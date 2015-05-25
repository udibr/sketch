#!/usr/bin/env python

from __future__ import division, print_function

import logging
import os

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)


import sys
import theano
# Automatically detect we are debugging and turn on maximal Theano debugging
if sys.gettrace() is not None:
    print("Debugging")
    theano.config.optimizer='fast_compile' #"None"  #
    theano.config.exception_verbosity='high'
    theano.config.compute_test_value = 'warn'

import theano.tensor as T
import numpy as np

import fuel

from argparse import ArgumentParser

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, Adam
from blocks.initialization import Constant

from blocks.graph import ComputationGraph, apply_dropout
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.saveload import SimpleExtension
from blocks.extensions.saveload import Dump, LoadFromDump
from blocks.bricks import Random, Initializable, Linear
from blocks.bricks.base import application
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.bricks.recurrent import LSTM, GatedRecurrent
from blocks.bricks.sequence_generators import SequenceGenerator, Readout
from fuel.datasets import H5PYDataset
from blocks.filter import VariableFilter
from blocks.bricks.parallel import Fork

from blocks_extras import GlorotBengio, AdaGrad, RecurrentStack

floatX = theano.config.floatX
fuel.config.floatX = floatX
from fuel.transformers import Mapping
from blocks.utils import named_copy, dict_union
logger = logging.getLogger(__name__)
from blocks.extensions.plot import Plot
import pprint

#-----------------------------------------------------------------------------
# this must be a global function so we can pickle it
import math
def _is_nan(log):
    return math.isnan(log.current_row.get('total_gradient_norm',0.))

def _transpose(data):
    return tuple(np.swapaxes(array,0,1) for array in data)

#----------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')  # allow plotting without a graphic display
from matplotlib import pylab as pl
from pylab import cm

def drawpoints(points,pad=0):
    skips = np.hstack((np.array([0]),points[:,2]))
    xys = np.vstack((np.zeros((1,2)),np.cumsum(points[:,:2],axis=0)))
#     xys = points[:,:2]

    xs=[]
    ys=[]
    x=[]
    y=[]
    for xy,s in zip(xys,skips):
        if s:
            if x is not None:
                xs.append(x)
                ys.append(y)
            x=[]
            y=[]
        x.append(xy[0])
        y.append(xy[1])

    for x,y in zip(xs,ys):
        if len(x) > 1:
            pl.plot(x,y,'k-')

    xmin,ymin = xys.min(axis=0)
    xmax,ymax = xys.max(axis=0)
    ax = pl.gca()
    ax.set_xlim(xmin-pad,xmax+pad)
    ax.set_ylim(ymin-pad,ymax+pad)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

class Sample(SimpleExtension):
    """Generate a sample sketch

    Parameters
    ----------
    steps : int Number of steps to samples
    path : prefix to all the files generated
    """
    def __init__(self, generator, N=8, steps=1200, path='samples', **kwargs):
        self.N = N
        self.path = path
        super(Sample, self).__init__(**kwargs)

        batch_size = self.N * self.N

        self.sample = ComputationGraph(generator.generate(
            n_steps=steps, batch_size=batch_size, iterate=True)
        ).get_theano_function()

    def do(self, callback_name, *args):
        batch_size = self.N * self.N

        res = self.sample()
        outputs = res[-2]
        pl.close('all')
        fig = pl.figure('Sample sketches')
        fig.set_size_inches(16,16)
        for i in range(batch_size):
            pl.subplot(self.N,self.N,i+1)
            drawpoints(outputs[:,i,:])
        fname = '%s-sketch.png'%self.path
        print('Writting to %s'%fname)
        pl.savefig(fname)

        fig = pl.figure('Pen down')
        fig.set_size_inches(17.5,17.5)
        img = 1-outputs[:,:,2]
        pl.imshow(img,cmap=cm.gray)
        pl.gca().set_xticks([])
        pl.gca().set_yticks([])
        pl.xlabel('iteration')
        pl.ylabel('sample')
        pl.title('Pen down for different samples vs. iteration step')
        fname = '%s-pen.png'%self.path
        print('Writting to %s'%fname)
        pl.savefig(fname)


#----------------------------------------------------------------------------
from blocks.bricks.sequence_generators import AbstractEmitter
class SketchEmitter(AbstractEmitter, Initializable, Random):
    """A mixture of gaussians emitter for x,y and logistic for pen-up/down.

    Parameters
    ----------
    mix_dim : int the number of gaussians to mix
    initial_output : int or a scalar :class:`~theano.Variable`
        The initial output.

    """
    def __init__(self, mix_dim=20, initial_output=0, **kwargs):
        self.initial_output = initial_output
        self.mix_dim = mix_dim
        super(SketchEmitter, self).__init__(**kwargs)

    def components(self, readouts):
        """
        All steps or single step

        :param readouts: [batch_size,get_dim('inputs')] or [steps,batch_size,get_dim('inputs')]
        :return: mean=[steps*batch_size,mix_dim,2], sigma=[steps*batch_size,mix_dim,2],
                corr=[steps*batch_size,mix_dim], weight=[steps*batch_size,mix_dim],
                penup=[steps*batch_size,1]
        """
        mix_dim = self.mix_dim  # get_dim('mix')
        output_norm = 2*mix_dim  # x,y
        readouts = readouts.reshape((-1, self.get_dim('inputs')))

        mean = readouts[:,0:output_norm].reshape((-1,mix_dim,2))  #20
        sigma = T.exp(readouts[:,output_norm:2*output_norm].reshape((-1,mix_dim,2))) + 1e-3  #21
        corr = T.tanh(readouts[:,2*output_norm:2*output_norm+mix_dim])  #22 [batch_size,mix_dim]
        weight = T.nnet.softmax(readouts[:,2*output_norm+mix_dim:2*output_norm+2*mix_dim]) #19 [batch_size,mix_dim]
        penup = T.nnet.sigmoid(readouts[:,2*output_norm+2*mix_dim:])  # 18 [batch_size,1]

        return mean, sigma, corr, weight, penup

    @application
    def emit(self, readouts):
        """
        Single step

        :param readouts: output of hidden layer [batch_size,get_dim('inputs')]
        """
        mean, sigma, corr, weight, penup = self.components(readouts)
        mix_dim = self.mix_dim  # get_dim('mix')
        batch_size = readouts.shape[0]

        nr = self.theano_rng.normal(
            size=(batch_size, mix_dim, 2),
            avg=0., std=1.) #, dtype=floatX)

        c = (1 - T.sqrt(1-corr**2))/(corr + 1e-12)
        c = c.dimshuffle((0, 1, 'x'))  # same for x and y
        nr = nr + nr[:,:,::-1]*c  # x+c*y and y + c*x

        nr = nr / T.sqrt(1+c**2)
        nr = nr * sigma
        nr = nr + mean
        # If I dont do dtype=floatX in the next line I get:
        #   File ".../github/Theano/theano/scan_module/scan_op.py", line 476, in make_node
        #    inner_sitsot_out.type.dtype))
        # ValueError: When compiling the inner function of scan the following error has been encountered: The initial state (`outputs_info` in scan nomenclature) of variable IncSubtensor{Set;:int64:}.0 (argument number 0) has dtype float32, while the result of the inner function (`fn`) has dtype float64. This can happen if the inner function of scan results in an upcast or downcast.
        weight = self.theano_rng.multinomial(pvals=weight, dtype=floatX)
        # right_boundry = T.cumsum(weight,axis=1)  # [...,1]
        # left_boundry = right_boundry - weight  # [0,...]
        # un0 = self.theano_rng.uniform(size=(batch_size,)).dimshuffle((0, 'x'))
        # weight = T.cast(left_boundry <= un0, floatX) * T.cast(un0 < right_boundry, floatX)  # 1 only in one bin

        xy = nr * weight[:,:,None]  # [batch_size,mix_dim,2]

        xy = xy.sum(axis=1) # [batch_size,2]

        un = self.theano_rng.uniform(size=(batch_size, 1)) #, dtype=floatX)
        penup = T.cast(un < penup, floatX) # .astype(floatX)  # [batch_size,1]

        res = T.concatenate([xy, penup], axis=1)
        return res

    def cost(self, readouts, outputs):
        """
        All steps or single step

        :param readouts: output of hidden layer [batch_size,input_dim] or [steps,batch_size,input_dim]
        :param outputs: sketch cordinates of next time period batch_size,3] or [steps, batch_size,3]
        :return: NLL [steps*batch_size]
        """
        nll_ndim = readouts.ndim - 1
        nll_shape = readouts.shape[:-1]
        outputs = outputs.reshape((-1,3))
        mean, sigma, corr, weight, penup = self.components(readouts)

        d = outputs[:,:2].dimshuffle((0, 'x', 1)) - mean  #25 duplicate the output over all mix_dim
        sigma2 = sigma[:,:,0] * sigma[:,:,1]  #25 [batch_size,mix_dim]
        z = d ** 2 / sigma ** 2
        z = z.sum(axis=-1) - 2 * corr * (d[:,:,0] * d[:,:,1]) / sigma2  #25 [batch_size,mix_dim]
        corr1 = 1 - corr ** 2 + 1e-6 #24
        n = - z / (2 * corr1)  #24 [batch_size,mix_dim]
        nmax = n.max(axis=-1, keepdims=True)
        n = n - nmax
        n = T.exp(n) / (2*np.pi*sigma2*T.sqrt(corr1))  #24
        nll = -T.log((n * weight).sum(axis=-1, keepdims=True) + 1e-8) - nmax +\
              T.nnet.binary_crossentropy(penup, outputs[:,2:]) #26 [batch_size]

        return nll.reshape(nll_shape, ndim=nll_ndim)

    @application
    def initial_outputs(self, batch_size):
        return T.ones((batch_size,3)) * self.initial_output #, dtype=floatX

    def get_dim(self, name):
        if name == 'inputs':
            return 2*(2*self.mix_dim)+2*self.mix_dim+1
        if name == 'outputs':
            return 3  # x,y,pen

        return super(SketchEmitter, self).get_dim(name)


#----------------------------------------------------------------------------
def main(name, epochs, batch_size, learning_rate,
         dim, mix_dim, old_model_name, max_length, bokeh, GRU, dropout,
         depth):

    #----------------------------------------------------------------------
    datasource = name

    def shnum(x):
        """ Convert a positive float into a short tag-usable string representation.
             E.g.: 0 -> 0, 0.005 -> 53, 100 -> 1-2
        """
        return '0' if x <= 0 else '%s%d' % (("%e"%x)[0], -np.floor(np.log10(x)))

    jobname = "%s-d%d-m%d-lr%s-b%d" % (datasource, dim, mix_dim,
                                       shnum(learning_rate), batch_size)
    if max_length != 600:
        jobname += '-L%d'%max_length

    if GRU:
        jobname += 'g'
    if dropout:
        jobname += 'D'
    if depth > 1:
        jobname += 'N%d'%depth

    print("\nRunning experiment %s" % jobname)
    print("         learning rate: %5.3f" % learning_rate) 
    print("             dimension: %d" % dim)
    print("         mix dimension: %d" % mix_dim)
    print()

    if old_model_name == 'continue':
        old_model_name = jobname

    #----------------------------------------------------------------------
    if GRU:
        transition = GatedRecurrent(dim=dim, name="transition")
    else:
        transition = LSTM(dim=dim, name="transition")

    if depth > 1:
        transition = RecurrentStack(dim=dim,
                                    depth=depth,
                                    name="transition",
                                    prototype=transition)

    emitter = SketchEmitter(mix_dim=mix_dim, name="emitter")
    readout = Readout(
        readout_dim=emitter.get_dim('inputs'),
        source_names=['states'],
        emitter=emitter,
        name="readout")
    normal_inputs = [name for name in transition.apply.sequences
                     if 'mask' not in name]
    fork = Fork(normal_inputs, prototype=Linear(use_bias=True))
    generator = SequenceGenerator(readout=readout, transition=transition,
                                  fork=fork)

    # Initialization settings
    generator.weights_init = GlorotBengio() # Orthogonal()
    generator.biases_init = Constant(0)

    # for LSTM you can not use Orthogonal because it has 1D weights connecting
    # the cells to in/out/forget gates (but you can on GatedRecurrent)
    # if not GRU:
    #     generator.push_initialization_config()
    #     transition.weights_init = GlorotBengio()

    # Build the cost computation graph
    x = T.tensor3('features',dtype=floatX)[:max_length,:,:]  # [steps,batch_size, 3]
    x.tag.test_value = np.ones((max_length,batch_size,3)).astype(np.float32)
    cost = generator.cost(x)
    cost.name = "sequence_log_likelihood"

    # Give an idea of what's going on
    model = Model(cost)
    params = model.get_params()
    logger.info("Parameters:\n" +
                pprint.pformat(
                    [(key, value.get_value().shape) for key, value
                     in params.items()],
                    width=120))
    model_size = 0
    for v in params.itervalues():
        s = v.get_value().shape
        model_size += s[0] * (s[1] if len(s) > 1 else 1)
    logger.info("Total number of parameters %d"%model_size)

    # Initialize parameters
    for brick in model.get_top_bricks():
        brick.initialize()

    # Define the training algorithm.
    cg = ComputationGraph(cost)
    if dropout:
        from blocks.roles import INPUT
        dropout_target = VariableFilter(roles=[INPUT],
                                        bricks=[readout],
                                        name_regex='states')(cg.variables)
        cg = apply_dropout(cg, dropout_target, 0.5)
        target_cost = cg.outputs[0]
    else:
        target_cost = cost

    algorithm = GradientDescent(
        cost=target_cost, params=cg.parameters,
        step_rule=CompositeRule([StepClipping(10.), AdaGrad(learning_rate)]))

    #------------------------------------------------------------
    observables = [cost]

    # Fetch variables useful for debugging
    (energies,) = VariableFilter(
        applications=[generator.readout.readout],
        name_regex="output")(cg.variables)
    (activations,) = VariableFilter(
        applications=[generator.transition.apply],
        name=generator.transition.apply.states[0])(cg.variables)
    min_energy = named_copy(energies.min(), "min_energy")
    max_energy = named_copy(energies.max(), "max_energy")
    mean_activation = named_copy(abs(activations).mean(),
                                 "mean_activation")
    observables += [min_energy, max_energy, mean_activation]

    observables += [algorithm.total_step_norm, algorithm.total_gradient_norm]
    for name, param in params.items():
        observables.append(named_copy(
            param.norm(2), name + "_norm"))
        observables.append(named_copy(
            algorithm.gradients[param].norm(2), name + "_grad_norm"))

    #------------------------------------------------------------
    datasource_fname = os.path.join(fuel.config.data_path, datasource,
                                    datasource+'.hdf5')

    train_ds = H5PYDataset(datasource_fname, #max_length=max_length,
                             which_set='train', sources=('features',),
                             load_in_memory=True)
    train_stream = DataStream(train_ds,
                              iteration_scheme=ShuffledScheme(
                                  train_ds.num_examples, batch_size))

    test_ds = H5PYDataset(datasource_fname, #max_length=max_length,
                            which_set='test', sources=('features',),
                            load_in_memory=True)
    test_stream  = DataStream(test_ds,
                              iteration_scheme=SequentialScheme(
                                  test_ds.num_examples, batch_size))

    train_stream = Mapping(train_stream, _transpose)
    test_stream = Mapping(test_stream, _transpose)

    def stream_stats(ds, label):
        itr = ds.get_epoch_iterator(as_dict=True)
        batch_count = 0
        examples_count = 0
        for batch in itr:
            batch_count += 1
            examples_count += batch['features'].shape[1]
        print('%s #batch %d #examples %d' % (label, batch_count, examples_count))

    stream_stats(train_stream, 'train')
    stream_stats(test_stream, 'test')
    #------------------------------------------------------------
    extensions = []
    if old_model_name:
        extensions.append(LoadFromDump(old_model_name))
        # or you can just load the weights without state using:
        # model.set_param_values(LoadFromDump(old_model_name).manager.load_parameters())
    extensions += [Timing(),
                   TrainingDataMonitoring(
                       observables, prefix="train", every_n_batches=10),
                   DataStreamMonitoring(
                       [cost],
                       test_stream,
                       prefix="test",
                       before_training=True, after_epoch=True),
                   Sample(generator, steps=max_length, before_training=True,
                          after_epoch=True),
                   Printing(every_n_batches=10, after_epoch=True),
                   Dump(jobname, after_epoch=True,every_n_batches=10),
                   ProgressBar(),
                   FinishAfter(after_n_epochs=epochs)
                    # This shows a way to handle NaN emerging during
                    # training: simply finish it.
                    .add_condition("after_batch", _is_nan),
                   ]

    if bokeh:
        extensions.append(Plot(
            'sketch',
            channels=[
                ['cost'],]))

    # Construct the main loop and start training!
    main_loop = MainLoop(
        model=model,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=extensions
        )

    main_loop.run()

#-----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str,
                default="handwriting", help="Name for this experiment")
    parser.add_argument("--epochs", type=int,
                default=500, help="Number of training epochs to do")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                default=85, help="Size of each mini-batch."
                                 " For max-length=600 on GTX 980 use"
                                 " 85 for LSTM, 178 for GRU")
    parser.add_argument("--lr", "--learning-rate", type=float,
                        dest="learning_rate",default=1e-3,
                        help="Learning rate")
    parser.add_argument("--dim", type=int,
                default=900, help="RNN state dimension")
    parser.add_argument("--mix-dim", type=int,
                default=20, help="number of gaussian mixtures")
    parser.add_argument("--model", type=str, dest="old_model_name",
                help="start from an old model created by a previous run."
                     " Or use continue")
    parser.add_argument("--max-length", type=int, default=600,
                        help="maximal number of steps in a single sequence")
    parser.add_argument("--bokeh", action='store_true', default=False,
                        help="Set if you want to use Bokeh ")
    parser.add_argument("--GRU", action='store_true', default=False,
                        help="Use GatedRecurrent network instead of LSTM.")
    parser.add_argument("-d","--dropout",action='store_true',default=False,
                        help="Use dropout")
    parser.add_argument("--depth", type=int,
                default=1, help="Number of recurrent layers to be stacked.")

    args = parser.parse_args()

    main(**vars(args))

