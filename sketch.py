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
debug = sys.gettrace() is not None
if debug:
    print("Debugging")
    theano.config.optimizer = 'fast_compile'  # or "None"
    theano.config.exception_verbosity = 'high'
    theano.config.compute_test_value = 'warn'
else:
    theano.config.compute_test_value = 'off'


import theano.tensor as T
import numpy as np

import fuel

from argparse import ArgumentParser

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping
from blocks.algorithms import Adam, RMSProp, AdaGrad, Scale, AdaDelta
from blocks.initialization import Constant

from blocks.graph import ComputationGraph, apply_dropout
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.saveload import SimpleExtension
from blocks.bricks import Random, Initializable
from blocks.bricks.base import application
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.bricks.recurrent import LSTM, GatedRecurrent
from blocks.bricks.recurrent import RecurrentStack
from blocks.bricks.sequence_generators import SequenceGenerator, Readout
from fuel.datasets import H5PYDataset
from blocks.filter import VariableFilter
from blocks.extensions.saveload import Checkpoint, Load
import cPickle as pickle

from blocks_extras import OrthogonalGlorot
from blocks.initialization import Uniform

floatX = theano.config.floatX
fuel.config.floatX = floatX
from fuel.transformers import Mapping
from blocks.utils import named_copy
logger = logging.getLogger(__name__)
import pprint

#-----------------------------------------------------------------------------
# this must be a global function so we can pickle it
import math
def _is_nan(log):
    return any(v != v for v in log.current_row.itervalues())

def _transpose(data):
    return tuple(np.swapaxes(array,0,1) for array in data)

#----------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')  # allow plotting without a graphic display
from matplotlib import pylab as pl
from pylab import cm

def drawpoints(points, xmin=None, ymin=None, xmax=None, ymax=None):
    skips = np.hstack((np.array([0]),points[:,2]))
    xys = np.vstack((np.zeros((1,2)),np.cumsum(points[:,:2],axis=0)))
    xys[:,:2] -= xys[:,:2].mean(axis=0)
#     xys = points[:,:2]

    xs=[]
    ys=[]
    x=[]
    y=[]
    for xy,s in zip(xys, skips):
        if s:
            if len(x) > 1:
                xs.append(x)
                ys.append(y)
            x=[]
            y=[]
        else:
            x.append(xy[0])
            y.append(xy[1])

    for x,y in zip(xs, ys):
        pl.plot(x, y, 'k-')

    if xmin is None:
        xmin,ymin = xys.min(axis=0)
        xmax,ymax = xys.max(axis=0)
    ax = pl.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
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

        epochstr = ""
        if self.main_loop:
            epochstr = "-{:03d}".format(self.main_loop.log.status['epochs_done'])

        res = self.sample()
        outputs = res[-2]

        mins = []
        maxs = []
        for i in range(batch_size):
            points = outputs[:,i,:]
            xys = np.vstack((np.zeros((1,2)),np.cumsum(points[:,:2],axis=0)))
            xys[:,:2] -= xys[:,:2].mean(axis=0)
            mins.append(xys.min(axis=0))
            maxs.append(xys.max(axis=0))
        xmin, ymin = np.array(mins).min(axis=0)
        xmax, ymax = np.array(maxs).max(axis=0)

        # find a new grid of subplot which best match the y/x ratio
        r = (ymax-ymin)/(xmax-xmin)
        reverse = r > 1.
        if reverse: r = 1./r
        bestapprox = 10
        for f in range(1,self.N):
            if self.N % f != 0:
                continue
            rapprox = 1./float(f*f)
            if abs(r-rapprox) < abs(r-bestapprox):
                bestapprox = rapprox
                bestf = f

        if reverse:
            h, w = self.N//bestf, self.N*bestf
            r = 1./r
            bestapprox = 1./bestapprox
        else:
            h, w = self.N*bestf, self.N//bestf

        if bestapprox > r:
            newydelta = (ymax-ymin)*bestapprox/r
            ymax += newydelta/2.
            ymin -= newydelta/2.
        else:
            newxdelta = (xmax-xmin)*r/bestapprox
            xmax += newxdelta/2.
            xmin -= newxdelta/2.

        pl.close('all')
        fig = pl.figure('Sample sketches')
        fig.set_size_inches(25,25)
        for i in range(batch_size):
            pl.subplot(h,w,i+1)
            drawpoints(outputs[:,i,:], xmin, ymin, xmax, ymax)
        fname = os.path.join(self.path,'sketch{}.png'.format(epochstr))
        print('Writting to %s'%fname)
        pl.subplots_adjust(left=0.005,right=0.995,bottom=0.005,top=0.995,
                           wspace=0.01,hspace=0.01)
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
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        fname = os.path.join(self.path,'pen{}.png'.format(epochstr))
        print('Writting to %s'%fname)
        pl.savefig(fname)


#----------------------------------------------------------------------------
from blocks.bricks.sequence_generators import AbstractEmitter
class SketchEmitter(AbstractEmitter, Initializable, Random):
    """A mixture of gaussians emitter for x,y and logistic for pen-up/down.

    Parameters
    ----------
    mix_dim : int the number of gaussians to mix

    """
    def __init__(self, mix_dim=20, epsilon=1e-5, **kwargs):
        self.mix_dim = mix_dim
        self.epsilon = epsilon
        super(SketchEmitter, self).__init__(**kwargs)

    def components(self, readouts):
        """
        Take components from readout.

        If we have steps flatten steps,bacth_size to steps*batch_size

        :param readouts: [batch_size,get_dim('inputs')] or
         [steps,batch_size,get_dim('inputs')]
        :return: mean=[steps*batch_size,mix_dim,2],
         sigma=[steps*batch_size,mix_dim,2],
         corr=[steps*batch_size,mix_dim],
         weight=[steps*batch_size,mix_dim],
         penup=[steps*batch_size,1]
        """
        mix_dim = self.mix_dim  # get_dim('mix')
        output_norm = 2*mix_dim  # x,y
        readouts = readouts.reshape((-1, self.get_dim('inputs')))

        mean = readouts[:,0:output_norm].reshape((-1,mix_dim,2))  # 20
        sigma = T.exp(readouts[:,output_norm:2*output_norm]
                      .reshape((-1,mix_dim,2)))  # 21
        # [batch_size,mix_dim]
        corr = T.tanh(readouts[:,2*output_norm:2*output_norm+mix_dim])  # 22
        # [batch_size,mix_dim]
        weight = T.nnet.softmax(
            readouts[:,2*output_norm+mix_dim:2*output_norm+2*mix_dim])  # 19
        # [batch_size,1]
        penup = T.nnet.sigmoid(readouts[:,2*output_norm+2*mix_dim:])  # 18

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

        c = (1 - T.sqrt(1-corr**2))/(corr + self.epsilon)
        c = c.dimshuffle((0, 1, 'x'))  # same for x and y
        nr = nr + nr[:,:,::-1]*c  # x+c*y and y + c*x

        nr = nr / T.sqrt(1+c**2)
        nr = nr * sigma
        nr = nr + mean
        # If I dont do dtype=floatX in the next line I get an error
        weight = self.theano_rng.multinomial(pvals=weight, dtype=floatX)
        # an alternative is the following code:
        # right_boundry = T.cumsum(weight,axis=1)  # [...,1]
        # left_boundry = right_boundry - weight  # [0,...]
        # un0 = self.theano_rng.uniform(size=(batch_size,)).dimshuffle((0, 'x'))
        # weight = T.cast(left_boundry <= un0, floatX) * \
        #          T.cast(un0 < right_boundry, floatX)  # 1 only in one bin

        xy = nr * weight[:,:,None]  # [batch_size,mix_dim,2]

        xy = xy.sum(axis=1) # [batch_size,2]

        un = self.theano_rng.uniform(size=(batch_size, 1)) #, dtype=floatX)
        penup = T.cast(un < penup, floatX) # .astype(floatX)  # [batch_size,1]

        res = T.concatenate([xy, penup], axis=1)
        return res

    def cost(self, readouts, outputs):
        """
        All steps or single step

        :param readouts: output of hidden layer [batch_size,input_dim] or
         [steps,batch_size,input_dim]
        :param outputs: sketch cordinates of next time period batch_size,3] or
         [steps, batch_size,3]
        :return: NLL [batch_size] or [steps,batch_size]
        """
        nll_ndim = readouts.ndim - 1
        nll_shape = readouts.shape[:-1]
        # if we have steps flatten steps,bacth_size to steps*batch_size
        outputs = outputs.reshape((-1,3))
        mean, sigma, corr, weight, penup = self.components(readouts)

        # duplicate the output over all mix_dim
        d = outputs[:,:2].dimshuffle((0, 'x', 1)) - mean  # 25
        # [batch_size,mix_dim]
        sigma2 = sigma[:,:,0] * sigma[:,:,1] + self.epsilon  # 25
        z = d ** 2 / sigma ** 2
        # [batch_size,mix_dim]
        z = z.sum(axis=-1) - 2 * corr * (d[:,:,0] * d[:,:,1]) / sigma2  # 25
        corr1 = 1 - corr ** 2 + self.epsilon  # 24
        n = - z / (2 * corr1)  # 24 [batch_size,mix_dim]
        nmax = n.max(axis=-1, keepdims=True)
        n = n - nmax
        n = T.exp(n) / (2*np.pi*sigma2*T.sqrt(corr1))  # 24
        # [batch_size]
        nll = -T.log((n * weight).sum(axis=-1, keepdims=True) + self.epsilon)
        nll -= nmax
        nll += T.nnet.binary_crossentropy(penup, outputs[:,2:])  # 26

        return nll.reshape(nll_shape, ndim=nll_ndim)

    @application
    def initial_outputs(self, batch_size):
        return T.zeros((batch_size,3))

    def get_dim(self, name):
        if name == 'inputs':
            return 2*(2*self.mix_dim)+2*self.mix_dim+1
        if name == 'outputs':
            return 3  # x,y,pen

        return super(SketchEmitter, self).get_dim(name)


#----------------------------------------------------------------------------
def main(name, epochs, batch_size, learning_rate,
         dim, mix_dim, old_model_name, max_length, bokeh, GRU, dropout,
         depth, max_grad, step_method, epsilon, sample, skip, uniform, top):

    #----------------------------------------------------------------------
    datasource = name

    def shnum(x):
        """ Convert a positive float into a short tag-usable string
             E.g.: 0 -> 0, 0.005 -> 53, 100 -> 1-2
        """
        return '0' if x <= 0 else '%s%d' % (("%e"%x)[0], -np.floor(np.log10(x)))

    jobname = "%s-%dX%dm%dd%dr%sb%de%s" % (datasource, depth, dim, mix_dim,
                                           int(dropout*10),
                                           shnum(learning_rate), batch_size,
                                           shnum(epsilon))
    if max_length != 600:
        jobname += '-L%d'%max_length

    if GRU:
        jobname += 'g'
    if max_grad != 5.:
        jobname += 'G%g'%max_grad
    if step_method != 'adam':
        jobname += step_method
    if skip:
        jobname += 'D'
        assert depth > 1
    if top:
        jobname += 'T'
        assert depth > 1
    if uniform > 0.:
        jobname += 'u%d'%int(uniform*100)

    if debug:
        jobname += ".debug"

    if sample:
        print("Sampling")
    else:
        print("\nRunning experiment %s" % jobname)
    if old_model_name:
        print("starting from model %s"%old_model_name)

    #----------------------------------------------------------------------
    transitions = [GatedRecurrent(dim=dim) if GRU else LSTM(dim=dim)
                   for _ in range(depth)]
    if depth > 1:
        transition = RecurrentStack(transitions, name="transition",
                                    skip_connections=skip or top)
        if skip:
            source_names=[RecurrentStack.suffix('states', d) for d in range(depth)]
        else:
            source_names=[RecurrentStack.suffix('states', depth-1)]
    else:
        transition = transitions[0]
        transition.name = "transition"
        source_names=['states']

    emitter = SketchEmitter(mix_dim=mix_dim,
                            epsilon=epsilon,
                            name="emitter")
    readout = Readout(
        readout_dim=emitter.get_dim('inputs'),
        source_names=source_names,
        emitter=emitter,
        name="readout")
    generator = SequenceGenerator(readout=readout, transition=transition)

    # Initialization settings
    if uniform > 0.:
        generator.weights_init = Uniform(width=uniform*2.)
    else:
        generator.weights_init = OrthogonalGlorot()
    generator.biases_init = Constant(0)

    # Build the cost computation graph [steps, batch_size, 3]
    x = T.tensor3('features', dtype=floatX)
    if debug:
        x.tag.test_value = np.ones((max_length,batch_size,3)).astype(floatX)
    x = x[:max_length,:,:]  # has to be after setting test_value
    cost = generator.cost(x)
    cost.name = "sequence_log_likelihood"

    # Give an idea of what's going on
    model = Model(cost)
    params = model.get_parameter_dict()
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

    #------------------------------------------------------------
    extensions = []
    if old_model_name:
        if old_model_name == 'continue':
            old_model_name = jobname
        with open(old_model_name + '_model', "rb") as f:
            old_model = pickle.load(f)
        model.set_parameter_values(old_model.get_parameter_values())
        del old_model
    else:
        # Initialize parameters
        for brick in model.get_top_bricks():
            brick.initialize()

    if sample:
        assert old_model_name and old_model_name != 'continue'
        Sample(generator, steps=max_length, path=old_model_name).do(None)
        exit(0)

    #------------------------------------------------------------
    # Define the training algorithm.
    cg = ComputationGraph(cost)
    if dropout > 0.:
        from blocks.roles import INPUT, OUTPUT
        dropout_target = VariableFilter(roles=[OUTPUT],
                                        bricks=transitions,
                                        name_regex='states')(cg.variables)
        print('# dropout %d' % len(dropout_target))
        cg = apply_dropout(cg, dropout_target, dropout)
        opt_cost = cg.outputs[0]
    else:
        opt_cost = cost

    if step_method == 'adam':
        step_rule = Adam(learning_rate)
    elif step_method == 'rmsprop':
        step_rule = RMSProp(learning_rate, decay_rate=0.95)
    elif step_method == 'adagrad':
        step_rule = AdaGrad(learning_rate)
    elif step_method == 'adadelta':
        step_rule = AdaDelta()
    elif step_method == 'scale':
        step_rule = Scale(learning_rate)
    else:
        raise Exception('Unknown sttep method %s'%step_method)

    step_rule = CompositeRule([StepClipping(max_grad), step_rule])

    algorithm = GradientDescent(
        cost=opt_cost, parameters=cg.parameters,
        step_rule=step_rule)

    #------------------------------------------------------------
    observables = [cost]

    # Fetch variables useful for debugging
    (energies,) = VariableFilter(
        applications=[generator.readout.readout],
        name_regex="output")(cg.variables)
    min_energy = named_copy(energies.min(), "min_energy")
    max_energy = named_copy(energies.max(), "max_energy")
    observables += [min_energy, max_energy]

    # (activations,) = VariableFilter(
    #     applications=[generator.transition.apply],
    #     name=generator.transition.apply.states[0])(cg.variables)
    # mean_activation = named_copy(abs(activations).mean(),
    #                              "mean_activation")
    # observables.append(mean_activation)

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
                             which_sets=['train'], sources=('features',),
                             load_in_memory=True)
    train_stream = DataStream(train_ds,
                              iteration_scheme=ShuffledScheme(
                                  train_ds.num_examples, batch_size))

    test_ds = H5PYDataset(datasource_fname, #max_length=max_length,
                            which_sets=['test'], sources=('features',),
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
        print('%s #batch %d #examples %d' %
              (label, batch_count, examples_count))

    stream_stats(train_stream, 'train')
    stream_stats(test_stream, 'test')

    extensions += [Timing(every_n_batches=10),
                   TrainingDataMonitoring(
                       observables, prefix="train",
                       every_n_batches=10),
                   DataStreamMonitoring(
                       [cost],  # without dropout
                       test_stream,
                       prefix="test",
                       on_resumption=True,
                       after_epoch=False,  # by default this is True
                       every_n_batches=100),
                   # all monitored data is ready so print it...
                   # (next steps may take more time and we want to see the
                   # results as soon as possible so print as soon as you can)
                   Printing(every_n_batches=10),
                   # perform multiple dumps at different intervals
                   # so if one of them breaks (has nan) we can hopefully
                   # find a model from few batches ago in the other
                   Checkpoint(jobname,
                              before_training=False, after_epoch=True,
                              save_separately=['log', 'model']),
                   Sample(generator, steps=max_length,
                          path=jobname+'.test',
                          every_n_batches=100),
                   ProgressBar(),
                   FinishAfter(after_n_epochs=epochs)
                    # This shows a way to handle NaN emerging during
                    # training: simply finish it.
                    .add_condition(["after_batch"], _is_nan),
                   ]

    if bokeh:
        from blocks.extras.extensions.plot import Plot
        extensions.append(Plot(
            'sketch',
            channels=[['cost']], every_n_batches=10))

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
                default=56, help="Size of each mini-batch.")
    parser.add_argument("--lr", "--learning-rate", type=float,
                        dest="learning_rate",default=1e-4,
                        help="Learning rate")
    parser.add_argument("--dim", type=int,
                default=700, help="RNN state dimension")
    parser.add_argument("--mix-dim", type=int,
                default=20, help="Number of gaussian mixtures")
    parser.add_argument("--model", type=str, dest="old_model_name",
                        metavar="old-model/continue",
                        help="Start from an old model created by a previous"
                             " run. Or use continue")
    parser.add_argument("--max-length", type=int, default=600,
                        help="Maximal number of steps in a single sequence")
    parser.add_argument("--bokeh", action='store_true', default=False,
                        help="Set if you want to use Bokeh ")
    parser.add_argument("--GRU", action='store_true', default=False,
                        help="Use GatedRecurrent network instead of LSTM.")
    parser.add_argument("-d","--dropout",type=float,default=0,
                        help="Use dropout. Set to 0 for no dropout.")
    parser.add_argument("--depth", type=int,
                        default=3,
                        help="Number of recurrent layers to be stacked.")
    parser.add_argument("-G", "--max-grad", type=float,
                        default=10.,
                        help="Maximal gradient limit")
    parser.add_argument("--step-method", type=str, default="adam",
                        choices=["adam", "scale", "rmsprop", "adagrad",
                                 "adadelta"],
                        help="What gradient step rule to use. Default adam.")
    parser.add_argument("--epsilon",type=float,default=1e-5,
                        help="Epsilon value for mixture of gaussians")
    parser.add_argument("--sample", action='store_true', default=False,
                        help="Just generate a sample without traning.")
    parser.add_argument("--skip", action='store_true', default=False,
                        help="To send the input to all layers and not just the"
                             " first. Also to use the states of all layers as"
                             " output and not just the last.")
    parser.add_argument("--top", action='store_true', default=False,
                        help="Same as skip but use only the output of top layer.")
    parser.add_argument("-u","--uniform",type=float,default=0,
                        help="Use uniform weight initialization. "
                             "Default use Orhtogonal / Glorot.")

    args = parser.parse_args()

    main(**vars(args))

