# -*- coding: utf-8 -*-
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.bricks import Initializable, Linear
from blocks.bricks.base import application
from blocks.bricks.parallel import Fork, Parallel
from theano.gof import utils

from copy import copy

from blocks.filter import get_application_call
from blocks.roles import VariableRole, add_role


class RecurrentStack(BaseRecurrent, Initializable):
    u"""Stack of recurrent networks.

    Build a stack of recurrent layers from a supplied list of
    BaseRecurrent objects. Each object must have a `sequences`,
    `contexts`, `states` and `outputs` parameters to its `apply` method.
    It is assumed that all `states` have a "states" element
    (this can be configured with `states_name` parameter.)

    The `sequences` (the "inputs") of the the first layer (layer 0) is
    used as the `sequence` of the stack (the instance of this class.)
    The "mask" element of the `sequences` is carried to all layers that
    have a "mask" element in their `sequences`.
    For all the following layers, all elements of the `sequences`, except
    for "mask", are forked from the "states" element of the of the
    `states` of the previous layer.

    The concatenated `states` of all layers is the `states` of the stack.
    In order to avoid conflict, the names of the elements appearing in the
    `states` of each layers are suffixed with a '_<n>' before combining
    them. The '<n>' is the number of the layer starting from 0.
    The same process is true for the `contexts` and `outputs` of the
    stack.

    Parameters
    ----------
    transitions : list
        List of recurrent units to use in each layer.
        Note: A suffix with layer number is added to transitions' names.
    prototype : :class:`.FeedForward`, optional
        The transformation applied to each states of each layer  when it is
         forked to the sequences of the next layer. By
        default a :class:`.Linear` transformation with biases is used.
    states_name : string
        In a stack of RNN the state of each layer is used as input to the
        next. This string identify which element in the output of each
        layer should be used for this task. By default the element is
        called "states". To be more precise, this is the name of the
        element in the outputs of the apply method of each transition
        (layer) that is used, via fork, as the sequences (input) of the
        next layer. The same element should also appear in the states
        parameter of the apply method.
    fast : bool
        Use the fast, but also memory consuming, implementation of this code.
        By default true. WARNING: there is something strange in the behaviour
        of the slow implementation (see last test below.) Only output used
        outside this class can be found using the output name. This problem
        does not appear in fast mode.
    distribute : bool
        By default False. When true, the input (sequences) is spread to all
        layer and not just the bottom layer. Each vector in the sequences to
        the first layer is passed through a linear transformation, without bias,
        to each of the sequences of the other layers and is added to the
        usually input comfing from a fork from the states of the previous layer.
        For this to work it is assumed that all layers have the same names in
        their sequences.

    Notes
    -----
    See :class:`.BaseRecurrent` for more initialization parameters.

    """
    def __init__(self, transitions, prototype=None, states_name="states",
                 fast=True, distribute=False, **kwargs):
        super(RecurrentStack, self).__init__(**kwargs)

        self.states_name = states_name

        for d, transition in enumerate(transitions):
            transition.name += '_' + str(d)
        self.transitions = transitions

        if prototype is None:
            # By default use Linear (with bias) as a default fork.
            # This overrides the bad default inside Fork which is without bias.
            # Yes I know use_bias=True us the default, but it is a big deal so
            # I write it down explicitly.
            prototype = Linear(use_bias=True)
        depth = len(transitions)
        self.forks = [Fork(self.normal_inputs(d), name='fork_' + str(d),
                           prototype=prototype)
                      for d in range(1, depth)]

        self.distributes = []
        if distribute:
            input_names = self.normal_inputs(0)
            input_dims = self.transitions[0].get_dims(input_names)
            for d in range(1, depth):
                assert cmp(self.normal_inputs(d), input_names) == 0
                self.distributes.append(Parallel(
                    input_names, input_dims,
                    self.transitions[d].get_dims(input_names),
                    child_prefix = 'distribute_' + str(d)))

        self.children = self.transitions + self.forks + self.distributes

        # Programmatically set the apply method
        self.apply = self.fast_apply if fast else self.low_memory_apply
        setattr(self.apply, 'sequences', transitions[0].apply.sequences)
        for t in ["states", "contexts", "outputs"]:
            v = [s + '_' + str(d)
                 for d, transition in enumerate(transitions)
                 for s in getattr(transition.apply, t)]
            setattr(self.apply, t, v)

    def normal_inputs(self, d):
        transition = self.transitions[d]
        return [name for name in transition.apply.sequences
                if 'mask' not in name]

    def _push_allocation_config(self):
        # Configure the forks that connect the "states" element in the `states`
        # of one layer to the elements in the `sequences` of the next layer,
        # excluding "mask".
        # This involves `get_dim` requests
        # to the transitions. To make sure that it answers
        # correctly we should finish its configuration first.
        for transition in self.transitions:
            transition.push_allocation_config()

        for d, fork in enumerate(self.forks):
            fork.input_dim = self.transitions[d].get_dim(self.states_name)
            fork.output_dims = self.transitions[d+1].get_dims(
                fork.output_names)
        for d, distribute in enumerate(self.distributes):
            distribute.input_dims = self.transitions[0].get_dims(
                self.normal_inputs(0))
            distribute.output_dims = self.transitions[d+1].get_dims(
                self.normal_inputs(d+1))

    def do_apply(self, *args, **kwargs):
        """Apply the stack of transitions.

        This is the undecorated implementation of the apply method.
        It is separated from the decorated apply method in order to allow
        usage of different docrations (wrappers) to be used.


        Parameters
        ----------
        iterate : bool
            If ``True`` iteration is made. By default ``True``.
        reverse : bool
            If ``True``, the sequences are processed in backward
            direction. ``False`` by default.
        return_initial_states : bool
            If ``True``, initial states are included in the returned
            state tensors. ``False`` by default.
        The names appearing in the sequences, states, contexts parameters of
            the apply method of each of the transitions in self.transitions.
            Each such name is suffixed with a layer number.

        Returns
        -------
        The outputs of all transitions.
        All return values are of type :class:`~tensor.TensorVariable`.

        """
        results = []
        last_states = None
        layer0_kwargs = None
        for d, transition in enumerate(self.transitions):
            sequences_names = transition.apply.sequences
            if d == 0:
                layer0_kwargs = dict(
                    (s, kwargs.get(s)) for s in sequences_names)
                layer_kwargs = dict(layer0_kwargs)
            else:
                inputs = self.forks[d-1].apply(last_states, as_list=True)
                if self.distributes:
                    distributes = self.distributes[d-1].apply(as_list=True,
                                                              **layer0_kwargs)
                    for i in range(len(inputs)):
                        inputs[i] += distributes[i]
                layer_kwargs = dict(zip(self.normal_inputs(d), inputs))
                if "mask" in sequences_names:
                    layer_kwargs["mask"] = kwargs.get("mask")

            for t in ["states", "contexts", "outputs"]:
                suffix = '_' + str(d)
                for s in getattr(transition.apply, t):
                    layer_kwargs[s] = kwargs.get(s + suffix)

            for k in ['iterate', 'reverse', 'return_initial_states']:
                if k in kwargs:
                    layer_kwargs[k] = kwargs[k]
            result = transition.apply(as_list=True, **layer_kwargs)

            results.extend(result)

            state_index = transition.apply.outputs.index(self.states_name)
            last_states = result[state_index]
            if kwargs.get('return_initial_states', False):
                # Note that the following line reset the tag
                last_states = last_states[1:]

        return tuple(results)

    @recurrent
    def low_memory_apply(self, *args, **kwargs):
        kwargs['iterate'] = False
        return self.do_apply(*args, **kwargs)

    @application
    def fast_apply(self, *args, **kwargs):
        return self.do_apply(*args, **kwargs)

    def get_dim(self, name):
        if name in self.transitions[0].apply.sequences:
            return self.transitions[0].get_dim(name)
        name, layer = name.split('_')
        transition = self.transitions[int(layer)]
        return transition.get_dim(name)

    @application
    def initial_state(self, state_name, batch_size, *args, **kwargs):
        state_name, layer = state_name.split('_')
        transition = self.transitions[int(layer)]
        return transition.initial_state(state_name, batch_size,
                                        *args, **kwargs)


#-----------------------------------------------------------------------------
# Testing

import numpy
import theano
from theano import tensor
import itertools
from numpy.testing import assert_allclose
from blocks.initialization import Constant
from blocks.bricks.recurrent import LSTM
from collections import OrderedDict

class tRecurrentStack(object):
    def setUp(self):
        depth = 4
        self.depth = depth
        dim = 3  # don't change, hardwired in the code
        tarnsitions = [LSTM(dim=dim) for _ in range(depth)]
        self.stack0 = RecurrentStack(tarnsitions,
                                     weights_init=Constant(2),
                                     biases_init=Constant(0))
        self.stack0.initialize()
        self.stack1 = RecurrentStack(tarnsitions,
                                     weights_init=Constant(2),
                                     biases_init=Constant(0),
                                     fast=False)
        self.stack1.initialize()

        self.stack2 = RecurrentStack(tarnsitions,
                                     weights_init=Constant(2),
                                     biases_init=Constant(0),
                                     distribute=True)
        self.stack2.initialize()

        self.stack3 = RecurrentStack(tarnsitions,
                                     weights_init=Constant(2),
                                     biases_init=Constant(0),
                                     distribute=True,
                                     fast=False)
        self.stack3.initialize()

    def do_one_step(self, stack, distribute=False):
        depth = self.depth
        kwargs = OrderedDict()
        kwargs['inputs'] = tensor.matrix('inputs')

        for d in range(depth):
            kwargs['states_' + str(d)] = tensor.matrix('states_' + str(d))
            kwargs['cells_' + str(d)] = tensor.matrix('cells_' + str(d))
        results = stack.apply(iterate=False, **kwargs)
        next_h = theano.function(inputs=list(kwargs.values()),
                                 outputs=results)

        # batch=2
        h0_val = 0.1 * numpy.array([[[1, 1, 0], [0, 1, 1]]]*depth,
                                   dtype=theano.config.floatX)
        c0_val = 0.1 * numpy.array([[[1, 1, 0], [0, 1, 1]]]*depth,
                                   dtype=theano.config.floatX)
        x_val = 0.1 * numpy.array([range(12), range(12, 24)],
                                  dtype=theano.config.floatX)
        # we will use same weights on all layers
        W_state2x_val = 2 * numpy.ones((3, 12), dtype=theano.config.floatX)
        W_state_val = 2 * numpy.ones((3, 12), dtype=theano.config.floatX)
        W_cell_to_in = 2 * numpy.ones((3,), dtype=theano.config.floatX)
        W_cell_to_out = 2 * numpy.ones((3,), dtype=theano.config.floatX)
        W_cell_to_forget = 2 * numpy.ones((3,), dtype=theano.config.floatX)
        W_input2input = 2 * numpy.ones((12, 12), dtype=theano.config.floatX)

        def sigmoid(x):
            return 1. / (1. + numpy.exp(-x))
        h1_val = []
        x_v = x_val
        args_val = [x_val]
        for d in range(depth):
            h0_v = h0_val[d]
            args_val.append(h0_v)
            c0_v = c0_val[d]
            args_val.append(c0_v)

            # omitting biases because they are zero
            activation = numpy.dot(h0_v, W_state_val) + x_v
            if distribute and d > 0:
                activation += numpy.dot(x_val, W_input2input)

            i_t = sigmoid(activation[:, :3] + c0_v * W_cell_to_in)
            f_t = sigmoid(activation[:, 3:6] + c0_v * W_cell_to_forget)
            next_cells = f_t * c0_v + i_t * numpy.tanh(activation[:, 6:9])
            o_t = sigmoid(activation[:, 9:12] +
                          next_cells * W_cell_to_out)
            h1_v = o_t * numpy.tanh(next_cells)
            # current layer output state transformed to input of next
            x_v = numpy.dot(h1_v, W_state2x_val)

            h1_val.append(h1_v)

        res = next_h(*args_val)
        for d in range(depth):
            assert_allclose(h1_val[d], res[d*2], rtol=1e-6)

    def test_one_step(self):
        self.do_one_step(self.stack0)
        self.do_one_step(self.stack1)
        self.do_one_step(self.stack2, distribute=True)
        self.do_one_step(self.stack3, distribute=True)

    def do_many_steps(self, stack, distribute=False):
        depth = self.depth

        kwargs = OrderedDict()
        kwargs['inputs'] = tensor.tensor3('inputs')
        kwargs['mask'] = tensor.matrix('mask')
        results = stack.apply(iterate=True, **kwargs)
        calc_h = theano.function(inputs=list(kwargs.values()),
                                 outputs=results)

        # 24 steps
        #  4 batch examples
        # 12 dimensions per step
        x_val = (0.1 * numpy.asarray(
            list(itertools.islice(itertools.permutations(range(12)), 0, 24)),
            dtype=theano.config.floatX))
        x_val = numpy.ones((24, 4, 12),
                           dtype=theano.config.floatX) * x_val[:, None, :]
        # mask the last third of steps
        mask_val = numpy.ones((24, 4), dtype=theano.config.floatX)
        mask_val[12:24, 3] = 0
        # unroll all states and cells for all steps and also initial value
        h_val = numpy.zeros((depth, 25, 4, 3), dtype=theano.config.floatX)
        c_val = numpy.zeros((depth, 25, 4, 3), dtype=theano.config.floatX)
        # we will use same weights on all layers
        W_state2x_val = 2 * numpy.ones((3, 12), dtype=theano.config.floatX)
        W_state_val = 2 * numpy.ones((3, 12), dtype=theano.config.floatX)
        W_cell_to_in = 2 * numpy.ones((3,), dtype=theano.config.floatX)
        W_cell_to_out = 2 * numpy.ones((3,), dtype=theano.config.floatX)
        W_cell_to_forget = 2 * numpy.ones((3,), dtype=theano.config.floatX)
        W_input2input = 2 * numpy.ones((12, 12), dtype=theano.config.floatX)

        def sigmoid(x):
            return 1. / (1. + numpy.exp(-x))

        for i in range(1, 25):
            x_v = x_val[i-1]
            h_vs = []
            c_vs = []
            for d in range(depth):
                h_v = h_val[d][i-1, :, :]
                c_v = c_val[d][i-1, :, :]
                activation = numpy.dot(h_v, W_state_val) + x_v
                if distribute and d > 0:
                    activation += numpy.dot(x_val[i-1], W_input2input)

                i_t = sigmoid(activation[:, :3] + c_v * W_cell_to_in)
                f_t = sigmoid(activation[:, 3:6] + c_v * W_cell_to_forget)
                c_v1 = f_t * c_v + i_t * numpy.tanh(activation[:, 6:9])
                o_t = sigmoid(activation[:, 9:12] +
                              c_v1 * W_cell_to_out)
                h_v1 = o_t * numpy.tanh(c_v1)
                h_v = (mask_val[i - 1, :, None] * h_v1 +
                       (1 - mask_val[i - 1, :, None]) * h_v)
                c_v = (mask_val[i - 1, :, None] * c_v1 +
                       (1 - mask_val[i - 1, :, None]) * c_v)
                # current layer output state transformed to input of next
                x_v = numpy.dot(h_v, W_state2x_val)

                h_vs.append(h_v)
                c_vs.append(c_v)

            for d in range(depth):
                h_val[d][i, :, :] = h_vs[d]
                c_val[d][i, :, :] = c_vs[d]

        res = calc_h(x_val, mask_val)
        for d in range(depth):
            assert_allclose(h_val[d][1:], res[d*2], rtol=1e-4)
            assert_allclose(c_val[d][1:], res[d*2+1], rtol=1e-4)

    def test_many_steps(self):
        self.do_many_steps(self.stack0)
        self.do_many_steps(self.stack1)
        self.do_many_steps(self.stack2, distribute=True)
        self.do_many_steps(self.stack3, distribute=True)


from blocks.bricks.base import application
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, TrivialEmitter)
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from numpy.testing import assert_equal

class TestEmitter(TrivialEmitter):
    @application
    def cost(self, readouts, outputs):
        """Compute MSE."""
        return ((readouts - outputs) ** 2).sum(axis=readouts.ndim - 1)


def test_recurrentstack_sequence_generator():
    """Test RecurrentStack behaviour inside a SequenceGenerator.

    """
    floatX = theano.config.floatX
    rng = numpy.random.RandomState(1234)

    output_dim = 1
    dim = 20
    batch_size = 30
    n_steps = 10

    depth=2
    transitions = [LSTM(dim=dim) for _ in range(depth)]
    transition = RecurrentStack(transitions,fast=True,
                                weights_init=Constant(2),
                                biases_init=Constant(0))
    generator = SequenceGenerator(
        Readout(readout_dim=output_dim, source_names=["states_%d"%(depth-1)],
                emitter=TestEmitter()),
        transition,
        weights_init=IsotropicGaussian(0.1), biases_init=Constant(0.0),
        seed=1234)
    generator.initialize()

    y = tensor.tensor3('y')

    cost = generator.cost(y)

    # Check that all states can be accessed and not just the state connected
    # to readout.
    cg = ComputationGraph(cost)
    from blocks.roles import INPUT, OUTPUT
    dropout_target = VariableFilter(roles=[INNER_OUTPUT],
                                    # bricks=transitions,
                                    # name_regex='*'
                                    )(cg.variables)
    assert_equal(len(dropout_target), depth)


if __name__ == "__main__":
    test = tRecurrentStack()
    test.setUp()
    test.test_one_step()
    test.test_many_steps()
    # test_recurrentstack_sequence_generator()