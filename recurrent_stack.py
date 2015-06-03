from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.bricks import Initializable, Linear
from blocks.bricks.base import application
from blocks.bricks.parallel import Fork

class RecurrentStack(BaseRecurrent, Initializable):
    u"""Stack of recurrent networks.

    Build a stack of recurrent layers from a supplied list of
    BaseRecurrent objects. Each object must have a `sequences`,
    `contexts`, `states` and `outputs` parameters to its `apply` method.
    It is assumed that the `sequences` of all objects has an "inputs"
    element. It is assumed that all `states` have a "states" element
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
    sync : bool
        If true, the input of each layer is forked from the states of the
        lower layer from the previous step of the RNN time sequence.
        If false (async, default) the input is forked from the states
        output of the same step.
        Looking at other examples, e.g., it looks like async is the
        correct way.
       https://github.com/karpathy/char-rnn/blob/master/model/LSTM.lua#L25

    Notes
    -----
    See :class:`.BaseRecurrent` for more initialization parameters.

    """
    def __init__(self, transitions, prototype=None, sync=False,
                 states_name="states", **kwargs):
        super(RecurrentStack, self).__init__(**kwargs)

        self.sync = sync
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
        self.forks = [Fork(self.normal_inputs(d), prototype=prototype)
                      for d in range(1, depth)]

        self.children = self.transitions + self.forks

        # Programmatically fix the parameters of the  recurrent decoration
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

    @recurrent(sequences=[], states=[], outputs=[], contexts=[])
    def apply(self, **kwargs):
        """Apply the stack of transitions.

        Parameters
        ----------
        All parameters are of type :class:`~tensor.TensorVariable`.

        Returns
        -------
        All return values are of type :class:`~tensor.TensorVariable`.

        """
        results = []
        for d, transition in enumerate(self.transitions):
            sequences_names = transition.apply.sequences
            if d == 0:
                layer_kwargs = dict(
                    (s, kwargs.get(s)) for s in sequences_names)
            else:
                if self.sync:
                    last_states = kwargs[self.states_name + "_" + str(d-1)]
                layer_kwargs = dict(zip(
                    self.normal_inputs(d),
                    self.forks[d-1].apply(last_states, as_list=True)))
                if "mask" in sequences_names:
                    layer_kwargs["mask"] = kwargs.get("mask")

            for t in ["states", "contexts", "outputs"]:
                suffix = '_' + str(d)
                for s in getattr(transition.apply, t):
                    layer_kwargs[s] = kwargs.get(s + suffix)

            result = transition.apply(iterate=False, **layer_kwargs)
            results.extend(result)
            if not self.sync:
                i = transition.apply.outputs.index(self.states_name)
                last_states = result[i]

        return tuple(results)

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
        depth = 2
        self.depth = depth
        self.sync = False
        dim = 3
        tarnsitions = [LSTM(dim=dim) for _ in range(depth)]
        self.stack = RecurrentStack(tarnsitions,
                                    weights_init=Constant(2),
                                    biases_init=Constant(0))
        self.stack.initialize()

    def test_one_step(self):
        depth = self.depth
        kwargs = OrderedDict()
        kwargs['inputs'] = tensor.matrix('inputs')

        for d in range(depth):
            kwargs['states_' + str(d)] = tensor.matrix('states_' + str(d))
            kwargs['cells_' + str(d)] = tensor.matrix('cells_' + str(d))
        results = self.stack.apply(iterate=False, **kwargs)
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
            if self.sync:
                # current layer input state transformed to input of next
                x_v = numpy.dot(h0_v, W_state2x_val)

            i_t = sigmoid(activation[:, :3] + c0_v * W_cell_to_in)
            f_t = sigmoid(activation[:, 3:6] + c0_v * W_cell_to_forget)
            next_cells = f_t * c0_v + i_t * numpy.tanh(activation[:, 6:9])
            o_t = sigmoid(activation[:, 9:12] +
                          next_cells * W_cell_to_out)
            h1_v = o_t * numpy.tanh(next_cells)
            if not self.sync:
                # current layer output state transformed to input of next
                x_v = numpy.dot(h1_v, W_state2x_val)

            h1_val.append(h1_v)

        res = next_h(*args_val)
        for d in range(depth):
            assert_allclose(h1_val[d], res[d*2], rtol=1e-6)

    def test_many_steps(self):
        depth = self.depth

        kwargs = OrderedDict()
        kwargs['inputs'] = tensor.tensor3('inputs')
        kwargs['mask'] = tensor.matrix('mask')
        results = self.stack.apply(iterate=True, **kwargs)
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
                if self.sync:
                    # current layer input state transformed to input of next
                    x_v = numpy.dot(h_v, W_state2x_val)
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
                if not self.sync:
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

if __name__ == "__main__":
    test = tRecurrentStack()
    test.setUp()
    test.test_one_step()
    test.test_many_steps()