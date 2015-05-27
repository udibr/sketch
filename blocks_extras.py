"""
This module contains classes that should be merged at some point inside blocks
"""
import copy
import numpy
import theano
from theano import tensor
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.bricks import Initializable, Linear
from blocks.utils import shared_floatx
from blocks.algorithms import StepRule
from blocks.initialization import NdarrayInitialization
from blocks.bricks.recurrent import LSTM
from blocks.bricks.base import application

from blocks.utils import shared_floatx_zeros
from blocks.roles import add_role, INITIAL_STATE


# should be in blocks.initialization
class GlorotBengio(NdarrayInitialization):
    """Initialize parameters with Glorot-Bengio method.

    Use the following gaussian parameters: mean=0 and std=sqrt(scale/Nin).
    In some circles this method is also called Xavier weight initialization.


    Parameters
    ----------
    scale : float
        1 for linear/tanh/sigmoid. 2 for RELU
    normal : bool
        Perform sampling from normal distribution. By defaut use uniform.

    Notes
    -----
    For more information, see [GLOROT]_.

    .. [GLOROT] Glorot et al. *Understanding the difficulty of training
      deep feedforward neural networks*, International conference on
      artificial intelligence and statistics, 249-256
      http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    """
    def __init__(self, scale=1, normal=False):
        self._scale = float(scale)
        self._normal = normal

    def generate(self, rng, shape):
        w = numpy.sqrt(self._scale/shape[-1])
        if self._normal:
            m = rng.normal(0., w, size=shape)
        else:
            m = rng.uniform(-w, w, size=shape)
        return m.astype(theano.config.floatX)

# should be in blocks.algorithms
class AdaGrad(StepRule):
    """Implements the AdaGrad learning rule.

    Parameters
    ----------
    learning_rate : float, optional
        Step size.
        Default value is set to 0.0002.
    epsilon : float, optional
        Stabilizing constant for one over root of sum of squares.
        Defaults to 1e-6.

    Notes
    -----
    For more information, see [ADAGRAD]_.

    .. [ADADGRAD] Duchi J, Hazan E, Singer Y.,
       *Adaptive subgradient methods for online learning and
        stochastic optimization*,
       http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf

    """
    def __init__(self, learning_rate=0.002, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def compute_step(self, param, previous_step):
        name = 'adagrad_sqs'
        if param.name:
            name += '_' + param.name
        ssq = shared_floatx(param.get_value() * 0.,
                            name=name)

        ssq_t = (tensor.sqr(previous_step) + ssq)
        step = (self.learning_rate * previous_step /
                (tensor.sqrt(ssq_t) + self.epsilon))

        updates = [(ssq, ssq_t)]

        return step, updates

# should be in blocks.bricks.recurrent
class RecurrentLSTM(BaseRecurrent, Initializable):
    u"""Stack of LSTM networks.

    This class acts as a plug-and-play replacement for a single LSTM network.
    Its interface looks like a single big LSTM.
    This behaviour is different from Bidirectional
    which only acts as a limited wrapper.

    Build a stack of LSTM layers of the same size. The inputs are
    feed to layer 0, the states of each layer are also feed as inputs
    to the next layer through a linear transformation with bias.

    Parameters
    ----------
    name : str, optional
        The name of this brick. The name of each layer will be
        followed by the name of the prototype followed by the layer number.
        By default, the brick receives the name of its class (lowercased).
    depth : int, optional
        Number of layers. By default 2.

    Notes
    -----
    See :class:`.BaseRecurrent` for more initialization parameters.

    """
    def __init__(self, dim, activation=None, depth=2, name=None,
                 lstm_name=None, **kwargs):
        super(RecurrentLSTM, self).__init__(name=name, **kwargs)
        # use the name allready processed by superclass
        name = self.name
        self.dim = dim

        self.children = []
        self.depth = depth
        for d in range(self.depth):
            layer_node = LSTM(dim, activation, name=lstm_name)
            layer_node.name = '%s_%s_%d'%(name, layer_node.name, d)

            if d > 0:
                # convert states of previous layer to inputs of new layer
                layer_name = '%s_%d_%d'%(name, d-1, d)
                input_dim = layer_node.get_dim('inputs')
                self.children.append(Linear(dim, input_dim,
                                            use_bias=True,
                                            name=layer_name))
            self.children.append(layer_node)

    @recurrent(sequences=['inputs', 'mask'], states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells'])
    def apply(self, inputs, states, cells, mask=None):
        """Apply the stack of Long Short Term Memory transition.

        Parameters
        ----------
        states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current states in the shape
            (batch_size, dim*depth). Required for `one_step` usage.
        cells : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current cells in the shape
            (batch_size, dim*depth). Required for `one_step` usage.
            The cells are also used as input to the next layer thorough a
            linear transformation with bias.
        inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs in the shape (batch_size,
            dim * 4). The inputs are feed to layer 0.
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            data available, 0 if not. Assumed to be 1-s only if not given.

        Returns
        -------
        states : :class:`~tensor.TensorVariable`
            Next states of the network.
        cells : :class:`~tensor.TensorVariable`
            Next cell activations of the network.

        """
        def slice_last(x, no):
            return x.T[no*self.dim: (no+1)*self.dim].T

        next_inputs = inputs
        next_states = []
        next_cells = []
        for d in range(self.depth):
            current_states = slice_last(states,d)
            current_cells = slice_last(cells,d)

            if d == 0:
                # SequenceGenerator has already built a linear transform in Fork
                current_inputs = next_inputs
            else:
                # convert cells of previous layer to input of new layer
                current_inputs = self.children[2*d-1].apply(next_inputs)

            current_next_states, current_next_cells = self.children[2*d].apply(
                inputs=current_inputs,
                states=current_states,
                cells=current_cells,
                mask=mask,
                iterate=False)
            next_states.append(current_next_states)
            next_cells.append(current_next_cells)

            next_inputs = current_states

        next_states = theano.tensor.concatenate(next_states, axis=-1)
        next_cells = theano.tensor.concatenate(next_cells, axis=-1)
        return next_states, next_cells

    def get_dim(self, name):
        if name in ['inputs', 'mask']:
            return self.children[0].get_dim(name)
        return self.children[0].get_dim(name) * self.depth

    @application
    def initial_state(self, state_name, batch_size, *args, **kwargs):
        states = []
        for d in range(self.depth):
            states.append(self.children[2*d].initial_state(state_name,
                                                           batch_size,
                                                           *args,
                                                           **kwargs))
        return theano.tensor.concatenate(states, axis=-1)

    # def _allocate(self):
    #     # The underscore is required to prevent collision with
    #     # the `initial_state` application method
    #     self.initial_state_ = shared_floatx_zeros((self.dim*self.depth,),
    #                                               name="initial_state")
    #     self.initial_cells = shared_floatx_zeros((self.dim*self.depth,),
    #                                              name="initial_cells")
    #     add_role(self.initial_state_, INITIAL_STATE)
    #     add_role(self.initial_cells, INITIAL_STATE)
    #
    #     self.params = [self.initial_state_, self.initial_cells]
    #
    # @application
    # def initial_state(self, state_name, batch_size, *args, **kwargs):
    #     if state_name == "states":
    #         return tensor.repeat(self.initial_state_[None, :], batch_size, 0)
    #     elif state_name == "cells":
    #         return tensor.repeat(self.initial_cells[None, :], batch_size, 0)
    #     raise ValueError("unknown state name " + state_name)


class RecurrentStack(BaseRecurrent, Initializable):
    u"""Stack of recurrent networks.

    This is a wrapper for a single layer recurrent network,
    similar to how Bidirectional acts. As a result you can not perform  a
    single step on the network.

    Build a stack of Recurrent layers of the same size. The inputs are
    feed to layer 0, the states of each layer are also feed as inputs
    to the next layer through a linear transformation with bias.

    Parameters
    ----------
    name : str, optional
        The name of this brick. The name of each layer will be
        followed by the name of the prototype followed by the layer number.
        By default, the brick receives the name of its class (lowercased).
    depth : int, optional
        Number of layers. By default 2.
    prototype : :class:`~blocks.bricks.recurrent.BaseRecurrent`
        A transformation prototype. A copy will be created for every
        input.  If ``None``, an  LSTM is used.

    Notes
    -----
    See :class:`.BaseRecurrent` for more initialization parameters.

    """
    def __init__(self, dim, activation=None, depth=2, name=None,
                 prototype=None, **kwargs):
        super(RecurrentLSTM, self).__init__(name=name, **kwargs)
        # use the name allready processed by superclass
        name = self.name
        self.dim = dim
        if not prototype:
            prototype = LSTM(dim, activation)
        self.prototype = prototype
        input_dim = prototype.get_dim('inputs')

        self.children = []
        self.depth = depth
        for d in range(self.depth):
            if d > 0:
                # convert cells of previous layer to input of new layer
                layer_name = '%s_%d_%d'%(name, d-1, d)
                self.children.append(Linear(dim, input_dim, use_bias=True,
                                            name=layer_name))
            layer_node = copy.deepcopy(self.prototype)
            layer_node.name = '%s_%s_%d'%(name, layer_node.name, d)
            self.children.append(layer_node)

    @recurrent(sequences=['inputs', 'mask'], states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells'])
    def apply(self, inputs, states, cells, mask=None):
        """Apply the stack of Long Short Term Memory transition.

        Parameters
        ----------
        states : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current states in the shape
            (batch_size, dim*depth). Required for `one_step` usage.
        cells : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of current cells in the shape
            (batch_size, dim*depth). Required for `one_step` usage.
            The cells are also used as input to the next layer thorough a
            linear transformation with bias.
        inputs : :class:`~tensor.TensorVariable`
            The 2 dimensional matrix of inputs in the shape (batch_size,
            dim * 4). The inputs are feed to layer 0.
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if there is
            data available, 0 if not. Assumed to be 1-s only if not given.

        Returns
        -------
        states : :class:`~tensor.TensorVariable`
            Next states of the network.
        cells : :class:`~tensor.TensorVariable`
            Next cell activations of the network.

        """
        def slice_last(x, no):
            return x.T[no*self.dim: (no+1)*self.dim].T

        next_inputs = inputs
        next_states = []
        next_cells = []
        for d in range(self.depth):
            current_states = slice_last(states,d)
            current_cells = slice_last(cells,d)

            if d == 0:
                # SequenceGenerator has already built a linear transform in Fork
                current_inputs = next_inputs
            else:
                # convert cells of previous layer to input of new layer
                current_inputs = self.children[2*d-1].apply(next_inputs)

            current_next_states, current_next_cells = self.children[2*d].apply(
                inputs=current_inputs,
                states=current_states,
                cells=current_cells,
                mask=mask,
                iterate=False)
            next_states.append(current_next_states)
            next_cells.append(current_next_cells)

            next_inputs = current_states

        next_states = theano.tensor.concatenate(next_states, axis=-1)
        next_cells = theano.tensor.concatenate(next_cells, axis=-1)
        return next_states, next_cells

    def get_dim(self, name):
        if name in ['inputs', 'mask']:
            return self.children[0].get_dim(name)
        return self.children[0].get_dim(name) * self.depth

    def _allocate(self):
        # The underscore is required to prevent collision with
        # the `initial_state` application method
        self.initial_state_ = shared_floatx_zeros((self.dim*self.depth,),
                                                  name="initial_state")
        self.initial_cells = shared_floatx_zeros((self.dim*self.depth,),
                                                 name="initial_cells")
        add_role(self.initial_state_, INITIAL_STATE)
        add_role(self.initial_cells, INITIAL_STATE)

        self.params = [self.initial_state_, self.initial_cells]

    # @application
    # def initial_state(self, state_name, batch_size, *args, **kwargs):
    #     states = []
    #     for d in range(self.depth):
    #         states.append(self.children[2*d].initial_state(state_name,
    #                                                        batch_size,
    #                                                        *args,
    #                                                        **kwargs))
    #     return theano.tensor.concatenate(states, axis=-1)

    @application
    def initial_state(self, state_name, batch_size, *args, **kwargs):
        if state_name == "states":
            return tensor.repeat(self.initial_state_[None, :], batch_size, 0)
        elif state_name == "cells":
            return tensor.repeat(self.initial_cells[None, :], batch_size, 0)
        raise ValueError("unknown state name " + state_name)
