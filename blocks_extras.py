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

# should be in blocks.initialization
class GlorotBengio(NdarrayInitialization):
    """Initialize parameters with Glorot-Bengio method.

    Use the following gaussian parameters: mean=0 and std=sqrt(scale/Nin).
    In some circles this method is also called Xavier weight initialization.


    Parameters
    ----------
    scale : float
        1 for linear/tanh/sigmoid. 2 for RELU

    Notes
    -----
    For more information, see [GLOROT]_.

    .. [GLOROT] Glorot et al. *Understanding the difficulty of training
      deep feedforward neural networks*, International conference on
      artificial intelligence and statistics, 249-256
      http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    """
    def __init__(self, scale=1):
        self._scale = float(scale)

    def generate(self, rng, shape):
        std = numpy.sqrt(self._scale/shape[-1])
        m = rng.normal(0., std, size=shape)
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
        ssq = shared_floatx(param.get_value() * 0.,
                            name='ssq_' + param.name)

        ssq_t = (tensor.sqr(previous_step) + ssq)
        step = (self.learning_rate * previous_step /
                (tensor.sqrt(ssq_t) + self.epsilon))

        updates = [(ssq, ssq_t)]

        return step, updates

# should be in blocks.bricks.recurrent
class RecurrentStack(BaseRecurrent, Initializable):
    u"""Stack of recurrent networks.

    Build a stack of Recurrent layers of the same size. The inputs are
    feed to layer 0, the cells of each layer are also feed as input to the next
    layer through a linear transformation with bias.

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
        super(RecurrentStack, self).__init__(name=name, **kwargs)
        self.dim = dim
        if not prototype:
            prototype = LSTM(dim, activation)
        self.prototype = prototype
        input_dim = prototype.get_dim('inputs')

        self.children = []
        self.depth = depth
        for d in range(self.depth):
            if d > 0:
                layer_name = '%s_%d_%d'%(self.name,d-1,d)
                self.children.append(Linear(dim, input_dim, use_bias=True,
                                            name=layer_name))
            layer_node = copy.deepcopy(self.prototype)
            # use the name allready processed by superclass
            layer_node.name = '%s_%s_%d'%(self.name, layer_node.name, d)
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

        last_cells = None
        next_states = []
        next_cells = []
        for d in range(self.depth):
            current_states = slice_last(states,d)
            current_cells = slice_last(cells,d)

            if d == 0:
                current_inputs = inputs
                current_mask = mask
            else:
                current_inputs = self.children[2*d-1].apply(last_cells)
                current_mask = None

            current_next_states, current_next_cells = self.children[2*d].apply(
                inputs=current_inputs,
                states=current_states,
                cells=current_cells,
                mask=current_mask,
                iterate=False)
            next_states.append(current_next_states)
            next_cells.append(current_next_cells)

            last_cells = current_cells

        next_states = theano.tensor.concatenate(next_states, axis=-1)
        next_cells = theano.tensor.concatenate(next_cells, axis=-1)
        return next_states, next_cells

    def get_dim(self, name):
        if name in ['inputs', 'mask']:
            return self.children[0].get_dim(name)
        return self.children[0].get_dim(name) * self.depth
