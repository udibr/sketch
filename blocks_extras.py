"""
This module contains classes that should be merged at some point inside blocks
"""
import copy
import numpy
import theano
from blocks.initialization import NdarrayInitialization, Orthogonal

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


class OrthogonalGlorot(GlorotBengio):
    """Initialize a random orthogonal matrix.


    """
    def __init__(self, *args, **kwargs):
        super(OrthogonalGlorot,self).__init__(*args, **kwargs)
        self.orth = Orthogonal()

    def generate(self, rng, shape):
        if len(shape) == 1:
            return super(OrthogonalGlorot,self).generate(rng,shape)

        N = shape[0]
        M = shape[1] // N
        if M > 1 and len(shape) == 2 and shape[1] == M*N:
            res = [self.orth.generate(rng,(N,N)) for i in range(M)]
            return numpy.concatenate(res,axis=-1)

        return self.orth.generate(rng, shape)
