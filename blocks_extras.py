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

import os.path
import logging

from blocks.extensions import SimpleExtension, TrainingExtension
from blocks.utils import reraise_as

logger = logging.getLogger(__name__)

LOADED_FROM = "loaded_from"
SAVED_TO = "saved_to"
from six.moves import cPickle

# from blocks.serialization import pickle_dump
def save_parameter_values(param_values, path):
    """Compactly save parameter values.

    This is a thin wrapper over `numpy.savez`. It deals with
    `numpy`'s vulnerability to slashes in file names.

    Parameters
    ----------
    param_values : dict of (parameter name, numpy array)
        The parameter values.
    path : str of file
        The destination for saving.

    """
    param_values = {name.replace("/", "-"): param
                    for name, param in param_values.items()}
    numpy.savez(path, **param_values)


def load_parameter_values(path):
    """Load parameter values saved by :func:`save_parameters`.

    This is a thin wrapper over `numpy.load`. It deals with
    `numpy`'s vulnerability to slashes in file names.

    Parameters
    ----------
    path : str or file
        The source for loading from.

    Returns
    -------
    A dictionary of (parameter name, numpy array) pairs.

    """
    source = numpy.load(path)
    param_values = {name.replace("-", "/"): value
                    for name, value in source.items()}
    source.close()
    return param_values


class MainLoopDumpManager(object):
    """Main loop dumping implementation.

    This class provides saving and loading logic that circumvents
    serialization of the most problematic parts: the model and the training
    algorithm (which typically has Theano functions as attributes). The
    on-disk representation used is a folder with a few files containing
    model parameters, log and state of the data iteration.

    Also see the module-level documentation.

    Parameters
    ----------
    folder : str
        The path to the dump root folder.

    """
    def __init__(self, folder):
        self.folder = folder

    @property
    def path_to_parameters(self):
        return os.path.join(self.folder, 'params.npz')

    @property
    def path_to_iteration_state(self):
        return os.path.join(self.folder, 'iterations_state.pkl')

    @property
    def path_to_log(self):
        # The extension is omitted for the log because advanced
        # log classes might have a better format for storing on the disk
        # then pickled file. Or alternatively, log will be dump as pure
        # text file of (time, key, value) triples. Currenly log is just
        # pickled though.
        return os.path.join(self.folder, 'log')

    def dump_parameters(self, main_loop):
        save_parameter_values(main_loop.model.get_param_values(),
                              self.path_to_parameters)

    def dump_iteration_state(self, main_loop):
        with open(self.path_to_iteration_state, "wb") as destination:
            cPickle.dump(main_loop.iteration_state, destination)

    def dump_log(self, main_loop):
        with open(self.path_to_log, "wb") as destination:
            cPickle.dump(main_loop.log, destination)

    def dump(self, main_loop):
        """Dumps the main loop to the root folder.

        See :mod:`blocks.dump`.

        Overwrites the old data if present.

        """
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        self.dump_parameters(main_loop)
        self.dump_iteration_state(main_loop)
        self.dump_log(main_loop)

    def load_parameters(self):
        return load_parameter_values(self.path_to_parameters)

    def load_iteration_state(self):
        with open(self.path_to_iteration_state, "rb") as source:
            return cPickle.load(source)

    def load_log(self):
        with open(self.path_to_log, "rb") as source:
            return cPickle.load(source)

    def load(self):
        return (self.load_parameters(),
                self.load_iteration_state(),
                self.load_log())

    def load_to(self, main_loop):
        """Loads the dump from the root folder into the main loop."""
        parameters, iteration_state, log = self.load()
        main_loop.model.set_param_values(parameters)
        main_loop.iteration_state = iteration_state
        main_loop.log = log

class LoadFromDump(TrainingExtension):
    """Loads a dump into the main loop.

    Makes a `LOADED_FROM` record in the log with the dump path.

    Parameters
    ----------
    state_path : str
        The path to the folder with dump.

    Notes
    -----
    Requires the model to be a Brick or a list of Bricks.

    """
    def __init__(self, state_path, **kwargs):
        super(LoadFromDump, self).__init__(**kwargs)
        self.manager = MainLoopDumpManager(state_path)

    def before_training(self):
        if not os.path.exists(self.manager.folder):
            logger.info("No dump found")
            return
        logger.info("Loading the state from {} into the main loop"
                    .format(self.manager.folder))
        try:
            self.manager.load_to(self.main_loop)
            self.main_loop.log.current_row[LOADED_FROM] = self.manager.folder
        except Exception:
            reraise_as("Failed to load the state")


class Dump(SimpleExtension):
    """Dumps the state of the main loop.

    Makes a `SAVED_TO` record in the log with the dumping destination
    in the case of success and ``None`` in the case of failure.

    Parameters
    ----------
    state_path : str
        The folder to dump the state to. Will be created it does not
        exist.

    Notes
    -----
    Requires the model to be a Brick or a list of Bricks.

    """
    def __init__(self, state_path, **kwargs):
        kwargs.setdefault("after_training", True)
        super(Dump, self).__init__(**kwargs)
        self.manager = MainLoopDumpManager(state_path)

    def do(self, callback_name, *args, **kwargs):
        try:
            self.main_loop.log.current_row[SAVED_TO] = (
                self.manager.folder)
            self.manager.dump(self.main_loop)
        except Exception:
            self.main_loop.log.current_row[SAVED_TO] = None
            raise
