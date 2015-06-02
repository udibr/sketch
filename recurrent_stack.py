from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.bricks import Initializable, Linear
from blocks.bricks.base import application
from blocks.bricks.parallel import Fork

class RecurrentStack(BaseRecurrent, Initializable):
    u"""Stack of recurrent networks.

    Build a stack of recurrent layers from a supplied list of BaseRecurrent
    objects. Each object must have a `sequences`, `contexts`, `states` and
    `outputs` parameters to its `apply` method.
    It is assumed that the `sequences` of all objects has an "inputs" element.
    It is assumed that all `states` have a "states" element.

    The `sequences` (the "inputs") of the the first layer (layer 0) is used
    as the `sequence` of the stack (the instance of this class.)
    The "masks" element of the `sequences` is carried to all layers that have
    a "masks" element in their `sequences`.
    For all the following layers, all elements of the `sequences`, except for
    "masks", are forked from the "states" element of the of the `states` of the
    previous layer.

    The concatenated `states` of all layers is the `states` of the stack.
    In order to avoid conflict, the names of the elements appearing in the
    `states` of each layers are suffixed with a '_<n>' before combining them.
    The '<n>' is the number of the layer starting from 0.
    The same process is true for the `contexts` and `outputs` of the stack.

    Parameters
    ----------
    transitions : list
        List of recurrent units to use in each layer.
    prototype : :class:`.FeedForward`, optional
        The transformation applied to each states of each layer  when it is
         forked to the sequences of the next layer. By
        default a :class:`.Linear` transformation with biases is used.

    Notes
    -----
    See :class:`.BaseRecurrent` for more initialization parameters.

    """
    def __init__(self, transitions, prototype=None, **kwargs):
        super(RecurrentStack, self).__init__(**kwargs)

        for d, transition in enumerate(transitions):
            transition.name += '_%d'%d
        self.transitions = transitions

        if prototype is None:
            # By default use Linear (with bias) as a default fork.
            # This overrides the bad default inside Fork which is without bias.
            # Yes I know use_bias=True us the defaut, but it is a big deal so
            # I write it down explicitly.
            prototype = Linear(use_bias=True)
        depth = len(transitions)
        self.forks=[Fork(self.normal_inputs(d), prototype=prototype)
                    for d in range(1,depth)]

        self.children = self.transitions + self.forks

        # Programmatically fix the parameters of the  recurrent decoration
        setattr(self.apply, 'sequences', transitions[0].apply.sequences)
        for t in ["states", "contexts", "outputs"]:
            v = ['%s_%d'%(s,d)
                 for d, transition in enumerate(transitions)
                 for s in getattr(transition.apply,t)]
            setattr(self.apply, t, v)

    def normal_inputs(self, d):
        transition = self.transitions[d]
        return [name for name in transition.apply.sequences if 'mask' not in name]

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
            fork.input_dim = self.transitions[d].get_dim("states")
            fork.output_dims = self.transitions[d+1].get_dims(fork.output_names)

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
        last_states = None
        results = []
        for d, transition in enumerate(self.transitions):
            if d == 0:
                layer_kwargs = dict(
                    (s,kwargs.get(s)) for s in transition.apply.sequences)
            else:
                layer_kwargs = dict(zip(
                    self.normal_inputs(d),
                    self.forks[d-1].apply(last_states, as_list=True)))
                if "masks" in transition.apply.sequences:
                    layer_kwargs["masks"] = kwargs.get("masks")

            for t in ["states", "contexts", "outputs"]:
                suffix = '_%d'%d
                for s in getattr(transition.apply, t):
                    layer_kwargs[s] = kwargs.get(s + suffix)

            result = transition.apply(iterate=False, **layer_kwargs)
            results.extend(result)

            # find the "states" member in the outputs and pass it to next layer
            last_states = result[transition.apply.outputs.index('states')]
        return tuple(results)

    def get_dim(self, name):
        if name in self.transitions[0].apply.sequences:
            return self.transitions[0].get_dim(name)
        name, layer = name.split('_')
        transition = self.transitions[int(layer)]
        return transition.get_dim(name)

    @application
    def initial_state(self, state_name, batch_size, *args, **kwargs):
        state_name,layer = state_name.split('_')
        transition = self.transitions[int(layer)]
        return transition.initial_state(state_name, batch_size,
                                        *args, **kwargs)
