import abc


class AbstractReplayBuffer(abc.ABC):
    @abc.abstractmethod
    def __init__(self, buffer_size: int, **kwargs):
        pass

    @abc.abstractmethod
    def sample(self, batch_size: int):
        """
        Sample a batch of transitions from the buffer.

        :param batch_size: The number of transitions to sample.
        :type batch_size: int

        :returns: A batch of sampled transitions.
        :rtype: TensorDict
        """

    @abc.abstractmethod
    def extend(self, transition):
        """
        Add a transition to the buffer.

        :param transition: The transition to add to the buffer.
        :type transition: TensorDict
        """

    @abc.abstractmethod
    def update(self, idx, transition):
        """
        Update a transition in the buffer.

        :param idx: The index of the transition to update.
        :type idx: int

        :param transition: The updated transition.
        :type transition: TensorDict
        """

    @abc.abstractmethod
    def __len__(self):
        """
        Return the number of transitions stored in the buffer.

        :returns: The number of transitions stored.
        :rtype: int
        """
