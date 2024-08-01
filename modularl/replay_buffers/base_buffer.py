import abc


class AbstractReplayBuffer(abc.ABC):
    @abc.abstractmethod
    def __init__(self, buffer_size: int, **kwargs):
        pass

    @abc.abstractmethod
    def sample(self, batch_size: int):
        """Sample a batch of experiences."""
        pass

    @abc.abstractmethod
    def extend(self, transition):
        """Add a transition to the buffer."""
        pass

    @abc.abstractmethod
    def update(self, idx, transition):
        """Update a transition in the buffer."""
        pass

    @abc.abstractmethod
    def __len__(self):
        """Return the number of transitions stored in the buffer."""
        pass
