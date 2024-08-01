from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.data.replay_buffers.samplers import (
    SamplerWithoutReplacement,
    RandomSampler,
    PrioritizedSampler,
)
from modularl.replay_buffers.base_buffer import AbstractReplayBuffer


class ReplayBuffer(AbstractReplayBuffer):
    """
    A replay buffer for storing and sampling transitions for reinforcement learning.

    Args:
        buffer_size (int): The maximum capacity of the replay buffer.
        sampling (str): The type of sampling to use. Options are 'random', 'prioritized', and 'without_replacement'.
        **kwargs: Additional keyword arguments to be passed to the base class.

    Attributes:
        storage (LazyMemmapStorage): The storage object for storing transitions.
        sampler (Sampler): The sampler object for sampling transitions.
        buffer (TensorDictReplayBuffer): The buffer object for managing the storage and sampling.

    Methods:
        sample(batch_size): Sample a batch of transitions from the buffer.
        extend(transition): Add a transition to the buffer.
        update(idx, transition): Update a transition in the buffer.

    """  # noqa: E501

    def __init__(self, buffer_size: int, sampling="random", **kwargs):
        super().__init__(buffer_size, **kwargs)
        self.storage = LazyMemmapStorage(buffer_size)
        if sampling == "random":
            self.sampler = RandomSampler()
        elif sampling == "prioritized":
            self.sampler = PrioritizedSampler()
        elif sampling == "without_replacement":
            self.sampler = SamplerWithoutReplacement()
        self.buffer = TensorDictReplayBuffer(
            storage=self.storage, sampler=self.sampler
        )

    def sample(self, batch_size: int):
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            sample (TensorDict): A batch of sampled transitions.

        """
        sample = self.buffer.sample(batch_size)
        return sample

    def extend(self, transition):
        """
        Add a transition to the buffer.

        Args:
            transition (TensorDict): The transition to add to the buffer.

        """
        self.buffer.extend(transition)

    def update(self, idx, transition):
        """
        Update a transition in the buffer.

        Args:
            idx (int): The index of the transition to update.
            transition (TensorDict): The updated transition.

        """
        self.buffer[idx] = transition

    def __len__(self):
        return len(self.buffer)
