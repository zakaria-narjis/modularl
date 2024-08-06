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

    :param buffer_size: The maximum capacity of the replay buffer.
    :type buffer_size: int

    :param sampling: The type of sampling to use. Options are 'random', 'prioritized', and 'without_replacement'.
    :type sampling: str

    :param \\**kwargs: Additional keyword arguments to be passed to the base class.

    :attributes:
        storage (LazyMemmapStorage): The storage object for storing transitions.
        sampler (Sampler): The sampler object for sampling transitions.
        buffer (TensorDictReplayBuffer): The buffer object for managing the storage and sampling.
    """  # noqa

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
            storage=self.storage, sampler=self.sampler, **kwargs
        )

    def sample(self, batch_size: int):

        sample = self.buffer.sample(batch_size)
        return sample

    def extend(self, transition):

        self.buffer.extend(transition)

    def update(self, idx, transition):

        self.buffer[idx] = transition

    def __len__(self):
        return len(self.buffer)
