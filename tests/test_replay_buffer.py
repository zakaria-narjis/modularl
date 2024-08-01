# test_replay_buffer.py

import pytest
import torch
from modularl.replay_buffers import ReplayBuffer
from tensordict import TensorDict


@pytest.fixture
def replay_buffer():
    return ReplayBuffer(buffer_size=1000, sampling="random")


def test_replay_buffer_init(replay_buffer):
    assert isinstance(replay_buffer, ReplayBuffer)
    assert replay_buffer.buffer.storage.max_size == 1000


def test_replay_buffer_extend(replay_buffer):
    transition = TensorDict(
        {
            "observations": torch.randn(10, 4),
            "actions": torch.randn(10, 2),
            "rewards": torch.randn(10),
            "next_observations": torch.randn(10, 4),
            "dones": torch.randint(0, 2, (10,)),
        },
        batch_size=[10],
    )

    initial_size = len(replay_buffer.buffer)
    replay_buffer.extend(transition)
    assert len(replay_buffer.buffer) == initial_size + 10


def test_replay_buffer_sample(replay_buffer):
    # Fill the buffer with some data
    for _ in range(5):
        transition = TensorDict(
            {
                "observations": torch.randn(10, 4),
                "actions": torch.randn(10, 2),
                "rewards": torch.randn(10),
                "next_observations": torch.randn(10, 4),
                "dones": torch.randint(0, 2, (10,)),
            },
            batch_size=[10],
        )
        replay_buffer.extend(transition)

    sample = replay_buffer.sample(32)
    assert isinstance(sample, TensorDict)
    assert sample.batch_size == torch.Size([32])
    assert "observations" in sample
    assert "actions" in sample
    assert "rewards" in sample
    assert "next_observations" in sample
    assert "dones" in sample


def test_replay_buffer_update(replay_buffer):
    # Add some data to the buffer
    transition = TensorDict(
        {
            "observations": torch.randn(10, 4),
            "actions": torch.randn(10, 2),
            "rewards": torch.randn(10),
            "next_observations": torch.randn(10, 4),
            "dones": torch.randint(0, 2, (10,)),
        },
        batch_size=[10],
    )
    replay_buffer.extend(transition)

    # Update a specific transition
    new_transition = TensorDict(
        {
            "observations": torch.ones(1, 4),
            "actions": torch.ones(1, 2),
            "rewards": torch.ones(1),
            "next_observations": torch.ones(1, 4),
            "dones": torch.ones(1),
        },
        batch_size=[1],
    )

    replay_buffer.update(0, new_transition)

    # Check if the update was successful
    updated_sample = replay_buffer.buffer[0]
    assert torch.all(updated_sample["observations"].eq(1))
    assert torch.all(updated_sample["actions"].eq(1))
    assert torch.all(updated_sample["rewards"].eq(1))
    assert torch.all(updated_sample["next_observations"].eq(1))
    assert torch.all(updated_sample["dones"].eq(1))
