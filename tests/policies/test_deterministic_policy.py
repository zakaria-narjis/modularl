import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from modularl.policies.deterministic_policy import DeterministicPolicy


class DummyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


@pytest.fixture
def deterministic_policy():
    obs_dim = 32
    action_dim = 10
    high_action = 2.0
    low_action = -2.0
    policy = DummyNetwork(obs_dim, 32)
    return DeterministicPolicy(
        observation_shape=obs_dim,
        action_shape=action_dim,
        high_action=high_action,
        low_action=low_action,
        policy=policy,
    )


def test_deterministic_policy(deterministic_policy):
    obs = torch.randn(1, 32)
    action = deterministic_policy(obs)
    assert action.shape == (1, 10), "Action shape is incorrect"
    assert torch.all(action <= 2) and torch.all(
        action >= -2
    ), "Action values are out of bounds"

    obs_batch = torch.randn(10, 32)
    actions = deterministic_policy(obs_batch)
    assert actions.shape == (10, 10), "Batch action shape is incorrect"
    assert torch.all(actions <= 2) and torch.all(
        actions >= -2
    ), "Batch action values are out of bounds"
