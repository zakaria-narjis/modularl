import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from modularl.policies.deterministic_policy import DeterministicPolicy


class DummyPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


@pytest.fixture
def deterministic_policy():
    obs_dim = 4
    action_dim = 2
    policy = DummyPolicy(obs_dim, action_dim)
    return DeterministicPolicy(policy)


def test_deterministic_policy(deterministic_policy):
    obs = torch.randn(1, 4)
    action = deterministic_policy(obs)
    assert action.shape == (1, 2), "Action shape is incorrect"
    assert torch.all(action <= 1) and torch.all(
        action >= -1
    ), "Action values are out of bounds"

    obs_batch = torch.randn(10, 4)
    actions = deterministic_policy(obs_batch)
    assert actions.shape == (10, 2), "Batch action shape is incorrect"
    assert torch.all(actions <= 1) and torch.all(
        actions >= -1
    ), "Batch action values are out of bounds"
