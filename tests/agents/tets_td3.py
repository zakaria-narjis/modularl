import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from modularl.agents.td3 import TD3
from modularl.replay_buffers import ReplayBuffer
import copy


class DummyActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class DummyCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, x, action):
        x = torch.cat([x, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


@pytest.fixture
def td3_agent():
    obs_dim = 4
    action_dim = 2
    actor = DummyActor(obs_dim, action_dim)
    critic1 = DummyCritic(obs_dim, action_dim)
    critic2 = DummyCritic(obs_dim, action_dim)
    replay_buffer = ReplayBuffer(obs_dim, action_dim, max_size=1000)
    agent = TD3(actor, critic1, critic2, replay_buffer)
    return agent


def test_td3_agent(td3_agent):
    initial_actor_state = copy.deepcopy(td3_agent.actor.state_dict())
    initial_critic1_state = copy.deepcopy(td3_agent.critic1.state_dict())
    initial_critic2_state = copy.deepcopy(td3_agent.critic2.state_dict())

    batch_obs = torch.randn(10, 4)
    batch_actions = torch.randn(10, 2)
    batch_rewards = torch.randn(10)
    batch_next_obs = torch.randn(10, 4)
    batch_dones = torch.randint(0, 2, (10,))

    td3_agent.observe(
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones
    )

    def check_state_dict_unchanged(initial_state, current_state):
        for key in initial_state:
            if not torch.allclose(initial_state[key], current_state[key]):
                return False
        return True

    assert check_state_dict_unchanged(
        initial_actor_state, td3_agent.actor.state_dict()
    ), "Actor parameters were updated before learning_starts"
    assert check_state_dict_unchanged(
        initial_critic1_state, td3_agent.critic1.state_dict()
    ), "Critic1 parameters were updated before learning_starts"
    assert check_state_dict_unchanged(
        initial_critic2_state, td3_agent.critic2.state_dict()
    ), "Critic2 parameters were updated before learning_starts"

    for _ in range(100):
        td3_agent.observe(
            batch_obs,
            batch_actions,
            batch_rewards,
            batch_next_obs,
            batch_dones,
        )

    def check_state_dict_updated(initial_state, updated_state, tolerance=1e-4):
        for key in initial_state:
            if not torch.allclose(
                initial_state[key], updated_state[key], atol=tolerance
            ):
                return True
        return False

    assert check_state_dict_updated(
        initial_actor_state, td3_agent.actor.state_dict()
    ), "Actor parameters were not updated after learning_starts"
    assert check_state_dict_updated(
        initial_critic1_state, td3_agent.critic1.state_dict()
    ), "Critic1 parameters were not updated after learning_starts"
    assert check_state_dict_updated(
        initial_critic2_state, td3_agent.critic2.state_dict()
    ), "Critic2 parameters were not updated after learning_starts"
