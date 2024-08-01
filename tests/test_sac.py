# test_sac.py

import pytest
import torch
import torch.nn as nn
from modularl.agents.sac import SAC
from modularl.replay_buffers import ReplayBuffer


class DummyActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(obs_dim, action_dim)

    def forward(self, x):
        return torch.tanh(self.fc(x))

    def get_action(self, x):
        action = self.forward(x)
        return action, torch.zeros_like(action), None


class DummyQFunction(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc = nn.Linear(obs_dim + action_dim, 1)

    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=-1)
        return self.fc(x)


@pytest.fixture
def sac_agent():
    obs_dim = 4
    action_dim = 2
    actor = DummyActor(obs_dim, action_dim)
    qf1 = DummyQFunction(obs_dim, action_dim)
    qf2 = DummyQFunction(obs_dim, action_dim)
    actor_optimizer = torch.optim.Adam(actor.parameters())
    q_optimizer = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()))
    replay_buffer = ReplayBuffer(buffer_size=1000)

    return SAC(
        actor=actor,
        qf1=qf1,
        qf2=qf2,
        actor_optimizer=actor_optimizer,
        q_optimizer=q_optimizer,
        replay_buffer=replay_buffer,
        batch_size=32,
        learning_starts=100,
        device="cpu",
    )


def test_sac_init(sac_agent):
    assert isinstance(sac_agent, SAC)
    assert sac_agent.device == "cpu"
    assert sac_agent.batch_size == 32
    assert sac_agent.learning_starts == 100


def test_sac_observe(sac_agent):
    batch_size = 10
    obs_dim = 4
    action_dim = 2

    batch_obs = torch.randn(batch_size, obs_dim)
    batch_actions = torch.randn(batch_size, action_dim)
    batch_rewards = torch.randn(batch_size)
    batch_next_obs = torch.randn(batch_size, obs_dim)
    batch_dones = torch.randint(0, 2, (batch_size,))

    initial_rb_size = len(sac_agent.rb)
    sac_agent.observe(
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones
    )
    assert len(sac_agent.rb) == initial_rb_size + batch_size


def test_sac_act_train(sac_agent):
    batch_size = 5
    obs_dim = 4

    batch_obs = torch.randn(batch_size, obs_dim)
    actions = sac_agent.act_train(batch_obs)

    assert actions.shape == (batch_size, 2)
    assert torch.all(actions >= -1) and torch.all(actions <= 1)


def test_sac_act_eval(sac_agent):
    batch_size = 5
    obs_dim = 4

    batch_obs = torch.randn(batch_size, obs_dim)
    actions = sac_agent.act_eval(batch_obs)

    assert actions.shape == (batch_size, 2)
    assert torch.all(actions >= -1) and torch.all(actions <= 1)


def test_sac_update(sac_agent):
    # Fill the replay buffer with some dummy data
    for _ in range(200):
        batch_obs = torch.randn(10, 4)
        batch_actions = torch.randn(10, 2)
        batch_rewards = torch.randn(10)
        batch_next_obs = torch.randn(10, 4)
        batch_dones = torch.randint(0, 2, (10,))
        sac_agent.observe(
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones
        )

    # Perform an update
    initial_actor_state = sac_agent.actor.state_dict()
    initial_qf1_state = sac_agent.qf1.state_dict()
    initial_qf2_state = sac_agent.qf2.state_dict()

    sac_agent.update()

    # Check if the parameters have been updated
    assert not torch.all(
        torch.eq(
            initial_actor_state["fc.weight"], sac_agent.actor.state_dict()["fc.weight"]
        )
    )
    assert not torch.all(
        torch.eq(
            initial_qf1_state["fc.weight"], sac_agent.qf1.state_dict()["fc.weight"]
        )
    )
    assert not torch.all(
        torch.eq(
            initial_qf2_state["fc.weight"], sac_agent.qf2.state_dict()["fc.weight"]
        )
    )
