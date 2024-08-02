# test_sac.py

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modularl.agents.sac import SAC
from modularl.replay_buffers import ReplayBuffer
import copy

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class DummyActor(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        high_action=1.0,
        low_action=-1.0,
        use_xavier=True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 16 * obs_dim)
        self.fc2 = nn.Linear(16 * obs_dim, 16 * obs_dim)
        self.fc_mean = nn.Linear(16 * obs_dim, action_dim)
        self.fc_logstd = nn.Linear(16 * obs_dim, action_dim)

        self.register_buffer(
            "action_scale",
            torch.tensor(
                (high_action - low_action) / 2.0, dtype=torch.float32
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (high_action + low_action) / 2.0, dtype=torch.float32
            ),
        )

        if use_xavier:
            self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)


class DummyQFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, use_xavier=True):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 16 * (obs_dim + action_dim))
        self.fc2 = nn.Linear(
            16 * (obs_dim + action_dim), 16 * (obs_dim + action_dim)
        )
        self.fc3 = nn.Linear(16 * (obs_dim + action_dim), 1)

        if use_xavier:
            self._initialize_weights()

    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)


@pytest.fixture
def sac_agent():
    obs_dim = 4
    action_dim = 2
    actor = DummyActor(obs_dim, action_dim)
    qf1 = DummyQFunction(obs_dim, action_dim)
    qf2 = DummyQFunction(obs_dim, action_dim)
    actor_optimizer = torch.optim.Adam(actor.parameters())
    q_optimizer = torch.optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters())
    )
    replay_buffer = ReplayBuffer(buffer_size=1000)
    agent = SAC(
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
    agent.init()
    return agent


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
    assert actions.shape == (batch_size, 2)


def test_sac_act_eval(sac_agent):
    batch_size = 5
    obs_dim = 4

    batch_obs = torch.randn(batch_size, obs_dim)
    actions = sac_agent.act_eval(batch_obs)

    assert actions.shape == (batch_size, 2)
    assert actions.shape == (batch_size, 2)


def test_sac_update(sac_agent):
    # Fill the replay buffer with some dummy data
    for _ in range(100):  # One less than learning_starts
        batch_obs = torch.randn(10, 4)
        batch_actions = torch.randn(10, 2)
        batch_rewards = torch.randn(10)
        batch_next_obs = torch.randn(10, 4)
        batch_dones = torch.randint(0, 2, (10,))
        sac_agent.observe(
            batch_obs,
            batch_actions,
            batch_rewards,
            batch_next_obs,
            batch_dones,
        )

    # Check that parameters haven't been updated yet
    initial_actor_state = copy.deepcopy(sac_agent.actor.state_dict())
    initial_qf1_state = copy.deepcopy(sac_agent.qf1.state_dict())
    initial_qf2_state = copy.deepcopy(sac_agent.qf2.state_dict())

    def check_state_dict_unchanged(initial_state, current_state):
        for key in initial_state:
            if not torch.allclose(initial_state[key], current_state[key]):
                return False
        return True

    assert check_state_dict_unchanged(
        initial_actor_state, sac_agent.actor.state_dict()
    ), "Actor parameters were updated before learning_starts"
    assert check_state_dict_unchanged(
        initial_qf1_state, sac_agent.qf1.state_dict()
    ), "QF1 parameters were updated before learning_starts"
    assert check_state_dict_unchanged(
        initial_qf2_state, sac_agent.qf2.state_dict()
    ), "QF2 parameters were updated before learning_starts"

    # Perform one more observation to trigger the update
    for _ in range(100):
        batch_obs = torch.randn(10, 4)
        batch_actions = torch.randn(10, 2)
        batch_rewards = torch.randn(10)
        batch_next_obs = torch.randn(10, 4)
        batch_dones = torch.randint(0, 2, (10,))
        sac_agent.observe(
            batch_obs,
            batch_actions,
            batch_rewards,
            batch_next_obs,
            batch_dones,
        )

    # Now check if the parameters have been updated
    def check_state_dict_updated(initial_state, updated_state, tolerance=1e-4):
        for key in initial_state:
            if not torch.allclose(
                initial_state[key], updated_state[key], atol=tolerance
            ):
                return True
        return False

    actor_updated = check_state_dict_updated(
        initial_actor_state, sac_agent.actor.state_dict()
    )
    qf1_updated = check_state_dict_updated(
        initial_qf1_state, sac_agent.qf1.state_dict()
    )
    qf2_updated = check_state_dict_updated(
        initial_qf2_state, sac_agent.qf2.state_dict()
    )
    assert (
        actor_updated
    ), "Actor parameters were not updated after learning_starts"
    assert qf1_updated, "QF1 parameters were not updated after learning_starts"
    assert qf2_updated, "QF2 parameters were not updated after learning_starts"
