import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modularl.agents.ddpg import DDPG
from modularl.replay_buffers import ReplayBuffer
import copy


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
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.high_action = high_action
        self.low_action = low_action
        self.fc1 = nn.Linear(obs_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
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
        return torch.tanh(self.fc3(x))

    def get_action(self, observation):
        actions = self(observation) * self.action_scale + self.action_bias
        return actions

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)


class DummyCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, use_xavier=True):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

        if use_xavier:
            self._initialize_weights()

    def forward(self, x, action):
        x = torch.cat([x, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)


@pytest.fixture
def ddpg_agent():
    obs_dim = 4
    action_dim = 2
    actor = DummyActor(obs_dim, action_dim)
    qf = DummyCritic(obs_dim, action_dim)
    actor_optimizer = torch.optim.Adam(actor.parameters())
    qf_optimizer = torch.optim.Adam(qf.parameters())
    replay_buffer = ReplayBuffer(buffer_size=1000)
    agent = DDPG(
        actor=actor,
        qf=qf,
        actor_optimizer=actor_optimizer,
        qf_optimizer=qf_optimizer,
        replay_buffer=replay_buffer,
        batch_size=32,
        learning_starts=100,
        device="cpu",
    )
    agent.init()
    return agent


def test_ddpg_init(ddpg_agent):
    assert isinstance(ddpg_agent, DDPG)
    assert ddpg_agent.device == "cpu"
    assert ddpg_agent.batch_size == 32
    assert ddpg_agent.learning_starts == 100


def test_ddpg_observe(ddpg_agent):
    batch_size = 10
    obs_dim = 4
    action_dim = 2

    batch_obs = torch.randn(batch_size, obs_dim)
    batch_actions = torch.randn(batch_size, action_dim)
    batch_rewards = torch.randn(batch_size)
    batch_next_obs = torch.randn(batch_size, obs_dim)
    batch_dones = torch.randint(0, 2, (batch_size,))

    initial_rb_size = len(ddpg_agent.rb)
    ddpg_agent.observe(
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones
    )
    assert len(ddpg_agent.rb) == initial_rb_size + batch_size


def test_ddpg_act_train(ddpg_agent):
    batch_size = 5
    obs_dim = 4

    batch_obs = torch.randn(batch_size, obs_dim)
    actions = ddpg_agent.act_train(batch_obs)
    assert actions.shape == (batch_size, 2)


def test_ddpg_act_eval(ddpg_agent):
    batch_size = 5
    obs_dim = 4

    batch_obs = torch.randn(batch_size, obs_dim)
    actions = ddpg_agent.act_eval(batch_obs)
    assert actions.shape == (batch_size, 2)


def test_ddpg_update(ddpg_agent):
    # Fill the replay buffer with some dummy data
    for _ in range(99):  # One less than learning_starts
        batch_obs = torch.randn(10, 4)
        batch_actions = torch.randn(10, 2)
        batch_rewards = torch.randn(10)
        batch_next_obs = torch.randn(10, 4)
        batch_dones = torch.randint(0, 2, (10,))
        ddpg_agent.observe(
            batch_obs,
            batch_actions,
            batch_rewards,
            batch_next_obs,
            batch_dones,
        )

    # Check that parameters haven't been updated yet
    initial_actor_state = copy.deepcopy(ddpg_agent.actor.state_dict())
    initial_qf_state = copy.deepcopy(ddpg_agent.qf.state_dict())

    def check_state_dict_unchanged(initial_state, current_state):
        for key in initial_state:
            if not torch.allclose(initial_state[key], current_state[key]):
                return False
        return True

    assert check_state_dict_unchanged(
        initial_actor_state, ddpg_agent.actor.state_dict()
    ), "Actor parameters were updated before learning_starts"
    assert check_state_dict_unchanged(
        initial_qf_state, ddpg_agent.qf.state_dict()
    ), "QF parameters were updated before learning_starts"

    # Perform more observations to trigger the update
    for _ in range(100):
        batch_obs = torch.randn(10, 4)
        batch_actions = torch.randn(10, 2)
        batch_rewards = torch.randn(10)
        batch_next_obs = torch.randn(10, 4)
        batch_dones = torch.randint(0, 2, (10,))
        ddpg_agent.observe(
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
        initial_actor_state, ddpg_agent.actor.state_dict()
    )
    qf_updated = check_state_dict_updated(
        initial_qf_state, ddpg_agent.qf.state_dict()
    )
    assert (
        actor_updated
    ), "Actor parameters were not updated after learning_starts"
    assert qf_updated, "QF parameters were not updated after learning_starts"
