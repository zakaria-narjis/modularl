import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modularl.agents.td3 import TD3
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
def td3_agent():
    obs_dim = 4
    action_dim = 2
    actor = DummyActor(obs_dim, action_dim)
    qf1 = DummyCritic(obs_dim, action_dim)
    qf2 = DummyCritic(obs_dim, action_dim)
    actor_optimizer = torch.optim.Adam(actor.parameters())
    q_optimizer = torch.optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters())
    )
    replay_buffer = ReplayBuffer(buffer_size=1000)
    agent = TD3(
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


def test_td3_init(td3_agent):
    assert isinstance(td3_agent, TD3)
    assert td3_agent.device == "cpu"
    assert td3_agent.batch_size == 32
    assert td3_agent.learning_starts == 100


def test_td3_observe(td3_agent):
    batch_size = 10
    obs_dim = 4
    action_dim = 2

    batch_obs = torch.randn(batch_size, obs_dim)
    batch_actions = torch.randn(batch_size, action_dim)
    batch_rewards = torch.randn(batch_size)
    batch_next_obs = torch.randn(batch_size, obs_dim)
    batch_dones = torch.randint(0, 2, (batch_size,))

    initial_rb_size = len(td3_agent.rb)
    td3_agent.observe(
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones
    )
    assert len(td3_agent.rb) == initial_rb_size + batch_size


def test_td3_act_train(td3_agent):
    batch_size = 5
    obs_dim = 4

    batch_obs = torch.randn(batch_size, obs_dim)
    actions = td3_agent.act_train(batch_obs)
    assert actions.shape == (batch_size, 2)


def test_td3_act_eval(td3_agent):
    batch_size = 5
    obs_dim = 4

    batch_obs = torch.randn(batch_size, obs_dim)
    actions = td3_agent.act_eval(batch_obs)
    assert actions.shape == (batch_size, 2)


def test_td3_update(td3_agent):
    # Fill the replay buffer with some dummy data
    for _ in range(99):  # One less than learning_starts
        batch_obs = torch.randn(10, 4)
        batch_actions = torch.randn(10, 2)
        batch_rewards = torch.randn(10)
        batch_next_obs = torch.randn(10, 4)
        batch_dones = torch.randint(0, 2, (10,))
        td3_agent.observe(
            batch_obs,
            batch_actions,
            batch_rewards,
            batch_next_obs,
            batch_dones,
        )

    # Check that parameters haven't been updated yet
    initial_actor_state = copy.deepcopy(td3_agent.actor.state_dict())
    initial_qf1_state = copy.deepcopy(td3_agent.qf1.state_dict())
    initial_qf2_state = copy.deepcopy(td3_agent.qf2.state_dict())

    def check_state_dict_unchanged(initial_state, current_state):
        for key in initial_state:
            if not torch.allclose(initial_state[key], current_state[key]):
                return False
        return True

    assert check_state_dict_unchanged(
        initial_actor_state, td3_agent.actor.state_dict()
    ), "Actor parameters were updated before learning_starts"
    assert check_state_dict_unchanged(
        initial_qf1_state, td3_agent.qf1.state_dict()
    ), "QF1 parameters were updated before learning_starts"
    assert check_state_dict_unchanged(
        initial_qf2_state, td3_agent.qf2.state_dict()
    ), "QF2 parameters were updated before learning_starts"

    # Perform more observations to trigger the update
    for _ in range(100):
        batch_obs = torch.randn(10, 4)
        batch_actions = torch.randn(10, 2)
        batch_rewards = torch.randn(10)
        batch_next_obs = torch.randn(10, 4)
        batch_dones = torch.randint(0, 2, (10,))
        td3_agent.observe(
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
        initial_actor_state, td3_agent.actor.state_dict()
    )
    qf1_updated = check_state_dict_updated(
        initial_qf1_state, td3_agent.qf1.state_dict()
    )
    qf2_updated = check_state_dict_updated(
        initial_qf2_state, td3_agent.qf2.state_dict()
    )
    assert (
        actor_updated
    ), "Actor parameters were not updated after learning_starts"
    assert qf1_updated, "QF1 parameters were not updated after learning_starts"
    assert qf2_updated, "QF2 parameters were not updated after learning_starts"
