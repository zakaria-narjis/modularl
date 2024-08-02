# test_abstract_agent.py

import pytest
import torch
from abc import ABC, abstractmethod


class AbstractAgentTest(ABC):
    @pytest.fixture
    @abstractmethod
    def agent(self):
        pass

    def test_init(self, agent):
        assert hasattr(agent, "device")
        assert hasattr(agent, "actor")
        assert hasattr(agent, "actor_optimizer")

    def test_observe(self, agent):
        batch_size = 10
        obs_dim = agent.actor.fc.in_features
        action_dim = agent.actor.fc.out_features

        batch_obs = torch.randn(batch_size, obs_dim)
        batch_actions = torch.randn(batch_size, action_dim)
        batch_rewards = torch.randn(batch_size)
        batch_next_obs = torch.randn(batch_size, obs_dim)
        batch_dones = torch.randint(0, 2, (batch_size,))

        initial_rb_size = len(agent.rb)
        agent.observe(
            batch_obs,
            batch_actions,
            batch_rewards,
            batch_next_obs,
            batch_dones,
        )
        assert len(agent.rb) == initial_rb_size + batch_size

    def test_act_train(self, agent):
        batch_size = 5
        obs_dim = agent.actor.fc.in_features
        action_dim = agent.actor.fc.out_features

        batch_obs = torch.randn(batch_size, obs_dim)
        actions = agent.act_train(batch_obs)

        assert actions.shape == (batch_size, action_dim)

    def test_act_eval(self, agent):
        batch_size = 5
        obs_dim = agent.actor.fc.in_features
        action_dim = agent.actor.fc.out_features

        batch_obs = torch.randn(batch_size, obs_dim)
        actions = agent.act_eval(batch_obs)

        assert actions.shape == (batch_size, action_dim)

    def test_update(self, agent):
        # Fill the replay buffer with some dummy data
        for _ in range(200):
            batch_obs = torch.randn(10, agent.actor.fc.in_features)
            batch_actions = torch.randn(10, agent.actor.fc.out_features)
            batch_rewards = torch.randn(10)
            batch_next_obs = torch.randn(10, agent.actor.fc.in_features)
            batch_dones = torch.randint(0, 2, (10,))
            agent.observe(
                batch_obs,
                batch_actions,
                batch_rewards,
                batch_next_obs,
                batch_dones,
            )

        # Perform an update
        initial_actor_state = agent.actor.state_dict()

        agent.update()

        # Check if the parameters have been updated
        assert not torch.all(
            torch.eq(
                initial_actor_state["fc.weight"],
                agent.actor.state_dict()["fc.weight"],
            )
        )


# Example usage:
# class TestSAC(AbstractAgentTest):
#     @pytest.fixture
#     def agent(self):
#         # Initialize and return your SAC agent here
#         pass
