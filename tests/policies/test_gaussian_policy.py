# test_gaussian_policy.py

import pytest
import torch
from modularl.policies.gaussian_policy import GaussianPolicy

@pytest.fixture
def gaussian_policy():
    observation_shape = 4
    action_shape = 2
    high_action = 1.0
    low_action = -1.0
    return GaussianPolicy(observation_shape, action_shape, high_action, low_action)

def test_gaussian_policy_init(gaussian_policy):
    assert isinstance(gaussian_policy, GaussianPolicy)
    assert gaussian_policy.observation_shape == 4
    assert gaussian_policy.action_shape == 2
    assert gaussian_policy.high_action == 1.0
    assert gaussian_policy.low_action == -1.0

def test_gaussian_policy_forward(gaussian_policy):
    batch_size = 10
    observation = torch.randn(batch_size, gaussian_policy.observation_shape)
    
    mean, log_std = gaussian_policy(observation)
    
    assert mean.shape == (batch_size, gaussian_policy.action_shape)
    assert log_std.shape == (batch_size, gaussian_policy.action_shape)
    assert torch.all(log_std >= -20) and torch.all(log_std <= 2)

def test_gaussian_policy_get_action(gaussian_policy):
    batch_size = 10
    observation = torch.randn(batch_size, gaussian_policy.observation_shape)
    
    action, log_prob, mean = gaussian_policy.get_action(observation)
    
    assert action.shape == (batch_size, gaussian_policy.action_shape)
    assert log_prob.shape == (batch_size, 1)
    assert mean.shape == (batch_size, gaussian_policy.action_shape)
    assert torch.all(action >= -1) and torch.all(action <= 1)

def test_gaussian_policy_deterministic_action(gaussian_policy):
    batch_size = 10
    observation = torch.randn(batch_size, gaussian_policy.observation_shape)
    
    deterministic_action = gaussian_policy.get_action(observation, deterministic=True)
    
    assert deterministic_action.shape == (batch_size, gaussian_policy.action_shape)
    assert torch.all(deterministic_action >= -1) and torch.all(deterministic_action <= 1)

def test_gaussian_policy_gradient_flow(gaussian_policy):
    batch_size = 10
    observation = torch.randn(batch_size, gaussian_policy.observation_shape, requires_grad=True)
    
    action, log_prob, _ = gaussian_policy.get_action(observation)
    loss = log_prob.sum()
    loss.backward()
    
    for param in gaussian_policy.parameters():
        assert param.grad is not None
        assert torch.sum(param.grad ** 2).item() > 0

