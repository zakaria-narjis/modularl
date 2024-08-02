import pytest
import torch
from modularl.q_functions.state_action_q_function import SAQNetwork

@pytest.fixture
def saq_network():
    observation_shape = 4
    action_shape = 2
    return SAQNetwork(observation_shape, action_shape)

def test_saq_network_init(saq_network):
    assert isinstance(saq_network, SAQNetwork)
    assert saq_network.observation_shape == 4
    assert saq_network.action_shape == 2

def test_saq_network_forward(saq_network):
    batch_size = 10
    observation = torch.randn(batch_size, saq_network.observation_shape)
    action = torch.randn(batch_size, saq_network.action_shape)
    
    q_value = saq_network(observation, action)
    
    assert q_value.shape == (batch_size, 1)

def test_saq_network_gradient_flow(saq_network):
    batch_size = 10
    observation = torch.randn(batch_size, saq_network.observation_shape, requires_grad=True)
    action = torch.randn(batch_size, saq_network.action_shape, requires_grad=True)
    
    q_value = saq_network(observation, action)
    loss = q_value.sum()
    loss.backward()
    
    for param in saq_network.parameters():
        assert param.grad is not None
        assert torch.sum(param.grad ** 2).item() > 0
