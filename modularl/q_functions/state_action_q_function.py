import torch
import torch.nn as nn
import torch.nn.init as init
from modularl.q_functions.q_functions import StateActionQFunction
from typing import Optional


class SAQNetwork(StateActionQFunction):
    """
    Initializes a fully-connected (s,a)-input Q-function network.

    :param observation_shape: The shape of the observation input.
    :type observation_shape: int

    :param action_shape: The shape of the action input.
    :type action_shape: int

    :param network: Custom neural network to represent the Q-function. If None, a default network is used. Defaults to None.
    :type network: nn.Module, optional

    :param use_xavier: Whether to use Xavier initialization for the network weights. Defaults to True.
    :type use_xavier: bool, optional

    Note:
        If no custom network is provided, a default network is created with three linear layers and ReLU activations. The output layer uses a Tanh activation to bound the actions.
        If a custom network is provided, it should be headless, meaning that this class will add an additional linear layer on top of the provided network for the policy output, with input size equal to the output features of the last layer in the provided network.
    """  # noqa

    def __init__(
        self,
        observation_shape: int,
        action_shape: int,
        network: Optional[nn.Module] = None,
        use_xavier=True,
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        if network is None:
            self.network = nn.Sequential(
                nn.Linear(
                    observation_shape + action_shape, 16 * observation_shape
                ),
                nn.ReLU(),
                nn.Linear(16 * observation_shape, 16 * observation_shape),
                nn.ReLU(),
                nn.Linear(16 * observation_shape, 1),
            )
        else:
            self.network = network
        if use_xavier:
            self._initialize_weights()

    def forward(self, observation, actions):

        q_input = torch.cat([observation, actions], 1)
        q_value = self.network(q_input)
        return q_value

    def _initialize_weights(self):
        """
        Initializes the network weights using Xavier initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
