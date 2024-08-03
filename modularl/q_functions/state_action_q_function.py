import torch
import torch.nn as nn
import torch.nn.init as init
from modularl.q_functions.q_functions import StateActionQFunction


class SAQNetwork(StateActionQFunction):
    def __init__(
        self, observation_shape: int, action_shape: int, use_xavier=True
    ):
        """
        Initializes a fully-connected (s,a)-input Q-function network.

        Args:
            observation_shape (int): The shape of the observation input.
            action_shape (int): The shape of the action input.
            use_xavier (bool): Whether to use Xavier initialization for the network weights. Default is True.
        """  # noqa
        super().__init__()
        self.observation_shape = observation_shape
        self.action_shape = action_shape

        self.network = nn.Sequential(
            nn.Linear(
                observation_shape + action_shape, 16 * observation_shape
            ),
            nn.ReLU(),
            nn.Linear(16 * observation_shape, 16 * observation_shape),
            nn.ReLU(),
            nn.Linear(16 * observation_shape, 1),
        )
        if use_xavier:
            self._initialize_weights()

    def forward(self, observation, actions):
        """
        Forward pass of the Q-function network.

        Args:
            observation (torch.Tensor): Batch Observation tensor.
            actions (torch.Tensor): Batch Action tensor.

        Returns:
            torch.Tensor: Q-value tensor.
        """
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
