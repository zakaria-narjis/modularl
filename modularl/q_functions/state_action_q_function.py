import torch
import torch.nn as nn
import torch.nn.init as init
from modularl.q_functions.q_functions import StateActionQFunction


class SAQNetwork(StateActionQFunction):
    """
    Initializes a fully-connected (s,a)-input Q-function network.

    :param observation_shape: The shape of the observation input.
    :type observation_shape: int

    :param action_shape: The shape of the action input.
    :type action_shape: int

    :param use_xavier: Whether to use Xavier initialization for the network weights. Defaults to True.
    :type use_xavier: bool, optional
    """  # noqa

    def __init__(
        self, observation_shape: int, action_shape: int, use_xavier=True
    ):
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
