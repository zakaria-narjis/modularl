import torch
import torch.nn as nn
import torch.nn.init as init
from modularl.policies.policy import AbstractPolicy
from typing import Any, Optional


class DeterministicPolicy(AbstractPolicy):
    def __init__(
        self,
        observation_shape: int,
        action_shape: int,
        high_action: float,
        low_action: float,
        network: Optional[nn.Module] = None,
        use_xavier: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        """
        Deterministic Policy for continuous action spaces

        Args:
            observation_shape (int): Dimension of the observation space
            action_shape (int): Dimension of the action space
            high_action (float): Upper bound of the action space
            low_action (float): Lower bound of the action space
            network (nn.Module, optional): Custom neural network to represent the policy. If None, a default network is used.
            use_xavier (bool, optional): Whether to use Xavier initialization for weights. Defaults to True.

        Note:
            If no custom network is provided, a default network is created with three linear layers and ReLU activations.
            The output layer uses a Tanh activation to bound the actions.

        Examples:
            >>> policy = DeterministicPolicy(512, 10, 1, -1)
            >>> custom_network = nn.Sequential(
            >>>     nn.Linear(512, 256),
            >>>     nn.ReLU(),
            >>>     nn.Linear(256, 128),
            >>>     nn.ReLU(),
            >>>     nn.Linear(128, 10),
            >>>     nn.Tanh()
            >>> )
            >>> policy = DeterministicPolicy(512, 10, 1, -1, network=custom_network)
        """  # noqa
        self.high_action = high_action
        self.low_action = low_action
        self.action_shape = action_shape
        self.observation_shape = observation_shape
        if network is None:
            self.network = nn.Sequential(
                nn.Linear(observation_shape, 16 * observation_shape),
                nn.ReLU(),
                nn.Linear(16 * observation_shape, 16 * observation_shape),
                nn.ReLU(),
                nn.Linear(16 * observation_shape, self.action_shape),
                nn.Tanh(),
            )
        else:
            self.network = network
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (self.high_action - self.low_action) / 2.0, dtype=torch.float32
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (self.high_action + self.low_action) / 2.0, dtype=torch.float32
            ),
        )
        if use_xavier:
            self._initialize_weights()

    def forward(self, observation):
        output = self.network(observation)
        return output

    def get_action(self, observation):
        """
        Get action from the policy

        Args:
            observation (torch.Tensor): Observation from the environment
        return:
            action (torch.Tensor): Action to be taken
        """  # noqa
        actions = self(observation) * self.action_scale + self.action_bias
        return actions

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
