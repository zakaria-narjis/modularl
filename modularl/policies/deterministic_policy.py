import torch
import torch.nn as nn
import torch.nn.init as init
from modularl.policies.policy import AbstractPolicy
from typing import Any, Optional


class DeterministicPolicy(AbstractPolicy):
    """
    Deterministic Policy for continuous action spaces.

    :param observation_shape: Dimension of the observation space.
    :type observation_shape: int

    :param action_shape: Dimension of the action space.
    :type action_shape: int

    :param high_action: Upper bound of the action space.
    :type high_action: float

    :param low_action: Lower bound of the action space.
    :type low_action: float

    :param network: Custom neural network to represent the policy. If None, a default network is used. Defaults to None.
    :type network: nn.Module, optional

    :param use_xavier: Whether to use Xavier initialization for weights. Defaults to True.
    :type use_xavier: bool, optional

    Note:
        If no custom network is provided, a default network is created with three linear layers and ReLU activations. The output layer uses a Tanh activation to bound the actions.
        If a custom network is provided, it should be headless, meaning that this class will add an additional linear layer on top of the provided network for the policy output, with input size equal to the output features of the last layer in the provided network.
    """  # noqa

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
            self.network = nn.Sequential(
                network,
                nn.Linear(network[-2].out_feature, self.action_shape),
                nn.Tanh(),
            )
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
