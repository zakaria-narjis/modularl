import torch
import torch.nn as nn
import torch.nn.init as init
from modularl.policies.policy import AbstractPolicy
from typing import Any, Optional, Tuple

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class GaussianPolicy(AbstractPolicy):
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
        Gaussian Policy for continuous action spaces

        Args:
            observation_shape (int): Dimension of the observation space
            action_shape (int): Dimension of the action space
            high_action (float): Upper bound of the action space
            low_action (float): Lower bound of the action space
            network (nn.Module, optional): Custom neural network to represent the policy. If None, a default network is used.
            use_xavier (bool, optional): Whether to use Xavier initialization for weights. Defaults to True.

        Note:
            The head of the network consists of two nn.Linear layers for mean and log_std, with input size equal to the output features of the last layer in the provided network.

        Examples:
            >>> policy = GaussianPolicy(512, 10, 1, -1)
            >>> custom_network = nn.Sequential(
            >>>     nn.Linear(512, 256),
            >>>     nn.ReLU(),
            >>>     nn.Linear(256, 128),
            >>>     nn.ReLU(),
            >>> )
            >>> policy = GaussianPolicy(512, 10, 1, -1, network=custom_network)
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
            )
        else:
            self.network = network

        self.fc_mean = nn.Linear(
            self.network[-2].out_features, self.action_shape
        )
        self.fc_logstd = nn.Linear(
            self.network[-2].out_features, self.action_shape
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

    def forward(
        self, observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.network(observation)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def get_action(self, observation: torch.Tensor):
        """
        Get action from the policy

        Args:
            observation (torch.Tensor): Observation from the environment
        Returns:
            action (torch.Tensor): Sampled action from the policy distribution (only if deterministic is False)
            log_prob (torch.Tensor): Log probability of the action (only if deterministic is False)
            mean (torch.Tensor): Mean of the action distribution
        """  # noqa
        mean, log_std = self(observation)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = (
            normal.rsample()
        )  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
