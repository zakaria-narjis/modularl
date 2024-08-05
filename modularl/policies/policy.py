import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from typing import Any


class AbstractPolicy(nn.Module, ABC):

    def __init__(self, **kwargs):
        super(AbstractPolicy, self).__init__()

    @abstractmethod
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the policy network

        Args:
            observation (torch.Tensor): Observation from the environment
        """  # noqa

    @abstractmethod
    def get_action(self, observation: torch.Tensor) -> Any:
        """
        Get action from the policy

        Args:
            observation (torch.Tensor): Observation from the environment
        return:
            action (torch.Tensor): Action to be taken
        """

    @abstractmethod
    def _initialize_weights(self) -> None:
        """
        Initialize weights of the policy
        """  # noqa
