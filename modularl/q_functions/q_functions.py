from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class StateQFunction(nn.Module, ABC):
    """Abstract Q-function with state input."""

    @abstractmethod
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Q-function network.
        :param observation: Batch Observation tensor.
        :return: Q-value tensor.
        """
        raise NotImplementedError()


class StateActionQFunction(nn.Module, ABC):
    """Abstract Q-function with state and action input."""

    @abstractmethod
    def forward(
        self, observation: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the Q-function network.
        :param observation: Batch Observation tensor.
        :param actions: Batch Action tensor.
        :return: Q-value tensor.
        """
        raise NotImplementedError()
