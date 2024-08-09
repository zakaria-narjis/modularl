from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class StateQFunction(nn.Module, ABC):
    """Abstract Q-function with state input."""

    @abstractmethod
    def forward(self, batch_observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Q-function network.

        :param batch_observation: Batch Observation tensor.
        :return: Q-value tensor.
        """  # noqa
        raise NotImplementedError()


class StateActionQFunction(nn.Module, ABC):
    """Abstract Q-function with state and action input."""

    @abstractmethod
    def forward(
        self, batch_observation: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the Q-function network.

        :param batch_observation: Batch Observation tensor.
        :param batch_actions: Batch Action tensor.
        :return: Q-value tensor.
        """  # noqa
        raise NotImplementedError()
