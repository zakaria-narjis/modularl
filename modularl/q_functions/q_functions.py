from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class StateQFunction(nn.Module, ABC):
    """Abstract Q-function with state input."""

    @abstractmethod
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Evaluates Q-function

        Args:
            x (ndarray): state input

        Returns:
            An instance of ActionValue that allows to calculate the Q-values
            for state x and every possible action
        """
        raise NotImplementedError()


class StateActionQFunction(nn.Module, ABC):
    """Abstract Q-function with state and action input."""

    @abstractmethod
    def forward(
        self, observation: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates Q-function

        Args:
            x (ndarray): state input
            a (ndarray): action input

        Returns:
            Q-value for state x and action a
        """
        raise NotImplementedError()
