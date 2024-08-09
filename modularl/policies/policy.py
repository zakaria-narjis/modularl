import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class AbstractPolicy(nn.Module, ABC):

    def __init__(self, **kwargs):
        super(AbstractPolicy, self).__init__()

    @abstractmethod
    def forward(self, batch_observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the policy network.

        Note:
            This method should not be used to get actions.
            For obtaining the batch actions, please use the `get_action` method.

        Args:
            batch_observation (torch.Tensor): Batch observation from the environment
        """  # noqa

    @abstractmethod
    def get_action(self, batch_observation: torch.Tensor) -> torch.Tensor:
        """
        Get action from the policy

        Args:
            batch_observation (torch.Tensor): Batch observation from the environment
        return:
            batch_action (torch.Tensor): Batch action to be taken
        """  # noqa

    @abstractmethod
    def _initialize_weights(self) -> None:
        """
        Initialize weights of the policy
        """  # noqa
