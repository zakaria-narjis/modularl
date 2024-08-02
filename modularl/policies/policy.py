import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from typing import Tuple


class AbstractPolicy(nn.Module, ABC):

    def __init__(
        self,
        observation_shape: int,
        action_shape: int,
        high_action: float,
        low_action: float,
    ):
        super(AbstractPolicy, self).__init__()

    @abstractmethod
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_action(
        self, observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def _initialize_weights(self) -> None:
        pass