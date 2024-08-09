import time
import torch
import torch.nn.functional as F
import copy
from tensordict import TensorDict
from modularl.agents.agent import AbstractAgent
from torchrl.data import TensorDictReplayBuffer
from typing import Callable, Optional, Any
from torch.utils.tensorboard import SummaryWriter


class DDPG(AbstractAgent):
    """
    Deep Deterministic Policy Gradient (DDPG) Agent

    :param actor: The actor network.
    :type actor: torch.nn.Module

    :param qf: The Q-function network.
    :type qf: torch.nn.Module

    :param actor_optimizer: Optimizer for the actor network.
    :type actor_optimizer: torch.optim.Optimizer

    :param qf_optimizer: Optimizer for the Q-function network.
    :type qf_optimizer: torch.optim.Optimizer

    :param replay_buffer: Replay buffer for storing experiences.
    :type replay_buffer: TensorDictReplayBuffer

    :param gamma: Discount factor for future rewards. Defaults to 0.99.
    :type gamma: float, optional

    :param batch_size: Number of samples per batch for training. Defaults to 32.
    :type batch_size: int, optional

    :param learning_starts: Number of steps before learning starts. Defaults to 0.
    :type learning_starts: int, optional

    :param tau: Soft update coefficient for target networks. Defaults to 0.005.
    :type tau: float, optional

    :param exploration_noise: Noise added to the actor policy during training. Defaults to 0.1.
    :type exploration_noise: float, optional

    :param policy_frequency: Frequency of delayed policy updates. Defaults to 2.
    :type policy_frequency: int, optional

    :param device: Device to run the agent on (e.g., "cpu" or "cuda"). Defaults to "cpu".
    :type device: str, optional

    :param burning_action_func: Function for generating initial exploratory actions. Defaults to None.
    :type burning_action_func: Callable, optional

    :param writer: Tensorboard writer for logging. Defaults to None.
    :type writer: SummaryWriter, optional
    """  # noqa: E501

    def __init__(
        self,
        actor: torch.nn.Module,
        qf: torch.nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        qf_optimizer: torch.optim.Optimizer,
        replay_buffer: TensorDictReplayBuffer,
        gamma: float = 0.99,
        batch_size: int = 32,
        learning_starts: int = 0,
        tau: float = 0.005,
        exploration_noise: float = 0.1,
        policy_frequency: int = 1,
        device: str = "cpu",
        burning_action_func: Optional[Callable] = None,
        writer: Optional[SummaryWriter] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.device = device
        self.writer = writer
        self.rb = replay_buffer
        self.batch_size = batch_size
        self.tau = tau
        self.burning_action_func = burning_action_func
        self.learning_starts = learning_starts
        self.gamma = gamma
        self.exploration_noise = exploration_noise
        self.policy_frequency = policy_frequency
        # Networks
        self.actor = actor.to(self.device)
        self.qf = qf.to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.qf_target = copy.deepcopy(self.qf).to(self.device)

        self.actor_optimizer = actor_optimizer
        self.qf_optimizer = qf_optimizer

    def init(self) -> None:
        self.start_time = time.time()
        self.global_step = 0

    def observe(
        self,
        batch_obs: torch.Tensor,
        batch_actions: torch.Tensor,
        batch_rewards: torch.Tensor,
        batch_next_obs: torch.Tensor,
        batch_dones: torch.Tensor,
    ) -> None:

        self.global_step += 1
        batch_transition = TensorDict(
            {
                "observations": batch_obs.clone(),
                "next_observations": batch_next_obs.clone(),
                "actions": batch_actions.clone(),
                "rewards": batch_rewards.clone(),
                "dones": batch_dones.clone(),
            },
            batch_size=[batch_obs.shape[0]],
        )
        self.rb.extend(batch_transition)
        self.update()

    def act_train(self, batch_obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if (
                self.global_step < self.learning_starts
                and self.burning_action_func is not None
            ):
                return self.burning_action_func(batch_obs).to(self.device)
            else:
                actions = self.actor.get_action(batch_obs.to(self.device))
                actions = actions + torch.normal(
                    0,
                    self.actor.action_scale * self.exploration_noise,
                    size=actions.shape,
                    device=self.device,
                )
                actions = actions.clamp(
                    self.actor.low_action, self.actor.high_action
                )
                return actions

    def act_eval(self, batch_obs: torch.Tensor) -> torch.Tensor:
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor.get_action(batch_obs.to(self.device))
        self.actor.train()
        return actions

    def update(self) -> None:
        if self.global_step > self.learning_starts:
            data = self.rb.sample(self.batch_size).to(self.device)

            with torch.no_grad():
                if self.gamma != 0:
                    next_state_actions = self.target_actor(
                        data["next_observations"]
                    )
                    next_q_value = self.qf_target(
                        data["next_observations"], next_state_actions
                    ).view(-1)
                    target_q_value = (
                        data["rewards"].flatten()
                        + (1 - data["dones"].flatten())
                        * self.gamma
                        * next_q_value
                    )
                else:
                    target_q_value = data["rewards"].flatten()

            # qf update
            current_q_value = self.qf(
                data["observations"], data["actions"]
            ).view(-1)
            qf_loss = F.mse_loss(current_q_value, target_q_value)

            self.qf_optimizer.zero_grad()
            qf_loss.backward()
            self.qf_optimizer.step()

            # Actor update
            if (
                self.global_step % self.policy_frequency == 0
            ):  # Delayed policy update
                actor_loss = -self.qf(
                    data["observations"],
                    self.actor.get_action(data["observations"]),
                ).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

            # Update target networks
            if self.gamma != 0:
                for param, target_param in zip(
                    self.actor.parameters(), self.target_actor.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data
                        + (1 - self.tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.qf.parameters(), self.qf_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data
                        + (1 - self.tau) * target_param.data
                    )

            if self.global_step % 100 == 0 and self.writer is not None:
                self.writer.add_scalar(
                    "losses/qf_values",
                    current_q_value.mean().item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "losses/qf_loss", qf_loss.item(), self.global_step
                )
                self.writer.add_scalar(
                    "losses/actor_loss", actor_loss.item(), self.global_step
                )
                self.writer.add_scalar(
                    "charts/SPS",
                    int(self.global_step / (time.time() - self.start_time)),
                    self.global_step,
                )
