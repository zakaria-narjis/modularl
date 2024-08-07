import time
import torch
import torch.nn.functional as F
import copy
from tensordict import TensorDict
from modularl.agents.agent import AbstractAgent
from torchrl.data import TensorDictReplayBuffer
from typing import Callable, Optional, Any
from torch.utils.tensorboard import SummaryWriter


class TD3(AbstractAgent):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent

    :param actor: The actor network.
    :type actor: torch.nn.Module

    :param qf1: The first Q-function network.
    :type qf1: torch.nn.Module

    :param qf2: The second Q-function network.
    :type qf2: torch.nn.Module

    :param actor_optimizer: Optimizer for the actor network.
    :type actor_optimizer: torch.optim.Optimizer

    :param q_optimizer: Optimizer for the Q-function networks.
    :type q_optimizer: torch.optim.Optimizer

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

    :param policy_noise: Noise added to the target policy during critic updates. Defaults to 0.2.
    :type policy_noise: float, optional

    :param noise_clip: Range to clip the target policy noise. Defaults to 0.5.
    :type noise_clip: float, optional

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
        qf1: torch.nn.Module,
        qf2: torch.nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_optimizer: torch.optim.Optimizer,
        replay_buffer: TensorDictReplayBuffer,
        gamma: float = 0.99,
        batch_size: int = 32,
        learning_starts: int = 0,
        tau: float = 0.005,
        exploration_noise: float = 0.1,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_frequency: int = 2,
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
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self.policy_frequency = policy_frequency

        # Networks
        self.actor = actor.to(self.device)
        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.qf1_target = copy.deepcopy(self.qf1).to(self.device)
        self.qf2_target = copy.deepcopy(self.qf2).to(self.device)

        self.actor_optimizer = actor_optimizer
        self.q_optimizer = q_optimizer

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
                actions = self.actor(batch_obs.to(self.device))
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
            actions = self.actor(batch_obs.to(self.device))
        self.actor.train()
        return actions

    def update(self) -> None:

        if self.global_step > self.learning_starts:
            data = self.rb.sample(self.batch_size).to(self.device)

            with torch.no_grad():
                if self.gamma != 0:
                    clipped_noise = (
                        torch.randn_like(data["actions"], device=self.device)
                        * self.policy_noise
                    ).clamp(
                        -self.noise_clip, self.noise_clip
                    ) * self.target_actor.action_scale

                    next_state_actions = (
                        self.target_actor(data["next_observations"])
                        + clipped_noise
                    ).clamp(self.actor.low_action, self.actor.high_action)

                    qf1_next_target = self.qf1_target(
                        data["next_observations"], next_state_actions
                    )
                    qf2_next_target = self.qf2_target(
                        data["next_observations"], next_state_actions
                    )
                    min_qf_next_target = torch.min(
                        qf1_next_target, qf2_next_target
                    )
                    next_q_value = data["rewards"].flatten() + (
                        1 - data["dones"].flatten()
                    ) * self.gamma * min_qf_next_target.view(-1)
                else:
                    next_q_value = data["rewards"].flatten()

            qf1_a_values = self.qf1(
                data["observations"], data["actions"]
            ).view(-1)
            qf2_a_values = self.qf2(
                data["observations"], data["actions"]
            ).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            self.q_optimizer.zero_grad()
            qf_loss.backward()
            self.q_optimizer.step()

            if self.global_step % self.policy_frequency == 0:
                actor_loss = -self.qf1(
                    data["observations"], self.actor(data["observations"])
                ).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                if self.gamma != 0:
                    # Update target networks
                    for param, target_param in zip(
                        self.actor.parameters(), self.target_actor.parameters()
                    ):
                        target_param.data.copy_(
                            self.tau * param.data
                            + (1 - self.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.qf1.parameters(), self.qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.tau * param.data
                            + (1 - self.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.qf2.parameters(), self.qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.tau * param.data
                            + (1 - self.tau) * target_param.data
                        )

            if self.global_step % 100 == 0 and self.writer is not None:
                self.writer.add_scalar(
                    "losses/qf1_values",
                    qf1_a_values.mean().item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "losses/qf2_values",
                    qf2_a_values.mean().item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    "losses/qf1_loss", qf1_loss.item(), self.global_step
                )
                self.writer.add_scalar(
                    "losses/qf2_loss", qf2_loss.item(), self.global_step
                )
                self.writer.add_scalar(
                    "losses/qf_loss", qf_loss.item() / 2.0, self.global_step
                )
                self.writer.add_scalar(
                    "losses/actor_loss", actor_loss.item(), self.global_step
                )
                self.writer.add_scalar(
                    "charts/SPS",
                    int(self.global_step / (time.time() - self.start_time)),
                    self.global_step,
                )
