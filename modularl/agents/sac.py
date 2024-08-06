import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
from tensordict import TensorDict
from modularl.agents.agent import AbstractAgent
from torchrl.data import TensorDictReplayBuffer
from typing import Callable, Optional, Any
from torch.utils.tensorboard import SummaryWriter


class SAC(AbstractAgent):
    """
    Soft Actor-Critic (SAC) Agent

    :param actor: The actor network (policy) to be used.
    :type actor: torch.nn.Module

    :param qf1: The first Q-function network.
    :type qf1: torch.nn.Module

    :param qf2: The second Q-function network.
    :type qf2: torch.nn.Module

    :param actor_optimizer: Optimizer for the actor network.
    :type actor_optimizer: torch.optim.Optimizer

    :param q_optimizer: Optimizer for both Q-function networks.
    :type q_optimizer: torch.optim.Optimizer

    :param replay_buffer: Replay buffer for storing experiences.
    :type replay_buffer: TensorDictReplayBuffer

    :param gamma: Discount factor for future rewards. Defaults to 0.99.
    :type gamma: float, optional

    :param entropy_lr: Learning rate for the entropy temperature. Defaults to 1e-3.
    :type entropy_lr: float, optional

    :param batch_size: Number of samples per batch for training. Defaults to 32.
    :type batch_size: int, optional

    :param learning_starts: Number of steps before learning starts. Defaults to 0.
    :type learning_starts: int, optional

    :param entropy_temperature: Initial entropy temperature. Defaults to 0.2.
    :type entropy_temperature: float, optional

    :param target_entropy: Target entropy for adaptive temperature adjustment. Defaults to None.
    :type target_entropy: float, optional

    :param tau: Soft update coefficient for target networks. Defaults to 0.005.
    :type tau: float, optional

    :param policy_frequency: Frequency of policy updates. Defaults to 1.
    :type policy_frequency: int, optional

    :param target_network_frequency: Frequency of target network updates. Defaults to 2.
    :type target_network_frequency: int, optional

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
        entropy_lr: float = 1e-3,
        batch_size: int = 32,
        learning_starts: int = 0,
        entropy_temperature: float = 0.2,
        target_entropy: Optional[float] = None,
        tau: float = 0.005,
        policy_frequency: int = 1,
        target_network_frequency: int = 2,
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
        # networks
        self.actor = actor.to(self.device)
        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)
        self.qf1_target = copy.deepcopy(self.qf1).to(self.device)
        self.qf2_target = copy.deepcopy(self.qf2).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.actor_optimizer = actor_optimizer
        self.q_optimizer = q_optimizer
        self.alpha = entropy_temperature
        self.entropy_lr = entropy_lr
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.target_entropy = target_entropy
        if self.target_entropy is not None:
            self.auto_tune_temp = True
        else:
            self.auto_tune_temp = False

        # entropy
        if self.auto_tune_temp:
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device
            )
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.entropy_lr)
        else:
            self.alpha = self.alpha

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
        """
        Generate actions for training based on the current policy.
        It uses a burning action function for initial exploration if specified,
        then switches to the learned policy.

        :param batch_obs: (torch.Tensor) A batch of observations from the environment.
        :return: (torch.Tensor) A batch of actions to be taken in the environment.

        Notes:
            - If the global step is less than `learning_starts` and a burning action
              function is provided, it uses that function for exploration.
            - Otherwise, it uses the current policy (actor) to generate actions.
        """  # noqa: E501
        if (
            self.global_step < self.learning_starts
            and self.burning_action_func is not None
        ):
            return self.burning_action_func(batch_obs).to(self.device)
        else:
            actions, _, _ = self.actor.get_action(batch_obs)
            actions = actions.detach()

        return actions

    def act_eval(self, batch_obs: torch.Tensor) -> torch.Tensor:

        self.qf1.eval().requires_grad_(False)
        self.qf2.eval().requires_grad_(False)
        self.actor.eval().requires_grad_(False)
        with torch.no_grad():
            actions, _, _ = self.actor.get_action(batch_obs.to(self.device))
        self.qf1.train().requires_grad_(True)
        self.qf2.train().requires_grad_(True)
        self.actor.train().requires_grad_(True)
        return actions

    def update(self) -> None:

        if self.global_step > self.learning_starts:
            data = self.rb.sample(self.batch_size).to(self.device)
            with torch.no_grad():
                if self.gamma != 0:
                    next_state_actions, next_state_log_pi, _ = (
                        self.actor.get_action(data["next_observations"])
                    )
                    qf1_next_target = self.qf1_target(
                        data["next_observations"], actions=next_state_actions
                    )
                    qf2_next_target = self.qf2_target(
                        data["next_observations"], actions=next_state_actions
                    )
                    min_qf_next_target = (
                        torch.min(qf1_next_target, qf2_next_target)
                        - self.alpha * next_state_log_pi
                    )
                    next_q_value = data["rewards"].flatten() + (
                        1 - data["dones"].to(torch.float32).flatten()
                    ) * self.gamma * (min_qf_next_target).view(-1)
                else:
                    next_q_value = data["rewards"].flatten()

            qf1_a_values = self.qf1(
                data["observations"], actions=data["actions"]
            ).view(-1)
            qf2_a_values = self.qf2(
                data["observations"], actions=data["actions"]
            ).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            self.q_optimizer.zero_grad()
            qf_loss.backward()
            self.q_optimizer.step()

            if (
                self.global_step % self.policy_frequency == 0
            ):  # TD 3 Delayed update support
                for _ in range(
                    self.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1 # noqa: E501
                    pi, log_pi, _ = self.actor.get_action(data["observations"])
                    qf1_pi = self.qf1(data["observations"], actions=pi)
                    qf2_pi = self.qf2(data["observations"], actions=pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    if self.auto_tune_temp:
                        alpha_loss = (
                            -self.log_alpha.exp()
                            * (log_pi + self.target_entropy).detach()
                        ).mean()
                        self.a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.a_optimizer.step()
                        self.alpha = self.log_alpha.exp().item()

            # update the target networks
            if self.gamma != 0:
                if self.global_step % self.target_network_frequency == 0:
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
                    "losses/alpha", self.alpha, self.global_step
                )
                self.writer.add_scalar(
                    "charts/SPS",
                    int(self.global_step / (time.time() - self.start_time)),
                    self.global_step,
                )
                if self.auto_tune_temp:
                    self.writer.add_scalar(
                        "losses/alpha_loss",
                        alpha_loss.item(),
                        self.global_step,
                    )
