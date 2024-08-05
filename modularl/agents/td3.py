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
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_frequency: int = 2,
        device: str = "cpu",
        burning_action_func: Optional[Callable] = None,
        writer: Optional[SummaryWriter] = None,
        **kwargs: Any
    ) -> None:
        """
        Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent

        Args:
            actor (torch.nn.Module): Actor network
            qf1 (torch.nn.Module): First Q-function network
            qf2 (torch.nn.Module): Second Q-function network
            actor_optimizer (torch.optim.Optimizer): Actor optimizer
            q_optimizer (torch.optim.Optimizer): Q-function optimizer
            replay_buffer (TensorDictReplayBuffer): Replay buffer
            gamma (float, optional): Discount factor. Defaults to 0.99.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            learning_starts (int, optional): Number of steps before learning starts. Defaults to 0.
            tau (float, optional): Target network update rate. Defaults to 0.005.
            policy_noise (float, optional): Noise added to target policy during critic update. Defaults to 0.2.
            noise_clip (float, optional): Range to clip target policy noise. Defaults to 0.5.
            policy_frequency (int, optional): Frequency of delayed policy updates. Defaults to 2.
            device (str, optional): Device to run the agent on. Defaults to "cpu".
            burning_action_func (Optional[Callable], optional): Function for generating initial exploratory actions. Defaults to None.
            writer (Optional[SummaryWriter], optional): Tensorboard writer for logging. Defaults to None.
        """  # noqa: E501
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
        """
        Observe the environment and store the transition in the replay buffer
        Args:
            batch_obs (torch.Tensor): Tensor containing the observations
            batch_actions (torch.Tensor): Tensor containing the actions
            batch_rewards (torch.Tensor): Tensor containing the rewards
            batch_next_obs (torch.Tensor): Tensor containing the next observations
            batch_dones (torch.Tensor): Tensor containing the dones
        """  # noqa: E501
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
        Generate actions for training based on the current policy with exploration noise.

        This method handles the exploration-exploitation trade-off during training.
        It uses a burning action function for initial exploration if specified,
        then switches to the learned policy with added Gaussian noise.

        Args:
            batch_obs (torch.Tensor): A batch of observations from the environment.

        Returns:
            torch.Tensor: A batch of actions to be taken in the environment.

        Notes:
            - If the global step is less than `learning_starts` and a burning action
            function is provided, it uses that function for exploration.
            - Otherwise, it uses the current policy (actor) to generate actions and adds
            Gaussian noise for exploration.
            - The noise is scaled by the actor's action scale and the policy noise parameter.
            - The final actions are clamped to be within the valid action range.
        """  # noqa: E501
        if (
            self.global_step < self.learning_starts
            and self.burning_action_func is not None
        ):
            return self.burning_action_func(batch_obs).to(self.device)
        else:
            actions = self.actor(batch_obs.to(self.device))
            actions = actions + torch.normal(
                0,
                self.actor.action_scale * self.policy_noise,
                size=actions.shape,
                device=self.device,
            )
            return actions.clamp(
                -self.actor.action_scale, self.actor.action_scale
            )

    def act_eval(self, batch_obs: torch.Tensor) -> torch.Tensor:
        """
        Returns the actions to take in evaluation mode.

        Args:
            batch_obs (torch.Tensor): The input observations (N,*).

        Returns:
            torch.Tensor: The actions to take(N,*).
        """
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(batch_obs.to(self.device))
        self.actor.train()
        return actions

    def update(self) -> None:
        """
        Update the SAC agent by performing a training step.

        This method implements the training logic for the SAC (Soft Actor-Critic) algorithm.
        It updates the Q-networks, the actor network, and the temperature parameter alpha.

        Returns:
            None
        """  # noqa: E501
        if self.global_step > self.learning_starts:
            data = self.rb.sample(self.batch_size).to(self.device)

            with torch.no_grad():
                if self.gamma != 0:
                    clipped_noise = (
                        torch.randn_like(data["actions"]) * self.policy_noise
                    ).clamp(-self.noise_clip, self.noise_clip)

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
