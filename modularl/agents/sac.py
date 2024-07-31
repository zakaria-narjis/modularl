
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
from tensordict import TensorDict

from torchrl.data import TensorDictReplayBuffer,LazyMemmapStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement



class SAC:

    def __init__(self,
                actor,
                qf1,
                qf2,
                actor_optimizer,
                q_optimizer,    
                replay_buffer,
                entropy_lr = 1e-3,
                batch_size = 32,
                learning_starts = 0,            
                entropy_temperature = 0.2,
                target_entropy = None,
                tau = 0.005,
                device = "cpu",
                burning_action_func=None,
                writer=None,):
        
        self.device = device
        self.writer = writer
        self.rb = replay_buffer
        self.batch_size = batch_size
        self.tau = tau
        self.burning_action_func = burning_action_func
        self.learning_starts = learning_starts
        #networks
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
        self.target_entropy = target_entropy

        if self.target_entropy is not None:
            self.auto_tune_temp = True
        else:
            self.auto_tune_temp = False
          
        # entropy
        if self.auto_tune_temp:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], 
                                          lr=self.entropy_lr,
                                          )
        else:
            self.alpha = self.alpha
        #ReplayBuffer
        # self.rb = TensorDictReplayBuffer(
        #                 storage=LazyMemmapStorage(args.buffer_size,), 
        #                 sampler=SamplerWithoutReplacement(),
        #                                                 )
        # self.rb = TensorDictReplayBuffer(storage=LazyMemmapStorage(self.buffer_size,) )
    def init(self,) -> None:
        self.start_time = time.time()
        self.global_step = 0

    def observe(self,batch_obs,batch_actions,batch_rewards,batch_next_obs,batch_dones):
        """
            Observe the environment and store the transition in the replay buffer
            Args:
                batch_obs: Tensor containing the observations
                batch_actions: Tensor containing the actions
                batch_rewards: Tensor containing the rewards
                batch_next_obs: Tensor containing the next observations
                batch_dones: Tensor containing the dones
        """
        self.global_step += 1
        batch_transition = TensorDict(
            {
                "observations":batch_obs.clone(),
                "next_observations":batch_next_obs.clone(),
                "actions":batch_actions.clone(),
                "rewards":batch_rewards.clone(),
                "dones":batch_dones.clone(),
            },
            batch_size = [batch_obs.shape[0]],
        )
        self.rb.extend(batch_transition)
        self.update()

    def act_train(self, batch_obs):
        torch.pi
        if self.global_step < self.args.learning_starts and self.args.use_burning_action==True:
            return torch.randint(low=self.args.low_action, 
                                 high=self.high_action, 
                                 size=(batch_obs.shape[0],self.args.action_shape)).to(self.device)  
        else:
            actions, _, _ = self.actor.get_action(batch_obs)
            actions = actions.detach()

        return actions
    
    def act_eval(self, obs):
        self.qf1.eval().requires_grad_(False)
        self.qf2.eval().requires_grad_(False)
        self.actor.eval().requires_grad_(False)
        with torch.no_grad():
            actions = self.actor.get_action(obs.to(self.device))
        self.qf1.train().requires_grad_(True)
        self.qf2.train().requires_grad_(True)
        self.actor.train().requires_grad_(True) 
        return actions
    
    def update(self,):
        # ALGO LOGIC: training.
        if self.global_step > self.args.learning_starts:
            data = self.rb.sample(self.args.batch_size).to(self.device)
            with torch.no_grad():
                if self.args.gamma!=0:
                    next_state_actions, next_state_log_pi, _ = self.actor.get_action(data["next_observations"])
                    qf1_next_target = self.qf1_target(data["next_observations"], actions=next_state_actions)
                    qf2_next_target = self.qf2_target(data["next_observations"], actions=next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                    next_q_value = data["rewards"].flatten() + (1 - data["dones"].to(torch.float32).flatten()) * self.args.gamma * (min_qf_next_target).view(-1)
                else:
                    next_q_value = data["rewards"].flatten()

            qf1_a_values = self.qf1(data["observations"], actions = data["actions"]).view(-1)
            qf2_a_values = self.qf2(data["observations"], actions = data["actions"]).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            self.q_optimizer.zero_grad()
            qf_loss.backward()
            self.q_optimizer.step()

            if self.global_step % self.args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    self.args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = self.actor.get_action(data["observations"])
                    qf1_pi = self.qf1(data["observations"], actions=pi)
                    qf2_pi = self.qf2(data["observations"], actions=pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    if self.args.autotune:
                        # with torch.no_grad():
                        #     _, log_pi, _ = self.actor.get_action(data["observations"])
                        alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
                        self.a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.a_optimizer.step()
                        self.alpha = self.log_alpha.exp().item()


            # update the target networks
            if self.args.gamma!=0:
                if self.global_step % self.args.target_network_frequency == 0:
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
                    for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                        target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

            if self.global_step % 100 == 0 and self.writer is not None:
                self.writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), self.global_step)
                self.writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), self.global_step)
                self.writer.add_scalar("losses/qf1_loss", qf1_loss.item(), self.global_step)
                self.writer.add_scalar("losses/qf2_loss", qf2_loss.item(), self.global_step)
                self.writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, self.global_step)
                self.writer.add_scalar("losses/actor_loss", actor_loss.item(), self.global_step)
                self.writer.add_scalar("losses/alpha", self.alpha, self.global_step)
                self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)
                if self.args.autotune:
                    self.writer.add_scalar("losses/alpha_loss", alpha_loss.item(), self.global_step)
        
        