# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCriticDKWMP
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.replay_buffer import ReplayBuffer

import escnn
import escnn.group
from escnn.group import CyclicGroup
from rsl_rl.utils.symm_utils import add_repr_to_gspace, SimpleEMLP, get_symm_tensor, G, representations_action, representations, representations_commands, representations_wm_feature



class AMPDKPPO:
    actor_critic: ActorCriticDKWMP

    def __init__(self,
                 actor_critic,
                 discriminator,
                 amp_data,
                 amp_normalizer,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 vel_predict_coef=1.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 amp_replay_buffer_size=100000,
                 min_std=None,
                 use_amp = False,
                 sym_coef = 1.0,
                 amp_coef = 1.0
                 ):

        self.sym_coef = sym_coef
        self.amp_coef = amp_coef
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.min_std = min_std
        self.use_amp = use_amp

        # Discriminator components
        if use_amp:
            self.discriminator = discriminator
            self.discriminator.to(self.device)
            print("Motion state dim is:", discriminator.input_dim // 2)
            self.amp_transition = RolloutStorage.Transition()
            self.amp_storage = ReplayBuffer(
                discriminator.input_dim // 2, amp_replay_buffer_size, device)
            self.amp_normalizer = amp_normalizer
        self.amp_data = amp_data
        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later

        # Optimizer for policy and discriminator.
        if  use_amp:
            params = [
                {'params': self.actor_critic.parameters(), 'name': 'actor_critic'},
                {'params': self.discriminator.trunk.parameters(),
                'weight_decay': 10e-4, 'name': 'amp_trunk'},
                {'params': self.discriminator.amp_linear.parameters(),
                'weight_decay': 10e-2, 'name': 'amp_head'}]
        else:
            params = [
                {'params': self.actor_critic.parameters(), 'name': 'actor_critic'},
                     ]
        self.optimizer = optim.Adam(params, lr=learning_rate)
        # 初始化DK优化器
        self. DK_optimizer = optim.Adam(
            self.actor_critic.deep_koopman.parameters(),
            lr=learning_rate
        )
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.vel_predict_coef = vel_predict_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape,
                     history_dim, wm_feature_dim):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape,
                                      action_shape, history_dim=history_dim,
                                      wm_feature_dim=wm_feature_dim, device=self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, amp_obs, history, wm_feature):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.history = history
        self.transition.wm_feature = wm_feature.detach()
        aug_obs, aug_critic_obs = obs.detach(), critic_obs.detach()
        self.transition.actions = self.actor_critic.act(aug_obs, history, wm_feature).detach()
        # self.actor_critic.eval()
        self.transition.values = self.actor_critic.evaluate(aug_critic_obs, wm_feature).detach()
        # self.actor_critic.train()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        if self.use_amp:
            self.amp_transition.observations = amp_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, amp_obs):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        not_done_idxs = (dones == False).nonzero().squeeze()
        if self.use_amp:
            self.amp_storage.insert(
                self.amp_transition.observations, amp_obs)  # state and next_state

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        if self.use_amp:
            self.amp_transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, wm_feature):
        aug_last_critic_obs = last_critic_obs.detach()
        # self.actor_critic.eval()
        last_values = self.actor_critic.evaluate(aug_last_critic_obs, wm_feature).detach()
        # self.actor_critic.train()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update_amp(self):
        mean_sym_loss = 0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_vel_predict_loss = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
            self.num_mini_batches)
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
            self.num_mini_batches)

        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):

            obs_batch, critic_obs_batch, actions_batch, history_batch, wm_feature_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch = sample
            aug_obs_batch = obs_batch.detach()
            self.actor_critic.act(aug_obs_batch, history_batch, wm_feature_batch, masks=masks_batch,
                                  hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            aug_critic_obs_batch = critic_obs_batch.detach()
            value_batch = self.actor_critic.evaluate(aug_critic_obs_batch, wm_feature_batch, masks=masks_batch,
                                                     hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            ## For symloss
            obs_history_batch_symmetry = get_symm_tensor(history_batch, G, representations)
            # 取出commands，翻转后放回obs batch
            commands_batch = aug_obs_batch[:, self.actor_critic.privileged_dim + 6:self.actor_critic.privileged_dim + 9]
            commands_symmetry = get_symm_tensor(commands_batch, G, representations_commands)
            # aug_obs_batch_symmetry = aug_ob
            aug_obs_batch_symmetry = torch.cat([
                aug_obs_batch[:, :self.actor_critic.privileged_dim],
                commands_symmetry,
                aug_obs_batch[:, self.actor_critic.privileged_dim + 9:]
            ], dim=-1)
            # 翻转 wm_feature ？需要在图像未编码前翻转，较为麻烦，除非从一开始就把所有图像翻转过一遍GRU
            actions_symmetry =  self.actor_critic.act(aug_obs_batch, obs_history_batch_symmetry, wm_feature_batch, masks=masks_batch,
                                  hidden_states=hid_states_batch[0])
            actions_symmetry_rerversed = get_symm_tensor(actions_symmetry, G, representations_action)
            mu_batch = self.actor_critic.action_mean
            sym_loss = (mu_batch - actions_symmetry_rerversed).pow(2).mean()

            # KL
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                    torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                    2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                               1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Linear vel predict loss
            predicted_linear_vel = self.actor_critic.get_linear_vel(aug_obs_batch, history_batch)
            target_linear_vel = aug_critic_obs_batch[:,
                                self.actor_critic.privileged_dim - 3: self.actor_critic.privileged_dim]
            vel_predict_loss = (predicted_linear_vel - target_linear_vel).pow(2).mean()

            # Discriminator loss.
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert

            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
            policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
            expert_loss = torch.nn.MSELoss()(
                expert_d, torch.ones(expert_d.size(), device=self.device))
            policy_loss = torch.nn.MSELoss()(
                policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen_loss = self.discriminator.compute_grad_pen(
                *sample_amp_expert, lambda_=10)
            
            # Deepkoopman loss
            DK_total_loss = self.actor_critic.deep_koopman.loss_fn(history_batch)["total_loss"]

            # Compute total loss.
            loss = (
                    self.sym_coef * sym_loss +
                    surrogate_loss +
                    self.vel_predict_coef * vel_predict_loss +
                    self.value_loss_coef * value_loss -
                    self.entropy_coef * entropy_batch.mean() +
                    self.amp_coef * amp_loss + 
                    grad_pen_loss)

            # update DK before AC
            # Deepkoopman gradient step
            for param in self.actor_critic.deep_koopman.parameters():
                param.requires_grad = True  # 开放所有参数
            self.DK_optimizer.zero_grad()
            DK_total_loss.backward()
            nn.utils.clip_grad_norm_(
                    self.actor_critic.deep_koopman.parameters(),
                    self.max_grad_norm,)
            self.DK_optimizer.step()
            for param in self.actor_critic.deep_koopman.parameters():
                param.requires_grad = False  # 冻结所有参数

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()




            if not self.actor_critic.fixed_std and self.min_std is not None:
                self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)

            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state.cpu().numpy())
                self.amp_normalizer.update(expert_state.cpu().numpy())

            mean_sym_loss += sym_loss.item()
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()
            mean_vel_predict_loss += vel_predict_loss.mean().item()

            mean_DK_loss = DK_total_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_sym_loss /= num_updates
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        mean_vel_predict_loss /= num_updates
        mean_DK_loss /= num_updates

        print("mean_DK_loss:", mean_DK_loss)
        self.storage.clear()

        return mean_sym_loss, mean_value_loss, mean_surrogate_loss, mean_vel_predict_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred, mean_DK_loss
    
    def update(self):
        mean_sym_loss = 0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_vel_predict_loss = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        amp_loss = 0
        grad_pen_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)



        for sample in generator:

            obs_batch, critic_obs_batch, actions_batch, history_batch, wm_feature_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch = sample
            aug_obs_batch = obs_batch.detach()
            self.actor_critic.act(aug_obs_batch, history_batch, wm_feature_batch, masks=masks_batch,
                                  hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            aug_critic_obs_batch = critic_obs_batch.detach()
            value_batch = self.actor_critic.evaluate(aug_critic_obs_batch, wm_feature_batch, masks=masks_batch,
                                                     hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy
            
            ## For symloss
            obs_history_batch_symmetry = get_symm_tensor(history_batch, G, representations)
            actions_symmetry =  self.actor_critic.act(aug_obs_batch, obs_history_batch_symmetry, wm_feature_batch, masks=masks_batch,
                                  hidden_states=hid_states_batch[0])
            actions_symmetry_rerversed = get_symm_tensor(actions_symmetry, G, representations_action)
            mu_batch = self.actor_critic.action_mean
            sym_loss = (mu_batch - actions_symmetry_rerversed).pow(2).mean()

            # KL
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                    torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                    2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                               1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Linear vel predict loss
            predicted_linear_vel = self.actor_critic.get_linear_vel(aug_obs_batch, history_batch)
            target_linear_vel = aug_critic_obs_batch[:,
                                self.actor_critic.privileged_dim - 3: self.actor_critic.privileged_dim]
            vel_predict_loss = (predicted_linear_vel - target_linear_vel).pow(2).mean()


            # Deepkoopman loss
            DK_total_loss = self.actor_critic.deep_koopman.loss_fn(history_batch)["total_loss"]



            # Compute total loss.
            loss = (
                    self.sym_coef * sym_loss +
                    surrogate_loss +
                    self.vel_predict_coef * vel_predict_loss +
                    self.value_loss_coef * value_loss -
                    self.entropy_coef * entropy_batch.mean() +
                    self.amp_coef * amp_loss + 
                    grad_pen_loss)  

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Deepkoopman gradient step
            self.DK_optimizer.zero_grad()
            DK_total_loss.backward()
            nn.utils.clip_grad_norm_(
                    self.actor_critic.deep_koopman.parameters(),
                    self.max_grad_norm,)
            self.DK_optimizer.step()

            if not self.actor_critic.fixed_std and self.min_std is not None:
                self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)

            mean_sym_loss += sym_loss.item()
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_amp_loss += amp_loss
            mean_grad_pen_loss += grad_pen_loss
            mean_vel_predict_loss += vel_predict_loss.mean().item()

            mean_DK_loss = DK_total_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_sym_loss /= num_updates
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        mean_vel_predict_loss /= num_updates
        mean_DK_loss /= num_updates

        print("mean_DK_loss:", mean_DK_loss)
        self.storage.clear()

        return mean_sym_loss, mean_value_loss, mean_surrogate_loss, mean_vel_predict_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred, mean_DK_loss

