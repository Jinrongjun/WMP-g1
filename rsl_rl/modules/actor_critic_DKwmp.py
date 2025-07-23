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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .state_estimator_DK import DeepKoopman

# 搞定KOOPMAN部分需要：1. 网络初始化 2. 改 act_interface

class ActorCriticDKWMP(nn.Module):
    is_recurrent = False

    # TODO: 上层传参num_history, prop_dim
    def __init__(self, 
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 encoder_hidden_dims=[256, 128],
                 wm_encoder_hidden_dims = [64, 32],
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 fixed_std=False,
                 latent_dim = 32,
                 height_dim=187,
                 privileged_dim=3 + 24,
                 history_dim = 42*5,
                 prop_dim = 69,
                 wm_feature_dim = 1536,
                 wm_latent_dim=16,
                 dk_latent_dim=128,
                 num_history=5,
                 use_observation_function=True,
                 w_delta_s = 0.0,
                 w_delta_a = 0.0,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(ActorCriticDKWMP, self).__init__()

        activation = get_activation(activation)
        self.w_delta_a = w_delta_a
        self.w_delta_s = w_delta_s

        self.latent_dim = latent_dim
        self.height_dim = height_dim
        self.privileged_dim = privileged_dim
        self.dk_latent_dim = dk_latent_dim
        self.prop_dim = prop_dim

        mlp_input_dim_a = latent_dim + 3 + wm_latent_dim #+ dk_latent_dim #latent vector + command + wm_latent
        mlp_input_dim_c = num_critic_obs + wm_latent_dim   # TODO: crtic要不要加上dk latent



        # Deep Koopman
        self.deep_koopman = DeepKoopman(
            num_obs=history_dim/num_history,
            num_history= num_history,
            num_latent=dk_latent_dim,
            num_action= num_actions,
            num_prop=prop_dim,
            use_observation_function=use_observation_function)


        # History Encoder
        # Koopman Compression Version
        # 实例化模型
        self.history_encoder = HistoryEncoder( dk_latent_dim= dk_latent_dim,
                                num_history = num_history,
                                encoder_hidden_dims = encoder_hidden_dims,
                                latent_dim = latent_dim,
                                activation = activation,
                                w_delta_s = w_delta_s,
                                DK = self.deep_koopman, 
                                num_action=num_actions)

        # Autoencoder version
        # encoder_layers = []
        # encoder_layers.append(nn.Linear(history_dim, encoder_hidden_dims[0]))
        # encoder_layers.append(activation)
        # for l in range(len(encoder_hidden_dims)):
        #     if l == len(encoder_hidden_dims) - 1:
        #         encoder_layers.append(nn.Linear(encoder_hidden_dims[l], latent_dim))
        #     else:
        #         encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
        #         encoder_layers.append(activation)
        # self.history_encoder = nn.Sequential(*encoder_layers)




        # World Model Feature Encoder
        wm_encoder_layers = []
        wm_encoder_layers.append(nn.Linear(wm_feature_dim, wm_encoder_hidden_dims[0]))
        wm_encoder_layers.append(activation)
        for l in range(len(wm_encoder_hidden_dims)):
            if l == len(wm_encoder_hidden_dims) - 1:
                wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_latent_dim))
            else:
                wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_encoder_hidden_dims[l + 1]))
                wm_encoder_layers.append(activation)
        self.wm_feature_encoder = nn.Sequential(*wm_encoder_layers)

        # Critic World Model Feature Encoder
        critic_wm_encoder_layers = []
        critic_wm_encoder_layers.append(nn.Linear(wm_feature_dim, wm_encoder_hidden_dims[0]))
        critic_wm_encoder_layers.append(activation)
        for l in range(len(wm_encoder_hidden_dims)):
            if l == len(wm_encoder_hidden_dims) - 1:
                critic_wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_latent_dim))
            else:
                critic_wm_encoder_layers.append(nn.Linear(wm_encoder_hidden_dims[l], wm_encoder_hidden_dims[l + 1]))
                critic_wm_encoder_layers.append(activation)
        self.critic_wm_feature_encoder = nn.Sequential(*critic_wm_encoder_layers)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                # actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)

        self.critic = nn.Sequential(*critic_layers)




        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.fixed_std = fixed_std
        std = init_noise_std * torch.ones(num_actions)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        std = self.std.to(mean.device)
        self.distribution = Normal(mean, mean * 0. + std)


    # act with KOOPMAN
    def act(self, observations, history, wm_feature, **kwargs):
        # 直接从observation中获取当前prop
        # latent_vector = self.history_encoder(history)
        # command = observations[:, self.privileged_dim + 6:self.privileged_dim + 9]
        # obs_without_command = torch.concat((observations[:, self.privileged_dim:self.privileged_dim + 6], observations[:, self.privileged_dim + 9:self.privileged_dim + self.prop_dim]),
        #                                    dim=-1)
        # dk_latent= self.deep_koopman.encode(obs_without_command)
        # wm_latent_vector = self.wm_feature_encoder(wm_feature)
        # concat_observations = torch.concat((latent_vector, command, wm_latent_vector, dk_latent),
        #                                    dim=-1)



        # Deep koopman encode
        # print("his_size:", history.size())
        DK_embeded_history = self.deep_koopman.history_encode(history)
        latent_vector = self.history_encoder(DK_embeded_history)
        command = observations[:, self.privileged_dim + 6:self.privileged_dim + 9]

        wm_latent_vector = self.wm_feature_encoder(wm_feature)
        concat_observations = torch.concat((latent_vector, command, wm_latent_vector),
                                           dim=-1)
        self.update_distribution(concat_observations)
        action = self.distribution.sample()
        propagated_state = self.deep_koopman.state_propagate(DK_embeded_history[..., -self.dk_latent_dim:], action)
        delta_action = self.w_delta_a * self.deep_koopman.forward_planner(propagated_state)
        modified_action = action + delta_action
        return modified_action

    def get_latent_vector(self, observations, history, **kwargs):
        DK_embeded_history = self.deep_koopman.history_encode(history)
        latent_vector = self.history_encoder(DK_embeded_history)
        return latent_vector

    def get_linear_vel(self, observations, history, **kwargs):
        DK_embeded_history = self.deep_koopman.history_encode(history)
        latent_vector = self.history_encoder(DK_embeded_history)
        linear_vel = latent_vector[:,-3:]
        return linear_vel

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, history, wm_feature):
        DK_embeded_history = self.deep_koopman.history_encode(history)
        latent_vector = self.history_encoder(DK_embeded_history)
        command = observations[:, self.privileged_dim + 6:self.privileged_dim + 9]
        dk_latent= self.deep_koopman.encode(observations[self.privileged_dim:self.privileged_dim + self.prop_dim])
        wm_latent_vector = self.wm_feature_encoder(wm_feature)
        concat_observations = torch.concat((latent_vector, command, wm_latent_vector, dk_latent),
                                           dim=-1)
        actions_mean = self.actor(concat_observations)
        propagated_state = self.deep_koopman.propagate(DK_embeded_history[..., -self.dk_latent_dim], actions_mean)
        delta_action = self.w_delta_a * self.deep_koopman.forward_planner(propagated_state)
        modified_actions_mean = actions_mean + delta_action
        return modified_actions_mean

    def evaluate(self, critic_observations, wm_feature,  **kwargs):
        wm_latent_vector = self.critic_wm_feature_encoder(wm_feature)
        # print("input data size:", critic_observations.shape)
        concat_observations = torch.concat((critic_observations, wm_latent_vector),
                                           dim=-1)


        value = self.critic(concat_observations)
        return value



class HistoryEncoder(nn.Module):
    def __init__(self, 
                 dk_latent_dim: int,
                 num_history: int,
                 encoder_hidden_dims: list,
                 latent_dim: int,
                 activation=nn.ELU(),
                 w_delta_s: float = 0.0, 
                 num_action: int = 29,
                 DK=None):
        super().__init__()
        
        # 参数校验
        assert dk_latent_dim > 0, "dk_latent_dim 必须为正整数"
        assert num_history > 0, "num_history 必须为正整数"
        assert len(encoder_hidden_dims) > 0, "encoder_hidden_dims 不能为空列表"

        # 初始化参数
        self.dk_latent_dim = dk_latent_dim
        self.num_history = num_history
        self.num_action = num_action
        self.w_delta_s = w_delta_s
        self.DK = DK  # 允许为None，表示不启用DK模块
        
        # 激活函数
        self.activation = activation
        
        # 构建编码器
        encoder_layers = []
        input_dim = (dk_latent_dim + num_action) * num_history
        
        # 输入层
        encoder_layers.append(nn.Linear(input_dim, encoder_hidden_dims[0]))
        encoder_layers.append(self.activation)
        
        # 隐藏层
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], latent_dim))
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
                encoder_layers.append(self.activation)
        
        # 封装为Sequential
        self.history_encoder = nn.Sequential(*encoder_layers)
        
        # 规范化层（可选）
        self.norm = nn.LayerNorm(latent_dim) if latent_dim > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入形状: (batch_size, num_history * (dk_latent_dim + num_action))
        输出形状: (batch_size, latent_dim)
        """
        if self.DK is not None:
            # DK模块处理
            traj = self.DK.Process_history_to_traj(x, self.dk_latent_dim, self.num_history, self.num_action)
            propagated = self.DK.state_propagate(traj['states'][:, 1:, :], traj['actions'])
            
            # 拼接传播结果与动作
            propagated_his = torch.cat((propagated, traj['actions']), dim=-1)
            propagated_his = propagated_his.reshape(-1, (self.num_history-1) * (self.dk_latent_dim + self.num_action))
            
            # 计算传播损失
            propagated_loss = self.w_delta_s * propagated_his - x[..., self.dk_latent_dim + self.num_action:]
            combined_x = torch.cat((x[..., -(self.dk_latent_dim + self.num_action):], propagated_loss), dim=-1)
        else:
            combined_x = x
        
        # 编码并规范化
        encoded = self.history_encoder(combined_x)
        return self.norm(encoded)
    


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None