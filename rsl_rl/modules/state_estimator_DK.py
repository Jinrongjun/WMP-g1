import torch.nn as nn
import torch
import numpy as np
from torch.distributions import Normal
from torch.nn import functional
from torchsummary import summary

"""
TODO:
# 1. 调整推理接口

# 2. 添加超参

3. 修改上层，网络推理 + 网络训练过程

线速度怎么处理？直接作为输入吗？还是需要预测？

"""


class DeepKoopman(nn.Module):
    def __init__(
        self,
        num_obs,
        num_history,
        num_latent,
        num_action,
        num_prop,
        activation="elu",
        encoder_hidden_dims=[512, 256],
        decoder_hidden_dims=[512, 256, 128],
        device = "cuda",
        weight_list=[0.5, 1e-5, 1e-7, 1e-5],
    ):
        super().__init__()
        self.device = device

        # decoder输出维度
        self.num_obs = num_obs

        # encoder输入维度
        self.num_his = num_history

        # 中间层压缩维度
        self.num_latent = num_latent

        # 非常重要的一步：把command减去，这三维不在给入的history里面, 同时再去掉action对应维度数
        self.num_prop = num_prop - 3 - num_action

        self.num_action = num_action

        # 初始化encoder
        self.encoder = encoder(
            self.num_prop,
            self.num_latent,    # 考虑要不要加速度预测（3维）
            activation,
            encoder_hidden_dims,
        )

        # 初始化decoder
        num_vel = 3
        self.decoder = decoder(
            self.num_latent,
            self.num_prop,# num_obs
            activation,
            decoder_hidden_dims,
        )

        self.propagate = nn.Linear(self.num_latent + num_action, num_latent, bias=False)

        # 初始化损失权重参数
        self.pred_loss_weight = weight_list[0]
        self.max_loss_weight = weight_list[1]
        self.weight_decay_weight = weight_list[2]
        self.metric_loss_weight = weight_list[3]


    def Process_history_to_traj(self, obs_history, num_prop,  num_history, num_action):
            """
            将观测历史转换为状态和动作的轨迹。
            :param obs_history: 形状为 [num_envs, num_obs * num_history] 的张量,
            :param num_obs: 单步状态的维度值
            :param num_history: 历史步数
            :param privileged_dim: 特权维度数量
            ## obs_history里面的数据结构: (注意: 没有command)
                                    (self.base_ang_vel  * self.obs_scales.ang_vel, #3
                                    self.projected_gravity, #3
                                    
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, #29
                                    self.dof_vel * self.obs_scales.dof_vel, #29
                                    sin_phase, #1
                                    cos_phase, #1
                                    self.actions, #29)   * num_history
            :return: 包含状态和动作的字典
            """
            
            # 重塑为[num_envs, num_history, num_obs]的三维张量
            obs_3d = obs_history.view(-1, num_history, num_prop + num_action)
            # print(obs_3d.size())
            # 提取状态序列（所有历史步的状态）
            states = obs_3d[:, :, :-1*num_action]  # [num_envs, num_history, state_dim]
            
            # 提取动作序列（从下一步观测中获取当前动作）
            # 使用roll实现位移索引，比切片快30%[1,6](@ref)
            next_obs = torch.roll(obs_3d, shifts=-1, dims=1)[:, :-1]  # [num_envs, num_history-1, num_obs]
            actions = next_obs[:, :, 8 + 2*num_action : 8 + 3*num_action]  # [num_envs, num_history-1, action_dim]
            # print("state size:", states.size())
            return {'states': states, 'actions': actions}

    



    # 生成latent
    def forward(self, state, action, cv=0):
        """
        纯单步下生成latent, 以及预测下一个状态的过程
        """
        latent= self.encode(state)
        # print("Propagate inout size:",self.num_latent + self.num_action)
        # print("input size:", latent.size(), action.size())
        latent_prediction = self.propagate(torch.cat((latent, action), dim=-1))
        state_prediction = self.decode(latent_prediction)
        state_reconstructed = self.decode(latent)
        return state_reconstructed, state_prediction, latent, latent_prediction
    

    # 损失函数
    def loss_fn(self, obs_history, kld_weight=1.0):
        mseloss = nn.MSELoss()
        l1loss = nn.L1Loss()
        Traj = self.Process_history_to_traj(obs_history, self.num_prop, self.num_his,  self.num_action)
        
        # 输入：num_history - 1 个过去的状态和动作
        # 输出：num_history - 1 个预测的状态和动作
        state_reconstructed, state_prediction, latent, latent_prediction = self.forward(Traj['states'][:, :-1, :], Traj['actions'])

        ae_loss = mseloss(Traj['states'][:, :-1, :], state_reconstructed)
        pred_loss = mseloss(Traj['states'][:, 1:, :], state_prediction)

        # linearity loss
        lin_loss = mseloss(latent_prediction, self.encoder(Traj['states'][:, 1:, :]))

        # largest loss
        inf_loss = torch.max(torch.abs(Traj['states'][:, :-1, :] - state_reconstructed)) + torch.max(torch.abs(Traj['states'][:, 1:, :] - state_prediction))

        # metric loss
        # print("Metrics:", latent_prediction.size(), latent.size(), state_prediction.size(), Traj['states'][:, :-1, :].size())
        metric_loss = l1loss(torch.norm(latent_prediction - latent, dim=-1), torch.norm(state_prediction - Traj['states'][:, :-1, :], dim=-1))

       
        # Frobenius norm of operator
        weight_loss = 0
        # 迭代 encoder 内部的 Sequential 模块
        for l in self.encoder.encoder:  # 注意：访问 self.encoder.encoder
            if isinstance(l, nn.Linear):
                weight_loss += torch.norm(l.weight.data)
        # 同理处理 decoder（假设 decoder 结构类似）
        for l in self.decoder.decoder:  # 假设 decoder 也有类似的内部 Sequential 模块
            if isinstance(l, nn.Linear):
                weight_loss += torch.norm(l.weight.data)


        total_loss = (
                self.pred_loss_weight * (ae_loss + pred_loss)
                + lin_loss
                + self.max_loss_weight * inf_loss
                + self.weight_decay_weight * weight_loss
                + self.metric_loss_weight * metric_loss
            )
        return {
            "total_loss": total_loss,
            "recon_loss": ae_loss,
            "pred_loss": pred_loss,
            "lin_loss": lin_loss,
            "inf_loss": inf_loss,
            "weight_loss": weight_loss,
            "metric_loss": metric_loss,
        }


    # encoder 的作用过程
    def encode(self, prop):
        # print(prop.size())
        if prop.size(-1) > self.num_prop:
            latent = self.encoder(prop[..., :-1*self.num_action])
        else:
            latent = self.encoder(prop)
        return latent


    # decoder作用过程
    def decode(self, z):
        output = self.decoder(z)
        return output

    # 只输出推理结果，即估计值
    def inference(self, prop):
        """
        return latent
        """
        if prop.size(-1) > self.num_prop:    
            state_reconstructed, state_prediction, latent, latent_prediction = self.forward(prop[..., :-1*self.num_action])
        else:
            state_reconstructed, state_prediction, latent, latent_prediction = self.forward(prop)
        return latent
    
    def history_encode(self, obs_history):
        # 1. 重塑输入数据：合并批次和历史步维度，便于批量编码
        obs_3d = obs_history.view(-1, self.num_prop + self.num_action)  # [batch * num_his, features]
        
        # 2. 批量编码：一次性处理所有历史步的数据
        all_encoded = self.encode(obs_3d)  # [batch * num_his, num_latent]
        
        # 3. 重塑结果：恢复批次和历史步维度，并拼接为最终输出
        DK_embeded_history = all_encoded.view(obs_history.shape[0], self.num_his * self.num_latent)  # [batch, num_his * num_latent]
        
        return DK_embeded_history
    
# 需要删除的接口： sample, reparameterize
    # # 返回所有值，包括估计值和用于估计的参数
    # def sample(self, obs_history, cv):
    #     """
    #     :return estimation = [z, vel]
    #     :dim(z) = num_latent
    #     :dim(vel) = 3
    #     """
    #     estimation, output = self.forward(obs_history, cv)
    #     return estimation,output

    # # 生成latent的过程
    # def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, cv) -> torch.Tensor:
    #     """
    #     :param mu: (Tensor) Mean of the latent Gaussian
    #     :param logvar: (Tensor) Standard deviation of the latent Gaussian
    #     :return: eps * std + mu
    #     """
    #     std = torch.exp(0.5 * logvar) * (1 - np.tanh(cv))
    #     eps = torch.randn_like(std)
    #     return eps * std + mu

# encoder 部分
class encoder(nn.Module):
    def __init__(self, input_size, output_size, activation, hidden_dims):
        """
        :param input_size: (Tensor) encoder input size, e.g., num_obs
        :param output_size: (Tensor) encoder output size, e.g., num_latent + 3
        :param activation: (str) Activation function to use
        :param hidden_dims: (list) List of hidden layer dimensions
        """

        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        if activation == "relu":
                self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        module = []
        module.append(nn.Linear(self.input_size, hidden_dims[0]))
        module.append(self.activation)
        for i in range(len(hidden_dims) - 1):
            module.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            module.append(self.activation)
        module.append(nn.Linear(hidden_dims[-1], self.output_size))
        self.encoder = nn.Sequential(*module)

    def forward(self, obs):
        return self.encoder(obs)

# TODO： sindy encoder
# class encoder_Sindy(nn.Module):



# decoder 部分
class decoder(nn.Module):
    def __init__(self, input_size, output_size, activation, hidden_dims):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        module = []
        module.append(nn.Linear(self.input_size, hidden_dims[0]))
        module.append(self.activation)
        for i in range(len(hidden_dims) - 1):
            module.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            module.append(self.activation)
        module.append(nn.Linear(hidden_dims[-1], self.output_size))
        self.decoder = nn.Sequential(*module)

    def forward(self, input):
        return self.decoder(input)


# TODO: sindy decoder
# class decoder_Sindy(nn.Module):