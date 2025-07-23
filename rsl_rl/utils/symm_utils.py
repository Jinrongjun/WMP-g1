import escnn
import escnn.group
from escnn.group import CyclicGroup, DihedralGroup, DirectProductGroup, Group, Representation
from escnn.nn import FieldType, EquivariantModule, GeometricTensor

from morpho_symm.utils.algebra_utils import gen_permutation_matrix
from morpho_symm.utils.rep_theory_utils import group_rep_from_gens

import torch
import torch.nn as nn
# from torch.distributions import Normal

import numpy as np
from typing import Union, List


### 本文件主要储存一些 "神经网络对称性" 构造过程中需要使用的辅助函数

# 设置numpy打印的精度
# np.set_printoptions(precision=1, suppress=True)

def add_repr_to_gspace(G:escnn.group.Group, 
                       permutation_conf:Union[List[int], np.ndarray],
                       reflection_conf:Union[List[int], np.ndarray],
                       name:str
    ):
    """将指定的变换添加到G这个group中, 供之后生成escnn的对称网络使用

    Args:
        G (escnn.group.Group): 一个escnn的group
        permutation_conf (Union[List[int], np.ndarray]): 位置变换矩阵
        reflection_conf (Union[List[int], np.ndarray]): 由1和-1指定的变换矩阵
        name (str): 这个变换的名字

    Returns:
        _type_: _description_
    """
    # 获取对应的配置文件，保证格式为 (n, 配置长度)
    permutation_conf = np.array(permutation_conf, dtype=int)
    reflection_conf = np.array(reflection_conf, dtype=float)
    if permutation_conf.ndim == 1: permutation_conf = permutation_conf[None]
    if reflection_conf.ndim == 1: reflection_conf = reflection_conf[None]
    
    # 获取配置个数, 长度
    (conf_num, conf_length) = permutation_conf.shape
    print(f"Conf name = {name}, length = {conf_length}")
    
    # 检查: 确保给的配置是正常的
    assert permutation_conf.shape == reflection_conf.shape, len(reflection_conf.shape)==2
    assert conf_num == len(G.generators)
    
    # 开始配置representations
    rep_joints = {G.identity: np.eye(conf_length, dtype=float)}
    for g_gen, perm, refx in zip(G.generators, permutation_conf, reflection_conf):
        refx = np.array(refx, dtype=float)
        rep_joints[g_gen] = gen_permutation_matrix(oneline_notation=perm, reflections=refx)

    # #将dict转化为representation.Representation
    rep_joints = group_rep_from_gens(G, rep_joints) 
    # 配置name
    rep_joints.name = name
    # 输入给G
    G.representations.update(**{name:rep_joints})
    return G

class SimpleEMLP(EquivariantModule):
    """EMLP网络构造

    Args:
        EquivariantModule (_type_): _description_
    """
    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 hidden_dims = [256, 256, 256],
                 bias: bool = True,
                 actor: bool = True,
                 activation: str = "ReLU"):
        super().__init__()
        self.out_type = out_type
        gspace = in_type.gspace
        group = gspace.fibergroup
        
        layer_in_type = in_type
        self.net = escnn.nn.SequentialModule()
        for n in range(len(hidden_dims)):
            layer_out_type = FieldType(gspace, [group.regular_representation] * int((hidden_dims[n] / group.order())))

            self.net.add_module(f"linear_{n}: in={layer_in_type.size}-out={layer_out_type.size}",
                             escnn.nn.Linear(layer_in_type, layer_out_type, bias=bias))
            self.net.add_module(f"act_{n}", self.get_activation(activation, layer_out_type))

            layer_in_type = layer_out_type

        if actor: 
            self.net.add_module(f"linear_{len(hidden_dims)}: in={layer_in_type.size}-out={out_type.size}",
                                escnn.nn.Linear(layer_in_type, out_type, bias=bias))
            self.extra_layer = None
        else:
            num_inv_features = len(layer_in_type.irreps)
            self.extra_layer = torch.nn.Linear(num_inv_features, out_type.size, bias=False)

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        x= self.net(x)
        if self.extra_layer:
            x = self.extra_layer(x.tensor)
        return x

    @staticmethod
    def get_activation(activation: str, hidden_type: FieldType) -> EquivariantModule:
        if activation.lower() == "relu":
            return escnn.nn.ReLU(hidden_type)
        elif activation.lower() == "elu":
            return escnn.nn.ELU(hidden_type)
        elif activation.lower() == "lrelu":
            return escnn.nn.LeakyReLU(hidden_type)
        else:
            raise NotImplementedError

    def evaluate_output_shape(self, input_shape):
        """Returns the output shape of the model given an input shape."""
        batch_size = input_shape[0]
        return batch_size, self.out_type.size

    def export(self):
        """Exports the model to a torch.nn.Sequential instance."""
        sequential = nn.Sequential()
        for name, module in self.net.named_children():
            sequential.add_module(name, module.export())
        return sequential

def get_symm_tensor(data:torch.Tensor, G:escnn.group.Group, reprs:List[str])->torch.Tensor:
    """将data这个torch.tensor转化为对称后的结构

    Args:
        data (torch.Tensor): (Batch, N)
        G (escnn.group.Group): 一个escnn的群论group
        reprs (List): represetations的列表，会按照顺序对data进行对称计算

    Returns:
        torch.Tensor: 返回的是对称后的结果，与data保持相同的device
    """
    # 要求data数据是torch.tensor
    assert isinstance(data, torch.Tensor), data.ndim <= 2
    # 整理data的shape和 repr要是列表
    data = data[None] if data.ndim == 1 else data
    reprs = [reprs] if not isinstance(reprs, List) else reprs
    # 获取device
    device = data.device
    # 开始转换
    curr_ind = 0
    res = []
    for repr in reprs:
        res.append(
            (torch.as_tensor(G.representations[repr](G.elements[1]), dtype=torch.float32, device=device) \
                @ data.T[curr_ind:curr_ind+G.representations[repr].size]
                ).T 
        )
        curr_ind += G.representations[repr].size
    # 确保reprs和data的维度是吻合的
    assert curr_ind == data.shape[-1]
    # 返回结果
    return torch.concat(res, dim=-1)



## 建立group和space
G = CyclicGroup(2)
gspace = escnn.gspaces.no_base_space(G)

# 添加需要的变换函数

#### obs使用的
# 3
add_repr_to_gspace(G, [0, 1, 2], [-1, 1, -1], 'base_ang_vel')
# 3
add_repr_to_gspace(G, [0, 1, 2], [1, -1, 1], 'projected_gravity')
# 29, actor obs 中几个29维度的子观测, 对称的方法一致，所以可以不变就能复用
add_repr_to_gspace(G, [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 13, 14, 22, 23, 24, 25, 26, 27, 28, 15, 16, 17, 18, 19, 20, 21], 
                   [1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1], 
                   'dof_pos')
# 2
add_repr_to_gspace(G, [0, 1], [-1, -1], 'phase')

# 3
add_repr_to_gspace(G, [0, 1, 2], [1, -1, 1], 'target_base_pos_delta')
# 4
add_repr_to_gspace(G, [0, 1, 2, 3], [-1, 1, -1, 1], 'target_base_quat_delta')
# 203
add_repr_to_gspace(G, [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 
                    100, 101, 102, 103, 104, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,],
                [1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1],
                'target_keyponts_delta')

# 3
add_repr_to_gspace(G, [0, 1, 2], [1, -1, -1], 'commands')

# add_repr_to_gspace(G, [], [], 'wm_feature')

#### critic 使用的
# 输入相比actor的obs还多了一些
# TODO: 这里需要添加一些新的变换

# 输出不变
add_repr_to_gspace(G, [0], [1], 'critic_value')

#### action使用的
add_repr_to_gspace(G, [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 13, 14, 22, 23, 24, 25, 26, 27, 28, 15, 16, 17, 18, 19, 20, 21],
                   [1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1],
                   'actions')

representations = ['base_ang_vel', 
                   'projected_gravity', 
                    *('actions',)*2, 
                    'phase',
                    *('actions',)] * 5

representations_commands = ['commands']

representations_wm_feature = ['wm_feature']

representations_action = ['actions']

representations_crtic = ['base_lin_vel', 
                         'base_ang_vel', 
                         'projected_gravity', 
                         'commands', 
                         *('dof_pos_vel_action',)*2, 
                         'phase', 
                         *('dof_pos_vel_action',),
                         'heights_obs']


"""一个标准的obs输入，镜面对称用到的transition排序"""
OBS_TRANS_NAME = [
            "base_ang_vel", 
            "projected_gravity", 
            "dof_pos", 
            'dof_pos', 
            'phase',
            'dof_pos',
            ]

"""一个标准的critic obs输入，镜面对称用到的transition排序"""
CRITIC_OBS_TRANS_NAME = OBS_TRANS_NAME + [ ]

CRITIC_VAL_TRANS_NAME = ['critic_value']


"""一个标准的action输入，镜面对称用到的transition排序"""
ACT_TRANS_NAME = ["actions"]


### 129+ 203 = 332
"""
self.obs_buf = torch.cat((  self.base_ang_vel * self.obs_scales.ang_vel, ### 3
        self.projected_gravity, ### 3
        (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, ### 29
        self.dof_vel * self.obs_scales.dof_vel,### 29
        self.actions,  ## 29
        self.target_base_pos_delta * 100.0,  ## 3
        self.target_base_quat_delta,  ## 4
        (self.target_dof_pos- self.default_dof_pos)* self.obs_scales.dof_pos,  ## 29
        self.target_keyponts_delta.view(self.num_envs, -1), ### 203   包含 quat ，不能乘系数
        ),dim=-1)
"""