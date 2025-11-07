# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data


class TemporalData(Data):

    def __init__(self,
                 x: Optional[torch.Tensor] = None,
                 positions: Optional[torch.Tensor] = None,
                 edge_index: Optional[torch.Tensor] = None,
                 edge_attrs: Optional[List[torch.Tensor]] = None,
                 y: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 padding_mask: Optional[torch.Tensor] = None,
                 bos_mask: Optional[torch.Tensor] = None,
                 rotate_angles: Optional[torch.Tensor] = None,
                 lane_vectors: Optional[torch.Tensor] = None,
                 is_intersections: Optional[torch.Tensor] = None,
                 turn_directions: Optional[torch.Tensor] = None,
                 traffic_controls: Optional[torch.Tensor] = None,
                 lane_actor_index: Optional[torch.Tensor] = None,
                 lane_actor_vectors: Optional[torch.Tensor] = None,
                 seq_id: Optional[int] = None,
                 **kwargs) -> None:
        if x is None:
            super(TemporalData, self).__init__()
            return
        super(TemporalData, self).__init__(x=x, positions=positions, edge_index=edge_index, y=y, num_nodes=num_nodes,
                                           padding_mask=padding_mask, bos_mask=bos_mask, rotate_angles=rotate_angles,
                                           lane_vectors=lane_vectors, is_intersections=is_intersections,
                                           turn_directions=turn_directions, traffic_controls=traffic_controls,
                                           lane_actor_index=lane_actor_index, lane_actor_vectors=lane_actor_vectors,
                                           seq_id=seq_id, **kwargs)
        if edge_attrs is not None:
            for t in range(self.x.size(1)):
                self[f'edge_attr_{t}'] = edge_attrs[t]

    def __inc__(self, key, value, *args):
        if key == 'lane_actor_index':
            return torch.tensor([[self['lane_vectors'].size(0)], [self.num_nodes]])
        else:
            return super().__inc__(key, value)


class DistanceDropEdge(object):

    def __init__(self, max_distance: Optional[float] = None) -> None:
        self.max_distance = max_distance

    def __call__(self,
                 edge_index: torch.Tensor,
                 edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.max_distance is None:
            return edge_index, edge_attr
        row, col = edge_index
        mask = torch.norm(edge_attr, p=2, dim=-1) < self.max_distance
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        edge_attr = edge_attr[mask]
        return edge_index, edge_attr


class ImprovedDropEdge(object):

    def __init__(self, alpha, beta, gamma, min_score=0.1, rate=0.8) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.min_score = min_score
        self.rate = rate

    def __call__(self,
                 edge_index: torch.Tensor,
                 edge_attr: torch.Tensor,
                 data,
                 t) -> Tuple[torch.Tensor, torch.Tensor]:

        # if self.max_distance is None:
        #     return edge_index, edge_attr
        # row, col = edge_index
        # mask = torch.norm(edge_attr, p=2, dim=-1) < self.max_distance
        # edge_index = torch.stack([row[mask], col[mask]], dim=0)
        # edge_attr = edge_attr[mask]

        # def aaa(edge_index, edge_attr, data, t):
        #     positions = data["positions"]
        #     velocity = data["velocity"]

        #     row, col = edge_index

        #     score_list = []
        #     # 提供给后面注意力部分使用，避免后面重复计算
        #     theta_list = torch.zeros_like(row).float()
        #     dv_list = torch.zeros_like(row).float()
        #     d_list = torch.zeros_like(row).float()

        #     for edge_i, (i, j) in enumerate(zip(row, col)):
        #         pi1 = positions[i, t]
        #         pi2 = positions[i, t + 1]
        #         pj1 = positions[j, t]
        #         pj2 = positions[j, t + 1]
        #         vi = velocity[i, t]
        #         vj = velocity[j, t]

        #         d = torch.norm(pj1 - pi1, p=2, dim=-1).item()
        #         dv = (vi - vj).abs().item()
        #         veci = pi2 - pi1
        #         vecj = pj2 - pj1
        #         theta = self.calculate_angle_radians(veci.cpu().numpy(), vecj.cpu().numpy())

        #         if theta is None:
        #             theta_list[edge_i] = torch.pi / 2.0
        #         else:
        #             theta_list[edge_i] = theta
        #         dv_list[edge_i] = dv

        #         if theta is not None:
        #             score = -self.alpha * d - self.beta * dv + self.gamma * torch.cos(theta)
        #         else:
        #             score = -self.alpha * d - self.beta * dv

        #         score_list.append([edge_i, score])

        #     score_list_t = deepcopy(score_list)

        #     score_list = sorted(score_list, key=lambda x: x[1], reverse=True)

        #     # #选出得分最高的部分
        #     # mask = torch.zeros_like(row).bool()
        #     # total_num = len(score_list)
        #     # remain_num = int(total_num * self.rate)
        #     # for i in range(remain_num):
        #     #     edge_i = score_list[i][0]
        #     #     mask[edge_i] = True

        #     # 选出得分大于阈值的部分
        #     mask = torch.zeros_like(row).bool()
        #     total_num = len(score_list)
        #     for i in range(total_num):
        #         if score_list[i][1] > self.min_score:
        #             edge_i = score_list[i][0]
        #             mask[edge_i] = True

        #     edge_index = torch.stack([row[mask], col[mask]], dim=0)
        #     theta_list = theta_list[mask]
        #     dv_list = dv_list[mask]
        #     d_list = d_list[mask]
        #     edge_attr = edge_attr[mask]

        #     return edge_index, edge_attr, theta_list, dv_list, d_list,score_list_t

        # # edge_index_0, edge_attr_0, theta_list_0, dv_list_0, score_list_t = aaa(deepcopy(edge_index), deepcopy(edge_attr), deepcopy(data), t)

        positions = data["positions"]
        velocity = data["velocity"]

        row, col = edge_index

        # 向量化计算所有边的特征
        pi1 = positions[row, t]  # shape: [E, 2]
        pi2 = positions[row, t + 1]  # shape: [E, 2]
        pj1 = positions[col, t]  # shape: [E, 2]
        pj2 = positions[col, t + 1]  # shape: [E, 2]
        vi = velocity[row, t]  # shape: [E, 1]
        vj = velocity[col, t]  # shape: [E, 1]

        # 计算距离d (L2范数)
        d = torch.norm(pj1 - pi1, p=2, dim=-1)  # shape: [E]

        # 计算速度差dv
        dv = torch.abs(vi - vj)  # shape: [E]

        # 计算角度theta (向量化)
        veci = pi2 - pi1  # shape: [E, 2]
        vecj = pj2 - pj1  # shape: [E, 2]
        # 使用向量化角度计算替代循环
        theta = self.vectorized_angle(veci, vecj)  # shape: [E]

        # 计算得分score (完全在GPU上计算)
        score = -self.alpha * d - self.beta * dv + self.gamma * torch.cos(theta)  # shape: [E]

        # 选择top-k边
        remain_num = int(edge_index.size(1) * self.rate)
        _, topk_indices = torch.topk(score, k=remain_num, largest=True)
        topk_indices, _ = topk_indices.sort()
        # 使用高级索引直接获取结果
        edge_index = edge_index[:, topk_indices]
        theta_list = theta[topk_indices]
        dv_list = dv[topk_indices]
        d_list = d[topk_indices]
        edge_attr = edge_attr[topk_indices]

        # # 创建mask
        # mask = score > self.min_score  # [E]
        # # 应用mask
        # edge_index = edge_index[:, mask]
        # theta_list = theta[mask]
        # dv_list = dv[mask]
        # d_list = d[mask]
        # edge_attr = edge_attr[mask]

        return edge_index, edge_attr, theta_list, dv_list, d_list
    def vectorized_angle(self, a, b):
        """向量化计算两个向量间的夹角(弧度)"""
        # 点积: a·b = |a||b|cosθ
        dot_product = (a * b).sum(dim=-1)  # shape: [E]

        # 模长乘积: |a||b|
        norm_product = torch.norm(a, p=2, dim=-1) * torch.norm(b, p=2, dim=-1)  # shape: [E]

        # 避免除零错误
        safe_norm = torch.clamp(norm_product, min=1e-6)

        # cosθ = (a·b)/(|a||b|)
        cos_theta = dot_product / safe_norm

        # 处理数值误差保证在[-1, 1]范围内
        clamped_cos = torch.clamp(cos_theta, -1.0, 1.0)

        # 反余弦得到角度(弧度)
        return torch.acos(clamped_cos)  # shape: [E]

    def calculate_angle_radians(self, v1, v2):
        # 计算点积
        dot_product = np.dot(v1, v2)
        # 计算模长
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        # 处理模长为零的异常情况
        if norm_v1 == 0 or norm_v2 == 0:
            # raise ValueError("向量长度不能为零")
            return None
        # 计算余弦值并限制范围（防止浮点误差）
        cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
        # 计算弧度值
        angle_rad = np.arccos(cos_theta)
        return angle_rad


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)