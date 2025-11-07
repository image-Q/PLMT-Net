import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import LaplaceNLLLoss
from losses import SoftTargetCrossEntropyLoss
from metrics import ADE
from metrics import FDE
from metrics import MR
from models import GlobalInteractor
from models import LocalEncoder
from models import MLPDecoder
from utils import TemporalData


class HiVT(pl.LightningModule):  # 定义lightning module

    def __init__(self,  # 类初始化
                 historical_steps: int,  # 历史时间步数量，模型考虑之前20帧
                 future_steps: int,  # 未来时间步数量，预测未来30帧
                 num_modes: int,  # 预测模态数量，6
                 rotate: bool,  # 处理数据时的旋转操作，true
                 node_dim: int,  # node feature 维度 节点
                 edge_dim: int,  # edge feature 维度 边
                 embed_dim: int,  # 嵌入向量维度
                 num_heads: int,  # 注意力机制中head数量
                 dropout: float,  # 用于正则化 防止过拟合的随机失活比例
                 num_temporal_layers: int,  # 模型中的两个子module,layer数量
                 num_global_layers: int,  # 全局交互的layer数量
                 local_radius: float,
                 parallel: bool,
                 lr: float,  # 优化器学习率
                 weight_decay: float,  # 权重衰减
                 T_max: int,  # 学习率调度器的周期
                 **kwargs) -> None:
        super(HiVT, self).__init__()
        self.save_hyperparameters()  # 保存输入的超参数，方便后续加载和记录。
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.parallel = parallel
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max

        self.local_encoder = LocalEncoder(historical_steps=historical_steps,
                                          node_dim=node_dim,
                                          edge_dim=edge_dim,
                                          embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          num_temporal_layers=num_temporal_layers,  # 时间维度上的网络深度。
                                          local_radius=local_radius,  # 定义局部交互范围。
                                          parallel=parallel)
        # 局部编码器，处理局部交互，生成局部嵌入特征。输入：包括节点和边的特征。
        self.global_interactor = GlobalInteractor(historical_steps=historical_steps,
                                                  embed_dim=embed_dim,
                                                  edge_dim=edge_dim,
                                                  num_modes=num_modes,
                                                  num_heads=num_heads,
                                                  num_layers=num_global_layers,  # 全局交互网络的深度。
                                                  dropout=dropout,
                                                  rotate=rotate)  # 是否对坐标进行旋转归一化。
        # 全局交互模块，对所有轨迹进行全局交互建模，生成全局嵌入。输入：局部嵌入特征。
        self.decoder = MLPDecoder(local_channels=embed_dim,
                                  global_channels=embed_dim,
                                  future_steps=future_steps,
                                  num_modes=num_modes,
                                  uncertain=True)
        # 解码器，根据局部和全局特征解码多模态预测轨迹和每种模态的权重。输出y_hat-预测的多模态轨迹。pi-每种轨迹模态的概率分布。
        self.reg_loss = LaplaceNLLLoss(reduction='mean')
        # 拉普拉斯负对数似然损失，用于回归任务。
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')
        # 交叉熵损失，用于分类模态权重。

        self.minADE = ADE()  # 平均偏差误差，衡量轨迹整体的精度。
        self.minFDE = FDE()  # 终点偏差误差，衡量轨迹终点的预测精度。
        self.minMR = MR()  # 容错率，衡量预测未覆盖真实轨迹的比例。

    def forward(self, data: TemporalData):
        if self.rotate:  # 计算旋转矩阵，根据rotate参数决定是否对轨迹坐标进行旋转归一化
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(data['rotate_angles'])
            cos_vals = torch.cos(data['rotate_angles'])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            if data.y is not None:
                data.y = torch.bmm(data.y, rotate_mat)
            data['rotate_mat'] = rotate_mat
        else:
            data['rotate_mat'] = None

        local_embed = self.local_encoder(data=data)  # 在一定半径范围内，50
        global_embed = self.global_interactor(data=data, local_embed=local_embed)  # 生成全局编码，所有agent交互
        # 调用local_encoder和global_interacto分别生成局部和全局嵌入
        y_hat, pi = self.decoder(local_embed=local_embed, global_embed=global_embed)  # 多模态轨迹y_hat,对应的目标轨迹pi
        return y_hat, pi  # 用decoder解码得到预测轨迹y_hat和模态权重pi

    def training_step(self, data, batch_idx):  # 训练过程
        y_hat, pi = self(data)  # 获取y_hat,pi
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        valid_steps = reg_mask.sum(dim=-1)  # 每个节点在时间步中的有效步数
        cls_mask = valid_steps > 0  # 选择至少一个有效步数的节点
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)  # 筛选最优
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        soft_target = F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0).t().detach()  # 计算分类loss
        cls_loss = self.cls_loss(pi[cls_mask], soft_target)
        loss = reg_loss + cls_loss
        self.log('train_reg_loss', reg_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss
    # 计算预测轨迹与真实轨迹的误差，使用 reg_loss 评估。
    # 根据模态权重 pi 计算分类损失 cls_loss。
    # 将两种损失加权合并，作为最终的训练损失。

    def validation_step(self, data, batch_idx):  # 验证过程
        y_hat, pi = self(data)
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        self.log('val_reg_loss', reg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        y_hat_agent = y_hat[:, data['agent_index'], :, : 2]
        y_agent = data.y[data['agent_index']]
        fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        best_mode_agent = fde_agent.argmin(dim=0)
        y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(data.num_graphs)]  # 得到best trajectory
        self.minADE.update(y_hat_best_agent, y_agent)
        self.minFDE.update(y_hat_best_agent, y_agent)
        self.minMR.update(y_hat_best_agent, y_agent)
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minMR', self.minMR, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        # 计算预测轨迹的误差。
        # 根据终点的偏差选择最优模态，用于更新 minADE、minFDE 和 minMR 指标。
        # 记录各指标，用于验证阶段的评估。

    def configure_optimizers(self):  # 优化器配置
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [scheduler]
    #     自定义权重衰减规则：
    #         权重分为两组：
    #             有衰减：线性层、卷积层等权重参数。
    #             无衰减：偏置参数和归一化层权重。
    #         防止某些层的参数意外遗漏。
    #     优化器：AdamW，适合深度学习的权重衰减优化方法。
    #     调度器：余弦退火学习率调度。

    @staticmethod
    def add_model_specific_args(parent_parser):  # 模型参数命令行配置
        parser = parent_parser.add_argument_group('HiVT')
        parser.add_argument('--historical_steps', type=int, default=20)
        parser.add_argument('--future_steps', type=int, default=30)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--rotate', type=bool, default=True)
        parser.add_argument('--node_dim', type=int, default=2)
        parser.add_argument('--edge_dim', type=int, default=2)
        parser.add_argument('--embed_dim', type=int, default=64)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num_temporal_layers', type=int, default=4)
        parser.add_argument('--num_global_layers', type=int, default=3)
        parser.add_argument('--local_radius', type=float, default=50)
        parser.add_argument('--parallel', type=bool, default=False)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        return parent_parser
    # 允许在命令行中传递参数，方便实验配置。
