from typing import List, Optional
import torch
import torch.nn as nn
from utils import init_weights


class SingleInputEmbedding(nn.Module):

    def __init__(self,
                 in_channel: int,  # 输入特征的通道数
                 out_channel: int) -> None:  # 输出特征的通道数
        super(SingleInputEmbedding, self).__init__()
        self.embed = nn.Sequential(  # 嵌入模块 self.embed 包括三组重复的操作
            # nn.Sequential：一个有序容器，按顺序堆叠一系列层，每次调用 forward 时会依次执行这些层。
            # 这种设计使得嵌入模块在特征表示学习上更具表现力，能够捕获输入数据的深层结构信息。
            nn.Linear(in_channel, out_channel),  # 全连接层，用于对输入特征进行线性变换。
            nn.LayerNorm(out_channel),  # 层归一化，对输入特征进行标准化，提升训练稳定性并使特征分布更平滑
            nn.ReLU(inplace=True),  # 非线性激活函数，提升模型非线性表达能力
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights)  # 为网络中的所有层自定义初始化权重，确保模型从一个良好的初始状态开始训练，提升收敛速度和模型性能。

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 接受一个形状为 (batch_size, in_channel) （批次大小，输入特征通道数）的输入张量 x。
        return self.embed(x)
        # 依次通过嵌入模块中的各层，最终返回一个形状为 (batch_size, out_channel) （批次大小，经嵌入模块变换后的特征通道数）的输出。


class MultipleInputEmbedding(nn.Module):  # 可嵌入多个输入特征的神经网络模块
    # 处理多个连续和可选的分类输入，并最终得到一个特征嵌入的输出。
    def __init__(self,
                 in_channels: List[int],  # 一个 List[int] 类型的参数，表示每个连续输入的特征通道数。
                 out_channel: int) -> None:  # 一个整数，表示输出特征的通道数。
        super(MultipleInputEmbedding, self).__init__()
        self.module_list = nn.ModuleList(  # 使用 nn.ModuleList 创建的多个 nn.Sequential 模块。每个模块用于处理一个连续输入，并将其转换为指定的输出通道数。
            [nn.Sequential(nn.Linear(in_channel, out_channel),  # 一个 nn.Linear 层，将输入通道映射到 out_channel。
                           nn.LayerNorm(out_channel),  # 一个 nn.LayerNorm 层，对输出进行层归一化。
                           nn.ReLU(inplace=True),  # 一个 nn.ReLU 激活函数，使用 inplace=True 以减少内存占用
                           nn.Linear(out_channel, out_channel))  # 一个最终的 nn.Linear 层，将特征进一步转换为out_channel。
             for in_channel in in_channels])
        # nn.ModuleList：一个存放子模块的容器，用于存储多个子网络。
        # 与普通的 Python 列表不同，ModuleList 会将内部模块注册为模型的子模块，从而让 PyTorch 自动管理它们的参数。
        self.aggr_embed = nn.Sequential(  # 用于对所有特征聚合后的输出进行进一步的层归一化和非线性处理。
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights)  # 对模型中的所有层应用 init_weights 函数，用于自定义权重初始化。

    def forward(self,  # 向前传播
                continuous_inputs: List[torch.Tensor],
                # 每个张量形状(batch_size,in_channels[i])。多个连续输入特征，每个特征有不同的通道数
                # categorical_inputs每个张量的形状与 continuous_inputs 类似，表示分类输入特征。
                categorical_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        for i in range(len(self.module_list)):
            continuous_inputs[i] = self.module_list[i](continuous_inputs[i])
            # 列表中的每个元素都通过相应的 self.module_list 中的模块进行处理。
        output = torch.stack(continuous_inputs).sum(dim=0)
        # 将所有处理后的连续输入按通道维度堆叠并相加。
        if categorical_inputs is not None:
            output += torch.stack(categorical_inputs).sum(dim=0)
            # 如果存在分类输入，则将所有处理后的分类输入同样堆叠并相加，并加到输出中。
        return self.aggr_embed(output)  # 对聚合后的结果进行进一步的处理，最终得到输出。(batch_size,out_channel)
