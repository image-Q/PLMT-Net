# 定义了一个可以在 PyTorch 模型中使用的 Laplace 负对数似然损失函数，
# 可以根据需要对损失进行平均、求和或不进行归约。

import torch  # 导入pytorch库，用于构建神经网络和进行张量操作
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块，其中包含定义各种神经网络层和损失函数的类。


# 定义了一个名为 LaplaceNLLLoss 的类，这个类继承自 nn.Module，
# 表示它是一个 PyTorch 模块，可以包含可学习的参数和计算方法。
class LaplaceNLLLoss(nn.Module):
    # 定义LaplaceNLLLoss类的构造函数，接受两个参数eps和reduction
    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps  # Laplace 分布的尺度参数的最小值，以防止分母为零。
        self.reduction = reduction  # 指定损失函数的归约方式，可以是 'mean'（默认，计算平均损失）、'sum'（求和损失）或 'none'（不进行归约）

    # 定义了损失函数的前向传播方法。这个方法接受两个张量输入：
    # pred：模型的预测输出，其中包含了 Laplace 分布的位置参数和尺度参数。
    # target：真实的目标标签。
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 将预测张量 pred 沿着最后一个维度（即参数维度）分割成两个部分，分别表示 Laplace 分布的位置参数和尺度参数。
        loc, scale = pred.chunk(2, dim=-1)
        # 克隆尺度参数张量，以避免修改原始张量。
        scale = scale.clone()
        # 上下文管理器，禁用梯度计算，并对尺度参数进行截断，确保其不小于 eps。
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        # 计算 Laplace 分布的负对数似然，使用 Laplace 分布的概率密度函数。
        # 这里用到了 PyTorch 的张量操作，包括对数函数 torch.log()、绝对值函数 torch.abs() 和除法操作 /。
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
        # 根据 self.reduction 的值，采取不同的归约方式：
        if self.reduction == 'mean':  # 如果 self.reduction == 'mean'，则返回损失的平均值。
            return nll.mean()
        elif self.reduction == 'sum':  # 如果 self.reduction == 'sum'，则返回损失的总和。
            return nll.sum()
        elif self.reduction == 'none':  # 如果 self.reduction == 'none'，则返回未进行归约的损失张量。
            return nll
        else:  # 如果 self.reduction 的值不在上述三种情况中，则抛出值错误。
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))
