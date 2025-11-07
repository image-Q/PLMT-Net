# 定义了一个名为 SoftTargetCrossEntropyLoss 的 PyTorch 自定义损失函数。
import torch  # 导入 PyTorch 库，用于进行张量操作和构建神经网络。
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块，其中包含定义各种神经网络层和损失函数的类。
import torch.nn.functional as F  # 入 PyTorch 中的函数模块，其中包含了各种激活函数和损失函数的实现。


# 定义了一个名为 SoftTargetCrossEntropyLoss 的类，这个类继承自 nn.Module，
# 表示它是一个 PyTorch 模块，可以包含可学习的参数和计算方法。
class SoftTargetCrossEntropyLoss(nn.Module):
    # 定义了 SoftTargetCrossEntropyLoss 类的构造函数。这个函数接受一个参数
    # reduction指定损失函数的归约方式，可以是 'mean'（默认，计算平均损失）、
    # 'sum'（求和损失）或 'none'（不进行归约）。
    def __init__(self, reduction: str = 'mean') -> None:
        # 调用父类 nn.Module 的构造函数，确保正确地初始化了这个自定义损失函数。
        super(SoftTargetCrossEntropyLoss, self).__init__()
        # 将构造函数中传入的参数保存为类的属性，以便在其他方法中使用。
        self.reduction = reduction

    # 定义了损失函数的前向传播方法。这个方法接受两个张量输入：
    # pred：模型的预测输出，通常是经过 softmax 处理后的概率分布。
    # target：真实的目标标签，通常是经过独热编码的标签或软标签。
    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        # 算交叉熵损失，根据预测值 pred 和目标值 target，
        # 采用 PyTorch 的函数模块中的 F.log_softmax() 计算 softmax 的对数，并与目标值相乘，
        # 然后使用 torch.sum() 沿着最后一个维度求和，得到每个样本的交叉熵损失。
        cross_entropy = torch.sum(-target * F.log_softmax(pred, dim=-1), dim=-1)
        # 根据 self.reduction 的值，采取不同的归约方式：
        if self.reduction == 'mean':  # 如果 self.reduction == 'mean'，则返回损失的平均值。
            return cross_entropy.mean()
        elif self.reduction == 'sum':  # 如果 self.reduction == 'sum'，则返回损失的总和。
            return cross_entropy.sum()
        elif self.reduction == 'none':  # 如果 self.reduction == 'none'，则返回未进行归约的损失张量。
            return cross_entropy
        else:  # 如果 self.reduction 的值不在上述三种情况中，则抛出值错误。
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))
