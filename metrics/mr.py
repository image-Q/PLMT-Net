# MS失错率
from typing import Any, Callable, Optional

import torch
from torchmetrics import Metric


class MR(Metric):

    def __init__(self,
                 miss_threshold: float = 2.0,  # 一个浮点数，表示错失率的阈值。
                 compute_on_step: bool = True,  # 一个布尔值，指定是否在每个步骤上计算指标。
                 dist_sync_on_step: bool = False,  # 一个布尔值，指定是否在每个步骤上在分布式设置中同步指标。
                 process_group: Optional[Any] = None,  # 一个可选的处理组对象，用于分布式设置。
                 dist_sync_fn: Callable = None) -> None:  # 一个可调用对象，用于在分布式设置中同步指标。
        super(MR, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                                 process_group=process_group, dist_sync_fn=dist_sync_fn)
        # 使用 add_state() 方法向指标中添加了一个状态，命名为 'sum'，初始值为 0.0，
        # 并指定了在分布式设置中如何对这个状态进行归约，这里是求和。
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        # 使用 add_state() 方法向指标中添加了另一个状态，命名为 'count'，初始值为 0，
        # 并指定了在分布式设置中如何对这个状态进行归约，这里也是求和。
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')
        # 将输入的 miss_threshold 参数保存到类属性中，以便后续使用。
        self.miss_threshold = miss_threshold

    # 定义了指标的更新方法。这个方法接受两个张量输入：
    # pred：模型的预测输出。
    # target：真实的目标标签。
    # 在这个方法中，错失率是通过计算预测值 pred 和目标值 target 中
    # 最后一个时间步的欧几里得距离是否大于 miss_threshold 得到的，
    # 然后统计大于阈值的数量，最后更新了指标的 'sum' 和 'count' 状态。
    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor) -> None:
        self.sum += (torch.norm(pred[:, -1] - target[:, -1], p=2, dim=-1) > self.miss_threshold).sum()
        self.count += pred.size(0)

    # 定义了计算指标的方法。这个方法不接受任何输入，在这个方法中，
    # 错失率是通过将 'sum' 状态除以 'count' 状态得到的。
    def compute(self) -> torch.Tensor:
        return self.sum / self.count
