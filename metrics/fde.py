# FDE最终位移误差
from typing import Any, Callable, Optional
import torch
from torchmetrics import Metric


class FDE(Metric):

    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None) -> None:
        super(FDE, self).__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step,
                                  process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    # 定义了指标的更新方法。这个方法接受两个张量输入：
    # pred：模型的预测输出。
    # target：真实的目标标签。
    # 在这个方法中，最终距离误差是通过计算预测值 pred 和目标值 target 中最后一个时间步的欧几里得距离得到的，
    # 然后求和，最后更新了指标的 'sum' 和 'count' 状态。
    def update(self,
               pred: torch.Tensor,
               target: torch.Tensor) -> None:
        self.sum += torch.norm(pred[:, -1] - target[:, -1], p=2, dim=-1).sum()
        self.count += pred.size(0)

    # 定义了计算指标的方法。这个方法不接受任何输入，在这个方法中，
    # 指标的计算是通过将 'sum' 状态除以 'count' 状态得到的。
    def compute(self) -> torch.Tensor:
        return self.sum / self.count
