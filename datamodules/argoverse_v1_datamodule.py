# 定义了一个pytorch lightning数据模块ArgoverseV1DataModule，用于处理ArgoverseV1数据集
# 导入了Callable, Optional，用于指定函数参数和返回值的类型，加代码的可读性和健壮性，提高代码的可维护性。
from typing import Callable, Optional
# 从 pytorch_lightning 库中导入 LightningDataModule 类，用于创建自定义数据模块。
from pytorch_lightning import LightningDataModule
# 从 torch_geometric 库中导入 DataLoader 类，用于加载图数据。
from torch_geometric.data import DataLoader
# 从自定义的 datasets 模块中导入 ArgoverseV1Dataset 类，用于加载 ArgoverseV1 数据集。
from datasets import ArgoverseV1Dataset


class ArgoverseV1DataModule(LightningDataModule):
    # 定义了一个名为 ArgoverseV1DataModule 的类，继承自 LightningDataModule 类。

    def __init__(self,  # 定义了初始化函数，用于初始化数据模块的各种属性。
                 root: str,  # 数据集的根目录，类型为字符串。
                 train_batch_size: int,  # 训练批量大小，类型为整数。
                 val_batch_size: int,  # 验证批量大小，类型为整数。
                 shuffle: bool = True,  # 是否在数据加载器中对数据进行随机洗牌，默认为 True，类型为布尔值。
                 num_workers: int = 8,  # 数据加载器中用于数据加载的进程数量，默认为 8，类型为整数。
                 pin_memory: bool = True,  # 是否将数据加载到 GPU 的固定内存中，默认为 True，类型为布尔值。
                 persistent_workers: bool = True,  # 是否使用持久化的工作进程，默认为 True，类型为布尔值。
                 train_transform: Optional[Callable] = None,  # 训练数据的转换函数，可选参数，默认为 None，类型为可调用对象。
                 val_transform: Optional[Callable] = None,  # 验证数据的转换函数，可选参数，默认为 None，类型为可调用对象。
                 local_radius: float = 50) -> None:  # 用于确定局部邻域的半径大小，默认为 50，类型为浮点数。
        super(ArgoverseV1DataModule, self).__init__()  # 调用父类的初始化函数，确保所有父类属性都被正确初始化。
        self.val_dataset = None
        self.train_dataset = None
        self.root = root  # 设置数据集的根目录。
        self.train_batch_size = train_batch_size  # 设置训练批量大小。
        self.val_batch_size = val_batch_size  # 设置验证批量大小。
        self.shuffle = shuffle  # 设置是否对数据进行随机洗牌。
        self.pin_memory = pin_memory  # 设置是否将数据加载到 GPU 的固定内存中。
        self.persistent_workers = persistent_workers  # 设置是否使用持久化的工作进程，提高加载效率。
        self.num_workers = num_workers  # 设置数据加载器中用于数据加载的进程数量。
        self.train_transform = train_transform  # 设置训练数据的转换函数。
        self.val_transform = val_transform  # 设置验证数据的转换函数。
        self.local_radius = local_radius  # 设置局部邻域的半径大小。

    def prepare_data(self) -> None:  # 定义了数据准备方法，用于在训练之前下载并准备数据集。只需要执行一次
        ArgoverseV1Dataset(self.root, 'train', self.train_transform, self.local_radius)  # 下载并准备训练数据集。
        ArgoverseV1Dataset(self.root, 'val', self.val_transform, self.local_radius)  # 下载并准备验证数据集。

    def setup(self, stage: Optional[str] = None) -> None:  # 定义了数据加载方法，用于设置训练和验证数据集。
        self.train_dataset = ArgoverseV1Dataset(self.root, 'train', self.train_transform, self.local_radius)  # 设置训练数据集。
        self.val_dataset = ArgoverseV1Dataset(self.root, 'val', self.val_transform, self.local_radius)   # 设置验证数据集。

    def train_dataloader(self):  # 定义了训练数据加载器方法，用于返回训练数据的 DataLoader 实例。
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,  # 返回训练数据加载器。
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):  # 定义了验证数据加载器方法，用于返回验证数据的 DataLoader 实例。
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,  # 返回验证数据加载器。
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)
