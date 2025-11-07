from argparse import ArgumentParser  # argparse是一个python模块，用途是命令行选项、参数、子命令的解释。
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint  # 训练过程中，保留训练数据
from datamodules import ArgoverseV1DataModule  # 导入数据集
from models.hivt import HiVT  # 导入HiVT模型

if __name__ == '__main__':
    pl.seed_everything(2022)  # 全局随机种子，保证实验可重复性。

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)  # 数据集存放根目录，要指定
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)  # 数据加载的并行线程数，影响速度
    parser.add_argument('--pin_memory', type=bool, default=True)  # 加速GPU数据传输的配置
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    # monitor模型选择依据的验证集指标
    parser.add_argument('--save_top_k', type=int, default=5)  # 保存最优的K个模型检查点
    parser = HiVT.add_model_specific_args(parser)  # 模型本身也会添加自己的特有参数
    args = parser.parse_args()

    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')  # 保留最佳模型
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[model_checkpoint])
    # 通过ModelCheckpoint回调保存最佳模型，根据验证集的指标
    model = HiVT(**vars(args))  # 模型初始化
    datamodule = ArgoverseV1DataModule.from_argparse_args(args)  # 数据准备
    trainer.fit(model, datamodule,ckpt_path="/root/autodl-tmp/HiVT/lightning_logs/version_20/checkpoints/epoch=50-step=328235.ckpt") 
    #trainer.fit(model, datamodule)  # 训练执行 model datamodule 通过lightning定义的两个参数
