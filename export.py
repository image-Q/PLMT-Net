from argparse import ArgumentParser
import torch
from torch_geometric.data import Data  # 用于构造模型所需的图数据
from models.hivt import HiVT  # 导入HiVT模型

def main():
    # 1. 固定 checkpoint 路径（你提供的 indir）
    ckpt_path = "/root/HiVT/lightning_logs/version_21/checkpoints/epoch=63-step=411903.ckpt"
    # 2. 固定 ONNX 输出路径
    onnx_path = "hivt_epoch63.onnx"

    # 解析模型特有参数（HiVT 需的配置，如隐藏层维度等，从 add_model_specific_args 继承）
    parser = ArgumentParser()
    parser = HiVT.add_model_specific_args(parser)
    # 若模型需要额外必选参数（如输入特征维度），可在此补充，示例：
    # parser.add_argument('--hidden_dim', type=int, default=128)
    args = parser.parse_args([])  # 空列表表示不依赖命令行传入，用默认值

    # --------------------------
    # 加载指定 checkpoint 的模型
    # --------------------------
    model = HiVT.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        **vars(args)  # 传入模型配置参数
    )
    model.eval()  # 切换到评估模式（关闭 dropout、BatchNorm）
    # 设备配置（自动用GPU或CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # --------------------------
    # 构造符合模型要求的图数据（Data 对象）
    # --------------------------
    # 关键：参数需与 Argoverse 数据集格式匹配（参考原训练时的输入形状）
    num_nodes = 10       # 单场景智能体数量（如10个车辆）
    seq_len = 50         # 历史轨迹长度（Argoverse 常用50帧，约2.5秒）
    node_feat_dim = 2    # 每个轨迹点特征（x、y 坐标）
    num_edges = num_nodes * (num_nodes - 1) * 2  # 无向边（全连接图）

    # 生成随机 dummy 数据（形状匹配即可，值不影响导出）
    traj = torch.randn(num_nodes, seq_len, node_feat_dim, device=device)  # 智能体轨迹
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).T  # 边索引（[2, 边数]）
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # 转为无向边

    # 构造模型输入的 Data 对象
    data = Data(
        traj=traj,          # 轨迹特征（属性名需与 HiVT.forward 中一致）
        edge_index=edge_index,  # 图边索引
        num_nodes=num_nodes     # 节点数量（模型需用此属性）
    ).to(device)

    # --------------------------
    # 用 JIT 包装模型，解决 Data 类型不兼容问题
    # --------------------------
    def jit_wrapper(traj, edge_index, num_nodes):
        # 内部重构 Data 对象，让 JIT 能识别属性访问
        dummy_data = Data(traj=traj, edge_index=edge_index, num_nodes=num_nodes)
        return model(dummy_data)  # 调用模型 forward

    # 生成 JIT 追踪模型（输入为 Data 拆解后的 Tensor）
    traced_model = torch.jit.trace(
        jit_wrapper,
        (data.traj, data.edge_index, torch.tensor(data.num_nodes, device=device))
    )

    # --------------------------
    # 导出 ONNX
    # --------------------------
    torch.onnx.export(
        model=traced_model,
        args=(data.traj, data.edge_index, torch.tensor(data.num_nodes, device=device)),
        f=onnx_path,
        export_params=True,  # 导出模型权重
        opset_version=12,    # 支持图相关算子（如 torch_scatter）
        do_constant_folding=True,  # 优化常量计算
        input_names=['traj', 'edge_index', 'num_nodes'],  # 输入节点名（后续部署用）
        output_names=['predicted_trajectory'],  # 输出节点名（预测的未来轨迹）
        dynamic_axes={       # 支持动态维度（适配不同场景大小）
            'traj': {0: 'num_nodes', 1: 'seq_len'},  # 智能体数量、历史长度可动态
            'edge_index': {1: 'num_edges'},          # 边数可动态
            'predicted_trajectory': {0: 'num_nodes', 1: 'pred_seq_len'}  # 预测长度可动态
        }
    )
    print(f"✅ ONNX 模型已导出至：{onnx_path}")
    print(f"📌 基于 checkpoint：{ckpt_path}")

if __name__ == '__main__':
    main()