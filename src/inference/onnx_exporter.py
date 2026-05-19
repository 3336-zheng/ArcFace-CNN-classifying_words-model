"""ONNX 模型导出器 - 将 TinyCNN backbone 导出为 ONNX 格式"""

import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.tiny_cnn import EmbedOnly


def export_to_onnx(ckpt_path, onnx_path="tiny_cnn_rpi.onnx", emb=64):
    """将训练好的模型导出为 ONNX 格式

    Args:
        ckpt_path: .pth 模型权重路径
        onnx_path: 输出 ONNX 文件路径
        emb: 嵌入向量维度
    """
    model = EmbedOnly(ckpt_path, emb=emb).eval()
    dummy = torch.randn(1, 40, 300)  # 1句 3秒 Fbank

    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['fbank'],
        output_names=['embedding'],
        opset_version=11,
        dynamic_axes=None
    )
    print(f"✅ ONNX 导出完成: {onnx_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="导出 TinyCNN 为 ONNX 格式")
    parser.add_argument("--ckpt", required=True, help="模型权重路径 (.pth)")
    parser.add_argument("--output", default="tiny_cnn_rpi.onnx", help="ONNX 输出路径")
    parser.add_argument("--emb", type=int, default=64, help="嵌入维度")
    args = parser.parse_args()

    export_to_onnx(args.ckpt, args.output, args.emb)
