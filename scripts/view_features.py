"""查看特征和标签文件的内容"""

import numpy as np
import json
import argparse


def view_features_and_labels(features_path, labels_path, mapping_path=None):
    """查看特征和标签文件内容"""
    features = np.load(features_path)
    labels = np.load(labels_path)

    reverse_mapping = {}
    if mapping_path:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            command_mapping = json.load(f)
        reverse_mapping = {v: k for k, v in command_mapping.items()}

    print("=" * 50)
    print("特征数据概览")
    print("=" * 50)
    print(f"特征矩阵形状: {features.shape}")
    print(f"标签数量: {len(labels)}")

    print(f"\n前60个样本的特征:")
    for i in range(min(60, len(features))):
        label_name = reverse_mapping.get(labels[i], "未知")
        print(f"样本 {i + 1} (标签: {label_name}):")
        print(f"  特征向量: {features[i]}")
        print(f"  特征范围: [{np.min(features[i]):.4f}, {np.max(features[i]):.4f}]")
        print()

    print(f"\n特征统计:")
    print(f"  所有特征范围: [{np.min(features):.4f}, {np.max(features):.4f}]")
    print(f"  特征均值: {np.mean(features):.4f}")
    print(f"  特征标准差: {np.std(features):.4f}")

    print(f"\n标签分布:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        label_name = reverse_mapping.get(label, "未知")
        print(f"  {label_name}: {count} 个样本")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查看特征和标签文件")
    parser.add_argument("--features", default="features.npy", help="特征文件路径")
    parser.add_argument("--labels", default="labels.npy", help="标签文件路径")
    parser.add_argument("--mapping", default=None, help="指令映射 JSON 文件路径")
    args = parser.parse_args()

    view_features_and_labels(args.features, args.labels, args.mapping)
