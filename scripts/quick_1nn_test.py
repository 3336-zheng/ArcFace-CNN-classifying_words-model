"""快速 1-NN 测试 - 评估特征区分度上限"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import argparse


def quick_test(features_path, labels_path, cv=3):
    """1-NN 和 RandomForest 交叉验证"""
    X = np.load(features_path)
    y = np.load(labels_path)

    print(f"数据加载完成: {X.shape[0]} 条样本, {X.shape[1]} 维特征, {len(np.unique(y))} 个类别")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1-NN 交叉验证
    print("\n===== 1-Nearest-Neighbour (上限测试) =====")
    acc_1nn = cross_val_score(
        KNeighborsClassifier(n_neighbors=1),
        X_scaled, y, cv=cv, scoring='accuracy', n_jobs=-1
    ).mean()
    print(f"1-NN {cv}折交叉验证平均准确率: {acc_1nn:.4f}")

    # RandomForest 对比
    print("\n===== RandomForest-100 (对比) =====")
    acc_rf = cross_val_score(
        RandomForestClassifier(n_estimators=100, random_state=42),
        X_scaled, y, cv=cv, scoring='accuracy', n_jobs=-1
    ).mean()
    print(f"RandomForest-100 {cv}折交叉验证平均准确率: {acc_rf:.4f}")

    # 小结
    print(f"\n=== 结果摘要 ===")
    print(f"1-NN 上限 : {acc_1nn:.1%}")
    print(f"RF-100    : {acc_rf:.1%}")
    print("若 1-NN > 80% → 特征区分度足够，换非线性模型即可；"
          "若 < 50% → 需升维或重新提 embedding。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="快速 1-NN 测试")
    parser.add_argument("--features", required=True, help="特征文件路径")
    parser.add_argument("--labels", required=True, help="标签文件路径")
    parser.add_argument("--cv", type=int, default=3, help="交叉验证折数")
    args = parser.parse_args()

    quick_test(args.features, args.labels, args.cv)
