"""逻辑回归验证 - 评估特征的线性可分性"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import argparse


def verify(features_path, labels_path, n_splits=5):
    """使用逻辑回归 + 5折交叉验证评估特征质量"""
    X = np.load(features_path)
    y = np.load(labels_path)

    print(f"数据: {X.shape[0]} 条样本, {X.shape[1]} 维特征, {len(np.unique(y))} 个类别")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=1000, multi_class='ovr')

    y_pred = np.zeros_like(y)
    for train_idx, test_idx in cv.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = clf.predict(X[test_idx])

    print(classification_report(y, y_pred, digits=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="逻辑回归特征验证")
    parser.add_argument("--features", default="features.npy", help="特征文件路径")
    parser.add_argument("--labels", default="labels.npy", help="标签文件路径")
    args = parser.parse_args()

    verify(args.features, args.labels)
