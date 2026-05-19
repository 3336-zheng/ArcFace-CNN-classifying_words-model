"""TinyCNN 模型定义 - 3层CNN + 全连接，用于语音特征提取"""

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """轻量级 CNN，输入 LogMel 特征，输出嵌入向量"""

    def __init__(self, emb=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(40, 64, 5, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, stride=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, emb)

    def forward(self, x):
        h = self.conv(x).squeeze(-1)
        return self.fc(h)


class EmbedOnly(nn.Module):
    """仅嵌入部分，用于 ONNX 导出"""

    def __init__(self, ckpt_path, emb=64):
        super().__init__()
        self.backbone = TinyCNN(emb=emb)
        self.backbone.load_state_dict(
            torch.load(ckpt_path, map_location='cpu')
        )

    def forward(self, x):
        return self.backbone(x)
