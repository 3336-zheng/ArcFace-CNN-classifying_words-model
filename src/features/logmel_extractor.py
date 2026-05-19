"""LogMel 特征提取器 - 使用 TinyCNN 提取 32 维嵌入向量"""

import torch
import numpy as np
import os
from tqdm import tqdm

from ..models.tiny_cnn import TinyCNN
from ..utils.audio import safe_load_audio, extract_logmel


class LogMelEmbeddingExtractor:
    """使用训练好的 TinyCNN 从 LogMel 特征中提取嵌入向量"""

    def __init__(self, ckpt_path, emb=32, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.emb = emb

        self.net = TinyCNN(emb=emb).to(self.device)
        self.net.load_state_dict(
            torch.load(ckpt_path, map_location=self.device)
        )
        self.net.eval()

    @torch.no_grad()
    def extract_embedding(self, audio_path):
        """提取单个音频的嵌入向量"""
        y, sr = safe_load_audio(audio_path)
        if y is None:
            return None

        if len(y) > 3 * sr:
            y = y[:3 * sr]

        logmel = extract_logmel(y, sr)
        x = torch.tensor(logmel, dtype=torch.float32).unsqueeze(0).to(self.device)
        emb = self.net(x).cpu().numpy().squeeze()
        return emb

    def extract_batch(self, data_list, desc="提取嵌入"):
        """批量提取嵌入向量"""
        X, y_labels = [], []

        for it in tqdm(data_list, desc=desc):
            e = self.extract_embedding(it["audio"]["path"])
            if e is not None:
                X.append(e)
                y_labels.append(it["sentence"])

        return np.array(X), np.array(y_labels)
