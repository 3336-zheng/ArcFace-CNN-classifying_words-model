"""手工 32维特征提取器 - MFCC + 频域统计 + 基频"""

import numpy as np
import librosa
import os
from scipy import stats
from tqdm import tqdm

from ..utils.audio import safe_load_audio


class HandcraftFeatureExtractor:
    """提取 32 维手工特征: 时域4 + 频域20 + 基频2 + MFCC统计6"""

    def __init__(self, sampling_rate=16000):
        self.sampling_rate = sampling_rate

    def extract_features(self, audio_path):
        """从音频文件提取 32 维特征"""
        try:
            y, sr = safe_load_audio(audio_path, self.sampling_rate)
            if y is None:
                return None
            if len(y) > sr * 3:
                y = y[(len(y) - sr * 3) // 2: (len(y) - sr * 3) // 2 + sr * 3]
            if len(y) == 0:
                print(f"音频为空: {audio_path}")
                return None

            features = []

            # 1. 时域 4 维
            rms = librosa.feature.rms(y=y)
            features.extend([np.mean(rms), np.std(rms), np.max(rms)])
            zcr = librosa.feature.zero_crossing_rate(y=y)
            features.append(np.mean(zcr))

            # 2. 频域 20 维
            try:
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                features.extend(np.mean(mfcc, axis=1))  # 13 维

                for i in range(4):  # 4 段 log-energy
                    band = y[i * len(y) // 4: (i + 1) * len(y) // 4]
                    features.append(np.log10(np.mean(band ** 2) + 1e-8))

                features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
                features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
                features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
                # 13 + 4 + 1 + 2 = 20 维
            except Exception as e:
                print(f"频域提取失败 {audio_path}: {e}")
                features.extend([0] * 20)

            # 3. 基频 2 维
            features.extend(self._extract_pitch(y, sr))

            # 4. MFCC 统计量 6 维（补齐到 32）
            features.append(np.max(mfcc))
            features.append(np.std(mfcc))
            features.append(np.min(mfcc))
            features.append(stats.skew(mfcc, axis=None))
            features.append(stats.kurtosis(mfcc, axis=None))
            features.append(np.mean(np.ptp(mfcc, axis=1)))

            feature_array = np.array(features)
            if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
                print(f"无效特征值: {audio_path}")
                return None
            return feature_array

        except Exception as e:
            print(f"处理文件 {audio_path} 时出错: {e}")
            return None

    def _extract_pitch(self, y, sr):
        """提取基频特征"""
        try:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            magnitude_threshold = np.median(magnitudes) * 0.1

            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                magnitude = magnitudes[index, t]
                if 50 < pitch < 500 and magnitude > magnitude_threshold:
                    pitch_values.append(pitch)

            if len(pitch_values) > 5:
                return [np.mean(pitch_values), np.std(pitch_values)]
            return [0, 0]
        except Exception:
            return [0, 0]

    def extract_features_batch(self, data_list, command_mapping, max_files=None):
        """批量提取特征"""
        print("开始批量提取特征...")

        if max_files and max_files < len(data_list):
            data_list = data_list[:max_files]
            print(f"测试模式: 只处理前 {max_files} 个文件")

        features_list = []
        labels_list = []
        failed_count = 0

        for item in tqdm(data_list, desc="提取特征"):
            if item["sentence"] not in command_mapping:
                failed_count += 1
                continue

            features = self.extract_features(item["audio_path"])
            if features is not None:
                features_list.append(features)
                labels_list.append(command_mapping[item["sentence"]])
            else:
                failed_count += 1

        print(f"\n特征提取完成: 成功 {len(features_list)}，失败 {failed_count}")
        if not features_list:
            print("错误: 没有成功提取任何特征!")
            return None, None
        return np.array(features_list), np.array(labels_list)
