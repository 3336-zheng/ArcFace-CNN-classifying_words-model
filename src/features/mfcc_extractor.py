"""MFCC 16维特征提取器 - 传统手工特征"""

import numpy as np
import librosa
import os
from tqdm import tqdm

from .preprocessor import DataPreprocessor
from ..utils.audio import safe_load_audio


class MFCCFeatureExtractor:
    """提取 MFCC 为主的 16 维手工特征"""

    def __init__(self, sampling_rate=16000):
        self.sampling_rate = sampling_rate

    def extract_features(self, audio_path):
        """从音频文件提取 16 维特征"""
        try:
            y, sr = safe_load_audio(audio_path, self.sampling_rate)
            if y is None:
                return None

            if len(y) > sr * 3:
                start = (len(y) - sr * 3) // 2
                y = y[start:start + sr * 3]
            if len(y) == 0:
                print(f"音频为空: {audio_path}")
                return None

            features = []

            # 时域特征 4 维
            rms = librosa.feature.rms(y=y)
            features.extend([np.mean(rms), np.std(rms), np.max(rms)])
            zcr = librosa.feature.zero_crossing_rate(y)
            features.append(np.mean(zcr))

            # 频域特征 8 维
            try:
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                features.extend(np.mean(mfcc, axis=1)[:8])
            except Exception as e:
                print(f"MFCC提取失败 {audio_path}: {e}")
                features.extend([0] * 8)

            try:
                features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
                features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
            except Exception:
                features.extend([0, 0])

            # 基频特征 2 维
            features.extend(self._extract_pitch(y, sr))

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


def main():
    """MFCC 16维特征提取流程"""
    import yaml

    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'default.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("=" * 50)
    print("MFCC 16维特征提取")
    print("=" * 50)

    preprocessor = DataPreprocessor()
    data = preprocessor.load_and_analyze_data(config['data']['jsonl_train'])
    command_mapping = preprocessor.create_command_mapping(
        min_samples=config['features']['min_samples_per_class']
    )
    preprocessor.save_command_mapping_json("command_mapping.json")

    extractor = MFCCFeatureExtractor()
    X, y = extractor.extract_features_batch(data, command_mapping)

    if X is not None:
        np.save("features.npy", X)
        np.save("labels.npy", y)
        print(f"✓ 特征已保存: {X.shape}")


if __name__ == "__main__":
    main()
