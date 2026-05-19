"""ONNX 推理器 - 在 PC 上使用 ONNX 模型进行音频命令识别"""

import onnxruntime as ort
import numpy as np
import librosa
import json
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class TinyCNNInferencePC:
    """PC 端 ONNX 推理器"""

    def __init__(self, onnx_path, command_mapping_path):
        self.session = ort.InferenceSession(onnx_path)

        with open(command_mapping_path, 'r', encoding='utf-8') as f:
            self.command2id = json.load(f)
        self.id2command = {v: k for k, v in self.command2id.items()}

        self.sr = 16000
        self.sec = 3
        self.frame = 300
        self.n_mels = 40

        print(f"✅ 模型加载成功，支持 {len(self.command2id)} 个命令")
        print("支持的命令列表:")
        for cmd, idx in self.command2id.items():
            print(f"  {idx:2d}: {cmd}")

    def preprocess_audio(self, audio_path):
        """预处理音频，与训练时完全一致"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)

            target_length = self.sec * sr
            if len(y) > target_length:
                start = (len(y) - target_length) // 2
                y = y[start:start + target_length]
            elif len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), 'constant')

            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=512, hop_length=160, n_mels=self.n_mels
            )
            logmel = librosa.power_to_db(mel, ref=np.max)

            if logmel.shape[1] < self.frame:
                logmel = np.pad(logmel, ((0, 0), (0, self.frame - logmel.shape[1])), 'constant')
            else:
                logmel = logmel[:, :self.frame]

            return logmel.astype(np.float32)

        except Exception as e:
            print(f"❌ 音频预处理失败: {e}")
            return None

    def extract_embedding(self, audio_path):
        """提取 32 维特征向量"""
        logmel = self.preprocess_audio(audio_path)
        if logmel is None:
            return None

        input_data = np.expand_dims(logmel, axis=0)  # (1, 40, 300)

        try:
            inputs = {'fbank': input_data}
            outputs = self.session.run(['embedding'], inputs)
            return outputs[0][0]
        except Exception as e:
            print(f"❌ 模型推理失败: {e}")
            return None

    def predict_single(self, audio_path, classifier=None, scaler=None):
        """预测单个音频"""
        embedding = self.extract_embedding(audio_path)
        if embedding is None:
            return None, None

        if classifier is not None and scaler is not None:
            embedding_scaled = scaler.transform([embedding])
            pred_id = classifier.predict(embedding_scaled)[0]
            confidence = np.max(classifier.predict_proba(embedding_scaled))
            command = self.id2command.get(pred_id, "未知命令")
            return command, confidence
        return embedding, None

    def batch_predict(self, audio_dir, classifier=None, scaler=None):
        """批量预测目录中的音频文件"""
        results = []
        audio_files = [
            f for f in os.listdir(audio_dir)
            if f.endswith(('.wav', '.mp3', '.m4a', '.flac'))
        ]

        for audio_file in audio_files:
            audio_path = os.path.join(audio_dir, audio_file)
            command, confidence = self.predict_single(audio_path, classifier, scaler)

            if command is not None:
                if confidence is not None:
                    results.append({
                        'file': audio_file,
                        'command': command,
                        'confidence': f"{confidence:.4f}",
                        'status': '✅ 成功'
                    })
                else:
                    embedding = self.extract_embedding(audio_path)
                    results.append({
                        'file': audio_file,
                        'embedding': embedding.tolist() if embedding is not None else [],
                        'status': '✅ 特征提取成功'
                    })
            else:
                results.append({
                    'file': audio_file,
                    'status': '❌ 失败'
                })

        return results


def create_classifier_from_features(feature_file, label_file):
    """从特征文件创建分类器"""
    try:
        X_train = np.load(feature_file)
        y_train = np.load(label_file)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        classifier = KNeighborsClassifier(n_neighbors=1)
        classifier.fit(X_train_scaled, y_train)

        print(f"✅ 分类器创建成功，训练样本: {len(X_train)}")
        return classifier, scaler
    except Exception as e:
        print(f"❌ 分类器创建失败: {e}")
        return None, None
