"""公共音频加载与预处理工具"""

import numpy as np
import librosa
import os


def safe_load_audio(audio_path, sr=16000, min_duration=0.5):
    """安全加载音频文件，处理各种异常情况

    Args:
        audio_path: 音频文件路径
        sr: 目标采样率
        min_duration: 最短音频时长（秒）

    Returns:
        (y, sr) 或 (None, None) 如果加载失败
    """
    if not os.path.exists(audio_path):
        print(f"文件不存在: {audio_path}")
        return None, None

    if os.path.getsize(audio_path) == 0:
        print(f"空文件: {audio_path}")
        return None, None

    try:
        y, sr = librosa.load(audio_path, sr=sr)
        if len(y) < sr * min_duration:
            print(f"音频过短: {audio_path} (长度: {len(y) / sr:.2f}秒)")
            return None, None
        return y, sr
    except Exception as e:
        print(f"加载音频失败 {audio_path}: {e}")
        return None, None


def pad_or_truncate(y, sr, max_duration=3):
    """将音频裁剪或填充到固定时长

    Args:
        y: 音频信号
        sr: 采样率
        max_duration: 目标时长（秒）

    Returns:
        处理后的音频信号
    """
    target_len = int(sr * max_duration)
    if len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), 'constant')
    return y


def spec_augment(logmel, freq_mask_param=8, time_mask_param=20, num_freq_masks=2, num_time_masks=2):
    """SpecAugment 数据增强 - 随机遮蔽频率和时间帧

    Args:
        logmel: LogMel 特征矩阵 (n_mels, n_frames)
        freq_mask_param: 频率遮蔽最大宽度
        time_mask_param: 时间遮蔽最大宽度
        num_freq_masks: 频率遮蔽次数
        num_time_masks: 时间遮蔽次数

    Returns:
        增强后的 LogMel 特征矩阵
    """
    augmented = logmel.copy()
    n_mels, n_frames = augmented.shape

    # 频率遮蔽
    for _ in range(num_freq_masks):
        f = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, max(n_mels - f, 1))
        augmented[f0:f0 + f, :] = 0

    # 时间遮蔽
    for _ in range(num_time_masks):
        t = np.random.randint(0, time_mask_param)
        t0 = np.random.randint(0, max(n_frames - t, 1))
        augmented[:, t0:t0 + t] = 0

    return augmented


def extract_logmel(y, sr=16000, n_mels=40, n_fft=512, hop_length=160, n_frames=300):
    """提取 LogMel 特征并调整到固定帧数

    Args:
        y: 音频信号
        sr: 采样率
        n_mels: 梅尔频带数
        n_fft: FFT 窗口大小
        hop_length: 帧移
        n_frames: 目标时间帧数

    Returns:
        LogMel 特征矩阵 (n_mels, n_frames)，float32
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    logmel = librosa.power_to_db(mel, ref=np.max)

    if logmel.shape[1] < n_frames:
        logmel = np.pad(logmel, ((0, 0), (0, n_frames - logmel.shape[1])), 'constant')
    else:
        logmel = logmel[:, :n_frames]

    return logmel.astype(np.float32)
