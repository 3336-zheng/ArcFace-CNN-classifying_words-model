# ArcFace-CNN-classifying_words-model_ONNX

基于 TinyCNN 的语音特征提取模型，采用 LogMel 特征作为输入，通过 2 层 CNN 提取特征并生成 32 维嵌入向量，支持 ONNX 导出用于边缘设备部署。

## 项目结构

```
├── configs/
│   └── default.yaml              # 集中管理路径和超参数
├── src/
│   ├── models/
│   │   └── tiny_cnn.py           # TinyCNN 模型定义
│   ├── features/
│   │   ├── preprocessor.py       # JSONL 数据预处理与指令映射
│   │   ├── mfcc_extractor.py     # MFCC 16维特征提取
│   │   ├── handcraft_extractor.py # 手工 32维特征提取
│   │   └── logmel_extractor.py   # LogMel + TinyCNN 嵌入提取
│   ├── inference/
│   │   ├── onnx_exporter.py      # ONNX 模型导出
│   │   └── onnx_inference.py     # ONNX 推理
│   └── utils/
│       └── audio.py              # 公共音频加载与预处理
├── scripts/
│   ├── verify.py                 # 逻辑回归特征验证
│   ├── quick_1nn_test.py         # 1-NN 快速测试
│   └── view_features.py          # 查看特征数据
├── requirements.txt
└── README.md
```

## 环境配置

```bash
conda activate whisper
pip install -r requirements.txt
```

## 网络架构

模型采用 **2 层 CNN + 1 层全连接** 结构:

1. **第一层卷积**: Conv1d(40→64, kernel=5, stride=2) + ReLU
2. **第二层卷积**: Conv1d(64→128, kernel=5, stride=2) + ReLU
3. **全局自适应平均池化**: 压缩时间维度至 1
4. **全连接层**: 128 → 32 维嵌入向量

## 使用方法

### 1. 配置参数

编辑 `configs/default.yaml`，设置数据路径和模型路径。

### 2. 特征提取

```bash
# MFCC 16维特征
python -m src.features.mfcc_extractor

# 手工 32维特征
python -m src.features.handcraft_extractor
```

### 3. ONNX 导出

```bash
python -m src.inference.onnx_exporter --ckpt <模型权重路径> --output tiny_cnn_rpi.onnx
```

### 4. ONNX 推理

```python
from src.inference.onnx_inference import TinyCNNInferencePC, create_classifier_from_features

# 初始化推理器
model = TinyCNNInferencePC("tiny_cnn_rpi.onnx", "command_mapping.json")

# 创建分类器
classifier, scaler = create_classifier_from_features("features.npy", "labels.npy")

# 预测
command, confidence = model.predict_single("test.wav", classifier, scaler)
```

### 5. 评估

```bash
# 逻辑回归验证
python scripts/verify.py --features features.npy --labels labels.npy

# 1-NN 快速测试
python scripts/quick_1nn_test.py --features features.npy --labels labels.npy

# 查看特征
python scripts/view_features.py --features features.npy --labels labels.npy
```

## 输入维度

- **输入特征**: LogMel (40 维梅尔频带 × 300 时间帧)
- **输出**: 32 维嵌入向量
- **采样率**: 16kHz
- **音频时长**: 3 秒

## 评估方法

采用 **5 折交叉验证 + 1-NN 分类器** 评估模型泛化能力，直接衡量嵌入空间的判别质量。

## 注意事项

- 本评估未使用独立测试集，通过训练集上的 5 折交叉验证估计模型泛化能力
- ONNX 导出仅保留嵌入部分（不含 ArcFace 分类头）
