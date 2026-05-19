# ArcFace-CNN-classifying_words-model_ONNX

基于 TinyCNN 的轻量级语音命令分类模型。接收 3 秒 16kHz 音频，通过 2 层 CNN 提取 LogMel 特征并生成 32 维嵌入向量，可用于语音指令识别，支持导出 ONNX 部署到树莓派等边缘设备。

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/3336-zheng/ArcFace-CNN-classifying_words-model_ONNX.git
cd ArcFace-CNN-classifying_words-model_ONNX
```

### 2. 创建环境并安装依赖

```bash
conda create -n arcface python=3.10 -y
conda activate arcface
pip install -r requirements.txt
```

> 如果已有可用的 conda 环境（如 `whisper`），直接激活后 `pip install -r requirements.txt` 即可。

### 3. 准备数据

编辑 `configs/default.yaml`，填入你的数据路径：

```yaml
data:
  jsonl_train: "/path/to/your/train.jsonl"      # 训练集，每行一个 JSON
  command_mapping: "/path/to/command_mapping.json" # 指令→ID 映射
  features_npy: "/path/to/features.npy"           # 提取后的特征文件
  labels_npy: "/path/to/labels.npy"               # 对应标签
model:
  checkpoint: "/path/to/tiny_cnn_32dim.pth"       # 训练好的模型权重
```

JSONL 格式示例（每行一条）：

```json
{"audio": {"path": "/path/to/audio.wav"}, "sentence": "开灯", "duration": 2.5}
```

### 4. 运行

```bash
# 提取 16 维 MFCC 手工特征
python -m src.features.mfcc_extractor

# 提取 32 维手工特征
python -m src.features.handcraft_extractor

# 导出 ONNX 模型
python -m src.inference.onnx_exporter --ckpt /path/to/model.pth --output tiny_cnn_rpi.onnx
```

## 项目结构

```
├── configs/
│   └── default.yaml               # 路径和超参数配置
├── src/
│   ├── models/
│   │   └── tiny_cnn.py            # TinyCNN 网络定义
│   ├── features/
│   │   ├── preprocessor.py        # JSONL 数据加载与指令映射
│   │   ├── mfcc_extractor.py      # MFCC 16 维特征提取
│   │   ├── handcraft_extractor.py # 手工 32 维特征提取（MFCC + 频域 + 基频）
│   │   └── logmel_extractor.py    # LogMel + TinyCNN 嵌入提取
│   ├── inference/
│   │   ├── onnx_exporter.py       # PyTorch → ONNX 导出
│   │   └── onnx_inference.py      # ONNX 推理 + 1-NN 分类
│   └── utils/
│       └── audio.py               # 音频加载、裁剪、LogMel 提取
├── scripts/
│   ├── verify.py                  # 逻辑回归验证（线性可分性）
│   ├── quick_1nn_test.py          # 1-NN + RandomForest 快速测试
│   └── view_features.py           # 查看特征统计信息
├── configs/
│   └── default.yaml
├── requirements.txt
└── README.md
```

## 模型架构

```
输入: LogMel (40 × 300)
  │
  ├─ Conv1d(40→64, k=5, s=2) + ReLU
  ├─ Conv1d(64→128, k=5, s=2) + ReLU
  ├─ AdaptiveAvgPool1d(1)        → 128 维
  └─ Linear(128→32)              → 32 维嵌入向量
```

- **输入**: 3 秒 16kHz 音频 → 40 维 LogMel × 300 帧
- **输出**: 32 维嵌入向量
- **训练**: 使用 ArcFace 分类头（导出时仅保留嵌入部分）

## ONNX 推理示例

```python
from src.inference.onnx_inference import TinyCNNInferencePC, create_classifier_from_features

# 加载模型
model = TinyCNNInferencePC("tiny_cnn_rpi.onnx", "command_mapping.json")

# 用已有特征训练 1-NN 分类器
classifier, scaler = create_classifier_from_features("features.npy", "labels.npy")

# 单条预测
command, confidence = model.predict_single("test.wav", classifier, scaler)
print(f"识别结果: {command}, 置信度: {confidence:.2%}")

# 批量预测
results = model.batch_predict("test_audios/", classifier, scaler)
```

## 评估脚本

```bash
# 逻辑回归 — 检验特征线性可分性
python scripts/verify.py --features features.npy --labels labels.npy

# 1-NN + RandomForest — 嵌入空间质量快速评估
python scripts/quick_1nn_test.py --features features.npy --labels labels.npy --cv 5

# 查看特征统计
python scripts/view_features.py --features features.npy --labels labels.npy
```

## 注意事项

- 评估使用 5 折交叉验证，无独立测试集
- ONNX 导出仅包含嵌入部分，不含 ArcFace 分类头
- 音频会被裁剪/填充到固定 3 秒，超出部分取中间段
