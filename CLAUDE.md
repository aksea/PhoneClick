# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

PhoneClick 是一个基于手机 IMU 数据的敲击识别项目，使用 tsai 库提供强大的时间序列 AI 模型支持。项目支持数据采集、预处理、模型训练和 iOS 平台部署。

## 常用开发命令

### 环境设置
```bash
# 安装项目依赖
pip install -r requirements.txt

# 运行基本示例
python examples/basic_usage.py
```

### 测试命令
```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_preprocessor.py
```

### 代码质量检查
```bash
# 代码格式化
black .

# 代码风格检查
flake8 .
```

## 核心架构

### 数据流水线
1. **数据采集** (`tools/data_collection/`) - 抽象 IMU 数据采集器接口
2. **数据预处理** (`tools/data_processing/`) - 清洗、滤波、标准化、窗口化
3. **模型训练** (`models/training/`) - 基于 tsai 库的时间序列模型训练
4. **模型部署** (`deployment/ios/`) - Core ML 和 ONNX 格式转换

### IMU 数据格式
- **特征维度**: 9 个特征 (ax, ay, az, gx, gy, gz, roll, pitch, yaw)
- **默认窗口大小**: 90 个样本点
- **默认采样率**: 100Hz
- **支持的标签**: 4 类敲击位置 (left, right, top, bottom)

### tsai 模型支持

项目集成了 tsai 库的多种时间序列分类模型：

#### 推荐模型选择
- **InceptionTime**: 适合长序列和复杂模式识别
- **ResNet**: 训练稳定，适合一般分类任务
- **FCN**: 全卷积网络，适合快速原型开发
- **LSTM**: 适合有长期依赖的时间序列
- **TST**: 时间序列 Transformer，适合复杂模式
- **MiniRocket**: 轻量级模型，极快训练，适合快速部署

#### 模型创建方式
```python
from models.training import TSModels, create_model

# 使用 tsai 模型
model = TSModels.create_model('inceptiontime', {
    'input_features': 9,
    'num_classes': 4,
    'seq_len': 90
})

# 通用创建函数
model = create_model('tsai', {
    'input_features': 9,
    'num_classes': 4,
    'seq_len': 90,
    'model_name': 'inceptiontime'
})
```

## 配置管理

所有项目参数通过 `configs/config.yaml` 集中管理：

- **data**: 数据配置（采样率、窗口大小、重叠率等）
- **model**: 模型配置（类型、参数、训练设置等）
- **tsai_models**: tsai 模型特定配置和推荐列表
- **training**: 训练配置（数据集分割、早停、学习率调度等）
- **deployment**: 部署配置（iOS 格式、量化设置等）

## iOS 部署

### 支持的格式
1. **Core ML** (推荐): 原生 iOS 支持，性能优化好
2. **ONNX Runtime**: 跨平台支持，标准化格式
3. **PyTorch Mobile**: 直接使用 PyTorch 模型

### 模型转换
```python
from deployment.ios import ModelConverter

converter = ModelConverter(model)

# 转换为 Core ML
converter.convert_to_coreml("model.mlmodel")

# 转换为 ONNX
converter.convert_to_onnx("model.onnx")

# 模型优化
converter.optimize_model("model.mlmodel", optimization_level="high")
```

## 开发模式和约定

### 模块化设计
- 每个功能模块都有清晰的接口定义
- 使用抽象基类定义采集器和处理器接口
- 配置驱动的设计模式，避免硬编码参数

### 数据处理流水线
```python
from tools.data_processing import IMUPreprocessor

preprocessor = IMUPreprocessor(config['data'])
processed_data = preprocessor.process_pipeline(
    raw_data,
    filter_config={'filter_type': 'lowpass', 'cutoff_freq': 10.0},
    normalize_config={'method': 'standard'}
)
```

### 自定义模型支持
除了 tsai 模型，项目还提供专门为 IMU 数据优化的自定义模型：
- **IMU CNN**: 专门为 IMU 数据优化的卷积神经网络
- **IMU LSTM**: 专门为 IMU 数据优化的 LSTM 网络

## 重要文件和目录

- `examples/basic_usage.py`: 完整的使用示例，展示所有主要功能
- `configs/config.yaml`: 集中的配置文件，包含所有参数设置
- `docs/deployment_guide.md`: 详细的部署指南文档
- `tools/data_processing/preprocessor.py`: 核心数据预处理实现
- `deployment/ios/converter.py`: iOS 平台模型转换实现

## 最佳实践

### 模型选择建议
- **快速原型**: 使用 FCN 或 MiniRocket
- **一般任务**: 使用 ResNet 或 LSTM  
- **复杂模式**: 使用 InceptionTime 或 TST
- **资源受限**: 使用 MiniRocket

### 性能优化
- 使用模型量化减少文件大小
- 应用适当的数据预处理流水线
- 根据实际需求选择合适的窗口大小和重叠率

### 部署注意事项
- iOS 平台优先选择 Core ML 格式
- 在真实设备上进行性能测试
- 考虑不同设备型号的兼容性