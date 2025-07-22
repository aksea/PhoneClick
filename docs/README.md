# PhoneClick 项目文档

## 项目概述

PhoneClick 是一个基于手机 IMU 数据的敲击识别项目，旨在训练和部署能够识别手机各侧敲击的深度学习模型。

## 项目架构

### 目录结构

```
PhoneClick/
├── data/                          # 数据目录
│   ├── raw/                       # 原始IMU数据
│   │   ├── left/                  # 左侧敲击数据
│   │   ├── right/                 # 右侧敲击数据
│   │   ├── top/                   # 顶部敲击数据
│   │   └── bottom/                # 底部敲击数据
│   ├── processed/                 # 预处理后的数据
│   └── labeled/                   # 标注后的数据
├── tools/                         # 工具类
│   ├── data_collection/           # 数据采集工具
│   ├── data_processing/           # 数据处理工具
│   └── data_labeling/             # 数据标注工具
├── models/                        # 模型相关
│   ├── training/                  # 模型训练代码
│   ├── evaluation/                # 模型评估代码
│   └── saved_models/              # 保存的模型文件
├── deployment/                    # 部署相关
│   ├── ios/                       # iOS部署代码
│   └── android/                   # Android部署代码
├── configs/                       # 配置文件
├── tests/                         # 测试代码
├── docs/                          # 文档
└── requirements.txt               # 依赖包
```

## 核心模块说明

### 1. 数据采集模块 (`tools/data_collection/`)

提供多平台 IMU 数据采集功能：

- **IMUDataCollector**: 基础数据采集器接口
- **MobileIMUCollector**: 移动端数据采集器
- **DesktopIMUCollector**: 桌面端数据采集器

### 2. 数据处理模块 (`tools/data_processing/`)

提供数据预处理和特征提取功能：

- **IMUPreprocessor**: 数据清洗、滤波、标准化
- **FeatureExtractor**: 特征提取
- **DataAugmentation**: 数据增强
- **IMUDataset**: 数据集类

### 3. 数据标注模块 (`tools/data_labeling/`)

提供可视化数据标注功能：

- **IMULabelTool**: 基于 PyQt5 的可视化标注工具

### 4. 模型训练模块 (`models/training/`)

提供深度学习模型训练功能：

- **CNNModel**: 1D 卷积神经网络
- **LSTMModel**: LSTM 网络
- **TransformerModel**: Transformer 网络
- **ModelTrainer**: 模型训练器

### 5. 模型评估模块 (`models/evaluation/`)

提供模型性能评估功能：

- **ModelEvaluator**: 模型评估器
- **ClassificationMetrics**: 分类指标计算
- **ResultVisualizer**: 结果可视化

### 6. 部署模块 (`deployment/`)

提供模型部署功能：

- **iOS 部署**: Core ML 和 ONNX 格式转换
- **Android 部署**: TensorFlow Lite 格式转换

## 使用流程

### 1. 数据采集

```python
from tools.data_collection import IMUDataCollector

# 创建数据采集器
collector = IMUDataCollector(sample_rate=100)

# 采集数据
data = collector.collect_data(duration=60)  # 采集60秒数据

# 保存数据
collector.save_data(data, "data/raw/right/imu_data.csv")
```

### 2. 数据预处理

```python
from tools.data_processing import IMUPreprocessor

# 创建预处理器
preprocessor = IMUPreprocessor()

# 处理数据
processed_data = preprocessor.process_pipeline(
    data,
    filter_config={'filter_type': 'lowpass', 'cutoff_freq': 10.0},
    normalize_config={'method': 'standard'}
)
```

### 3. 数据标注

```python
from tools.data_labeling import IMULabelTool

# 启动标注工具
tool = IMULabelTool()
tool.show()
```

### 4. 模型训练

```python
from models.training import CNNModel, ModelTrainer

# 创建模型
model = CNNModel(input_features=9, num_classes=4)

# 训练模型
trainer = ModelTrainer(model)
trainer.train(train_loader, val_loader, epochs=100)
```

### 5. 模型部署

```python
from deployment.ios import ModelConverter

# 转换模型
converter = ModelConverter(model)
converter.convert_to_coreml("model.mlmodel")
```

## 配置说明

项目使用 YAML 配置文件管理参数：

- **数据配置**: 采样率、窗口大小、重叠率等
- **模型配置**: 模型类型、隐藏层大小、学习率等
- **训练配置**: 批次大小、训练轮数、早停等
- **部署配置**: 模型格式、量化设置等

## 开发指南

### 环境设置

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 配置环境变量（如需要）

### 代码规范

- 使用 Google 风格的 docstring
- 遵循 PEP 8 代码风格
- 添加类型注解
- 编写单元测试

### 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_preprocessor.py
```

## 部署指南

详细的部署说明请参考 `docs/deployment_guide.md`。

## 常见问题

### Q: 可以使用 PyTorch 训练模型吗？

A: 是的！PyTorch 提供了多种方式将模型部署到移动端：

- iOS: Core ML、ONNX Runtime、PyTorch Mobile
- Android: TensorFlow Lite、ONNX Runtime

### Q: 如何优化模型性能？

A:

- 使用模型量化减少文件大小
- 应用模型剪枝减少计算量
- 使用更高效的模型架构

### Q: 如何处理实时推理？

A:

- 优化数据预处理流程
- 使用适当的批处理大小
- 考虑 GPU 加速（如果可用）

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。
