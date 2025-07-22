# PhoneClick - 手机敲击识别项目

基于手机 IMU 数据的敲击识别模型训练和部署项目，使用 tsai 库提供强大的时间序列 AI 模型支持。

## 项目结构

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
│   └── ios/                       # iOS部署代码
├── configs/                       # 配置文件
├── tests/                         # 测试代码
├── docs/                          # 文档
└── requirements.txt               # 依赖包
```

## 功能特性

- **数据采集**: 支持多平台 IMU 数据采集
- **数据标注**: 可视化标注工具，支持手动标注敲击事件
- **模型训练**: 基于 tsai 库的深度学习模型训练，支持多种时间序列模型
- **模型部署**: 支持 iOS 平台部署（Core ML 和 ONNX 格式）
- **实时识别**: 低延迟的实时敲击识别

## tsai 库模型支持

本项目集成了 tsai 库，提供多种专门用于时间序列分类的模型：

### 推荐模型

- **InceptionTime**: 基于 Inception 架构，适合长序列时间序列分类
- **ResNet**: 残差网络，训练稳定，适合一般任务
- **FCN**: 全卷积网络，简单高效，适合快速原型
- **LSTM**: 长短期记忆网络，适合有长期依赖的时间序列
- **TST**: 时间序列 Transformer，适合复杂模式识别
- **MiniRocket**: 轻量级 Rocket 模型，极快训练，适合快速部署

### 自定义模型

- **IMU CNN**: 专门为 IMU 数据优化的卷积神经网络
- **IMU LSTM**: 专门为 IMU 数据优化的 LSTM 网络

## 安装和使用

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行示例

```bash
python examples/basic_usage.py
```

### 3. 使用 tsai 模型

```python
from models.training import TSModels

# 创建InceptionTime模型
model = TSModels.create_model('inceptiontime', {
    'input_features': 9,
    'num_classes': 4,
    'seq_len': 90
})

# 查看可用模型
available_models = TSModels.get_available_models()
print(available_models)
```

## 配置说明

项目使用 YAML 配置文件管理参数：

- **数据配置**: 采样率、窗口大小、重叠率等
- **模型配置**: 模型类型、tsai 模型名称、序列长度等
- **训练配置**: 批次大小、训练轮数、学习率调度等
- **部署配置**: iOS 模型格式、量化设置等

## 部署指南

详细的部署说明请参考 `docs/deployment_guide.md`。

### iOS 部署选项

1. **Core ML** (推荐): 原生 iOS 支持，性能优化好
2. **ONNX Runtime**: 跨平台支持，模型格式标准化
3. **PyTorch Mobile**: 直接使用 PyTorch 模型

## 开发指南

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

## 常见问题

### Q: 为什么选择 tsai 库？

A: tsai 库专门为时间序列 AI 设计，提供了多种经过验证的时间序列分类模型，特别适合 IMU 数据这类时序数据。

### Q: 推荐使用哪个模型？

A:

- **快速原型**: FCN 或 MiniRocket
- **一般任务**: ResNet 或 LSTM
- **复杂模式**: InceptionTime 或 TST
- **资源受限**: MiniRocket

### Q: 如何优化模型性能？

A:

- 使用模型量化减少文件大小
- 应用模型剪枝减少计算量
- 使用更高效的模型架构

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。
