# 模型部署指南

## 概述

本文档介绍如何将训练好的 PyTorch 模型部署到 iOS 和 Android 平台。

## iOS 部署

### 1. 使用 PyTorch 训练模型

是的，你可以使用 PyTorch 训练模型！PyTorch 提供了多种方式将模型部署到 iOS：

### 2. 部署选项

#### 选项 1: Core ML (推荐)

- **优点**: 原生 iOS 支持，性能优化好，支持神经网络加速
- **缺点**: 仅支持 iOS 平台
- **工具**: `coremltools`

#### 选项 2: ONNX Runtime

- **优点**: 跨平台支持，模型格式标准化
- **缺点**: 需要额外的运行时库
- **工具**: `onnx`, `onnxruntime`

#### 选项 3: PyTorch Mobile

- **优点**: 直接使用 PyTorch 模型，无需转换
- **缺点**: 模型文件较大，性能可能不如优化后的格式
- **工具**: `torch.jit`

### 3. 部署步骤

#### Core ML 部署流程

1. **安装依赖**

```bash
pip install coremltools
```

2. **转换模型**

```python
from deployment.ios.converter import ModelConverter

# 加载训练好的模型
model = load_trained_model()
converter = ModelConverter(model)

# 转换为Core ML格式
converter.convert_to_coreml("model.mlmodel")
```

3. **在 iOS 项目中使用**

```swift
import CoreML

// 加载模型
let model = try MyModel()

// 准备输入数据
let input = try MLMultiArray(shape: [1, 90, 9], dataType: .float32)
// ... 填充数据

// 运行推理
let prediction = try model.prediction(input: input)
```

#### ONNX 部署流程

1. **转换模型**

```python
converter.convert_to_onnx("model.onnx")
```

2. **在 iOS 项目中使用**

```swift
import ONNXRuntime

// 创建推理会话
let session = try ORTSession(path: "model.onnx")

// 准备输入
let input = try ORTValue(tensorData: data, elementType: .float32, shape: [1, 90, 9])

// 运行推理
let outputs = try session.run(withInputs: ["input": input])
```

### 4. 性能优化

#### 模型量化

```python
# 量化模型以减少文件大小和提高推理速度
converter.optimize_model("model.mlmodel", optimization_level="high")
```

#### 批处理优化

- 使用适当的批处理大小
- 考虑实时推理的延迟要求

## Android 部署

### 1. 部署选项

#### 选项 1: TensorFlow Lite (推荐)

- **优点**: 谷歌官方支持，性能优化好
- **工具**: `tflite`

#### 选项 2: ONNX Runtime

- **优点**: 跨平台一致性
- **工具**: `onnxruntime-android`

### 2. 部署步骤

#### TensorFlow Lite 部署

1. **转换模型**

```python
import tensorflow as tf

# 将PyTorch模型转换为TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
tflite_model = converter.convert()

# 保存模型
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```

2. **在 Android 项目中使用**

```kotlin
import org.tensorflow.lite.Interpreter

// 加载模型
val interpreter = Interpreter(loadModelFile())

// 准备输入数据
val inputArray = Array(1) { Array(90) { FloatArray(9) } }
// ... 填充数据

// 运行推理
val outputArray = Array(1) { FloatArray(4) }
interpreter.run(inputArray, outputArray)
```

## 实时推理优化

### 1. 数据预处理优化

- 在移动端实现高效的数据预处理
- 使用 SIMD 指令优化计算

### 2. 模型推理优化

- 使用适当的线程数
- 考虑 GPU 加速（如果可用）

### 3. 内存管理

- 重用输入/输出缓冲区
- 避免频繁的内存分配

## 测试和验证

### 1. 模型验证

```python
# 验证转换后的模型
test_data = torch.randn(1, 90, 9)
converter.validate_model("model.mlmodel", test_data)
```

### 2. 性能测试

- 测量推理延迟
- 测试内存使用情况
- 验证准确性

### 3. 集成测试

- 在真实设备上测试
- 测试不同设备型号的兼容性

## 常见问题

### 1. 模型文件过大

- 使用模型量化
- 考虑模型剪枝
- 使用更小的模型架构

### 2. 推理延迟过高

- 优化数据预处理
- 使用更快的模型架构
- 考虑模型蒸馏

### 3. 内存使用过多

- 减少批处理大小
- 优化内存管理
- 使用内存映射文件

## 最佳实践

1. **选择合适的部署格式**

   - iOS: 优先考虑 Core ML
   - Android: 优先考虑 TensorFlow Lite

2. **模型优化**

   - 训练时考虑部署约束
   - 使用模型压缩技术

3. **测试策略**

   - 在多种设备上测试
   - 进行压力测试

4. **版本管理**
   - 保持模型版本的一致性
   - 记录模型转换参数
