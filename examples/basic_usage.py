"""
PhoneClick 项目基本使用示例

展示如何使用项目的各个模块进行数据采集、处理和模型训练。
"""

import os
import sys
import yaml
import pandas as pd
import torch
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tools.data_processing import IMUPreprocessor
from models.training import TSModels, CustomModels, create_model
from deployment.ios import ModelConverter


def load_config():
    """加载配置文件"""
    config_path = project_root / "configs" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def process_data_example():
    """数据处理示例"""
    print("=== 数据处理示例 ===")
    
    # 加载配置
    config = load_config()
    
    # 创建预处理器
    preprocessor = IMUPreprocessor(config['data'])
    
    # 假设我们有一些原始数据
    # 这里使用示例数据，实际使用时应该从文件加载
    sample_data = pd.DataFrame({
        'ax': [0.1, 0.2, 0.3] * 30,
        'ay': [0.2, 0.3, 0.4] * 30,
        'az': [0.3, 0.4, 0.5] * 30,
        'gx': [0.01, 0.02, 0.03] * 30,
        'gy': [0.02, 0.03, 0.04] * 30,
        'gz': [0.03, 0.04, 0.05] * 30,
        'roll': [0.1, 0.2, 0.3] * 30,
        'pitch': [0.2, 0.3, 0.4] * 30,
        'yaw': [0.3, 0.4, 0.5] * 30,
        'label': [0, 1, 0] * 30
    })
    
    print(f"原始数据形状: {sample_data.shape}")
    
    # 处理数据
    processed_data = preprocessor.process_pipeline(
        sample_data,
        filter_config={'filter_type': 'lowpass', 'cutoff_freq': 10.0},
        normalize_config={'method': 'standard'}
    )
    
    print(f"处理后数据形状: {processed_data.shape}")
    print("数据处理完成！\n")


def tsai_models_example():
    """tsai模型示例"""
    print("=== tsai模型示例 ===")
    
    # 加载配置
    config = load_config()
    
    # 获取可用模型列表
    available_models = TSModels.get_available_models()
    print(f"可用的tsai模型: {available_models}")
    
    # 显示推荐模型
    recommended_models = config['tsai_models']['recommended']
    print(f"推荐的模型: {recommended_models}")
    
    # 创建不同的tsai模型
    model_config = {
        'input_features': config['model']['input_features'],
        'num_classes': config['model']['num_classes'],
        'seq_len': config['model']['seq_len'],
        'hidden_size': config['model']['hidden_size']
    }
    
    # 测试几个推荐的模型
    for model_name in recommended_models[:3]:  # 只测试前3个
        try:
            print(f"\n创建 {model_name} 模型...")
            model = TSModels.create_model(model_name, model_config)
            
            # 获取模型信息
            model_info = TSModels.get_model_info(model_name)
            print(f"描述: {model_info['description']}")
            print(f"适用场景: {model_info['best_for']}")
            
            # 计算模型参数数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"总参数数量: {total_params:,}")
            print(f"可训练参数数量: {trainable_params:,}")
            
            # 测试前向传播
            test_input = torch.randn(1, model_config['seq_len'], model_config['input_features'])
            with torch.no_grad():
                output = model(test_input)
            print(f"输出形状: {output.shape}")
            
        except Exception as e:
            print(f"创建 {model_name} 模型失败: {e}")
    
    print("tsai模型示例完成！\n")


def custom_models_example():
    """自定义模型示例"""
    print("=== 自定义模型示例 ===")
    
    # 加载配置
    config = load_config()
    
    model_config = {
        'input_features': config['model']['input_features'],
        'num_classes': config['model']['num_classes'],
        'seq_len': config['model']['seq_len'],
        'hidden_size': config['model']['hidden_size']
    }
    
    # 创建IMU CNN模型
    print("创建IMU CNN模型...")
    imu_cnn = CustomModels.create_imu_cnn(**model_config)
    
    total_params = sum(p.numel() for p in imu_cnn.parameters())
    print(f"IMU CNN参数数量: {total_params:,}")
    
    # 创建IMU LSTM模型
    print("创建IMU LSTM模型...")
    imu_lstm = CustomModels.create_imu_lstm(**model_config)
    
    total_params = sum(p.numel() for p in imu_lstm.parameters())
    print(f"IMU LSTM参数数量: {total_params:,}")
    
    # 测试前向传播
    test_input = torch.randn(1, model_config['seq_len'], model_config['input_features'])
    
    with torch.no_grad():
        cnn_output = imu_cnn(test_input)
        lstm_output = imu_lstm(test_input)
    
    print(f"CNN输出形状: {cnn_output.shape}")
    print(f"LSTM输出形状: {lstm_output.shape}")
    
    print("自定义模型示例完成！\n")


def model_creation_example():
    """模型创建示例"""
    print("=== 模型创建示例 ===")
    
    # 加载配置
    config = load_config()
    
    # 使用通用创建函数
    model_config = {
        'input_features': config['model']['input_features'],
        'num_classes': config['model']['num_classes'],
        'seq_len': config['model']['seq_len'],
        'hidden_size': config['model']['hidden_size']
    }
    
    # 创建tsai模型
    print("创建tsai模型...")
    tsai_model = create_model('tsai', {
        **model_config,
        'model_name': 'inceptiontime'
    })
    
    # 创建自定义模型
    print("创建自定义模型...")
    custom_model = create_model('custom', {
        **model_config,
        'model_name': 'imu_cnn'
    })
    
    # 计算参数数量
    tsai_params = sum(p.numel() for p in tsai_model.parameters())
    custom_params = sum(p.numel() for p in custom_model.parameters())
    
    print(f"tsai模型参数数量: {tsai_params:,}")
    print(f"自定义模型参数数量: {custom_params:,}")
    
    print("模型创建示例完成！\n")


def deployment_example():
    """模型部署示例"""
    print("=== 模型部署示例 ===")
    
    # 创建示例模型
    model = create_model('tsai', {
        'input_features': 9,
        'num_classes': 4,
        'seq_len': 90,
        'hidden_size': 128,
        'model_name': 'inceptiontime'
    })
    model.eval()
    
    # 创建转换器
    converter = ModelConverter(model)
    
    # 转换为Core ML格式
    coreml_path = project_root / "models" / "saved_models" / "tsai_model.mlmodel"
    coreml_path.parent.mkdir(parents=True, exist_ok=True)
    
    success = converter.convert_to_coreml(str(coreml_path))
    if success:
        print(f"Core ML模型已保存到: {coreml_path}")
    
    # 转换为ONNX格式
    onnx_path = project_root / "models" / "saved_models" / "tsai_model.onnx"
    success = converter.convert_to_onnx(str(onnx_path))
    if success:
        print(f"ONNX模型已保存到: {onnx_path}")
    
    print("模型部署示例完成！\n")


def main():
    """主函数"""
    print("PhoneClick 项目基本使用示例")
    print("=" * 50)
    
    try:
        # 数据处理示例
        process_data_example()
        
        # tsai模型示例
        tsai_models_example()
        
        # 自定义模型示例
        custom_models_example()
        
        # 模型创建示例
        model_creation_example()
        
        # 模型部署示例
        deployment_example()
        
        print("所有示例执行完成！")
        print("\n推荐使用tsai库的以下模型进行IMU敲击识别:")
        print("- InceptionTime: 适合长序列，多尺度特征提取")
        print("- ResNet: 训练稳定，适合一般任务")
        print("- MiniRocket: 极快训练，适合快速原型")
        print("- TST: 复杂模式识别，适合精细分类")
        
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        print("请确保已安装所有依赖包: pip install -r requirements.txt")
        print("特别是tsai库: pip install tsai")


if __name__ == "__main__":
    main() 