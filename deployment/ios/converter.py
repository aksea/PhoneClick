"""
iOS模型转换器

提供PyTorch模型到iOS平台的转换功能。
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import os


class ModelConverter:
    """
    iOS模型转换器基类
    
    提供PyTorch模型到iOS平台的转换功能。
    """
    
    def __init__(self, model: nn.Module, config: Dict = None):
        """
        初始化转换器
        
        Args:
            model: PyTorch模型
            config: 转换配置
        """
        self.model = model
        self.config = config or {}
        
    def convert_to_coreml(self, output_path: str, 
                         input_shape: Tuple[int, ...] = (1, 90, 9)) -> bool:
        """
        转换为Core ML格式
        
        Args:
            output_path: 输出文件路径
            input_shape: 输入形状
            
        Returns:
            bool: 是否转换成功
        """
        try:
            # 设置模型为评估模式
            self.model.eval()
            
            # 创建示例输入
            example_input = torch.randn(input_shape)
            
            # 使用torch.jit.trace创建脚本模型
            traced_model = torch.jit.trace(self.model, example_input)
            
            # 转换为Core ML格式
            import coremltools as ct
            
            # 定义输入格式
            input_format = ct.Shape(shape=input_shape)
            
            # 转换模型
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=input_shape, name="input")],
                minimum_deployment_target=ct.target.iOS15
            )
            
            # 保存模型
            coreml_model.save(output_path)
            
            print(f"Core ML模型已保存到: {output_path}")
            return True
            
        except Exception as e:
            print(f"Core ML转换失败: {e}")
            return False
    
    def convert_to_onnx(self, output_path: str, 
                       input_shape: Tuple[int, ...] = (1, 90, 9)) -> bool:
        """
        转换为ONNX格式
        
        Args:
            output_path: 输出文件路径
            input_shape: 输入形状
            
        Returns:
            bool: 是否转换成功
        """
        try:
            # 设置模型为评估模式
            self.model.eval()
            
            # 创建示例输入
            example_input = torch.randn(input_shape)
            
            # 导出ONNX模型
            torch.onnx.export(
                self.model,
                example_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            print(f"ONNX模型已保存到: {output_path}")
            return True
            
        except Exception as e:
            print(f"ONNX转换失败: {e}")
            return False
    
    def optimize_model(self, model_path: str, optimization_level: str = "default") -> bool:
        """
        优化模型
        
        Args:
            model_path: 模型文件路径
            optimization_level: 优化级别
            
        Returns:
            bool: 是否优化成功
        """
        try:
            if model_path.endswith('.mlmodel'):
                # Core ML模型优化
                import coremltools as ct
                
                model = ct.models.MLModel(model_path)
                optimized_model = ct.compression_utils.affine_quantize_weights(
                    model, mode="linear"
                )
                
                # 保存优化后的模型
                optimized_path = model_path.replace('.mlmodel', '_optimized.mlmodel')
                optimized_model.save(optimized_path)
                
                print(f"优化后的Core ML模型已保存到: {optimized_path}")
                return True
                
            elif model_path.endswith('.onnx'):
                # ONNX模型优化
                import onnx
                from onnxruntime.tools import optimize_model
                
                # 加载ONNX模型
                onnx_model = onnx.load(model_path)
                
                # 优化模型
                optimized_model = optimize_model(onnx_model)
                
                # 保存优化后的模型
                optimized_path = model_path.replace('.onnx', '_optimized.onnx')
                onnx.save(optimized_model, optimized_path)
                
                print(f"优化后的ONNX模型已保存到: {optimized_path}")
                return True
                
        except Exception as e:
            print(f"模型优化失败: {e}")
            return False
    
    def validate_model(self, model_path: str, test_data: torch.Tensor) -> bool:
        """
        验证转换后的模型
        
        Args:
            model_path: 模型文件路径
            test_data: 测试数据
            
        Returns:
            bool: 验证是否通过
        """
        try:
            if model_path.endswith('.mlmodel'):
                # 验证Core ML模型
                import coremltools as ct
                
                model = ct.models.MLModel(model_path)
                
                # 准备输入数据
                input_data = test_data.numpy()
                
                # 运行推理
                predictions = model.predict({"input": input_data})
                
                print("Core ML模型验证通过")
                return True
                
            elif model_path.endswith('.onnx'):
                # 验证ONNX模型
                import onnxruntime as ort
                
                session = ort.InferenceSession(model_path)
                
                # 准备输入数据
                input_data = test_data.numpy()
                
                # 运行推理
                predictions = session.run(None, {"input": input_data})
                
                print("ONNX模型验证通过")
                return True
                
        except Exception as e:
            print(f"模型验证失败: {e}")
            return False 