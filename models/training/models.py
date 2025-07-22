"""
基于tsai库的时间序列深度学习模型

提供多种专门用于时间序列分类的模型架构。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from tsai.models.all import *
from tsai.basics import *


class TSModels:
    """
    tsai库时间序列模型封装
    
    提供多种专门用于时间序列分类的模型。
    """
    
    @staticmethod
    def create_model(model_name: str, config: Dict) -> nn.Module:
        """
        创建tsai模型
        
        Args:
            model_name: 模型名称
            config: 模型配置
            
        Returns:
            nn.Module: 模型实例
        """
        # 提取配置参数
        input_features = config.get('input_features', 9)
        num_classes = config.get('num_classes', 4)
        seq_len = config.get('seq_len', 90)
        hidden_size = config.get('hidden_size', 128)
        
        # 创建模型
        if model_name == 'inceptiontime':
            return InceptionTime(c_in=input_features, c_out=num_classes, seq_len=seq_len)
        elif model_name == 'resnet':
            return ResNet(c_in=input_features, c_out=num_classes, seq_len=seq_len)
        elif model_name == 'fcn':
            return FCN(c_in=input_features, c_out=num_classes, seq_len=seq_len)
        elif model_name == 'lstm':
            return LSTM(c_in=input_features, c_out=num_classes, seq_len=seq_len)
        elif model_name == 'lstm_plus':
            return LSTMPlus(c_in=input_features, c_out=num_classes, seq_len=seq_len)
        elif model_name == 'gru':
            return GRU(c_in=input_features, c_out=num_classes, seq_len=seq_len)
        elif model_name == 'gru_plus':
            return GRUPlus(c_in=input_features, c_out=num_classes, seq_len=seq_len)
        elif model_name == 'transformer':
            return TransformerModel(c_in=input_features, c_out=num_classes, seq_len=seq_len)
        elif model_name == 'tst':
            return TST(c_in=input_features, c_out=num_classes, seq_len=seq_len)
        elif model_name == 'xcm':
            return XCM(c_in=input_features, c_out=num_classes, seq_len=seq_len)
        elif model_name == 'xresnet1d':
            return XResNet1d(c_in=input_features, c_out=num_classes, seq_len=seq_len)
        elif model_name == 'xresnet1d_plus':
            return XResNet1dPlus(c_in=input_features, c_out=num_classes, seq_len=seq_len)
        elif model_name == 'omniscale':
            return OmniScale(c_in=input_features, c_out=num_classes, seq_len=seq_len)
        elif model_name == 'tcn':
            return TCN(c_in=input_features, c_out=num_classes, seq_len=seq_len)
        elif model_name == 'tst_plus':
            return TSTPlus(c_in=input_features, c_out=num_classes, seq_len=seq_len)
        elif model_name == 'mini_rocket':
            return MiniRocket(c_in=input_features, c_out=num_classes, seq_len=seq_len)
        elif model_name == 'rocket':
            return Rocket(c_in=input_features, c_out=num_classes, seq_len=seq_len)
        else:
            raise ValueError(f"不支持的tsai模型: {model_name}")
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        获取可用的模型列表
        
        Returns:
            List[str]: 可用模型名称列表
        """
        return [
            'inceptiontime', 'resnet', 'fcn', 'lstm', 'lstm_plus',
            'gru', 'gru_plus', 'transformer', 'tst', 'xcm',
            'xresnet1d', 'xresnet1d_plus', 'omniscale', 'tcn',
            'tst_plus', 'mini_rocket', 'rocket'
        ]
    
    @staticmethod
    def get_model_info(model_name: str) -> Dict:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            Dict: 模型信息
        """
        model_info = {
            'inceptiontime': {
                'description': 'InceptionTime - 基于Inception架构的时间序列分类模型',
                'paper': 'InceptionTime: Finding AlexNet for Time Series Classification',
                'strengths': ['多尺度特征提取', '残差连接', '适合长序列'],
                'best_for': '长序列时间序列分类'
            },
            'resnet': {
                'description': 'ResNet - 残差网络用于时间序列分类',
                'paper': 'Deep Residual Learning for Image Recognition',
                'strengths': ['残差连接', '梯度流动', '训练稳定'],
                'best_for': '一般时间序列分类任务'
            },
            'fcn': {
                'description': 'FCN - 全卷积网络用于时间序列分类',
                'paper': 'Time Series Classification from Scratch with Deep Neural Networks',
                'strengths': ['简单高效', '参数少', '训练快'],
                'best_for': '快速原型和基线模型'
            },
            'lstm': {
                'description': 'LSTM - 长短期记忆网络',
                'paper': 'Long Short-Term Memory',
                'strengths': ['序列建模能力强', '记忆长期依赖'],
                'best_for': '有长期依赖的时间序列'
            },
            'transformer': {
                'description': 'Transformer - 基于注意力机制的模型',
                'paper': 'Attention Is All You Need',
                'strengths': ['并行计算', '全局建模', '可解释性'],
                'best_for': '需要全局建模的复杂序列'
            },
            'tst': {
                'description': 'TST - 时间序列Transformer',
                'paper': 'Time Series Transformer',
                'strengths': ['专门为时间序列设计', '位置编码', '多头注意力'],
                'best_for': '复杂时间序列模式识别'
            },
            'mini_rocket': {
                'description': 'MiniRocket - 轻量级Rocket模型',
                'paper': 'MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification',
                'strengths': ['极快训练', '确定性', '参数少'],
                'best_for': '快速部署和资源受限环境'
            }
        }
        
        return model_info.get(model_name, {
            'description': '未知模型',
            'paper': '未知',
            'strengths': [],
            'best_for': '未知'
        })


class CustomModels:
    """
    自定义模型类
    
    提供一些专门为IMU数据设计的自定义模型。
    """
    
    @staticmethod
    def create_imu_cnn(input_features: int = 9, num_classes: int = 4, 
                      seq_len: int = 90, hidden_size: int = 128) -> nn.Module:
        """
        创建专门用于IMU数据的CNN模型
        
        Args:
            input_features: 输入特征数
            num_classes: 输出类别数
            seq_len: 序列长度
            hidden_size: 隐藏层大小
            
        Returns:
            nn.Module: CNN模型
        """
        return IMUCNN(input_features, num_classes, seq_len, hidden_size)
    
    @staticmethod
    def create_imu_lstm(input_features: int = 9, num_classes: int = 4,
                       seq_len: int = 90, hidden_size: int = 128) -> nn.Module:
        """
        创建专门用于IMU数据的LSTM模型
        
        Args:
            input_features: 输入特征数
            num_classes: 输出类别数
            seq_len: 序列长度
            hidden_size: 隐藏层大小
            
        Returns:
            nn.Module: LSTM模型
        """
        return IMULSTM(input_features, num_classes, seq_len, hidden_size)


class IMUCNN(nn.Module):
    """
    专门用于IMU数据的CNN模型
    
    针对IMU数据特点优化的卷积神经网络。
    """
    
    def __init__(self, input_features: int = 9, num_classes: int = 4,
                 seq_len: int = 90, hidden_size: int = 128, dropout: float = 0.3):
        """
        初始化IMU CNN模型
        
        Args:
            input_features: 输入特征数
            num_classes: 输出类别数
            seq_len: 序列长度
            hidden_size: 隐藏层大小
            dropout: Dropout率
        """
        super(IMUCNN, self).__init__()
        
        self.input_features = input_features
        self.num_classes = num_classes
        self.seq_len = seq_len
        
        # 1D卷积层 - 针对IMU数据优化
        self.conv1 = nn.Conv1d(input_features, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        
        # 池化层
        self.pool = nn.MaxPool1d(2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * (seq_len // 8), hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_features)
            
        Returns:
            torch.Tensor: 输出张量 (batch_size, num_classes)
        """
        # 调整输入维度 (batch_size, input_features, seq_len)
        x = x.transpose(1, 2)
        
        # 卷积层
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class IMULSTM(nn.Module):
    """
    专门用于IMU数据的LSTM模型
    
    针对IMU数据特点优化的LSTM网络。
    """
    
    def __init__(self, input_features: int = 9, num_classes: int = 4,
                 seq_len: int = 90, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.3):
        """
        初始化IMU LSTM模型
        
        Args:
            input_features: 输入特征数
            num_classes: 输出类别数
            seq_len: 序列长度
            hidden_size: 隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout率
        """
        super(IMULSTM, self).__init__()
        
        self.input_features = input_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(input_features, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_features)
            
        Returns:
            torch.Tensor: 输出张量 (batch_size, num_classes)
        """
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 自注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 全局平均池化
        x = attn_out.mean(dim=1)
        
        # 全连接层
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def create_model(model_type: str, config: Dict) -> nn.Module:
    """
    创建模型实例
    
    Args:
        model_type: 模型类型 ('tsai' 或 'custom')
        config: 模型配置
        
    Returns:
        nn.Module: 模型实例
    """
    if model_type == 'tsai':
        model_name = config.get('model_name', 'inceptiontime')
        return TSModels.create_model(model_name, config)
    elif model_type == 'custom':
        model_name = config.get('model_name', 'imu_cnn')
        if model_name == 'imu_cnn':
            return CustomModels.create_imu_cnn(**config)
        elif model_name == 'imu_lstm':
            return CustomModels.create_imu_lstm(**config)
        else:
            raise ValueError(f"不支持的自定义模型: {model_name}")
    else:
        raise ValueError(f"不支持的模型类型: {model_type}") 