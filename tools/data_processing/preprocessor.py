"""
IMU数据预处理器

提供IMU数据的清洗、滤波和标准化功能。
"""

import pandas as pd
import numpy as np
from scipy import signal
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class IMUPreprocessor:
    """
    IMU数据预处理器
    
    提供数据清洗、滤波、标准化等功能。
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化预处理器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.scalers = {}
        self.is_fitted = False
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据
        
        Args:
            data: 原始数据
            
        Returns:
            pd.DataFrame: 清洗后的数据
        """
        # 移除重复行
        data = data.drop_duplicates()
        
        # 处理缺失值
        data = self._handle_missing_values(data)
        
        # 移除异常值
        data = self._remove_outliers(data)
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            data: 数据DataFrame
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        # 对于IMU数据，使用线性插值填充缺失值
        imu_columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'roll', 'pitch', 'yaw']
        
        for col in imu_columns:
            if col in data.columns:
                data[col] = data[col].interpolate(method='linear')
        
        return data
    
    def _remove_outliers(self, data: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        移除异常值
        
        Args:
            data: 数据DataFrame
            threshold: 异常值阈值（标准差的倍数）
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        imu_columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'roll', 'pitch', 'yaw']
        
        for col in imu_columns:
            if col in data.columns:
                mean = data[col].mean()
                std = data[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                
                # 将异常值替换为边界值
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        
        return data
    
    def apply_filter(self, data: pd.DataFrame, filter_type: str = 'lowpass', 
                    cutoff_freq: float = 10.0, sample_rate: float = 100.0) -> pd.DataFrame:
        """
        应用滤波器
        
        Args:
            data: 数据DataFrame
            filter_type: 滤波器类型 ('lowpass', 'highpass', 'bandpass')
            cutoff_freq: 截止频率
            sample_rate: 采样率
            
        Returns:
            pd.DataFrame: 滤波后的数据
        """
        imu_columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
        
        # 设计滤波器
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        if filter_type == 'lowpass':
            b, a = signal.butter(4, normalized_cutoff, btype='low')
        elif filter_type == 'highpass':
            b, a = signal.butter(4, normalized_cutoff, btype='high')
        else:
            raise ValueError(f"不支持的滤波器类型: {filter_type}")
        
        # 应用滤波器
        for col in imu_columns:
            if col in data.columns:
                data[col] = signal.filtfilt(b, a, data[col])
        
        return data
    
    def normalize_data(self, data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        标准化数据
        
        Args:
            data: 数据DataFrame
            method: 标准化方法 ('standard', 'minmax')
            
        Returns:
            pd.DataFrame: 标准化后的数据
        """
        imu_columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'roll', 'pitch', 'yaw']
        
        for col in imu_columns:
            if col in data.columns:
                if method == 'standard':
                    scaler = StandardScaler()
                elif method == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    raise ValueError(f"不支持的标准化方法: {method}")
                
                # 拟合并转换数据
                data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1)).flatten()
                self.scalers[col] = scaler
        
        self.is_fitted = True
        return data
    
    def create_windows(self, data: pd.DataFrame, window_size: int = 90, 
                      overlap: float = 0.5) -> List[pd.DataFrame]:
        """
        创建滑动窗口
        
        Args:
            data: 数据DataFrame
            window_size: 窗口大小
            overlap: 重叠率
            
        Returns:
            List[pd.DataFrame]: 窗口列表
        """
        windows = []
        step_size = int(window_size * (1 - overlap))
        
        for i in range(0, len(data) - window_size + 1, step_size):
            window = data.iloc[i:i + window_size].copy()
            windows.append(window)
        
        return windows
    
    def process_pipeline(self, data: pd.DataFrame, 
                        filter_config: Dict = None,
                        normalize_config: Dict = None) -> pd.DataFrame:
        """
        完整的数据处理流水线
        
        Args:
            data: 原始数据
            filter_config: 滤波配置
            normalize_config: 标准化配置
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        # 清洗数据
        data = self.clean_data(data)
        
        # 应用滤波器
        if filter_config:
            data = self.apply_filter(data, **filter_config)
        
        # 标准化
        if normalize_config:
            data = self.normalize_data(data, **normalize_config)
        
        return data 