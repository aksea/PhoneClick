"""
基础IMU数据采集器

提供IMU数据采集的基础接口和功能。
"""

import time
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np


class IMUDataCollector(ABC):
    """
    IMU数据采集器基类
    
    提供IMU数据采集的基础接口，包括数据采集、存储和导出功能。
    """
    
    def __init__(self, sample_rate: int = 100, buffer_size: int = 1000):
        """
        初始化数据采集器
        
        Args:
            sample_rate: 采样率 (Hz)
            buffer_size: 缓冲区大小
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.is_collecting = False
        self.data_buffer = []
        self.start_time = None
        
    @abstractmethod
    def start_collection(self) -> bool:
        """
        开始数据采集
        
        Returns:
            bool: 是否成功启动采集
        """
        pass
    
    @abstractmethod
    def stop_collection(self) -> bool:
        """
        停止数据采集
        
        Returns:
            bool: 是否成功停止采集
        """
        pass
    
    @abstractmethod
    def get_imu_data(self) -> Dict[str, float]:
        """
        获取当前IMU数据
        
        Returns:
            Dict[str, float]: 包含ax, ay, az, gx, gy, gz, roll, pitch, yaw的数据字典
        """
        pass
    
    def collect_data(self, duration: float) -> pd.DataFrame:
        """
        采集指定时长的数据
        
        Args:
            duration: 采集时长 (秒)
            
        Returns:
            pd.DataFrame: 采集到的数据
        """
        if not self.start_collection():
            raise RuntimeError("无法启动数据采集")
        
        start_time = time.time()
        data_list = []
        
        while time.time() - start_time < duration:
            imu_data = self.get_imu_data()
            timestamp = time.time() - start_time
            imu_data['timestamp'] = timestamp
            data_list.append(imu_data)
            time.sleep(1.0 / self.sample_rate)
        
        self.stop_collection()
        
        return pd.DataFrame(data_list)
    
    def save_data(self, data: pd.DataFrame, filepath: str) -> bool:
        """
        保存数据到CSV文件
        
        Args:
            data: 要保存的数据
            filepath: 文件路径
            
        Returns:
            bool: 是否保存成功
        """
        try:
            data.to_csv(filepath, index=False)
            return True
        except Exception as e:
            print(f"保存数据失败: {e}")
            return False
    
    def get_data_info(self, data: pd.DataFrame) -> Dict:
        """
        获取数据统计信息
        
        Args:
            data: 数据DataFrame
            
        Returns:
            Dict: 统计信息字典
        """
        info = {
            'total_samples': len(data),
            'duration': data['timestamp'].max() - data['timestamp'].min(),
            'sample_rate': len(data) / (data['timestamp'].max() - data['timestamp'].min()),
            'columns': list(data.columns),
            'missing_values': data.isnull().sum().to_dict()
        }
        return info 