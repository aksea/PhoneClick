"""
数据采集模块

提供多平台IMU数据采集功能，支持iOS、Android和桌面平台。
"""

from .collector import IMUDataCollector
from .mobile_collector import MobileIMUCollector
from .desktop_collector import DesktopIMUCollector

__all__ = [
    'IMUDataCollector',
    'MobileIMUCollector', 
    'DesktopIMUCollector'
] 