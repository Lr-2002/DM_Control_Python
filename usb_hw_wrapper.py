
#!/usr/bin/env python3
"""
USB Hardware Wrapper

提供对usb_class的包装，可以包装现有实例或创建新实例
"""

from Python.src import usb_class


class USBHardwareWrapper:
    """
    USB硬件接口包装类
    
    可以包装现有的usb_class实例，保持原有功能的同时允许重写特定方法
    """
    
    def __init__(self, usb_hw_or_nom_baud, dat_baud=None, sn=None):
        """
        初始化USB硬件接口包装器
        
        可以以两种方式初始化:
        1. 传入现有的usb_hw实例: wrapper = USBHardwareWrapper(usb_hw)
        2. 创建新的实例: wrapper = USBHardwareWrapper(nom_baud, dat_baud, sn)
        
        Args:
            usb_hw_or_nom_baud: 现有的usb_hw实例或标称波特率
            dat_baud: 数据波特率（如果创建新实例）
            sn: 设备序列号（如果创建新实例）
        """
        # 检查是否传入了现有的usb_hw实例
        if dat_baud is None and sn is None:
            # 包装现有实例
            self.usb_hw = usb_hw_or_nom_baud
        else:
            # 创建新实例
            self.usb_hw = usb_class(usb_hw_or_nom_baud, dat_baud, sn)
    
    def __getattr__(self, name):
        """
        转发所有未定义的方法调用到底层usb_class实例
        
        Args:
            name: 方法名
            
        Returns:
            底层usb_class实例的方法
        """
        return getattr(self.usb_hw, name)
    
    def __enter__(self):
        """
        支持上下文管理器协议
        """
        self.usb_hw.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        支持上下文管理器协议
        """
        return self.usb_hw.__exit__(exc_type, exc_val, exc_tb)


# 简单包装函数，可以直接使用: usb_hw = wrapper(usb_hw)
def wrapper(usb_hw):
    """
    包装现有的usb_hw实例
    
    Args:
        usb_hw: 现有的usb_hw实例
        
    Returns:
        USBHardwareWrapper实例
    """
    return USBHardwareWrapper(usb_hw)


# 为了方便使用，提供与原始usb_class相同的构造函数接口
def create_usb_hardware(nom_baud, dat_baud, sn):
    """
    创建USB硬件接口实例
    
    Args:
        nom_baud: 标称波特率
        dat_baud: 数据波特率
        sn: 设备序列号
        
    Returns:
        USBHardwareWrapper实例
    """
    return USBHardwareWrapper(nom_baud, dat_baud, sn)
