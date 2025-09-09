#!/usr/bin/env python3
"""
USB Hardware Wrapper 使用示例
"""

from Python.src import usb_class
from usb_hw_wrapper import wrapper

# 示例1: 创建一个新的usb_hw实例
# usb_hw = usb_class(1000000, 5000000, "F561E08C892274DB09496BCC1102DBC5")

# 示例2: 包装现有的usb_hw实例
def example_with_wrapper():
    # 创建原始实例
    original_usb_hw = usb_class(1000000, 5000000, "F561E08C892274DB09496BCC1102DBC5")
    
    # 包装原始实例
    wrapped_usb_hw = wrapper(original_usb_hw)
    
    # 现在可以使用wrapped_usb_hw，它具有与original_usb_hw相同的所有功能
    # 但你也可以通过继承USBHardwareWrapper类来重写特定方法
    
    # 例如，你可以重写setFrameCallback方法来添加额外的功能
    original_callback = None
    
    def enhanced_callback(frame):
        print(f"Enhanced callback: Received frame with ID: {hex(frame.head.id)}")
        # 调用原始回调
        if original_callback:
            original_callback(frame)
    
    # 保存原始回调
    original_callback = wrapped_usb_hw.setFrameCallback
    
    # 设置增强的回调
    wrapped_usb_hw.setFrameCallback = enhanced_callback
    
    return wrapped_usb_hw

# 示例3: 创建一个自定义的包装类
class MyCustomUSBWrapper:
    def __init__(self, usb_hw):
        self.usb_hw = wrapper(usb_hw)  # 使用我们的包装器
        
        # 添加自定义功能
        self.debug_mode = False
    
    def enable_debug(self):
        """启用调试模式"""
        self.debug_mode = True
        print("Debug mode enabled")
    
    def disable_debug(self):
        """禁用调试模式"""
        self.debug_mode = False
        print("Debug mode disabled")
    
    def fdcanFrameSend(self, data, canId):
        """重写发送帧的方法，添加调试信息"""
        if self.debug_mode:
            print(f"Sending CAN frame: ID={hex(canId)}, Data={[hex(x) for x in data]}")
        
        # 调用原始方法
        return self.usb_hw.fdcanFrameSend(data, canId)
    
    # 其他方法通过__getattr__转发
    def __getattr__(self, name):
        return getattr(self.usb_hw, name)

if __name__ == "__main__":
    # 这里只是示例代码，实际运行时可能需要根据实际情况调整
    print("USB Hardware Wrapper 使用示例")
    print("注意: 这个脚本只是演示包装器的用法，不会实际执行USB通信")
