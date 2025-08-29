#!/usr/bin/env python3
"""
单电机测试脚本
控制0x01电机发送-5Nm MIT扭矩
"""

import time
import signal
import sys
import serial
from DM_CAN import MotorControl, Motor, DM_Motor_Type

# 全局变量
dm_can = None
motor = None
running = True

def signal_handler(sig, frame):
    """信号处理函数"""
    global running, motor, dm_can
    print("\n收到中断信号，正在停止电机...")
    running = False
    
    if motor:
        try:
            # 停止电机
            dm_can.controlMIT(motor, 0, 0, 0, 0, 0)
            print("电机已停止")
        except:
            pass
    
    if dm_can:
        try:
            dm_can.close()
            print("CAN连接已关闭")
        except:
            pass
    
    sys.exit(0)

def main():
    global dm_can, motor, running
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 初始化串口和CAN通信
        print("初始化串口连接...")
        serial_device = serial.Serial('/dev/cu.usbmodem00000000050C1', 921600, timeout=0.1)
        
        print("初始化CAN通信...")
        dm_can = MotorControl(serial_device)
        
        # 创建0x01电机对象 (ID=0x01, 类型=DM10010L)
        print("创建电机对象...")
        # motor = Motor(DM_Motor_Type.DM10010L, 0x01, 0x00)
        motor = Motor(DM_Motor_Type.DM4340, 0x02, 0x00)
        dm_can.addMotor(motor)
        
        # 使能电机
        print("使能电机...")
        dm_can.enable(motor)
        time.sleep(3)
        input()
        print("开始发送-5Nm MIT扭矩...")
        print("按Ctrl+C停止")
        
        # 控制循环
        while running:
            try:
                # 发送MIT控制命令: kp=0, kd=0, position=0, velocity=0, torque=-5Nm
                dm_can.controlMIT(motor, kp=0, kd=0, q=0, dq=0, tau=-1.0)
                
                # 读取电机状态
                pos = motor.getPosition()
                vel = motor.getVelocity()
                torque = motor.getTorque()
                
                print(f"位置: {pos:.3f} rad, 速度: {vel:.3f} rad/s, 扭矩: {torque:.3f} Nm")
                
                time.sleep(0.01)  # 100Hz控制频率
                
            except Exception as e:
                print(f"控制过程中出错: {e}")
                break
    
    except Exception as e:
        print(f"初始化失败: {e}")
    
    finally:
        # 清理资源
        if motor and dm_can:
            try:
                print("停止电机...")
                dm_can.controlMIT(motor, 0, 0, 0, 0, 0)
                dm_can.disable(motor)
            except:
                pass
        
        if dm_can:
            try:
                dm_can.close()
            except:
                pass
        
        print("测试结束")

if __name__ == "__main__":
    main()
