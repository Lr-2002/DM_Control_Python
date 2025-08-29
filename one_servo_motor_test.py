#!/usr/bin/env python3
"""
舵机控制测试脚本
使用ServoController类测试3个舵机的位置控制和读取
"""

import time
import signal
import sys
import serial
from DM_CAN import MotorControl, ServoController

# 全局变量
dm_can = None
servo_controller = None
running = True

def signal_handler(sig, frame):
    """信号处理函数"""
    global running, servo_controller, dm_can
    print("\n收到中断信号，正在停止舵机...")
    running = False
    
    if servo_controller:
        try:
            # 将所有舵机设置为0位置
            servo_controller.set_all_servo_positions([0, 0, 0])
            print("舵机已停止")
        except:
            pass
    
    if dm_can:
        try:
            dm_can.close()
            print("CAN连接已关闭")
        except:
            pass
    
    sys.exit(0)

def test_servo_positions():
    """测试舵机位置设置和读取"""
    global servo_controller
    
    print("\n=== 舵机位置测试 ===")
    
    # 测试位置1: [787, 787, 0] (对应原始的[13, 3, 13, 3, 0, 0])
    print("设置位置1: [787, 787, 0]")
    servo_controller.set_all_servo_positions([787, 787, 787])
    time.sleep(2)
    
    # 读取位置和速度
    positions = servo_controller.get_servo_positions()
    if positions:
        print(f"读取位置: {positions}")
    
    velocities = servo_controller.get_servo_velocities()
    if velocities:
        print(f"读取速度: {velocities}")
    
    time.sleep(1)
    
    # 测试位置2: [0, 1536, 0] (对应原始的[0, 0, 6, 0, 0, 0])
    print("\n设置位置2: [0, 1536, 0]")
    servo_controller.set_all_servo_positions([0, 1536, 0])
    time.sleep(2)
    
    # 读取位置和速度
    positions = servo_controller.get_servo_positions()
    if positions:
        print(f"读取位置: {positions}")
    
    velocities = servo_controller.get_servo_velocities()
    if velocities:
        print(f"读取速度: {velocities}")

def test_individual_servos():
    """测试单个舵机控制"""
    global servo_controller
    
    print("\n=== 单个舵机测试 ===")
    
    # 先设置所有舵机为0
    servo_controller.set_all_servo_positions([0, 0, 0])
    time.sleep(1)
    
    # 逐个测试每个舵机
    for i in range(3):
        print(f"\n测试舵机 {i} (A/B/C):")
        
        # 设置单个舵机位置
        test_pos = 500 + i * 100  # 不同的测试位置
        servo_controller.set_servo_position(i, test_pos)
        time.sleep(1)
        
        # 读取该舵机位置
        pos = servo_controller.get_servo_position(i)
        vel = servo_controller.get_servo_velocity(i)
        
        print(f"  设置位置: {test_pos}")
        print(f"  实际位置: {pos}")
        print(f"  当前速度: {vel}")
        
        # 相对移动测试
        print(f"  相对移动 +100")
        servo_controller.move_servo_relative(i, 100)
        time.sleep(1)
        
        new_pos = servo_controller.get_servo_position(i)
        print(f"  移动后位置: {new_pos}")

def main():
    global dm_can, servo_controller, running
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 初始化串口和CAN通信
        print("初始化串口连接...")
        serial_device = serial.Serial('/dev/cu.usbmodem00000000050C1', 921600, timeout=0.1)
        
        print("初始化CAN通信...")
        dm_can = MotorControl(serial_device)
        
        # 创建舵机控制器
        print("创建舵机控制器...")
        servo_controller = ServoController(dm_can)
        
        # 初始化舵机位置为0
        print("初始化舵机位置...")
        servo_controller.set_all_servo_positions([0, 0, 0])
        time.sleep(2)
        
        print("按Enter开始测试...")
        input()
        
        # 运行测试
        while running:
            print("\n选择测试模式:")
            print("1. 位置测试 (复现原始代码逻辑)")
            print("2. 单个舵机测试")
            print("3. 连续监控")
            print("4. 连续执行1")
            print("5. 退出")
            
            try:
                choice = input("请选择 (1-4): ").strip()
                
                if choice == '1':
                    test_servo_positions()
                elif choice == '2':
                    test_individual_servos()
                elif choice == '3':
                    print("连续监控模式 (Ctrl+C退出)")
                    while running:
                        positions = servo_controller.get_servo_positions()
                        velocities = servo_controller.get_servo_velocities()
                        print(f"位置: {positions}, 速度: {velocities}")
                        time.sleep(1)
                elif choice == '4':
                    while running:
                        test_servo_positions()
                        time.sleep(1)
                elif choice == '5':
                    break
                else:
                    print("无效选择，请重试")
                    
            except KeyboardInterrupt:
                break
                
    except Exception as e:
        print(f"错误: {e}")
    finally:
        # 清理资源
        if servo_controller:
            servo_controller.set_all_servo_positions([0, 0, 0])
        if dm_can:
            dm_can.close()

if __name__ == "__main__":
    main()
