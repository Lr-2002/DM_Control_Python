#!/usr/bin/env python3
"""
舵机UMC测试脚本
使用现有的ServoController直接控制舵机
"""

import time
import sys
import serial
from DM_CAN import MotorControl, ServoController

class ServoUMCTest:
    def __init__(self):
        """初始化舵机测试器"""
        self.dm_can = None
        self.servo_controller = None
        self.serial_device = None
        
    def initialize(self, port='/dev/cu.usbmodem00000000050C1', baudrate=921600):
        """
        初始化舵机控制系统
        
        Args:
            port: 串口设备路径
            baudrate: 波特率
        """
        try:
            print("=== 初始化舵机控制系统 ===")
            
            # 初始化串口
            print(f"连接串口: {port} @ {baudrate}")
            self.serial_device = serial.Serial(port, baudrate, timeout=0.1)
            
            # 初始化CAN通信
            print("初始化CAN通信...")
            self.dm_can = MotorControl(self.serial_device)
            
            # 创建舵机控制器
            print("创建舵机控制器...")
            self.servo_controller = ServoController(self.dm_can)
            
            # 初始化舵机位置
            print("初始化舵机位置为0...")
            self.servo_controller.set_all_servo_positions([0, 0, 0])
            time.sleep(1)
            
            print("✓ 舵机控制系统初始化完成")
            return True
            
        except Exception as e:
            print(f"✗ 初始化失败: {e}")
            return False
    
    def test_basic_control(self):
        """测试基本舵机控制功能"""
        print("\n=== 基本舵机控制测试 ===")
        
        # 测试位置序列
        test_positions = [
            [0, 0, 0],      # 初始位置
            [500, 0, 0],    # 舵机A移动
            [500, 500, 0],  # 舵机B移动
            [500, 500, 500], # 舵机C移动
            [0, 500, 500],  # 舵机A回零
            [0, 0, 500],    # 舵机B回零
            [0, 0, 0],      # 全部回零
        ]
        
        for i, positions in enumerate(test_positions):
            print(f"\n步骤 {i+1}: 设置位置 {positions}")
            
            # 设置位置
            success = self.servo_controller.set_all_servo_positions(positions)
            if success:
                print("✓ 位置命令发送成功")
            else:
                print("✗ 位置命令发送失败")
                continue
            
            time.sleep(2)  # 等待运动完成
            
            # 读取反馈
            actual_positions = self.servo_controller.get_servo_positions()
            velocities = self.servo_controller.get_servo_velocities()
            
            if actual_positions:
                print(f"  实际位置: {actual_positions}")
            else:
                print("  位置读取失败")
                
            if velocities:
                print(f"  当前速度: {velocities}")
            else:
                print("  速度读取失败")
    
    def test_individual_servos(self):
        """测试单个舵机控制"""
        print("\n=== 单个舵机控制测试 ===")
        
        # 先回零
        self.servo_controller.set_all_servo_positions([0, 0, 0])
        time.sleep(1)
        
        # 逐个测试每个舵机
        for servo_id in range(3):
            servo_name = ['A', 'B', 'C'][servo_id]
            print(f"\n测试舵机 {servo_name} (索引 {servo_id}):")
            
            # 设置测试位置
            test_pos = 300 + servo_id * 100
            print(f"  设置位置: {test_pos}")
            
            success = self.servo_controller.set_servo_position(servo_id, test_pos)
            if success:
                print("  ✓ 位置设置成功")
            else:
                print("  ✗ 位置设置失败")
                continue
            
            time.sleep(1.5)
            
            # 读取单个舵机位置
            pos = self.servo_controller.get_servo_position(servo_id)
            vel = self.servo_controller.get_servo_velocity(servo_id)
            
            print(f"  实际位置: {pos}")
            print(f"  当前速度: {vel}")
            
            # 相对移动测试
            print(f"  相对移动 +100")
            self.servo_controller.move_servo_relative(servo_id, 100)
            time.sleep(1)
            
            new_pos = self.servo_controller.get_servo_position(servo_id)
            print(f"  移动后位置: {new_pos}")
    
    def interactive_control(self):
        """交互式舵机控制"""
        print("\n=== 交互式舵机控制 ===")
        print("命令格式:")
        print("  set <a> <b> <c>    - 设置3个舵机位置")
        print("  move <id> <pos>    - 设置单个舵机位置 (id: 0-2)")
        print("  rel <id> <delta>   - 相对移动舵机")
        print("  read               - 读取所有舵机状态")
        print("  zero               - 所有舵机回零")
        print("  quit               - 退出")
        
        while True:
            try:
                cmd = input("\n> ").strip().lower().split()
                
                if not cmd:
                    continue
                    
                if cmd[0] == "quit":
                    break
                    
                elif cmd[0] == "set" and len(cmd) == 4:
                    try:
                        positions = [int(cmd[1]), int(cmd[2]), int(cmd[3])]
                        success = self.servo_controller.set_all_servo_positions(positions)
                        print("✓ 设置成功" if success else "✗ 设置失败")
                    except ValueError:
                        print("✗ 无效的位置值")
                        
                elif cmd[0] == "move" and len(cmd) == 3:
                    try:
                        servo_id = int(cmd[1])
                        position = int(cmd[2])
                        if 0 <= servo_id <= 2:
                            success = self.servo_controller.set_servo_position(servo_id, position)
                            print("✓ 移动成功" if success else "✗ 移动失败")
                        else:
                            print("✗ 舵机ID必须在0-2之间")
                    except ValueError:
                        print("✗ 无效的参数")
                        
                elif cmd[0] == "rel" and len(cmd) == 3:
                    try:
                        servo_id = int(cmd[1])
                        delta = int(cmd[2])
                        if 0 <= servo_id <= 2:
                            success = self.servo_controller.move_servo_relative(servo_id, delta)
                            print("✓ 相对移动成功" if success else "✗ 相对移动失败")
                        else:
                            print("✗ 舵机ID必须在0-2之间")
                    except ValueError:
                        print("✗ 无效的参数")
                        
                elif cmd[0] == "read":
                    positions = self.servo_controller.get_servo_positions()
                    velocities = self.servo_controller.get_servo_velocities()
                    
                    if positions:
                        print(f"位置: A={positions[0]}, B={positions[1]}, C={positions[2]}")
                    else:
                        print("位置读取失败")
                        
                    if velocities:
                        print(f"速度: A={velocities[0]}, B={velocities[1]}, C={velocities[2]}")
                    else:
                        print("速度读取失败")
                        
                elif cmd[0] == "zero":
                    success = self.servo_controller.set_all_servo_positions([0, 0, 0])
                    print("✓ 回零成功" if success else "✗ 回零失败")
                    
                else:
                    print("✗ 未知命令或参数错误")
                    
            except KeyboardInterrupt:
                break
        
        print("退出交互模式")
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.servo_controller:
                print("舵机回零...")
                self.servo_controller.set_all_servo_positions([0, 0, 0])
                time.sleep(1)
                
            if self.dm_can:
                print("关闭CAN连接...")
                self.dm_can.close()
                
            if self.serial_device:
                print("关闭串口连接...")
                self.serial_device.close()
                
            print("清理完成")
            
        except Exception as e:
            print(f"清理过程中出错: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="舵机UMC测试脚本")
    parser.add_argument("--port", default="/dev/cu.usbmodem00000000050C1", 
                       help="串口设备路径")
    parser.add_argument("--baudrate", type=int, default=921600, 
                       help="串口波特率")
    parser.add_argument("--mode", choices=["basic", "individual", "interactive"], 
                       default="interactive", help="测试模式")
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = ServoUMCTest()
    
    try:
        # 初始化
        if not tester.initialize(args.port, args.baudrate):
            print("初始化失败，退出")
            return
        
        # 运行测试
        if args.mode == "basic":
            tester.test_basic_control()
        elif args.mode == "individual":
            tester.test_individual_servos()
        elif args.mode == "interactive":
            tester.interactive_control()
            
    except KeyboardInterrupt:
        print("\n测试被中断")
    except Exception as e:
        print(f"运行错误: {e}")
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()
