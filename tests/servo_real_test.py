#!/usr/bin/env python3
"""
真实舵机测试脚本
直接使用UMC系统控制舵机
"""

import time
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 直接导入现有的工作模块
from IC_ARM import ICARM

class ServoTester:
    def __init__(self):
        """初始化舵机测试器"""
        self.icarm = None
        self.servo_id = 1
        
    def initialize(self):
        """初始化IC_ARM系统"""
        print("=== 初始化IC_ARM系统 ===")
        
        try:
            # 创建IC_ARM实例
            self.icarm = ICARM()
            
            # 手动添加舵机到系统
            print("添加舵机到系统...")
            from Python.unified_motor_control import MotorInfo, MotorType
            
            servo_info = MotorInfo(
                motor_id=self.servo_id,
                motor_type=MotorType.SERVO,
                name="test_servo",
                can_id=0x09,  # 舵机发送CAN ID
                rx_id=0x19    # 舵机接收CAN ID
            )
            
            # 直接添加到motor_manager
            success = self.icarm.motor_manager.add_motor(
                motor_id=self.servo_id,
                motor_type='servo',
                motor_info=servo_info
            )
            
            if not success:
                print("错误: 舵机添加失败!")
                return False
                
            print("✓ 系统初始化完成")
            return True
            
        except Exception as e:
            print(f"初始化失败: {e}")
            return False
    
    def test_servo_control(self):
        """测试舵机控制"""
        if not self.icarm:
            print("错误: 系统未初始化")
            return
            
        try:
            print("\n=== 舵机控制测试 ===")
            
            # 使能舵机
            print("1. 使能舵机...")
            result = self.icarm.motor_manager.enable_motor(self.servo_id)
            print(f"   结果: {'成功' if result else '失败'}")
            time.sleep(1)
            
            # 测试位置控制
            positions = [0.0, 0.5, -0.5, 1.0, -1.0, 0.0]  # 弧度
            
            for i, pos in enumerate(positions):
                print(f"2.{i+1} 设置位置: {pos:.2f} rad ({pos*180/3.14159:.1f}°)")
                
                result = self.icarm.motor_manager.set_command(
                    motor_id=self.servo_id,
                    pos=pos,
                    vel=0.5,  # 速度
                    kp=0, kd=0, tau=0  # 舵机协议不使用这些参数
                )
                
                print(f"   命令发送: {'成功' if result else '失败'}")
                
                # 读取反馈
                time.sleep(0.5)
                feedback = self.icarm.motor_manager.read_feedback(self.servo_id)
                if feedback:
                    print(f"   反馈: 位置={feedback.get('position', 0):.3f}rad, "
                          f"速度={feedback.get('velocity', 0):.3f}rad/s")
                else:
                    print("   反馈: 无数据")
                
                time.sleep(2)  # 等待运动完成
            
            # 失能舵机
            print("3. 失能舵机...")
            result = self.icarm.motor_manager.disable_motor(self.servo_id)
            print(f"   结果: {'成功' if result else '失败'}")
            
        except Exception as e:
            print(f"测试过程中出错: {e}")
    
    def interactive_mode(self):
        """交互式控制模式"""
        if not self.icarm:
            print("错误: 系统未初始化")
            return
            
        print("\n=== 交互式舵机控制 ===")
        print("命令:")
        print("  enable     - 使能舵机")
        print("  disable    - 失能舵机")
        print("  pos <度数> - 设置位置 (度)")
        print("  read       - 读取反馈")
        print("  quit       - 退出")
        
        # 默认使能舵机
        self.icarm.motor_manager.enable_motor(self.servo_id)
        print("舵机已使能")
        
        while True:
            try:
                cmd = input("\n> ").strip().lower()
                
                if cmd == "quit":
                    break
                elif cmd == "enable":
                    result = self.icarm.motor_manager.enable_motor(self.servo_id)
                    print("✓ 使能成功" if result else "✗ 使能失败")
                    
                elif cmd == "disable":
                    result = self.icarm.motor_manager.disable_motor(self.servo_id)
                    print("✓ 失能成功" if result else "✗ 失能失败")
                    
                elif cmd.startswith("pos "):
                    try:
                        angle_deg = float(cmd.split()[1])
                        angle_rad = angle_deg * 3.14159 / 180.0
                        
                        result = self.icarm.motor_manager.set_command(
                            motor_id=self.servo_id,
                            pos=angle_rad,
                            vel=1.0,
                            kp=0, kd=0, tau=0
                        )
                        
                        if result:
                            print(f"✓ 设置位置 {angle_deg}° ({angle_rad:.3f}rad)")
                        else:
                            print("✗ 位置设置失败")
                            
                    except (ValueError, IndexError):
                        print("✗ 无效的角度值")
                        
                elif cmd == "read":
                    feedback = self.icarm.motor_manager.read_feedback(self.servo_id)
                    if feedback:
                        pos_deg = feedback.get('position', 0) * 180.0 / 3.14159
                        print(f"位置: {feedback.get('position', 0):.3f}rad ({pos_deg:.1f}°)")
                        print(f"速度: {feedback.get('velocity', 0):.3f}rad/s")
                        print(f"力矩: {feedback.get('torque', 0):.3f}Nm")
                    else:
                        print("✗ 无反馈数据")
                        
                else:
                    print("✗ 未知命令")
                    
            except KeyboardInterrupt:
                break
        
        # 清理
        self.icarm.motor_manager.disable_motor(self.servo_id)
        print("\n舵机已失能，退出")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="舵机真实硬件测试")
    parser.add_argument("--mode", choices=["test", "interactive"], 
                       default="test", help="运行模式")
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = ServoTester()
    
    # 初始化
    if not tester.initialize():
        print("初始化失败，退出")
        return
    
    try:
        if args.mode == "test":
            tester.test_servo_control()
        elif args.mode == "interactive":
            tester.interactive_mode()
            
    except KeyboardInterrupt:
        print("\n测试被中断")
    except Exception as e:
        print(f"运行错误: {e}")
    finally:
        # 清理
        if tester.icarm:
            tester.icarm.motor_manager.disable_motor(tester.servo_id)
        print("清理完成")

if __name__ == "__main__":
    main()
