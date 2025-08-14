#!/usr/bin/env python3
"""
测试回零功能
演示IC ARM的home_to_zero和set_zero_position功能
"""

from IC_ARM import ICARM
import time
import numpy as np

def main():
    """测试回零功能"""
    print("=== 测试IC ARM回零功能 ===\n")
    
    arm = ICARM()
    
    try:
        print("1. 连接到IC ARM...")
        if not arm.connect():
            print("连接失败，程序退出")
            return
        
        print("✓ 连接成功\n")
        
        # 显示当前位置
        print("2. 显示当前位置:")
        current_pos = arm.get_joint_positions()
        if current_pos is not None:
            current_pos_deg = [np.degrees(pos) for pos in current_pos]
            print(f"当前位置: {[f'{p:.2f}°' for p in current_pos_deg]}")
            
            # 计算距离零位的总距离
            total_distance = sum(abs(pos) for pos in current_pos)
            print(f"距离零位总距离: {np.degrees(total_distance):.1f}°\n")
        else:
            print("无法获取当前位置")
            return
        
        # 询问用户是否要执行回零
        response = input("是否执行回零操作? (y/n): ").lower().strip()
        if response != 'y':
            print("用户取消回零操作")
            return
        
        print("\n3. 开始回零操作...")
        print("使用平滑轨迹回到零位...")
        
        # 执行回零操作
        success = arm.home_to_zero(speed=0.3, timeout=30.0)
        
        if success:
            print("\n✓ 回零操作成功完成!")
            
            # 再次显示位置确认
            final_pos = arm.get_joint_positions()
            if final_pos is not None:
                final_pos_deg = [np.degrees(pos) for pos in final_pos]
                print(f"最终位置: {[f'{p:.3f}°' for p in final_pos_deg]}")
                
                max_error = max(abs(pos) for pos in final_pos)
                print(f"最大误差: {np.degrees(max_error):.3f}°")
        else:
            print("\n✗ 回零操作失败")
        
        print("\n4. 测试软件零位设置功能...")
        
        # 演示软件零位设置
        response = input("是否测试软件零位设置? (y/n): ").lower().strip()
        if response == 'y':
            success = arm.set_zero_position()
            if success:
                print("✓ 软件零位设置成功")
            else:
                print("✗ 软件零位设置失败")
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
    finally:
        # 安全断开连接
        print("\n5. 断开连接...")
        arm.disconnect()
        print("✓ 程序结束")

def test_homing_with_different_speeds():
    """测试不同速度的回零操作"""
    print("=== 测试不同回零速度 ===\n")
    
    arm = ICARM()
    
    try:
        if not arm.connect():
            print("连接失败")
            return
        
        speeds = [0.1, 0.3, 0.5, 1.0]  # 不同的回零速度
        
        for speed in speeds:
            print(f"\n测试回零速度: {speed} rad/s")
            
            current_pos = arm.get_joint_positions()
            if current_pos is None:
                continue
                
            # 只有当前不在零位时才测试
            max_distance = max(abs(pos) for pos in current_pos)
            if max_distance > np.radians(1.0):  # 大于1度才回零
                print(f"当前最大偏差: {np.degrees(max_distance):.1f}°")
                
                start_time = time.time()
                success = arm.home_to_zero(speed=speed, timeout=60.0)
                end_time = time.time()
                
                if success:
                    print(f"✓ 回零成功，用时: {end_time - start_time:.1f}秒")
                else:
                    print(f"✗ 回零失败")
            else:
                print("已在零位附近，跳过测试")
                
    except Exception as e:
        print(f"测试出错: {e}")
    finally:
        arm.disconnect()

if __name__ == "__main__":
    # 运行主测试
    main()
    
    # 可选: 运行速度测试
    # test_homing_with_different_speeds()
