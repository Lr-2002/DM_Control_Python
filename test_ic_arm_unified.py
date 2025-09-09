#!/usr/bin/env python3
"""
测试脚本：验证IC_ARM与unified_motor_control的集成
"""

import time
import numpy as np
from IC_ARM import ICARM

def test_basic_initialization():
    """测试基本初始化"""
    print("=== 测试1: 基本初始化 ===")
    try:
        # 初始化IC_ARM（使用统一电机控制系统）
        arm = ICARM(debug=True, gc=False, use_ht=False)
        print("✓ IC_ARM初始化成功")
        
        # 测试电机信息读取
        arm._read_motor_info()
        print("✓ 电机信息读取成功")
        
        return arm
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_state_reading(arm):
    """测试状态读取功能"""
    print("\n=== 测试2: 状态读取 ===")
    try:
        # 测试获取关节位置
        positions = arm.get_joint_positions()
        print(f"✓ 关节位置: {positions}")
        
        # 测试获取关节速度
        velocities = arm.get_joint_velocities()
        print(f"✓ 关节速度: {velocities}")
        
        # 测试获取关节力矩
        torques = arm.get_joint_torques()
        print(f"✓ 关节力矩: {torques}")
        
        # 测试获取完整状态
        complete_state = arm.get_complete_state()
        print(f"✓ 完整状态获取成功，包含 {len(complete_state)} 个字段")
        
        return True
    except Exception as e:
        print(f"✗ 状态读取失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_motor_control(arm):
    """测试电机控制功能"""
    print("\n=== 测试3: 电机控制 ===")
    try:
        # 获取当前位置作为起始位置
        current_positions = arm.get_joint_positions()
        print(f"当前位置: {np.degrees(current_positions)}")
        
        # 测试单个关节控制
        print("测试单个关节控制...")
        test_position = current_positions[0] + 0.1  # 第一个关节移动0.1弧度
        success = arm.set_joint_position(0, test_position)
        if success:
            print("✓ 单个关节控制成功")
        else:
            print("✗ 单个关节控制失败")
        
        time.sleep(1.0)
        
        # 测试批量关节控制
        print("测试批量关节控制...")
        target_positions = current_positions + np.array([0.05, -0.05, 0.03, -0.03, 0.02, -0.02])
        success = arm.set_joint_positions(target_positions)
        if success:
            print("✓ 批量关节控制成功")
        else:
            print("✗ 批量关节控制失败")
        
        time.sleep(2.0)
        
        # 回到原始位置
        print("回到原始位置...")
        success = arm.set_joint_positions(current_positions)
        if success:
            print("✓ 回到原始位置成功")
        else:
            print("✗ 回到原始位置失败")
        
        return True
    except Exception as e:
        print(f"✗ 电机控制测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_high_level_functions(arm):
    """测试高级功能"""
    print("\n=== 测试4: 高级功能 ===")
    try:
        # 测试设置零位
        print("测试设置零位...")
        success = arm.set_zero_position()
        if success:
            print("✓ 设置零位成功")
        else:
            print("✗ 设置零位失败")
        
        # 测试回零功能
        print("测试回零功能...")
        # 先移动到一个非零位置
        test_positions = np.array([0.2, -0.2, 0.15, -0.15, 0.1, -0.1])
        arm.set_joint_positions(test_positions)
        time.sleep(2.0)
        
        # 执行回零
        success = arm.home_to_zero(speed=0.3, timeout=15.0)
        if success:
            print("✓ 回零功能成功")
        else:
            print("✗ 回零功能失败")
        
        return True
    except Exception as e:
        print(f"✗ 高级功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance(arm):
    """测试性能"""
    print("\n=== 测试5: 性能测试 ===")
    try:
        # 测试状态读取频率
        print("测试状态读取频率...")
        start_time = time.time()
        num_reads = 100
        
        for i in range(num_reads):
            arm._refresh_all_states_fast()
        
        elapsed = time.time() - start_time
        frequency = num_reads / elapsed
        print(f"✓ 状态读取频率: {frequency:.1f} Hz")
        
        # 测试控制频率
        print("测试控制频率...")
        current_positions = arm.get_joint_positions()
        start_time = time.time()
        num_commands = 50
        
        for i in range(num_commands):
            # 发送微小的位置变化
            test_positions = current_positions + 0.01 * np.sin(i * 0.1)
            arm.set_joint_positions(test_positions)
            time.sleep(0.01)  # 100Hz控制频率
        
        elapsed = time.time() - start_time
        frequency = num_commands / elapsed
        print(f"✓ 控制频率: {frequency:.1f} Hz")
        
        return True
    except Exception as e:
        print(f"✗ 性能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("IC_ARM与unified_motor_control集成测试")
    print("=" * 50)
    
    # 测试1: 基本初始化
    arm = test_basic_initialization()
    if arm is None:
        print("初始化失败，终止测试")
        return
    
    try:
        # 测试2: 状态读取
        if not test_state_reading(arm):
            print("状态读取测试失败")
        
        # # 测试3: 电机控制
        # if not test_motor_control(arm):
        #     print("电机控制测试失败")
        
        # # 测试4: 高级功能
        # if not test_high_level_functions(arm):
        #     print("高级功能测试失败")
        
        # # 测试5: 性能测试
        # if not test_performance(arm):
        #     print("性能测试失败")
        
        print("\n" + "=" * 50)
        print("✓ 所有测试完成")
        
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"\n测试过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保电机安全失能
        try:
            arm.disable_all_motors()
            print("✓ 电机已安全失能")
        except:
            print("✗ 电机失能失败")

if __name__ == "__main__":
    main()
