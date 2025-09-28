#!/usr/bin/env python3
"""
测试轨迹执行性能优化效果
验证日志记录优化是否解决了降速问题
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# 添加项目路径到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ic_arm_control.control.IC_ARM import ICARM


def generate_test_trajectory(duration=5.0, frequency=50.0):
    """
    生成测试轨迹

    Args:
        duration: 轨迹持续时间（秒）
        frequency: 控制频率（Hz）

    Returns:
        list: 轨迹点列表
    """
    # 创建简单的正弦波轨迹
    num_points = int(duration * frequency)
    trajectory = []

    for i in range(num_points):
        t = i / frequency

        # 生成6个关节的轨迹点（正弦波）
        positions = []
        for joint in range(6):
            amplitude = 10.0 * (joint + 1)  # 不同关节不同幅度
            frequency_hz = 0.5 + joint * 0.1  # 不同关节不同频率
            pos = amplitude * np.sin(2 * np.pi * frequency_hz * t)
            positions.append(pos)

        # 添加时间戳
        point = positions + [t]
        trajectory.append(point)

    return trajectory


def test_trajectory_performance():
    """测试轨迹执行性能"""
    print("=== 轨迹执行性能测试 ===")

    try:
        # 初始化IC ARM（禁用日志记录以测试性能）
        print("初始化IC ARM...")
        arm = ICARM(debug=False, gc=False, enable_buffered_control=False)

        # 生成测试轨迹
        trajectory_duration = 3.0  # 3秒测试
        test_trajectory = generate_test_trajectory(duration=trajectory_duration, frequency=100.0)

        print(f"测试轨迹: {len(test_trajectory)} 个点, 持续时间: {trajectory_duration} 秒")

        # 测试1: 启用日志记录
        print("\n--- 测试1: 启用日志记录 ---")
        start_time = time.time()
        success1 = arm.execute_trajectory_points(
            test_trajectory,
            verbose=True,
            smooth_start=False,
            smooth_end=False,
            enable_logging=True
        )
        end_time = time.time()

        if success1:
            fps1 = len(test_trajectory) / (end_time - start_time)
            print(f"启用日志记录时的平均FPS: {fps1:.1f}")
        else:
            print("启用日志记录时执行失败")
            fps1 = 0

        # 等待系统稳定
        time.sleep(1.0)

        # 测试2: 禁用日志记录
        print("\n--- 测试2: 禁用日志记录 ---")
        start_time = time.time()
        success2 = arm.execute_trajectory_points(
            test_trajectory,
            verbose=True,
            smooth_start=False,
            smooth_end=False,
            enable_logging=False
        )
        end_time = time.time()

        if success2:
            fps2 = len(test_trajectory) / (end_time - start_time)
            print(f"禁用日志记录时的平均FPS: {fps2:.1f}")
        else:
            print("禁用日志记录时执行失败")
            fps2 = 0

        # 性能比较
        print("\n=== 性能比较结果 ===")
        if success1 and success2:
            improvement = (fps2 - fps1) / fps1 * 100
            print(f"启用日志记录: {fps1:.1f} FPS")
            print(f"禁用日志记录: {fps2:.1f} FPS")
            print(f"性能提升: {improvement:+.1f}%")

            if improvement > 10:
                print("✅ 优化效果显著！")
            elif improvement > 0:
                print("⚠ 性能有轻微提升")
            else:
                print("❌ 性能没有改善，可能存在其他瓶颈")
        elif success2:
            print("✅ 禁用日志记录后执行成功")
            print("❌ 启用日志记录时执行失败")
        else:
            print("❌ 两种模式都执行失败，需要检查其他问题")

        # 清理
        arm.close()

    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()


def test_logging_impact():
    """测试日志记录对性能的具体影响"""
    print("\n=== 日志记录影响测试 ===")

    try:
        print("初始化IC ARM...")
        arm = ICARM(debug=False, gc=False, enable_buffered_control=False)

        # 测试不同状态读取方法的性能
        print("\n测试不同状态读取方法的性能...")

        methods = [
            ('normal', lambda: arm._read_all_states(refresh=True, enable_logging=True)),
            ('fast', lambda: arm._read_all_states_fast()),
            ('cached', lambda: arm._read_all_states_cached())
        ]

        for method_name, method_func in methods:
            print(f"\n--- 测试 {method_name} 方法 ---")

            # 预热
            for _ in range(10):
                method_func()

            # 性能测试
            test_duration = 2.0
            start_time = time.time()
            count = 0

            while time.time() - start_time < test_duration:
                method_func()
                count += 1

            actual_duration = time.time() - start_time
            fps = count / actual_duration

            print(f"方法: {method_name}")
            print(f"调用次数: {count}")
            print(f"持续时间: {actual_duration:.3f} 秒")
            print(f"平均FPS: {fps:.1f} Hz")

        arm.close()

    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("IC ARM 轨迹执行性能优化验证")
    print("=" * 50)

    # 检查参数
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
    else:
        test_type = "all"

    if test_type in ["all", "trajectory"]:
        test_trajectory_performance()

    if test_type in ["all", "logging"]:
        test_logging_impact()

    print("\n测试完成！")


if __name__ == "__main__":
    main()