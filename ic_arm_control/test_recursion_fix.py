#!/usr/bin/env python3
"""
测试递归修复 - 验证自动轨迹提取不再造成无限循环
"""

import sys
import time
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from ic_arm_control.tools.unified_trajectory_manager import create_unified_manager
from ic_arm_control.control.async_logger_unified import create_unified_logger

def test_recursion_fix():
    """测试递归修复"""
    print("=== 测试递归修复 ===")

    try:
        # 创建统一管理器
        print("1. 创建统一轨迹管理器...")
        manager = create_unified_manager("test_recursion_fix")
        print("✓ 管理器创建成功")

        # 创建统一日志器
        print("2. 创建统一日志器...")
        logger = create_unified_logger(manager, auto_trajectory_extraction=True)
        print("✓ 日志器创建成功")

        # 启动日志器
        print("3. 启动日志器...")
        logger.start()
        print("✓ 日志器启动成功")

        # 记录一些测试数据
        print("4. 记录测试数据...")
        num_joints = 3
        test_duration = 5.0  # 5秒测试
        sample_rate = 20     # 20Hz

        initial_stats = logger.get_stats()
        print(f"  初始状态: {initial_stats['total_logs']} 日志, {initial_stats['buffer_size']} 缓冲")

        start_time = time.time()
        for i in range(int(test_duration * sample_rate)):
            t = i / sample_rate

            # 生成测试数据
            positions = np.array([
                0.5 * np.sin(2 * np.pi * 0.5 * t),
                0.3 * np.sin(2 * np.pi * 0.3 * t + np.pi/4),
                0.2 * np.sin(2 * np.pi * 0.7 * t)
            ])

            velocities = np.array([
                0.5 * 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * t),
                0.3 * 2 * np.pi * 0.3 * np.cos(2 * np.pi * 0.3 * t + np.pi/4),
                0.2 * 2 * np.pi * 0.7 * np.cos(2 * np.pi * 0.7 * t)
            ])

            torques = np.zeros(num_joints)

            # 记录电机状态
            logger.log_motor_states(positions, velocities, torques)

            # 每1秒检查一次状态
            if i % sample_rate == 0:
                current_stats = logger.get_stats()
                print(f"  t={t:.1f}s: {current_stats['total_logs']} 日志, {current_stats['buffer_size']} 缓冲")

            # 控制循环速度
            time.sleep(1.0 / sample_rate)

            # 检查是否陷入递归（如果处理时间过长，可能有问题）
            if time.time() - start_time > test_duration * 2:  # 超过2倍预期时间
                print("❌ 检测到可能的无限循环！")
                break

        elapsed_time = time.time() - start_time
        print(f"  数据记录完成，耗时: {elapsed_time:.2f}秒")

        # 等待自动轨迹提取完成
        print("5. 等待自动轨迹提取...")
        time.sleep(2)

        # 强制提取一次
        print("6. 强制提取轨迹...")
        forced_traj = logger.force_trajectory_extraction()
        if forced_traj:
            print(f"✓ 强制提取成功: {forced_traj}")
        else:
            print("⚠ 强制提取无结果")

        # 检查最终状态
        final_stats = logger.get_stats()
        print(f"\n7. 最终状态:")
        print(f"  总日志数: {final_stats['total_logs']}")
        print(f"  丢弃日志: {final_stats['dropped_logs']}")
        print(f"  缓冲区大小: {final_stats['buffer_size']}")
        print(f"  是否正在提取: {getattr(logger, '_is_extracting_trajectory', False)}")

        # 检查管理器中的轨迹
        trajectories = manager.list_trajectories()
        print(f"  管理器中轨迹数: {len(trajectories)}")

        # 停止日志器
        print("8. 清理资源...")
        logger.stop()
        manager.cleanup()
        print("✓ 资源清理完成")

        # 验证结果
        success = True
        if elapsed_time > test_duration * 1.5:  # 不应超过预期时间的1.5倍
            print(f"⚠ 警告: 执行时间过长 ({elapsed_time:.2f}s)")
            success = False

        if final_stats['total_logs'] == 0:
            print("❌ 没有记录到任何日志")
            success = False

        if trajectories:
            print("✓ 成功提取到轨迹")
        else:
            print("⚠ 没有提取到轨迹（可能数据不足）")

        print(f"\n=== 测试{'成功' if success else '失败'} ===")
        return success

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_recursion_fix()
    sys.exit(0 if success else 1)