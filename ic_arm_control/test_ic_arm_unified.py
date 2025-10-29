#!/usr/bin/env python3
"""
测试IC_ARM系统使用统一日志管理器
验证统一轨迹管理系统的集成
"""

import sys
import time
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from ic_arm_control.control.IC_ARM import ICARM

def test_ic_arm_unified_logger():
    """测试IC_ARM使用统一日志管理器"""
    print("=== IC_ARM 统一日志系统测试 ===")

    try:
        # 创建IC_ARM实例（使用模拟模式）
        print("1. 创建IC_ARM实例...")
        arm = ICARM(simulation_mode=True, enable_gc=False)
        print("✓ ICARM实例创建成功")

        # 检查统一管理器
        print("\n2. 检查统一管理器...")
        if hasattr(arm, 'unified_manager'):
            print(f"✓ 统一轨迹管理器已创建: {arm.unified_manager.session_id}")
            print(f"  会话目录: {arm.unified_manager.session_dir}")
        else:
            print("❌ 统一轨迹管理器未创建")
            return False

        # 检查统一日志器
        print("\n3. 检查统一日志器...")
        if hasattr(arm, 'logger'):
            stats = arm.logger.get_stats()
            print(f"✓ 统一日志器已启动")
            print(f"  日志目录: {stats['session_dir']}")
            print(f"  自动轨迹提取: {'启用' if stats['auto_extraction'] else '禁用'}")
            print(f"  JSON格式: {'启用' if stats['json_enabled'] else '禁用'}")
            print(f"  CSV格式: {'启用' if stats['csv_enabled'] else '禁用'}")
        else:
            print("❌ 统一日志器未创建")
            return False

        # 测试日志记录
        print("\n4. 测试日志记录...")
        num_joints = arm.motor_count

        # 生成一些模拟数据
        positions = np.random.uniform(-0.5, 0.5, num_joints)
        velocities = np.random.uniform(-1.0, 1.0, num_joints)
        torques = np.zeros(num_joints)

        # 记录电机状态
        arm.logger.log_motor_states(positions, velocities, torques)
        print("✓ 电机状态记录成功")

        # 记录控制事件
        arm.logger.log_control_event("test_event", {
            "test_value": 42,
            "description": "测试统一日志系统"
        })
        print("✓ 控制事件记录成功")

        # 等待一段时间让日志写入
        time.sleep(1)

        # 检查日志统计
        updated_stats = arm.logger.get_stats()
        print(f"\n5. 日志统计:")
        print(f"  总日志数: {updated_stats['total_logs']}")
        print(f"  丢弃日志: {updated_stats['dropped_logs']}")
        print(f"  队列大小: {updated_stats['queue_size']}")

        # 测试自动轨迹提取
        print("\n6. 测试自动轨迹提取...")
        if updated_stats['auto_extraction']:
            print("✓ 自动轨迹提取已启用")

            # 强制提取轨迹
            recent_traj = arm.logger.force_trajectory_extraction()
            if recent_traj:
                print(f"✓ 强制提取轨迹成功: {recent_traj}")
            else:
                print("⚠ 暂无足够数据提取轨迹")
        else:
            print("❌ 自动轨迹提取未启用")

        # 检查统一管理器的轨迹
        print("\n7. 检查统一管理器轨迹...")
        trajectories = arm.unified_manager.list_trajectories()
        print(f"  轨迹总数: {len(trajectories)}")

        if trajectories:
            print("  最近轨迹:")
            for traj_id in trajectories[-3:]:  # 显示最近3个
                traj_info = arm.unified_manager.get_trajectory_info(traj_id)
                if traj_info:
                    print(f"    - {traj_id}: {traj_info['metadata']['trajectory_type']}")

        print("\n8. 清理资源...")
        arm.close()
        print("✓ 资源清理完成")

        print("\n=== 测试完成 ===")
        print("✓ IC_ARM统一日志系统集成成功!")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ic_arm_unified_logger()
    sys.exit(0 if success else 1)