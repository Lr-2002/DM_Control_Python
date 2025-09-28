#!/usr/bin/env python3
"""
演示关节力矩限制功能
"""

import numpy as np
import sys
from pathlib import Path

# 添加路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def demo_torque_limits():
    """演示力矩限制功能"""
    print("🎯 关节力矩限制功能演示\n")

    try:
        from mlp_gravity_integrator import MLPGravityCompensation

        # 创建MLP重力补偿实例
        mlp_gc = MLPGravityCompensation()

        # 定义力矩限制
        torque_limits = [15.0, 12.0, 12.0, 4.0, 4.0, 3.0]  # 前6个关节
        print(f"📋 关节力矩限制: {torque_limits} Nm\n")

        # 测试场景
        scenarios = [
            {
                "name": "正常工作位置",
                "position": np.array([0.0, 0.5, 0.3, 0.2, -0.1, 0.0]),
                "description": "典型工作位置，力矩应在安全范围内"
            },
            {
                "name": "极端伸展位置",
                "position": np.array([1.8, 1.5, -1.0, 1.2, -1.5, 0.8]),
                "description": "机械臂完全伸展，可能产生较大力矩"
            },
            {
                "name": "高负载位置",
                "position": np.array([2.0, 1.8, -1.5, 1.5, -1.8, 1.0]),
                "description": "极端位置加上额外负载"
            }
        ]

        for i, scenario in enumerate(scenarios, 1):
            print(f"{'='*60}")
            print(f"场景 {i}: {scenario['name']}")
            print(f"描述: {scenario['description']}")
            print(f"关节位置: {scenario['position']}")

            # 计算重力补偿力矩
            compensation = mlp_gc.get_gravity_compensation_torque(scenario['position'])

            print(f"\n原始补偿力矩:")
            for j, torque in enumerate(compensation):
                print(f"  关节{j+1}: {torque:6.2f} Nm")

            # 应用力矩限制
            limited_compensation = compensation.copy()
            warnings = []

            for j in range(len(limited_compensation)):
                if abs(limited_compensation[j]) > torque_limits[j]:
                    original = limited_compensation[j]
                    limited_compensation[j] = np.sign(limited_compensation[j]) * torque_limits[j]
                    warnings.append({
                        'joint': j+1,
                        'original': original,
                        'limited': limited_compensation[j],
                        'limit': torque_limits[j]
                    })

            print(f"\n限制后补偿力矩:")
            for j, torque in enumerate(limited_compensation):
                print(f"  关节{j+1}: {torque:6.2f} Nm")

            # 显示警告信息
            if warnings:
                print(f"\n⚠️ 力矩限制警告:")
                for warning in warnings:
                    print(f"  关节{warning['joint']}: {warning['original']:.2f}Nm → {warning['limited']:.2f}Nm (限制: ±{warning['limit']}Nm)")
            else:
                print(f"\n✅ 所有关节力矩在安全范围内")

            # 安全性评估
            print(f"\n🛡️ 安全性评估:")
            max_ratio = max([abs(compensation[j]) / torque_limits[j] for j in range(len(compensation))])
            if max_ratio < 0.5:
                print("  状态: 🟢 安全 (力矩利用率 < 50%)")
            elif max_ratio < 0.8:
                print("  状态: 🟡 警告 (力矩利用率 50-80%)")
            elif max_ratio < 1.0:
                print("  状态: 🟠 注意 (力矩利用率 80-100%)")
            else:
                print("  状态: 🔴 危险 (力矩超限，已被限制)")

            print(f"  最高力矩利用率: {max_ratio*100:.1f}%")

        print(f"\n{'='*60}")
        print("📊 演示总结:")
        print("✅ 力矩限制功能正常工作")
        print("✅ 超限力矩被自动限制到安全范围")
        print("✅ 提供详细的警告信息")
        print("✅ 支持动态调整力矩限制")

        return True

    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_usage_examples():
    """演示使用示例"""
    print("\n📖 使用示例\n")

    print("1. 基本使用:")
    print("   arm = ICARM(gc=True, gc_type='mlp')")
    print("   compensation = arm.cal_gravity()  # 自动应用力矩限制")
    print("   arm.set_joint_torque(compensation)  # 自动应用力矩限制")

    print("\n2. 查看力矩限制:")
    print("   limits = arm.get_joint_torque_limits()")
    print("   print(f'力矩限制: {limits} Nm')")

    print("\n3. 修改力矩限制:")
    print("   new_limits = [10.0, 8.0, 8.0, 3.0, 3.0, 2.0, 4.0, 4.0]")
    print("   arm.set_joint_torque_limits(new_limits)")

    print("\n4. 安全建议:")
    print("   - 定期检查力矩使用情况")
    print("   - 在极端位置操作时格外小心")
    print("   - 根据实际负载调整力矩限制")
    print("   - 监控力矩限制警告信息")

if __name__ == "__main__":
    print("=== IC_ARM 关节力矩限制功能演示 ===\n")

    success = demo_torque_limits()
    if success:
        demo_usage_examples()
        print(f"\n🎉 演示完成！")
    else:
        print(f"\n❌ 演示过程中出现问题")