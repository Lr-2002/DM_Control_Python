#!/usr/bin/env python3
"""
加载和使用辨识参数的简单演示
"""

import numpy as np
import os
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

from minimum_gc import MinimumGravityCompensation

def load_and_test_parameters():
    """加载并测试辨识参数"""
    print("=== 辨识参数加载和使用演示 ===\n")

    # 1. 直接使用重力补偿器 (自动加载最新参数)
    print("1. 初始化重力补偿器:")
    gc = MinimumGravityCompensation()

    # 2. 获取参数信息
    param_info = gc.get_parameter_info()
    print(f"   参数数量: {param_info['num_base_params']}")
    print(f"   参数格式: {param_info['param_format']}")
    print(f"   参数范围: [{param_info['param_range'][0]:.6f}, {param_info['param_range'][1]:.6f}]")
    print()

    # 3. 测试不同位置的力矩计算
    print("2. 测试力矩计算:")

    # 测试位置1: 零位
    q1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tau1 = gc.calculate_gravity_torque(q1)
    print(f"   零位 ({q1}):")
    print(f"   重力力矩: {tau1[0]}")

    # 测试位置2: 弯曲配置
    q2 = np.array([0.5, -0.3, 0.2, 0.4, -0.2, 0.1])
    tau2 = gc.calculate_gravity_torque(q2)
    print(f"   弯曲配置 ({np.degrees(q2)}°):")
    print(f"   重力力矩: {tau2[0]}")

    # 测试位置3: 伸展配置
    q3 = np.array([-0.4, 0.5, -0.3, 0.6, 0.2, -0.1])
    tau3 = gc.calculate_gravity_torque(q3)
    print(f"   伸展配置 ({np.degrees(q3)}°):")
    print(f"   重力力矩: {tau3[0]}")

    # 4. 测试完整动力学计算
    print("\n3. 测试完整动力学:")
    dq = np.array([0.1, 0.2, 0.1, 0.15, 0.1, 0.05])
    ddq = np.array([0.5, 0.3, 0.2, 0.4, 0.1, 0.2])

    tau_total = gc.calculate_torque(q2, dq, ddq)
    tau_gravity = gc.calculate_gravity_torque(q2)

    print(f"   输入位置: {np.degrees(q2)}°")
    print(f"   输入速度: {np.degrees(dq)}°/s")
    print(f"   输入加速度: {np.degrees(ddq)}°/s²")
    print(f"   总力矩: {tau_total[0]}")
    print(f"   重力力矩: {tau_gravity[0]}")
    print(f"   动态力矩: {tau_total[0] - tau_gravity[0]}")

    # 5. 多点测试
    print("\n4. 多点批量测试:")
    n_points = 5
    q_batch = np.random.uniform(-0.5, 0.5, (n_points, 6))
    tau_batch = gc.calculate_gravity_torque(q_batch)

    print(f"   批量计算 {n_points} 个配置:")
    for i in range(n_points):
        print(f"   配置 {i+1}: {np.degrees(q_batch[i])[:3]}° → 力矩: {tau_batch[i][:3]}")

    print(f"\n✅ 参数加载和测试完成!")
    print(f"   辨识参数已成功加载并可用于实时控制。")

    return gc

if __name__ == "__main__":
    gc = load_and_test_parameters()