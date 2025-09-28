#!/usr/bin/env python3
"""
测试常用范围余弦轨迹的速度优化
"""

import numpy as np
import json
import os
from excitation_trajectory_generator import ExcitationTrajectoryGenerator

def test_cosine_trajectory():
    """测试优化后的余弦轨迹"""
    print("=== 测试优化后的常用范围余弦轨迹 ===")

    # 初始化轨迹生成器
    urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/urdfs/ic_arm_8_dof/urdf/ic_arm_8_dof.urdf"
    generator = ExcitationTrajectoryGenerator(urdf_path, joint_margin=0.05)

    # 轨迹参数
    duration = 20.0
    sampling_rate = 200
    t = np.linspace(0, duration, int(sampling_rate * duration))

    # 常用角度设置
    common_angles_deg = {
        1: [-30, 10],     # Joint1: [-30°, 10°]
        2: [-100, 10],    # Joint2: [-100°, 10°]
        3: [-20, 70],     # Joint3: [-20°, 70°]
        4: [-100, 70],    # Joint4: [-100°, 70°]
        5: [-90, 90],     # Joint5: [-90°, 90°]
        6: [-120, 120]    # Joint6: [-120°, 120°]
    }

    # 转换为弧度
    common_angles_rad = {}
    for joint_id, angle_range_deg in common_angles_deg.items():
        angle_range_rad = [np.deg2rad(angle_range_deg[0]), np.deg2rad(angle_range_deg[1])]
        common_angles_rad[joint_id] = angle_range_rad

    # 为每个关节生成轨迹
    all_positions = np.zeros((len(t), generator.num_joints))

    for joint_idx in range(generator.num_joints):
        joint_id = joint_idx + 1
        joint_info = generator.joint_info[joint_idx]
        safe_limits = generator.safe_joint_limits[joint_idx]

        # 获取该关节的常用角度范围
        common_range = common_angles_rad[joint_id]
        common_min, common_max = common_range
        common_center = (common_min + common_max) / 2
        common_amplitude = (common_max - common_min) * 0.4  # 降低振幅到40%

        print(f"\nJoint{joint_id}:")
        print(f"  常用范围: [{np.rad2deg(common_min):.1f}°, {np.rad2deg(common_max):.1f}°]")
        print(f"  中心位置: {np.rad2deg(common_center):.1f}°")
        print(f"  振幅: {np.rad2deg(common_amplitude):.1f}°")

        # 余弦函数组合，大幅降低频率以减少速度
        trajectory = common_center
        frequencies = [0.01, 0.02, 0.035, 0.05]  # 显著降低频率
        phases = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        for i, freq in enumerate(frequencies):
            amp = common_amplitude * (0.4 ** i)  # 更快的振幅递减
            phase = phases[i] + np.random.uniform(0, np.pi/4)
            trajectory += amp * np.cos(2 * np.pi * freq * t + phase)

        # 确保在常用范围内
        trajectory = np.clip(trajectory, common_min, common_max)

        all_positions[:, joint_idx] = trajectory

        # 计算实际的运动范围
        actual_min = np.min(trajectory)
        actual_max = np.max(trajectory)
        actual_range = np.rad2deg(actual_max - actual_min)
        print(f"  实际范围: [{np.rad2deg(actual_min):.1f}°, {np.rad2deg(actual_max):.1f}°] = {actual_range:.1f}°")

    # 计算速度和加速度
    dt = 1.0 / sampling_rate
    velocities = np.zeros_like(all_positions)
    accelerations = np.zeros_like(all_positions)

    for joint_idx in range(generator.num_joints):
        # 速度计算
        velocities[1:-1, joint_idx] = (all_positions[2:, joint_idx] - all_positions[:-2, joint_idx]) / (2 * dt)
        velocities[0, joint_idx] = (all_positions[1, joint_idx] - all_positions[0, joint_idx]) / dt
        velocities[-1, joint_idx] = (all_positions[-1, joint_idx] - all_positions[-2, joint_idx]) / dt

        # 加速度计算
        accelerations[1:-1, joint_idx] = (velocities[2:, joint_idx] - velocities[:-2, joint_idx]) / (2 * dt)
        accelerations[0, joint_idx] = (velocities[1, joint_idx] - velocities[0, joint_idx]) / dt
        accelerations[-1, joint_idx] = (velocities[-1, joint_idx] - velocities[-2, joint_idx]) / dt

    # 检查速度限制
    max_velocities = []
    velocity_utilizations = []

    print(f"\n=== 速度利用率分析 ===")
    for joint_idx in range(generator.num_joints):
        joint_info = generator.joint_info[joint_idx]
        max_vel_limit = joint_info['velocity']
        joint_max_vel = np.max(np.abs(velocities[:, joint_idx]))
        joint_vel_util = (joint_max_vel / max_vel_limit) * 100

        max_velocities.append(joint_max_vel)
        velocity_utilizations.append(joint_vel_util)

        status = "✅" if joint_vel_util <= 90 else "⚠️" if joint_vel_util <= 100 else "❌"
        print(f"  Joint{joint_idx+1}: {joint_vel_util:.1f}% {status} (最大速度: {joint_max_vel:.3f} rad/s, 限制: {max_vel_limit:.3f} rad/s)")

    # 创建输出数据
    output_data = {
        'time': t.tolist(),
        'positions': all_positions.tolist(),
        'velocities': velocities.tolist(),
        'accelerations': accelerations.tolist(),
        'trajectory_info': {
            'type': 'common_range_cosine_optimized',
            'description': '优化后的常用范围余弦轨迹',
            'common_angles': common_angles_rad,
            'duration': duration,
            'sampling_rate': sampling_rate,
            'joint_velocities': max_velocities,
            'joint_velocity_utilizations': velocity_utilizations
        }
    }

    # 保存文件
    filename = "trajectory_common_angles_cosine_optimized.json"
    filepath = os.path.join("/Users/lr-2002/project/instantcreation/IC_arm_control", filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n已保存优化后的轨迹: {filename}")

    # 统计总结
    max_utilization = max(velocity_utilizations)
    avg_utilization = np.mean(velocity_utilizations)
    safe_joints = sum(1 for util in velocity_utilizations if util <= 90)

    print(f"\n=== 优化效果总结 ===")
    print(f"  最高速度利用率: {max_utilization:.1f}%")
    print(f"  平均速度利用率: {avg_utilization:.1f}%")
    print(f"  安全关节数量: {safe_joints}/{len(velocity_utilizations)}")

    if max_utilization <= 90:
        print(f"  ✅ 优化成功！所有关节都在安全范围内")
    elif max_utilization <= 100:
        print(f"  ⚠️ 优化较好，但仍有接近极限的关节")
    else:
        print(f"  ❌ 仍需进一步优化")

if __name__ == "__main__":
    test_cosine_trajectory()