#!/usr/bin/env python3
"""
为Joint3生成以0为中心的轨迹，特别针对在0附近的动力学辨识
"""

import numpy as np
import json
import os
from excitation_trajectory_generator import ExcitationTrajectoryGenerator

def generate_joint3_zero_centered_trajectory():
    """为Joint3生成以0为中心的轨迹"""

    print("=== 为Joint3生成0附近活动的轨迹 ===")

    # 初始化轨迹生成器
    urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/urdfs/ic_arm_8_dof/urdf/ic_arm_8_dof.urdf"
    generator = ExcitationTrajectoryGenerator(urdf_path, joint_margin=0.05)

    # Joint3信息
    joint3_idx = 2  # Joint3是第3个关节(0-based索引)
    joint3_info = generator.joint_info[joint3_idx]
    joint3_safe_limits = generator.safe_joint_limits[joint3_idx]

    print(f"Joint3原始范围: [{joint3_info['lower']:.3f}, {joint3_info['upper']:.3f}] rad")
    print(f"Joint3安全范围: [{joint3_safe_limits['lower']:.3f}, {joint3_safe_limits['upper']:.3f}] rad")
    print(f"Joint3速度限制: {joint3_info['velocity']:.3f} rad/s")

    # 为Joint3设计以0为中心的轨迹
    duration = 20.0  # 增加持续时间到20秒
    sampling_rate = 200
    t = np.linspace(0, duration, int(sampling_rate * duration))

    # 计算以0为中心的振幅
    max_velocity = joint3_info['velocity']
    zero_position = 0.0

    # 确保振幅不会超出安全范围
    max_positive_amplitude = min(joint3_safe_limits['upper'] - zero_position, zero_position - joint3_safe_limits['lower'])
    max_positive_amplitude = min(max_positive_amplitude, 0.8)  # 限制最大振幅

    # 基于速度限制计算最大振幅
    max_freq = 1.0  # Hz
    omega_max = 2 * np.pi * max_freq
    velocity_limited_amplitude = max_velocity / (omega_max * 1.5)  # 1.5倍安全余量

    # 取较小的振幅
    amplitude = min(max_positive_amplitude, velocity_limited_amplitude)
    print(f"使用振幅: {amplitude:.3f} rad")

    # 生成多种轨迹类型
    trajectories = {}

    # 1. 低频大振幅正弦波
    print("生成低频大振幅正弦波...")
    freq1 = 0.1  # Hz
    trajectory1 = zero_position + amplitude * np.sin(2 * np.pi * freq1 * t)
    trajectories['low_freq_sine'] = trajectory1

    # 2. 多频率组合，强调低频
    print("生成多频率组合轨迹...")
    trajectory2 = np.zeros_like(t)
    frequencies = [0.05, 0.1, 0.2, 0.3, 0.5]
    amplitudes = [amplitude, amplitude*0.7, amplitude*0.5, amplitude*0.3, amplitude*0.2]

    for freq, amp in zip(frequencies, amplitudes):
        trajectory2 += amp * np.sin(2 * np.pi * freq * t)

    trajectory2 += zero_position
    trajectories['multi_freq_low'] = trajectory2

    # 3. 方波近似（用多个谐波）
    print("生成方波近似轨迹...")
    trajectory3 = np.zeros_like(t)
    # 使用多个奇次谐波模拟方波
    for k in range(1, 8, 2):  # 1, 3, 5, 7次谐波
        harmonic_amp = amplitude * 4 / (np.pi * k)
        trajectory3 += harmonic_amp * np.sin(2 * np.pi * k * 0.2 * t)

    trajectory3 += zero_position
    trajectories['square_wave_approx'] = trajectory3

    # 4. 三角波近似
    print("生成三角波轨迹...")
    trajectory4 = np.zeros_like(t)
    # 使用多个谐波模拟三角波
    for k in range(1, 6):
        harmonic_amp = amplitude * 8 / (np.pi**2 * k**2) * (-1)**((k-1)//2)
        trajectory4 += harmonic_amp * np.sin(2 * np.pi * k * 0.15 * t)

    trajectory4 += zero_position
    trajectories['triangle_wave'] = trajectory4

    # 5. 在0附近的随机游走
    print("生成随机游走轨迹...")
    trajectory5 = np.zeros_like(t)
    current_pos = zero_position

    # 生成随机步长
    dt = t[1] - t[0]
    max_step = amplitude * 0.02  # 减小每步最大变化，避免超速

    for i in range(1, len(t)):
        # 生成随机步长
        step = np.random.uniform(-max_step, max_step)
        current_pos += step

        # 确保不超出范围
        current_pos = np.clip(current_pos, zero_position - amplitude, zero_position + amplitude)
        trajectory5[i] = current_pos

    trajectories['random_walk'] = trajectory5

    # 为其他关节生成简单的轨迹（保持静止或小幅度运动）
    print("为其他关节生成简单轨迹...")
    other_joints_trajectories = []

    for joint_idx in range(generator.num_joints):
        if joint_idx != joint3_idx:
            # 其他关节保持接近初始位置的小幅度运动
            safe_limits = generator.safe_joint_limits[joint_idx]
            center = (safe_limits['lower'] + safe_limits['upper']) / 2
            small_amplitude = 0.05  # 很小的振幅

            joint_trajectory = center + small_amplitude * np.sin(2 * np.pi * 0.05 * t)
            other_joints_trajectories.append(joint_trajectory)
        else:
            other_joints_trajectories.append(None)  # 占位

    # 保存轨迹文件
    output_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control"

    for traj_name, joint3_trajectory in trajectories.items():
        # 构建完整的6关节轨迹
        all_positions = np.zeros((len(t), generator.num_joints))

        for joint_idx in range(generator.num_joints):
            if joint_idx == joint3_idx:
                all_positions[:, joint_idx] = joint3_trajectory
            else:
                all_positions[:, joint_idx] = other_joints_trajectories[joint_idx]

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

        # 创建输出数据
        output_data = {
            'time': t.tolist(),
            'positions': all_positions.tolist(),
            'velocities': velocities.tolist(),
            'accelerations': accelerations.tolist(),
            'trajectory_info': {
                'type': traj_name,
                'joint3_center': zero_position,
                'joint3_amplitude': amplitude,
                'joint3_max_velocity': np.max(np.abs(velocities[:, joint3_idx])),
                'duration': duration,
                'sampling_rate': sampling_rate
            }
        }

        # 保存文件
        filename = f"trajectory_joint3_zero_{traj_name}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"已保存: {filename}")
        print(f"  Joint3位置范围: [{np.min(joint3_trajectory):.3f}, {np.max(joint3_trajectory):.3f}] rad")
        print(f"  Joint3最大速度: {np.max(np.abs(velocities[:, joint3_idx])):.3f} rad/s")
        print(f"  Joint3速度利用率: {(np.max(np.abs(velocities[:, joint3_idx])) / max_velocity) * 100:.1f}%")

    print(f"\n=== Joint3 0附近活动轨迹生成完成 ===")
    print(f"共生成了 {len(trajectories)} 种专门针对Joint3的轨迹")
    print(f"所有轨迹都以0为中心，振幅为 {amplitude:.3f} rad")

    return trajectories

if __name__ == "__main__":
    np.random.seed(42)  # 确保可重复性
    generate_joint3_zero_centered_trajectory()