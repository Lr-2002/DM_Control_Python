#!/usr/bin/env python3
"""
专门为Joint3生成的激励轨迹
基于常用角度范围 [-20°, 90°]，重点测试Joint3的动力学特性
"""

import numpy as np
import json
import os
from excitation_trajectory_generator import ExcitationTrajectoryGenerator

def generate_joint3_trajectories():
    """生成专门针对Joint3的激励轨迹"""

    print("=== Joint3专用激励轨迹生成 ===")

    # Joint3的常用角度范围：[-20°, 90°]
    joint3_common_range_deg = [-20, 90]
    joint3_common_range_rad = [np.deg2rad(angle) for angle in joint3_common_range_deg]
    joint3_center = np.mean(joint3_common_range_rad)
    joint3_amplitude = (joint3_common_range_rad[1] - joint3_common_range_rad[0]) * 0.8  # 使用80%的范围

    print(f"Joint3常用角度范围: {joint3_common_range_deg}° = [{joint3_common_range_rad[0]:.3f}, {joint3_common_range_rad[1]:.3f}] rad")
    print(f"中心位置: {joint3_center:.3f} rad ({np.rad2deg(joint3_center):.1f}°)")
    print(f"最大幅值: {joint3_amplitude:.3f} rad ({np.rad2deg(joint3_amplitude):.1f}°)")

    # 初始化轨迹生成器
    urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/urdfs/ic_arm_8_dof/urdf/ic_arm_8_dof.urdf"
    generator = ExcitationTrajectoryGenerator(urdf_path, joint_margin=0.05)

    # 获取Joint3的安全限制
    joint3_idx = 2  # Joint3是第3个关节（0-based索引）
    joint3_safe_limits = generator.safe_joint_limits[joint3_idx]
    joint3_velocity_limit = generator.joint_info[joint3_idx]['velocity']

    print(f"Joint3安全限制: [{joint3_safe_limits['lower']:.3f}, {joint3_safe_limits['upper']:.3f}] rad")
    print(f"Joint3速度限制: {joint3_velocity_limit} rad/s")

    # 轨迹参数
    duration = 30.0
    sampling_rate = 200
    t = np.linspace(0, duration, int(sampling_rate * duration))

    # 生成多种轨迹类型，特别针对Joint3
    trajectory_types = {
        'joint3_full_range': 'Joint3全范围运动',
        'joint3_center_bias': 'Joint3中心偏置运动',
        'joint3_multi_freq': 'Joint3多频率组合',
        'joint3_low_freq_gravity': 'Joint3低频重力辨识',
        'joint3_high_freq_inertia': 'Joint3高频惯性辨识',
        'joint3_friction_sweep': 'Joint3摩擦扫频',
        'joint3_asymmetric_motion': 'Joint3非对称运动',
        'joint3_sinusoidal_patterns': 'Joint3正弦模式组合',
        'joint3_random_excitation': 'Joint3随机激励',
        'joint3_chirp_signal': 'Joint3扫频信号'
    }

    trajectories = {}

    for traj_type, description in trajectory_types.items():
        print(f"\n生成{description}轨迹...")

        # 为所有关节生成轨迹，但重点设计Joint3
        all_positions = np.zeros((len(t), generator.num_joints))

        for joint_idx in range(generator.num_joints):
            joint_info = generator.joint_info[joint_idx]
            safe_limits = generator.safe_joint_limits[joint_idx]

            if joint_idx == 2:  # Joint3 - 重点设计
                trajectory = _generate_joint3_trajectory(t, traj_type, joint3_center, joint3_amplitude, joint3_velocity_limit)
            else:  # 其他关节 - 保持小幅度运动
                other_center = (safe_limits['lower'] + safe_limits['upper']) / 2
                other_amplitude = (safe_limits['upper'] - safe_limits['lower']) * 0.1  # 10%的小幅度
                trajectory = other_center + other_amplitude * 0.1 * np.sin(2 * np.pi * 0.1 * t)

            # 确保在安全范围内
            trajectory = np.clip(trajectory, safe_limits['lower'], safe_limits['upper'])
            all_positions[:, joint_idx] = trajectory

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
        joint3_max_vel = np.max(np.abs(velocities[:, 2]))
        joint3_vel_util = (joint3_max_vel / joint3_velocity_limit) * 100

        # 创建输出数据
        output_data = {
            'time': t.tolist(),
            'positions': all_positions.tolist(),
            'velocities': velocities.tolist(),
            'accelerations': accelerations.tolist(),
            'trajectory_info': {
                'type': traj_type,
                'description': description,
                'joint3_common_range': joint3_common_range_deg,
                'duration': duration,
                'sampling_rate': sampling_rate,
                'joint3_max_velocity': joint3_max_vel,
                'joint3_velocity_utilization': joint3_vel_util,
                'focus_joint': 'Joint3'
            }
        }

        # 保存文件
        filename = f"trajectory_joint3_{traj_type}.json"
        filepath = os.path.join("/Users/lr-2002/project/instantcreation/IC_arm_control", filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"已保存: {filename}")
        print(f"  Joint3速度利用率: {joint3_vel_util:.1f}% (最大速度: {joint3_max_vel:.3f} rad/s)")

    print(f"\n=== Joint3专用轨迹生成完成 ===")
    print(f"共生成了 {len(trajectory_types)} 种Joint3专用轨迹类型")
    return trajectories

def _generate_joint3_trajectory(t, traj_type, center, amplitude, velocity_limit):
    """生成Joint3的特定轨迹"""
    sampling_rate = 200  # 采样率

    if traj_type == 'joint3_full_range':
        # 全范围运动 - 覆盖整个常用角度范围
        frequencies = [0.05, 0.1, 0.2]
        amplitudes = amplitude * np.array([0.5, 0.3, 0.2])
        trajectory = center
        for freq, amp in zip(frequencies, amplitudes):
            trajectory += amp * np.sin(2 * np.pi * freq * t)

    elif traj_type == 'joint3_center_bias':
        # 中心偏置运动 - 考虑Joint3的非对称常用范围
        # Joint3常用范围：[-20°, 90°]，中心在35°
        bias_center = np.deg2rad(35)  # 偏置中心
        frequencies = [0.03, 0.08, 0.15]
        amplitudes = amplitude * 0.6 * np.array([0.4, 0.3, 0.2])
        trajectory = bias_center
        for freq, amp in zip(frequencies, amplitudes):
            trajectory += amp * np.sin(2 * np.pi * freq * t)

    elif traj_type == 'joint3_multi_freq':
        # 多频率组合 - 宽频激励
        frequencies = np.logspace(np.log10(0.01), np.log10(2.0), 8)  # 对数间距
        amplitudes = amplitude / len(frequencies)
        trajectory = center
        for freq in frequencies:
            phase = np.random.uniform(0, 2*np.pi)
            trajectory += amplitudes * np.sin(2 * np.pi * freq * t + phase)

    elif traj_type == 'joint3_low_freq_gravity':
        # 低频重力辨识 - 非常慢的运动
        frequencies = [0.02, 0.05, 0.08]
        amplitudes = amplitude * np.array([0.6, 0.3, 0.1])
        trajectory = center
        for freq, amp in zip(frequencies, amplitudes):
            trajectory += amp * np.sin(2 * np.pi * freq * t)

    elif traj_type == 'joint3_high_freq_inertia':
        # 高频惯性辨识 - 高频小振幅
        frequencies = [1.0, 2.0, 3.0, 4.0]
        amplitudes = amplitude * 0.1 * np.array([0.4, 0.3, 0.2, 0.1])
        trajectory = center
        for freq, amp in zip(frequencies, amplitudes):
            trajectory += amp * np.sin(2 * np.pi * freq * t)

    elif traj_type == 'joint3_friction_sweep':
        # 摩擦扫频 - 速度扫频
        f_start, f_end = 0.01, 1.0
        sweep_duration = t[-1] - t[0]
        instantaneous_freq = f_start * (f_end/f_start)**(t/sweep_duration)
        phase = 2 * np.pi * f_start * sweep_duration * ((f_end/f_start)**(t/sweep_duration) - 1) / np.log(f_end/f_start)
        trajectory = center + amplitude * 0.3 * np.sin(phase)

    elif traj_type == 'joint3_asymmetric_motion':
        # 非对称运动 - 考虑Joint3的非对称常用范围
        # 向上运动（正方向）幅度更大，向下运动（负方向）幅度较小
        trajectory = center
        # 主要分量：向上大幅度运动
        trajectory += amplitude * 0.6 * np.maximum(0, np.sin(2 * np.pi * 0.1 * t))
        # 次要分量：向下小幅度运动
        trajectory += amplitude * 0.3 * np.minimum(0, np.sin(2 * np.pi * 0.1 * t + np.pi))

    elif traj_type == 'joint3_sinusoidal_patterns':
        # 正弦模式组合 - 多种正弦模式
        patterns = [
            (0.05, 0.4),   # 低频大幅
            (0.2, 0.3),    # 中频中幅
            (0.8, 0.2),    # 高频小幅
            (1.5, 0.1)     # 超高频微幅
        ]
        trajectory = center
        for freq, amp_ratio in patterns:
            trajectory += amplitude * amp_ratio * np.sin(2 * np.pi * freq * t)

    elif traj_type == 'joint3_random_excitation':
        # 随机激励 - 伪随机信号
        np.random.seed(43)  # 可重复的随机种子
        trajectory = center
        # 添加多个随机相位和频率的正弦波
        for i in range(5):
            freq = np.random.uniform(0.05, 1.0)
            phase = np.random.uniform(0, 2*np.pi)
            amp = amplitude * 0.2 * np.random.uniform(0.5, 1.0)
            trajectory += amp * np.sin(2 * np.pi * freq * t + phase)

    elif traj_type == 'joint3_chirp_signal':
        # 线性扫频信号 - 用于频率响应分析
        f0, f1 = 0.1, 2.0  # 起始和结束频率
        duration = 30.0  # 持续时间
        chirp_signal = amplitude * 0.3 * np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
        trajectory = center + chirp_signal

    else:
        # 默认轨迹
        trajectory = center + amplitude * 0.5 * np.sin(2 * np.pi * 0.1 * t)

    return trajectory

if __name__ == "__main__":
    generate_joint3_trajectories()