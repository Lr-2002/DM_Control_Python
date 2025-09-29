#!/usr/bin/env python3
"""
专门为Joint1生成的激励轨迹
基于常用角度范围 [-15°, 25°]，重点解决Joint1的激励不足问题
"""

import numpy as np
import json
import os
from excitation_trajectory_generator import ExcitationTrajectoryGenerator

def generate_joint1_trajectories():
    """生成专门针对Joint1的激励轨迹"""

    print("=== Joint1专用激励轨迹生成 ===")

    # Joint1的常用角度范围：[-15°, 25°]
    joint1_common_range_deg = [-15, 25]
    joint1_common_range_rad = [np.deg2rad(angle) for angle in joint1_common_range_deg]
    joint1_center = np.mean(joint1_common_range_rad)
    joint1_amplitude = (joint1_common_range_rad[1] - joint1_common_range_rad[0]) * 0.9  # 使用90%的范围

    print(f"Joint1常用角度范围: {joint1_common_range_deg}° = [{joint1_common_range_rad[0]:.3f}, {joint1_common_range_rad[1]:.3f}] rad")
    print(f"中心位置: {joint1_center:.3f} rad ({np.rad2deg(joint1_center):.1f}°)")
    print(f"最大幅值: {joint1_amplitude:.3f} rad ({np.rad2deg(joint1_amplitude):.1f}°)")

    # 初始化轨迹生成器
    urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/urdfs/ic_arm_8_dof/urdf/ic_arm_8_dof.urdf"
    generator = ExcitationTrajectoryGenerator(urdf_path, joint_margin=0.05)

    # 获取Joint1的安全限制
    joint1_idx = 0  # Joint1是第1个关节（0-based索引）
    joint1_safe_limits = generator.safe_joint_limits[joint1_idx]
    joint1_velocity_limit = generator.joint_info[joint1_idx]['velocity']

    print(f"Joint1安全限制: [{joint1_safe_limits['lower']:.3f}, {joint1_safe_limits['upper']:.3f}] rad")
    print(f"Joint1速度限制: {joint1_velocity_limit} rad/s")

    # 轨迹参数
    duration = 40.0  # 更长的持续时间以获得更好的数据
    sampling_rate = 200
    t = np.linspace(0, duration, int(sampling_rate * duration))

    # 计算Joint1的安全速度限制（使用80%的安全裕度）
    joint1_safe_velocity = joint1_velocity_limit * 0.8
    print(f"Joint1安全速度限制: {joint1_safe_velocity:.3f} rad/s")

    # 生成多种轨迹类型，专门针对Joint1的激励不足问题
    trajectory_types = {
        'joint1_wide_excitation': 'Joint1宽范围激励',
        'joint1_high_frequency': 'Joint1高频激励',
        'joint1_multi_sine': 'Joint1多正弦组合',
        'joint1_chirp_sweep': 'Joint1扫频信号',
        'joint1_random_walk': 'Joint1随机游走',
        'joint1_step_excitation': 'Joint1阶跃激励',
        'joint1_low_freq_gravity': 'Joint1低频重力辨识',
        'joint1_persistent_excitation': 'Joint1持续激励',
        'joint1_asymmetric_motion': 'Joint1非对称运动',
        'joint1_full_spectrum': 'Joint1全谱激励'
    }

    trajectories = {}

    for traj_type, description in trajectory_types.items():
        print(f"\n生成{description}轨迹...")

        # 为所有关节生成轨迹，但重点设计Joint1
        all_positions = np.zeros((len(t), generator.num_joints))

        for joint_idx in range(generator.num_joints):
            joint_info = generator.joint_info[joint_idx]
            safe_limits = generator.safe_joint_limits[joint_idx]

            if joint_idx == 0:  # Joint1 - 重点设计
                trajectory = _generate_joint1_trajectory(t, traj_type, joint1_center, joint1_amplitude, joint1_velocity_limit)
            else:  # 其他关节 - 保持小幅度运动
                other_center = (safe_limits['lower'] + safe_limits['upper']) / 2
                other_amplitude = (safe_limits['upper'] - safe_limits['lower']) * 0.05  # 5%的小幅度
                trajectory = other_center + other_amplitude * 0.1 * np.sin(2 * np.pi * 0.05 * t)

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
        joint1_max_vel = np.max(np.abs(velocities[:, 0]))
        joint1_vel_util = (joint1_max_vel / joint1_velocity_limit) * 100

        # 如果超过安全速度，给出警告
        if joint1_max_vel > joint1_safe_velocity:
            print(f"  ⚠️ 超过安全速度限制: {joint1_max_vel:.3f} > {joint1_safe_velocity:.3f} rad/s")

        # 创建输出数据
        output_data = {
            'time': t.tolist(),
            'positions': all_positions.tolist(),
            'velocities': velocities.tolist(),
            'accelerations': accelerations.tolist(),
            'trajectory_info': {
                'type': traj_type,
                'description': description,
                'joint1_common_range': joint1_common_range_deg,
                'duration': duration,
                'sampling_rate': sampling_rate,
                'joint1_max_velocity': joint1_max_vel,
                'joint1_velocity_utilization': joint1_vel_util,
                'focus_joint': 'Joint1',
                'purpose': '解决Joint1激励不足问题'
            }
        }

        # 保存文件
        filename = f"trajectory_joint1_{traj_type}.json"
        filepath = os.path.join("/Users/lr-2002/project/instantcreation/IC_arm_control", filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"已保存: {filename}")
        print(f"  Joint1速度利用率: {joint1_vel_util:.1f}% (最大速度: {joint1_max_vel:.3f} rad/s)")

    print(f"\n=== Joint1专用轨迹生成完成 ===")
    print(f"共生成了 {len(trajectory_types)} 种Joint1专用轨迹类型")
    return trajectories

def _generate_joint1_trajectory(t, traj_type, center, amplitude, velocity_limit):
    """生成Joint1的特定轨迹"""
    sampling_rate = 200  # 采样率

    if traj_type == 'joint1_wide_excitation':
        # 宽范围激励 - 覆盖整个常用角度范围
        frequencies = [0.02, 0.05, 0.1, 0.2]
        amplitudes = amplitude * np.array([0.4, 0.3, 0.2, 0.1])
        trajectory = center
        for freq, amp in zip(frequencies, amplitudes):
            trajectory += amp * np.sin(2 * np.pi * freq * t)

    elif traj_type == 'joint1_high_frequency':
        # 高频激励 - 专门辨识高频动力学特性
        frequencies = [0.5, 1.0, 1.5, 2.0, 2.5]
        amplitudes = amplitude * 0.2 * np.array([0.4, 0.3, 0.2, 0.1, 0.05])
        trajectory = center
        for freq, amp in zip(frequencies, amplitudes):
            phase = np.random.uniform(0, 2*np.pi)
            trajectory += amp * np.sin(2 * np.pi * freq * t + phase)

    elif traj_type == 'joint1_multi_sine':
        # 多正弦组合 - 使用Schroeder相位序列
        n_frequencies = 15
        frequencies = np.linspace(0.01, 2.0, n_frequencies)
        amplitudes = amplitude / n_frequencies * np.ones(n_frequencies)

        # Schroeder相位序列以减少峰值因子
        phases = np.zeros(n_frequencies)
        for i in range(n_frequencies):
            phases[i] = np.sum(frequencies[:i+1])

        trajectory = center
        for i, (freq, amp, phase) in enumerate(zip(frequencies, amplitudes, phases)):
            trajectory += amp * np.sin(2 * np.pi * freq * t + phase)

    elif traj_type == 'joint1_chirp_sweep':
        # 线性扫频信号 - 用于频率响应分析
        f0, f1 = 0.01, 3.0  # 更宽的频率范围
        duration = 40.0
        chirp_signal = amplitude * 0.4 * np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
        trajectory = center + chirp_signal

    elif traj_type == 'joint1_random_walk':
        # 随机游走 - 模拟实际使用情况
        np.random.seed(123)
        trajectory = center * np.ones_like(t)

        # 生成随机步长
        n_steps = 100
        step_duration = len(t) // n_steps
        for i in range(n_steps):
            start_idx = i * step_duration
            end_idx = min((i + 1) * step_duration, len(t))

            # 随机步长
            step_size = np.random.uniform(-0.3, 0.3) * amplitude
            trajectory[start_idx:end_idx] += step_size

        # 平滑处理
        from scipy import ndimage
        trajectory = ndimage.gaussian_filter1d(trajectory, sigma=2.0)

    elif traj_type == 'joint1_step_excitation':
        # 阶跃激励 - 用于辨识瞬态响应
        trajectory = center * np.ones_like(t)

        # 添加多个阶跃输入
        step_times = [5, 10, 15, 20, 25, 30, 35]
        step_amplitudes = [0.5, -0.3, 0.4, -0.6, 0.3, -0.4, 0.2]

        for step_time, step_amp in zip(step_times, step_amplitudes):
            step_idx = int(step_time * sampling_rate)
            if step_idx < len(t):
                trajectory[step_idx:] += amplitude * step_amp

    elif traj_type == 'joint1_low_freq_gravity':
        # 低频重力辨识 - 非常慢的运动
        frequencies = [0.005, 0.01, 0.02]
        amplitudes = amplitude * np.array([0.5, 0.3, 0.2])
        trajectory = center
        for freq, amp in zip(frequencies, amplitudes):
            trajectory += amp * np.sin(2 * np.pi * freq * t)

    elif traj_type == 'joint1_persistent_excitation':
        # 持续激励 - 保证持续激励条件
        n_components = 20
        frequencies = np.logspace(np.log10(0.01), np.log10(2.0), n_components)

        # 使用随机相位
        np.random.seed(42)
        phases = np.random.uniform(0, 2*np.pi, n_components)

        trajectory = center
        for i, freq in enumerate(frequencies):
            amp = amplitude / n_components
            trajectory += amp * np.sin(2 * np.pi * freq * t + phases[i])

    elif traj_type == 'joint1_asymmetric_motion':
        # 非对称运动 - 模拟实际工作负载
        trajectory = center

        # 慢速正向运动 + 快速负向运动
        slow_freq = 0.05
        fast_freq = 0.15

        # 正向运动（慢速大幅）
        trajectory += amplitude * 0.4 * np.maximum(0, np.sin(2 * np.pi * slow_freq * t))
        # 负向运动（快速小幅）
        trajectory += amplitude * 0.2 * np.minimum(0, np.sin(2 * np.pi * fast_freq * t + np.pi))

    elif traj_type == 'joint1_full_spectrum':
        # 全谱激励 - 覆盖所有重要频率
        # 使用多个频率分量
        freq_bands = [
            (0.01, 0.1, 5),      # 超低频
            (0.1, 0.5, 8),       # 低频
            (0.5, 1.5, 10),      # 中频
            (1.5, 3.0, 7)        # 高频
        ]

        trajectory = center
        total_amp_ratio = 0

        for f_min, f_max, n_freq in freq_bands:
            frequencies = np.linspace(f_min, f_max, n_freq)
            amp_ratio = 0.2  # 每个频段分配20%的幅值
            total_amp_ratio += amp_ratio

            for freq in frequencies:
                phase = np.random.uniform(0, 2*np.pi)
                trajectory += amplitude * (amp_ratio / n_freq) * np.sin(2 * np.pi * freq * t + phase)

    else:
        # 默认轨迹
        trajectory = center + amplitude * 0.5 * np.sin(2 * np.pi * 0.1 * t)

    return trajectory

if __name__ == "__main__":
    generate_joint1_trajectories()