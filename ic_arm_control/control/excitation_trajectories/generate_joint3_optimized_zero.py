#!/usr/bin/env python3
"""
为Joint3生成优化的0附近活动轨迹，在安全范围内最大化激励
"""

import numpy as np
import json
import os
from excitation_trajectory_generator import ExcitationTrajectoryGenerator

def generate_joint3_optimized_zero_trajectory():
    """为Joint3生成优化的0附近轨迹"""

    print("=== 为Joint3生成优化的0附近活动轨迹 ===")

    # 初始化轨迹生成器
    urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/urdfs/ic_arm_8_dof/urdf/ic_arm_8_dof.urdf"
    generator = ExcitationTrajectoryGenerator(urdf_path, joint_margin=0.05)

    # Joint3信息
    joint3_idx = 2
    joint3_info = generator.joint_info[joint3_idx]
    joint3_safe_limits = generator.safe_joint_limits[joint3_idx]

    print(f"Joint3原始范围: [{joint3_info['lower']:.3f}, {joint3_info['upper']:.3f}] rad")
    print(f"Joint3安全范围: [{joint3_safe_limits['lower']:.3f}, {joint3_safe_limits['upper']:.3f}] rad")
    print(f"Joint3速度限制: {joint3_info['velocity']:.3f} rad/s")

    # 计算以0为中心的最大可能振幅
    zero_position = 0.0
    max_amplitude_positive = joint3_safe_limits['upper'] - zero_position
    max_amplitude_negative = zero_position - joint3_safe_limits['lower']
    max_amplitude = min(max_amplitude_positive, max_amplitude_negative)

    print(f"理论最大振幅: {max_amplitude:.3f} rad")

    duration = 25.0  # 更长的持续时间
    sampling_rate = 200
    t = np.linspace(0, duration, int(sampling_rate * duration))

    trajectories = {}

    # 1. 大振幅低频正弦波 (最大化位置范围)
    print("生成大振幅低频正弦波...")
    freq1 = 0.05  # 很低的频率
    amplitude1 = min(max_amplitude * 0.9, 0.8)  # 使用90%的最大振幅
    trajectory1 = zero_position + amplitude1 * np.sin(2 * np.pi * freq1 * t)
    trajectories['large_amplitude_sine'] = trajectory1

    # 2. 非常低频的多频率组合
    print("生成超低频多频率组合...")
    amplitude2 = min(max_amplitude * 0.8, 0.7)
    frequencies = [0.02, 0.05, 0.1, 0.15, 0.2]  # 超低频
    trajectory2 = np.zeros_like(t)

    for i, freq in enumerate(frequencies):
        # 使用递减的振幅
        harmonic_amp = amplitude2 * (0.6 ** i)
        trajectory2 += harmonic_amp * np.sin(2 * np.pi * freq * t)

    trajectory2 += zero_position
    trajectories['ultra_low_freq_multi'] = trajectory2

    # 3. 分段恒定速度轨迹 (在正负振幅之间切换)
    print("生成分段恒定速度轨迹...")
    amplitude3 = min(max_amplitude * 0.85, 0.75)
    max_velocity = joint3_info['velocity']
    segment_duration = 2.0  # 每段2秒
    trajectory3 = np.zeros_like(t)

    current_pos = zero_position
    current_velocity = max_velocity * 0.8  # 使用80%的最大速度
    direction = 1

    for i, time_val in enumerate(t):
        segment_time = time_val % segment_duration

        if segment_time < segment_duration / 2:
            # 向正方向移动
            if current_pos < amplitude3:
                current_pos += current_velocity * (1/sampling_rate)
                current_pos = min(current_pos, amplitude3)
        else:
            # 向负方向移动
            if current_pos > -amplitude3:
                current_pos -= current_velocity * (1/sampling_rate)
                current_pos = max(current_pos, -amplitude3)

        trajectory3[i] = current_pos

    trajectories['piecewise_constant'] = trajectory3

    # 4. 梯形波 (在0附近有更多停留)
    print("生成梯形波轨迹...")
    amplitude4 = min(max_amplitude * 0.8, 0.7)
    trajectory4 = np.zeros_like(t)
    period = 8.0  # 8秒周期

    for i, time_val in enumerate(t):
        phase = (time_val % period) / period * 2 * np.pi

        if phase < np.pi/2:
            # 上升段
            pos = amplitude4 * (2 * phase / np.pi)
        elif phase < 3*np.pi/2:
            # 平顶段
            pos = amplitude4
        else:
            # 下降段
            pos = amplitude4 * (2 - 2 * phase / np.pi)

        trajectory4[i] = pos

    trajectories['trapezoidal_wave'] = trajectory4

    # 5. 频率渐变正弦波
    print("生成频率渐变轨迹...")
    amplitude5 = min(max_amplitude * 0.75, 0.65)
    trajectory5 = np.zeros_like(t)

    for i, time_val in enumerate(t):
        # 频率从0.02Hz渐变到0.5Hz再回到0.02Hz
        freq_cycle = 20.0  # 20秒完成一个频率周期
        freq_normalized = (time_val % freq_cycle) / freq_cycle
        if freq_normalized < 0.5:
            freq = 0.02 + (0.5 - 0.02) * (2 * freq_normalized)
        else:
            freq = 0.5 - (0.5 - 0.02) * (2 * (freq_normalized - 0.5))

        trajectory5[i] = zero_position + amplitude5 * np.sin(2 * np.pi * freq * time_val)

    trajectories['frequency_sweep'] = trajectory5

    # 为其他关节生成静止轨迹
    print("为其他关节生成静止轨迹...")
    other_joints_trajectories = []

    for joint_idx in range(generator.num_joints):
        if joint_idx != joint3_idx:
            # 其他关节保持初始位置
            safe_limits = generator.safe_joint_limits[joint_idx]
            center = (safe_limits['lower'] + safe_limits['upper']) / 2
            joint_trajectory = np.full_like(t, center)
            other_joints_trajectories.append(joint_trajectory)
        else:
            other_joints_trajectories.append(None)

    # 保存轨迹文件并检查速度
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

        # 检查Joint3的速度
        joint3_max_velocity = np.max(np.abs(velocities[:, joint3_idx]))
        joint3_vel_utilization = (joint3_max_velocity / joint3_info['velocity']) * 100

        # 创建输出数据
        output_data = {
            'time': t.tolist(),
            'positions': all_positions.tolist(),
            'velocities': velocities.tolist(),
            'accelerations': accelerations.tolist(),
            'trajectory_info': {
                'type': traj_name,
                'joint3_center': zero_position,
                'joint3_amplitude': amplitude1 if traj_name == 'large_amplitude_sine' else amplitude2 if traj_name == 'ultra_low_freq_multi' else amplitude3 if traj_name == 'piecewise_constant' else amplitude4 if traj_name == 'trapezoidal_wave' else amplitude5,
                'joint3_max_velocity': joint3_max_velocity,
                'joint3_vel_utilization': joint3_vel_utilization,
                'joint3_pos_range': [np.min(joint3_trajectory), np.max(joint3_trajectory)],
                'duration': duration,
                'sampling_rate': sampling_rate
            }
        }

        # 保存文件
        filename = f"trajectory_joint3_zero_opt_{traj_name}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"已保存: {filename}")
        print(f"  Joint3位置范围: [{np.min(joint3_trajectory):.3f}, {np.max(joint3_trajectory):.3f}] rad")
        print(f"  活动范围: {(np.max(joint3_trajectory) - np.min(joint3_trajectory)):.3f} rad")
        print(f"  最大速度: {joint3_max_velocity:.3f} rad/s")
        print(f"  速度利用率: {joint3_vel_utilization:.1f}%")

        # 安全检查
        if joint3_vel_utilization > 100:
            print(f"  ⚠️  超速警告!")
        elif joint3_vel_utilization > 90:
            print(f"  ⚠️  接近速度极限")
        elif joint3_vel_utilization > 60:
            print(f"  ✅ 速度利用良好")
        else:
            print(f"  ⚠️  速度利用率偏低")

    print(f"\n=== Joint3优化0附近活动轨迹生成完成 ===")
    print(f"共生成了 {len(trajectories)} 种专门针对Joint3的轨迹")
    print(f"所有轨迹都以0为中心，最大化了在0附近的活动范围")

    return trajectories

if __name__ == "__main__":
    generate_joint3_optimized_zero_trajectory()