#!/usr/bin/env python3
"""
生成傅立叶相关轨迹并保存为JSON文件
"""

import numpy as np
import json
import os
from excitation_trajectory_generator import ExcitationTrajectoryGenerator

def generate_fourier_trajectories():
    """生成所有傅立叶相关的轨迹"""

    # 初始化轨迹生成器
    urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/urdfs/ic_arm_8_dof/urdf/ic_arm_8_dof.urdf"
    generator = ExcitationTrajectoryGenerator(urdf_path, joint_margin=0.05)  # 减少余量到5%

    # 为每个关节生成参数
    params = []
    for joint_idx in range(generator.num_joints):
        param = generator._generate_initial_params(joint_idx, 5)  # 5个谐波
        params.append(param)

    # 生成所有傅立叶相关轨迹
    trajectories = {}

    # 1. 多频率轨迹
    print("生成多频率轨迹...")
    freq = 0.5
    duration = 15.0
    sampling_rate = 200
    t = np.linspace(0, duration, int(sampling_rate * duration))

    # 对每个关节生成轨迹
    all_joints_data = []
    for joint_idx in range(generator.num_joints):
        # 生成低频多频率，避免超过速度限制
        frequencies = [0.15, 0.3, 0.45, 0.6, 0.75]  # 降低频率确保安全
        trajectory = generator._multi_frequency_trajectory(t, params[joint_idx], frequencies, duration)

        joint_data = {
            'joint_id': joint_idx + 1,
            'time_points': t.tolist(),
            'positions': trajectory.tolist(),
            'type': 'multi_frequency',
            'frequencies': frequencies,
            'duration': duration,
            'amplitude': generator.safe_joint_limits[joint_idx]['range'] / 2
        }
        all_joints_data.append(joint_data)

    trajectories['multi_frequency'] = all_joints_data

    # 2. Chirp轨迹
    print("生成Chirp轨迹...")
    all_joints_data = []
    for joint_idx in range(generator.num_joints):
        # 保守的扫频范围，确保安全
        trajectory = generator._swept_sine_trajectory(t, params[joint_idx], 0.05, 0.8, duration)

        joint_data = {
            'joint_id': joint_idx + 1,
            'time_points': t.tolist(),
            'positions': trajectory.tolist(),
            'type': 'chirp',
            'f_start': 0.1,
            'f_end': 5.0,
            'duration': duration,
            'amplitude': generator.safe_joint_limits[joint_idx]['range'] / 2
        }
        all_joints_data.append(joint_data)

    trajectories['chirp'] = all_joints_data

    # 3. Schroeder轨迹
    print("生成Schroeder轨迹...")
    all_joints_data = []
    for joint_idx in range(generator.num_joints):
        trajectory = generator._schroeder_trajectory(t, params[joint_idx], 10, duration)

        joint_data = {
            'joint_id': joint_idx + 1,
            'time_points': t.tolist(),
            'positions': trajectory.tolist(),
            'type': 'schroeder',
            'num_harmonics': 10,
            'duration': duration,
            'amplitude': generator.safe_joint_limits[joint_idx]['range'] / 2
        }
        all_joints_data.append(joint_data)

    trajectories['schroeder'] = all_joints_data

    # 4. 伪随机轨迹
    print("生成伪随机轨迹...")
    all_joints_data = []
    for joint_idx in range(generator.num_joints):
        # 使用8个谐波
        num_harmonics = 8
        trajectory = generator._pseudo_random_trajectory(t, params[joint_idx], num_harmonics, duration)

        joint_data = {
            'joint_id': joint_idx + 1,
            'time_points': t.tolist(),
            'positions': trajectory.tolist(),
            'type': 'pseudo_random',
            'num_harmonics': num_harmonics,
            'duration': duration,
            'amplitude': generator.safe_joint_limits[joint_idx]['range'] / 2
        }
        all_joints_data.append(joint_data)

    trajectories['pseudo_random'] = all_joints_data

    # 5. 相位调制轨迹
    print("生成相位调制轨迹...")
    all_joints_data = []
    for joint_idx in range(generator.num_joints):
        trajectory = generator._phase_modulated_trajectory(t, params[joint_idx], 1.0, 2.0, duration)

        joint_data = {
            'joint_id': joint_idx + 1,
            'time_points': t.tolist(),
            'positions': trajectory.tolist(),
            'type': 'phase_modulated',
            'carrier_freq': 1.0,
            'modulation_freq': 2.0,
            'duration': duration,
            'amplitude': generator.safe_joint_limits[joint_idx]['range'] / 2
        }
        all_joints_data.append(joint_data)

    trajectories['phase_modulated'] = all_joints_data

    # 6. 正弦和轨迹
    print("生成正弦和轨迹...")
    all_joints_data = []
    for joint_idx in range(generator.num_joints):
        # 使用3个低频正弦分量
        num_components = 3
        trajectory = generator._sum_of_sines_trajectory(t, params[joint_idx], num_components, duration)

        joint_data = {
            'joint_id': joint_idx + 1,
            'time_points': t.tolist(),
            'positions': trajectory.tolist(),
            'type': 'sum_of_sines',
            'num_components': num_components,
            'duration': duration,
            'amplitude': generator.safe_joint_limits[joint_idx]['range'] / 2
        }
        all_joints_data.append(joint_data)

    trajectories['sum_of_sines'] = all_joints_data

    # 保存轨迹文件
    output_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control"

    for trajectory_type, joint_data in trajectories.items():
        filename = f"trajectory_fourier_{trajectory_type}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'trajectory_type': trajectory_type,
                'num_joints': generator.num_joints,
                'sampling_rate': sampling_rate,
                'duration': duration,
                'joint_data': joint_data
            }, f, indent=2, ensure_ascii=False)

        print(f"已保存: {filepath}")
        print(f"  类型: {trajectory_type}")
        print(f"  关节数: {generator.num_joints}")
        print(f"  采样点数: {len(t)}")
        print(f"  持续时间: {duration}s")

    print(f"\n=== 所有傅立叶轨迹生成完成 ===")
    print(f"共生成了 {len(trajectories)} 种轨迹类型")
    print(f"每种轨迹包含 {generator.num_joints} 个关节的数据")

    return trajectories

if __name__ == "__main__":
    generate_fourier_trajectories()