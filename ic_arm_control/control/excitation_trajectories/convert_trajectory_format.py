#!/usr/bin/env python3
"""
将生成的傅立叶轨迹格式转换为轨迹执行器期望的格式
"""

import numpy as np
import json
import os
from typing import Dict, List

def convert_trajectory_format(input_file: str, output_file: str) -> None:
    """
    转换轨迹文件格式

    输入格式:
    {
        "trajectory_type": "multi_frequency",
        "num_joints": 6,
        "sampling_rate": 200,
        "duration": 15.0,
        "joint_data": [
            {
                "joint_id": 1,
                "time_points": [...],
                "positions": [...],
                ...
            }
        ]
    }

    输出格式:
    {
        "time": [...],
        "positions": [[...], [...], ...],
        "velocities": [[...], [...], ...],
        "accelerations": [[...], [...], ...]
    }
    """

    print(f"转换轨迹文件: {input_file} -> {output_file}")

    # 加载输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    # 获取基本信息
    num_joints = input_data['num_joints']
    duration = input_data['duration']
    sampling_rate = input_data['sampling_rate']
    joint_data = input_data['joint_data']

    # 计算时间点
    num_points = len(joint_data[0]['time_points'])
    time_array = np.array(joint_data[0]['time_points'])

    # 创建位置矩阵 (num_points x num_joints)
    positions_matrix = np.zeros((num_points, num_joints))

    # 填充位置数据
    for joint_info in joint_data:
        joint_id = joint_info['joint_id'] - 1  # 转换为0-based索引
        positions = np.array(joint_info['positions'])
        positions_matrix[:, joint_id] = positions

    # 计算速度和加速度
    dt = 1.0 / sampling_rate
    velocities_matrix = np.zeros_like(positions_matrix)
    accelerations_matrix = np.zeros_like(positions_matrix)

    # 使用中心差分计算速度和加速度
    for i in range(num_joints):
        # 速度计算 (中心差分)
        velocities_matrix[1:-1, i] = (positions_matrix[2:, i] - positions_matrix[:-2, i]) / (2 * dt)
        velocities_matrix[0, i] = (positions_matrix[1, i] - positions_matrix[0, i]) / dt  # 前向差分
        velocities_matrix[-1, i] = (positions_matrix[-1, i] - positions_matrix[-2, i]) / dt  # 后向差分

        # 加速度计算 (中心差分)
        accelerations_matrix[1:-1, i] = (velocities_matrix[2:, i] - velocities_matrix[:-2, i]) / (2 * dt)
        accelerations_matrix[0, i] = (velocities_matrix[1, i] - velocities_matrix[0, i]) / dt  # 前向差分
        accelerations_matrix[-1, i] = (velocities_matrix[-1, i] - velocities_matrix[-2, i]) / dt  # 后向差分

    # 创建输出数据结构
    output_data = {
        'time': time_array.tolist(),
        'positions': positions_matrix.tolist(),
        'velocities': velocities_matrix.tolist(),
        'accelerations': accelerations_matrix.tolist()
    }

    # 保存输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"转换完成!")
    print(f"  数据点数: {num_points}")
    print(f"  关节数: {num_joints}")
    print(f"  持续时间: {duration}s")
    print(f"  采样率: {sampling_rate}Hz")
    print(f"  位置范围: [{positions_matrix.min():.3f}, {positions_matrix.max():.3f}] rad")
    print(f"  速度范围: [{velocities_matrix.min():.3f}, {velocities_matrix.max():.3f}] rad/s")
    print(f"  加速度范围: [{accelerations_matrix.min():.3f}, {accelerations_matrix.max():.3f}] rad/s²")

def convert_all_fourier_trajectories():
    """转换所有傅立叶轨迹文件"""

    # 输入文件目录
    input_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control"
    output_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control"

    # 傅立叶轨迹文件列表
    fourier_files = [
        "trajectory_fourier_multi_frequency.json",
        "trajectory_fourier_chirp.json",
        "trajectory_fourier_schroeder.json",
        "trajectory_fourier_pseudo_random.json",
        "trajectory_fourier_phase_modulated.json",
        "trajectory_fourier_sum_of_sines.json"
    ]

    print("=== 转换所有傅立叶轨迹文件 ===")

    for input_file in fourier_files:
        input_path = os.path.join(input_dir, input_file)

        if os.path.exists(input_path):
            # 生成输出文件名（去掉"fourier_"前缀）
            output_filename = input_file.replace("fourier_", "")
            output_path = os.path.join(output_dir, output_filename)

            try:
                convert_trajectory_format(input_path, output_path)
            except Exception as e:
                print(f"转换失败 {input_file}: {e}")
        else:
            print(f"文件不存在: {input_path}")

    print("\n=== 所有轨迹转换完成 ===")

if __name__ == "__main__":
    convert_all_fourier_trajectories()