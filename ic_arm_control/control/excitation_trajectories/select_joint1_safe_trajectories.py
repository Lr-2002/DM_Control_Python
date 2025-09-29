#!/usr/bin/env python3
"""
选择和优化Joint1的安全轨迹
"""

import numpy as np
import json
import os
import glob

def analyze_and_select_joint1_trajectories():
    """分析并选择Joint1的安全轨迹"""

    print("=== Joint1安全轨迹分析 ===")

    # 查找所有Joint1轨迹文件
    trajectory_files = glob.glob("/Users/lr-2002/project/instantcreation/IC_arm_control/trajectory_joint1_*.json")

    if not trajectory_files:
        print("未找到Joint1轨迹文件")
        return []

    print(f"找到 {len(trajectory_files)} 个Joint1轨迹文件")

    # 分析每个轨迹
    safe_trajectories = []
    unsafe_trajectories = []

    joint1_velocity_limit = 3.14  # rad/s
    joint1_safe_velocity = joint1_velocity_limit * 0.8  # 80%安全裕度

    for file_path in trajectory_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            trajectory_info = data['trajectory_info']
            max_velocity = trajectory_info['joint1_max_velocity']
            vel_utilization = trajectory_info['joint1_velocity_utilization']

            # 检查是否安全
            is_safe = max_velocity <= joint1_safe_velocity

            if is_safe:
                safe_trajectories.append({
                    'file': os.path.basename(file_path),
                    'max_velocity': max_velocity,
                    'vel_utilization': vel_utilization,
                    'description': trajectory_info['description']
                })
            else:
                unsafe_trajectories.append({
                    'file': os.path.basename(file_path),
                    'max_velocity': max_velocity,
                    'vel_utilization': vel_utilization,
                    'description': trajectory_info['description']
                })

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

    # 输出分析结果
    print(f"\n安全轨迹 ({len(safe_trajectories)} 个):")
    for traj in safe_trajectories:
        print(f"  {traj['file']}: {traj['vel_utilization']:.1f}% ({traj['max_velocity']:.3f} rad/s) - {traj['description']}")

    print(f"\n不安全轨迹 ({len(unsafe_trajectories)} 个):")
    for traj in unsafe_trajectories:
        print(f"  {traj['file']}: {traj['vel_utilization']:.1f}% ({traj['max_velocity']:.3f} rad/s) - {traj['description']}")

    # 选择最佳轨迹
    if safe_trajectories:
        # 按速度利用率排序（选择利用率较高但安全的轨迹）
        safe_trajectories.sort(key=lambda x: x['vel_utilization'], reverse=True)

        print(f"\n推荐的Joint1安全轨迹 (按速度利用率排序):")
        for i, traj in enumerate(safe_trajectories[:5]):  # 显示前5个
            print(f"  {i+1}. {traj['file']}: {traj['vel_utilization']:.1f}% - {traj['description']}")

        # 创建推荐轨迹列表
        recommended_files = [traj['file'] for traj in safe_trajectories[:5]]
        print(f"\n推荐轨迹文件: {recommended_files}")

        return recommended_files
    else:
        print("\n⚠️ 没有找到安全的轨迹，需要生成新的安全轨迹")
        return create_safe_joint1_trajectories()

def create_safe_joint1_trajectories():
    """创建安全的Joint1轨迹"""

    print("\n=== 创建安全的Joint1轨迹 ===")

    # 导入轨迹生成器
    from generate_joint1_trajectories import _generate_joint1_trajectory

    # 轨迹参数
    duration = 40.0
    sampling_rate = 200
    t = np.linspace(0, duration, int(sampling_rate * duration))

    # Joint1参数
    joint1_center = np.deg2rad(5.0)  # 5度中心
    joint1_amplitude = np.deg2rad(36.0) * 0.6  # 减小幅值
    joint1_safe_velocity = 3.14 * 0.8  # 安全速度

    # 创建安全的轨迹类型
    safe_trajectory_types = {
        'joint1_safe_wide': 'Joint1安全宽范围激励',
        'joint1_safe_multi_freq': 'Joint1安全多频激励',
        'joint1_safe_chirp': 'Joint1安全扫频信号',
        'joint1_safe_gravity': 'Joint1安全重力辨识',
        'joint1_safe_persistent': 'Joint1安全持续激励'
    }

    recommended_files = []

    for traj_type, description in safe_trajectory_types.items():
        print(f"\n生成{description}...")

        # 生成轨迹
        trajectory = _generate_joint1_trajectory(t, traj_type, joint1_center, joint1_amplitude, joint1_safe_velocity)

        # 计算速度
        dt = 1.0 / sampling_rate
        velocities = np.zeros_like(trajectory)
        velocities[1:-1] = (trajectory[2:] - trajectory[:-2]) / (2 * dt)
        velocities[0] = (trajectory[1] - trajectory[0]) / dt
        velocities[-1] = (trajectory[-1] - trajectory[-2]) / dt

        # 检查最大速度
        max_vel = np.max(np.abs(velocities))
        vel_util = (max_vel / 3.14) * 100

        # 如果仍然超过安全速度，进一步缩放
        if max_vel > joint1_safe_velocity:
            scale_factor = joint1_safe_velocity / max_vel * 0.9  # 保留10%裕度
            trajectory = joint1_center + (trajectory - joint1_center) * scale_factor
            velocities = velocities * scale_factor
            max_vel = np.max(np.abs(velocities))
            vel_util = (max_vel / 3.14) * 100

        print(f"  最终速度利用率: {vel_util:.1f}% (最大速度: {max_vel:.3f} rad/s)")

        # 创建完整的多关节数据
        all_positions = np.zeros((len(t), 6))
        all_positions[:, 0] = trajectory  # 只设置Joint1，其他关节为0

        # 计算其他关节的运动量
        for joint_idx in range(1, 6):
            # 其他关节保持非常小的运动
            other_center = 0.0
            other_amplitude = 0.01  # 很小的幅值
            all_positions[:, joint_idx] = other_center + other_amplitude * np.sin(2 * np.pi * 0.05 * t)

        # 计算所有关节的速度和加速度
        all_velocities = np.zeros_like(all_positions)
        all_accelerations = np.zeros_like(all_positions)

        for joint_idx in range(6):
            # 速度计算
            all_velocities[1:-1, joint_idx] = (all_positions[2:, joint_idx] - all_positions[:-2, joint_idx]) / (2 * dt)
            all_velocities[0, joint_idx] = (all_positions[1, joint_idx] - all_positions[0, joint_idx]) / dt
            all_velocities[-1, joint_idx] = (all_positions[-1, joint_idx] - all_positions[-2, joint_idx]) / dt

            # 加速度计算
            all_accelerations[1:-1, joint_idx] = (all_velocities[2:, joint_idx] - all_velocities[:-2, joint_idx]) / (2 * dt)
            all_accelerations[0, joint_idx] = (all_velocities[1, joint_idx] - all_velocities[0, joint_idx]) / dt
            all_accelerations[-1, joint_idx] = (all_velocities[-1, joint_idx] - all_velocities[-2, joint_idx]) / dt

        # 创建输出数据
        output_data = {
            'time': t.tolist(),
            'positions': all_positions.tolist(),
            'velocities': all_velocities.tolist(),
            'accelerations': all_accelerations.tolist(),
            'trajectory_info': {
                'type': traj_type,
                'description': description,
                'joint1_common_range': [-15, 25],
                'duration': duration,
                'sampling_rate': sampling_rate,
                'joint1_max_velocity': max_vel,
                'joint1_velocity_utilization': vel_util,
                'focus_joint': 'Joint1',
                'purpose': '安全优化的Joint1轨迹',
                'safety_factor': '80%速度限制'
            }
        }

        # 保存文件
        filename = f"trajectory_joint1_safe_{traj_type.split('_')[-1]}.json"
        filepath = os.path.join("/Users/lr-2002/project/instantcreation/IC_arm_control", filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"已保存: {filename}")
        recommended_files.append(filename)

    print(f"\n创建了 {len(recommended_files)} 个安全的Joint1轨迹")
    return recommended_files

def generate_joint1_recommendation_report():
    """生成Joint1轨迹使用建议报告"""

    print("=== Joint1轨迹使用建议 ===")

    # 获取推荐的轨迹文件
    recommended_files = analyze_and_select_joint1_trajectories()

    if not recommended_files:
        print("无法获取推荐轨迹")
        return

    print(f"\n使用建议:")
    print(f"1. 数据收集阶段:")
    print(f"   - 首先使用 'persistent' (持续激励) 轨迹收集基础数据")
    print(f"   - 然后使用 'wide' (宽范围) 轨迹收集重力辨识数据")
    print(f"   - 最后使用 'chirp' (扫频) 轨迹收集频率响应数据")

    print(f"\n2. 参数辨识阶段:")
    print(f"   - 使用多频轨迹进行系统辨识")
    print(f"   - 确保数据覆盖Joint1的整个工作范围 [-15°, 25°]")
    print(f"   - 重点关注低频特性（重力影响）")

    print(f"\n3. 验证阶段:")
    print(f"   - 使用辨识结果进行重力补偿验证")
    print(f"   - 如果R²仍然较低，考虑收集更多数据")
    print(f"   - 可以尝试组合多个轨迹的数据进行联合辨识")

    # 创建简单的使用脚本
    usage_script = f"""#!/bin/bash
# Joint1轨迹数据收集脚本

echo "开始Joint1专用数据收集..."

# 推荐的轨迹文件
TRAJECTORIES=({" ".join([f'"{f}"' for f in recommended_files[:3]])})

for traj in "${{TRAJECTORIES[@]}}"; do
    echo "执行轨迹: $traj"
    # 在这里添加实际的控制命令
    # python3 trajectory_executor.py $traj
    sleep 1
done

echo "Joint1数据收集完成"
"""

    # 保存使用脚本
    script_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/joint1_data_collection.sh"
    with open(script_path, 'w') as f:
        f.write(usage_script)

    print(f"\n4. 自动化脚本:")
    print(f"   已创建数据收集脚本: {script_path}")
    print(f"   推荐按顺序执行以下轨迹:")
    for i, filename in enumerate(recommended_files[:3], 1):
        print(f"   {i}. {filename}")

if __name__ == "__main__":
    generate_joint1_recommendation_report()