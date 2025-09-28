#!/usr/bin/env python3
"""
分析基于常用角度的轨迹质量
"""

import numpy as np
import json
import os

def analyze_common_angles_trajectories():
    """分析常用角度轨迹"""

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

    print("=== 基于常用角度的轨迹分析 ===")
    print("各关节常用角度范围:")
    for joint_id, angle_range in common_angles_rad.items():
        center_deg = np.rad2deg((angle_range[0] + angle_range[1]) / 2)
        range_deg = np.rad2deg(angle_range[1] - angle_range[0])
        print(f"  Joint{joint_id}: {center_deg:.0f}° ± {range_deg/2:.0f}°")

    print()

    # 查找轨迹文件
    trajectory_files = []
    for filename in os.listdir("/Users/lr-2002/project/instantcreation/IC_arm_control"):
        if filename.startswith("trajectory_common_angles_") and filename.endswith(".json"):
            trajectory_files.append(filename)

    max_velocity_limit = 1.57  # rad/s

    safe_trajectories = []

    for filename in trajectory_files:
        filepath = f"/Users/lr-2002/project/instantcreation/IC_arm_control/{filename}"

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            positions = np.array(data['positions'])
            velocities = np.array(data['velocities'])

            # 检查每个关节
            all_safe = True
            joint_stats = []

            for joint_idx in range(6):
                joint_id = joint_idx + 1
                joint_positions = positions[:, joint_idx]
                joint_velocities = velocities[:, joint_idx]

                # 计算统计信息
                pos_range = np.max(joint_positions) - np.min(joint_positions)
                pos_mean = np.mean(joint_positions)
                vel_max = np.max(np.abs(joint_velocities))
                vel_utilization = (vel_max / max_velocity_limit) * 100

                # 检查是否在常用角度范围内
                common_range = common_angles_rad[joint_id]
                time_in_common_range = np.sum((joint_positions >= common_range[0]) &
                                            (joint_positions <= common_range[1])) / len(joint_positions) * 100

                # 计算在常用中心附近的时间
                common_center = (common_range[0] + common_range[1]) / 2
                common_amplitude = (common_range[1] - common_range[0]) / 4
                time_near_center = np.sum(np.abs(joint_positions - common_center) < common_amplitude) / len(joint_positions) * 100

                # 安全检查
                is_safe = vel_utilization <= 90
                if not is_safe:
                    all_safe = False

                joint_stats.append({
                    'joint_id': joint_id,
                    'pos_range_deg': np.rad2deg(pos_range),
                    'pos_mean_deg': np.rad2deg(pos_mean),
                    'vel_max': vel_max,
                    'vel_utilization': vel_utilization,
                    'time_in_common_range': time_in_common_range,
                    'time_near_center': time_near_center,
                    'is_safe': is_safe
                })

            if all_safe:
                safe_trajectories.append({
                    'filename': filename,
                    'joint_stats': joint_stats
                })

            print(f"{filename}:")
            safety_status = "✅ 全部安全" if all_safe else "⚠️  有关节超速"
            print(f"  安全状态: {safety_status}")

            for stat in joint_stats:
                status_icon = "✅" if stat['is_safe'] else "❌"
                center_deg = np.rad2deg((common_angles_rad[stat['joint_id']][0] + common_angles_rad[stat['joint_id']][1]) / 2)
                print(f"  Joint{stat['joint_id']}: {status_icon} {stat['vel_utilization']:.1f}% "
                      f"(范围: {stat['pos_range_deg']:.1f}°, 常用范围内: {stat['time_in_common_range']:.1f}%, "
                      f"中心附近: {stat['time_near_center']:.1f}%)")
            print()

        except Exception as e:
            print(f"❌ 分析 {filename} 时出错: {e}")

    # 推荐最佳轨迹
    print("=== 推荐轨迹 ===")
    if safe_trajectories:
        print("🎯 安全且在常用角度范围内的轨迹:")

        # 计算综合评分
        for traj in safe_trajectories:
            # 评分标准：常用范围内时间 + 中心附近时间 + 速度利用率(不超过90%)
            total_score = 0
            for stat in traj['joint_stats']:
                score = (stat['time_in_common_range'] + stat['time_near_center']) * min(stat['vel_utilization']/90, 1.0)
                total_score += score

            traj['total_score'] = total_score / 6  # 平均分

        # 按评分排序
        safe_trajectories.sort(key=lambda x: x['total_score'], reverse=True)

        for i, traj in enumerate(safe_trajectories[:2]):  # 显示前2个
            print(f"{i+1}. {traj['filename']}")
            print(f"   综合评分: {traj['total_score']:.1f}")
            for stat in traj['joint_stats']:
                center_deg = np.rad2deg((common_angles_rad[stat['joint_id']][0] + common_angles_rad[stat['joint_id']][1]) / 2)
                print(f"   Joint{stat['joint_id']} ({center_deg:.0f}°): "
                      f"范围{stat['pos_range_deg']:.1f}°, 常用内{stat['time_in_common_range']:.1f}%, "
                      f"速度{stat['vel_utilization']:.1f}%")
            print()

    return safe_trajectories

if __name__ == "__main__":
    analyze_common_angles_trajectories()