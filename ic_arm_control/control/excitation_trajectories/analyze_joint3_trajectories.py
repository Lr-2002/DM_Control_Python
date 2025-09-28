#!/usr/bin/env python3
"""
分析专门为Joint3生成的0附近活动轨迹
"""

import numpy as np
import json
import os

def analyze_joint3_trajectories():
    """分析Joint3专用轨迹"""

    # 查找所有Joint3专用轨迹文件
    trajectory_files = []
    for filename in os.listdir("/Users/lr-2002/project/instantcreation/IC_arm_control"):
        if filename.startswith("trajectory_joint3_zero") and filename.endswith(".json"):
            trajectory_files.append(filename)

    print("=== Joint3专用轨迹分析 ===")
    print(f"找到 {len(trajectory_files)} 个Joint3专用轨迹")
    print()

    max_velocity_limit = 1.57  # rad/s
    joint3_idx = 2

    results = []

    for filename in trajectory_files:
        filepath = f"/Users/lr-2002/project/instantcreation/IC_arm_control/{filename}"

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            positions = np.array(data['positions'])
            velocities = np.array(data['velocities'])

            # 获取Joint3的数据
            joint3_positions = positions[:, joint3_idx]
            joint3_velocities = velocities[:, joint3_idx]

            # 计算统计信息
            pos_range = joint3_positions.max() - joint3_positions.min()
            pos_mean = np.mean(joint3_positions)
            pos_std = np.std(joint3_positions)
            vel_max = abs(joint3_velocities).max()
            vel_utilization = (vel_max / max_velocity_limit) * 100

            # 计算在0附近的活动程度
            time_near_zero = np.sum(np.abs(joint3_positions) < 0.1) / len(joint3_positions) * 100
            time_very_near_zero = np.sum(np.abs(joint3_positions) < 0.05) / len(joint3_positions) * 100

            # 安全评估
            if vel_utilization > 100:
                safety_status = "❌ 超速"
            elif vel_utilization > 90:
                safety_status = "⚠️  接近极限"
            elif vel_utilization > 40:
                safety_status = "✅ 良好"
            else:
                safety_status = "⚠️  利用率低"

            # 激励质量评估
            excitation_score = pos_range * min(vel_utilization/100, 1.0) * (time_near_zero/100)

            results.append({
                'filename': filename,
                'pos_range': pos_range,
                'pos_mean': pos_mean,
                'pos_std': pos_std,
                'vel_max': vel_max,
                'vel_utilization': vel_utilization,
                'time_near_zero': time_near_zero,
                'time_very_near_zero': time_very_near_zero,
                'safety_status': safety_status,
                'excitation_score': excitation_score
            })

            print(f"{filename}:")
            print(f"  位置范围: {pos_range:.3f} rad")
            print(f"  平均位置: {pos_mean:.3f} rad")
            print(f"  位置标准差: {pos_std:.3f} rad")
            print(f"  最大速度: {vel_max:.3f} rad/s")
            print(f"  速度利用率: {vel_utilization:.1f}%")
            print(f"  在±0.1rad内时间: {time_near_zero:.1f}%")
            print(f"  在±0.05rad内时间: {time_very_near_zero:.1f}%")
            print(f"  安全状态: {safety_status}")
            print(f"  激励质量评分: {excitation_score:.3f}")
            print()

        except Exception as e:
            print(f"❌ 分析 {filename} 时出错: {e}")

    # 排序并推荐
    safe_trajectories = [r for r in results if r['vel_max'] <= max_velocity_limit * 0.9]

    print("=== 推荐轨迹 ===")

    if safe_trajectories:
        # 按激励质量排序
        safe_trajectories.sort(key=lambda x: x['excitation_score'], reverse=True)

        print("🎯 安全且高激励质量的轨迹:")
        for i, traj in enumerate(safe_trajectories[:3]):  # 显示前3个
            print(f"  {i+1}. {traj['filename']}")
            print(f"     活动范围: {traj['pos_range']:.3f} rad")
            print(f"     速度利用率: {traj['vel_utilization']:.1f}%")
            print(f"     在0附近时间: {traj['time_near_zero']:.1f}%")
            print(f"     激励评分: {traj['excitation_score']:.3f}")

    # 显示所有轨迹的对比
    print(f"\n=== 所有轨迹按激励质量排序 ===")
    all_trajectories_sorted = sorted(results, key=lambda x: x['excitation_score'], reverse=True)

    for i, traj in enumerate(all_trajectories_sorted):
        status_icon = "🟢" if traj['vel_max'] <= max_velocity_limit * 0.9 else "🔴"
        print(f"{i+1:2d}. {status_icon} {traj['filename']}")
        print(f"     范围: {traj['pos_range']:.3f}rad, 速度: {traj['vel_utilization']:.1f}%, 0附近: {traj['time_near_zero']:.1f}%, 评分: {traj['excitation_score']:.3f}")

    return results

if __name__ == "__main__":
    analyze_joint3_trajectories()