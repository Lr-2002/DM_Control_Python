#!/usr/bin/env python3
"""
分析轨迹激励范围，并提供优化建议
"""

import numpy as np
import json
import os

def analyze_trajectories():
    """分析所有轨迹的激励范围"""

    trajectory_files = [
        "trajectory_multi_frequency.json",
        "trajectory_chirp.json",
        "trajectory_schroeder.json",
        "trajectory_pseudo_random.json",
        "trajectory_phase_modulated.json",
        "trajectory_sum_of_sines.json"
    ]

    max_velocity_limit = 1.57  # rad/s

    print("=== 轨迹激励范围分析 ===")
    print(f"电机速度限制: {max_velocity_limit} rad/s")
    print()

    results = []

    for filename in trajectory_files:
        filepath = f"/Users/lr-2002/project/instantcreation/IC_arm_control/{filename}"

        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)

            positions = np.array(data['positions'])
            velocities = np.array(data['velocities'])

            # 计算实际范围
            pos_range = positions.max() - positions.min()
            vel_max = abs(velocities).max()
            vel_utilization = (vel_max / max_velocity_limit) * 100

            # 评估结果
            if vel_utilization > 100:
                safety_status = "❌ 超速"
            elif vel_utilization > 90:
                safety_status = "⚠️  接近极限"
            elif vel_utilization > 60:
                safety_status = "✅ 良好"
            else:
                safety_status = "⚠️  利用率低"

            results.append({
                'filename': filename,
                'pos_range': pos_range,
                'vel_max': vel_max,
                'vel_utilization': vel_utilization,
                'safety_status': safety_status
            })

            print(f"{filename}:")
            print(f"  位置范围: {pos_range:.3f} rad")
            print(f"  最大速度: {vel_max:.3f} rad/s")
            print(f"  速度利用率: {vel_utilization:.1f}%")
            print(f"  安全状态: {safety_status}")
            print()

    # 总结和建议
    print("=== 总结和建议 ===")
    safe_trajectories = [r for r in results if r['vel_max'] <= max_velocity_limit * 0.9]
    unsafe_trajectories = [r for r in results if r['vel_max'] > max_velocity_limit]

    if unsafe_trajectories:
        print(f"❌ 有 {len(unsafe_trajectories)} 个轨迹超速:")
        for t in unsafe_trajectories:
            print(f"   - {t['filename']}: {t['vel_max']:.3f} rad/s ({t['vel_utilization']:.1f}%)")

    if safe_trajectories:
        print(f"✅ 有 {len(safe_trajectories)} 个轨迹安全:")
        for t in safe_trajectories:
            print(f"   - {t['filename']}: {t['vel_max']:.3f} rad/s ({t['vel_utilization']:.1f}%)")

    # 找到最佳平衡的轨迹
    balanced_trajectories = [r for r in safe_trajectories if 60 <= r['vel_utilization'] <= 90]
    if balanced_trajectories:
        print(f"\n🎯 推荐使用的轨迹 (平衡了安全性和激励效果):")
        for t in balanced_trajectories:
            print(f"   - {t['filename']}: {t['vel_max']:.3f} rad/s, 位置范围 {t['pos_range']:.3f} rad")

    return results

if __name__ == "__main__":
    analyze_trajectories()