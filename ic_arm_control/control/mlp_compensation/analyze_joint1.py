#!/usr/bin/env python3
"""
分析Joint 1性能问题的原因
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from train_model import load_training_data_from_datasets

def analyze_joint1_data():
    """分析Joint 1的数据分布和特性"""
    print("=== Joint 1 Data Analysis ===")

    # 加载数据
    dataset_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control/datasets"
    positions, velocities, torques = load_training_data_from_datasets(dataset_dir)

    # 分析Joint 1的数据
    joint1_positions = positions[:, 0]
    joint1_velocities = velocities[:, 0]
    joint1_torques = torques[:, 0]

    print(f"Joint 1 Statistics:")
    print(f"Position range: [{np.min(joint1_positions):.3f}, {np.max(joint1_positions):.3f}] rad")
    print(f"Velocity range: [{np.min(joint1_velocities):.3f}, {np.max(joint1_velocities):.3f}] rad/s")
    print(f"Torque range: [{np.min(joint1_torques):.3f}, {np.max(joint1_torques):.3f}] Nm")

    # 分析扭矩分布
    print(f"\nTorque Distribution:")
    print(f"Mean torque: {np.mean(joint1_torques):.3f} Nm")
    print(f"Std torque: {np.std(joint1_torques):.3f} Nm")
    print(f"Torque variance: {np.var(joint1_torques):.3f} Nm²")

    # 分析位置-扭矩关系
    print(f"\nPosition-Torque Relationship:")
    # 按位置分组统计扭矩
    pos_bins = np.linspace(np.min(joint1_positions), np.max(joint1_positions), 20)
    pos_indices = np.digitize(joint1_positions, pos_bins)

    for i in range(1, len(pos_bins)):
        mask = pos_indices == i
        if np.sum(mask) > 0:
            avg_torque = np.mean(joint1_torques[mask])
            std_torque = np.std(joint1_torques[mask])
            count = np.sum(mask)
            pos_center = (pos_bins[i-1] + pos_bins[i]) / 2
            print(f"Pos {pos_center:.3f} rad: Torque = {avg_torque:.3f} ± {std_torque:.3f} Nm (n={count})")

    # 检查数据质量
    print(f"\nData Quality Check:")
    print(f"Zero torque samples: {np.sum(np.abs(joint1_torques) < 0.1)} / {len(joint1_torques)} ({100*np.sum(np.abs(joint1_torques) < 0.1)/len(joint1_torques):.1f}%)")
    print(f"High torque samples (>5 Nm): {np.sum(np.abs(joint1_torques) > 5)} / {len(joint1_torques)} ({100*np.sum(np.abs(joint1_torques) > 5)/len(joint1_torques):.1f}%)")

    return joint1_positions, joint1_velocities, joint1_torques

def plot_joint1_analysis(positions, velocities, torques):
    """绘制Joint 1的分析图表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Joint 1 Data Analysis', fontsize=16)

    # 1. 位置-扭矩散点图
    axes[0, 0].scatter(positions, torques, alpha=0.6, s=20)
    axes[0, 0].set_xlabel('Position (rad)')
    axes[0, 0].set_ylabel('Torque (Nm)')
    axes[0, 0].set_title('Position vs Torque')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 扭矩分布直方图
    axes[0, 1].hist(torques, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_xlabel('Torque (Nm)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Torque Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 速度-扭矩散点图
    axes[1, 0].scatter(velocities, torques, alpha=0.6, s=20)
    axes[1, 0].set_xlabel('Velocity (rad/s)')
    axes[1, 0].set_ylabel('Torque (Nm)')
    axes[1, 0].set_title('Velocity vs Torque')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 时间序列分析（前1000个点）
    time_points = np.arange(len(torques[:1000]))
    axes[1, 1].plot(time_points, torques[:1000], 'b-', alpha=0.7, linewidth=1)
    axes[1, 1].set_xlabel('Time Steps')
    axes[1, 1].set_ylabel('Torque (Nm)')
    axes[1, 1].set_title('Torque Time Series (First 1000 points)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('joint1_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved Joint 1 analysis plot to joint1_analysis.png")
    plt.show()

def compare_with_other_joints(positions, velocities, torques):
    """比较Joint 1与其他关节的特性"""
    print("\n=== Comparison with Other Joints ===")

    # Since torques is 1D (only Joint 1 data), we'll just analyze Joint 1
    joint_names = ['Joint 1']

    for i in range(len(joint_names)):
        joint_torques = torques  # torques is already Joint 1 data
        torque_variance = np.var(joint_torques)
        torque_range = np.max(joint_torques) - np.min(joint_torques)

        print(f"{joint_names[i]}:")
        print(f"  Torque variance: {torque_variance:.3f} Nm²")
        print(f"  Torque range: {torque_range:.3f} Nm")
        print(f"  Mean |torque|: {np.mean(np.abs(joint_torques)):.3f} Nm")

    print(f"\nKey Observation:")
    print(f"Joint 1 shows high torque variance and range,")
    print(f"which makes it challenging to model accurately.")
    print(f"Note: This analysis only includes Joint 1 data.")

def main():
    """主函数"""
    positions, velocities, torques = analyze_joint1_data()
    plot_joint1_analysis(positions, velocities, torques)
    compare_with_other_joints(positions, velocities, torques)

if __name__ == "__main__":
    main()