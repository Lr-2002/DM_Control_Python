#!/usr/bin/env python3
"""
测试保守轨迹的辨识效果
"""

import numpy as np
import pandas as pd
import json
import os
from multi_joint_identification import MultiJointIdentification

def simulate_trajectory_data(trajectory_file, output_csv_file):
    """
    模拟轨迹数据 - 添加噪声和动力学效应
    """
    print(f"=== 模拟轨迹数据 ===")

    # 加载轨迹
    with open(trajectory_file, 'r') as f:
        trajectory_data = json.load(f)

    time = np.array(trajectory_data['time'])
    positions = np.array(trajectory_data['positions'])
    velocities = np.array(trajectory_data['velocities'])
    accelerations = np.array(trajectory_data['accelerations'])

    n_samples, n_joints = positions.shape

    print(f"轨迹数据: {n_samples} 个采样点, {n_joints} 个关节")
    print(f"持续时间: {time[-1]:.1f} 秒")

    # 模拟动力学参数 (基于典型机器人的数值)
    # 惯性参数
    inertia_params = [
        0.5,   # Joint1
        1.2,   # Joint2
        0.8,   # Joint3
        1.5,   # Joint4
        0.6,   # Joint5
        0.3    # Joint6
    ]

    # 摩擦参数
    friction_params = [
        0.1,   # Joint1
        0.15,  # Joint2
        0.08,  # Joint3
        0.12,  # Joint4
        0.07,  # Joint5
        0.05   # Joint6
    ]

    # 重力参数
    gravity_params = [
        2.0,   # Joint1
        5.0,   # Joint2
        3.0,   # Joint3
        4.0,   # Joint4
        1.5,   # Joint5
        0.8    # Joint6
    ]

    # 计算理论力矩
    theoretical_torques = np.zeros_like(positions)

    for i in range(n_samples):
        for j in range(n_joints):
            # 简化的动力学模型
            pos = positions[i, j]
            vel = velocities[i, j]
            acc = accelerations[i, j]

            # 惯性项
            inertia_torque = inertia_params[j] * acc

            # 摩擦项
            friction_torque = friction_params[j] * vel

            # 重力项
            gravity_torque = gravity_params[j] * np.sin(pos)

            theoretical_torques[i, j] = inertia_torque + friction_torque + gravity_torque

    # 添加测量噪声
    noise_level = 0.02  # 2% 噪声
    noise = np.random.normal(0, noise_level, theoretical_torques.shape)
    measured_torques = theoretical_torques + noise

    # 创建DataFrame
    data = {}
    for j in range(n_joints):
        joint_id = j + 1
        data[f'm{joint_id}_pos_actual'] = positions[:, j]
        data[f'm{joint_id}_vel_actual'] = velocities[:, j]
        data[f'm{joint_id}_acc_actual'] = accelerations[:, j]
        data[f'm{joint_id}_torque'] = measured_torques[:, j]

    df = pd.DataFrame(data)

    # 保存到CSV
    df.to_csv(output_csv_file, index=False)

    print(f"模拟数据已保存到: {output_csv_file}")
    print(f"力矩范围:")
    for j in range(n_joints):
        torque_range = [
            measured_torques[:, j].min(),
            measured_torques[:, j].max()
        ]
        print(f"  Joint{j+1}: [{torque_range[0]:.3f}, {torque_range[1]:.3f}] Nm")

    return df

def test_identification_with_conservative_trajectory():
    """测试保守轨迹的辨识效果"""

    # 输入文件
    trajectory_file = "/Users/lr-2002/project/instantcreation/IC_arm_control/conservative_gravity_identification.json"
    simulated_data_file = "/Users/lr-2002/project/instantcreation/IC_arm_control/simulated_conservative_data.csv"

    # 模拟数据
    simulated_df = simulate_trajectory_data(trajectory_file, simulated_data_file)

    print(f"\n=== 运行辨识测试 ===")

    # 创建辨识器
    identifier = MultiJointIdentification(n_joints=6)

    # 运行辨识
    results = identifier.identify_all_joints(simulated_df, regularization=0.1)

    # 统计结果
    valid_results = [r for r in results if r is not None]

    if valid_results:
        print(f"\n=== 辨识结果总结 ===")
        print(f"成功辨识关节数: {len(valid_results)}/{len(results)}")

        avg_r2 = np.mean([r['r2'] for r in valid_results])
        avg_rmse = np.mean([r['rmse'] for r in valid_results])

        print(f"平均R²分数: {avg_r2:.4f}")
        print(f"平均RMS误差: {avg_rmse:.6f} Nm")

        print(f"\n各关节性能:")
        for result in valid_results:
            joint_id = result['joint_id']
            r2 = result['r2']
            rmse = result['rmse']
            model_name = result.get('model_name', 'Unknown')

            status = "✅" if r2 > 0.8 else "⚠️" if r2 > 0.6 else "❌"
            print(f"  Joint{joint_id}: {r2:.4f} {status} ({model_name}, RMSE: {rmse:.4f})")

        # 保存结果
        param_file, report_file = identifier.save_results()
        print(f"\n详细结果已保存到: {report_file}")

        return True
    else:
        print("❌ 辨识失败，没有有效结果")
        return False

if __name__ == "__main__":
    print("保守轨迹辨识效果测试")
    print("=" * 50)

    success = test_identification_with_conservative_trajectory()

    if success:
        print(f"\n✅ 测试完成 - 保守轨迹可用于参数辨识")
    else:
        print(f"\n❌ 测试失败 - 需要进一步优化")