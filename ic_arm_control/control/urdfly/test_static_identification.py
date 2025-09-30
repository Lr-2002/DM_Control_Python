#!/usr/bin/env python3
"""
测试静态数据辨识功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multi_joint_identification import MultiJointIdentification

def generate_static_test_data():
    """生成模拟的静态测试数据"""
    print("生成模拟静态数据...")

    # 时间序列
    n_points = 1000
    t = np.linspace(0, 10, n_points)

    # 模拟关节1的静态数据（不同位置的重力矩）
    # 位置缓慢变化
    q1 = np.linspace(-np.pi/4, np.pi/4, n_points) + 0.01 * np.sin(2*np.pi*0.5*t)

    # 速度接近零（小量噪声）
    dq1 = 0.001 * np.random.randn(n_points)

    # 加速度接近零
    ddq1 = 0.0001 * np.random.randn(n_points)

    # 模拟重力矩（主要来自sin和cos分量）
    gravity_torque = 2.5 * np.sin(q1) + 0.8 * np.cos(q1) + 0.3 * np.sin(2*q1)

    # 添加小量噪声
    tau1 = gravity_torque + 0.02 * np.random.randn(n_points)

    # 创建DataFrame
    data = pd.DataFrame({
        'm1_pos_actual': q1,
        'm1_vel_actual': dq1,
        'm1_acc_actual': ddq1,
        'm1_torque': tau1
    })

    print(f"生成的数据统计:")
    print(f"  位置范围: [{np.degrees(q1.min()):.1f}°, {np.degrees(q1.max()):.1f}°]")
    print(f"  速度标准差: {dq1.std():.6f} rad/s")
    print(f"  加速度标准差: {ddq1.std():.6f} rad/s²")
    print(f"  力矩范围: [{tau1.min():.3f}, {tau1.max():.3f}] Nm")

    return data

def test_static_identification():
    """测试静态数据辨识"""
    print("=== 静态数据辨识测试 ===\n")

    # 生成测试数据
    data = generate_static_test_data()

    # 创建辨识器
    identifier = MultiJointIdentification(n_joints=1)

    print("1. 使用传统方法（动态模式）:")
    results_dynamic = identifier.identify_all_joints(data, data_mode="dynamic")

    # 重置辨识器
    identifier = MultiJointIdentification(n_joints=1)

    print("\n2. 使用静态数据专用方法:")
    results_static = identifier.identify_all_joints(data, data_mode="static")

    # 比较结果
    print("\n=== 结果对比 ===")
    if results_dynamic and results_dynamic[0]:
        r2_dynamic = results_dynamic[0]['r2']
        print(f"动态模式 R²: {r2_dynamic:.4f}")

    if results_static and results_static[0]:
        r2_static = results_static[0]['r2']
        print(f"静态模式 R²: {r2_static:.4f}")

        # 显示辨识的参数
        print(f"\n静态模式辨识参数:")
        for i, (name, coef) in enumerate(zip(results_static[0]['feature_names'],
                                             results_static[0]['coefficients'])):
            if abs(coef) > 1e-6:
                print(f"  {name}: {coef:.6f}")

    # 绘制比较图
    if results_static and results_static[0]:
        plot_identification_results(data, results_static[0], identifier)

def plot_identification_results(data, result, identifier):
    """绘制辨识结果"""
    q = data['m1_pos_actual'].values
    dq = data['m1_vel_actual'].values
    ddq = data['m1_acc_actual'].values
    tau_actual = data['m1_torque'].values

    # 预测力矩
    tau_pred = identifier.predict_joint(1, q, dq, ddq)

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 位置vs时间
    axes[0, 0].plot(np.degrees(q))
    axes[0, 0].set_title('Joint Position vs Time')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Position (degrees)')
    axes[0, 0].grid(True)

    # 2. 力矩对比
    axes[0, 1].plot(tau_actual, label='Actual', alpha=0.7)
    axes[0, 1].plot(tau_pred, label='Predicted', alpha=0.7)
    axes[0, 1].set_title(f'Torque Comparison (R²: {result["r2"]:.4f})')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Torque (Nm)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. 力矩vs位置
    axes[1, 0].scatter(np.degrees(q), tau_actual, alpha=0.5, s=5, label='Actual')
    axes[1, 0].scatter(np.degrees(q), tau_pred, alpha=0.5, s=5, label='Predicted')
    axes[1, 0].set_title('Torque vs Position')
    axes[1, 0].set_xlabel('Position (degrees)')
    axes[1, 0].set_ylabel('Torque (Nm)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 4. 残差
    residual = tau_actual - tau_pred
    axes[1, 1].plot(residual, alpha=0.7)
    axes[1, 1].set_title(f'Prediction Residual (RMSE: {result["rmse"]:.4f})')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Residual (Nm)')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('/Users/lr-2002/project/instantcreation/IC_arm_control/control/urdfly/static_identification_test.png',
                dpi=300, bbox_inches='tight')
    print(f"\n结果图表已保存到: static_identification_test.png")
    plt.show()

if __name__ == "__main__":
    test_static_identification()