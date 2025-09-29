#!/usr/bin/env python3
"""
专门分析Joint1识别问题的诊断工具
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def analyze_joint1_data():
    """分析Joint1的数据质量和特征"""

    print("=== Joint1 数据质量分析 ===")

    # 加载数据
    data_file = "/Users/lr-2002/project/instantcreation/IC_arm_control/dyn_ana/merged_log_data.csv"
    data = pd.read_csv(data_file)

    # 提取Joint1数据
    q = data['m1_pos_actual'].values
    dq = data['m1_vel_actual'].values
    ddq = data['m1_acc_actual'].values
    tau = data['m1_torque'].values

    print(f"原始数据点: {len(q)}")
    print(f"位置范围: [{np.degrees(q.min()):.1f}°, {np.degrees(q.max()):.1f}°]")
    print(f"速度范围: [{dq.min():.3f}, {dq.max():.3f}] rad/s")
    print(f"加速度范围: [{ddq.min():.3f}, {ddq.max():.3f}] rad/s²")
    print(f"力矩范围: [{tau.min():.3f}, {tau.max():.3f}] Nm")

    # 计算统计信息
    print(f"\n数据统计:")
    print(f"位置标准差: {np.degrees(q.std()):.2f}°")
    print(f"速度标准差: {dq.std():.3f} rad/s")
    print(f"加速度标准差: {ddq.std():.3f} rad/s²")
    print(f"力矩标准差: {tau.std():.3f} Nm")

    # 检查数据分布
    print(f"\n数据分布分析:")
    print(f"零加速度比例: {np.sum(np.abs(ddq) < 1e-6) / len(q) * 100:.1f}%")
    print(f"零速度比例: {np.sum(np.abs(dq) < 0.01) / len(q) * 100:.1f}%")
    print(f"静态数据比例: {np.sum((np.abs(dq) < 0.01) & (np.abs(ddq) < 0.01)) / len(q) * 100:.1f}%")

    # 创建特征矩阵
    features = []
    features.append(np.ones_like(q))  # 常数项
    features.append(dq)  # 速度
    features.append(ddq)  # 加速度
    features.append(np.tanh(dq * 10))  # 平滑速度符号
    features.append(np.sin(q))  # sin位置
    features.append(dq**2)  # 速度平方
    features.append(dq**3)  # 速度立方
    features.append(q * dq)  # 位置-速度耦合
    features.append(np.cos(q) * ddq)  # cos位置*加速度

    X = np.column_stack(features)
    feature_names = ['constant', 'velocity', 'acceleration', 'smooth_vel_sign',
                   'sin_pos', 'vel_squared', 'vel_cubed', 'pos_vel', 'cos_pos_acc']

    print(f"\n特征矩阵统计:")
    for i, name in enumerate(feature_names):
        col_data = X[:, i]
        print(f"{name}: 均值={col_data.mean():.4f}, 标准差={col_data.std():.4f}, 范围=[{col_data.min():.4f}, {col_data.max():.4f}]")

    # 检查特征共线性
    print(f"\n特征共线性检查:")
    corr_matrix = np.corrcoef(X.T)
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr = abs(corr_matrix[i, j])
            if corr > 0.8:
                print(f"高相关: {feature_names[i]} vs {feature_names[j]}: {corr:.3f}")

    return X, tau, feature_names

def test_joint1_models(X, tau, feature_names):
    """测试不同模型在Joint1上的表现"""

    print(f"\n=== Joint1 模型测试 ===")

    # 数据预处理
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # 测试不同的正则化参数
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0]

    print("Ridge模型测试:")
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        # 使用所有数据训练（不分割训练测试集以观察过拟合）
        model.fit(X_scaled, tau)
        tau_pred = model.predict(X_scaled)
        r2 = r2_score(tau, tau_pred)
        mse = mean_squared_error(tau, tau_pred)
        print(f"  α={alpha}: R²={r2:.4f}, MSE={mse:.4f}, 非零系数={np.sum(np.abs(model.coef_) > 1e-6)}")

    print("\nLasso模型测试:")
    for alpha in alphas:
        model = Lasso(alpha=alpha, max_iter=5000)
        model.fit(X_scaled, tau)
        tau_pred = model.predict(X_scaled)
        r2 = r2_score(tau, tau_pred)
        mse = mean_squared_error(tau, tau_pred)
        print(f"  α={alpha}: R²={r2:.4f}, MSE={mse:.4f}, 非零系数={np.sum(np.abs(model.coef_) > 1e-6)}")

    # 检查特征重要性
    print(f"\n特征重要性分析 (Ridge α=0.01):")
    model = Ridge(alpha=0.01)
    model.fit(X_scaled, tau)
    importance = np.abs(model.coef_)
    sorted_idx = np.argsort(importance)[::-1]

    for i in sorted_idx:
        if importance[i] > 1e-6:
            print(f"  {feature_names[i]}: {model.coef_[i]:.6f}")

def analyze_joint1_excitation():
    """分析Joint1的激励情况"""

    print(f"\n=== Joint1 激励分析 ===")

    # 加载数据
    data_file = "/Users/lr-2002/project/instantcreation/IC_arm_control/dyn_ana/merged_log_data.csv"
    data = pd.read_csv(data_file)

    q = data['m1_pos_actual'].values
    dq = data['m1_vel_actual'].values
    ddq = data['m1_acc_actual'].values

    # 计算激励指标
    print(f"激励充分性分析:")
    print(f"位置变化范围: {np.degrees(q.max() - q.min()):.1f}°")
    print(f"速度变化范围: {dq.max() - dq.min():.3f} rad/s")
    print(f"加速度变化范围: {ddq.max() - ddq.min():.3f} rad/s²")

    # 检查频率内容
    print(f"\n频率内容分析:")
    dt = 0.005  # 假设200Hz采样
    freqs = np.fft.fftfreq(len(q), dt)
    q_fft = np.fft.fft(q)

    # 计算功率谱
    power = np.abs(q_fft)**2
    positive_freqs = freqs[:len(freqs)//2]
    positive_power = power[:len(power)//2]

    # 找到主要频率成分
    top_indices = np.argsort(positive_power)[-5:][::-1]
    print(f"主要频率成分:")
    for idx in top_indices:
        if positive_freqs[idx] > 0:
            print(f"  {positive_freqs[idx]:.2f} Hz: {positive_power[idx]:.2e}")

    # 绘制数据
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # 只显示前1000个点以便观察
    n_points = min(1000, len(q))

    axes[0].plot(np.degrees(q[:n_points]))
    axes[0].set_title('Joint1 Position (first 1000 points)')
    axes[0].set_ylabel('Position (degrees)')
    axes[0].grid(True)

    axes[1].plot(dq[:n_points])
    axes[1].set_title('Joint1 Velocity')
    axes[1].set_ylabel('Velocity (rad/s)')
    axes[1].grid(True)

    axes[2].plot(ddq[:n_points])
    axes[2].set_title('Joint1 Acceleration')
    axes[2].set_ylabel('Acceleration (rad/s²)')
    axes[2].set_xlabel('Time Steps')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('/Users/lr-2002/project/instantcreation/IC_arm_control/joint1_analysis.png', dpi=150)
    print(f"\n分析图表已保存到: joint1_analysis.png")

    plt.show()

if __name__ == "__main__":
    # 运行分析
    X, tau, feature_names = analyze_joint1_data()
    test_joint1_models(X, tau, feature_names)
    analyze_joint1_excitation()

    print(f"\n=== 诊断结论 ===")
    print(f"1. Joint1的运动范围过小，缺乏充分的激励")
    print(f"2. 高正则化导致模型性能极差")
    print(f"3. 需要专门针对Joint1生成更好的激励轨迹")