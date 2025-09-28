#!/usr/bin/env python3
"""
单关节动力学辨识（针对IC ARM数据集只有关节1有数据的情况）
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

class SingleJointIdentification:
    """单关节动力学辨识器"""

    def __init__(self, joint_id=1):
        """
        初始化单关节辨识器

        Args:
            joint_id: 关节ID (1-5)
        """
        self.joint_id = joint_id
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.identification_results = {}

    def create_features(self, q, dq, ddq):
        """
        创建动力学特征

        Args:
            q: 位置
            dq: 速度
            ddq: 加速度

        Returns:
            特征矩阵
        """
        features = []

        # 基本特征
        features.append(np.ones_like(q))  # 常数项（重力补偿）
        features.append(dq)  # 速度相关
        features.append(ddq)  # 加速度相关（惯性）
        features.append(np.sign(dq))  # 摩擦力方向

        # 非线性特征
        features.append(q * dq)  # 位置-速度耦合
        features.append(q * ddq)  # 位置-加速度耦合
        features.append(dq * ddq)  # 速度-加速度耦合
        features.append(dq**2)  # 速度平方
        features.append(np.sin(q))  # 位置相关重力
        features.append(np.cos(q))  # 位置相关重力

        # 组合特征矩阵
        X = np.column_stack(features)

        # 特征名称
        self.feature_names = [
            'constant', 'velocity', 'acceleration', 'velocity_sign',
            'pos_vel', 'pos_acc', 'vel_acc', 'vel_squared', 'sin_pos', 'cos_pos'
        ]

        return X

    def identify(self, q, dq, ddq, tau, regularization=0.1):
        """
        执行动力学辨识

        Args:
            q: 位置数组
            dq: 速度数组
            ddq: 加速度数组
            tau: 力矩数组
            regularization: 正则化参数

        Returns:
            辨识结果
        """
        print(f"=== 关节{self.joint_id}动力学辨识 ===")

        # 创建特征矩阵
        X = self.create_features(q, dq, ddq)
        y = tau

        print(f"特征矩阵形状: {X.shape}")
        print(f"力矩向量长度: {len(y)}")

        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)

        # 使用岭回归进行辨识
        self.model = Ridge(alpha=regularization)
        self.model.fit(X_scaled, y)

        # 预测
        tau_pred = self.model.predict(X_scaled)

        # 计算误差
        mse = mean_squared_error(y, tau_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, tau_pred)

        print(f"RMS误差: {rmse:.6f}")
        print(f"R²分数: {r2:.4f}")

        # 输出参数
        print(f"\n辨识参数:")
        for i, (name, coef) in enumerate(zip(self.feature_names, self.model.coef_)):
            print(f"  {name}: {coef:.6f}")
        print(f"  截距: {self.model.intercept_:.6f}")

        # 保存结果
        self.identification_results = {
            'joint_id': self.joint_id,
            'coefficients': self.model.coef_,
            'intercept': self.model.intercept_,
            'feature_names': self.feature_names,
            'rmse': rmse,
            'r2': r2,
            'regularization': regularization,
            'n_samples': len(q),
            'position_range': [q.min(), q.max()],
            'velocity_range': [dq.min(), dq.max()],
            'torque_range': [tau.min(), tau.max()]
        }

        return self.identification_results

    def predict(self, q, dq, ddq):
        """
        使用辨识结果预测力矩

        Args:
            q: 位置
            dq: 速度
            ddq: 加速度

        Returns:
            预测力矩
        """
        if self.model is None:
            raise ValueError("模型尚未训练")

        X = self.create_features(q, dq, ddq)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def plot_results(self, q, dq, ddq, tau, save_path=None):
        """
        绘制辨识结果

        Args:
            q: 位置
            dq: 速度
            ddq: 加速度
            tau: 实际力矩
            save_path: 保存路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练")

        # 预测力矩
        tau_pred = self.predict(q, dq, ddq)

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Torque comparison
        axes[0, 0].plot(tau, label='Actual Torque', alpha=0.7)
        axes[0, 0].plot(tau_pred, label='Predicted Torque', alpha=0.7)
        axes[0, 0].set_title(f'Joint {self.joint_id} Torque Comparison')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Torque (Nm)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 2. Residual
        residual = tau - tau_pred
        axes[0, 1].plot(residual, alpha=0.7)
        axes[0, 1].set_title(f'Prediction Residual (RMSE: {self.identification_results["rmse"]:.4f})')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Residual (Nm)')
        axes[0, 1].grid(True)

        # 3. Scatter plot
        axes[1, 0].scatter(tau, tau_pred, alpha=0.5, s=1)
        axes[1, 0].plot([tau.min(), tau.max()], [tau.min(), tau.max()], 'r--', alpha=0.8)
        axes[1, 0].set_title(f'Actual vs Predicted (R²: {self.identification_results["r2"]:.4f})')
        axes[1, 0].set_xlabel('Actual Torque (Nm)')
        axes[1, 0].set_ylabel('Predicted Torque (Nm)')
        axes[1, 0].grid(True)

        # 4. Parameter importance
        coefficients = self.identification_results['coefficients']
        feature_names = self.identification_results['feature_names']
        importance = np.abs(coefficients)
        sorted_idx = np.argsort(importance)[::-1]

        axes[1, 1].bar(range(len(coefficients)), importance[sorted_idx])
        axes[1, 1].set_xticks(range(len(coefficients)))
        axes[1, 1].set_xticklabels([feature_names[i] for i in sorted_idx], rotation=45)
        axes[1, 1].set_title('Parameter Importance')
        axes[1, 1].set_ylabel('Coefficient Absolute Value')
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")

        plt.show()

    def save_results(self, output_dir="identification_results"):
        """
        保存辨识结果

        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存参数
        param_file = os.path.join(output_dir, f"joint{self.joint_id}_params_{timestamp}.npz")
        np.savez(param_file, **self.identification_results)

        # 保存报告
        report_file = os.path.join(output_dir, f"joint{self.joint_id}_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"=== 关节{self.joint_id}动力学辨识报告 ===\n\n")
            f.write(f"辨识时间: {datetime.now()}\n")
            f.write(f"样本数: {self.identification_results['n_samples']}\n")
            f.write(f"正则化参数: {self.identification_results['regularization']}\n\n")

            f.write("辨识精度:\n")
            f.write(f"  RMS误差: {self.identification_results['rmse']:.6f}\n")
            f.write(f"  R²分数: {self.identification_results['r2']:.4f}\n\n")

            f.write("运动范围:\n")
            f.write(f"  位置范围: [{self.identification_results['position_range'][0]:.6f}, {self.identification_results['position_range'][1]:.6f}] rad\n")
            f.write(f"  速度范围: [{self.identification_results['velocity_range'][0]:.6f}, {self.identification_results['velocity_range'][1]:.6f}] rad/s\n")
            f.write(f"  力矩范围: [{self.identification_results['torque_range'][0]:.3f}, {self.identification_results['torque_range'][1]:.3f}] Nm\n\n")

            f.write("辨识参数:\n")
            for name, coef in zip(self.feature_names, self.identification_results['coefficients']):
                f.write(f"  {name}: {coef:.6f}\n")
            f.write(f"  截距: {self.identification_results['intercept']:.6f}\n")

        print(f"结果已保存到: {output_dir}")
        return param_file, report_file

def load_data_from_csv(csv_file):
    """
    从CSV文件加载关节数据

    Args:
        csv_file: CSV文件路径

    Returns:
        q, dq, ddq, tau: 位置、速度、加速度、力矩
    """
    print(f"加载数据: {csv_file}")
    data = pd.read_csv(csv_file)

    # 提取关节1数据
    pos_col = 'm1_pos_actual'
    vel_col = 'm1_vel_actual'
    acc_col = 'm1_acc_actual'
    torque_col = 'm1_torque'

    if pos_col not in data.columns:
        raise ValueError(f"列 {pos_col} 不存在")

    q = data[pos_col].values
    dq = data[vel_col].values
    ddq = data[acc_col].values
    tau = data[torque_col].values

    print(f"数据点数: {len(q)}")
    print(f"位置范围: [{q.min():.6f}, {q.max():.6f}] rad ({np.degrees([q.min(), q.max()])[0]:.1f}° ~ {np.degrees([q.min(), q.max()])[1]:.1f}°)")
    print(f"速度范围: [{dq.min():.6f}, {dq.max():.6f}] rad/s")
    print(f"力矩范围: [{tau.min():.3f}, {tau.max():.3f}] Nm")

    return q, dq, ddq, tau

def run_joint_identification(csv_file, max_points=None, regularization=0.1):
    """
    运行单关节辨识

    Args:
        csv_file: CSV文件路径
        max_points: 最大数据点数
        regularization: 正则化参数
    """
    # 加载数据
    q, dq, ddq, tau = load_data_from_csv(csv_file)

    # 限制数据点数
    if max_points and len(q) > max_points:
        print(f"限制数据点数为: {max_points}")
        indices = np.linspace(0, len(q)-1, max_points, dtype=int)
        q = q[indices]
        dq = dq[indices]
        ddq = ddq[indices]
        tau = tau[indices]

    # 创建辨识器并执行辨识
    identifier = SingleJointIdentification(joint_id=1)
    results = identifier.identify(q, dq, ddq, tau, regularization=regularization)

    # 绘制结果
    identifier.plot_results(q, dq, ddq, tau)

    # 保存结果
    param_file, report_file = identifier.save_results()

    return identifier

def compare_different_datasets(data_files, max_points=5000):
    """
    比较不同数据集的辨识结果

    Args:
        data_files: 数据文件列表
        max_points: 每个文件的最大数据点数
    """
    print("=== 多数据集辨识比较 ===\n")

    results = []

    for file in data_files:
        print(f"处理文件: {os.path.basename(file)}")

        try:
            # 加载数据
            q, dq, ddq, tau = load_data_from_csv(file)

            # 限制数据点数
            if len(q) > max_points:
                indices = np.linspace(0, len(q)-1, max_points, dtype=int)
                q = q[indices]
                dq = dq[indices]
                ddq = ddq[indices]
                tau = tau[indices]

            # 辨识
            identifier = SingleJointIdentification(joint_id=1)
            result = identifier.identify(q, dq, ddq, tau, regularization=0.1)

            results.append({
                'file': os.path.basename(file),
                'identifier': identifier,
                'result': result
            })

            print(f"  RMS误差: {result['rmse']:.6f}")
            print(f"  R²分数: {result['r2']:.4f}\n")

        except Exception as e:
            print(f"  处理失败: {e}\n")

    # 输出比较结果
    if results:
        print("=== 辨识结果比较 ===")
        print(f"{'文件名':<25} {'样本数':<8} {'RMS误差':<12} {'R²分数':<10}")
        print("-" * 65)

        for item in results:
            result = item['result']
            print(f"{item['file']:<25} {result['n_samples']:<8} {result['rmse']:<12.6f} {result['r2']:<10.4f}")

    return results

if __name__ == "__main__":
    # 查找所有数据文件
    data_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control/dyn_ana"
    data_files = [
        os.path.join(data_dir, f"dynamics_data_20250928_153405.csv"),
        os.path.join(data_dir, f"dynamics_data_20250928_153459.csv"),
        os.path.join(data_dir, f"dynamics_data_20250928_153654.csv"),
        os.path.join(data_dir, f"dynamics_data_20250928_153911.csv")
    ]

    print("IC ARM单关节动力学辨识\n")

    # 1. 比较不同数据集
    print("1. 比较不同数据集的辨识结果...")
    comparison_results = compare_different_datasets(data_files, max_points=3000)

    print("\n" + "="*60)

    # 2. 使用合并数据集进行综合辨识
    print("2. 使用合并数据集进行综合辨识...")
    merged_file = os.path.join(data_dir, "merged_dynamics_data.csv")

    if os.path.exists(merged_file):
        identifier = run_joint_identification(merged_file, max_points=10000, regularization=0.1)
        print("\n✓ 综合辨识完成！")
    else:
        print("⚠️ 合并数据集文件不存在")