#!/usr/bin/env python3
"""
多关节动力学辨识（支持5个关节）
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

class MultiJointIdentification:
    """多关节动力学辨识器"""

    def __init__(self, n_joints=6):
        """
        初始化多关节辨识器

        Args:
            n_joints: 关节数量
        """
        self.n_joints = n_joints
        self.models = []  # 每个关节的模型
        self.scalers = []  # 每个关节的数据标准化器
        self.feature_names = []
        self.identification_results = []

    def create_features(self, q, dq, ddq, joint_id):
        """
        为指定关节创建动力学特征

        Args:
            q: 该关节的位置
            dq: 该关节的速度
            ddq: 该关节的加速度
            joint_id: 关节ID

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

        # 特征名称（只需设置一次）
        if not self.feature_names:
            self.feature_names = [
                'constant', 'velocity', 'acceleration', 'velocity_sign',
                'pos_vel', 'pos_acc', 'vel_acc', 'vel_squared', 'sin_pos', 'cos_pos'
            ]

        return X

    def identify_joint(self, joint_id, q, dq, ddq, tau, regularization=0.1):
        """
        辨识单个关节的动力学参数

        Args:
            joint_id: 关节ID (1-5)
            q: 位置数组
            dq: 速度数组
            ddq: 加速度数组
            tau: 力矩数组
            regularization: 正则化参数

        Returns:
            辨识结果
        """
        print(f"=== 关节{joint_id}动力学辨识 ===")

        # 创建特征矩阵
        X = self.create_features(q, dq, ddq, joint_id)
        y = tau

        print(f"特征矩阵形状: {X.shape}")
        print(f"力矩向量长度: {len(y)}")

        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 使用岭回归进行辨识
        model = Ridge(alpha=regularization)
        model.fit(X_scaled, y)

        # 预测
        tau_pred = model.predict(X_scaled)

        # 计算误差
        mse = mean_squared_error(y, tau_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, tau_pred)

        print(f"RMS误差: {rmse:.6f}")
        print(f"R²分数: {r2:.4f}")

        # 输出参数
        print(f"\n辨识参数:")
        for i, (name, coef) in enumerate(zip(self.feature_names, model.coef_)):
            print(f"  {name}: {coef:.6f}")
        print(f"  截距: {model.intercept_:.6f}")

        # 保存结果
        result = {
            'joint_id': joint_id,
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'feature_names': self.feature_names,
            'rmse': rmse,
            'r2': r2,
            'regularization': regularization,
            'n_samples': len(q),
            'position_range': [q.min(), q.max()],
            'velocity_range': [dq.min(), dq.max()],
            'torque_range': [tau.min(), tau.max()]
        }

        return model, scaler, result

    def identify_all_joints(self, data, regularization=0.1):
        """
        辨识所有关节的动力学参数

        Args:
            data: 包含所有关节数据的DataFrame
            regularization: 正则化参数

        Returns:
            所有关节的辨识结果
        """
        print("=== 多关节动力学辨识 ===")

        for joint_id in range(1, self.n_joints + 1):
            print(f"\n辨识关节 {joint_id}...")

            # 提取该关节数据
            pos_col = f'm{joint_id}_pos_actual'
            vel_col = f'm{joint_id}_vel_actual'
            acc_col = f'm{joint_id}_acc_actual'
            torque_col = f'm{joint_id}_torque'

            if pos_col not in data.columns:
                print(f"⚠️ 关节 {joint_id} 数据不存在，跳过")
                continue

            q = data[pos_col].values
            dq = data[vel_col].values
            ddq = data[acc_col].values
            tau = data[torque_col].values

            # 执行辨识
            model, scaler, result = self.identify_joint(joint_id, q, dq, ddq, tau, regularization)

            # 保存模型和结果
            self.models.append(model)
            self.scalers.append(scaler)
            self.identification_results.append(result)

        print(f"\n✓ 完成 {len(self.identification_results)} 个关节的辨识")
        return self.identification_results

    def predict_joint(self, joint_id, q, dq, ddq):
        """
        使用辨识结果预测指定关节的力矩

        Args:
            joint_id: 关节ID (1-5)
            q: 位置
            dq: 速度
            ddq: 加速度

        Returns:
            预测力矩
        """
        if joint_id < 1 or joint_id > len(self.models):
            raise ValueError(f"关节 {joint_id} 模型不存在")

        model = self.models[joint_id - 1]
        scaler = self.scalers[joint_id - 1]

        X = self.create_features(q, dq, ddq, joint_id)
        X_scaled = scaler.transform(X)
        return model.predict(X_scaled)

    def predict_all_joints(self, q_all, dq_all, ddq_all):
        """
        预测所有关节的力矩

        Args:
            q_all: 所有关节的位置 (n_samples, n_joints)
            dq_all: 所有关节的速度 (n_samples, n_joints)
            ddq_all: 所有关节的加速度 (n_samples, n_joints)

        Returns:
            所有关节的预测力矩 (n_samples, n_joints)
        """
        n_samples = q_all.shape[0]
        predictions = np.zeros((n_samples, self.n_joints))

        for joint_id in range(1, self.n_joints + 1):
            if joint_id <= len(self.models):
                q = q_all[:, joint_id - 1]
                dq = dq_all[:, joint_id - 1]
                ddq = ddq_all[:, joint_id - 1]

                predictions[:, joint_id - 1] = self.predict_joint(joint_id, q, dq, ddq)

        return predictions

    def plot_joint_results(self, joint_id, q, dq, ddq, tau, save_path=None):
        """
        绘制单个关节的辨识结果

        Args:
            joint_id: 关节ID
            q: 位置
            dq: 速度
            ddq: 加速度
            tau: 实际力矩
            save_path: 保存路径
        """
        if joint_id > len(self.models):
            raise ValueError(f"关节 {joint_id} 模型不存在")

        # 预测力矩
        tau_pred = self.predict_joint(joint_id, q, dq, ddq)
        result = self.identification_results[joint_id - 1]

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Torque comparison
        axes[0, 0].plot(tau, label='Actual Torque', alpha=0.7)
        axes[0, 0].plot(tau_pred, label='Predicted Torque', alpha=0.7)
        axes[0, 0].set_title(f'Joint {joint_id} Torque Comparison')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Torque (Nm)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 2. Residual
        residual = tau - tau_pred
        axes[0, 1].plot(residual, alpha=0.7)
        axes[0, 1].set_title(f'Prediction Residual (RMSE: {result["rmse"]:.4f})')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Residual (Nm)')
        axes[0, 1].grid(True)

        # 3. Scatter plot
        axes[1, 0].scatter(tau, tau_pred, alpha=0.5, s=1)
        axes[1, 0].plot([tau.min(), tau.max()], [tau.min(), tau.max()], 'r--', alpha=0.8)
        axes[1, 0].set_title(f'Actual vs Predicted (R²: {result["r2"]:.4f})')
        axes[1, 0].set_xlabel('Actual Torque (Nm)')
        axes[1, 0].set_ylabel('Predicted Torque (Nm)')
        axes[1, 0].grid(True)

        # 4. Parameter importance
        coefficients = result['coefficients']
        importance = np.abs(coefficients)
        sorted_idx = np.argsort(importance)[::-1]

        axes[1, 1].bar(range(len(coefficients)), importance[sorted_idx])
        axes[1, 1].set_xticks(range(len(coefficients)))
        axes[1, 1].set_xticklabels([self.feature_names[i] for i in sorted_idx], rotation=45)
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
        保存所有关节的辨识结果

        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存所有结果
        all_results = {
            'n_joints': len(self.identification_results),
            'feature_names': self.feature_names,
            'results': self.identification_results
        }

        param_file = os.path.join(output_dir, f"multi_joint_params_{timestamp}.npz")
        np.savez(param_file, **all_results)

        # 生成综合报告
        report_file = os.path.join(output_dir, f"multi_joint_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Multi-Joint Dynamics Identification Report ===\n\n")
            f.write(f"Generation Time: {datetime.now()}\n")
            f.write(f"Number of Joints: {len(self.identification_results)}\n\n")

            for result in self.identification_results:
                joint_id = result['joint_id']
                f.write(f"【Joint {joint_id}】\n")
                f.write(f"  Samples: {result['n_samples']}\n")
                f.write(f"  RMS Error: {result['rmse']:.6f}\n")
                f.write(f"  R² Score: {result['r2']:.4f}\n")
                f.write(f"  Position Range: [{result['position_range'][0]:.6f}, {result['position_range'][1]:.6f}] rad\n")
                f.write(f"  Velocity Range: [{result['velocity_range'][0]:.6f}, {result['velocity_range'][1]:.6f}] rad/s\n")
                f.write(f"  Torque Range: [{result['torque_range'][0]:.3f}, {result['torque_range'][1]:.3f}] Nm\n")
                f.write(f"  Regularization: {result['regularization']}\n")

                f.write(f"  Key Parameters:\n")
                f.write(f"    sin_pos: {result['coefficients'][8]:.6f}\n")
                f.write(f"    cos_pos: {result['coefficients'][9]:.6f}\n")
                f.write(f"    intercept: {result['intercept']:.6f}\n\n")

            # 总体统计
            avg_rmse = np.mean([r['rmse'] for r in self.identification_results])
            avg_r2 = np.mean([r['r2'] for r in self.identification_results])

            f.write("【Overall Statistics】\n")
            f.write(f"  Average RMS Error: {avg_rmse:.6f}\n")
            f.write(f"  Average R² Score: {avg_r2:.4f}\n")

        print(f"结果已保存到: {output_dir}")
        return param_file, report_file

def load_data_from_csv(csv_file):
    """
    从CSV文件加载多关节数据

    Args:
        csv_file: CSV文件路径

    Returns:
        data: 包含所有关节数据的DataFrame
    """
    print(f"Loading data: {csv_file}")
    data = pd.read_csv(csv_file)

    print(f"Data points: {len(data)}")

    # 显示每个关节的数据范围
    for i in range(1, 6):
        pos_col = f'm{i}_pos_actual'
        if pos_col in data.columns:
            pos_data = data[pos_col]
            pos_range = np.degrees([pos_data.min(), pos_data.max()])
            pos_std = np.degrees(pos_data.std())

            print(f"Joint {i}:")
            print(f"  Position range: [{pos_range[0]:.1f}° ~ {pos_range[1]:.1f}°]")
            print(f"  Position std: {pos_std:.1f}°")

    return data

def run_multi_joint_identification(csv_file, max_points=None, regularization=0.1):
    """
    运行多关节辨识

    Args:
        csv_file: CSV文件路径
        max_points: 最大数据点数
        regularization: 正则化参数
    """
    # 加载数据
    data = load_data_from_csv(csv_file)

    # 限制数据点数
    if max_points and len(data) > max_points:
        print(f"Limiting data points to: {max_points}")
        data = data.sample(n=max_points, random_state=42).reset_index(drop=True)

    # 创建辨识器并执行辨识
    identifier = MultiJointIdentification(n_joints=6)
    results = identifier.identify_all_joints(data, regularization=regularization)

    # 绘制每个关节的结果
    for i, result in enumerate(results):
        joint_id = result['joint_id']
        pos_col = f'm{joint_id}_pos_actual'
        vel_col = f'm{joint_id}_vel_actual'
        acc_col = f'm{joint_id}_acc_actual'
        torque_col = f'm{joint_id}_torque'

        q = data[pos_col].values
        dq = data[vel_col].values
        ddq = data[acc_col].values
        tau = data[torque_col].values

        print(f"\nPlotting results for Joint {joint_id}...")
        identifier.plot_joint_results(joint_id, q, dq, ddq, tau)

    # 保存结果
    param_file, report_file = identifier.save_results()

    return identifier

if __name__ == "__main__":
    # 使用转换后的日志数据
    data_file = "/Users/lr-2002/project/instantcreation/IC_arm_control/dyn_ana/merged_log_data.csv"

    print("Multi-Joint Dynamics Identification\n")

    # 运行多关节辨识
    identifier = run_multi_joint_identification(
        data_file,
        # max_points=15000,  # 限制数据点数以控制计算时间
        regularization=0.05
    )

    print("\n✓ Multi-joint identification completed!")