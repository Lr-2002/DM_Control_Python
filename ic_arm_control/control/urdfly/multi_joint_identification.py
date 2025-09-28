#!/usr/bin/env python3
"""
多关节动力学辨识（支持5个关节）
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
        为指定关节创建动力学特征 - 改进版本

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
        features.append(dq)  # 速度相关（粘性摩擦）
        features.append(ddq)  # 加速度相关（惯性）

        # 摩擦力特征（使用连续函数代替sign）
        features.append(np.tanh(dq * 10))  # 平滑的速度方向特征

        # 重力相关特征（只使用sin避免共线性）
        features.append(np.sin(q))  # 位置相关重力

        # 非线性动力学特征
        features.append(dq**2)  # 速度平方（空气阻力等）
        features.append(dq**3)  # 速度立方（高阶摩擦）

        # 耦合特征（简化版本）
        features.append(q * dq)  # 位置-速度耦合

        # 惯性变化特征
        features.append(np.cos(q) * ddq)  # 位置相关的惯性变化

        # 组合特征矩阵
        X = np.column_stack(features)

        # 特征名称（只需设置一次）
        if not self.feature_names:
            self.feature_names = [
                'constant', 'velocity', 'acceleration', 'smooth_velocity_sign',
                'sin_pos', 'vel_squared', 'vel_cubed', 'pos_vel', 'cos_pos_acc'
            ]

        return X

    def preprocess_data(self, q, dq, ddq, tau, joint_id):
        """
        数据预处理 - 过滤低质量数据

        Args:
            q: 位置数组
            dq: 速度数组
            ddq: 加速度数组
            tau: 力矩数组
            joint_id: 关节ID

        Returns:
            过滤后的数据
        """
        print(f"  原始数据点: {len(q)}")

        # 移除零加速度数据点（这些对惯性辨识无用）
        zero_acc_mask = np.abs(ddq) < 1e-6
        if np.sum(zero_acc_mask) > len(q) * 0.5:
            print(f"  警告: {np.sum(zero_acc_mask)/len(q)*100:.1f}% 的加速度接近零")

        # 移除异常值
        tau_iqr = np.percentile(tau, [25, 75])
        tau_range = tau_iqr[1] - tau_iqr[0]
        tau_lower = tau_iqr[0] - 3 * tau_range
        tau_upper = tau_iqr[1] + 3 * tau_range
        outlier_mask = (tau < tau_lower) | (tau > tau_upper)

        # 移除静力矩数据点（加速度和速度都很小）
        static_mask = (np.abs(dq) < 0.01) & (np.abs(ddq) < 0.01)

        # 组合过滤条件
        valid_mask = ~(zero_acc_mask | outlier_mask | static_mask)

        if np.sum(valid_mask) < 100:  # 如果有效数据太少，放宽条件
            print(f"  有效数据不足 ({np.sum(valid_mask)}), 放宽过滤条件")
            valid_mask = ~outlier_mask  # 只移除明显的异常值

        q_filtered = q[valid_mask]
        dq_filtered = dq[valid_mask]
        ddq_filtered = ddq[valid_mask]
        tau_filtered = tau[valid_mask]

        print(f"  过滤后数据点: {len(q_filtered)}")
        print(f"  位置范围: [{np.degrees(q_filtered.min()):.1f}°, {np.degrees(q_filtered.max()):.1f}°]")
        print(f"  速度范围: [{dq_filtered.min():.3f}, {dq_filtered.max():.3f}] rad/s")
        print(f"  加速度范围: [{ddq_filtered.min():.3f}, {ddq_filtered.max():.3f}] rad/s²")

        return q_filtered, dq_filtered, ddq_filtered, tau_filtered

    def identify_joint(self, joint_id, q, dq, ddq, tau, regularization=0.1):
        """
        辨识单个关节的动力学参数 - 改进版本

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

        # 数据预处理
        q_proc, dq_proc, ddq_proc, tau_proc = self.preprocess_data(q, dq, ddq, tau, joint_id)

        if len(q_proc) < 50:
            print(f"  警告: 有效数据不足 ({len(q_proc)} 个点)，辨识结果可能不可靠")
            return None, None, None

        # 创建特征矩阵
        X = self.create_features(q_proc, dq_proc, ddq_proc, joint_id)
        y = tau_proc

        print(f"特征矩阵形状: {X.shape}")

        # 使用鲁棒缩放处理异常值
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # 模型选择和超参数优化
        models = {
            'Ridge': Ridge(),
            'Lasso': Lasso(max_iter=2000),
            'ElasticNet': ElasticNet(max_iter=2000)
        }

        param_grid = {
            'Ridge': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
            'Lasso': {'alpha': [0.001, 0.01, 0.1, 1.0]},
            'ElasticNet': {'alpha': [0.001, 0.01, 0.1], 'l1_ratio': [0.1, 0.5, 0.9]}
        }

        best_model = None
        best_score = -np.inf
        best_params = None
        best_model_name = None

        # 网格搜索选择最佳模型
        for model_name in models:
            print(f"  测试 {model_name} 模型...")
            grid_search = GridSearchCV(
                models[model_name],
                param_grid[model_name],
                cv=5,
                scoring='r2',
                n_jobs=-1
            )
            grid_search.fit(X_scaled, y)

            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_model_name = model_name

        print(f"  最佳模型: {best_model_name}")
        print(f"  最佳参数: {best_params}")
        print(f"  交叉验证R²: {best_score:.4f}")

        # 使用最佳模型进行预测
        tau_pred = best_model.predict(X_scaled)

        # 计算误差
        mse = mean_squared_error(y, tau_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, tau_pred)

        print(f"  RMS误差: {rmse:.6f}")
        print(f"  R²分数: {r2:.4f}")

        # 输出重要参数
        print(f"\n  辨识参数:")
        for i, (name, coef) in enumerate(zip(self.feature_names, best_model.coef_)):
            if abs(coef) > 1e-6:  # 只显示非零参数
                print(f"    {name}: {coef:.6f}")
        print(f"    截距: {best_model.intercept_:.6f}")

        # 保存结果
        result = {
            'joint_id': joint_id,
            'coefficients': best_model.coef_,
            'intercept': best_model.intercept_,
            'feature_names': self.feature_names,
            'rmse': rmse,
            'r2': r2,
            'model_name': best_model_name,
            'best_params': best_params,
            'cv_score': best_score,
            'n_samples': len(q_proc),
            'n_original_samples': len(q),
            'position_range': [q_proc.min(), q_proc.max()],
            'velocity_range': [dq_proc.min(), dq_proc.max()],
            'torque_range': [tau_proc.min(), tau_proc.max()]
        }

        return best_model, scaler, result

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

            if result is not None:
                # 保存模型和结果
                self.models.append(model)
                self.scalers.append(scaler)
                self.identification_results.append(result)
            else:
                print(f"  关节 {joint_id} 辨识失败")
                self.identification_results.append(None)

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
            f.write(f"=== Multi-Joint Dynamics Identification Report (Improved) ===\n\n")
            f.write(f"Generation Time: {datetime.now()}\n")
            f.write(f"Number of Joints: {len(self.identification_results)}\n\n")

            for result in self.identification_results:
                if result is None:
                    continue

                joint_id = result['joint_id']
                f.write(f"【Joint {joint_id}】\n")
                f.write(f"  Model: {result.get('model_name', 'Unknown')}\n")
                f.write(f"  Best Parameters: {result.get('best_params', {})}\n")
                f.write(f"  Cross-validation R²: {result.get('cv_score', 'N/A'):.4f}\n")
                f.write(f"  Samples: {result['n_samples']} (from {result['n_original_samples']})\n")
                f.write(f"  RMS Error: {result['rmse']:.6f}\n")
                f.write(f"  R² Score: {result['r2']:.4f}\n")
                f.write(f"  Position Range: [{np.degrees(result['position_range'][0]):.1f}°, {np.degrees(result['position_range'][1]):.1f}°]\n")
                f.write(f"  Velocity Range: [{result['velocity_range'][0]:.3f}, {result['velocity_range'][1]:.3f}] rad/s\n")
                f.write(f"  Torque Range: [{result['torque_range'][0]:.3f}, {result['torque_range'][1]:.3f}] Nm\n")

                # 重要参数
                f.write(f"  Significant Parameters:\n")
                for i, (name, coef) in enumerate(zip(result['feature_names'], result['coefficients'])):
                    if abs(coef) > 1e-6:  # 只显示非零参数
                        f.write(f"    {name}: {coef:.6f}\n")
                f.write(f"    intercept: {result['intercept']:.6f}\n\n")

            # 总体统计
            valid_results = [r for r in self.identification_results if r is not None]
            if valid_results:
                avg_rmse = np.mean([r['rmse'] for r in valid_results])
                avg_r2 = np.mean([r['r2'] for r in valid_results])
                avg_cv = np.mean([r.get('cv_score', r['r2']) for r in valid_results])

                f.write("【Overall Statistics】\n")
                f.write(f"  Average RMS Error: {avg_rmse:.6f}\n")
                f.write(f"  Average R² Score: {avg_r2:.4f}\n")
                f.write(f"  Average CV R² Score: {avg_cv:.4f}\n")
                f.write(f"  Successfully Identified Joints: {len(valid_results)}/{self.n_joints}\n")

                # 模型使用统计
                model_usage = {}
                for r in valid_results:
                    model_name = r.get('model_name', 'Unknown')
                    model_usage[model_name] = model_usage.get(model_name, 0) + 1
                f.write(f"  Model Usage: {model_usage}\n")

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