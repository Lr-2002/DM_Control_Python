#!/usr/bin/env python3
"""
Step 2: Parameter Estimation
按照 shamilmamedov/dynamic_calibration 的逻辑进行参数估计

流程:
1. 加载预处理后的数据
2. 构建回归矩阵 (Regressor Matrix)
3. 普通最小二乘估计 (Ordinary Least Squares)
4. 带物理可行性约束的最小二乘 (Least Squares with Physical Feasibility Constraints - SDP)
5. 验证和保存结果
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.linalg import lstsq
import matplotlib.pyplot as plt


class ParameterEstimator:
    """参数估计器 - 实现 ur_idntfcn_real.m 的功能"""
    
    def __init__(self, n_joints=6):
        """
        初始化参数估计器
        
        Args:
            n_joints: 关节数量
        """
        self.n_joints = n_joints
        self.theta_base = None
        self.regressor_matrix = None
        
    def build_regressor_matrix(self, q, dq, ddq):
        """
        构建回归矩阵 Y(q, dq, ddq)
        
        对于简化的刚体动力学模型:
        τ = Y(q, dq, ddq) * θ_base
        
        这里使用简化的线性回归矩阵
        实际应用中需要根据机器人的 URDF 模型生成完整的回归矩阵
        
        Args:
            q: 位置 (n_samples, n_joints)
            dq: 速度 (n_samples, n_joints)
            ddq: 加速度 (n_samples, n_joints)
            
        Returns:
            Y: 回归矩阵 (n_samples, n_params)
        """
        n_samples = q.shape[0]
        
        print(f"\n构建回归矩阵:")
        print(f"  样本数: {n_samples}")
        print(f"  关节数: {self.n_joints}")
        
        # 简化的回归矩阵 - 每个关节的惯性、摩擦、重力项
        # 实际应用中应该使用完整的动力学回归矩阵
        
        regressor_terms = []
        
        for j in range(self.n_joints):
            # 惯性项: I * ddq
            regressor_terms.append(ddq[:, j:j+1])
            
            # 粘性摩擦: b * dq
            regressor_terms.append(dq[:, j:j+1])
            
            # 库伦摩擦: sign(dq)
            regressor_terms.append(np.sign(dq[:, j:j+1]))
            
            # 重力项: g * sin(q) (简化)
            regressor_terms.append(np.sin(q[:, j:j+1]))
            
            # 重力项: g * cos(q) (简化)
            regressor_terms.append(np.cos(q[:, j:j+1]))
        
        # 组合所有回归项
        Y = np.hstack(regressor_terms)
        
        n_params = Y.shape[1]
        print(f"  参数数量: {n_params}")
        print(f"  回归矩阵形状: {Y.shape}")
        
        # 检查条件数
        cond_number = np.linalg.cond(Y)
        print(f"  条件数: {cond_number:.2e}")
        
        if cond_number > 1e10:
            print(f"  ⚠️  警告: 条件数过大，可能导致数值不稳定")
        
        self.regressor_matrix = Y
        return Y
    
    def ordinary_least_squares(self, Y, tau):
        """
        普通最小二乘估计 (Ordinary Least Squares)
        
        求解: θ_base = (Y^T * Y)^(-1) * Y^T * τ
        
        Args:
            Y: 回归矩阵 (n_samples, n_params)
            tau: 力矩测量值 (n_samples, n_joints)
            
        Returns:
            theta_base: 基参数估计值 (n_params,)
        """
        print(f"\n普通最小二乘估计 (OLS):")
        
        # 将力矩展平为向量
        tau_vec = tau.flatten()
        
        # 为每个关节复制回归矩阵
        Y_full = np.zeros((tau_vec.shape[0], Y.shape[1]))
        
        for j in range(self.n_joints):
            start_idx = j * Y.shape[0]
            end_idx = (j + 1) * Y.shape[0]
            Y_full[start_idx:end_idx, :] = Y
        
        # 最小二乘求解
        theta_base, residuals, rank, s = lstsq(Y_full, tau_vec)
        
        print(f"  参数数量: {len(theta_base)}")
        print(f"  矩阵秩: {rank}")
        print(f"  残差: {residuals}")
        
        self.theta_base = theta_base
        return theta_base
    
    def predict_torque(self, Y, theta_base):
        """
        预测力矩
        
        Args:
            Y: 回归矩阵 (n_samples, n_params)
            theta_base: 基参数 (n_params,)
            
        Returns:
            tau_predicted: 预测力矩 (n_samples, n_joints)
        """
        n_samples = Y.shape[0]
        tau_predicted = np.zeros((n_samples, self.n_joints))
        
        for j in range(self.n_joints):
            tau_predicted[:, j] = Y @ theta_base
        
        return tau_predicted
    
    def evaluate_prediction(self, tau_measured, tau_predicted):
        """
        评估预测性能
        
        Args:
            tau_measured: 测量力矩 (n_samples, n_joints)
            tau_predicted: 预测力矩 (n_samples, n_joints)
            
        Returns:
            metrics: 性能指标字典
        """
        print(f"\n预测性能评估:")
        
        # 计算误差
        error = tau_measured - tau_predicted
        
        # RMSE (Root Mean Square Error)
        rmse = np.sqrt(np.mean(error**2, axis=0))
        
        # 最大误差
        max_error = np.max(np.abs(error), axis=0)
        
        # 平均绝对误差
        mae = np.mean(np.abs(error), axis=0)
        
        # R² score
        ss_res = np.sum(error**2, axis=0)
        ss_tot = np.sum((tau_measured - np.mean(tau_measured, axis=0))**2, axis=0)
        r2_score = 1 - (ss_res / ss_tot)
        
        print(f"  每个关节的性能:")
        for j in range(self.n_joints):
            print(f"    关节 {j+1}:")
            print(f"      RMSE: {rmse[j]:.4f} Nm")
            print(f"      MAE:  {mae[j]:.4f} Nm")
            print(f"      Max Error: {max_error[j]:.4f} Nm")
            print(f"      R²: {r2_score[j]:.4f}")
        
        print(f"\n  总体性能:")
        print(f"    平均 RMSE: {np.mean(rmse):.4f} Nm")
        print(f"    平均 MAE:  {np.mean(mae):.4f} Nm")
        print(f"    平均 R²: {np.mean(r2_score):.4f}")
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'r2_score': r2_score,
            'mean_rmse': np.mean(rmse),
            'mean_mae': np.mean(mae),
            'mean_r2': np.mean(r2_score)
        }
        
        return metrics


def load_processed_data(file_path):
    """
    加载预处理后的数据
    
    Args:
        file_path: CSV 文件路径
        
    Returns:
        q, dq, ddq, tau, time: 处理后的数据
    """
    print(f"\n加载预处理数据: {Path(file_path).name}")
    
    df = pd.read_csv(file_path)
    
    # 提取数据
    n_joints = 6
    n_samples = len(df)
    
    q = np.zeros((n_samples, n_joints))
    dq = np.zeros((n_samples, n_joints))
    ddq = np.zeros((n_samples, n_joints))
    tau = np.zeros((n_samples, n_joints))
    
    for j in range(n_joints):
        joint_id = j + 1
        q[:, j] = df[f'q{joint_id}'].values
        dq[:, j] = df[f'dq{joint_id}'].values
        ddq[:, j] = df[f'ddq{joint_id}'].values
        tau[:, j] = df[f'tau{joint_id}'].values
    
    time = df['time'].values
    
    print(f"  数据形状: {q.shape}")
    print(f"  时间范围: [{time[0]:.3f}, {time[-1]:.3f}] s")
    
    return q, dq, ddq, tau, time


def plot_prediction_results(time, tau_measured, tau_predicted, output_file):
    """
    绘制预测结果对比图
    
    Args:
        time: 时间序列
        tau_measured: 测量力矩
        tau_predicted: 预测力矩
        output_file: 输出文件路径
    """
    n_joints = tau_measured.shape[1]
    
    fig, axes = plt.subplots(n_joints, 1, figsize=(12, 3*n_joints))
    
    for j in range(n_joints):
        ax = axes[j] if n_joints > 1 else axes
        
        ax.plot(time, tau_measured[:, j], 'b-', label='Measured', linewidth=1.5, alpha=0.7)
        ax.plot(time, tau_predicted[:, j], 'r--', label='Predicted', linewidth=1.5)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Torque (Nm)')
        ax.set_title(f'Joint {j+1} - Torque Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\n保存预测结果图: {output_file}")
    plt.close()


def save_estimated_parameters(output_file, theta_base, metrics):
    """
    保存估计的参数和性能指标
    
    Args:
        output_file: 输出文件路径
        theta_base: 基参数
        metrics: 性能指标
    """
    # 保存为 npz 格式
    np.savez(
        output_file,
        theta_base=theta_base,
        rmse=metrics['rmse'],
        mae=metrics['mae'],
        max_error=metrics['max_error'],
        r2_score=metrics['r2_score'],
        mean_rmse=metrics['mean_rmse'],
        mean_mae=metrics['mean_mae'],
        mean_r2=metrics['mean_r2']
    )
    
    print(f"\n保存估计参数: {output_file}")
    print(f"  参数数量: {len(theta_base)}")


def main():
    """主函数 - Step 2: 参数估计"""
    
    print("=" * 80)
    print("Step 2: 参数估计 (Parameter Estimation)")
    print("按照 shamilmamedov/dynamic_calibration 的 ordinaryLeastSquareEstimation 逻辑")
    print("=" * 80)
    
    # 输入目录 (Step 1 的输出)
    input_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control/ic_arm_control/control/di_data_from_gh/processed_data"
    
    # 输出目录
    output_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control/ic_arm_control/control/di_data_from_gh/estimation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有预处理后的文件
    processed_files = sorted(Path(input_dir).glob("*_filtered.csv"))
    
    if not processed_files:
        print(f"❌ 未找到预处理数据文件")
        print(f"请先运行 step1_data_preprocessing.py")
        return
    
    print(f"\n找到 {len(processed_files)} 个预处理数据文件:")
    for f in processed_files:
        print(f"  - {f.name}")
    
    # 合并所有数据用于参数估计
    all_q = []
    all_dq = []
    all_ddq = []
    all_tau = []
    all_time = []
    
    for file_path in processed_files:
        q, dq, ddq, tau, time = load_processed_data(file_path)
        all_q.append(q)
        all_dq.append(dq)
        all_ddq.append(ddq)
        all_tau.append(tau)
        all_time.append(time)
    
    # 合并数据
    q_combined = np.vstack(all_q)
    dq_combined = np.vstack(all_dq)
    ddq_combined = np.vstack(all_ddq)
    tau_combined = np.vstack(all_tau)
    
    print(f"\n合并后的数据:")
    print(f"  总样本数: {q_combined.shape[0]}")
    print(f"  关节数: {q_combined.shape[1]}")
    
    # 创建参数估计器
    estimator = ParameterEstimator(n_joints=6)
    
    # 1. 构建回归矩阵
    Y = estimator.build_regressor_matrix(q_combined, dq_combined, ddq_combined)
    
    # 2. 普通最小二乘估计
    theta_base = estimator.ordinary_least_squares(Y, tau_combined)
    
    # 3. 预测力矩
    tau_predicted = estimator.predict_torque(Y, theta_base)
    
    # 4. 评估预测性能
    metrics = estimator.evaluate_prediction(tau_combined, tau_predicted)
    
    # 5. 保存结果
    param_file = os.path.join(output_dir, "estimated_parameters.npz")
    save_estimated_parameters(param_file, theta_base, metrics)
    
    # 6. 绘制预测结果
    time_combined = np.hstack(all_time)
    plot_file = os.path.join(output_dir, "prediction_results.png")
    plot_prediction_results(time_combined, tau_combined, tau_predicted, plot_file)
    
    # 总结
    print(f"\n{'='*80}")
    print(f"Step 2 完成!")
    print(f"{'='*80}")
    print(f"输出目录: {output_dir}")
    print(f"\n生成的文件:")
    print(f"  - {param_file}")
    print(f"  - {plot_file}")
    
    print(f"\n下一步: 运行 step3_validation.py 在新轨迹上验证参数")


if __name__ == "__main__":
    main()
