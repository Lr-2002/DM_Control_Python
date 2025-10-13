#!/usr/bin/env python3
"""
Step 3: Validation
按照 shamilmamedov/dynamic_calibration 的逻辑进行验证

流程:
1. 加载估计的参数
2. 在新的轨迹上测试
3. 预测力矩并与测量值对比
4. 生成验证报告
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
from step2_parameter_estimation import ParameterEstimator, load_processed_data


def validate_on_trajectory(estimator, theta_base, q, dq, ddq, tau_measured, trajectory_name="Test"):
    """
    在轨迹上验证参数
    
    Args:
        estimator: 参数估计器
        theta_base: 估计的基参数
        q, dq, ddq: 轨迹数据
        tau_measured: 测量力矩
        trajectory_name: 轨迹名称
        
    Returns:
        metrics: 验证指标
    """
    print(f"\n{'='*80}")
    print(f"验证轨迹: {trajectory_name}")
    print(f"{'='*80}")
    
    # 1. 构建回归矩阵
    Y = estimator.build_regressor_matrix(q, dq, ddq)
    
    # 2. 预测力矩
    tau_predicted = estimator.predict_torque(Y, theta_base)
    
    # 3. 评估性能
    metrics = estimator.evaluate_prediction(tau_measured, tau_predicted)
    
    return tau_predicted, metrics


def plot_validation_comparison(time, tau_measured, tau_predicted, trajectory_name, output_file):
    """
    绘制验证对比图
    
    Args:
        time: 时间序列
        tau_measured: 测量力矩
        tau_predicted: 预测力矩
        trajectory_name: 轨迹名称
        output_file: 输出文件路径
    """
    n_joints = tau_measured.shape[1]
    
    fig, axes = plt.subplots(n_joints, 1, figsize=(14, 3*n_joints))
    
    for j in range(n_joints):
        ax = axes[j] if n_joints > 1 else axes
        
        # 绘制测量值和预测值
        ax.plot(time, tau_measured[:, j], 'b-', label='Measured', linewidth=1.5, alpha=0.7)
        ax.plot(time, tau_predicted[:, j], 'r--', label='Predicted', linewidth=1.5)
        
        # 绘制误差
        error = tau_measured[:, j] - tau_predicted[:, j]
        ax.fill_between(time, 0, error, alpha=0.3, color='gray', label='Error')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Torque (Nm)')
        ax.set_title(f'Joint {j+1} - Validation on {trajectory_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        rmse = np.sqrt(np.mean(error**2))
        max_error = np.max(np.abs(error))
        ax.text(0.02, 0.98, f'RMSE: {rmse:.4f} Nm\nMax Error: {max_error:.4f} Nm',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\n保存验证对比图: {output_file}")
    plt.close()


def generate_validation_report(all_metrics, output_file):
    """
    生成验证报告
    
    Args:
        all_metrics: 所有轨迹的验证指标
        output_file: 输出文件路径
    """
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("动力学参数辨识验证报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("验证方法: 按照 shamilmamedov/dynamic_calibration 的逻辑\n\n")
        
        for traj_name, metrics in all_metrics.items():
            f.write(f"\n轨迹: {traj_name}\n")
            f.write("-" * 80 + "\n")
            
            f.write(f"每个关节的性能:\n")
            for j in range(len(metrics['rmse'])):
                f.write(f"  关节 {j+1}:\n")
                f.write(f"    RMSE: {metrics['rmse'][j]:.4f} Nm\n")
                f.write(f"    MAE:  {metrics['mae'][j]:.4f} Nm\n")
                f.write(f"    Max Error: {metrics['max_error'][j]:.4f} Nm\n")
                f.write(f"    R²: {metrics['r2_score'][j]:.4f}\n")
            
            f.write(f"\n总体性能:\n")
            f.write(f"  平均 RMSE: {metrics['mean_rmse']:.4f} Nm\n")
            f.write(f"  平均 MAE:  {metrics['mean_mae']:.4f} Nm\n")
            f.write(f"  平均 R²: {metrics['mean_r2']:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("验证完成\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n保存验证报告: {output_file}")


def main():
    """主函数 - Step 3: 验证"""
    
    print("=" * 80)
    print("Step 3: 验证 (Validation)")
    print("按照 shamilmamedov/dynamic_calibration 的验证逻辑")
    print("=" * 80)
    
    # 加载估计的参数
    param_file = "/Users/lr-2002/project/instantcreation/IC_arm_control/ic_arm_control/control/di_data_from_gh/estimation_results/estimated_parameters.npz"
    
    if not os.path.exists(param_file):
        print(f"❌ 未找到参数文件: {param_file}")
        print(f"请先运行 step2_parameter_estimation.py")
        return
    
    print(f"\n加载估计参数: {param_file}")
    param_data = np.load(param_file)
    theta_base = param_data['theta_base']
    
    print(f"  参数数量: {len(theta_base)}")
    print(f"  训练 RMSE: {param_data['mean_rmse']:.4f} Nm")
    print(f"  训练 R²: {param_data['mean_r2']:.4f}")
    
    # 验证数据目录
    validation_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control/ic_arm_control/control/di_data_from_gh/processed_data"
    output_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control/ic_arm_control/control/di_data_from_gh/validation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找验证数据
    validation_files = sorted(Path(validation_dir).glob("*_filtered.csv"))
    
    if not validation_files:
        print(f"❌ 未找到验证数据")
        return
    
    print(f"\n找到 {len(validation_files)} 个验证轨迹:")
    for f in validation_files:
        print(f"  - {f.name}")
    
    # 创建参数估计器
    estimator = ParameterEstimator(n_joints=6)
    
    # 在每个轨迹上验证
    all_metrics = {}
    
    for file_path in validation_files:
        trajectory_name = file_path.stem.replace('_filtered', '')
        
        # 加载数据
        q, dq, ddq, tau_measured, time = load_processed_data(file_path)
        
        # 验证
        tau_predicted, metrics = validate_on_trajectory(
            estimator, theta_base, q, dq, ddq, tau_measured, trajectory_name
        )
        
        all_metrics[trajectory_name] = metrics
        
        # 绘制对比图
        plot_file = os.path.join(output_dir, f"{trajectory_name}_validation.png")
        plot_validation_comparison(time, tau_measured, tau_predicted, trajectory_name, plot_file)
    
    # 生成验证报告
    report_file = os.path.join(output_dir, "validation_report.txt")
    generate_validation_report(all_metrics, report_file)
    
    # 总结
    print(f"\n{'='*80}")
    print(f"Step 3 完成!")
    print(f"{'='*80}")
    print(f"输出目录: {output_dir}")
    
    print(f"\n验证结果总结:")
    for traj_name, metrics in all_metrics.items():
        print(f"  {traj_name}:")
        print(f"    平均 RMSE: {metrics['mean_rmse']:.4f} Nm")
        print(f"    平均 R²: {metrics['mean_r2']:.4f}")
    
    print(f"\n如果预测不满意，建议:")
    print(f"  1. 调整 step1_data_preprocessing.py 中的滤波器参数")
    print(f"  2. 收集更多激励轨迹数据")
    print(f"  3. 检查回归矩阵的构建是否正确")


if __name__ == "__main__":
    main()
