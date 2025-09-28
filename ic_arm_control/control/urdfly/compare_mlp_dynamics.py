#!/usr/bin/env python3
"""
比较MLP重力补偿与动力学辨识结果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# 添加路径
current_dir = Path(__file__).parent
mlp_dir = current_dir.parent / "mlp_compensation"
sys.path.insert(0, str(mlp_dir))

def load_dynamics_identification_results():
    """加载动力学辨识结果"""
    results = {}

    # 查找辨识结果文件
    results_dir = current_dir / "identification_results"
    if results_dir.exists():
        npz_files = list(results_dir.glob("joint1_*.npz"))
        if npz_files:
            # 使用最新的结果文件
            latest_file = max(npz_files, key=os.path.getctime)
            data = np.load(latest_file)
            results['dynamics'] = {
                'coefficients': data['coefficients'],
                'intercept': data['intercept'],
                'feature_names': data['feature_names'],
                'rmse': data['rmse'],
                'r2': data['r2'],
                'n_samples': data['n_samples']
            }
            print(f"加载动力学辨识结果: {latest_file.name}")

    return results

def load_mlp_performance():
    """加载MLP性能数据"""
    results = {}

    try:
        # 导入MLP模块
        from mlp_gravity_integrator import MLPGravityCompensation

        # 创建MLP实例 - 使用完整的路径
        mlp_dir = current_dir.parent / "mlp_compensation"
        model_files = [
            mlp_dir / "mlp_gravity_model_improved.pkl",
            mlp_dir / "mlp_gravity_model_new.pkl",
            mlp_dir / "mlp_gravity_model.pkl"
        ]

        mlp_gc = None
        for model_file in model_files:
            try:
                mlp_gc = MLPGravityCompensation(model_path=str(model_file))
                if mlp_gc.is_initialized:
                    print(f"成功加载MLP模型: {model_file.name}")
                    break
            except Exception as e:
                print(f"加载失败 {model_file.name}: {e}")
                continue

        if mlp_gc is None or not mlp_gc.is_initialized:
            print("所有MLP模型都加载失败")
            return results

        if mlp_gc.is_initialized:
            # 获取性能统计
            stats = mlp_gc.get_performance_stats()
            results['mlp'] = {
                'total_predictions': stats['total_predictions'],
                'avg_prediction_time': stats['avg_prediction_time'],
                'frequency_hz': stats['frequency_hz'],
                'is_initialized': mlp_gc.is_initialized
            }
            print(f"MLP初始化成功，预测频率: {stats['frequency_hz']:.1f} Hz")

    except Exception as e:
        print(f"MLP加载失败: {e}")

    return results

def create_test_positions(n_samples=1000):
    """创建测试位置"""
    # 在关节1的运动范围内生成测试位置
    positions = np.linspace(-0.65, 0.18, n_samples)  # 基于实际数据范围
    return positions

def compare_predictions(dynamics_results, mlp_results, test_positions):
    """比较两种方法的预测结果"""
    if 'dynamics' not in dynamics_results or 'mlp' not in mlp_results:
        print("缺少必要的结果数据")
        return None

    print("\n=== MLP与动力学辨识比较 ===")

    # 动力学辨识预测
    print("1. 动力学辨识预测...")
    dynamics_coefs = dynamics_results['dynamics']['coefficients']
    dynamics_intercept = dynamics_results['dynamics']['intercept']

    # 使用简化的动力学模型（仅考虑重力和速度阻尼）
    dynamics_torques = []
    for q in test_positions:
        # 简化模型：主要考虑重力项（sin和cos）
        gravity_term = dynamics_coefs[8] * np.sin(q) + dynamics_coefs[9] * np.cos(q)
        velocity_term = dynamics_coefs[1] * 0  # 假设速度为0（静态重力补偿）
        torque = gravity_term + velocity_term + dynamics_intercept
        dynamics_torques.append(torque)

    dynamics_torques = np.array(dynamics_torques)

    # MLP预测
    print("2. MLP预测...")
    try:
        from mlp_gravity_integrator import MLPGravityCompensation

        # 尝试加载可用的MLP模型
        mlp_dir = current_dir.parent / "mlp_compensation"
        model_files = [
            mlp_dir / "mlp_gravity_model_improved.pkl",
            mlp_dir / "mlp_gravity_model_new.pkl",
            mlp_dir / "mlp_gravity_model.pkl"
        ]

        mlp_gc = None
        for model_file in model_files:
            try:
                mlp_gc = MLPGravityCompensation(model_path=str(model_file))
                if mlp_gc.is_initialized:
                    break
            except:
                continue

        if mlp_gc and mlp_gc.is_initialized:
            # 创建完整的关节位置向量（其他关节设为0）
            joint_positions = np.zeros((len(test_positions), 6))
            joint_positions[:, 0] = test_positions  # 只设置关节1

            mlp_torques = mlp_gc.get_gravity_compensation_torque(joint_positions)
            mlp_torques = mlp_torques[:, 0]  # 只取关节1
        else:
            print("MLP未初始化，使用零力矩")
            mlp_torques = np.zeros_like(test_positions)

    except Exception as e:
        print(f"MLP预测失败: {e}")
        mlp_torques = np.zeros_like(test_positions)

    return dynamics_torques, mlp_torques

def plot_comparison(test_positions, dynamics_torques, mlp_torques, save_path=None):
    """绘制比较结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Torque curve comparison
    axes[0, 0].plot(np.degrees(test_positions), dynamics_torques, 'b-', label='Dynamics ID', linewidth=2)
    axes[0, 0].plot(np.degrees(test_positions), mlp_torques, 'r-', label='MLP', linewidth=2)
    axes[0, 0].set_xlabel('Joint 1 Position (deg)')
    axes[0, 0].set_ylabel('Gravity Compensation Torque (Nm)')
    axes[0, 0].set_title('Torque Curve Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. Torque difference
    torque_diff = mlp_torques - dynamics_torques
    axes[0, 1].plot(np.degrees(test_positions), torque_diff, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Joint 1 Position (deg)')
    axes[0, 1].set_ylabel('Torque Difference (Nm)')
    axes[0, 1].set_title('MLP - Dynamics ID')
    axes[0, 1].grid(True)

    # 3. Scatter plot
    axes[1, 0].scatter(dynamics_torques, mlp_torques, alpha=0.6, s=10)
    axes[1, 0].plot([dynamics_torques.min(), dynamics_torques.max()],
                   [dynamics_torques.min(), dynamics_torques.max()], 'k--', alpha=0.8)
    axes[1, 0].set_xlabel('Dynamics ID Torque (Nm)')
    axes[1, 0].set_ylabel('MLP Torque (Nm)')
    axes[1, 0].set_title('Torque Correlation')
    axes[1, 0].grid(True)

    # 4. Statistics
    correlation = np.corrcoef(dynamics_torques, mlp_torques)[0, 1]
    mean_diff = np.mean(np.abs(torque_diff))
    std_diff = np.std(torque_diff)

    axes[1, 1].text(0.1, 0.8, f'Correlation: {correlation:.4f}', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.7, f'Mean Abs Diff: {mean_diff:.4f} Nm', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.6, f'Std Diff: {std_diff:.4f} Nm', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.5, f'Max Abs Diff: {np.max(np.abs(torque_diff)):.4f} Nm', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.4, f'Dynamics Range: [{dynamics_torques.min():.2f}, {dynamics_torques.max():.2f}] Nm', transform=axes[1, 1].transAxes, fontsize=10)
    axes[1, 1].text(0.1, 0.3, f'MLP Range: [{mlp_torques.min():.2f}, {mlp_torques.max():.2f}] Nm', transform=axes[1, 1].transAxes, fontsize=10)
    axes[1, 1].set_title('Statistics')
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"比较图表已保存到: {save_path}")

    plt.show()

    return {
        'correlation': correlation,
        'mean_abs_diff': mean_diff,
        'std_diff': std_diff,
        'max_abs_diff': np.max(np.abs(torque_diff))
    }

def generate_comparison_report(dynamics_results, mlp_results, comparison_stats, output_file=None):
    """生成比较报告"""
    if output_file is None:
        output_file = current_dir / "mlp_dynamics_comparison_report.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== MLP重力补偿与动力学辨识比较报告 ===\n\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n\n")

        # 动力学辨识结果
        if 'dynamics' in dynamics_results:
            dyn = dynamics_results['dynamics']
            f.write("【动力学辨识结果】\n")
            f.write(f"  样本数: {dyn['n_samples']}\n")
            f.write(f"  RMS误差: {dyn['rmse']:.6f}\n")
            f.write(f"  R²分数: {dyn['r2']:.4f}\n")
            f.write(f"  主要参数:\n")
            f.write(f"    sin_pos系数: {dyn['coefficients'][8]:.6f}\n")
            f.write(f"    cos_pos系数: {dyn['coefficients'][9]:.6f}\n")
            f.write(f"    截距: {dyn['intercept']:.6f}\n\n")

        # MLP结果
        if 'mlp' in mlp_results:
            mlp = mlp_results['mlp']
            f.write("【MLP结果】\n")
            f.write(f"  初始化状态: {'成功' if mlp['is_initialized'] else '失败'}\n")
            f.write(f"  预测频率: {mlp['frequency_hz']:.1f} Hz\n")
            f.write(f"  平均预测时间: {mlp['avg_prediction_time']:.3f} ms\n")
            f.write(f"  总预测次数: {mlp['total_predictions']:,}\n\n")

        # 比较统计
        if comparison_stats:
            f.write("【比较统计】\n")
            f.write(f"  相关系数: {comparison_stats['correlation']:.4f}\n")
            f.write(f"  平均绝对差异: {comparison_stats['mean_abs_diff']:.4f} Nm\n")
            f.write(f"  差异标准差: {comparison_stats['std_diff']:.4f} Nm\n")
            f.write(f"  最大绝对差异: {comparison_stats['max_abs_diff']:.4f} Nm\n\n")

        # 方法特点分析
        f.write("【方法特点分析】\n")
        f.write("动力学辨识:\n")
        f.write("  ✓ 基于物理模型，参数有明确物理意义\n")
        f.write("  ✓ 计算简单快速，实时性好\n")
        f.write("  ✓ 需要较少的训练数据\n")
        f.write("  ⚠️ 模型简化，可能忽略复杂的非线性效应\n")
        f.write("  ⚠️ 对数据质量和激励要求较高\n\n")

        f.write("MLP:\n")
        f.write("  ✓ 能学习复杂的非线性关系\n")
        f.write("  ✓ 对噪声和异常值鲁棒性较好\n")
        f.write("  ✓ 可自动提取特征，无需手动设计\n")
        f.write("  ⚠️ 计算量较大，需要GPU加速以获得高频性能\n")
        f.write("  ⚠️ 参数物理意义不明确，黑盒模型\n")
        f.write("  ⚠️ 需要大量训练数据\n\n")

        # 建议
        f.write("【建议】\n")
        f.write("1. 实时控制场景：优先使用动力学辨识，计算速度快，实时性有保障\n")
        f.write("2. 离线分析场景：可使用MLP获得更精确的重力补偿效果\n")
        f.write("3. 混合方案：使用动力学辨识作为基础，MLP作为补偿和优化\n")
        f.write("4. 安全考虑：两种方法都应设置适当的力矩限制\n")

    print(f"比较报告已保存到: {output_file}")

def main():
    """主函数"""
    print("MLP重力补偿与动力学辨识比较分析\n")

    # 1. 加载结果
    print("1. 加载辨识结果...")
    dynamics_results = load_dynamics_identification_results()
    mlp_results = load_mlp_performance()

    if not dynamics_results:
        print("❌ 未找到动力学辨识结果")
        return False

    if not mlp_results:
        print("❌ MLP加载失败")
        return False

    # 2. 创建测试数据
    print("\n2. 创建测试位置...")
    test_positions = create_test_positions(n_samples=1000)
    print(f"测试位置范围: {np.degrees([test_positions.min(), test_positions.max()])[0]:.1f}° ~ {np.degrees([test_positions.min(), test_positions.max()])[1]:.1f}°")

    # 3. 比较预测
    print("\n3. 比较预测结果...")
    dynamics_torques, mlp_torques = compare_predictions(dynamics_results, mlp_results, test_positions)

    if dynamics_torques is not None:
        # 4. 绘制比较图表
        print("\n4. 绘制比较图表...")
        comparison_stats = plot_comparison(test_positions, dynamics_torques, mlp_torques)

        # 5. 生成报告
        print("\n5. 生成比较报告...")
        generate_comparison_report(dynamics_results, mlp_results, comparison_stats)

        print("\n✓ 比较分析完成！")
        return True
    else:
        print("❌ 比较失败")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)