#!/usr/bin/env python3
"""
完整的动力学辨识流程
结合数据预处理和Pinocchio参数线性化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os
import json

from static_data_preprocessor import StaticDataPreprocessor
from pinocchio_linearization import PinocchioLinearizer

class CompleteDynamicsPipeline:
    """完整的动力学辨识流程"""

    def __init__(self, urdf_path: str):
        """
        初始化流程

        Args:
            urdf_path: URDF文件路径
        """
        self.urdf_path = urdf_path
        self.preprocessor = None
        self.linearizer = None
        self.pipeline_results = {}

    def run_complete_pipeline(self, data_file: str,
                             output_dir: str = "dynamics_identification_results") -> Dict:
        """
        运行完整的辨识流程

        Args:
            data_file: 原始数据文件路径
            output_dir: 输出目录

        Returns:
            pipeline_results: 流程结果
        """
        print("="*60)
        print("开始完整的动力学辨识流程")
        print("="*60)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 步骤1: 数据预处理
        print("\n步骤1: 数据预处理")
        preprocess_results = self._run_preprocessing(data_file, output_dir)
        self.pipeline_results['preprocessing'] = preprocess_results

        # 步骤2: 准备辨识数据
        print("\n步骤2: 准备辨识数据")
        identification_data = self._prepare_identification_data(preprocess_results, output_dir)
        self.pipeline_results['identification_data'] = identification_data

        # 步骤3: Pinocchio线性化
        print("\n步骤3: Pinocchio参数线性化")
        linearization_results = self._run_linearization(identification_data, output_dir)
        self.pipeline_results['linearization'] = linearization_results

        # 步骤4: 参数辨识
        print("\n步骤4: 动力学参数辨识")
        identification_results = self._run_identification(linearization_results, output_dir)
        self.pipeline_results['identification'] = identification_results

        # 步骤5: 结果验证
        print("\n步骤5: 辨识结果验证")
        validation_results = self._validate_results(identification_results, output_dir)
        self.pipeline_results['validation'] = validation_results

        # 步骤6: 生成报告
        print("\n步骤6: 生成辨识报告")
        self._generate_report(output_dir)

        # 保存完整结果
        self._save_complete_results(output_dir)

        print(f"\n{'='*60}")
        print("完整辨识流程完成!")
        print(f"结果保存在: {output_dir}")
        print(f"{'='*60}")

        return self.pipeline_results

    def _run_preprocessing(self, data_file: str, output_dir: str) -> Dict:
        """运行数据预处理"""
        print("运行数据预处理...")

        # 创建预处理器
        self.preprocessor = StaticDataPreprocessor(
            velocity_threshold=0.5,  # More lenient thresholds
            acceleration_threshold=5.0,
            outlier_method='iqr',
            window_size=5
        )

        # 加载数据
        data = self.preprocessor.load_data(data_file)

        # 筛选稳态数据
        steady_data = self.preprocessor.filter_steady_state(data)

        # 保存稳态数据
        steady_file = os.path.join(output_dir, "steady_state_data.csv")
        steady_data.to_csv(steady_file, index=False)
        print(f"稳态数据已保存: {steady_file}")

        # 对每个关节进行预处理
        all_results = {}
        for joint_id in range(1, 7):
            joint_result = self.preprocessor.preprocess_joint(
                steady_data,
                joint_id,
                apply_filter=True,
                normalize=False,  # 动力学辨识不需要标准化
                classify_direction=True
            )
            all_results[joint_id] = joint_result

        # 保存预处理结果
        processed_dir = self.preprocessor.save_processed_data(all_results, output_dir)

        return {
            'preprocessor': self.preprocessor,
            'all_results': all_results,
            'processed_dir': processed_dir,
            'steady_data': steady_data
        }

    def _prepare_identification_data(self, preprocess_results: Dict, output_dir: str) -> Dict:
        """准备辨识数据"""
        print("准备辨识数据...")

        # 从预处理结果中提取数据
        steady_data = preprocess_results['steady_data']

        # 从稳态数据中提取完整的6-DOF数据
        if len(steady_data) == 0:
            print("警告: 没有稳态数据可用，使用原始数据")
            steady_data = preprocess_results['preprocessor'].load_data(
                "/Users/lr-2002/project/instantcreation/IC_arm_control/dyn_ana/merged_log_data.csv"
            )

        # 提取完整的6-DOF状态数据
        full_q = []
        full_dq = []
        full_ddq = []
        full_tau = []

        # 构建完整的6-DOF向量
        for i in range(1, 7):
            pos_col = f'm{i}_pos_actual'
            vel_col = f'm{i}_vel_actual'
            acc_col = f'm{i}_acc_actual'
            torque_col = f'm{i}_torque'

            if pos_col in steady_data.columns:
                full_q.append(steady_data[pos_col].values)
                full_dq.append(steady_data[vel_col].values)
                full_ddq.append(steady_data[acc_col].values)
                full_tau.append(steady_data[torque_col].values)

        # 转换为numpy数组并调整维度
        q_matrix = np.column_stack(full_q)  # (n_samples, 6)
        dq_matrix = np.column_stack(full_dq)  # (n_samples, 6)
        ddq_matrix = np.column_stack(full_ddq)  # (n_samples, 6)
        tau_matrix = np.column_stack(full_tau)  # (n_samples, 6)

        print(f"完整6-DOF数据形状: q={q_matrix.shape}, dq={dq_matrix.shape}, ddq={ddq_matrix.shape}, tau={tau_matrix.shape}")

        # 保存辨识数据
        data_file = os.path.join(output_dir, "identification_data.npz")
        np.savez(data_file, q=q_matrix, dq=dq_matrix, ddq=ddq_matrix, tau=tau_matrix)
        print(f"辨识数据已保存: {data_file}")

        return {
            'q_matrix': q_matrix,
            'dq_matrix': dq_matrix,
            'ddq_matrix': ddq_matrix,
            'tau_matrix': tau_matrix,
            'data_file': data_file
        }

    def _run_linearization(self, identification_data: Dict, output_dir: str) -> Dict:
        """运行Pinocchio线性化"""
        print("运行Pinocchio线性化...")

        # 初始化线性化器
        self.linearizer = PinocchioLinearizer(self.urdf_path)

        # 提取基参数
        theta_base, param_info = self.linearizer.extract_base_parameters()

        # 保存基参数信息
        param_info_file = os.path.join(output_dir, "base_parameters_info.json")
        with open(param_info_file, 'w') as f:
            json.dump(param_info, f, indent=2, default=str)
        print(f"基参数信息已保存: {param_info_file}")

        # 使用完整的6-DOF数据
        q_matrix = identification_data['q_matrix']
        dq_matrix = identification_data['dq_matrix']
        ddq_matrix = identification_data['ddq_matrix']
        tau_matrix = identification_data['tau_matrix']

        print(f"使用6-DOF数据形状:")
        print(f"  q: {q_matrix.shape}")
        print(f"  dq: {dq_matrix.shape}")
        print(f"  ddq: {ddq_matrix.shape}")
        print(f"  tau: {tau_matrix.shape}")

        # 线性化动力学
        Y, tau_vec = self.linearizer.linearize_dynamics(
            q_matrix, dq_matrix, ddq_matrix, tau_matrix
        )

        # 保存线性化结果
        linearization_file = os.path.join(output_dir, "linearization_results.npz")
        np.savez(linearization_file,
                 Y=Y,
                 tau_vec=tau_vec,
                 q=q_matrix,
                 dq=dq_matrix,
                 ddq=ddq_matrix,
                 tau=tau_matrix)
        print(f"线性化结果已保存: {linearization_file}")

        return {
            'theta_base': theta_base,
            'param_info': param_info,
            'regressor_matrix': Y,
            'torque_vector': tau_vec,
            'combined_data': {
                'q': q_matrix,
                'dq': dq_matrix,
                'ddq': ddq_matrix,
                'tau': tau_matrix
            }
        }

    def _run_identification(self, linearization_results: Dict, output_dir: str) -> Dict:
        """运行参数辨识"""
        print("运行参数辨识...")

        Y = linearization_results['regressor_matrix']
        tau_vec = linearization_results['torque_vector']

        # 尝试不同的辨识方法
        methods = ['least_squares', 'ridge', 'lasso']
        identification_results = {}

        for method in methods:
            print(f"\n使用 {method} 方法...")

            try:
                theta_identified, identification_info = self.linearizer.identify_parameters(
                    Y, tau_vec, method=method, regularization=0.01
                )

                identification_results[method] = {
                    'theta': theta_identified,
                    'info': identification_info
                }

                print(f"{method} 方法完成:")
                print(f"  RMSE: {identification_info['rmse']:.6f}")
                print(f"  R²: {identification_info['r2']:.6f}")

            except Exception as e:
                print(f"{method} 方法失败: {e}")
                identification_results[method] = None

        # 选择最佳方法
        best_method = None
        best_r2 = -np.inf

        for method, result in identification_results.items():
            if result is not None and result['info']['r2'] > best_r2:
                best_r2 = result['info']['r2']
                best_method = method

        print(f"\n最佳方法: {best_method} (R² = {best_r2:.6f})")

        # 使用最佳方法更新模型
        if best_method is not None:
            theta_best = identification_results[best_method]['theta']
            self.linearizer.update_model_parameters(theta_best)

            # 保存最佳参数
            best_params_file = os.path.join(output_dir, f"identified_parameters_{best_method}.npz")
            self.linearizer.save_identified_parameters(
                theta_best, identification_results[best_method]['info'], best_params_file
            )

        return {
            'all_methods': identification_results,
            'best_method': best_method,
            'best_r2': best_r2
        }

    def _validate_results(self, identification_results: Dict, output_dir: str) -> Dict:
        """验证辨识结果"""
        print("验证辨识结果...")

        best_method = identification_results['best_method']

        if best_method is None:
            print("没有可用的辨识结果进行验证")
            return {}

        best_result = identification_results['all_methods'][best_method]
        theta_identified = best_result['theta']

        # 提取原始基参数
        theta_base = self.pipeline_results['linearization']['theta_base']

        # 参数对比
        param_comparison = {}
        n_params = min(len(theta_identified), len(theta_base), len(self.linearizer.base_param_names))
        print(f"对比参数数量: {n_params} (辨识: {len(theta_identified)}, 原始: {len(theta_base)}, 名称: {len(self.linearizer.base_param_names)})")

        for i in range(n_params):
            param_name = self.linearizer.base_param_names[i]
            param_comparison[param_name] = {
                'original': float(theta_base[i]),
                'identified': float(theta_identified[i]),
                'difference': float(theta_identified[i] - theta_base[i]),
                'relative_change': float((theta_identified[i] - theta_base[i]) / (theta_base[i] + 1e-10))
            }

        # 计算预测误差
        Y = self.pipeline_results['linearization']['regressor_matrix']
        tau_vec = self.pipeline_results['linearization']['torque_vector']

        # 使用回归矩阵的前24列来匹配实际的参数数量
        Y_actual = Y[:, :24]  # 只使用前24列

        # 也需要相应地截取原始基参数
        theta_base_actual = theta_base[:24]

        print(f"回归矩阵形状: {Y.shape} -> {Y_actual.shape}")
        print(f"原始基参数: {len(theta_base)} -> {len(theta_base_actual)}")
        print(f"辨识参数: {len(theta_identified)}")

        tau_pred_original = Y_actual @ theta_base_actual
        tau_pred_identified = Y_actual @ theta_identified

        # 计算误差指标
        validation_info = {
            'original_rmse': np.sqrt(np.mean((tau_vec - tau_pred_original) ** 2)),
            'identified_rmse': np.sqrt(np.mean((tau_vec - tau_pred_identified) ** 2)),
            'improvement_ratio': np.sqrt(np.mean((tau_vec - tau_pred_original) ** 2)) /
                               np.sqrt(np.mean((tau_vec - tau_pred_identified) ** 2)),
            'max_error_original': np.max(np.abs(tau_vec - tau_pred_original)),
            'max_error_identified': np.max(np.abs(tau_vec - tau_pred_identified))
        }

        # 保存验证结果
        validation_file = os.path.join(output_dir, "validation_results.json")
        save_data = {
            'param_comparison': param_comparison,
            'validation_info': validation_info,
            'best_method': best_method
        }

        with open(validation_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"验证结果:")
        print(f"  原始参数RMSE: {validation_info['original_rmse']:.6f}")
        print(f"  辨识参数RMSE: {validation_info['identified_rmse']:.6f}")
        print(f"  改进比例: {validation_info['improvement_ratio']:.2f}x")

        # 绘制验证图表
        self._plot_validation_results(tau_vec, tau_pred_original, tau_pred_identified, output_dir)

        return {
            'param_comparison': param_comparison,
            'validation_info': validation_info,
            'save_file': validation_file
        }

    def _plot_validation_results(self, tau_vec: np.ndarray, tau_pred_original: np.ndarray,
                                tau_pred_identified: np.ndarray, output_dir: str):
        """绘制验证结果图表"""
        print("绘制验证图表...")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 原始参数预测
        axes[0, 0].scatter(tau_vec, tau_pred_original, alpha=0.5, s=1)
        axes[0, 0].plot([tau_vec.min(), tau_vec.max()], [tau_vec.min(), tau_vec.max()], 'r--')
        axes[0, 0].set_title('原始参数预测')
        axes[0, 0].set_xlabel('实际力矩')
        axes[0, 0].set_ylabel('预测力矩')
        axes[0, 0].grid(True)

        # 2. 辨识参数预测
        axes[0, 1].scatter(tau_vec, tau_pred_identified, alpha=0.5, s=1)
        axes[0, 1].plot([tau_vec.min(), tau_vec.max()], [tau_vec.min(), tau_vec.max()], 'r--')
        axes[0, 1].set_title('辨识参数预测')
        axes[0, 1].set_xlabel('实际力矩')
        axes[0, 1].set_ylabel('预测力矩')
        axes[0, 1].grid(True)

        # 3. 残差对比
        residual_original = tau_vec - tau_pred_original
        residual_identified = tau_vec - tau_pred_identified

        axes[1, 0].hist(residual_original, bins=50, alpha=0.7, label='原始参数', density=True)
        axes[1, 0].hist(residual_identified, bins=50, alpha=0.7, label='辨识参数', density=True)
        axes[1, 0].set_title('残差分布对比')
        axes[1, 0].set_xlabel('残差')
        axes[1, 0].set_ylabel('密度')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 4. 误差时间序列
        sample_indices = np.arange(len(tau_vec))[:1000]  # 只显示前1000个点
        axes[1, 1].plot(sample_indices, residual_original[sample_indices], alpha=0.7, label='原始参数')
        axes[1, 1].plot(sample_indices, residual_identified[sample_indices], alpha=0.7, label='辨识参数')
        axes[1, 1].set_title('预测误差时间序列')
        axes[1, 1].set_xlabel('样本索引')
        axes[1, 1].set_ylabel('误差')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "validation_results.png"), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"验证图表已保存: validation_results.png")

    def _generate_report(self, output_dir: str):
        """生成辨识报告"""
        print("生成辨识报告...")

        report_file = os.path.join(output_dir, "identification_report.md")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 动力学参数辨识报告\n\n")
            f.write(f"生成时间: {pd.Timestamp.now()}\n\n")

            # 预处理结果
            f.write("## 1. 数据预处理\n\n")
            preprocess_results = self.pipeline_results['preprocessing']['all_results']
            for joint_id, result in preprocess_results.items():
                f.write(f"### 关节{joint_id}\n")
                f.write(f"- 原始数据点: {result['original_points']}\n")
                f.write(f"- 最终数据点: {result['final_points']}\n")
                if 'positive_direction' in result:
                    f.write(f"- 正方向到达: {result['positive_direction']} 点\n")
                    f.write(f"- 负方向到达: {result['negative_direction']} 点\n")
                f.write("\n")

            # 线性化结果
            f.write("## 2. 线性化结果\n\n")
            linearization_results = self.pipeline_results['linearization']
            f.write(f"- 回归矩阵形状: {linearization_results['regressor_matrix'].shape}\n")
            f.write(f"- 基参数数量: {len(linearization_results['theta_base'])}\n")
            f.write("\n")

            # 辨识结果
            f.write("## 3. 参数辨识\n\n")
            identification_results = self.pipeline_results['identification']
            best_method = identification_results['best_method']
            f.write(f"### 最佳方法: {best_method}\n")
            if best_method is not None:
                best_info = identification_results['all_methods'][best_method]['info']
                f.write(f"- RMSE: {best_info['rmse']:.6f}\n")
                f.write(f"- R²: {best_info['r2']:.6f}\n")
                f.write(f"- 最大误差: {best_info['max_error']:.6f}\n")
            f.write("\n")

            # 验证结果
            f.write("## 4. 结果验证\n\n")
            validation_results = self.pipeline_results['validation']
            if validation_results:
                validation_info = validation_results['validation_info']
                f.write(f"### 预测精度\n")
                f.write(f"- 原始参数RMSE: {validation_info['original_rmse']:.6f}\n")
                f.write(f"- 辨识参数RMSE: {validation_info['identified_rmse']:.6f}\n")
                f.write(f"- 改进比例: {validation_info['improvement_ratio']:.2f}x\n")
            f.write("\n")

        print(f"辨识报告已生成: {report_file}")

    def _save_complete_results(self, output_dir: str):
        """保存完整结果"""
        print("保存完整结果...")

        # 转换结果为可序列化的格式
        serializable_results = self._make_serializable(self.pipeline_results)

        # 保存完整结果
        complete_file = os.path.join(output_dir, "complete_pipeline_results.json")
        with open(complete_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        print(f"完整结果已保存: {complete_file}")

    def _make_serializable(self, obj):
        """使对象可序列化"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        else:
            return obj


def main():
    """主函数"""
    # 配置参数
    data_file = "/Users/lr-2002/project/instantcreation/IC_arm_control/dyn_ana/merged_log_data.csv"
    urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/urdfs/ic_arm_8_dof/urdf/ic_arm_8_dof.urdf"
    output_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control/control/urdfly/dynamics_identification_results"

    # 创建并运行流程
    pipeline = CompleteDynamicsPipeline(urdf_path)
    results = pipeline.run_complete_pipeline(data_file, output_dir)

    return results


if __name__ == "__main__":
    results = main()