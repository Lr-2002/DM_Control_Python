#!/usr/bin/env python3
"""
静态数据预处理模块
专门用于动力学辨识前的数据清洗和预处理
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
import os

class StaticDataPreprocessor:
    """静态数据预处理器"""

    def __init__(self,
                 velocity_threshold: float = 0.5,  # Much more lenient: 0.5 rad/s (~29 deg/s)
                 acceleration_threshold: float = 5.0,  # Much more lenient: 5.0 rad/s²
                 outlier_method: str = 'iqr',
                 window_size: int = 5):
        """
        初始化预处理器

        Args:
            velocity_threshold: 速度阈值，小于此值认为是稳态
            acceleration_threshold: 加速度阈值
            outlier_method: 异常值检测方法 ('iqr' 或 'zscore')
            window_size: 中值滤波窗口大小
        """
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        self.outlier_method = outlier_method
        self.window_size = window_size

        self.preprocess_stats = {}

    def load_data(self, data_file: str) -> pd.DataFrame:
        """加载原始数据"""
        print(f"加载数据: {data_file}")
        data = pd.read_csv(data_file)
        print(f"原始数据点: {len(data)}")

        # 显示原始数据统计
        self._show_data_stats(data, "原始数据")

        return data

    def _show_data_stats(self, data: pd.DataFrame, title: str):
        """显示数据统计信息"""
        print(f"\n{title}统计:")
        print(f"  数据点数: {len(data)}")

        for i in range(1, 7):
            pos_col = f'm{i}_pos_actual'
            vel_col = f'm{i}_vel_actual'
            acc_col = f'm{i}_acc_actual'
            torque_col = f'm{i}_torque'

            if pos_col in data.columns:
                pos_data = data[pos_col]
                vel_data = data[vel_col]
                acc_data = data[acc_col]
                torque_data = data[torque_col]

                print(f"  关节{i}:")
                print(f"    位置范围: [{np.degrees(pos_data.min()):.1f}°, {np.degrees(pos_data.max()):.1f}°]")
                print(f"    速度范围: [{vel_data.min():.4f}, {vel_data.max():.4f}] rad/s")
                print(f"    加速度范围: [{acc_data.min():.4f}, {acc_data.max():.4f}] rad/s²")
                print(f"    力矩范围: [{torque_data.min():.4f}, {torque_data.max():.4f}] Nm")

    def filter_steady_state(self, data: pd.DataFrame) -> pd.DataFrame:
        """筛选稳态数据样本"""
        print("\n=== 筛选稳态数据 ===")

        # 创建稳态标志
        steady_mask = pd.Series(True, index=data.index)

        for i in range(1, 7):
            vel_col = f'm{i}_vel_actual'
            acc_col = f'm{i}_acc_actual'

            if vel_col in data.columns:
                # 速度和加速度都要小于阈值
                vel_steady = np.abs(data[vel_col]) < self.velocity_threshold
                acc_steady = np.abs(data[acc_col]) < self.acceleration_threshold

                joint_steady = vel_steady & acc_steady
                steady_mask = steady_mask & joint_steady

                steady_ratio = joint_steady.sum() / len(data) * 100
                print(f"  关节{i}稳态比例: {steady_ratio:.1f}%")

        # 应用过滤
        steady_data = data[steady_mask].copy()
        steady_ratio = len(steady_data) / len(data) * 100

        print(f"稳态数据筛选结果:")
        print(f"  原始数据点: {len(data)}")
        print(f"  稳态数据点: {len(steady_data)}")
        print(f"  保留比例: {steady_ratio:.1f}%")

        self._show_data_stats(steady_data, "稳态数据")

        return steady_data

    def remove_outliers(self, data: pd.DataFrame, joint_id: int) -> pd.DataFrame:
        """移除异常点"""
        print(f"\n=== 关节{joint_id}异常值处理 ===")

        pos_col = f'm{joint_id}_pos_actual'
        vel_col = f'm{joint_id}_vel_actual'
        torque_col = f'm{joint_id}_torque'

        if pos_col not in data.columns:
            return data

        # 原始数据
        original_data = len(data)

        # Check if data is empty
        if original_data == 0:
            print("  警告: 输入数据为空，跳过异常值处理")
            return data

        # 方法1: IQR方法
        if self.outlier_method == 'iqr':
            # 计算力矩的IQR
            q1 = data[torque_col].quantile(0.25)
            q3 = data[torque_col].quantile(0.75)
            iqr = q3 - q1

            # 定义异常值边界
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # 标记异常值
            outlier_mask = (data[torque_col] < lower_bound) | (data[torque_col] > upper_bound)

        # 方法2: Z-score方法
        elif self.outlier_method == 'zscore':
            torque_mean = data[torque_col].mean()
            torque_std = data[torque_col].std()

            # 计算Z-score
            z_scores = np.abs((data[torque_col] - torque_mean) / torque_std)
            outlier_mask = z_scores > 3

        # 移除异常值
        cleaned_data = data[~outlier_mask].copy()

        # 统计
        outliers_removed = original_data - len(cleaned_data)
        outlier_ratio = outliers_removed / original_data * 100 if original_data > 0 else 0

        print(f"  异常值移除: {outliers_removed} 个 ({outlier_ratio:.1f}%)")
        print(f"  清理后数据点: {len(cleaned_data)}")

        # 可视化异常值检测结果
        self._plot_outlier_detection(data, joint_id, outlier_mask, cleaned_data)

        return cleaned_data

    def _plot_outlier_detection(self, data: pd.DataFrame, joint_id: int,
                               outlier_mask: pd.Series, cleaned_data: pd.DataFrame):
        """绘制异常值检测结果"""
        pos_col = f'm{joint_id}_pos_actual'
        torque_col = f'm{joint_id}_torque'

        if pos_col not in data.columns:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Position sequence
        ax1.plot(np.degrees(data[pos_col]), label='Original Data', alpha=0.7)
        ax1.scatter(data.index[outlier_mask], np.degrees(data.loc[outlier_mask, pos_col]),
                   color='red', s=20, label='Outliers')
        ax1.set_title(f'Joint {joint_id} Position Sequence (Outlier Detection)')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Position (deg)')
        ax1.legend()
        ax1.grid(True)

        # Torque vs Position scatter plot
        ax2.scatter(np.degrees(data[pos_col]), data[torque_col],
                   alpha=0.3, s=5, label='Original Data')
        ax2.scatter(np.degrees(data.loc[outlier_mask, pos_col]),
                   data.loc[outlier_mask, torque_col],
                   color='red', s=20, label='Outliers')
        ax2.set_title(f'Joint {joint_id} Torque vs Position')
        ax2.set_xlabel('Position (deg)')
        ax2.set_ylabel('Torque (Nm)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f'/Users/lr-2002/project/instantcreation/IC_arm_control/control/urdfly/joint{joint_id}_outlier_detection.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  异常值检测图已保存: joint{joint_id}_outlier_detection.png")

    def classify_by_direction(self, data: pd.DataFrame, joint_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """按到达方向分类数据"""
        print(f"\n=== 关节{joint_id}方向分类 ===")

        pos_col = f'm{joint_id}_pos_actual'
        vel_col = f'm{joint_id}_vel_actual'

        if pos_col not in data.columns:
            return data, pd.DataFrame()

        # 计算位置的一阶差分来估计运动方向
        pos_diff = data[pos_col].diff()

        # 标记到达方向
        # 正方向到达: 位置差分为正（之前在向正方向运动）
        positive_direction = pos_diff > 0
        negative_direction = pos_diff < 0

        # 分别提取两个方向的数据
        pos_data = data[positive_direction].copy()
        neg_data = data[negative_direction].copy()

        print(f"  正方向到达: {len(pos_data)} 个点")
        print(f"  负方向到达: {len(neg_data)} 个点")

        # 可视化方向分类
        self._plot_direction_classification(data, joint_id, positive_direction, negative_direction)

        return pos_data, neg_data

    def _plot_direction_classification(self, data: pd.DataFrame, joint_id: int,
                                     pos_direction: pd.Series, neg_direction: pd.Series):
        """绘制方向分类结果"""
        pos_col = f'm{joint_id}_pos_actual'
        torque_col = f'm{joint_id}_torque'

        if pos_col not in data.columns:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Position sequence colored by direction
        ax1.plot(np.degrees(data[pos_col]), alpha=0.3, color='gray', label='All Data')
        ax1.scatter(data.index[pos_direction], np.degrees(data.loc[pos_direction, pos_col]),
                   color='blue', s=10, label='Positive Direction')
        ax1.scatter(data.index[neg_direction], np.degrees(data.loc[neg_direction, pos_col]),
                   color='red', s=10, label='Negative Direction')
        ax1.set_title(f'Joint {joint_id} Position Sequence (By Direction)')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Position (deg)')
        ax1.legend()
        ax1.grid(True)

        # Torque vs Position colored by direction
        ax2.scatter(np.degrees(data.loc[pos_direction, pos_col]),
                   data.loc[pos_direction, torque_col],
                   color='blue', alpha=0.5, s=10, label='Positive Direction')
        ax2.scatter(np.degrees(data.loc[neg_direction, pos_col]),
                   data.loc[neg_direction, torque_col],
                   color='red', alpha=0.5, s=10, label='Negative Direction')
        ax2.set_title(f'Joint {joint_id} Torque vs Position (By Direction)')
        ax2.set_xlabel('Position (deg)')
        ax2.set_ylabel('Torque (Nm)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f'/Users/lr-2002/project/instantcreation/IC_arm_control/control/urdfly/joint{joint_id}_direction_classification.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  方向分类图已保存: joint{joint_id}_direction_classification.png")

    def normalize_data(self, data: pd.DataFrame, method: str = 'standard') -> Tuple[pd.DataFrame, Dict]:
        """数据标准化"""
        print(f"\n=== 数据标准化 ({method}) ===")

        normalized_data = data.copy()
        scalers = {}

        for i in range(1, 7):
            torque_col = f'm{i}_torque'

            if torque_col in data.columns:
                if method == 'standard':
                    # 标准化: (x - mean) / std
                    mean_val = data[torque_col].mean()
                    std_val = data[torque_col].std()

                    normalized_data[torque_col] = (data[torque_col] - mean_val) / std_val
                    scalers[torque_col] = {'mean': mean_val, 'std': std_val, 'method': 'standard'}

                    print(f"  关节{i}: mean={mean_val:.4f}, std={std_val:.4f}")

                elif method == 'minmax':
                    # 最小最大标准化: (x - min) / (max - min)
                    min_val = data[torque_col].min()
                    max_val = data[torque_col].max()

                    normalized_data[torque_col] = (data[torque_col] - min_val) / (max_val - min_val)
                    scalers[torque_col] = {'min': min_val, 'max': max_val, 'method': 'minmax'}

                    print(f"  关节{i}: min={min_val:.4f}, max={max_val:.4f}")

        return normalized_data, scalers

    def apply_median_filter(self, data: pd.DataFrame, joint_id: int) -> pd.DataFrame:
        """应用中值滤波"""
        print(f"\n=== 关节{joint_id}中值滤波 ===")

        torque_col = f'm{joint_id}_torque'

        if torque_col not in data.columns:
            return data

        filtered_data = data.copy()

        # 对力矩数据应用中值滤波
        filtered_torque = median_filter(data[torque_col], size=self.window_size)
        filtered_data[torque_col] = filtered_torque

        # 计算滤波前后的差异
        rmse_before = np.sqrt(np.mean(data[torque_col]**2))  # 假设理想力矩为0
        rmse_after = np.sqrt(np.mean(filtered_torque**2))

        print(f"  滤波窗口大小: {self.window_size}")
        print(f"  滤波前RMSE: {rmse_before:.6f}")
        print(f"  滤波后RMSE: {rmse_after:.6f}")

        # 可视化滤波效果
        self._plot_median_filter_effect(data, joint_id, filtered_torque)

        return filtered_data

    def _plot_median_filter_effect(self, data: pd.DataFrame, joint_id: int, filtered_torque: np.ndarray):
        """绘制中值滤波效果"""
        torque_col = f'm{joint_id}_torque'
        pos_col = f'm{joint_id}_pos_actual'

        if torque_col not in data.columns:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Torque sequence comparison
        ax1.plot(data[torque_col], alpha=0.7, label='Original Torque')
        ax1.plot(filtered_torque, linewidth=2, label='Filtered Torque')
        ax1.set_title(f'Joint {joint_id} Torque Sequence (Median Filter)')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Torque (Nm)')
        ax1.legend()
        ax1.grid(True)

        # Torque vs Position comparison
        if pos_col in data.columns:
            ax2.scatter(np.degrees(data[pos_col]), data[torque_col],
                       alpha=0.3, s=5, label='Original Data')
            ax2.scatter(np.degrees(data[pos_col]), filtered_torque,
                       alpha=0.7, s=10, label='Filtered Data')
            ax2.set_title(f'Joint {joint_id} Torque vs Position (Filter Effect)')
            ax2.set_xlabel('Position (deg)')
            ax2.set_ylabel('Torque (Nm)')
            ax2.legend()
            ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f'/Users/lr-2002/project/instantcreation/IC_arm_control/control/urdfly/joint{joint_id}_median_filter.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  中值滤波效果图已保存: joint{joint_id}_median_filter.png")

    def preprocess_joint(self, data: pd.DataFrame, joint_id: int,
                       apply_filter: bool = True,
                       normalize: bool = True,
                       classify_direction: bool = True) -> Dict:
        """预处理单个关节的数据"""
        print(f"\n{'='*50}")
        print(f"预处理关节 {joint_id}")
        print(f"{'='*50}")

        joint_data = data.copy()
        preprocessing_results = {
            'joint_id': joint_id,
            'original_points': len(joint_data),
            'steps': []
        }

        # 1. 移除异常值
        print("步骤1: 移除异常值")
        cleaned_data = self.remove_outliers(joint_data, joint_id)
        preprocessing_results['steps'].append({
            'step': 'remove_outliers',
            'points_before': len(joint_data),
            'points_after': len(cleaned_data)
        })

        # 2. 中值滤波
        if apply_filter:
            print("步骤2: 中值滤波")
            filtered_data = self.apply_median_filter(cleaned_data, joint_id)
            preprocessing_results['steps'].append({
                'step': 'median_filter',
                'points_before': len(cleaned_data),
                'points_after': len(filtered_data)
            })
            joint_data = filtered_data
        else:
            joint_data = cleaned_data

        # 3. 方向分类
        if classify_direction:
            print("步骤3: 方向分类")
            pos_data, neg_data = self.classify_by_direction(joint_data, joint_id)
            preprocessing_results['positive_direction'] = len(pos_data)
            preprocessing_results['negative_direction'] = len(neg_data)
        else:
            pos_data = joint_data
            neg_data = pd.DataFrame()

        # 4. 数据标准化
        if normalize:
            print("步骤4: 数据标准化")
            normalized_data, scalers = self.normalize_data(joint_data)
            preprocessing_results['scalers'] = scalers
            preprocessing_results['steps'].append({
                'step': 'normalization',
                'method': 'standard'
            })
        else:
            normalized_data = joint_data

        preprocessing_results['final_points'] = len(joint_data)
        preprocessing_results['processed_data'] = normalized_data
        preprocessing_results['positive_data'] = pos_data
        preprocessing_results['negative_data'] = neg_data

        return preprocessing_results

    def save_processed_data(self, results: Dict, output_dir: str = "processed_static_data"):
        """保存处理后的数据"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存每个关节的处理结果
        for joint_result in results.values():
            joint_id = joint_result['joint_id']

            # 保存处理后的数据
            if 'processed_data' in joint_result:
                output_file = os.path.join(output_dir, f'joint{joint_id}_processed.csv')
                joint_result['processed_data'].to_csv(output_file, index=False)
                print(f"关节{joint_id}处理后数据已保存: {output_file}")

            # 保存正方向数据
            if 'positive_data' in joint_result and len(joint_result['positive_data']) > 0:
                output_file = os.path.join(output_dir, f'joint{joint_id}_positive_direction.csv')
                joint_result['positive_data'].to_csv(output_file, index=False)
                print(f"关节{joint_id}正方向数据已保存: {output_file}")

            # 保存负方向数据
            if 'negative_data' in joint_result and len(joint_result['negative_data']) > 0:
                output_file = os.path.join(output_dir, f'joint{joint_id}_negative_direction.csv')
                joint_result['negative_data'].to_csv(output_file, index=False)
                print(f"关节{joint_id}负方向数据已保存: {output_file}")

        # 保存处理统计信息
        import json
        stats_file = os.path.join(output_dir, 'preprocessing_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"预处理统计信息已保存: {stats_file}")

        return output_dir


def main():
    """主函数 - 演示完整的数据预处理流程"""

    # 输入数据文件
    data_file = "/Users/lr-2002/project/instantcreation/IC_arm_control/dyn_ana/merged_log_data.csv"

    # 创建预处理器
    preprocessor = StaticDataPreprocessor(
        velocity_threshold=1e-3,
        acceleration_threshold=1e-2,
        outlier_method='iqr',
        window_size=5
    )

    # 1. 加载数据
    data = preprocessor.load_data(data_file)

    # 2. 筛选稳态数据
    steady_data = preprocessor.filter_steady_state(data)

    # 3. 对每个关节进行预处理
    all_results = {}

    for joint_id in range(1, 7):
        joint_result = preprocessor.preprocess_joint(
            steady_data,
            joint_id,
            apply_filter=True,
            normalize=False,  # 动力学辨识通常不需要标准化
            classify_direction=True
        )
        all_results[joint_id] = joint_result

    # 4. 保存处理结果
    output_dir = preprocessor.save_processed_data(all_results)

    print(f"\n{'='*50}")
    print("数据预处理完成!")
    print(f"{'='*50}")
    print(f"输出目录: {output_dir}")

    # 打印汇总统计
    print("\n预处理汇总:")
    for joint_id, result in all_results.items():
        print(f"  关节{joint_id}: {result['original_points']} -> {result['final_points']} 个点")

    return all_results


if __name__ == "__main__":
    results = main()