#!/usr/bin/env python3
"""
加载和使用辨识参数的工具类
"""

import numpy as np
import json
import os
from typing import Dict, Optional, Tuple
from minimum_gc import MinimumGravityCompensation

class IdentifiedParametersLoader:
    """辨识参数加载器"""

    def __init__(self, results_dir: str = None):
        """
        初始化加载器

        Args:
            results_dir: 结果目录路径
        """
        if results_dir is None:
            results_dir = os.path.join(
                os.path.dirname(__file__),
                'dynamics_identification_results'
            )

        self.results_dir = results_dir
        self.gc = None
        self.params = None
        self.param_names = None
        self.identification_info = None

    def load_parameters(self, method: str = 'least_squares') -> Dict:
        """
        加载指定方法的辨识参数

        Args:
            method: 辨识方法 ('least_squares', 'ridge', 'lasso')

        Returns:
            参数信息字典
        """
        param_file = os.path.join(self.results_dir, f'identified_parameters_{method}.npz')

        if not os.path.exists(param_file):
            raise FileNotFoundError(f"参数文件不存在: {param_file}")

        # 加载参数文件
        data = np.load(param_file, allow_pickle=True)

        self.params = data['identified_params']
        self.param_names = data['param_names']
        self.identification_info = data['identification_info'].item()

        print(f"成功加载{method}方法参数:")
        print(f"  参数数量: {len(self.params)}")
        print(f"  RMSE: {self.identification_info['rmse']:.4f}")
        print(f"  R²: {self.identification_info['r2']:.4f}")

        return {
            'params': self.params,
            'param_names': self.param_names,
            'info': self.identification_info
        }

    def init_gravity_compensator(self, method: str = 'least_squares') -> MinimumGravityCompensation:
        """
        初始化重力补偿器

        Args:
            method: 辨识方法

        Returns:
            重力补偿器实例
        """
        param_file = os.path.join(self.results_dir, f'identified_parameters_{method}.npz')

        self.gc = MinimumGravityCompensation(param_file=param_file)
        return self.gc

    def load_validation_results(self) -> Dict:
        """加载验证结果"""
        validation_file = os.path.join(self.results_dir, 'validation_results.json')

        if not os.path.exists(validation_file):
            raise FileNotFoundError(f"验证结果文件不存在: {validation_file}")

        with open(validation_file, 'r') as f:
            validation_data = json.load(f)

        return validation_data

    def get_parameter_summary(self) -> Dict:
        """获取参数统计摘要"""
        if self.params is None:
            raise ValueError("请先加载参数")

        return {
            'num_params': len(self.params),
            'param_range': [float(self.params.min()), float(self.params.max())],
            'param_std': float(self.params.std()),
            'param_mean': float(self.params.mean()),
            'nonzero_params': int(np.sum(np.abs(self.params) > 1e-10)),
            'method': self.identification_info.get('method', 'unknown'),
            'rmse': self.identification_info.get('rmse', 0.0),
            'r2': self.identification_info.get('r2', 0.0)
        }

    def compare_parameters(self, other_loader: 'IdentifiedParametersLoader') -> Dict:
        """比较两组参数"""
        if self.params is None or other_loader.params is None:
            raise ValueError("请先加载两组参数")

        diff = self.params - other_loader.params
        rel_diff = diff / (self.params + 1e-10)

        return {
            'absolute_diff': np.max(np.abs(diff)),
            'relative_diff': np.max(np.abs(rel_diff)),
            'mse': np.mean(diff**2),
            'correlation': np.corrcoef(self.params, other_loader.params)[0, 1]
        }


def demo_usage():
    """演示使用方法"""
    print("=== 辨识参数加载演示 ===")

    # 1. 创建加载器
    loader = IdentifiedParametersLoader()

    # 2. 加载最佳参数
    print("\n1. 加载最小二乘法参数:")
    params_data = loader.load_parameters('least_squares')

    # 3. 获取参数摘要
    print("\n2. 参数统计摘要:")
    summary = loader.get_parameter_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # 4. 初始化重力补偿器
    print("\n3. 初始化重力补偿器:")
    gc = loader.init_gravity_compensator()

    # 5. 测试重力补偿
    print("\n4. 测试重力补偿:")
    q_test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    gravity_torque = gc.calculate_gravity_torque(q_test)
    print(f"  输入位置: {q_test}")
    print(f"  重力力矩: {gravity_torque}")

    # 6. 加载验证结果
    print("\n5. 验证结果:")
    validation = loader.load_validation_results()
    print(f"  原始RMSE: {validation['validation_info']['original_rmse']:.4f}")
    print(f"  辨识RMSE: {validation['validation_info']['identified_rmse']:.4f}")
    print(f"  改进比例: {validation['validation_info']['improvement_ratio']:.2f}x")

    print("\n✅ 演示完成!")
    return loader, gc


if __name__ == "__main__":
    demo_usage()