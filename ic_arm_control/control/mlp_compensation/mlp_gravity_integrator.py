#!/usr/bin/env python3
"""
MLP Gravity Compensation Integration for IC_ARM
将MLP重力补偿模型集成到IC_ARM系统中
"""

import numpy as np
import pickle
import time
import sys
from pathlib import Path
from typing import Optional, Tuple, Union
import logging

# Add current directory to path
sys.path.append('.')
from mlp_gravity_compensation import LightweightMLPGravityCompensation


class MLPGravityCompensation:
    """MLP重力补偿类 - 兼容IC_ARM的接口"""

    def __init__(self, model_path: str = "mlp_gravity_model_improved.pkl",
                 enable_enhanced: bool = True, debug: bool = False, max_torques=None):
        """
        初始化MLP重力补偿

        Args:
            model_path: MLP模型文件路径
            enable_enhanced: 是否启用增强特征
            debug: 调试模式
            max_torques: 各关节最大力矩限制列表 (Nm)
        """
        self.model_path = model_path
        self.enable_enhanced = enable_enhanced
        self.debug = debug
        self.is_initialized = False
        self.mlp_system = None
        self.use_gc = True
        self.max_torques = max_torques or [15.0, 12.0, 12.0, 4.0, 4.0, 3.0]  # 默认力矩限制
        self.last_prediction_time = 0
        self.prediction_count = 0

        # 性能统计
        self.performance_stats = {
            'total_predictions': 0,
            'avg_prediction_time': 0,
            'max_prediction_time': 0,
            'min_prediction_time': float('inf')
        }

        # 初始化模型
        self._initialize_model()

    def _initialize_model(self) -> bool:
        """初始化MLP模型"""
        try:
            if not Path(self.model_path).exists():
                print(f"❌ MLP模型文件不存在: {self.model_path}")
                return False

            print(f"🔄 加载MLP重力补偿模型: {self.model_path}")

            # 加载模型数据
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            # 创建MLP系统
            self.mlp_system = LightweightMLPGravityCompensation(
                hidden_layer_sizes=model_data['hidden_layer_sizes'],
                max_iter=model_data['max_iter'],
                random_state=model_data['random_state'],
                max_torques=self.max_torques
            )

            # 恢复模型参数
            self.mlp_system.mlps = model_data['mlps']
            self.mlp_system.input_scaler = model_data['input_scaler']
            self.mlp_system.output_scaler = model_data['output_scaler']
            self.mlp_system.train_scores = model_data['train_scores']
            self.mlp_system.val_scores = model_data.get('val_scores', [])
            self.mlp_system.is_trained = model_data['is_trained']

            # 设置增强训练标志
            if 'enhanced_training' in model_data:
                self.mlp_system.train_enhanced = model_data['enhanced_training']
                self.mlp_system.enhanced_feature_dim = model_data.get('enhanced_feature_dim', 18)
                self.enable_enhanced = model_data['enhanced_training']

            self.is_initialized = True
            print(f"✅ MLP重力补偿模型加载成功")
            print(f"   - 增强特征: {'是' if self.enable_enhanced else '否'}")
            print(f"   - 模型参数: {sum(len(mlp.coefs_[0]) for mlp in self.mlp_system.mlps):,}")

            return True

        except Exception as e:
            print(f"❌ MLP模型初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _enhance_features(self, positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        """
        生成增强特征

        Args:
            positions: 关节位置 (N, 6)
            velocities: 关节速度 (N, 6)

        Returns:
            enhanced_features: 增强特征 (N, 18)
        """
        if not self.enable_enhanced:
            return np.concatenate([positions, velocities], axis=1)

        enhanced_features = []

        # 原始特征
        enhanced_features.append(positions)
        enhanced_features.append(velocities)

        # Joint 1 特定特征
        joint1_pos = positions[:, 0:1]
        joint1_vel = velocities[:, 0:1]

        # 非线性特征
        enhanced_features.append(joint1_pos ** 2)  # 位置平方
        enhanced_features.append(joint1_vel ** 2)  # 速度平方
        enhanced_features.append(joint1_pos * joint1_vel)  # 交叉项

        # 三角函数特征
        enhanced_features.append(np.sin(joint1_pos))  # sin
        enhanced_features.append(np.cos(joint1_pos))  # cos

        # 方向特征
        enhanced_features.append(np.sign(joint1_vel))  # 速度方向

        return np.concatenate(enhanced_features, axis=1)

    def get_gravity_compensation_torque(self, positions: np.ndarray) -> np.ndarray:
        """
        计算重力补偿力矩 - 兼容IC_ARM接口

        Args:
            positions: 关节位置数组 (6,) 或 (N, 6)

        Returns:
            compensation_torque: 重力补偿力矩 (6,) 或 (N, 6)
        """
        if not self.is_initialized or not self.use_gc:
            # 返回零力矩
            if positions.ndim == 1:
                return np.zeros(6)
            else:
                return np.zeros((positions.shape[0], 6))

        try:
            start_time = time.time()

            # 确保输入是2D数组
            if positions.ndim == 1:
                positions = positions.reshape(1, -1)
                single_sample = True
            else:
                single_sample = False

            # 假设速度为零（重力补偿主要考虑静态位置）
            velocities = np.zeros_like(positions)

            # 生成特征
            if self.enable_enhanced:
                features = self._enhance_features(positions, velocities)
                prediction = self.mlp_system.predict_enhanced(features)
            else:
                prediction = self.mlp_system.predict(positions, velocities)

            # 更新性能统计
            prediction_time = (time.time() - start_time) * 1000  # ms
            self._update_performance_stats(prediction_time)

            if single_sample:
                return prediction.flatten()
            else:
                return prediction

        except Exception as e:
            print(f"❌ 重力补偿计算失败: {e}")
            if positions.ndim == 1:
                return np.zeros(6)
            else:
                return np.zeros((positions.shape[0], 6))

    def _update_performance_stats(self, prediction_time: float):
        """更新性能统计"""
        self.prediction_count += 1
        self.performance_stats['total_predictions'] += 1

        # 更新平均时间
        current_avg = self.performance_stats['avg_prediction_time']
        new_avg = (current_avg * (self.prediction_count - 1) + prediction_time) / self.prediction_count
        self.performance_stats['avg_prediction_time'] = new_avg

        # 更新最值
        self.performance_stats['max_prediction_time'] = max(
            self.performance_stats['max_prediction_time'], prediction_time
        )
        self.performance_stats['min_prediction_time'] = min(
            self.performance_stats['min_prediction_time'], prediction_time
        )

    def calculate_torque(self, positions: np.ndarray, velocities: np.ndarray,
                        accelerations: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算完整动力学力矩 - 兼容IC_ARM接口

        Args:
            positions: 关节位置
            velocities: 关节速度
            accelerations: 关节加速度 (可选)

        Returns:
            torque: 计算的力矩
        """
        # MLP模型只计算重力补偿部分
        gravity_torque = self.get_gravity_compensation_torque(positions)

        # 如果没有提供速度和加速度，只返回重力补偿
        if accelerations is None:
            return gravity_torque

        # TODO: 可以在这里添加科里奥利力和惯性力计算
        # 目前只返回重力补偿
        return gravity_torque

    def calculate_coriolis_torque(self, positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        """
        计算科里奥利力矩 - 兼容IC_ARM接口

        Args:
            positions: 关节位置
            velocities: 关节速度

        Returns:
            coriolis_torque: 科里奥利力矩
        """
        # MLP模型目前不计算科里奥利力，返回零
        if positions.ndim == 1:
            return np.zeros(6)
        else:
            return np.zeros((positions.shape[0], 6))

    def enable(self):
        """启用重力补偿"""
        self.use_gc = True
        print("✅ MLP重力补偿已启用")

    def disable(self):
        """禁用重力补偿"""
        self.use_gc = False
        print("❌ MLP重力补偿已禁用")

    def is_enabled(self) -> bool:
        """检查是否启用"""
        return self.use_gc and self.is_initialized

    def get_performance_stats(self) -> dict:
        """获取性能统计"""
        stats = self.performance_stats.copy()
        if stats['total_predictions'] > 0:
            stats['frequency_hz'] = 1000 / stats['avg_prediction_time']
        else:
            stats['frequency_hz'] = 0
        return stats

    def print_performance_summary(self):
        """打印性能摘要"""
        stats = self.get_performance_stats()
        print("=== MLP重力补偿性能摘要 ===")
        print(f"总预测次数: {stats['total_predictions']:,}")
        print(f"平均预测时间: {stats['avg_prediction_time']:.3f} ms")
        print(f"最大预测时间: {stats['max_prediction_time']:.3f} ms")
        print(f"最小预测时间: {stats['min_prediction_time']:.3f} ms")
        print(f"预测频率: {stats['frequency_hz']:.1f} Hz")
        print(f"模型状态: {'✅ 正常' if self.is_initialized else '❌ 未初始化'}")
        print(f"补偿状态: {'✅ 启用' if self.use_gc else '❌ 禁用'}")

    def reset_performance_stats(self):
        """重置性能统计"""
        self.performance_stats = {
            'total_predictions': 0,
            'avg_prediction_time': 0,
            'max_prediction_time': 0,
            'min_prediction_time': float('inf')
        }
        self.prediction_count = 0


# 兼容性别名 - 与IC_ARM现有接口保持一致
StaticGravityCompensation = MLPGravityCompensation


def create_mlp_gc_instance(model_path: str = "mlp_gravity_model_improved.pkl",
                          debug: bool = False) -> MLPGravityCompensation:
    """
    创建MLP重力补偿实例的工厂函数

    Args:
        model_path: 模型文件路径
        debug: 调试模式

    Returns:
        MLP重力补偿实例
    """
    return MLPGravityCompensation(model_path=model_path, debug=debug)


def test_mlp_gravity_compensation():
    """测试MLP重力补偿功能"""
    print("=== 测试MLP重力补偿 ===")

    # 创建实例
    mlp_gc = create_mlp_gc_instance(debug=True)

    if not mlp_gc.is_initialized:
        print("❌ 测试失败：模型初始化失败")
        return False

    # 测试基本功能
    print("\n1. 测试基本力矩计算...")
    test_positions = np.array([0.0, 0.5, 1.0, 0.2, -0.3, 0.8])
    torque = mlp_gc.get_gravity_compensation_torque(test_positions)
    print(f"测试位置: {test_positions}")
    print(f"计算力矩: {torque}")
    print(f"力矩范围: [{np.min(torque):.3f}, {np.max(torque):.3f}] Nm")

    # 测试批量计算
    print("\n2. 测试批量计算...")
    batch_positions = np.random.uniform(-np.pi, np.pi, (10, 6))
    batch_torques = mlp_gc.get_gravity_compensation_torque(batch_positions)
    print(f"批量计算: {batch_positions.shape} -> {batch_torques.shape}")

    # 测试性能
    print("\n3. 测试计算性能...")
    n_tests = 1000
    start_time = time.time()
    for _ in range(n_tests):
        mlp_gc.get_gravity_compensation_torque(test_positions)
    avg_time = (time.time() - start_time) / n_tests * 1000
    frequency = 1000 / avg_time
    print(f"平均计算时间: {avg_time:.3f} ms")
    print(f"计算频率: {frequency:.1f} Hz")

    # 显示性能统计
    print("\n4. 性能统计:")
    mlp_gc.print_performance_summary()

    print("\n✅ MLP重力补偿测试完成")
    return True


if __name__ == "__main__":
    test_mlp_gravity_compensation()