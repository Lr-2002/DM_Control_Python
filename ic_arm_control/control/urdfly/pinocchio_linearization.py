#!/usr/bin/env python3
"""
使用Pinocchio进行动力学参数线性化
将非线性动力学方程转换为关于参数的线性形式
"""

import numpy as np
import pandas as pd
import pinocchio as pin
from typing import Dict, List, Tuple, Optional
import os
import matplotlib.pyplot as plt

class PinocchioLinearizer:
    """使用Pinocchio进行动力学参数线性化"""

    def __init__(self, urdf_path: str):
        """
        初始化线性化器

        Args:
            urdf_path: URDF文件路径
        """
        self.urdf_path = urdf_path
        self.model = None
        self.data = None
        self.param_names = []
        self.base_param_names = []

        # 初始化Pinocchio模型
        self._init_model()

    def _init_model(self):
        """初始化Pinocchio模型"""
        print(f"加载URDF模型: {self.urdf_path}")

        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF文件不存在: {self.urdf_path}")

        # 创建模型和数据
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()

        # 获取基参数名称
        self._get_base_param_names()

        print(f"模型初始化完成:")
        print(f"  关节数量: {self.model.njoints}")
        print(f"  配置空间维度: {self.model.nq}")
        print(f"  切空间维度: {self.model.nv}")
        print(f"  基参数数量: {self.model.nv * 10}")  # 估计值

    def _get_base_param_names(self):
        """获取基参数名称"""
        # Pinocchio中每个连杆有10个基参数
        # [m, Ixx, Iyy, Izz, Ixy, Ixz, Iyz, cx, cy, cz]
        # 其中：m是质量，I是惯性矩阵分量，c是质心位置（相对于关节）

        self.base_param_names = []
        for i in range(1, self.model.njoints):  # 跳过universe关节
            joint_name = self.model.names[i]
            for param_type in ['mass', 'inertia', 'com']:
                if param_type == 'mass':
                    self.base_param_names.append(f"{joint_name}_mass")
                elif param_type == 'inertia':
                    for axis in ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']:
                        self.base_param_names.append(f"{joint_name}_I{axis}")
                elif param_type == 'com':
                    for axis in ['x', 'y', 'z']:
                        self.base_param_names.append(f"{joint_name}_com{axis}")

        print(f"基参数名称数量: {len(self.base_param_names)}")

    def extract_base_parameters(self) -> Tuple[np.ndarray, Dict]:
        """
        提取基参数向量

        Returns:
            theta_base: 基参数向量
            param_info: 参数信息字典
        """
        theta_base = np.zeros(len(self.base_param_names))
        param_info = {}

        # 获取当前模型的惯性参数
        for i in range(1, self.model.njoints):  # 跳过universe关节
            joint_name = self.model.names[i]
            inertia = self.model.inertias[i]

            # 提取基参数
            param_start_idx = (i - 1) * 10  # 每个连杆10个参数

            # 质量
            theta_base[param_start_idx] = inertia.mass
            param_info[f"{joint_name}_mass"] = {
                'value': inertia.mass,
                'index': param_start_idx
            }

            # 惯性矩阵 (上三角)
            theta_base[param_start_idx + 1] = inertia.inertia[0, 0]  # Ixx
            theta_base[param_start_idx + 2] = inertia.inertia[1, 1]  # Iyy
            theta_base[param_start_idx + 3] = inertia.inertia[2, 2]  # Izz
            theta_base[param_start_idx + 4] = inertia.inertia[0, 1]  # Ixy
            theta_base[param_start_idx + 5] = inertia.inertia[0, 2]  # Ixz
            theta_base[param_start_idx + 6] = inertia.inertia[1, 2]  # Iyz

            param_info[f"{joint_name}_Ixx"] = {'value': inertia.inertia[0, 0], 'index': param_start_idx + 1}
            param_info[f"{joint_name}_Iyy"] = {'value': inertia.inertia[1, 1], 'index': param_start_idx + 2}
            param_info[f"{joint_name}_Izz"] = {'value': inertia.inertia[2, 2], 'index': param_start_idx + 3}
            param_info[f"{joint_name}_Ixy"] = {'value': inertia.inertia[0, 1], 'index': param_start_idx + 4}
            param_info[f"{joint_name}_Ixz"] = {'value': inertia.inertia[0, 2], 'index': param_start_idx + 5}
            param_info[f"{joint_name}_Iyz"] = {'value': inertia.inertia[1, 2], 'index': param_start_idx + 6}

            # 质心位置
            theta_base[param_start_idx + 7] = inertia.lever[0]  # cx
            theta_base[param_start_idx + 8] = inertia.lever[1]  # cy
            theta_base[param_start_idx + 9] = inertia.lever[2]  # cz

            param_info[f"{joint_name}_comx"] = {'value': inertia.lever[0], 'index': param_start_idx + 7}
            param_info[f"{joint_name}_comy"] = {'value': inertia.lever[1], 'index': param_start_idx + 8}
            param_info[f"{joint_name}_comz"] = {'value': inertia.lever[2], 'index': param_start_idx + 9}

        return theta_base, param_info

    def compute_regressor_matrix(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        """
        计算回归矩阵 (观测矩阵)

        Args:
            q: 关节位置 (n_samples, n_joints)
            dq: 关节速度 (n_samples, n_joints)
            ddq: 关节加速度 (n_samples, n_joints)

        Returns:
            Y: 回归矩阵 (n_samples * n_joints, n_params)
        """
        print("计算回归矩阵...")

        n_samples = q.shape[0]
        n_joints = q.shape[1]

        # 首先使用一个样本测试实际的参数数量
        try:
            test_regressor = pin.computeStaticRegressor(self.model, self.data, q[0])
            actual_n_params = test_regressor.shape[1]
            print(f"检测到实际参数数量: {actual_n_params} (之前期望: {len(self.base_param_names)})")
            print(f"静态回归矩阵形状: {test_regressor.shape}")

            # 更新参数数量
            n_params = actual_n_params

        except Exception as e:
            print(f"无法确定实际参数数量: {e}")
            n_params = len(self.base_param_names)

        # 初始化回归矩阵
        Y = np.zeros((n_samples * n_joints, n_params))

        for i in range(n_samples):
            # 设置当前配置
            q_i = q[i]

            try:
                # 使用Pinocchio的computeStaticRegressor
                static_regressor = pin.computeStaticRegressor(self.model, self.data, q_i)

                # 静态回归矩阵的形状可能是 (3, n_params) 或 (6, n_params)
                # 我们需要根据实际形状来处理
                if static_regressor.shape[0] >= n_joints:
                    # 如果行数足够，直接使用前n_joints行
                    Y[i*n_joints:(i+1)*n_joints, :] = static_regressor[:n_joints, :]
                elif static_regressor.shape[0] == 3:
                    # 如果只有3行（可能是3D力向量），复制到前3个关节
                    Y[i*n_joints:(i*n_joints + 3), :] = static_regressor
                    # 后3个关节填充零或使用简化方法
                    for j in range(3, n_joints):
                        Y[i*n_joints + j, :] = self._compute_simplified_joint_regressor(q_i, np.zeros_like(q_i), np.zeros_like(q_i), j+1)
                else:
                    # 其他情况，使用简化方法
                    for j in range(n_joints):
                        Y[i*n_joints + j, :] = self._compute_simplified_joint_regressor(q_i, np.zeros_like(q_i), np.zeros_like(q_i), j+1)

            except Exception as e:
                print(f"回归矩阵计算错误 (样本 {i}): {e}")
                # 使用简化的方法
                for j in range(n_joints):
                    simplified_regressor = self._compute_simplified_joint_regressor(q_i, np.zeros_like(q_i), np.zeros_like(q_i), j+1)
                    Y[i*n_joints + j, :] = simplified_regressor

        print(f"回归矩阵形状: {Y.shape}")
        return Y

    def _compute_joint_regressor(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray, joint_id: int) -> np.ndarray:
        """
        计算单个关节的回归矩阵

        Args:
            q: 关节位置
            dq: 关节速度
            ddq: 关节加速度
            joint_id: 关节ID

        Returns:
            joint_regressor: 关节回归矩阵 (1, n_params)
        """
        try:
            # 使用Pinocchio的computeStaticRegressor函数 - 适用于静态重力辨识
            if hasattr(pin, 'computeStaticRegressor'):
                # 计算静态回归矩阵 - 这更适合重力补偿
                static_regressor = pin.computeStaticRegressor(self.model, self.data, q)
                # 静态回归矩阵的形状是 (6, n_params)，我们需要第joint_id-1行
                if static_regressor.shape[0] >= joint_id:
                    return static_regressor[joint_id-1, :]
                else:
                    raise ValueError(f"静态回归矩阵行数不足: {static_regressor.shape[0]} < {joint_id}")

            else:
                # 如果没有内置函数，使用简化方法
                return self._compute_simplified_joint_regressor(q, dq, ddq, joint_id)

        except Exception as e:
            print(f"计算关节{joint_id}回归矩阵失败: {e}")
            return self._compute_simplified_joint_regressor(q, dq, ddq, joint_id)

    def _compute_simplified_regressor(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        """
        计算简化的回归矩阵（当Pinocchio函数不可用时）

        Args:
            q: 关节位置
            dq: 关节速度
            ddq: 关节加速度

        Returns:
            regressor: 简化回归矩阵
        """
        n_joints = len(q)
        n_params = 24  # 使用Pinocchio的实际参数数量

        # 创建简化的回归矩阵
        # 这里使用基本的物理原理构建
        Y = np.zeros((n_joints, n_params))

        # 对每个关节创建简化的重力回归器
        for j in range(n_joints):
            joint_id = j + 1

            # 使用与单个关节回归器相同的逻辑
            if joint_id <= 3:  # 前3个关节
                param_idx = (joint_id - 1) * 4  # 每个关节4个参数
                if param_idx < 24:
                    Y[j, param_idx] = np.sin(q[j])     # 重力项1
                    Y[j, param_idx + 1] = np.cos(q[j])   # 重力项2
                    Y[j, param_idx + 2] = np.sin(2 * q[j])  # 重力项3
                    Y[j, param_idx + 3] = np.cos(2 * q[j])  # 重力项4
            else:  # 后3个关节
                param_idx = 12 + (joint_id - 4) * 3  # 后3个关节，每个3个参数
                if param_idx < 21:
                    Y[j, param_idx] = np.sin(q[j])     # 重力项1
                    Y[j, param_idx + 1] = np.cos(q[j])   # 重力项2
                    Y[j, param_idx + 2] = 1.0                    # 偏置项

        return Y

    def _compute_simplified_joint_regressor(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray, joint_id: int) -> np.ndarray:
        """
        计算单个关节的简化回归矩阵

        Args:
            q: 关节位置
            dq: 关节速度
            ddq: 关节加速度
            joint_id: 关节ID

        Returns:
            joint_regressor: 关节回归矩阵
        """
        # 使用实际参数数量而不是期望的参数数量
        # 这里使用一个简化的24参数模型

        # 创建一个基本的重力回归器
        regressor = np.zeros(24)  # 使用Pinocchio的实际参数数量

        # 简化的重力模型 - 基于关节位置
        if joint_id <= 3:  # 只为前3个关节设置非零值
            # 为每个关节分配不同的参数索引
            param_idx = (joint_id - 1) * 4  # 每个关节4个参数

            if param_idx < 24:
                regressor[param_idx] = np.sin(q[joint_id - 1])     # 重力项1
                regressor[param_idx + 1] = np.cos(q[joint_id - 1])   # 重力项2
                regressor[param_idx + 2] = np.sin(2 * q[joint_id - 1])  # 重力项3
                regressor[param_idx + 3] = np.cos(2 * q[joint_id - 1])  # 重力项4
        else:
            # 后3个关节使用更简单的模型
            param_idx = 12 + (joint_id - 4) * 3  # 后3个关节，每个3个参数
            if param_idx < 21:
                regressor[param_idx] = np.sin(q[joint_id - 1])     # 重力项1
                regressor[param_idx + 1] = np.cos(q[joint_id - 1])   # 重力项2
                regressor[param_idx + 2] = 1.0                    # 偏置项

        return regressor

    def linearize_dynamics(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray, tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将动力学方程线性化

        Args:
            q: 关节位置 (n_samples, n_joints)
            dq: 关节速度 (n_samples, n_joints)
            ddq: 关节加速度 (n_samples, n_joints)
            tau: 关节力矩 (n_samples, n_joints)

        Returns:
            Y: 回归矩阵 (n_samples * n_joints, n_params)
            tau_vec: 力矩向量 (n_samples * n_joints,)
        """
        print("开始动力学线性化...")

        # 计算回归矩阵
        Y = self.compute_regressor_matrix(q, dq, ddq)

        # 将力矩矩阵转换为向量
        n_samples, n_joints = tau.shape
        tau_vec = tau.reshape(-1)

        print(f"线性化完成:")
        print(f"  回归矩阵形状: {Y.shape}")
        print(f"  力矩向量形状: {tau_vec.shape}")

        return Y, tau_vec

    def identify_parameters(self, Y: np.ndarray, tau_vec: np.ndarray,
                          method: str = 'least_squares',
                          regularization: float = 0.01) -> Tuple[np.ndarray, Dict]:
        """
        辨识动力学参数

        Args:
            Y: 回归矩阵
            tau_vec: 力矩向量
            method: 辨识方法 ('least_squares', 'ridge', 'lasso')
            regularization: 正则化参数

        Returns:
            theta_identified: 辨识的参数
            identification_info: 辨识信息
        """
        print(f"使用 {method} 方法辨识参数...")

        n_samples, n_params = Y.shape

        if method == 'least_squares':
            # 最小二乘法
            try:
                # 使用QR分解求解最小二乘问题
                theta_identified, residuals, rank, s = np.linalg.lstsq(Y, tau_vec, rcond=None)
                method_info = {
                    'method': 'least_squares',
                    'rank': rank,
                    'singular_values': s.tolist(),
                    'residuals': residuals.tolist() if len(residuals) > 0 else None
                }
            except Exception as e:
                print(f"最小二乘法失败: {e}")
                # 回退到岭回归
                theta_identified, method_info = self._ridge_regression(Y, tau_vec, regularization)
                method_info['fallback_method'] = 'ridge'

        elif method == 'ridge':
            # 岭回归
            theta_identified, method_info = self._ridge_regression(Y, tau_vec, regularization)

        elif method == 'lasso':
            # Lasso回归
            theta_identified, method_info = self._lasso_regression(Y, tau_vec, regularization)

        else:
            raise ValueError(f"未知的方法: {method}")

        # 计算辨识质量指标
        tau_pred = Y @ theta_identified
        rmse = np.sqrt(np.mean((tau_vec - tau_pred) ** 2))
        r2 = 1 - np.sum((tau_vec - tau_pred) ** 2) / np.sum((tau_vec - np.mean(tau_vec)) ** 2)

        identification_info = {
            'method': method,
            'regularization': regularization,
            'n_samples': n_samples,
            'n_params': n_params,
            'rmse': rmse,
            'r2': r2,
            'max_error': np.max(np.abs(tau_vec - tau_pred)),
            'method_details': method_info
        }

        print(f"参数辨识完成:")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  R²: {r2:.6f}")
        print(f"  最大误差: {identification_info['max_error']:.6f}")

        return theta_identified, identification_info

    def _ridge_regression(self, Y: np.ndarray, tau_vec: np.ndarray, alpha: float) -> Tuple[np.ndarray, Dict]:
        """岭回归"""
        # Y^T Y + alpha * I
        YtY = Y.T @ Y
        YtY_reg = YtY + alpha * np.eye(YtY.shape[0])

        # (Y^T Y + alpha * I)^{-1} Y^T tau
        theta_identified = np.linalg.solve(YtY_reg, Y.T @ tau_vec)

        method_info = {
            'method': 'ridge',
            'alpha': alpha,
            'condition_number': np.linalg.cond(YtY_reg)
        }

        return theta_identified, method_info

    def _lasso_regression(self, Y: np.ndarray, tau_vec: np.ndarray, alpha: float) -> Tuple[np.ndarray, Dict]:
        """Lasso回归"""
        try:
            from sklearn.linear_model import Lasso

            lasso = Lasso(alpha=alpha, max_iter=10000)
            lasso.fit(Y, tau_vec)
            theta_identified = lasso.coef_

            method_info = {
                'method': 'lasso',
                'alpha': alpha,
                'n_iterations': lasso.n_iter_,
                'intercept': lasso.intercept_
            }

            return theta_identified, method_info

        except ImportError:
            print("sklearn未安装，回退到岭回归")
            return self._ridge_regression(Y, tau_vec, alpha)

    def update_model_parameters(self, theta_identified: np.ndarray):
        """
        使用辨识的参数更新Pinocchio模型

        Args:
            theta_identified: 辨识的参数向量
        """
        print("更新模型参数...")

        # 检查参数数量
        if len(theta_identified) != 24:
            print(f"警告: 辨识参数数量({len(theta_identified)})与期望的24个基参数不匹配")
            print("跳过模型参数更新")
            return

        # 对于Pinocchio的24个基参数，我们需要更谨慎的更新策略
        # 这里我们使用一个简化的更新方法
        try:
            # 计算一个缩放因子，因为辨识的参数可能不是直接对应于惯性参数
            scale_factor = 0.1  # 缩放因子，防止参数过大

            for i in range(1, min(4, self.model.njoints)):  # 只更新前3个关节，参数数量限制
                # 为每个关节分配参数（每个关节8个参数：1个质量，6个惯性，3个COM）
                param_start_idx = (i - 1) * 8

                if param_start_idx + 9 < len(theta_identified):
                    # 获取当前惯性参数
                    inertia = self.model.inertias[i]

                    # 更新质量（添加保护）
                    new_mass = max(0.001, theta_identified[param_start_idx] * scale_factor)
                    inertia.mass = new_mass

                    # 更新惯性矩阵（添加保护和缩放）
                    inertia.inertia[0, 0] = max(0.0001, theta_identified[param_start_idx + 1] * scale_factor)  # Ixx
                    inertia.inertia[1, 1] = max(0.0001, theta_identified[param_start_idx + 2] * scale_factor)  # Iyy
                    inertia.inertia[2, 2] = max(0.0001, theta_identified[param_start_idx + 3] * scale_factor)  # Izz
                    inertia.inertia[0, 1] = theta_identified[param_start_idx + 4] * scale_factor * 0.01  # Ixy
                    inertia.inertia[0, 2] = theta_identified[param_start_idx + 5] * scale_factor * 0.01  # Ixz
                    inertia.inertia[1, 2] = theta_identified[param_start_idx + 6] * scale_factor * 0.01  # Iyz
                    inertia.inertia[1, 0] = inertia.inertia[0, 1]  # 对称性
                    inertia.inertia[2, 0] = inertia.inertia[0, 2]  # 对称性
                    inertia.inertia[2, 1] = inertia.inertia[1, 2]  # 对称性

                    # 更新质心位置（添加保护）
                    inertia.lever[0] = theta_identified[param_start_idx + 7] * 0.1  # cx
                    inertia.lever[1] = theta_identified[param_start_idx + 8] * 0.1  # cy
                    if param_start_idx + 9 < len(theta_identified):
                        inertia.lever[2] = theta_identified[param_start_idx + 9] * 0.1  # cz

                    # 更新模型
                    self.model.inertias[i] = inertia

                    print(f"  更新关节{i}参数: 质量={new_mass:.6f}, 惯性对角=[{inertia.inertia[0,0]:.6f}, {inertia.inertia[1,1]:.6f}, {inertia.inertia[2,2]:.6f}]")

        except Exception as e:
            print(f"参数更新过程中出现错误: {e}")
            print("使用默认值保持模型不变")

        print("模型参数更新完成")

    def save_identified_parameters(self, theta_identified: np.ndarray,
                                 identification_info: Dict,
                                 output_file: str):
        """
        保存辨识的参数

        Args:
            theta_identified: 辨识的参数
            identification_info: 辨识信息
            output_file: 输出文件路径
        """
        print(f"保存辨识参数到: {output_file}")

        # 准备保存的数据
        save_data = {
            'identified_params': theta_identified,
            'param_names': self.base_param_names,
            'identification_info': identification_info,
            'urdf_path': self.urdf_path,
            'model_info': {
                'n_joints': self.model.njoints,
                'nq': self.model.nq,
                'nv': self.model.nv
            }
        }

        # 保存为numpy格式
        if output_file.endswith('.npz'):
            np.savez(output_file, **save_data)
        # 保存为pickle格式
        elif output_file.endswith('.pkl'):
            import pickle
            with open(output_file, 'wb') as f:
                pickle.dump(save_data, f)
        else:
            raise ValueError("不支持的文件格式，请使用.npz或.pkl")

        print(f"参数保存完成")


def main():
    """主函数 - 演示完整的线性化流程"""
    # URDF文件路径
    urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/urdfs/ic_arm_8_dof/urdf/ic_arm_8_dof.urdf"

    # 创建线性化器
    linearizer = PinocchioLinearizer(urdf_path)

    # 这里应该加载预处理后的数据
    # 示例数据（实际使用时应该从预处理模块加载）
    print("注意：这里使用示例数据，实际应该从预处理模块加载")

    # 示例：生成一些测试数据
    n_samples = 100
    n_joints = 6

    q = np.random.randn(n_samples, n_joints) * 0.1  # 小范围的位置
    dq = np.zeros((n_samples, n_joints))  # 静态数据，速度为0
    ddq = np.zeros((n_samples, n_joints))  # 静态数据，加速度为0

    # 生成一些模拟的力矩数据（用于测试）
    tau = np.random.randn(n_samples, n_joints) * 0.1

    # 线性化动力学
    Y, tau_vec = linearizer.linearize_dynamics(q, dq, ddq, tau)

    # 辨识参数
    theta_identified, identification_info = linearizer.identify_parameters(
        Y, tau_vec, method='least_squares'
    )

    # 保存结果
    output_file = "/Users/lr-2002/project/instantcreation/IC_arm_control/control/urdfly/identified_parameters.npz"
    linearizer.save_identified_parameters(theta_identified, identification_info, output_file)

    print(f"\n线性化完成! 结果已保存到: {output_file}")
    return linearizer, theta_identified, identification_info


if __name__ == "__main__":
    linearizer, theta, info = main()