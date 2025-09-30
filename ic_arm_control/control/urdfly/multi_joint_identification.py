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

    def create_features(self, q, dq, ddq, joint_id, data_mode="auto"):
        """
        为指定关节创建动力学特征 - 改进版本

        Args:
            q: 该关节的位置
            dq: 该关节的速度
            ddq: 该关节的加速度
            joint_id: 关节ID
            data_mode: "auto"自动检测, "static"静态数据, "dynamic"动态数据

        Returns:
            特征矩阵
        """
        # 检测数据类型
        if data_mode == "auto":
            vel_std = np.std(dq)
            acc_std = np.std(ddq)
            if vel_std < 0.005 and acc_std < 0.005:
                data_mode = "static"

        if data_mode == "static":
            return self._create_static_features(q, dq, ddq, joint_id)
        else:
            return self._create_dynamic_features(q, dq, ddq, joint_id)

    def _create_static_features(self, q, dq, ddq, joint_id):
        """
        为静态数据创建特征 - 专注于重力辨识
        """
        features = []

        # 基本重力特征
        features.append(np.ones_like(q))  # 常数项
        features.append(np.sin(q))  # sin重力分量
        features.append(np.cos(q))  # cos重力分量

        # 重力矩的高阶项
        features.append(np.sin(2*q))  # 二次谐波
        features.append(np.cos(2*q))  # 二次谐波
        features.append(np.sin(3*q))  # 三次谐波
        features.append(np.cos(3*q))  # 三次谐波

        # 静摩擦特征（即使速度接近零，仍有方向信息）
        features.append(np.tanh(dq * 100))  # 极敏感的方向特征
        features.append(np.sign(dq + 1e-10))  # 强制符号（偏移避免零）

        # 位置的平方和立方项（重力矩的非线性）
        features.append(q**2)
        features.append(q**3)

        # 组合特征矩阵
        X = np.column_stack(features)

        # 特征名称（只需设置一次）
        if not self.feature_names:
            self.feature_names = [
                'constant', 'sin_pos', 'cos_pos', 'sin_2pos', 'cos_2pos',
                'sin_3pos', 'cos_3pos', 'static_friction_sign', 'forced_friction_sign',
                'pos_squared', 'pos_cubed'
            ]

        return X

    def _create_dynamic_features(self, q, dq, ddq, joint_id):
        """
        为动态数据创建特征 - 原有逻辑
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

    def preprocess_data(self, q, dq, ddq, tau, joint_id, data_mode="auto"):
        """
        数据预处理 - 针对静态数据优化的版本

        Args:
            q: 位置数组
            dq: 速度数组
            ddq: 加速度数组
            tau: 力矩数组
            joint_id: 关节ID
            data_mode: "auto"自动检测, "static"静态数据, "dynamic"动态数据

        Returns:
            过滤后的数据
        """
        print(f"  原始数据点: {len(q)}")

        # 检测数据类型
        vel_std = np.std(dq)
        acc_std = np.std(ddq)

        if data_mode == "auto":
            if vel_std < 0.005 and acc_std < 0.005:
                data_mode = "static"
                print(f"  检测到静态数据 (vel_std={vel_std:.6f}, acc_std={acc_std:.6f})")
            else:
                data_mode = "dynamic"
                print(f"  检测到动态数据 (vel_std={vel_std:.6f}, acc_std={acc_std:.6f})")

        # 移除异常值
        tau_iqr = np.percentile(tau, [25, 75])
        tau_range = tau_iqr[1] - tau_iqr[0]
        tau_lower = tau_iqr[0] - 3 * tau_range
        tau_upper = tau_iqr[1] + 3 * tau_range
        outlier_mask = (tau < tau_lower) | (tau > tau_upper)

        if data_mode == "static":
            # 静态数据处理策略 - 保留静态数据进行重力辨识
            print("  使用静态数据处理模式")

            # 移除异常值但保留静态数据点
            valid_mask = ~outlier_mask

            # 为静态数据添加额外的过滤条件
            # 确保位置有足够变化（重力辨识需要不同位置的力矩数据）
            pos_range = np.max(q) - np.min(q)
            if pos_range < 0.1:  # 位置变化太小
                print(f"  警告: 位置变化范围太小 ({np.degrees(pos_range):.1f}°)，重力辨识效果可能不佳")

            # 计算统计信息
            static_torque_std = np.std(tau[valid_mask])
            print(f"  静态力矩标准差: {static_torque_std:.6f} Nm")

            if static_torque_std < 0.01:
                print(f"  警告: 力矩变化太小，可能无法辨识重力参数")

        else:
            # 动态数据处理策略 - 原有逻辑
            print("  使用动态数据处理模式")

            # 移除零加速度数据点（这些对惯性辨识无用）
            zero_acc_mask = np.abs(ddq) < 1e-6
            if np.sum(zero_acc_mask) > len(q) * 0.5:
                print(f"  警告: {np.sum(zero_acc_mask)/len(q)*100:.1f}% 的加速度接近零")

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

    def identify_joint(self, joint_id, q, dq, ddq, tau, regularization=0.1, data_mode="auto"):
        """
        辨识单个关节的动力学参数 - 改进版本

        Args:
            joint_id: 关节ID (1-5)
            q: 位置数组
            dq: 速度数组
            ddq: 加速度数组
            tau: 力矩数组
            regularization: 正则化参数
            data_mode: "auto"自动检测, "static"静态数据, "dynamic"动态数据

        Returns:
            辨识结果
        """
        print(f"=== 关节{joint_id}动力学辨识 ===")

        # 数据预处理
        q_proc, dq_proc, ddq_proc, tau_proc = self.preprocess_data(q, dq, ddq, tau, joint_id, data_mode)

        if len(q_proc) < 50:
            print(f"  警告: 有效数据不足 ({len(q_proc)} 个点)，辨识结果可能不可靠")
            return None, None, None

        # 创建特征矩阵
        X = self.create_features(q_proc, dq_proc, ddq_proc, joint_id, data_mode)
        y = tau_proc

        print(f"特征矩阵形状: {X.shape}")

        # 使用鲁棒缩放处理异常值
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # 特殊处理Joint1
        if joint_id == 1:
            # Joint1使用更温和的正则化和不同的模型
            return self._identify_joint1_special(X_scaled, y, q_proc, dq_proc, ddq_proc, tau_proc, scaler)

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

    def _identify_joint1_special(self, X_scaled, y, q_proc, dq_proc, ddq_proc, tau_proc, scaler):
        """
        专门为Joint1设计的辨识方法
        """
        print("  使用Joint1专用辨识方法...")

        # Joint1的专用参数网格（更小的正则化）
        param_grid = {
            'Ridge': {'alpha': [0.0001, 0.001, 0.01, 0.1]},
            'Lasso': {'alpha': [0.0001, 0.001, 0.01]},
            'ElasticNet': {'alpha': [0.0001, 0.001, 0.01], 'l1_ratio': [0.1, 0.3, 0.5]}
        }

        models = {
            'Ridge': Ridge(max_iter=5000),
            'Lasso': Lasso(max_iter=5000),
            'ElasticNet': ElasticNet(max_iter=5000)
        }

        best_model = None
        best_score = -np.inf
        best_params = None
        best_model_name = None

        # 网格搜索
        for model_name in models:
            print(f"  测试 {model_name} 模型 (Joint1专用)...")
            grid_search = GridSearchCV(
                models[model_name],
                param_grid[model_name],
                cv=3,  # 减少交叉验证折数以增加数据量
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
        print(f"\n  辨识参数 (Joint1):")
        for i, (name, coef) in enumerate(zip(self.feature_names, best_model.coef_)):
            if abs(coef) > 1e-6:  # 只显示非零参数
                print(f"    {name}: {coef:.6f}")
        print(f"    截距: {best_model.intercept_:.6f}")

        # 保存结果
        result = {
            'joint_id': 1,
            'coefficients': best_model.coef_,
            'intercept': best_model.intercept_,
            'feature_names': self.feature_names,
            'rmse': rmse,
            'r2': r2,
            'model_name': best_model_name,
            'best_params': best_params,
            'cv_score': best_score,
            'n_samples': len(q_proc),
            'n_original_samples': len(q_proc),
            'position_range': [q_proc.min(), q_proc.max()],
            'velocity_range': [dq_proc.min(), dq_proc.max()],
            'torque_range': [tau_proc.min(), tau_proc.max()],
            'special_treatment': 'Joint1专用方法'
        }

        return best_model, scaler, result

    def identify_joint_static_special(self, joint_id, q, dq, ddq, tau, regularization=0.01):
        """
        专门为静态数据设计的辨识方法 - 物理约束优化
        """
        print(f"=== 关节{joint_id}静态数据专用辨识 ===")

        # 数据预处理 - 静态模式
        q_proc, dq_proc, ddq_proc, tau_proc = self.preprocess_data(q, dq, ddq, tau, joint_id, "static")

        if len(q_proc) < 30:
            print(f"  警告: 有效数据不足 ({len(q_proc)} 个点)，辨识结果可能不可靠")
            return None, None, None

        # 验证数据质量
        pos_range = np.max(q_proc) - np.min(q_proc)
        torque_range = np.max(tau_proc) - np.min(tau_proc)

        # 根据力矩变化范围选择策略
        if torque_range < 0.05:
            print(f"  力矩变化范围过小 ({torque_range:.3f} Nm)，使用简化模型")
            return self._identify_static_simplified(joint_id, q_proc, dq_proc, ddq_proc, tau_proc)

        # 创建静态数据专用特征
        X = self._create_static_features(q_proc, dq_proc, ddq_proc, joint_id)
        y = tau_proc

        print(f"特征矩阵形状: {X.shape}")

        # 对静态数据使用更强的正则化和物理约束
        from sklearn.linear_model import RidgeCV, LassoCV
        from sklearn.preprocessing import StandardScaler

        # 使用StandardScaler而不是RobustScaler，因为静态数据通常更干净
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 针对重力辨识优化的alpha范围
        alphas = np.logspace(-4, -1, 20)  # 0.0001 to 0.1

        # 使用RidgeCV进行交叉验证
        ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='r2')
        ridge_cv.fit(X_scaled, y)

        print(f"  最佳alpha: {ridge_cv.alpha_:.6f}")
        print(f"  交叉验证R²: {ridge_cv.score(X_scaled, y):.4f}")

        # 使用最佳模型进行预测
        tau_pred = ridge_cv.predict(X_scaled)

        # 计算误差
        mse = mean_squared_error(y, tau_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, tau_pred)

        print(f"  RMS误差: {rmse:.6f}")
        print(f"  R²分数: {r2:.4f}")

        # 输出重要参数
        print(f"\n  辨识参数 (静态数据专用):")
        for i, (name, coef) in enumerate(zip(self.feature_names, ridge_cv.coef_)):
            if abs(coef) > 1e-6:  # 只显示非零参数
                print(f"    {name}: {coef:.6f}")
        print(f"    截距: {ridge_cv.intercept_:.6f}")

        print(f"\n  物理验证:")
        print(f"    位置变化范围: {np.degrees(pos_range):.1f}°")
        print(f"    力矩变化范围: {torque_range:.3f} Nm")

        if pos_range < 0.05:  # 约3度
            print(f"    ⚠️ 位置变化范围太小，重力辨识可能不准确")
        if torque_range < 0.1:
            print(f"    ⚠️ 力矩变化范围太小，可能无法辨识重力参数")

        # 保存结果
        result = {
            'joint_id': joint_id,
            'coefficients': ridge_cv.coef_,
            'intercept': ridge_cv.intercept_,
            'feature_names': self.feature_names,
            'rmse': rmse,
            'r2': r2,
            'model_name': 'RidgeCV_Static',
            'best_params': {'alpha': ridge_cv.alpha_},
            'cv_score': ridge_cv.score(X_scaled, y),
            'n_samples': len(q_proc),
            'n_original_samples': len(q),
            'position_range': [q_proc.min(), q_proc.max()],
            'velocity_range': [dq_proc.min(), dq_proc.max()],
            'torque_range': [tau_proc.min(), tau_proc.max()],
            'special_treatment': 'Static data optimized',
            'pos_range_deg': float(np.degrees(pos_range)),
            'torque_range_nm': float(torque_range)
        }

        return ridge_cv, scaler, result

    def _identify_static_simplified(self, joint_id, q, dq, ddq, tau):
        """
        简化模型，用于力矩变化范围很小的静态数据
        """
        print(f"  使用简化模型进行辨识...")

        # 使用最基础的特征
        features = []
        features.append(np.ones_like(q))  # 常数项
        features.append(np.sin(q))  # 基本重力分量
        features.append(np.cos(q))  # 基本重力分量

        X = np.column_stack(features)
        y = tau

        # 简化的特征名称
        simplified_features = ['constant', 'sin_pos', 'cos_pos']

        # 使用强正则化
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 使用固定的强正则化参数
        model = Ridge(alpha=0.1)
        model.fit(X_scaled, y)

        # 预测
        tau_pred = model.predict(X_scaled)

        # 计算误差
        mse = mean_squared_error(y, tau_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, tau_pred)

        print(f"  简化模型R²: {r2:.4f}")
        print(f"  简化模型RMSE: {rmse:.6f}")

        print(f"\n  辨识参数 (简化模型):")
        for i, (name, coef) in enumerate(zip(simplified_features, model.coef_)):
            print(f"    {name}: {coef:.6f}")
        print(f"    截距: {model.intercept_:.6f}")

        # 保存结果
        result = {
            'joint_id': joint_id,
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'feature_names': simplified_features,
            'rmse': rmse,
            'r2': r2,
            'model_name': 'Ridge_Simplified',
            'best_params': {'alpha': 0.1},
            'cv_score': r2,
            'n_samples': len(q),
            'n_original_samples': len(q),
            'position_range': [q.min(), q.max()],
            'velocity_range': [dq.min(), dq.max()],
            'torque_range': [tau.min(), tau.max()],
            'special_treatment': 'Static data simplified',
            'pos_range_deg': float(np.degrees(np.max(q) - np.min(q))),
            'torque_range_nm': float(np.max(tau) - np.min(tau)),
            'note': 'Simplified model due to small torque range'
        }

        return model, scaler, result

    def identify_all_joints(self, data, regularization=0.1, data_mode="auto"):
        """
        辨识所有关节的动力学参数

        Args:
            data: 包含所有关节数据的DataFrame
            regularization: 正则化参数
            data_mode: "auto"自动检测, "static"静态数据, "dynamic"动态数据

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

            # 根据数据模式选择辨识方法
            if data_mode == "static":
                print(f"  使用静态数据专用辨识方法...")
                model, scaler, result = self.identify_joint_static_special(joint_id, q, dq, ddq, tau, regularization)
            else:
                model, scaler, result = self.identify_joint(joint_id, q, dq, ddq, tau, regularization, data_mode)

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

    def predict_joint(self, joint_id, q, dq, ddq, data_mode="auto"):
        """
        使用辨识结果预测指定关节的力矩

        Args:
            joint_id: 关节ID (1-5)
            q: 位置
            dq: 速度
            ddq: 加速度
            data_mode: 数据模式，用于选择特征创建方法

        Returns:
            预测力矩
        """
        if joint_id < 1 or joint_id > len(self.models):
            raise ValueError(f"关节 {joint_id} 模型不存在")

        model = self.models[joint_id - 1]
        scaler = self.scalers[joint_id - 1]
        result = self.identification_results[joint_id - 1]

        # 根据模型的特殊处理选择特征创建方法
        if result.get('special_treatment') == 'Static data simplified':
            # 简化模型使用基础特征
            features = []
            features.append(np.ones_like(np.array(q)))
            features.append(np.sin(q))
            features.append(np.cos(q))
            X = np.column_stack(features)
        elif result.get('special_treatment') == 'Static data optimized':
            # 静态数据模型使用静态特征
            X = self._create_static_features(q, dq, ddq, joint_id)
        else:
            # 动态数据模型使用动态特征
            X = self._create_dynamic_features(q, dq, ddq, joint_id)

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

        result = self.identification_results[joint_id - 1]

        # 根据模型的特殊处理选择预测方法
        if result.get('special_treatment') == 'Static data optimized':
            tau_pred = self.predict_joint(joint_id, q, dq, ddq)
        else:
            tau_pred = self.predict_joint(joint_id, q, dq, ddq)

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

def run_multi_joint_identification(csv_file, max_points=None, regularization=0.1, data_mode="auto"):
    """
    运行多关节辨识

    Args:
        csv_file: CSV文件路径
        max_points: 最大数据点数
        regularization: 正则化参数
        data_mode: "auto"自动检测, "static"静态数据, "dynamic"动态数据
    """
    # 加载数据
    data = load_data_from_csv(csv_file)

    # 限制数据点数
    if max_points and len(data) > max_points:
        print(f"Limiting data points to: {max_points}")
        data = data.sample(n=max_points, random_state=42).reset_index(drop=True)

    # 创建辨识器并执行辨识
    identifier = MultiJointIdentification(n_joints=6)
    results = identifier.identify_all_joints(data, regularization=regularization, data_mode=data_mode)

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
    import sys

    # 使用转换后的日志数据
    data_file = "/Users/lr-2002/project/instantcreation/IC_arm_control/dyn_ana/merged_log_data.csv"

    # 解析命令行参数
    data_mode = "auto"
    if len(sys.argv) > 1:
        if sys.argv[1] in ["static", "dynamic", "auto"]:
            data_mode = sys.argv[1]
            print(f"使用指定数据模式: {data_mode}")

    print("Multi-Joint Dynamics Identification\n")
    print(f"数据模式: {data_mode}")
    print()

    # 运行多关节辨识
    identifier = run_multi_joint_identification(
        data_file,
        # max_points=15000,  # 限制数据点数以控制计算时间
        regularization=0.05,
        data_mode=data_mode
    )

    print("\n✓ Multi-joint identification completed!")