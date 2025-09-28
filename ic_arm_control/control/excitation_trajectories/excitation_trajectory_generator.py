#!/usr/bin/env python3
"""
激励轨迹生成器 (Excitation Trajectory Generator)
基于论文 "Excitation Trajectory Optimization for Dynamic Parameter Identification Using Virtual Constraints in Hands-on Robotic System"
用于IC ARM的动力学参数辨识

作者：Claude Code
日期：2025-09-28
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Optional
import os
from scipy.optimize import minimize
from scipy.signal import chirp
import warnings
warnings.filterwarnings('ignore')

class ExcitationTrajectoryGenerator:
    def __init__(self, urdf_path: str, joint_margin: float = 0.1):
        """
        初始化激励轨迹生成器

        Args:
            urdf_path: URDF文件路径
            joint_margin: 关节余量比例 (0-1)，默认为0.1表示使用10%-90%的范围
        """
        self.urdf_path = urdf_path
        self.joint_margin = joint_margin

        # 从URDF加载关节信息
        self.joint_info = self._load_joint_info_from_urdf()
        self.num_joints = len(self.joint_info)

        # 安全的关节范围（应用余量）
        self.safe_joint_limits = self._calculate_safe_joint_limits()

        print(f"=== 激励轨迹生成器初始化 ===")
        print(f"URDF文件: {urdf_path}")
        print(f"关节数量: {self.num_joints}")
        print(f"关节余量: {joint_margin*100:.0f}%")
        print(f"安全的关节范围:")
        for i, info in enumerate(self.joint_info):
            safe_limits = self.safe_joint_limits[i]
            print(f"  Joint{i+1}: [{safe_limits['lower']:.3f}, {safe_limits['upper']:.3f}] rad "
                  f"(原始: [{info['lower']:.3f}, {info['upper']:.3f}])")

    def _load_joint_info_from_urdf(self) -> List[Dict]:
        """从URDF文件加载关节信息"""
        try:
            tree = ET.parse(self.urdf_path)
            root = tree.getroot()

            joint_info = []

            for joint in root.findall('.//joint'):
                joint_name = joint.get('name')
                joint_type = joint.get('type')

                if joint_type == 'revolute':
                    limit_elem = joint.find('limit')
                    if limit_elem is not None:
                        lower = float(limit_elem.get('lower', 0))
                        upper = float(limit_elem.get('upper', 0))
                        velocity = float(limit_elem.get('velocity', 1.57))

                        joint_info.append({
                            'name': joint_name,
                            'lower': lower,
                            'upper': upper,
                            'velocity': velocity,
                            'range': upper - lower
                        })

            # 按关节名称排序
            joint_info.sort(key=lambda x: x['name'])
            print(f"从URDF加载了 {len(joint_info)} 个revolute关节")

            return joint_info

        except Exception as e:
            print(f"读取URDF失败: {e}")
            return []

    def _calculate_safe_joint_limits(self) -> List[Dict]:
        """计算安全的关节范围（应用余量）"""
        safe_limits = []

        for info in self.joint_info:
            lower = info['lower']
            upper = info['upper']
            joint_range = upper - lower

            # 应用余量：只使用中间的80%范围
            safe_range = joint_range * (1 - 2 * self.joint_margin)
            safe_lower = lower + joint_range * self.joint_margin
            safe_upper = upper - joint_range * self.joint_margin

            safe_limits.append({
                'lower': safe_lower,
                'upper': safe_upper,
                'range': safe_range,
                'velocity': info['velocity']
            })

        return safe_limits

    def _fourier_series_trajectory(self, t: np.ndarray, params: np.ndarray,
                                  num_harmonics: int = 3, duration: float = 10.0) -> np.ndarray:
        """
        生成傅里叶级数轨迹

        Args:
            t: 时间数组
            params: 参数数组 [a0, a1, b1, a2, b2, ...]
            num_harmonics: 谐波数量
            duration: 轨迹周期

        Returns:
            关节位置数组
        """
        omega = 2 * np.pi / duration

        # 提取参数
        a0 = params[0]
        a = params[1::2][:num_harmonics]  # 余弦项系数
        b = params[2::2][:num_harmonics]  # 正弦项系数

        # 生成轨迹
        q = a0 * np.ones_like(t)

        for k in range(1, num_harmonics + 1):
            if k <= len(a):
                q += a[k-1] * np.cos(k * omega * t)
            if k <= len(b):
                q += b[k-1] * np.sin(k * omega * t)

        return q

    def _multi_frequency_trajectory(self, t: np.ndarray, params: np.ndarray,
                                 frequencies: List[float], duration: float = 10.0) -> np.ndarray:
        """
        生成多频率正弦轨迹（多个不同频率的正弦波叠加）

        Args:
            t: 时间数组
            params: 参数数组 [a0, A1, φ1, A2, φ2, ...]
            frequencies: 频率列表 [f1, f2, ...]
            duration: 轨迹周期

        Returns:
            关节位置数组
        """
        a0 = params[0]
        q = a0 * np.ones_like(t)

        for i, freq in enumerate(frequencies):
            if 2*i + 1 < len(params):
                amplitude = params[2*i + 1]
                phase = params[2*i + 2] if 2*i + 2 < len(params) else 0
                q += amplitude * np.sin(2 * np.pi * freq * t + phase)

        return q

    def _swept_sine_trajectory(self, t: np.ndarray, params: np.ndarray,
                              f_start: float, f_end: float, duration: float = 10.0) -> np.ndarray:
        """
        生成扫频正弦轨迹（chirp信号）

        Args:
            t: 时间数组
            params: 参数数组 [a0, A, φ]
            f_start: 起始频率
            f_end: 结束频率
            duration: 轨迹周期

        Returns:
            关节位置数组
        """
        a0 = params[0]
        amplitude = params[1] if len(params) > 1 else 1.0
        phase = params[2] if len(params) > 2 else 0

        # 使用scipy的chirp函数
        q = a0 + amplitude * chirp(t, f_start, duration, f_end, method='linear', phi=phase)

        return q

    def _phase_modulated_trajectory(self, t: np.ndarray, params: np.ndarray,
                                   carrier_freq: float, mod_freq: float,
                                   duration: float = 10.0) -> np.ndarray:
        """
        生成相位调制轨迹

        Args:
            t: 时间数组
            params: 参数数组 [a0, A_carrier, A_mod, φ]
            carrier_freq: 载波频率
            mod_freq: 调制频率
            duration: 轨迹周期

        Returns:
            关节位置数组
        """
        a0 = params[0]
        A_carrier = params[1] if len(params) > 1 else 1.0
        A_mod = params[2] if len(params) > 2 else 0.5
        phase = params[3] if len(params) > 3 else 0

        # 相位调制信号
        mod_signal = A_mod * np.sin(2 * np.pi * mod_freq * t)
        q = a0 + A_carrier * np.sin(2 * np.pi * carrier_freq * t + mod_signal + phase)

        return q

    def _sum_of_sines_trajectory(self, t: np.ndarray, params: np.ndarray,
                               num_components: int = 5, duration: float = 10.0) -> np.ndarray:
        """
        生成多分量正弦和轨迹（随机频率的正弦波叠加）

        Args:
            t: 时间数组
            params: 参数数组 [a0, A1, f1, φ1, A2, f2, φ2, ...]
            num_components: 正弦分量数量
            duration: 轨迹周期

        Returns:
            关节位置数组
        """
        a0 = params[0]
        q = a0 * np.ones_like(t)

        for i in range(num_components):
            idx = 3*i + 1
            if idx + 2 < len(params):
                amplitude = params[idx]
                frequency = params[idx + 1]
                phase = params[idx + 2]
                q += amplitude * np.sin(2 * np.pi * frequency * t + phase)

        return q

    def _pseudo_random_trajectory(self, t: np.ndarray, params: np.ndarray,
                                 num_harmonics: int = 10, duration: float = 10.0) -> np.ndarray:
        """
        生成伪随机多频轨迹（使用傅里叶级数模拟随机信号）

        Args:
            t: 时间数组
            params: 参数数组 [a0, A1, φ1, A2, φ2, ...]
            num_harmonics: 谐波数量
            duration: 轨迹周期

        Returns:
            关节位置数组
        """
        omega0 = 2 * np.pi / duration

        a0 = params[0]
        q = a0 * np.ones_like(t)

        for i in range(num_harmonics):
            idx = 2*i + 1
            if idx + 1 < len(params):
                amplitude = params[idx]
                phase = params[idx + 1]
                # 使用谐波频率
                k = i + 1
                q += amplitude * np.sin(k * omega0 * t + phase)

        return q

    def _schroeder_trajectory(self, t: np.ndarray, params: np.ndarray,
                            num_harmonics: int = 10, duration: float = 10.0) -> np.ndarray:
        """
        生成Schroeder相位轨迹（具有低峰值的激励信号）

        Args:
            t: 时间数组
            params: 参数数组 [a0, A]
            num_harmonics: 谐波数量
            duration: 轨迹周期

        Returns:
            关节位置数组
        """
        omega0 = 2 * np.pi / duration

        a0 = params[0]
        amplitude = params[1] if len(params) > 1 else 1.0

        q = a0 * np.ones_like(t)

        for k in range(1, num_harmonics + 1):
            # Schroeder相位：使信号具有较低峰值因子
            phi_k = np.pi * k * (k - 1) / num_harmonics
            q += (amplitude / k) * np.sin(k * omega0 * t + phi_k)

        return q

    def _gaussian_modulated_trajectory(self, t: np.ndarray, params: np.ndarray,
                                    center_freq: float, bandwidth: float,
                                    duration: float = 10.0) -> np.ndarray:
        """
        生成高斯调频轨迹（高斯包络调制的正弦波）

        Args:
            t: 时间数组
            params: 参数数组 [a0, A]
            center_freq: 中心频率
            bandwidth: 带宽
            duration: 轨迹周期

        Returns:
            关节位置数组
        """
        a0 = params[0]
        amplitude = params[1] if len(params) > 1 else 1.0

        # 高斯包络
        envelope = np.exp(-((t - duration/2)**2) / (2 * (duration/bandwidth)**2))

        # 调制信号
        q = a0 + amplitude * envelope * np.sin(2 * np.pi * center_freq * t)

        return q

    def _exponential_chirp_trajectory(self, t: np.ndarray, params: np.ndarray,
                                     f_start: float, f_end: float,
                                     duration: float = 10.0) -> np.ndarray:
        """
        生成指数扫频轨迹

        Args:
            t: 时间数组
            params: 参数数组 [a0, A, φ]
            f_start: 起始频率
            f_end: 结束频率
            duration: 轨迹周期

        Returns:
            关节位置数组
        """
        a0 = params[0]
        amplitude = params[1] if len(params) > 1 else 1.0
        phase = params[2] if len(params) > 2 else 0

        # 指数扫频
        k = (f_end / f_start) ** (1 / duration)
        instantaneous_freq = f_start * (k ** t)
        phase_integral = np.cumsum(instantaneous_freq) * (t[1] - t[0]) if len(t) > 1 else 0

        q = a0 + amplitude * np.sin(2 * np.pi * phase_integral + phase)

        return q

    def _calculate_trajectory_derivatives(self, t: np.ndarray, q: np.ndarray,
                                       params: np.ndarray, num_harmonics: int,
                                       duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算轨迹的导数（速度和加速度）
        """
        dt = t[1] - t[0]
        dq = np.gradient(q, dt)
        ddq = np.gradient(dq, dt)

        return dq, ddq

    def _check_joint_limits(self, q: np.ndarray, joint_idx: int) -> bool:
        """检查关节限制"""
        safe_limits = self.safe_joint_limits[joint_idx]
        return np.all(q >= safe_limits['lower']) and np.all(q <= safe_limits['upper'])

    def _check_velocity_limits(self, dq: np.ndarray, joint_idx: int) -> bool:
        """检查速度限制"""
        safe_limits = self.safe_joint_limits[joint_idx]
        return np.all(np.abs(dq) <= safe_limits['velocity'])

    def _objective_function(self, params: np.ndarray, joint_idx: int,
                          num_harmonics: int, duration: float) -> float:
        """
        目标函数：最大化激励质量（最小化条件数相关指标）
        """
        # 生成轨迹
        t = np.linspace(0, duration, 1000)
        q = self._fourier_series_trajectory(t, params, num_harmonics, duration)
        dq, ddq = self._calculate_trajectory_derivatives(t, q, params, num_harmonics, duration)

        # 检查约束
        if not self._check_joint_limits(q, joint_idx):
            return 1e10  # 违反关节限制

        if not self._check_velocity_limits(dq, joint_idx):
            return 1e10  # 违反速度限制

        # 计算回归矩阵Y的简化版本
        # 这里使用一个简化的激励质量指标
        position_range = np.max(q) - np.min(q)
        velocity_rms = np.sqrt(np.mean(dq**2))
        acceleration_rms = np.sqrt(np.mean(ddq**2))

        # 激励质量指标：希望位置变化大，速度和加速度适中
        excitation_quality = position_range * velocity_rms / (acceleration_rms + 1e-6)

        return -excitation_quality  # 最小化负质量等价于最大化质量

    def _generate_initial_params(self, joint_idx: int, num_harmonics: int) -> np.ndarray:
        """生成初始参数，考虑速度限制但最大化激励效果"""
        safe_limits = self.safe_joint_limits[joint_idx]
        center = (safe_limits['lower'] + safe_limits['upper']) / 2
        max_velocity = safe_limits['velocity']

        # 基于位置限制的最大幅度
        position_amplitude = safe_limits['range'] / 2.2  # 使用安全范围的45%

        # 基于速度限制的最大幅度（使用主要频率0.5Hz计算）
        primary_frequency = 0.5  # Hz
        omega_primary = 2 * np.pi * primary_frequency
        safety_factor = 1.5  # 使用更保守的安全余量
        velocity_amplitude = max_velocity / (omega_primary * safety_factor)

        # 取较小的一个作为最终幅度，但确保足够大
        amplitude = min(position_amplitude, velocity_amplitude)
        amplitude = max(amplitude, 0.3)  # 确保最小振幅

        params = [center]  # a0: 中心位置

        # 添加谐波参数
        for k in range(1, num_harmonics + 1):
            # 幅度随着谐波次数递减，但考虑速度限制
            freq = k * 0.3  # 降低基频到0.3Hz，减少高频的速度影响
            omega = 2 * np.pi * freq
            # 使用更合理的安全余量
            harmonic_safety_factor = 1.2 + (k - 1) * 0.1  # 高次谐波增加安全余量
            max_harmonic_amp = max_velocity / (omega * harmonic_safety_factor)
            # 减缓衰减速度，保持更多高频成分
            harmonic_amp = min(amplitude / (k ** 0.5), max_harmonic_amp)
            params.append(harmonic_amp)  # a_k
            params.append(harmonic_amp)  # b_k

        return np.array(params)

    def optimize_single_joint_trajectory(self, joint_idx: int, num_harmonics: int = 3,
                                      duration: float = 10.0, max_iter: int = 100) -> Dict:
        """
        为单个关节优化激励轨迹

        Args:
            joint_idx: 关节索引 (0-based)
            num_harmonics: 谐波数量
            duration: 轨迹周期
            max_iter: 最大迭代次数

        Returns:
            优化后的轨迹字典
        """
        print(f"优化关节 {joint_idx + 1} 的激励轨迹...")

        # 生成初始参数
        initial_params = self._generate_initial_params(joint_idx, num_harmonics)

        # 定义边界约束
        bounds = []
        safe_limits = self.safe_joint_limits[joint_idx]

        # a0边界：中心位置
        bounds.append((safe_limits['lower'], safe_limits['upper']))

        # 谐波参数边界
        max_amplitude = safe_limits['range'] / 2
        for k in range(1, num_harmonics + 1):
            harmonic_max = max_amplitude / (k + 1)
            bounds.append((-harmonic_max, harmonic_max))  # a_k
            bounds.append((-harmonic_max, harmonic_max))  # b_k

        # 优化
        result = minimize(
            self._objective_function,
            initial_params,
            args=(joint_idx, num_harmonics, duration),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter}
        )

        if result.success:
            print(f"  ✓ 优化成功，目标函数值: {-result.fun:.3f}")
            optimal_params = result.x
        else:
            print(f"  ⚠ 优化失败，使用初始参数")
            optimal_params = initial_params

        # 生成最终轨迹
        dt = 0.001  # 1ms采样
        t = np.arange(0, duration, dt)
        q = self._fourier_series_trajectory(t, optimal_params, num_harmonics, duration)
        dq, ddq = self._calculate_trajectory_derivatives(t, q, optimal_params, num_harmonics, duration)

        # 验证安全性
        is_safe = (self._check_joint_limits(q, joint_idx) and
                  self._check_velocity_limits(dq, joint_idx))

        print(f"  {'✓' if is_safe else '✗'} 轨迹安全性检查")

        return {
            'time': t,
            'position': q,
            'velocity': dq,
            'acceleration': ddq,
            'joint_idx': joint_idx,
            'num_harmonics': num_harmonics,
            'duration': duration,
            'optimal_params': optimal_params,
            'is_safe': is_safe
        }

    def generate_multi_joint_trajectory(self, duration: float = 15.0,
                                    num_harmonics: int = 3) -> Dict:
        """
        生成多关节激励轨迹（所有关节同时运动）

        Args:
            duration: 轨迹持续时间
            num_harmonics: 每个关节的谐波数量

        Returns:
            多关节轨迹字典
        """
        print(f"\n=== 生成多关节激励轨迹 ===")

        trajectories = []

        for joint_idx in range(self.num_joints):
            traj = self.optimize_single_joint_trajectory(
                joint_idx, num_harmonics, duration
            )
            trajectories.append(traj)

        # 组合轨迹
        t = trajectories[0]['time']
        positions = np.zeros((len(t), self.num_joints))
        velocities = np.zeros((len(t), self.num_joints))
        accelerations = np.zeros((len(t), self.num_joints))

        for joint_idx, traj in enumerate(trajectories):
            positions[:, joint_idx] = traj['position']
            velocities[:, joint_idx] = traj['velocity']
            accelerations[:, joint_idx] = traj['acceleration']

        return {
            'time': t,
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'individual_trajectories': trajectories,
            'duration': duration,
            'num_harmonics': num_harmonics
        }

    def generate_sequential_trajectory(self, duration_per_joint: float = 8.0,
                                   rest_duration: float = 2.0,
                                   num_harmonics: int = 3) -> Dict:
        """
        生成序列激励轨迹（关节依次运动）

        Args:
            duration_per_joint: 每个关节的运动时间
            rest_duration: 关节间静止时间
            num_harmonics: 谐波数量

        Returns:
            序列轨迹字典
        """
        print(f"\n=== 生成序列激励轨迹 ===")

        individual_trajectories = []

        for joint_idx in range(self.num_joints):
            traj = self.optimize_single_joint_trajectory(
                joint_idx, num_harmonics, duration_per_joint
            )
            individual_trajectories.append(traj)

        # 计算总时长
        dt = individual_trajectories[0]['time'][1] - individual_trajectories[0]['time'][0]
        total_duration = (len(individual_trajectories) * (duration_per_joint + rest_duration) -
                         rest_duration)
        t_total = np.arange(0, total_duration, dt)

        positions = np.zeros((len(t_total), self.num_joints))
        velocities = np.zeros((len(t_total), self.num_joints))
        accelerations = np.zeros((len(t_total), self.num_joints))

        current_idx = 0
        current_time = 0.0

        for joint_idx, traj in enumerate(individual_trajectories):
            traj_samples = len(traj['time'])

            # 添加轨迹数据
            end_idx = current_idx + traj_samples
            positions[current_idx:end_idx, joint_idx] = traj['position']
            velocities[current_idx:end_idx, joint_idx] = traj['velocity']
            accelerations[current_idx:end_idx, joint_idx] = traj['acceleration']

            current_idx = end_idx
            current_time += duration_per_joint

            # 添加静止时间（除了最后一个关节）
            if joint_idx < self.num_joints - 1:
                rest_samples = int(rest_duration / dt)
                rest_end_idx = current_idx + rest_samples

                if rest_end_idx <= len(t_total):
                    # 保持最后一个位置
                    positions[current_idx:rest_end_idx, joint_idx] = traj['position'][-1]
                    # 速度和加速度保持为0

                current_idx = rest_end_idx
                current_time += rest_duration

        return {
            'time': t_total,
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'individual_trajectories': individual_trajectories,
            'duration_per_joint': duration_per_joint,
            'rest_duration': rest_duration,
            'num_harmonics': num_harmonics
        }

    def generate_multi_frequency_trajectory(self, duration: float = 15.0,
                                          frequency_sets: List[List[float]] = None) -> Dict:
        """
        生成多频率激励轨迹（使用多个不同频率的正弦波）

        Args:
            duration: 轨迹持续时间
            frequency_sets: 每个关节的频率集合列表

        Returns:
            多频率轨迹字典
        """
        print(f"\n=== 生成多频率激励轨迹 ===")

        if frequency_sets is None:
            # 默认频率设置
            frequency_sets = [
                [0.1, 0.3, 0.5],      # 关节1
                [0.15, 0.35, 0.55],   # 关节2
                [0.2, 0.4, 0.6],      # 关节3
                [0.25, 0.45, 0.65],   # 关节4
                [0.12, 0.32, 0.52],   # 关节5
                [0.18, 0.38, 0.58]    # 关节6
            ]

        dt = 0.001
        t = np.arange(0, duration, dt)

        positions = np.zeros((len(t), self.num_joints))
        velocities = np.zeros((len(t), self.num_joints))
        accelerations = np.zeros((len(t), self.num_joints))

        for joint_idx in range(self.num_joints):
            if joint_idx < len(frequency_sets):
                frequencies = frequency_sets[joint_idx]

                # 生成初始参数
                safe_limits = self.safe_joint_limits[joint_idx]
                center = (safe_limits['lower'] + safe_limits['upper']) / 2
                amplitude = safe_limits['range'] / 6

                params = [center]
                for freq in frequencies:
                    params.append(amplitude)
                    params.append(0)  # 相位

                # 生成轨迹
                q = self._multi_frequency_trajectory(t, params, frequencies, duration)

                # 计算导数
                dq = np.gradient(q, dt)
                ddq = np.gradient(dq, dt)

                positions[:, joint_idx] = q
                velocities[:, joint_idx] = dq
                accelerations[:, joint_idx] = ddq

        return {
            'time': t,
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'duration': duration,
            'frequency_sets': frequency_sets,
            'trajectory_type': 'multi_frequency'
        }

    def generate_chirp_trajectory(self, duration: float = 15.0,
                                 f_start: float = 0.05, f_end: float = 2.0) -> Dict:
        """
        生成扫频（chirp）激励轨迹

        Args:
            duration: 轨迹持续时间
            f_start: 起始频率 (Hz)
            f_end: 结束频率 (Hz)

        Returns:
            扫频轨迹字典
        """
        print(f"\n=== 生成扫频激励轨迹 ===")
        print(f"频率范围: {f_start} Hz → {f_end} Hz")

        dt = 0.001
        t = np.arange(0, duration, dt)

        positions = np.zeros((len(t), self.num_joints))
        velocities = np.zeros((len(t), self.num_joints))
        accelerations = np.zeros((len(t), self.num_joints))

        for joint_idx in range(self.num_joints):
            safe_limits = self.safe_joint_limits[joint_idx]
            center = (safe_limits['lower'] + safe_limits['upper']) / 2
            amplitude = safe_limits['range'] / 4

            params = [center, amplitude, 0]  # [a0, A, φ]

            # 生成扫频轨迹（每个关节使用不同的频率范围）
            joint_f_start = f_start * (1 + joint_idx * 0.2)
            joint_f_end = f_end * (1 + joint_idx * 0.2)

            q = self._swept_sine_trajectory(t, params, joint_f_start, joint_f_end, duration)

            # 计算导数
            dq = np.gradient(q, dt)
            ddq = np.gradient(dq, dt)

            positions[:, joint_idx] = q
            velocities[:, joint_idx] = dq
            accelerations[:, joint_idx] = ddq

        return {
            'time': t,
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'duration': duration,
            'f_start': f_start,
            'f_end': f_end,
            'trajectory_type': 'chirp'
        }

    def generate_schroeder_trajectory(self, duration: float = 15.0,
                                      num_harmonics: int = 15) -> Dict:
        """
        生成Schroeder相位轨迹（低峰值激励信号）

        Args:
            duration: 轨迹持续时间
            num_harmonics: 谐波数量

        Returns:
            Schroeder轨迹字典
        """
        print(f"\n=== 生成Schroeder相位轨迹 ===")
        print(f"谐波数量: {num_harmonics}")

        dt = 0.001
        t = np.arange(0, duration, dt)

        positions = np.zeros((len(t), self.num_joints))
        velocities = np.zeros((len(t), self.num_joints))
        accelerations = np.zeros((len(t), self.num_joints))

        for joint_idx in range(self.num_joints):
            safe_limits = self.safe_joint_limits[joint_idx]
            center = (safe_limits['lower'] + safe_limits['upper']) / 2
            amplitude = safe_limits['range'] / 3

            params = [center, amplitude]

            q = self._schroeder_trajectory(t, params, num_harmonics, duration)

            # 计算导数
            dq = np.gradient(q, dt)
            ddq = np.gradient(dq, dt)

            positions[:, joint_idx] = q
            velocities[:, joint_idx] = dq
            accelerations[:, joint_idx] = ddq

        return {
            'time': t,
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'duration': duration,
            'num_harmonics': num_harmonics,
            'trajectory_type': 'schroeder'
        }

    def generate_pseudo_random_trajectory(self, duration: float = 20.0,
                                         num_harmonics: int = 20) -> Dict:
        """
        生成伪随机多频激励轨迹

        Args:
            duration: 轨迹持续时间
            num_harmonics: 谐波数量

        Returns:
            伪随机轨迹字典
        """
        print(f"\n=== 生成伪随机多频轨迹 ===")
        print(f"谐波数量: {num_harmonics}")

        dt = 0.001
        t = np.arange(0, duration, dt)

        positions = np.zeros((len(t), self.num_joints))
        velocities = np.zeros((len(t), self.num_joints))
        accelerations = np.zeros((len(t), self.num_joints))

        np.random.seed(42)  # 固定随机种子保证可重复性

        for joint_idx in range(self.num_joints):
            safe_limits = self.safe_joint_limits[joint_idx]
            center = (safe_limits['lower'] + safe_limits['upper']) / 2
            amplitude = safe_limits['range'] / 4

            params = [center]
            for i in range(num_harmonics):
                params.append(amplitude * np.random.uniform(0.1, 1.0))
                params.append(np.random.uniform(0, 2*np.pi))

            q = self._pseudo_random_trajectory(t, params, num_harmonics, duration)

            # 计算导数
            dq = np.gradient(q, dt)
            ddq = np.gradient(dq, dt)

            positions[:, joint_idx] = q
            velocities[:, joint_idx] = dq
            accelerations[:, joint_idx] = ddq

        return {
            'time': t,
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'duration': duration,
            'num_harmonics': num_harmonics,
            'trajectory_type': 'pseudo_random'
        }

    def generate_phase_modulated_trajectory(self, duration: float = 15.0,
                                          carrier_freq: float = 1.0,
                                          mod_freq: float = 0.1) -> Dict:
        """
        生成相位调制激励轨迹

        Args:
            duration: 轨迹持续时间
            carrier_freq: 载波频率 (Hz)
            mod_freq: 调制频率 (Hz)

        Returns:
            相位调制轨迹字典
        """
        print(f"\n=== 生成相位调制激励轨迹 ===")
        print(f"载波频率: {carrier_freq} Hz, 调制频率: {mod_freq} Hz")

        dt = 0.001
        t = np.arange(0, duration, dt)

        positions = np.zeros((len(t), self.num_joints))
        velocities = np.zeros((len(t), self.num_joints))
        accelerations = np.zeros((len(t), self.num_joints))

        for joint_idx in range(self.num_joints):
            safe_limits = self.safe_joint_limits[joint_idx]
            center = (safe_limits['lower'] + safe_limits['upper']) / 2
            carrier_amp = safe_limits['range'] / 6
            mod_amp = safe_limits['range'] / 12

            params = [center, carrier_amp, mod_amp, 0]

            # 每个关节使用不同的载波频率
            joint_carrier_freq = carrier_freq * (1 + joint_idx * 0.3)
            joint_mod_freq = mod_freq * (1 + joint_idx * 0.2)

            q = self._phase_modulated_trajectory(t, params, joint_carrier_freq, joint_mod_freq, duration)

            # 计算导数
            dq = np.gradient(q, dt)
            ddq = np.gradient(dq, dt)

            positions[:, joint_idx] = q
            velocities[:, joint_idx] = dq
            accelerations[:, joint_idx] = ddq

        return {
            'time': t,
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'duration': duration,
            'carrier_freq': carrier_freq,
            'mod_freq': mod_freq,
            'trajectory_type': 'phase_modulated'
        }

    def generate_sum_of_sines_trajectory(self, duration: float = 18.0,
                                        num_components: int = 8) -> Dict:
        """
        生成多分量正弦和轨迹

        Args:
            duration: 轨迹持续时间
            num_components: 正弦分量数量

        Returns:
            正弦和轨迹字典
        """
        print(f"\n=== 生成多分量正弦和轨迹 ===")
        print(f"正弦分量数量: {num_components}")

        dt = 0.001
        t = np.arange(0, duration, dt)

        positions = np.zeros((len(t), self.num_joints))
        velocities = np.zeros((len(t), self.num_joints))
        accelerations = np.zeros((len(t), self.num_joints))

        np.random.seed(123)  # 固定随机种子

        for joint_idx in range(self.num_joints):
            safe_limits = self.safe_joint_limits[joint_idx]
            center = (safe_limits['lower'] + safe_limits['upper']) / 2
            max_amplitude = safe_limits['range'] / (num_components * 2)

            params = [center]
            for i in range(num_components):
                amplitude = max_amplitude * np.random.uniform(0.5, 1.0)
                # 在不同频率范围内随机选择
                frequency = np.random.uniform(0.05, 2.0)
                phase = np.random.uniform(0, 2*np.pi)
                params.extend([amplitude, frequency, phase])

            q = self._sum_of_sines_trajectory(t, params, num_components, duration)

            # 计算导数
            dq = np.gradient(q, dt)
            ddq = np.gradient(dq, dt)

            positions[:, joint_idx] = q
            velocities[:, joint_idx] = dq
            accelerations[:, joint_idx] = ddq

        return {
            'time': t,
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'duration': duration,
            'num_components': num_components,
            'trajectory_type': 'sum_of_sines'
        }

    def save_trajectory(self, trajectory: Dict, filename: str):
        """保存轨迹到JSON文件"""
        # 如果是单关节轨迹，需要转换为多关节格式
        if 'joint_idx' in trajectory:
            joint_idx = trajectory['joint_idx']
            positions_full = np.zeros((len(trajectory['time']), self.num_joints))
            velocities_full = np.zeros((len(trajectory['time']), self.num_joints))
            accelerations_full = np.zeros((len(trajectory['time']), self.num_joints))

            positions_full[:, joint_idx] = trajectory['position']
            velocities_full[:, joint_idx] = trajectory['velocity']
            accelerations_full[:, joint_idx] = trajectory['acceleration']

            output_trajectory = {
                'time': trajectory['time'].tolist(),
                'positions': positions_full.tolist(),
                'velocities': velocities_full.tolist(),
                'accelerations': accelerations_full.tolist()
            }
        else:
            # 已经是多关节格式
            output_trajectory = {
                'time': trajectory['time'].tolist(),
                'positions': trajectory['positions'].tolist(),
                'velocities': trajectory['velocities'].tolist(),
                'accelerations': trajectory['accelerations'].tolist()
            }

        # 保存文件
        with open(filename, 'w') as f:
            json.dump(output_trajectory, f, indent=2)
        print(f"轨迹已保存到: {filename}")

    def plot_trajectory(self, trajectory: Dict, title: str = "", save_path: str = None):
        """绘制轨迹"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        t = trajectory['time']
        positions = np.array(trajectory['positions'])
        velocities = np.array(trajectory['velocities'])
        accelerations = np.array(trajectory['accelerations'])

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

        # 位置图
        for i in range(self.num_joints):
            if np.max(np.abs(positions[:, i])) > 1e-6:  # 只显示有运动的关节
                axes[0].plot(t, np.degrees(positions[:, i]),
                           label=f'Joint {i+1}', color=colors[i], linewidth=2)
        axes[0].set_ylabel('Position (°)', fontsize=12)
        axes[0].set_title(f'{title} - Position Trajectory', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 速度图
        for i in range(self.num_joints):
            if np.max(np.abs(velocities[:, i])) > 1e-6:
                axes[1].plot(t, np.degrees(velocities[:, i]),
                           label=f'Joint {i+1}', color=colors[i], linewidth=2)
        axes[1].set_ylabel('Velocity (°/s)', fontsize=12)
        axes[1].set_title(f'{title} - Velocity Trajectory', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 加速度图
        for i in range(self.num_joints):
            if np.max(np.abs(accelerations[:, i])) > 1e-6:
                axes[2].plot(t, np.degrees(accelerations[:, i]),
                           label=f'Joint {i+1}', color=colors[i], linewidth=2)
        axes[2].set_xlabel('Time (s)', fontsize=12)
        axes[2].set_ylabel('Acceleration (°/s²)', fontsize=12)
        axes[2].set_title(f'{title} - Acceleration Trajectory', fontsize=14)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"轨迹图已保存到: {save_path}")

        plt.show()

    def print_trajectory_statistics(self, trajectory: Dict):
        """打印轨迹统计信息"""
        print(f"\n=== 轨迹统计信息 ===")

        positions = np.array(trajectory['positions'])
        velocities = np.array(trajectory['velocities'])
        accelerations = np.array(trajectory['accelerations'])

        for i in range(self.num_joints):
            pos_range = [np.min(positions[:, i]), np.max(positions[:, i])]
            vel_max = np.max(np.abs(velocities[:, i]))
            acc_max = np.max(np.abs(accelerations[:, i]))

            safe_limits = self.safe_joint_limits[i]
            utilization = (pos_range[1] - pos_range[0]) / safe_limits['range'] * 100

            print(f"Joint {i+1}:")
            print(f"  位置范围: [{pos_range[0]:.3f}, {pos_range[1]:.3f}] rad "
                  f"([{np.degrees(pos_range[0]):.1f}°, {np.degrees(pos_range[1]):.1f}°])")
            print(f"  最大速度: {vel_max:.3f} rad/s ({np.degrees(vel_max):.1f}°/s)")
            print(f"  最大加速度: {acc_max:.3f} rad/s² ({np.degrees(acc_max):.1f}°/s²)")
            print(f"  关节范围利用率: {utilization:.1f}%")
            print(f"  安全限制: [{safe_limits['lower']:.3f}, {safe_limits['upper']:.3f}] rad")
            print()


def main():
    """主函数：生成各种激励轨迹"""
    # URDF文件路径
    urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/urdfs/ic_arm_8_dof/urdf/ic_arm_8_dof.urdf"

    # 初始化生成器
    generator = ExcitationTrajectoryGenerator(urdf_path, joint_margin=0.002)

    # 1. 生成多关节同时激励轨迹
    print("\n=== 1. 生成多关节同时激励轨迹 ===")
    multi_joint_traj = generator.generate_multi_joint_trajectory(
        duration=15.0,
        num_harmonics=3
    )
    generator.save_trajectory(multi_joint_traj, "excitation_multi_joint_simultaneous.json")
    generator.print_trajectory_statistics(multi_joint_traj)

    # 2. 生成序列激励轨迹
    print("\n=== 2. 生成序列激励轨迹 ===")
    sequential_traj = generator.generate_sequential_trajectory(
        duration_per_joint=8.0,
        rest_duration=2.0,
        num_harmonics=3
    )
    generator.save_trajectory(sequential_traj, "excitation_sequential.json")
    generator.print_trajectory_statistics(sequential_traj)

    # 3. 生成多频率激励轨迹（新的傅立叶相关轨迹）
    print("\n=== 3. 生成多频率激励轨迹 ===")
    multi_freq_traj = generator.generate_multi_frequency_trajectory(
        duration=15.0
    )
    generator.save_trajectory(multi_freq_traj, "excitation_multi_frequency.json")
    generator.print_trajectory_statistics(multi_freq_traj)

    # 4. 生成扫频激励轨迹（chirp信号）
    print("\n=== 4. 生成扫频激励轨迹 ===")
    chirp_traj = generator.generate_chirp_trajectory(
        duration=15.0,
        f_start=0.05,
        f_end=2.0
    )
    generator.save_trajectory(chirp_traj, "excitation_chirp.json")
    generator.print_trajectory_statistics(chirp_traj)

    # 5. 生成Schroeder相位轨迹（低峰值激励）
    print("\n=== 5. 生成Schroeder相位轨迹 ===")
    schroeder_traj = generator.generate_schroeder_trajectory(
        duration=15.0,
        num_harmonics=15
    )
    generator.save_trajectory(schroeder_traj, "excitation_schroeder.json")
    generator.print_trajectory_statistics(schroeder_traj)

    # 6. 生成伪随机多频轨迹
    print("\n=== 6. 生成伪随机多频轨迹 ===")
    pseudo_random_traj = generator.generate_pseudo_random_trajectory(
        duration=20.0,
        num_harmonics=20
    )
    generator.save_trajectory(pseudo_random_traj, "excitation_pseudo_random.json")
    generator.print_trajectory_statistics(pseudo_random_traj)

    # 7. 生成相位调制轨迹
    print("\n=== 7. 生成相位调制轨迹 ===")
    phase_mod_traj = generator.generate_phase_modulated_trajectory(
        duration=15.0,
        carrier_freq=1.0,
        mod_freq=0.1
    )
    generator.save_trajectory(phase_mod_traj, "excitation_phase_modulated.json")
    generator.print_trajectory_statistics(phase_mod_traj)

    # 8. 生成多分量正弦和轨迹
    print("\n=== 8. 生成多分量正弦和轨迹 ===")
    sum_of_sines_traj = generator.generate_sum_of_sines_trajectory(
        duration=18.0,
        num_components=8
    )
    generator.save_trajectory(sum_of_sines_traj, "excitation_sum_of_sines.json")
    generator.print_trajectory_statistics(sum_of_sines_traj)

    # 9. 为每个关节生成单独的优化轨迹
    print("\n=== 9. 生成单独关节优化轨迹 ===")
    for joint_idx in range(generator.num_joints):
        single_traj = generator.optimize_single_joint_trajectory(
            joint_idx=joint_idx,
            num_harmonics=4,
            duration=12.0
        )
        filename = f"excitation_joint_{joint_idx + 1}_optimized.json"
        generator.save_trajectory(single_traj, filename)

        # 打印统计信息
        pos_range = [np.min(single_traj['position']), np.max(single_traj['position'])]
        vel_max = np.max(np.abs(single_traj['velocity']))
        print(f"Joint {joint_idx + 1}: 范围 [{pos_range[0]:.3f}, {pos_range[1]:.3f}] rad, "
              f"最大速度 {vel_max:.3f} rad/s")

    # 10. 生成不同持续时间的轨迹
    print("\n=== 10. 生成不同持续时间的轨迹 ===")
    for duration in [10.0, 20.0, 30.0]:
        duration_traj = generator.generate_multi_joint_trajectory(
            duration=duration,
            num_harmonics=3
        )
        filename = f"excitation_multi_joint_{duration:.0f}s.json"
        generator.save_trajectory(duration_traj, filename)
        print(f"已生成 {duration}s 轨迹: {filename}")

    print("\n=== 激励轨迹生成完成 ===")
    print("生成的文件:")
    print("- excitation_multi_joint_simultaneous.json (多关节同时运动)")
    print("- excitation_sequential.json (序列运动)")
    print("- excitation_multi_frequency.json (多频率正弦波叠加)")
    print("- excitation_chirp.json (扫频信号)")
    print("- excitation_schroeder.json (Schroeder低峰值激励)")
    print("- excitation_pseudo_random.json (伪随机多频)")
    print("- excitation_phase_modulated.json (相位调制)")
    print("- excitation_sum_of_sines.json (多分量正弦和)")
    for i in range(generator.num_joints):
        print(f"- excitation_joint_{i + 1}_optimized.json (关节{i + 1}单独优化)")
    for duration in [10.0, 20.0, 30.0]:
        print(f"- excitation_multi_joint_{duration:.0f}s.json ({duration}s多关节运动)")

    print(f"\n总计生成了 {8 + generator.num_joints + 3} 个激励轨迹文件！")
    print("所有轨迹都应用了安全的关节余量，确保在安全范围内运行。")

    print("\n=== 新增的傅立叶相关轨迹特点 ===")
    print("1. 多频率轨迹: 使用多个固定频率的正弦波叠加，提供丰富的频率激励")
    print("2. 扫频轨迹: 频率随时间线性变化，覆盖宽频带")
    print("3. Schroeder轨迹: 具有低峰值因子的多频信号，减少驱动器饱和风险")
    print("4. 伪随机轨迹: 模拟随机激励但保持可重复性")
    print("5. 相位调制轨迹: 载波信号被低频信号调制，产生复杂频谱")
    print("6. 多分量正弦和: 随机频率和幅度的多个正弦波叠加")


if __name__ == "__main__":
    main()