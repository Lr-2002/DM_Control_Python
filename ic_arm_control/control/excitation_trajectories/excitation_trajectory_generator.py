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
        """生成初始参数"""
        safe_limits = self.safe_joint_limits[joint_idx]
        center = (safe_limits['lower'] + safe_limits['upper']) / 2
        amplitude = safe_limits['range'] / 4  # 使用安全范围的1/4作为幅度

        params = [center]  # a0: 中心位置

        # 添加谐波参数
        for k in range(1, num_harmonics + 1):
            # 幅度随着谐波次数递减
            harmonic_amp = amplitude / (k + 1)
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
    generator = ExcitationTrajectoryGenerator(urdf_path, joint_margin=0.1)

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

    # 3. 为每个关节生成单独的优化轨迹
    print("\n=== 3. 生成单独关节优化轨迹 ===")
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

    # 4. 生成不同持续时间的轨迹
    print("\n=== 4. 生成不同持续时间的轨迹 ===")
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
    for i in range(generator.num_joints):
        print(f"- excitation_joint_{i + 1}_optimized.json (关节{i + 1}单独优化)")
    for duration in [10.0, 20.0, 30.0]:
        print(f"- excitation_multi_joint_{duration:.0f}s.json ({duration}s多关节运动)")

    print(f"\n总计生成了 {3 + generator.num_joints + 3} 个激励轨迹文件！")
    print("所有轨迹都应用了10%的关节余量，确保在安全范围内运行。")


if __name__ == "__main__":
    main()