#!/usr/bin/env python3
"""
优化的动力学参数辨识轨迹生成器
基于系统辨识理论和持续激励条件设计

作者：Claude Code
日期：2025-09-28
"""

import numpy as np
import json
import os
from scipy import signal
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class OptimalExcitationTrajectoryGenerator:
    """优化的激励轨迹生成器 - 专为动力学参数辨识设计"""

    def __init__(self, urdf_path: str, safety_factor: float = 0.8):
        """
        初始化优化轨迹生成器

        Args:
            urdf_path: URDF文件路径
            safety_factor: 安全系数 (0-1)
        """
        self.urdf_path = urdf_path
        self.safety_factor = safety_factor

        # 从URDF加载关节限制
        self.joint_limits = self._load_joint_limits()
        self.n_joints = len(self.joint_limits)

        # 系统辨识参数
        self.sample_rate = 200  # Hz - 适中的采样率
        self.min_duration = 30.0  # 最小持续时间
        self.max_frequency = 5.0   # 最大频率 (Hz) - 降低以避免超速
        self.min_frequency = 0.05  # 最小频率 (Hz)

        print(f"=== 优化激励轨迹生成器初始化 ===")
        print(f"关节数量: {self.n_joints}")
        print(f"采样率: {self.sample_rate} Hz")
        print(f"安全系数: {safety_factor}")

    def _load_joint_limits(self) -> List[Dict]:
        """从URDF加载关节限制"""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(self.urdf_path)
            root = tree.getroot()

            joint_limits = []
            for joint in root.findall('.//joint'):
                joint_name = joint.get('name')
                joint_type = joint.get('type')

                if joint_type == 'revolute':
                    limit_elem = joint.find('limit')
                    if limit_elem is not None:
                        lower = float(limit_elem.get('lower', 0))
                        upper = float(limit_elem.get('upper', 0))
                        velocity = float(limit_elem.get('velocity', 1.57))

                        joint_limits.append({
                            'name': joint_name,
                            'lower': lower,
                            'upper': upper,
                            'velocity_limit': velocity,
                            'range': upper - lower,
                            'center': (lower + upper) / 2
                        })

            return joint_limits
        except Exception as e:
            print(f"加载URDF失败: {e}")
            # 返回默认的6关节限制
            return [
                {'name': 'joint1', 'lower': -1.383, 'upper': 0.301, 'velocity_limit': 1.57, 'range': 1.684, 'center': -0.541},
                {'name': 'joint2', 'lower': -3.064, 'upper': 0.434, 'velocity_limit': 1.57, 'range': 3.498, 'center': -1.315},
                {'name': 'joint3', 'lower': -0.485, 'upper': 2.041, 'velocity_limit': 1.57, 'range': 2.526, 'center': 0.778},
                {'name': 'joint4', 'lower': -3.007, 'upper': 2.260, 'velocity_limit': 1.57, 'range': 5.267, 'center': -0.374},
                {'name': 'joint5', 'lower': -1.770, 'upper': 1.740, 'velocity_limit': 1.57, 'range': 3.510, 'center': -0.015},
                {'name': 'joint6', 'lower': -2.071, 'upper': 3.859, 'velocity_limit': 1.57, 'range': 5.930, 'center': 0.894}
            ]

    def design_optimal_frequencies(self, n_frequencies: int = 8) -> np.ndarray:
        """
        设计最优频率分布 - 对数间距覆盖动力学带宽

        Args:
            n_frequencies: 频率数量

        Returns:
            最优频率数组 (Hz)
        """
        # 对数间距频率 - 更好的参数分离
        frequencies = np.logspace(
            np.log10(self.min_frequency),
            np.log10(self.max_frequency),
            n_frequencies
        )

        print(f"最优频率设计:")
        print(f"  频率范围: {self.min_frequency:.2f} - {self.max_frequency:.1f} Hz")
        print(f"  频率数量: {n_frequencies}")
        print(f"  频率分布: {frequencies}")

        return frequencies

    def schroeder_multisine(self, t: np.ndarray, frequencies: np.ndarray,
                          amplitudes: np.ndarray) -> np.ndarray:
        """
        生成Schroeder相位多正弦信号 - 低波峰因子

        Args:
            t: 时间数组
            frequencies: 频率数组
            amplitudes: 幅值数组

        Returns:
            Schroeder多正弦信号
        """
        signal = np.zeros_like(t)

        # 计算Schroeder相位
        phases = np.zeros_like(frequencies)
        for i in range(len(frequencies)):
            if i == 0:
                phases[i] = 0
            else:
                phases[i] = phases[i-1] + amplitudes[i-1]**2

        # 生成信号
        for i, (f, A, phi) in enumerate(zip(frequencies, amplitudes, phases)):
            signal += A * np.sin(2 * np.pi * f * t + phi)

        return signal

    def generate_parameter_specific_trajectory(self, trajectory_type: str,
                                            duration: float = 45.0) -> Dict:
        """
        生成针对特定参数类型的优化轨迹

        Args:
            trajectory_type: 轨迹类型 ('inertia', 'friction', 'gravity', 'coriolis', 'comprehensive')
            duration: 持续时间

        Returns:
            轨迹数据字典
        """
        print(f"\n=== 生成{trajectory_type}辨识轨迹 ===")

        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples)

        # 为每个关节设计轨迹
        all_positions = np.zeros((n_samples, self.n_joints))

        for joint_idx in range(self.n_joints):
            joint_limit = self.joint_limits[joint_idx]

            # 计算安全幅值
            pos_amplitude = (joint_limit['range'] * self.safety_factor) / 2
            vel_amplitude = joint_limit['velocity_limit'] * self.safety_factor

            if trajectory_type == 'inertia':
                # 惯性参数辨识 - 需要高加速度激励
                trajectory = self._generate_inertia_excitation(t, pos_amplitude, joint_idx)
            elif trajectory_type == 'friction':
                # 摩擦参数辨识 - 需要全速度范围覆盖
                trajectory = self._generate_friction_excitation(t, pos_amplitude, vel_amplitude, joint_idx)
            elif trajectory_type == 'gravity':
                # 重力参数辨识 - 需要全工作空间覆盖
                trajectory = self._generate_gravity_excitation(t, pos_amplitude, joint_idx)
            elif trajectory_type == 'coriolis':
                # 科氏力参数辨识 - 需要多关节耦合
                trajectory = self._generate_coriolis_excitation(t, pos_amplitude, joint_idx)
            elif trajectory_type == 'comprehensive':
                # 综合辨识 - 所有参数类型
                trajectory = self._generate_comprehensive_excitation(t, pos_amplitude, joint_idx)
            else:
                raise ValueError(f"未知的轨迹类型: {trajectory_type}")

            # 确保在安全范围内
            trajectory = np.clip(trajectory,
                               joint_limit['lower'],
                               joint_limit['upper'])

            all_positions[:, joint_idx] = trajectory

        # 计算速度和加速度
        dt = 1.0 / self.sample_rate
        velocities = np.zeros_like(all_positions)
        accelerations = np.zeros_like(all_positions)

        for joint_idx in range(self.n_joints):
            # 使用中心差分法计算导数
            velocities[1:-1, joint_idx] = (all_positions[2:, joint_idx] - all_positions[:-2, joint_idx]) / (2 * dt)
            velocities[0, joint_idx] = (all_positions[1, joint_idx] - all_positions[0, joint_idx]) / dt
            velocities[-1, joint_idx] = (all_positions[-1, joint_idx] - all_positions[-2, joint_idx]) / dt

            accelerations[1:-1, joint_idx] = (velocities[2:, joint_idx] - velocities[:-2, joint_idx]) / (2 * dt)
            accelerations[0, joint_idx] = (velocities[1, joint_idx] - velocities[0, joint_idx]) / dt
            accelerations[-1, joint_idx] = (velocities[-1, joint_idx] - velocities[-2, joint_idx]) / dt

        # 检查约束
        self._check_trajectory_constraints(all_positions, velocities, accelerations)

        # 创建输出数据
        trajectory_data = {
            'time': t.tolist(),
            'positions': all_positions.tolist(),
            'velocities': velocities.tolist(),
            'accelerations': accelerations.tolist(),
            'trajectory_info': {
                'type': trajectory_type,
                'description': f'优化的{trajectory_type}参数辨识轨迹',
                'duration': duration,
                'sampling_rate': self.sample_rate,
                'safety_factor': self.safety_factor,
                'joint_limits': self.joint_limits
            }
        }

        return trajectory_data

    def _generate_inertia_excitation(self, t: np.ndarray, amplitude: float,
                                   joint_idx: int) -> np.ndarray:
        """生成惯性参数辨识激励 - 高频、高加速度"""

        # 降低频率和幅值以避免超速
        frequencies = np.array([1.0, 2.5, 4.0])  # 中高频
        amplitudes = amplitude * np.array([0.2, 0.1, 0.05])  # 大幅降低幅值

        # Schroeder多正弦信号
        trajectory = self.schroeder_multisine(t, frequencies, amplitudes)

        # 添加直流偏置到关节中心
        joint_center = self.joint_limits[joint_idx]['center']
        trajectory += joint_center

        return trajectory

    def _generate_friction_excitation(self, t: np.ndarray, pos_amplitude: float,
                                    vel_amplitude: float, joint_idx: int) -> np.ndarray:
        """生成摩擦参数辨识激励 - 全速度范围覆盖"""

        # 降低扫频范围以避免超速
        f_start, f_end = 0.02, 0.5  # Hz - 大幅降低频率
        sweep_duration = t[-1] - t[0]

        # 对数扫频信号
        instantaneous_freq = f_start * (f_end/f_start)**(t/sweep_duration)
        phase = 2 * np.pi * f_start * sweep_duration * ( (f_end/f_start)**(t/sweep_duration) - 1 ) / np.log(f_end/f_start)

        # 位置信号 - 降低幅值
        trajectory = pos_amplitude * 0.3 * np.sin(phase)

        # 添加低速振荡用于Stribeck效应辨识
        low_freq_component = 0.05 * pos_amplitude * np.sin(2 * np.pi * 0.05 * t)
        trajectory += low_freq_component

        # 添加直流偏置
        joint_center = self.joint_limits[joint_idx]['center']
        trajectory += joint_center

        return trajectory

    def _generate_gravity_excitation(self, t: np.ndarray, amplitude: float,
                                   joint_idx: int) -> np.ndarray:
        """生成重力参数辨识激励 - 慢速全工作空间覆盖"""

        # 非常慢的频率用于重力辨识
        frequencies = np.array([0.1, 0.2, 0.3])  # 超低频
        amplitudes = amplitude * np.array([0.6, 0.3, 0.1])

        # 多正弦信号
        trajectory = np.zeros_like(t)
        for i, (f, A) in enumerate(zip(frequencies, amplitudes)):
            trajectory += A * np.sin(2 * np.pi * f * t)

        # 添加缓慢的线性漂移
        drift_component = 0.3 * amplitude * np.sin(2 * np.pi * 0.05 * t)
        trajectory += drift_component

        # 确保覆盖整个工作空间
        joint_center = self.joint_limits[joint_idx]['center']
        trajectory += joint_center

        return trajectory

    def _generate_coriolis_excitation(self, t: np.ndarray, amplitude: float,
                                    joint_idx: int) -> np.ndarray:
        """生成科氏力参数辨识激励 - 多关节耦合"""

        # 降低频率和幅值以避免超速
        frequencies = np.array([0.5, 1.5, 3.0])  # 中低频
        amplitudes = amplitude * np.array([0.2, 0.1, 0.05])  # 大幅降低幅值

        # 为不同关节设置不同的相位关系以产生耦合
        phase_offset = joint_idx * np.pi / 3  # 60度相位差

        trajectory = np.zeros_like(t)
        for i, (f, A) in enumerate(zip(frequencies, amplitudes)):
            trajectory += A * np.sin(2 * np.pi * f * t + phase_offset)

        # 添加直流偏置
        joint_center = self.joint_limits[joint_idx]['center']
        trajectory += joint_center

        return trajectory

    def _generate_comprehensive_excitation(self, t: np.ndarray, amplitude: float,
                                         joint_idx: int) -> np.ndarray:
        """生成综合参数辨识激励 - 所有参数类型"""

        # 大幅降低频率和幅值
        frequencies = np.array([0.05, 0.15, 0.4, 1.0, 2.5])  # 低中频
        amplitudes = amplitude * np.array([0.15, 0.12, 0.1, 0.07, 0.04])

        # Schroeder多正弦信号
        trajectory = self.schroeder_multisine(t, frequencies, amplitudes)

        # 添加特殊激励成分 - 大幅降低
        # 1. 低频大位移（重力）
        gravity_component = 0.2 * amplitude * np.sin(2 * np.pi * 0.08 * t)
        # 2. 中频耦合（科氏力）
        coriolis_component = 0.1 * amplitude * np.sin(2 * np.pi * 0.8 * t + joint_idx * np.pi/4)
        # 3. 高频小位移（惯性）
        inertia_component = 0.05 * amplitude * np.sin(2 * np.pi * 3.0 * t)

        trajectory += gravity_component + coriolis_component + inertia_component

        # 添加直流偏置
        joint_center = self.joint_limits[joint_idx]['center']
        trajectory += joint_center

        return trajectory

    def _check_trajectory_constraints(self, positions: np.ndarray,
                                   velocities: np.ndarray,
                                   accelerations: np.ndarray):
        """检查轨迹约束"""
        print(f"\n=== 轨迹约束检查 ===")

        max_velocities = []
        max_accelerations = []
        velocity_utilizations = []

        for joint_idx in range(self.n_joints):
            joint_limit = self.joint_limits[joint_idx]

            # 检查位置限制
            pos_min, pos_max = positions[:, joint_idx].min(), positions[:, joint_idx].max()
            pos_range = pos_max - pos_min

            # 检查速度限制
            joint_max_vel = np.max(np.abs(velocities[:, joint_idx]))
            vel_limit = joint_limit['velocity_limit']
            vel_utilization = (joint_max_vel / vel_limit) * 100

            # 检查加速度限制
            joint_max_acc = np.max(np.abs(accelerations[:, joint_idx]))
            acc_limit = vel_limit * 10  # 假设加速度限制为速度限制的10倍
            acc_utilization = (joint_max_acc / acc_limit) * 100

            max_velocities.append(joint_max_vel)
            max_accelerations.append(joint_max_acc)
            velocity_utilizations.append(vel_utilization)

            status = "✅" if vel_utilization <= 90 else "⚠️" if vel_utilization <= 100 else "❌"
            print(f"  Joint{joint_idx+1}: {vel_utilization:.1f}% {status} "
                  f"(最大速度: {joint_max_vel:.3f} rad/s, 位置范围: {np.degrees(pos_range):.1f}°)")

        # 统计
        max_utilization = max(velocity_utilizations)
        avg_utilization = np.mean(velocity_utilizations)
        safe_joints = sum(1 for util in velocity_utilizations if util <= 90)

        print(f"\n约束检查总结:")
        print(f"  最高速度利用率: {max_utilization:.1f}%")
        print(f"  平均速度利用率: {avg_utilization:.1f}%")
        print(f"  安全关节数量: {safe_joints}/{len(velocity_utilizations)}")

        if max_utilization <= 90:
            print(f"  ✅ 轨迹设计安全")
        elif max_utilization <= 100:
            print(f"  ⚠️ 轨迹接近限制")
        else:
            print(f"  ❌ 轨迹超出限制")

    def save_trajectory(self, trajectory_data: Dict, filename: str,
                       output_dir: str = "/Users/lr-2002/project/instantcreation/IC_arm_control"):
        """保存轨迹数据"""
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)

        print(f"轨迹已保存到: {filepath}")
        return filepath

    def generate_all_optimal_trajectories(self):
        """生成所有优化的激励轨迹"""
        print("\n" + "="*60)
        print("生成所有优化的动力学参数辨识轨迹")
        print("="*60)

        trajectory_types = [
            ('inertia', '惯性参数辨识轨迹'),
            ('friction', '摩擦参数辨识轨迹'),
            ('gravity', '重力参数辨识轨迹'),
            ('coriolis', '科氏力参数辨识轨迹'),
            ('comprehensive', '综合参数辨识轨迹')
        ]

        generated_files = []

        for traj_type, description in trajectory_types:
            print(f"\n{'='*20} {description} {'='*20}")

            # 生成轨迹
            trajectory_data = self.generate_parameter_specific_trajectory(
                traj_type, duration=45.0
            )

            # 保存文件
            filename = f"optimal_excitation_{traj_type}.json"
            filepath = self.save_trajectory(trajectory_data, filename)
            generated_files.append(filepath)

        print(f"\n{'='*60}")
        print(f"优化轨迹生成完成!")
        print(f"共生成了 {len(generated_files)} 种轨迹类型:")
        for i, (traj_type, description) in enumerate(trajectory_types):
            print(f"  {i+1}. {description}")
        print(f"{'='*60}")

        return generated_files

def main():
    """主函数 - 生成所有优化轨迹"""
    # URDF文件路径
    urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/urdfs/ic_arm_8_dof/urdf/ic_arm_8_dof.urdf"

    # 创建优化轨迹生成器
    generator = OptimalExcitationTrajectoryGenerator(urdf_path, safety_factor=0.8)

    # 生成所有优化轨迹
    generated_files = generator.generate_all_optimal_trajectories()

    print(f"\n✅ 优化激励轨迹生成完成!")
    print(f"生成的文件:")
    for filepath in generated_files:
        print(f"  - {filepath}")

if __name__ == "__main__":
    main()