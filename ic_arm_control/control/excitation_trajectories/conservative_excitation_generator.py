#!/usr/bin/env python3
"""
保守的激励轨迹生成器 - 专为安全有效的参数辨识设计
重点关注重力参数辨识，确保在安全限制内

作者：Claude Code
日期：2025-09-28
"""

import numpy as np
import json
import os
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

class ConservativeExcitationGenerator:
    """保守的激励轨迹生成器 - 确保安全约束"""

    def __init__(self, urdf_path: str, safety_factor: float = 0.6):
        """
        初始化保守轨迹生成器

        Args:
            urdf_path: URDF文件路径
            safety_factor: 安全系数 (0-1) - 使用更保守的值
        """
        self.urdf_path = urdf_path
        self.safety_factor = safety_factor

        # 从URDF加载关节限制
        self.joint_limits = self._load_joint_limits()
        self.n_joints = len(self.joint_limits)

        # 保守的系统辨识参数
        self.sample_rate = 100  # Hz - 降低采样率
        self.min_duration = 60.0  # 增加持续时间以获得更好的低频分辨率
        self.max_frequency = 2.0   # 最大频率 (Hz) - 大幅降低
        self.min_frequency = 0.02  # 最小频率 (Hz) - 更低频率

        print(f"=== 保守激励轨迹生成器初始化 ===")
        print(f"关节数量: {self.n_joints}")
        print(f"采样率: {self.sample_rate} Hz")
        print(f"安全系数: {safety_factor}")
        print(f"频率范围: {self.min_frequency}-{self.max_frequency} Hz")

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

    def generate_safe_gravity_trajectory(self, duration: float = 60.0) -> Dict:
        """生成安全的重力参数辨识轨迹"""

        print(f"\n=== 生成安全重力参数辨识轨迹 ===")

        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples)

        # 为每个关节设计安全的轨迹
        all_positions = np.zeros((n_samples, self.n_joints))

        for joint_idx in range(self.n_joints):
            joint_limit = self.joint_limits[joint_idx]

            # 计算保守的幅值
            pos_amplitude = (joint_limit['range'] * self.safety_factor) / 3  # 更保守
            vel_amplitude = joint_limit['velocity_limit'] * self.safety_factor * 0.5  # 非常保守

            # 生成超低频多正弦信号
            trajectory = self._generate_ultra_low_frequency_excitation(
                t, pos_amplitude, joint_idx
            )

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
                'type': 'safe_gravity_identification',
                'description': '保守的安全重力参数辨识轨迹',
                'duration': duration,
                'sampling_rate': self.sample_rate,
                'safety_factor': self.safety_factor,
                'joint_limits': self.joint_limits,
                'design_notes': '超低频设计，确保安全约束，专注于重力参数辨识'
            }
        }

        return trajectory_data

    def _generate_ultra_low_frequency_excitation(self, t: np.ndarray, amplitude: float,
                                               joint_idx: int) -> np.ndarray:
        """生成超低频激励 - 专为重力辨识设计"""

        # 超低频设计 - 主要用于重力辨识
        base_frequencies = np.array([0.02, 0.05, 0.1, 0.2])  # 超低频

        # 为不同关节使用略有不同的频率以增加解耦性
        frequency_multipliers = [1.0, 1.2, 0.8, 1.5, 0.9, 1.1]
        freq_multiplier = frequency_multipliers[joint_idx % len(frequency_multipliers)]

        frequencies = base_frequencies * freq_multiplier

        # 保守的幅值分配
        amplitudes = amplitude * np.array([0.4, 0.3, 0.2, 0.1])

        # 生成多正弦信号
        trajectory = np.zeros_like(t)
        for i, (f, A) in enumerate(zip(frequencies, amplitudes)):
            phase = joint_idx * np.pi / 6  # 为不同关节添加相位差
            trajectory += A * np.sin(2 * np.pi * f * t + phase)

        # 添加非常缓慢的线性漂移组件
        drift_freq = 0.005  # Hz - 超慢漂移
        drift_amplitude = amplitude * 0.3
        drift_component = drift_amplitude * np.sin(2 * np.pi * drift_freq * t)

        trajectory += drift_component

        # 添加直流偏置到关节中心
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
            acc_limit = vel_limit * 5  # 保守的加速度限制
            acc_utilization = (joint_max_acc / acc_limit) * 100

            max_velocities.append(joint_max_vel)
            max_accelerations.append(joint_max_acc)
            velocity_utilizations.append(vel_utilization)

            status = "✅" if vel_utilization <= 80 else "⚠️" if vel_utilization <= 90 else "❌"
            print(f"  Joint{joint_idx+1}: {vel_utilization:.1f}% {status} "
                  f"(最大速度: {joint_max_vel:.3f} rad/s, 位置范围: {np.degrees(pos_range):.1f}°)")

        # 统计
        max_utilization = max(velocity_utilizations)
        avg_utilization = np.mean(velocity_utilizations)
        safe_joints = sum(1 for util in velocity_utilizations if util <= 80)

        print(f"\n约束检查总结:")
        print(f"  最高速度利用率: {max_utilization:.1f}%")
        print(f"  平均速度利用率: {avg_utilization:.1f}%")
        print(f"  安全关节数量: {safe_joints}/{len(velocity_utilizations)}")

        if max_utilization <= 80:
            print(f"  ✅ 轨迹设计安全 - 适合重力参数辨识")
        elif max_utilization <= 90:
            print(f"  ⚠️ 轨迹较为保守 - 可以使用")
        else:
            print(f"  ❌ 轨迹超出安全限制")

    def save_trajectory(self, trajectory_data: Dict, filename: str,
                       output_dir: str = "/Users/lr-2002/project/instantcreation/IC_arm_control"):
        """保存轨迹数据"""
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)

        print(f"轨迹已保存到: {filepath}")
        return filepath

    def analyze_excitation_quality(self, trajectory_data: Dict) -> Dict:
        """分析激励质量"""
        positions = np.array(trajectory_data['positions'])
        velocities = np.array(trajectory_data['velocities'])
        accelerations = np.array(trajectory_data['accelerations'])

        n_samples, n_joints = positions.shape

        analysis = {
            'duration': trajectory_data['trajectory_info']['duration'],
            'sampling_rate': trajectory_data['trajectory_info']['sampling_rate'],
            'n_samples': n_samples,
            'n_joints': n_joints,
            'joint_analysis': []
        }

        for joint_idx in range(n_joints):
            # 计算各种统计量
            pos_data = positions[:, joint_idx]
            vel_data = velocities[:, joint_idx]
            acc_data = accelerations[:, joint_idx]

            joint_analysis = {
                'joint_id': joint_idx + 1,
                'position': {
                    'range_deg': float(np.degrees(pos_data.max() - pos_data.min())),
                    'std_deg': float(np.degrees(pos_data.std())),
                    'mean_deg': float(np.degrees(pos_data.mean()))
                },
                'velocity': {
                    'max': float(np.max(np.abs(vel_data))),
                    'rms': float(np.sqrt(np.mean(vel_data**2))),
                    'std': float(vel_data.std())
                },
                'acceleration': {
                    'max': float(np.max(np.abs(acc_data))),
                    'rms': float(np.sqrt(np.mean(acc_data**2))),
                    'std': float(acc_data.std())
                },
                'frequency_content': {
                    'dominant_freq': self._estimate_dominant_frequency(pos_data, trajectory_data['trajectory_info']['sampling_rate']),
                    'bandwidth': self._estimate_bandwidth(pos_data, trajectory_data['trajectory_info']['sampling_rate'])
                }
            }

            analysis['joint_analysis'].append(joint_analysis)

        return analysis

    def _estimate_dominant_frequency(self, signal: np.ndarray, fs: float) -> float:
        """估计主导频率"""
        from scipy import signal as scipy_signal

        # 计算功率谱密度
        freqs, psd = scipy_signal.welch(signal, fs, nperseg=min(1024, len(signal)//4))

        # 找到主导频率（忽略DC分量）
        valid_freqs = freqs[1:]  # 忽略0Hz
        valid_psd = psd[1:]
        dominant_freq_idx = np.argmax(valid_psd)

        return float(valid_freqs[dominant_freq_idx])

    def _estimate_bandwidth(self, signal: np.ndarray, fs: float) -> float:
        """估计信号带宽"""
        from scipy import signal as scipy_signal

        # 计算功率谱密度
        freqs, psd = scipy_signal.welch(signal, fs, nperseg=min(1024, len(signal)//4))

        # 找到包含90%能量的带宽
        total_power = np.sum(psd)
        cumulative_power = np.cumsum(psd)
        threshold = 0.9 * total_power

        bandwidth_idx = np.where(cumulative_power >= threshold)[0]
        if len(bandwidth_idx) > 0:
            return float(freqs[bandwidth_idx[0]])
        else:
            return float(freqs[-1])

def main():
    """主函数 - 生成保守的安全轨迹"""
    # URDF文件路径
    urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/urdfs/ic_arm_8_dof/urdf/ic_arm_8_dof.urdf"

    # 创建保守轨迹生成器
    generator = ConservativeExcitationGenerator(urdf_path, safety_factor=0.6)

    # 生成安全重力辨识轨迹
    trajectory_data = generator.generate_safe_gravity_trajectory(duration=60.0)

    # 保存轨迹
    filename = "conservative_gravity_identification.json"
    filepath = generator.save_trajectory(trajectory_data, filename)

    # 分析激励质量
    print(f"\n=== 激励质量分析 ===")
    quality_analysis = generator.analyze_excitation_quality(trajectory_data)

    print(f"轨迹基本信息:")
    print(f"  持续时间: {quality_analysis['duration']} 秒")
    print(f"  采样率: {quality_analysis['sampling_rate']} Hz")
    print(f"  数据点数: {quality_analysis['n_samples']}")
    print(f"  关节数量: {quality_analysis['n_joints']}")

    print(f"\n各关节分析:")
    for joint_data in quality_analysis['joint_analysis']:
        joint_id = joint_data['joint_id']
        pos_analysis = joint_data['position']
        vel_analysis = joint_data['velocity']
        acc_analysis = joint_data['acceleration']
        freq_analysis = joint_data['frequency_content']

        print(f"  Joint{joint_id}:")
        print(f"    位置范围: {pos_analysis['range_deg']:.1f}° (标准差: {pos_analysis['std_deg']:.1f}°)")
        print(f"    速度: 最大 {vel_analysis['max']:.3f} rad/s, RMS {vel_analysis['rms']:.3f} rad/s")
        print(f"    加速度: 最大 {acc_analysis['max']:.3f} rad/s², RMS {acc_analysis['rms']:.3f} rad/s²")
        print(f"    主导频率: {freq_analysis['dominant_freq']:.3f} Hz")

    print(f"\n✅ 保守激励轨迹生成完成!")
    print(f"生成的文件: {filepath}")

if __name__ == "__main__":
    main()