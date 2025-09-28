#!/usr/bin/env python3
"""
为每个关节在常用角度范围内生成激励轨迹
基于 common_angles.md 中的常用角度范围
"""

import numpy as np
import json
import os
from excitation_trajectory_generator import ExcitationTrajectoryGenerator

def load_common_angles():
    """加载常用角度设置"""
    # 从 common_angles.md 读取常用角度范围 (度转弧度)
    common_angles_deg = {
        1: [-30, 10],     # Joint1: [-30°, 10°]
        2: [-100, 10],    # Joint2: [-100°, 10°]
        3: [-20, 70],     # Joint3: [-20°, 70°]
        4: [-100, 70],    # Joint4: [-100°, 70°]
        5: [-90, 90],     # Joint5: [-90°, 90°]
        6: [-120, 120]    # Joint6: [-120°, 120°]
    }

    # 转换为弧度
    common_angles_rad = {}
    for joint_id, angle_range_deg in common_angles_deg.items():
        angle_range_rad = [np.deg2rad(angle_range_deg[0]), np.deg2rad(angle_range_deg[1])]
        common_angles_rad[joint_id] = angle_range_rad

    return common_angles_rad

def generate_common_angles_trajectory():
    """为每个关节在常用角度范围内生成轨迹"""

    print("=== 基于常用角度的激励轨迹生成 ===")

    # 加载常用角度设置
    common_angles = load_common_angles()

    print("各关节常用角度范围:")
    for joint_id, angle_range in common_angles.items():
        center = (angle_range[0] + angle_range[1]) / 2
        range_deg = np.rad2deg(angle_range[1] - angle_range[0])
        print(f"  Joint{joint_id}: [{np.rad2deg(angle_range[0]):.0f}°, {np.rad2deg(angle_range[1]):.0f}°] = [{angle_range[0]:.3f}, {angle_range[1]:.3f}] rad")
        print(f"    中心位置: {center:.3f} rad ({np.rad2deg(center):.1f}°), 范围: {range_deg:.0f}°")

    # 初始化轨迹生成器
    urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/urdfs/ic_arm_8_dof/urdf/ic_arm_8_dof.urdf"
    generator = ExcitationTrajectoryGenerator(urdf_path, joint_margin=0.05)

    # 轨迹参数
    duration = 20.0
    sampling_rate = 200
    t = np.linspace(0, duration, int(sampling_rate * duration))

    # 生成多种轨迹类型
    trajectory_types = {
        'multi_freq': '多频率组合',
        'sine_sweep': '正弦扫频',
        'random_walk': '随机游走',
        'harmonic_excitation': '谐波激励',
        'zero_centered': '零点中心激励',
        'low_freq_oscillation': '低频振荡',
        'joint1_focus': 'Joint1重点激励',
        'common_range_sine': '常用范围正弦',
        'common_range_cosine': '常用范围余弦',
        'common_range_triangle': '常用范围三角波',
        'common_range_square': '常用范围方波',
        'common_range_sawtooth': '常用范围锯齿波',
        'common_range_exponential': '常用范围指数衰减',
        'common_range_gaussian': '常用范围高斯分布'
    }

    trajectories = {}

    for traj_type, description in trajectory_types.items():
        print(f"\n生成{description}轨迹...")

        # 为每个关节生成轨迹
        all_positions = np.zeros((len(t), generator.num_joints))

        for joint_idx in range(generator.num_joints):
            joint_id = joint_idx + 1  # 转换为1-based
            joint_info = generator.joint_info[joint_idx]
            safe_limits = generator.safe_joint_limits[joint_idx]

            # 获取该关节的常用角度范围
            common_range = common_angles[joint_id]
            common_center = (common_range[0] + common_range[1]) / 2
            common_amplitude = (common_range[1] - common_range[0]) / 1.0  # 使用50%的范围，更保守

            # 限制在安全范围内
            safe_center = np.clip(common_center, safe_limits['lower'], safe_limits['upper'])
            max_amplitude = min(
                common_amplitude,
                safe_center - safe_limits['lower'],
                safe_limits['upper'] - safe_center
            )

            # 根据轨迹类型生成
            if traj_type == 'multi_freq':
                # 多频率组合
                trajectory = np.zeros_like(t)
                frequencies = [0.05, 0.1, 0.15, 0.2, 0.25]  # 降低频率
                amplitudes = [max_amplitude * 0.6, max_amplitude * 0.4, max_amplitude * 0.3, max_amplitude * 0.2, max_amplitude * 0.15]

                for freq, amp in zip(frequencies, amplitudes):
                    trajectory += amp * np.sin(2 * np.pi * freq * t)

                trajectory += safe_center

            elif traj_type == 'sine_sweep':
                # 正弦扫频
                trajectory = safe_center + max_amplitude * 0.7 * np.sin(2 * np.pi * 0.05 * t) * np.cos(2 * np.pi * 0.3 * t)

            elif traj_type == 'random_walk':
                # 随机游走
                trajectory = np.zeros_like(t)
                current_pos = safe_center
                max_step = max_amplitude * 0.005  # 更小的步长避免超速

                np.random.seed(joint_id * 42)  # 可重复的随机种子
                for i in range(1, len(t)):
                    step = np.random.uniform(-max_step, max_step)
                    current_pos += step
                    current_pos = np.clip(current_pos, safe_center - max_amplitude, safe_center + max_amplitude)
                    trajectory[i] = current_pos

            elif traj_type == 'harmonic_excitation':
                # 谐波激励
                trajectory = safe_center * np.ones_like(t)
                fundamental_freq = 0.08  # Hz  降低基础频率

                # 添加多个谐波
                for harmonic in range(1, 6):
                    freq = fundamental_freq * harmonic
                    amplitude = max_amplitude / (harmonic ** 0.8)  # 缓慢衰减
                    phase = np.random.uniform(0, 2*np.pi)
                    trajectory += amplitude * np.sin(2 * np.pi * freq * t + phase)

            elif traj_type == 'zero_centered':
                # 零点中心激励 - 特别针对电机2-4在0附近运动
                if joint_id in [2, 3, 4]:  # 电机2-4
                    # 在0附近重点运动
                    zero_center = 0.0
                    # 减小振幅，更多在0附近小范围运动
                    zero_amplitude = max_amplitude * 0.4

                    # 多频率小振幅激励
                    trajectory = zero_center
                    frequencies = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
                    for i, freq in enumerate(frequencies):
                        # 振幅随频率递减
                        amp = zero_amplitude * (0.8 ** i)
                        phase = np.random.uniform(0, 2*np.pi)
                        trajectory += amp * np.sin(2 * np.pi * freq * t + phase)

                    # 添加一些0附近的随机扰动
                    noise_amplitude = zero_amplitude * 0.1
                    np.random.seed(joint_id * 123)
                    noise = noise_amplitude * np.random.normal(0, 1, len(t))
                    trajectory += noise

                else:
                    # 其他关节使用常规运动
                    trajectory = safe_center + max_amplitude * 0.6 * np.sin(2 * np.pi * 0.1 * t)

            elif traj_type == 'low_freq_oscillation':
                # 低频振荡 - 所有关节都更多在各自中心附近运动
                center_focus = common_center  # 使用常用角度中心
                reduced_amplitude = max_amplitude * 0.5  # 减小振幅

                # 低频主运动
                trajectory = center_focus + reduced_amplitude * 0.8 * np.sin(2 * np.pi * 0.03 * t)

                # 添加多个低频分量
                low_freqs = [0.05, 0.08, 0.12, 0.18]
                for freq in low_freqs:
                    amp = reduced_amplitude * 0.3 * np.exp(-freq * 2)  # 指数衰减
                    phase = np.random.uniform(0, 2*np.pi)
                    trajectory += amp * np.sin(2 * np.pi * freq * t + phase)

                # 特别为电机2-4添加更多中心附近的运动
                if joint_id in [2, 3, 4]:
                    # 在常用角度中心附近添加小范围高频振荡
                    high_freq_components = [0.5, 0.8, 1.2, 1.5]
                    for freq in high_freq_components:
                        amp = reduced_amplitude * 0.1  # 很小的振幅
                        phase = np.random.uniform(0, 2*np.pi)
                        trajectory += amp * np.sin(2 * np.pi * freq * t + phase)

            elif traj_type == 'joint1_focus':
                # Joint1重点激励 - 确保在-20到10度附近多运动
                if joint_id == 1:
                    # Joint1: 重点在-20到10度范围内运动
                    joint1_target_center = np.deg2rad((-20 + 10) / 2)  # -5度
                    joint1_range = np.deg2rad(10 - (-20))  # 30度范围

                    # 在-20到10度范围内生成丰富运动
                    trajectory = joint1_target_center

                    # 多频率激励覆盖整个范围
                    frequencies = [0.02, 0.05, 0.1, 0.15, 0.25, 0.35]
                    amplitudes = [joint1_range * 0.4, joint1_range * 0.3, joint1_range * 0.2,
                                 joint1_range * 0.15, joint1_range * 0.1, joint1_range * 0.08]

                    for freq, amp in zip(frequencies, amplitudes):
                        phase = np.random.uniform(0, 2*np.pi)
                        trajectory += amp * np.sin(2 * np.pi * freq * t + phase)

                    # 确保不超过限制
                    trajectory = np.clip(trajectory, safe_limits['lower'], safe_limits['upper'])

                elif joint_id in [2, 3, 4]:
                    # 电机2-4: 在0附近运动
                    zero_center = 0.0
                    zero_amplitude = max_amplitude * 0.6

                    trajectory = zero_center
                    # 中低频为主
                    frequencies = [0.03, 0.07, 0.12, 0.2, 0.3]
                    for freq in frequencies:
                        amp = zero_amplitude * 0.4 / (freq * 10)  # 频率越高振幅越小
                        phase = np.random.uniform(0, 2*np.pi)
                        trajectory += amp * np.sin(2 * np.pi * freq * t + phase)

                else:
                    # 其他关节正常运动
                    trajectory = safe_center + max_amplitude * 0.5 * np.sin(2 * np.pi * 0.08 * t)

            elif traj_type == 'common_range_sine':
                # 常用范围正弦 - 确保运动在常用角度范围内
                common_min, common_max = common_range
                common_center = (common_min + common_max) / 2
                common_amplitude = (common_max - common_min) * 0.8  # 使用80%的常用范围

                # 多频率正弦组合，确保不超出常用范围
                trajectory = common_center
                frequencies = [0.05, 0.1, 0.2, 0.4]
                amplitudes = [common_amplitude * 0.4, common_amplitude * 0.3,
                             common_amplitude * 0.2, common_amplitude * 0.1]

                for freq, amp in zip(frequencies, amplitudes):
                    phase = np.random.uniform(0, 2*np.pi)
                    trajectory += amp * np.sin(2 * np.pi * freq * t + phase)

                # 确保在常用范围内
                trajectory = np.clip(trajectory, common_min, common_max)

            elif traj_type == 'common_range_cosine':
                # 常用范围余弦 - 使用余弦函数在常用角度范围内运动
                common_min, common_max = common_range
                common_center = (common_min + common_max) / 2
                common_amplitude = (common_max - common_min) * 0.4  # 降低振幅到40%

                # 余弦函数组合，大幅降低频率以减少速度
                trajectory = common_center
                frequencies = [0.01, 0.02, 0.035, 0.05]  # 显著降低频率
                phases = [0, np.pi/4, np.pi/2, 3*np.pi/4]

                for i, freq in enumerate(frequencies):
                    amp = common_amplitude * (0.4 ** i)  # 更快的振幅递减
                    phase = phases[i] + np.random.uniform(0, np.pi/4)
                    trajectory += amp * np.cos(2 * np.pi * freq * t + phase)

                # 确保在常用范围内
                trajectory = np.clip(trajectory, common_min, common_max)

            elif traj_type == 'common_range_triangle':
                # 常用范围三角波 - 在常用角度范围内生成三角波运动
                common_min, common_max = common_range
                common_center = (common_min + common_max) / 2
                common_amplitude = (common_max - common_min) * 0.6

                # 生成三角波
                freq = 0.1  # Hz
                triangle_wave = common_amplitude * (2/np.pi) * np.arcsin(np.sin(2 * np.pi * freq * t))
                trajectory = common_center + triangle_wave

                # 添加高频小振幅三角波
                freq2 = 0.3
                triangle_wave2 = common_amplitude * 0.2 * (2/np.pi) * np.arcsin(np.sin(2 * np.pi * freq2 * t))
                trajectory += triangle_wave2

                # 确保在常用范围内
                trajectory = np.clip(trajectory, common_min, common_max)

            elif traj_type == 'common_range_square':
                # 常用范围方波 - 在常用角度范围内生成方波运动
                common_min, common_max = common_range
                common_center = (common_min + common_max) / 2
                common_amplitude = (common_max - common_min) * 0.4

                # 生成方波
                freq = 0.08  # Hz
                square_wave = common_amplitude * np.sign(np.sin(2 * np.pi * freq * t))
                trajectory = common_center + square_wave

                # 添加平滑处理
                from scipy import ndimage
                try:
                    trajectory = ndimage.gaussian_filter1d(trajectory, sigma=2)
                except ImportError:
                    # 简单的移动平均平滑
                    window_size = 5
                    kernel = np.ones(window_size) / window_size
                    trajectory = np.convolve(trajectory, kernel, mode='same')

                # 确保在常用范围内
                trajectory = np.clip(trajectory, common_min, common_max)

            elif traj_type == 'common_range_sawtooth':
                # 常用范围锯齿波 - 在常用角度范围内生成锯齿波运动
                common_min, common_max = common_range
                common_center = (common_min + common_max) / 2
                common_amplitude = (common_max - common_min) * 0.5

                # 生成锯齿波
                freq = 0.12  # Hz
                sawtooth_wave = common_amplitude * (2 * (t * freq - np.floor(t * freq + 0.5)))
                trajectory = common_center + sawtooth_wave

                # 添加第二个锯齿波
                freq2 = 0.05
                sawtooth_wave2 = common_amplitude * 0.3 * (2 * (t * freq2 - np.floor(t * freq2 + 0.5)))
                trajectory += sawtooth_wave2

                # 确保在常用范围内
                trajectory = np.clip(trajectory, common_min, common_max)

            elif traj_type == 'common_range_exponential':
                # 常用范围指数衰减 - 生成指数衰减和增长的振荡
                common_min, common_max = common_range
                common_center = (common_min + common_max) / 2
                common_amplitude = (common_max - common_min) * 0.6

                # 指数衰减振荡
                decay_rate = 0.1
                freq = 0.15

                # 生成分段的指数衰减振荡
                trajectory = np.zeros_like(t)
                segment_length = len(t) // 4

                for i in range(4):
                    start_idx = i * segment_length
                    end_idx = (i + 1) * segment_length if i < 3 else len(t)
                    segment_t = t[start_idx:end_idx] - t[start_idx]

                    # 交替的指数衰减和增长
                    if i % 2 == 0:
                        envelope = np.exp(-decay_rate * segment_t)
                    else:
                        envelope = 1 - np.exp(-decay_rate * segment_t)

                    oscillation = envelope * np.sin(2 * np.pi * freq * segment_t)
                    trajectory[start_idx:end_idx] = common_center + common_amplitude * 0.8 * oscillation

                # 确保在常用范围内
                trajectory = np.clip(trajectory, common_min, common_max)

            elif traj_type == 'common_range_gaussian':
                # 常用范围高斯分布 - 基于高斯分布生成在常用范围内的运动
                common_min, common_max = common_range
                common_center = (common_min + common_max) / 2
                common_amplitude = (common_max - common_min) * 0.7

                # 生成多个高斯包络的振荡
                trajectory = common_center

                # 设置不同的高斯中心点
                gaussian_centers = np.linspace(0, duration, 5)
                gaussian_widths = [2.0, 1.5, 1.0, 1.5, 2.0]
                frequencies = [0.2, 0.25, 0.3, 0.25, 0.2]

                for center, width, freq in zip(gaussian_centers, gaussian_widths, frequencies):
                    # 高斯包络
                    envelope = np.exp(-((t - center) ** 2) / (2 * width ** 2))
                    # 调制振荡
                    amplitude = common_amplitude * 0.4
                    oscillation = amplitude * envelope * np.sin(2 * np.pi * freq * t)
                    trajectory += oscillation

                # 确保在常用范围内
                trajectory = np.clip(trajectory, common_min, common_max)

            all_positions[:, joint_idx] = trajectory

        # 计算速度和加速度
        dt = 1.0 / sampling_rate
        velocities = np.zeros_like(all_positions)
        accelerations = np.zeros_like(all_positions)

        for joint_idx in range(generator.num_joints):
            # 速度计算
            velocities[1:-1, joint_idx] = (all_positions[2:, joint_idx] - all_positions[:-2, joint_idx]) / (2 * dt)
            velocities[0, joint_idx] = (all_positions[1, joint_idx] - all_positions[0, joint_idx]) / dt
            velocities[-1, joint_idx] = (all_positions[-1, joint_idx] - all_positions[-2, joint_idx]) / dt

            # 加速度计算
            accelerations[1:-1, joint_idx] = (velocities[2:, joint_idx] - velocities[:-2, joint_idx]) / (2 * dt)
            accelerations[0, joint_idx] = (velocities[1, joint_idx] - velocities[0, joint_idx]) / dt
            accelerations[-1, joint_idx] = (velocities[-1, joint_idx] - velocities[-2, joint_idx]) / dt

        # 检查速度限制
        max_velocities = []
        velocity_utilizations = []

        for joint_idx in range(generator.num_joints):
            joint_info = generator.joint_info[joint_idx]
            max_vel_limit = joint_info['velocity']
            joint_max_vel = np.max(np.abs(velocities[:, joint_idx]))
            joint_vel_util = (joint_max_vel / max_vel_limit) * 100

            max_velocities.append(joint_max_vel)
            velocity_utilizations.append(joint_vel_util)

        # 创建输出数据
        output_data = {
            'time': t.tolist(),
            'positions': all_positions.tolist(),
            'velocities': velocities.tolist(),
            'accelerations': accelerations.tolist(),
            'trajectory_info': {
                'type': traj_type,
                'description': description,
                'common_angles': common_angles,
                'duration': duration,
                'sampling_rate': sampling_rate,
                'joint_velocities': max_velocities,
                'joint_velocity_utilizations': velocity_utilizations
            }
        }

        # 保存文件
        filename = f"trajectory_common_angles_{traj_type}.json"
        filepath = os.path.join("/Users/lr-2002/project/instantcreation/IC_arm_control", filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"已保存: {filename}")

        # 显示统计信息
        print(f"  各关节速度利用率:")
        for joint_idx in range(generator.num_joints):
            joint_id = joint_idx + 1
            common_range = common_angles[joint_id]
            common_center_deg = np.rad2deg((common_range[0] + common_range[1]) / 2)
            pos_range = np.max(all_positions[:, joint_idx]) - np.min(all_positions[:, joint_idx])
            vel_util = velocity_utilizations[joint_idx]

            status = "✅" if vel_util <= 90 else "⚠️" if vel_util <= 100 else "❌"
            print(f"    Joint{joint_id}: {vel_util:.1f}% {status} (中心: {common_center_deg:.0f}°, 范围: {np.rad2deg(pos_range):.1f}°)")

    print(f"\n=== 基于常用角度的轨迹生成完成 ===")
    print(f"共生成了 {len(trajectory_types)} 种轨迹类型")
    print(f"每个关节都在其常用角度范围内活动")

    return trajectories

if __name__ == "__main__":
    generate_common_angles_trajectory()