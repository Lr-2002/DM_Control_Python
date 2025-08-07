#!/usr/bin/env python3
"""
动力学辨识轨迹生成器
为IC ARM生成用于动力学参数辨识的激励轨迹
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from typing import List, Tuple, Dict

class TrajectoryGenerator:
    def __init__(self, num_motors=5):
        """
        初始化轨迹生成器
        
        Args:
            num_motors: 电机数量
        """
        self.num_motors = num_motors
        self.motor_names = [f"m{i+1}" for i in range(num_motors)]
        
    def generate_single_motor_trajectory(self, motor_id: int, amplitude: float = 90.0, 
                                       duration: float = 10.0, num_cycles: int = 2) -> Dict:
        """
        生成单个电机的平滑正反转轨迹
        
        Args:
            motor_id: 电机ID (1-5)
            amplitude: 转动幅度 (度)
            duration: 轨迹持续时间 (秒)
            num_cycles: 正反转循环次数
            
        Returns:
            包含时间、位置、速度、加速度的字典
        """
        dt = 0.001  # 10ms采样间隔
        t = np.arange(0, duration, dt)
        
        amplitude_rad = np.radians(amplitude)
        
        # 使用单一的平滑函数确保连续性
        # 采用修正的正弦波加上平滑的包络
        
        # 使用修正的正弦函数: sin²(πt/T) 来确保初始和结束速度都为0
        # 这个函数在 t=0 和 t=T 时位置、速度都为0
        
        # 对于多个循环，使用复合函数
        if num_cycles == 1:
            # 单循环: 使用 sin² 函数
            phase = np.pi * t / duration
            position = amplitude_rad * np.sin(phase)**2
            velocity = amplitude_rad * np.pi / duration * np.sin(2 * phase)
            acceleration = 2 * amplitude_rad * (np.pi / duration)**2 * np.cos(2 * phase)
        else:
            # 多循环: 使用包络调制的正弦波
            # 外层包络: sin²(πt/T) 确保边界条件
            # 内层振荡: sin(2π*num_cycles*t/T) 实现多循环
            
            envelope_phase = np.pi * t / duration
            envelope = np.sin(envelope_phase)**2
            
            oscillation_phase = 2 * np.pi * num_cycles * t / duration
            oscillation = np.sin(oscillation_phase)
            
            # 组合位置
            position = amplitude_rad * envelope * oscillation
            
            # 解析求导得到速度和加速度
            envelope_dot = 2 * np.pi / duration * np.sin(envelope_phase) * np.cos(envelope_phase)
            oscillation_dot = 2 * np.pi * num_cycles / duration * np.cos(oscillation_phase)
            
            velocity = amplitude_rad * (envelope_dot * oscillation + envelope * oscillation_dot)
            
            # 加速度(二阶导数)
            envelope_ddot = 2 * (np.pi / duration)**2 * (np.cos(envelope_phase)**2 - np.sin(envelope_phase)**2)
            oscillation_ddot = -(2 * np.pi * num_cycles / duration)**2 * np.sin(oscillation_phase)
            
            acceleration = amplitude_rad * (
                envelope_ddot * oscillation + 
                2 * envelope_dot * oscillation_dot + 
                envelope * oscillation_ddot
            )
        
        # 创建所有电机的位置数组
        all_positions = np.zeros((len(t), self.num_motors))
        all_velocities = np.zeros((len(t), self.num_motors))
        all_accelerations = np.zeros((len(t), self.num_motors))
        
        # 只有指定电机运动，其他保持零位
        motor_idx = motor_id - 1  # 转换为0-based索引
        all_positions[:, motor_idx] = position
        all_velocities[:, motor_idx] = velocity
        all_accelerations[:, motor_idx] = acceleration
        
        return {
            'time': t,
            'positions': all_positions,
            'velocities': all_velocities,
            'accelerations': all_accelerations,
            'motor_id': motor_id,
            'amplitude_deg': amplitude,
            'num_cycles': num_cycles
        }
    
    def generate_sequential_trajectories(self, motor_sequence: List[int] = [5, 4, 3, 2],
                                       amplitude: float = 90.0, 
                                       duration_per_motor: float = 10.0,
                                       rest_duration: float = 2.0) -> Dict:
        """
        生成按序列运动的轨迹 (5-4-3-2)
        
        Args:
            motor_sequence: 电机运动序列
            amplitude: 转动幅度 (度)
            duration_per_motor: 每个电机运动持续时间 (秒)
            rest_duration: 电机间静止时间 (秒)
            
        Returns:
            完整的序列轨迹
        """
        dt = 0.01
        total_duration = len(motor_sequence) * (duration_per_motor + rest_duration)
        t_total = np.arange(0, total_duration, dt)
        
        all_positions = np.zeros((len(t_total), self.num_motors))
        all_velocities = np.zeros((len(t_total), self.num_motors))
        all_accelerations = np.zeros((len(t_total), self.num_motors))
        
        current_time = 0
        
        for motor_id in motor_sequence:
            # 生成单个电机轨迹
            single_traj = self.generate_single_motor_trajectory(
                motor_id, amplitude, duration_per_motor, num_cycles=1
            )
            
            # 计算时间索引
            start_idx = int(current_time / dt)
            end_idx = start_idx + len(single_traj['time'])
            
            # 添加到总轨迹中
            if end_idx <= len(t_total):
                all_positions[start_idx:end_idx] = single_traj['positions']
                all_velocities[start_idx:end_idx] = single_traj['velocities']
                all_accelerations[start_idx:end_idx] = single_traj['accelerations']
            
            # 更新当前时间 (包括静止时间)
            current_time += duration_per_motor + rest_duration
        
        return {
            'time': t_total,
            'positions': all_positions,
            'velocities': all_velocities,
            'accelerations': all_accelerations,
            'motor_sequence': motor_sequence,
            'amplitude_deg': amplitude
        }
    
    def generate_complex_trajectory(self, duration: float = 30.0, 
                                  amplitudes: List[float] = None) -> Dict:
        """
        生成复合激励轨迹 (多个电机同时运动，不同频率)
        
        Args:
            duration: 轨迹持续时间
            amplitudes: 各电机的幅度列表
            
        Returns:
            复合轨迹
        """
        if amplitudes is None:
            amplitudes = [90, 75, 60, 45, 30]  # 不同电机不同幅度
        
        dt = 0.01
        t = np.arange(0, duration, dt)
        
        all_positions = np.zeros((len(t), self.num_motors))
        all_velocities = np.zeros((len(t), self.num_motors))
        all_accelerations = np.zeros((len(t), self.num_motors))
        
        # 为每个电机分配不同的频率组合
        frequencies = [
            [0.05, 0.15],  # m1: 低频 + 中频
            [0.08, 0.12],  # m2: 
            [0.06, 0.18],  # m3:
            [0.10, 0.20],  # m4:
            [0.04, 0.16]   # m5: 
        ]
        
        for motor_idx in range(self.num_motors):
            if motor_idx < len(amplitudes):
                amp_rad = np.radians(amplitudes[motor_idx])
                
                # 多频率组合
                position = 0
                velocity = 0
                acceleration = 0
                
                for freq in frequencies[motor_idx]:
                    omega = 2 * np.pi * freq
                    # 每个频率分量的权重
                    weight = 1.0 / len(frequencies[motor_idx])
                    
                    position += weight * amp_rad * np.sin(omega * t)
                    velocity += weight * amp_rad * omega * np.cos(omega * t)
                    acceleration += -weight * amp_rad * omega**2 * np.sin(omega * t)
                
                all_positions[:, motor_idx] = position
                all_velocities[:, motor_idx] = velocity
                all_accelerations[:, motor_idx] = acceleration
        
        return {
            'time': t,
            'positions': all_positions,
            'velocities': all_velocities,
            'accelerations': all_accelerations,
            'amplitudes_deg': amplitudes,
            'frequencies': frequencies
        }
    
    def save_trajectory(self, trajectory: Dict, filename: str):
        """保存轨迹到文件"""
        # 转换numpy数组为列表以便JSON序列化
        trajectory_copy = trajectory.copy()
        for key in ['time', 'positions', 'velocities', 'accelerations']:
            if key in trajectory_copy:
                trajectory_copy[key] = trajectory_copy[key].tolist()
        
        with open(filename, 'w') as f:
            json.dump(trajectory_copy, f, indent=2)
        print(f"轨迹已保存到: {filename}")
    
    def plot_trajectory(self, trajectory: Dict, title: str = "Motor Trajectory"):
        """可视化轨迹"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        t = trajectory['time']
        positions = trajectory['positions']
        velocities = trajectory['velocities']
        accelerations = trajectory['accelerations']
        
        # 位置图
        for i in range(self.num_motors):
            axes[0].plot(t, np.degrees(positions[:, i]), label=f'Motor {i+1}')
        axes[0].set_ylabel('Position (deg)')
        axes[0].set_title(f'{title} - Position')
        axes[0].legend()
        axes[0].grid(True)
        
        # 速度图
        for i in range(self.num_motors):
            axes[1].plot(t, np.degrees(velocities[:, i]), label=f'Motor {i+1}')
        axes[1].set_ylabel('Velocity (deg/s)')
        axes[1].set_title(f'{title} - Velocity')
        axes[1].legend()
        axes[1].grid(True)
        
        # 加速度图
        for i in range(self.num_motors):
            axes[2].plot(t, np.degrees(accelerations[:, i]), label=f'Motor {i+1}')
        axes[2].set_ylabel('Acceleration (deg/s²)')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_title(f'{title} - Acceleration')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    """主函数 - 生成各种测试轨迹"""
    generator = TrajectoryGenerator()
    
    print("=== IC ARM 动力学辨识轨迹生成器 ===\n")
    
    # 1. 生成单个电机轨迹 (从5号电机开始)
    print("1. 生成单个电机轨迹...")
    for motor_id in [5, 4, 3, 2]:
        print(f"   生成电机 {motor_id} 的轨迹...")
        single_traj = generator.generate_single_motor_trajectory(
            motor_id=motor_id,
            amplitude=90.0,
            duration=10.0,
            num_cycles=2
        )
        
        # 保存轨迹
        filename = f"trajectory_motor_{motor_id}_single.json"
        generator.save_trajectory(single_traj, filename)
        
        # 可视化第一个轨迹作为示例
        if motor_id == 5:
            generator.plot_trajectory(single_traj, f"Motor {motor_id} Single Trajectory")
    
    # 2. 生成序列轨迹 (5-4-3-2)
    print("\n2. 生成序列轨迹 (5-4-3-2)...")
    sequential_traj = generator.generate_sequential_trajectories(
        motor_sequence=[5, 4, 3, 2],
        amplitude=90.0,
        duration_per_motor=10.0,
        rest_duration=2.0
    )
    
    generator.save_trajectory(sequential_traj, "trajectory_sequential_5432.json")
    generator.plot_trajectory(sequential_traj, "Sequential Trajectory (5-4-3-2)")
    
    # 3. 生成复合轨迹
    print("\n3. 生成复合激励轨迹...")
    complex_traj = generator.generate_complex_trajectory(
        duration=30.0,
        amplitudes=[90, 75, 60, 45, 30]
    )
    
    generator.save_trajectory(complex_traj, "trajectory_complex_multifreq.json")
    generator.plot_trajectory(complex_traj, "Complex Multi-Frequency Trajectory")
    
    print("\n=== 轨迹生成完成 ===")
    print("生成的文件:")
    print("- trajectory_motor_5_single.json")
    print("- trajectory_motor_4_single.json") 
    print("- trajectory_motor_3_single.json")
    print("- trajectory_motor_2_single.json")
    print("- trajectory_sequential_5432.json")
    print("- trajectory_complex_multifreq.json")

if __name__ == "__main__":
    main()
