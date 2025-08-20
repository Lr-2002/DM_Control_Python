#!/usr/bin/env python3
"""
动力学辨识轨迹生成器
为IC ARM生成用于动力学参数辨识的激励轨迹
"""

from ssl import ALERT_DESCRIPTION_DECOMPRESSION_FAILURE
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
									   duration: float = 10.0, num_cycles: int = 2,
									   min_angle: float = None, max_angle: float = None) -> Dict:
		"""
		生成单个电机的平滑正反转轨迹
		
		Args:
			motor_id: 电机ID (1-5)
			amplitude: 转动幅度 (度) - 当min_angle和max_angle未指定时使用
			duration: 轨迹持续时间 (秒)
			num_cycles: 正反转循环次数
			min_angle: 最小角度 (度) - 如果指定，将覆盖amplitude设置
			max_angle: 最大角度 (度) - 如果指定，将覆盖amplitude设置
			
		Returns:
			包含时间、位置、速度、加速度的字典
		"""
		dt = 0.001  # 1ms采样间隔
		t = np.arange(0, duration, dt)
		
		# 处理自定义运动范围
		if min_angle is not None and max_angle is not None:
			min_rad = np.radians(min_angle)
			max_rad = np.radians(max_angle)
			# 从0开始，运动范围是min_angle到max_angle
			start_rad = 0.0
			actual_amplitude_deg = max_angle - min_angle
		else:
			# 使用传统的对称幅度
			amplitude_rad = np.radians(amplitude)
			start_rad = 0.0
			actual_amplitude_deg = amplitude
		
		# 使用单一的平滑函数确保连续性
		# 采用修正的正弦波加上平滑的包络
		
		# 使用修正的正弦函数: sin²(πt/T) 来确保初始和结束速度都为0
		# 这个函数在 t=0 和 t=T 时位置、速度都为0
		
		# 处理自定义运动范围和传统幅度
		if min_angle is not None and max_angle is not None:
			# 自定义范围：从0°到min_angle，然后到max_angle
			# 使用单一连续函数避免分段突变
			
			position = np.zeros_like(t)
			velocity = np.zeros_like(t)
			acceleration = np.zeros_like(t)
			
			# 使用连续的分段五次多项式确保C²连续性
			# 轨迹路径: 0 -> min_angle -> max_angle -> 0
			# 时间分配: 25% -> 50% -> 25%
			t1 = duration * 0.25  # 0 to min_angle
			t2 = duration * 0.50  # min_angle to max_angle  
			t3 = duration * 0.25  # max_angle to 0
			
			for i, time_val in enumerate(t):
				if time_val <= t1:
					# 阶段1: 0 -> min_angle
					tau = time_val / t1  # 归一化时间 [0, 1]
					# 使用五次多项式: 10τ³ - 15τ⁴ + 6τ⁵ (确保C²连续)
					s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
					position[i] = min_rad * s
					# 一阶导数
					s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / t1
					velocity[i] = min_rad * s_dot
					# 二阶导数
					s_ddot = (60 * tau - 180 * tau**2 + 120 * tau**3) / (t1**2)
					acceleration[i] = min_rad * s_ddot
					
				elif time_val <= t1 + t2:
					# 阶段2: min_angle -> max_angle
					local_t = time_val - t1
					tau = local_t / t2  # 归一化时间 [0, 1]
					s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
					position[i] = min_rad + (max_rad - min_rad) * s
					s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / t2
					velocity[i] = (max_rad - min_rad) * s_dot
					s_ddot = (60 * tau - 180 * tau**2 + 120 * tau**3) / (t2**2)
					acceleration[i] = (max_rad - min_rad) * s_ddot
					
				else:
					# 阶段3: max_angle -> 0
					local_t = time_val - t1 - t2
					tau = local_t / t3  # 归一化时间 [0, 1]
					s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
					position[i] = max_rad * (1 - s)  # 从max_rad降到0
					s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / t3
					velocity[i] = -max_rad * s_dot
					s_ddot = (60 * tau - 180 * tau**2 + 120 * tau**3) / (t3**2)
					acceleration[i] = -max_rad * s_ddot
					
		else:
			# 传统的对称幅度轨迹
			if num_cycles == 1:
				# 单循环: 使用 sin² 函数
				phase = np.pi * t / duration
				position = start_rad + amplitude_rad * np.sin(phase)**2
				velocity = amplitude_rad * np.pi / duration * np.sin(2 * phase)
				acceleration = 2 * amplitude_rad * (np.pi / duration)**2 * np.cos(2 * phase)
			else:
				# 多循环: 使用包络调制的正弦波
				envelope_phase = np.pi * t / duration
				envelope = np.sin(envelope_phase)**2
				
				oscillation_phase = 2 * np.pi * num_cycles * t / duration
				oscillation = np.sin(oscillation_phase)
				
				# 组合位置
				position = start_rad + amplitude_rad * envelope * oscillation
				
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
			'amplitude_deg': actual_amplitude_deg,
			'min_angle_deg': min_angle if min_angle is not None else -amplitude,
			'max_angle_deg': max_angle if max_angle is not None else amplitude,
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
								  amplitudes: List[float] = None,
								  motor1_range: List[float] = None) -> Dict:
		"""
		生成复合激励轨迹 (多个电机同时运动，不同频率)
		
		Args:
			duration: 轨迹持续时间
			amplitudes: 各电机的幅度列表 (仅用于电机2-5)
			motor1_range: 电机1的运动范围 [min_angle, max_angle] (度)
			
		Returns:
			复合轨迹
		"""
		if amplitudes is None:
			amplitudes = [90, 75, 60, 45, 30]  # 不同电机不同幅度
		
		if motor1_range is None:
			motor1_range = [-30, 0]  # 电机1默认范围: -30° 到 0°
		
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
			if motor_idx == 0:  # 电机1 (索引0) - 使用自定义范围
				min_angle_rad = np.radians(motor1_range[0])
				max_angle_rad = np.radians(motor1_range[1])
				center_rad = (min_angle_rad + max_angle_rad) / 2
				amplitude_rad = (max_angle_rad - min_angle_rad) / 2
				
				# 多频率组合，但在指定范围内
				position = center_rad  # 从中心位置开始
				velocity = 0
				acceleration = 0
				
				for freq in frequencies[motor_idx]:
					omega = 2 * np.pi * freq
					# 每个频率分量的权重
					weight = 1.0 / len(frequencies[motor_idx])
					
					position += weight * amplitude_rad * np.sin(omega * t)
					velocity += weight * amplitude_rad * omega * np.cos(omega * t)
					acceleration += -weight * amplitude_rad * omega**2 * np.sin(omega * t)
				
				all_positions[:, motor_idx] = position
				all_velocities[:, motor_idx] = velocity
				all_accelerations[:, motor_idx] = acceleration
				
			elif motor_idx < len(amplitudes):  # 电机2-5 - 保持原有逻辑
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
			'motor1_range_deg': motor1_range,
			'frequencies': frequencies
		}
	
	def load_trajectory(self, filename: str) -> Dict:
		"""从文件加载轨迹"""
		with open(filename, 'r') as f:
			trajectory = json.load(f)
		
		# 转换列表为numpy数组
		for key in ['time', 'positions', 'velocities', 'accelerations']:
			if key in trajectory:
				trajectory[key] = np.array(trajectory[key])
		
		return trajectory
	
	def concatenate_trajectories_sequential(self, trajectory_files: List[str], 
										  rest_duration: float = 2.0) -> Dict:
		"""
		拼接轨迹 - 单个轴序列运动 (一个轴动完再动下一个轴)
		
		Args:
			trajectory_files: 轨迹文件路径列表
			rest_duration: 轨迹间的静止时间 (秒)
			
		Returns:
			拼接后的轨迹
		"""
		print(f"开始序列拼接 {len(trajectory_files)} 个轨迹文件...")
		
		all_trajectories = []
		for filename in trajectory_files:
			traj = self.load_trajectory(filename)
			all_trajectories.append(traj)
			print(f"  加载: {filename}")
		
		# 计算总时长和采样参数
		dt = all_trajectories[0]['time'][1] - all_trajectories[0]['time'][0]
		rest_samples = int(rest_duration / dt)
		
		total_samples = 0
		for traj in all_trajectories:
			total_samples += len(traj['time']) + rest_samples
		total_samples -= rest_samples  # 最后一个轨迹后不需要静止时间
		
		# 初始化拼接后的数组
		concatenated_time = np.zeros(total_samples)
		concatenated_positions = np.zeros((total_samples, self.num_motors))
		concatenated_velocities = np.zeros((total_samples, self.num_motors))
		concatenated_accelerations = np.zeros((total_samples, self.num_motors))
		
		current_idx = 0
		current_time = 0.0
		
		for i, traj in enumerate(all_trajectories):
			traj_samples = len(traj['time'])
			
			# 添加轨迹数据
			end_idx = current_idx + traj_samples
			concatenated_time[current_idx:end_idx] = traj['time'] + current_time
			concatenated_positions[current_idx:end_idx] = traj['positions']
			concatenated_velocities[current_idx:end_idx] = traj['velocities']
			concatenated_accelerations[current_idx:end_idx] = traj['accelerations']
			
			current_idx = end_idx
			current_time += traj['time'][-1] + dt
			
			# 添加静止时间 (除了最后一个轨迹)
			if i < len(all_trajectories) - 1:
				rest_end_idx = current_idx + rest_samples
				concatenated_time[current_idx:rest_end_idx] = np.arange(
					current_time, current_time + rest_duration, dt
				)
				# 位置保持最后状态，速度和加速度为0
				concatenated_positions[current_idx:rest_end_idx] = concatenated_positions[current_idx-1]
				# 速度和加速度已经初始化为0
				
				current_idx = rest_end_idx
				current_time += rest_duration
		
		return {
			'time': concatenated_time,
			'positions': concatenated_positions,
			'velocities': concatenated_velocities,
			'accelerations': concatenated_accelerations,
			'source_files': trajectory_files,
			'rest_duration': rest_duration,
			'concatenation_type': 'sequential'
		}
	
	def concatenate_trajectories_simultaneous(self, trajectory_files: List[str]) -> Dict:
		"""
		拼接轨迹 - 多轴同时运动 (所有轴的轨迹直接叠加)
		
		Args:
			trajectory_files: 轨迹文件路径列表
			
		Returns:
			拼接后的轨迹
		"""
		print(f"开始同时拼接 {len(trajectory_files)} 个轨迹文件...")
		
		all_trajectories = []
		for filename in trajectory_files:
			traj = self.load_trajectory(filename)
			all_trajectories.append(traj)
			print(f"  加载: {filename}")
		
		# 找到最长的时间序列
		max_samples = max(len(traj['time']) for traj in all_trajectories)
		reference_time = None
		
		for traj in all_trajectories:
			if len(traj['time']) == max_samples:
				reference_time = traj['time']
				break
		
		# 初始化拼接后的数组
		concatenated_positions = np.zeros((max_samples, self.num_motors))
		concatenated_velocities = np.zeros((max_samples, self.num_motors))
		concatenated_accelerations = np.zeros((max_samples, self.num_motors))
		
		# 叠加所有轨迹
		for traj in all_trajectories:
			traj_samples = len(traj['time'])
			
			# 如果轨迹长度不同，只取较短的部分
			samples_to_use = min(traj_samples, max_samples)
			
			concatenated_positions[:samples_to_use] += traj['positions'][:samples_to_use]
			concatenated_velocities[:samples_to_use] += traj['velocities'][:samples_to_use]
			concatenated_accelerations[:samples_to_use] += traj['accelerations'][:samples_to_use]
		
		return {
			'time': reference_time,
			'positions': concatenated_positions,
			'velocities': concatenated_velocities,
			'accelerations': concatenated_accelerations,
			'source_files': trajectory_files,
			'concatenation_type': 'simultaneous'
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
	
	def plot_trajectory(self, trajectory: Dict, title: str = "Motor Trajectory", save_path: str = None):
		"""可视化轨迹"""
		fig, axes = plt.subplots(3, 1, figsize=(14, 12))
		
		t = trajectory['time']
		positions = trajectory['positions']
		velocities = trajectory['velocities']
		accelerations = trajectory['accelerations']
		
		# 检测有运动的电机
		active_motors = []
		for i in range(self.num_motors):
			if np.max(np.abs(positions[:, i])) > 1e-6:
				active_motors.append(i)
		
		colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
		
		# 位置图
		for i in active_motors:
			axes[0].plot(t, np.degrees(positions[:, i]), 
						label=f'Motor {i+1}', color=colors[i], linewidth=2)
		axes[0].set_ylabel('位置 (°)', fontsize=12)
		axes[0].set_title(f'{title} - 位置轨迹', fontsize=14)
		axes[0].legend()
		axes[0].grid(True, alpha=0.3)
		
		# 速度图
		for i in active_motors:
			axes[1].plot(t, np.degrees(velocities[:, i]), 
						label=f'Motor {i+1}', color=colors[i], linewidth=2)
		axes[1].set_ylabel('速度 (°/s)', fontsize=12)
		axes[1].set_title(f'{title} - 速度轨迹', fontsize=14)
		axes[1].legend()
		axes[1].grid(True, alpha=0.3)
		
		# 加速度图
		for i in active_motors:
			axes[2].plot(t, np.degrees(accelerations[:, i]), 
						label=f'Motor {i+1}', color=colors[i], linewidth=2)
		axes[2].set_xlabel('时间 (s)', fontsize=12)
		axes[2].set_ylabel('加速度 (°/s²)', fontsize=12)
		axes[2].set_title(f'{title} - 加速度轨迹', fontsize=14)
		axes[2].legend()
		axes[2].grid(True, alpha=0.3)
		
		plt.tight_layout()
		
		if save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
			print(f"轨迹图表已保存到: {save_path}")
		
		plt.show()
	
	def plot_motor_detail(self, trajectory: Dict, motor_id: int, title: str = None, save_path: str = None):
		"""
		绘制单个电机的详细分析图
		
		Args:
			trajectory: 轨迹数据字典
			motor_id: 电机ID (1-5)
			title: 图表标题
			save_path: 保存路径（可选）
		"""
		if not (1 <= motor_id <= 5):
			print(f"电机ID必须在1-5之间，当前: {motor_id}")
			return
		
		motor_idx = motor_id - 1
		
		if title is None:
			title = f"电机 {motor_id} 详细轨迹分析"
		
		t = trajectory['time']
		position = np.degrees(trajectory['positions'][:, motor_idx])
		velocity = np.degrees(trajectory['velocities'][:, motor_idx])
		acceleration = np.degrees(trajectory['accelerations'][:, motor_idx])
		
		# 检查是否有运动
		if np.max(np.abs(trajectory['positions'][:, motor_idx])) < 1e-6:
			print(f"警告: 电机 {motor_id} 没有明显运动")
			return
		
		# 创建子图
		fig, axes = plt.subplots(4, 1, figsize=(14, 16))
		fig.suptitle(title, fontsize=16, fontweight='bold')
		
		color = '#2ca02c'  # 绿色
		
		# 位置图
		ax1 = axes[0]
		ax1.plot(t, position, color=color, linewidth=2.5)
		ax1.set_ylabel('位置 (°)', fontsize=12)
		ax1.set_title(f'电机 {motor_id} 位置轨迹', fontsize=14)
		ax1.grid(True, alpha=0.3)
		ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
		
		# 添加统计信息
		pos_stats = f"范围: [{position.min():.1f}°, {position.max():.1f}°], 幅度: {position.max() - position.min():.1f}°"
		ax1.text(0.02, 0.98, pos_stats, transform=ax1.transAxes, 
				verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
		
		# 速度图
		ax2 = axes[1]
		ax2.plot(t, velocity, color=color, linewidth=2.5)
		ax2.set_ylabel('速度 (°/s)', fontsize=12)
		ax2.set_title(f'电机 {motor_id} 速度轨迹', fontsize=14)
		ax2.grid(True, alpha=0.3)
		ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
		
		vel_stats = f"最大速度: {velocity.max():.1f}°/s, 最小速度: {velocity.min():.1f}°/s"
		ax2.text(0.02, 0.98, vel_stats, transform=ax2.transAxes, 
				verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
		
		# 加速度图
		ax3 = axes[2]
		ax3.plot(t, acceleration, color=color, linewidth=2.5)
		ax3.set_ylabel('加速度 (°/s²)', fontsize=12)
		ax3.set_title(f'电机 {motor_id} 加速度轨迹', fontsize=14)
		ax3.grid(True, alpha=0.3)
		ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
		
		acc_stats = f"最大加速度: {acceleration.max():.1f}°/s², 最小加速度: {acceleration.min():.1f}°/s²"
		ax3.text(0.02, 0.98, acc_stats, transform=ax3.transAxes, 
				verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
		
		# 相空间图 (位置 vs 速度)
		ax4 = axes[3]
		ax4.plot(position, velocity, color=color, linewidth=2, alpha=0.7)
		ax4.scatter(position[0], velocity[0], color='green', s=100, marker='o', label='起点', zorder=5)
		ax4.scatter(position[-1], velocity[-1], color='red', s=100, marker='s', label='终点', zorder=5)
		ax4.set_xlabel('位置 (°)', fontsize=12)
		ax4.set_ylabel('速度 (°/s)', fontsize=12)
		ax4.set_title(f'电机 {motor_id} 相空间图 (位置 vs 速度)', fontsize=14)
		ax4.grid(True, alpha=0.3)
		ax4.legend()
		
		# 调整布局
		plt.tight_layout()
		
		# 保存图片
		if save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
			print(f"详细分析图表已保存到: {save_path}")
		
		plt.show()
		
		# 分析连续性
		self.analyze_continuity(trajectory, motor_id)
	
	def analyze_continuity(self, trajectory: Dict, motor_id: int):
		"""
		分析轨迹连续性
		
		Args:
			trajectory: 轨迹数据字典
			motor_id: 电机ID (1-5)
		"""
		if not (1 <= motor_id <= 5):
			print(f"电机ID必须在1-5之间，当前: {motor_id}")
			return
		
		motor_idx = motor_id - 1
		
		t = trajectory['time']
		position = trajectory['positions'][:, motor_idx]
		velocity = trajectory['velocities'][:, motor_idx]
		acceleration = trajectory['accelerations'][:, motor_idx]
		
		print(f"\n=== 电机 {motor_id} 连续性分析 ===")
		
		# 计算数值导数
		dt = np.diff(t)
		numerical_velocity = np.diff(position) / dt
		numerical_acceleration = np.diff(velocity[:-1]) / dt[:-1]
		
		# 比较解析导数和数值导数
		velocity_error = np.abs(velocity[1:] - numerical_velocity)
		acceleration_error = np.abs(acceleration[2:] - numerical_acceleration)
		
		print(f"速度连续性:")
		print(f"  最大误差: {np.degrees(velocity_error.max()):.6f}°/s")
		print(f"  平均误差: {np.degrees(velocity_error.mean()):.6f}°/s")
		print(f"  RMS误差: {np.degrees(np.sqrt(np.mean(velocity_error**2))):.6f}°/s")
		
		print(f"加速度连续性:")
		print(f"  最大误差: {np.degrees(acceleration_error.max()):.6f}°/s²")
		print(f"  平均误差: {np.degrees(acceleration_error.mean()):.6f}°/s²")
		print(f"  RMS误差: {np.degrees(np.sqrt(np.mean(acceleration_error**2))):.6f}°/s²")
		
		# 检查边界条件
		print(f"边界条件:")
		print(f"  起始位置: {np.degrees(position[0]):.6f}°")
		print(f"  起始速度: {np.degrees(velocity[0]):.6f}°/s")
		print(f"  终止位置: {np.degrees(position[-1]):.6f}°")
		print(f"  终止速度: {np.degrees(velocity[-1]):.6f}°/s")

def main():
	"""主函数 - 生成各种测试轨迹"""
	print("=== IC ARM 动力学辨识轨迹生成器 ===\n")
	
	generator = TrajectoryGenerator()
	
	# 1. 生成单个电机轨迹 (包括1号电机)
	print("1. 生成单个电机轨迹...")
	motor_configs = {
		1: {"min_angle": -30.0, "max_angle": -4.0},  # 1号电机: -30° 到 -4°
		2: {"min_angle": -100.0, "max_angle": 30.0},  # 2号电机: -100° 到 +30°
		3: {"amplitude": 90.0},  # 3号电机: 对称±90°
		4: {"amplitude": 90.0},  # 4号电机: 对称±90°
		5: {"amplitude": 90.0}   # 5号电机: 对称±90°
	}
	
	for motor_id in [1, 2, 3, 4, 5]:
		print(f"   生成电机 {motor_id} 的轨迹...")
		config = motor_configs[motor_id]
		
		if "min_angle" in config and "max_angle" in config:
			# 使用自定义范围
			single_traj = generator.generate_single_motor_trajectory(
				motor_id=motor_id,
				duration=10.0,
				num_cycles=2,
				min_angle=config["min_angle"],
				max_angle=config["max_angle"]
			)
		else:
			# 使用对称幅度
			single_traj = generator.generate_single_motor_trajectory(
				motor_id=motor_id,
				amplitude=config["amplitude"],
				duration=10.0,
				num_cycles=2
			)
		
		# 保存轨迹
		filename = f"trajectory_motor_{motor_id}_single.json"
		generator.save_trajectory(single_traj, filename)
		
		# 可视化1号电机轨迹作为示例
		if motor_id == 1:
			generator.plot_trajectory(single_traj, f"Motor {motor_id} Single Trajectory (-30° to -4°)")
	
	# 2. 生成完整序列轨迹 (1-5-4-3-2)
	print("\n2. 生成完整序列轨迹 (1-5-4-3-2)...")
	
	# 方法1: 使用文件拼接
	trajectory_files = [
		"trajectory_motor_1_single.json",
		"trajectory_motor_5_single.json", 
		"trajectory_motor_4_single.json",
		"trajectory_motor_3_single.json",
		"trajectory_motor_2_single.json"
	]
	
	concatenated_traj = generator.concatenate_trajectories_sequential(
		trajectory_files=trajectory_files,
		rest_duration=2.0
	)
	
	generator.save_trajectory(concatenated_traj, "trajectory_concatenated_sequential.json")
	generator.plot_trajectory(concatenated_traj, "Complete Sequential Trajectory (1-5-4-3-2)")
	
	# 方法2: 同时运动轨迹
	print("\n3. 生成同时运动轨迹...")
	simultaneous_traj = generator.concatenate_trajectories_simultaneous(trajectory_files)
	generator.save_trajectory(simultaneous_traj, "trajectory_concatenated_simultaneous.json")
	generator.plot_trajectory(simultaneous_traj, "Simultaneous Multi-Motor Trajectory")
	
	# 4. 生成复合轨迹 (更新电机1范围)
	print("\n4. 生成复合激励轨迹...")
	complex_traj = generator.generate_complex_trajectory(
		duration=30.0,
		amplitudes=[90, 75, 60, 45, 30],  # 电机2-5的幅度
		motor1_range=[-30, -4]  # 电机1在-30°到-4°范围内运动
	)
	
	generator.save_trajectory(complex_traj, "trajectory_complex_multifreq.json")
	generator.plot_trajectory(complex_traj, "Complex Multi-Frequency Trajectory")
	
	print("\n=== 轨迹生成完成 ===")
	print("生成的文件:")
	print("- trajectory_motor_1_single.json  (新增: -30° to -4°)")
	print("- trajectory_motor_2_single.json  (-100° to +30°)")
	print("- trajectory_motor_3_single.json  (±90°)")
	print("- trajectory_motor_4_single.json  (±90°)")
	print("- trajectory_motor_5_single.json  (±90°)")
	print("- trajectory_concatenated_sequential.json  (1-5-4-3-2 序列)")
	print("- trajectory_concatenated_simultaneous.json  (所有电机同时)")
	print("- trajectory_complex_multifreq.json  (多频率复合轨迹)")

if __name__ == "__main__":
	main()
