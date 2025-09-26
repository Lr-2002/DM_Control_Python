#!/usr/bin/env python3
"""
动力学辨识轨迹生成器 (安全增强版)
为IC ARM生成用于动力学参数辨识的激励轨迹
包含关节限制、速度约束和碰撞检测
"""

from ssl import ALERT_DESCRIPTION_DECOMPRESSION_FAILURE
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
import os

class TrajectoryGenerator:
	def __init__(self, urdf_path=None, max_velocity=3.0, joint_limit_buffer=0.2, num_motors=None):
		"""
		初始化轨迹生成器
		
		Args:
			urdf_path: URDF文件路径，用于读取关节限制和自动检测关节数量
			max_velocity: 最大速度限制 (rad/s)
			joint_limit_buffer: 关节限制缓冲区比例 (0-1)
			num_motors: 手动指定电机数量（可选，优先使用URDF自动检测）
		"""
		self.max_velocity = max_velocity
		self.joint_limit_buffer = joint_limit_buffer
		self.joint_limits = {}
		
		# 从URDF加载关节限制并自动检测关节数量
		if urdf_path and os.path.exists(urdf_path):
			detected_joints = self._load_joint_limits_from_urdf(urdf_path)
			self.num_motors = len(detected_joints) if detected_joints else (num_motors or 5)
		else:
			self.num_motors = num_motors or 5
			print(f"⚠ 未提供URDF文件，使用默认电机数量: {self.num_motors}")
		
		print(f"轨迹生成器初始化完成:")
		print(f"  电机数量: {self.num_motors}")
		print(f"  最大速度: {self.max_velocity} rad/s")
		print(f"  关节限制缓冲区: {self.joint_limit_buffer*100:.1f}%")
		
		# 打印关节限制信息
		self._print_joint_limits()
	
	def _load_joint_limits_from_urdf(self, urdf_path):
		"""从URDF文件读取关节限制并返回检测到的revolute关节列表"""
		try:
			tree = ET.parse(urdf_path)
			root = tree.getroot()
			
			detected_joints = []
			
			for joint in root.findall('.//joint'):
				joint_name = joint.get('name')
				joint_type = joint.get('type')
				
				if joint_type == 'revolute':
					limit_elem = joint.find('limit')
					if limit_elem is not None:
						lower = float(limit_elem.get('lower', '-3.14'))
						upper = float(limit_elem.get('upper', '3.14'))
						velocity = float(limit_elem.get('velocity', '3.14'))
						
						# 映射joint名称到motor索引
						if joint_name.startswith('joint'):
							joint_num = int(joint_name.replace('joint', ''))
							detected_joints.append(joint_num)
							
							# 存储关节限制（使用0-based索引）
							self.joint_limits[joint_num-1] = {
								'lower': lower,
								'upper': upper,
								'velocity': min(velocity, self.max_velocity),
								'range': upper - lower,
								'name': joint_name
							}
			
			detected_joints.sort()  # 确保顺序
			print(f"✓ 从URDF检测到 {len(detected_joints)} 个revolute关节: {detected_joints}")
			print(f"✓ 加载了 {len(self.joint_limits)} 个关节限制")
			
			return detected_joints
			
		except Exception as e:
			print(f"✗ 读取URDF失败: {e}")
			self._set_default_joint_limits()
			return None
	
	def _set_default_joint_limits(self):
		"""设置默认关节限制"""
		default_limits = [
			{'lower': -1.23, 'upper': 0.17, 'velocity': 3.0},  # joint1
			{'lower': -0.18, 'upper': 3.47, 'velocity': 3.0},  # joint2  
			{'lower': -0.65, 'upper': 3.49, 'velocity': 3.0},  # joint3
			{'lower': -3.46, 'upper': 1.97, 'velocity': 3.0},  # joint4
			{'lower': -1.69, 'upper': 1.80, 'velocity': 3.0},  # joint5
		]
		
		for i, limits in enumerate(default_limits):
			if i < self.num_motors:
				self.joint_limits[i] = {
					'lower': limits['lower'],
					'upper': limits['upper'], 
					'velocity': min(limits['velocity'], self.max_velocity),
					'range': limits['upper'] - limits['lower']
				}
		
		print(f"✓ 使用默认关节限制")
	
	def _print_joint_limits(self):
		"""打印关节限制信息"""
		print(f"\n=== 关节限制信息 ===")
		print(f"{'关节':<8} {'下限(rad)':<12} {'上限(rad)':<12} {'范围(rad)':<12} {'最大速度(rad/s)':<15}")
		print("-" * 70)
		
		for i in range(self.num_motors):
			if i in self.joint_limits:
				limits = self.joint_limits[i]
				print(f"joint{i+1:<3} {limits['lower']:<12.3f} {limits['upper']:<12.3f} "
					  f"{limits['range']:<12.3f} {limits['velocity']:<15.3f}")
			else:
				print(f"joint{i+1:<3} {'未定义':<12} {'未定义':<12} {'未定义':<12} {'未定义':<15}")
	
	def _apply_velocity_ramping(self, position, velocity, joint_idx):
		"""
		在关节限制边界附近应用速度渐变
		
		Args:
			position: 位置数组
			velocity: 速度数组  
			joint_idx: 关节索引 (0-based)
			
		Returns:
			调整后的速度数组
		"""
		if joint_idx not in self.joint_limits:
			return velocity
		
		limits = self.joint_limits[joint_idx]
		lower_limit = limits['lower']
		upper_limit = limits['upper']
		joint_range = limits['range']
		
		# 计算渐变区域
		ramp_zone = joint_range * self.joint_limit_buffer
		lower_ramp_start = lower_limit + ramp_zone
		upper_ramp_start = upper_limit - ramp_zone
		
		ramped_velocity = velocity.copy()
		
		for i, pos in enumerate(position):
			ramp_factor = 1.0
			
			# 接近下限时减速
			if pos < lower_ramp_start:
				if pos <= lower_limit:
					ramp_factor = 0.1  # 最小速度
				else:
					# 线性渐变
					ramp_factor = 0.1 + 0.9 * (pos - lower_limit) / ramp_zone
			
			# 接近上限时减速  
			elif pos > upper_ramp_start:
				if pos >= upper_limit:
					ramp_factor = 0.1  # 最小速度
				else:
					# 线性渐变
					ramp_factor = 0.1 + 0.9 * (upper_limit - pos) / ramp_zone
			
			ramped_velocity[i] *= ramp_factor
		
		return ramped_velocity
	
	def _constrain_to_joint_limits(self, position, joint_idx):
		"""
		将位置约束到关节限制范围内
		
		Args:
			position: 位置数组
			joint_idx: 关节索引 (0-based)
			
		Returns:
			约束后的位置数组
		"""
		if joint_idx not in self.joint_limits:
			return position
		
		limits = self.joint_limits[joint_idx]
		lower_limit = limits['lower']
		upper_limit = limits['upper']
		
		# 应用缓冲区
		buffer_size = limits['range'] * self.joint_limit_buffer
		safe_lower = lower_limit + buffer_size
		safe_upper = upper_limit - buffer_size
		
		return np.clip(position, safe_lower, safe_upper)
	
	def _limit_velocity(self, velocity, joint_idx):
		"""
		限制关节速度
		
		Args:
			velocity: 速度数组
			joint_idx: 关节索引 (0-based)
			
		Returns:
			限制后的速度数组
		"""
		if joint_idx not in self.joint_limits:
			max_vel = self.max_velocity
		else:
			max_vel = self.joint_limits[joint_idx]['velocity']
		
		return np.clip(velocity, -max_vel, max_vel)
	
	def create_mujoco_scene_with_obstacle(self, urdf_path):
		"""
		创建包含50mm圆柱体障碍物的MuJoCo场景XML
		
		Args:
			urdf_path: 机器人URDF文件路径
			
		Returns:
			MuJoCo XML场景字符串
		"""
		xml_content = f'''
<mujoco model="robot_with_obstacle">
	<compiler angle="radian" meshdir="meshes/"/>
	
	<option timestep="0.002" iterations="50" solver="Newton" jacobian="dense"/>
	
	<worldbody>
		<!-- 地面 -->
		<geom name="floor" pos="0 0 -0.1" size="2 2 0.05" type="box" rgba="0.8 0.8 0.8 1"/>
		
		<!-- 50mm直径圆柱体障碍物 (从base往下-y方向) -->
		<body name="obstacle" pos="0 -0.15 0">
			<geom name="cylinder_obstacle" 
				  type="cylinder" 
				  size="0.025 0.5" 
				  rgba="1 0 0 0.7"
				  pos="0 0 0"/>
		</body>
		
		<!-- 机器人 -->
		<body name="robot_base" pos="0 0 0">
			<include file="{urdf_path}"/>
		</body>
	</worldbody>
	
	<contact>
		<exclude body1="robot_base" body2="obstacle"/>
	</contact>
</mujoco>
'''
		return xml_content
	
	def validate_trajectory_safety(self, trajectory, check_collisions=True):
		"""
		验证轨迹安全性
		
		Args:
			trajectory: 轨迹字典
			check_collisions: 是否检查碰撞
			
		Returns:
			验证结果字典
		"""
		validation_results = {
			'is_safe': True,
			'violations': [],
			'warnings': [],
			'statistics': {}
		}
		
		positions = trajectory['positions']
		velocities = trajectory['velocities']
		accelerations = trajectory['accelerations']
		
		# 1. 检查关节限制
		num_trajectory_motors = positions.shape[1]
		for motor_idx in range(min(self.num_motors, num_trajectory_motors)):
			if motor_idx in self.joint_limits:
				limits = self.joint_limits[motor_idx]
				motor_positions = positions[:, motor_idx]
				motor_velocities = velocities[:, motor_idx]
				
				# 位置限制检查
				pos_violations = np.where((motor_positions < limits['lower']) | 
										(motor_positions > limits['upper']))[0]
				if len(pos_violations) > 0:
					validation_results['is_safe'] = False
					validation_results['violations'].append({
						'type': 'position_limit',
						'motor': motor_idx + 1,
						'count': len(pos_violations),
						'first_violation_time': trajectory['time'][pos_violations[0]]
					})
				
				# 速度限制检查
				max_vel = limits['velocity']
				vel_violations = np.where(np.abs(motor_velocities) > max_vel)[0]
				if len(vel_violations) > 0:
					validation_results['is_safe'] = False
					validation_results['violations'].append({
						'type': 'velocity_limit',
						'motor': motor_idx + 1,
						'count': len(vel_violations),
						'max_velocity': np.max(np.abs(motor_velocities)),
						'limit': max_vel
					})
				
				# 统计信息
				motor_accelerations = accelerations[:, motor_idx] if motor_idx < accelerations.shape[1] else np.zeros_like(motor_positions)
				validation_results['statistics'][f'motor_{motor_idx+1}'] = {
					'pos_range': [np.min(motor_positions), np.max(motor_positions)],
					'vel_range': [np.min(motor_velocities), np.max(motor_velocities)],
					'max_abs_vel': np.max(np.abs(motor_velocities)),
					'max_abs_acc': np.max(np.abs(motor_accelerations))
				}
		
		# 2. 碰撞检查 (如果启用)
		if check_collisions:
			collision_count = self._check_trajectory_collisions(trajectory)
			if collision_count > 0:
				validation_results['is_safe'] = False
				validation_results['violations'].append({
					'type': 'collision',
					'count': collision_count
				})
		
		return validation_results
	
	def _generate_motor_configs_from_urdf(self):
		"""
		从URDF文件自动生成电机配置
		
		Returns:
			dict: 电机配置字典
		"""
		motor_configs = {}
		
		for joint_id in range(1, self.num_motors + 1):
			joint_name = f"joint{joint_id}"
			if joint_name in self.joint_limits:
				limits = self.joint_limits[joint_name]
				min_angle_deg = np.degrees(limits['lower'])
				max_angle_deg = np.degrees(limits['upper'])
				
				# 计算安全范围 (应用缓冲区)
				range_deg = max_angle_deg - min_angle_deg
				buffer_deg = range_deg * self.joint_limit_buffer
				safe_min = min_angle_deg + buffer_deg
				safe_max = max_angle_deg - buffer_deg
				
				# 根据关节范围选择配置方式
				if range_deg < 90:  # 小范围关节使用min/max配置
					motor_configs[joint_id] = {
						"min_angle": safe_min,
						"max_angle": safe_max
					}
				else:  # 大范围关节使用对称幅度配置
					center_deg = (safe_min + safe_max) / 2
					amplitude_deg = min(abs(safe_max - center_deg), abs(center_deg - safe_min))
					motor_configs[joint_id] = {
						"amplitude": amplitude_deg
					}
			else:
				# 默认配置
				motor_configs[joint_id] = {"amplitude": 30.0}
		
		print(f"\n=== 从URDF自动生成的电机配置 ===")
		for motor_id, config in motor_configs.items():
			if "min_angle" in config:
				print(f"电机{motor_id}: 范围 [{config['min_angle']:.1f}°, {config['max_angle']:.1f}°]")
			else:
				print(f"电机{motor_id}: 对称幅度 ±{config['amplitude']:.1f}°")
		print()
		
		return motor_configs
	
	def _check_trajectory_collisions(self, trajectory):
		"""
		使用简化的几何检查来检测与圆柱体障碍物的碰撞
		
		Args:
			trajectory: 轨迹字典
			
		Returns:
			碰撞次数
		"""
		# 简化的碰撞检查 - 基于关节角度的几何估算
		# 这里使用简化模型，实际应用中需要完整的运动学计算
		
		positions = trajectory['positions']
		collision_count = 0
		
		# 障碍物位置: (0, -0.15, 0), 半径: 0.025m
		obstacle_pos = np.array([0, -0.15, 0])
		obstacle_radius = 0.025
		
		# 简化检查：如果关节2角度过大，可能导致link2接近障碍物
		joint2_positions = positions[:, 1]  # joint2 (motor2)
		
		# 当joint2角度 > 2.0 rad时，可能有碰撞风险
		risky_positions = np.where(joint2_positions > 2.0)[0]
		collision_count = len(risky_positions)
		
		return collision_count
	
	def apply_safety_constraints_to_trajectory(self, trajectory):
		"""
		对现有轨迹应用安全约束
		
		Args:
			trajectory: 输入轨迹字典
			
		Returns:
			安全约束后的轨迹字典
		"""
		safe_trajectory = trajectory.copy()
		positions = trajectory['positions'].copy()
		velocities = trajectory['velocities'].copy()
		
		# 对每个关节应用约束
		for motor_idx in range(self.num_motors):
			# 1. 约束位置
			positions[:, motor_idx] = self._constrain_to_joint_limits(
				positions[:, motor_idx], motor_idx)
			
			# 2. 限制速度
			velocities[:, motor_idx] = self._limit_velocity(
				velocities[:, motor_idx], motor_idx)
			
			# 3. 应用速度渐变
			velocities[:, motor_idx] = self._apply_velocity_ramping(
				positions[:, motor_idx], velocities[:, motor_idx], motor_idx)
		
		# 4. 重新计算加速度
		dt = trajectory['time'][1] - trajectory['time'][0]
		accelerations = np.zeros_like(velocities)
		for motor_idx in range(self.num_motors):
			accelerations[:, motor_idx] = np.gradient(velocities[:, motor_idx], dt)
		
		safe_trajectory['positions'] = positions
		safe_trajectory['velocities'] = velocities
		safe_trajectory['accelerations'] = accelerations
		safe_trajectory['safety_applied'] = True
		
		return safe_trajectory
		
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
		
		# 应用安全约束
		motor_idx = motor_id - 1  # 转换为0-based索引
		
		# 1. 约束位置到关节限制范围内
		position = self._constrain_to_joint_limits(position, motor_idx)
		
		# 2. 限制速度
		velocity = self._limit_velocity(velocity, motor_idx)
		
		# 3. 在关节限制边界附近应用速度渐变
		velocity = self._apply_velocity_ramping(position, velocity, motor_idx)
		
		# 4. 重新计算加速度（基于调整后的速度）
		dt_val = t[1] - t[0]
		acceleration = np.gradient(velocity, dt_val)
		
		# 创建所有电机的位置数组
		all_positions = np.zeros((len(t), self.num_motors))
		all_velocities = np.zeros((len(t), self.num_motors))
		all_accelerations = np.zeros((len(t), self.num_motors))
		
		# 只有指定电机运动，其他保持零位
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
	
	def generate_chirp_trajectory(self, duration: float = 20.0, 
								 motor_configs: Dict = None,
								 freq_range: List[float] = [0.05, 0.5]) -> Dict:
		"""
		生成扫频轨迹 (频率从低到高线性变化)
		
		Args:
			duration: 轨迹持续时间
			motor_configs: 电机配置字典，包含每个电机的运动范围或幅度
			freq_range: 频率范围 [start_freq, end_freq] (Hz)
			
		Returns:
			扫频轨迹
		"""
		if motor_configs is None:
			motor_configs = {
				1: {"min_angle": -30.0, "max_angle": 4.0},
				2: {"min_angle": -120.0, "max_angle": 40.0},
				3: {"amplitude": 100.0},
				4: {"amplitude": 100.0},
				5: {"amplitude": 100.0}
			}
		
		dt = 0.01
		t = np.arange(0, duration, dt)
		
		all_positions = np.zeros((len(t), self.num_motors))
		all_velocities = np.zeros((len(t), self.num_motors))
		all_accelerations = np.zeros((len(t), self.num_motors))
		
		# 线性扫频参数
		f0, f1 = freq_range
		k = (f1 - f0) / duration  # 频率变化率
		
		for motor_idx in range(self.num_motors):
			motor_id = motor_idx + 1
			config = motor_configs.get(motor_id, {})
			
			# 每个电机使用不同的扫频范围
			motor_f0 = f0 + motor_idx * 0.02
			motor_f1 = f1 + motor_idx * 0.05
			motor_k = (motor_f1 - motor_f0) / duration
			
			phase = 2 * np.pi * (motor_f0 * t + motor_k * t**2 / 2)
			inst_freq = motor_f0 + motor_k * t
			
			if "min_angle" in config and "max_angle" in config:
				# 电机1和2: 使用指定范围
				min_angle_rad = np.radians(config["min_angle"])
				max_angle_rad = np.radians(config["max_angle"])
				center_rad = (min_angle_rad + max_angle_rad) / 2
				amplitude_rad = (max_angle_rad - min_angle_rad) / 2
				
				position = center_rad + amplitude_rad * np.sin(phase)
				velocity = amplitude_rad * 2 * np.pi * inst_freq * np.cos(phase)
				acceleration = (amplitude_rad * 2 * np.pi * 
							   (motor_k * np.cos(phase) - (2 * np.pi * inst_freq)**2 * np.sin(phase)))
				
			elif "amplitude" in config:
				# 电机3-5: 使用对称幅度
				amp_rad = np.radians(config["amplitude"])
				
				position = amp_rad * np.sin(phase)
				velocity = amp_rad * 2 * np.pi * inst_freq * np.cos(phase)
				acceleration = (amp_rad * 2 * np.pi * 
							   (motor_k * np.cos(phase) - (2 * np.pi * inst_freq)**2 * np.sin(phase)))
			
			else:
				# 默认配置
				position = velocity = acceleration = np.zeros_like(t)
			
			all_positions[:, motor_idx] = position
			all_velocities[:, motor_idx] = velocity
			all_accelerations[:, motor_idx] = acceleration
		
		return {
			'time': t,
			'positions': all_positions,
			'velocities': all_velocities,
			'accelerations': all_accelerations,
			'motor_configs': motor_configs,
			'freq_range': freq_range,
			'trajectory_type': 'chirp'
		}
	
	def generate_random_trajectory(self, duration: float = 25.0,
								  motor_configs: Dict = None,
								  bandwidth: float = 0.3,
								  seed: int = 42) -> Dict:
		"""
		生成随机轨迹 (带限白噪声)
		
		Args:
			duration: 轨迹持续时间
			motor_configs: 电机配置字典，包含每个电机的运动范围或幅度
			bandwidth: 带宽限制 (Hz)
			seed: 随机种子
			
		Returns:
			随机轨迹
		"""
		np.random.seed(seed)
		
		if motor_configs is None:
			motor_configs = {
				1: {"min_angle": -30.0, "max_angle": 4.0},
				2: {"min_angle": -120.0, "max_angle": 40.0},
				3: {"amplitude": 100.0},
				4: {"amplitude": 100.0},
				5: {"amplitude": 100.0}
			}
		
		dt = 0.01
		t = np.arange(0, duration, dt)
		N = len(t)
		
		all_positions = np.zeros((N, self.num_motors))
		all_velocities = np.zeros((N, self.num_motors))
		all_accelerations = np.zeros((N, self.num_motors))
		
		# 频域滤波生成带限随机信号
		freqs = np.fft.fftfreq(N, dt)
		
		for motor_idx in range(self.num_motors):
			motor_id = motor_idx + 1
			config = motor_configs.get(motor_id, {})
			
			# 生成白噪声
			noise = np.random.randn(N)
			noise_fft = np.fft.fft(noise)
			
			# 带限滤波器
			filter_mask = np.abs(freqs) <= bandwidth
			filtered_fft = noise_fft * filter_mask
			
			# 逆变换得到时域信号
			filtered_signal = np.real(np.fft.ifft(filtered_fft))
			normalized_signal = filtered_signal / np.std(filtered_signal)
			
			if "min_angle" in config and "max_angle" in config:
				# 电机1和2: 使用指定范围
				min_angle_rad = np.radians(config["min_angle"])
				max_angle_rad = np.radians(config["max_angle"])
				center_rad = (min_angle_rad + max_angle_rad) / 2
				amplitude_rad = (max_angle_rad - min_angle_rad) / 2
				
				position = center_rad + amplitude_rad * 0.5 * normalized_signal
				
			elif "amplitude" in config:
				# 电机3-5: 使用对称幅度
				amp_rad = np.radians(config["amplitude"])
				position = amp_rad * 0.5 * normalized_signal
			
			else:
				# 默认配置
				position = np.zeros_like(t)
			
			# 数值求导得到速度和加速度
			velocity = np.gradient(position, dt)
			acceleration = np.gradient(velocity, dt)
			
			all_positions[:, motor_idx] = position
			all_velocities[:, motor_idx] = velocity
			all_accelerations[:, motor_idx] = acceleration
		
		return {
			'time': t,
			'positions': all_positions,
			'velocities': all_velocities,
			'accelerations': all_accelerations,
			'motor_configs': motor_configs,
			'bandwidth': bandwidth,
			'seed': seed,
			'trajectory_type': 'random'
		}
	
	def generate_phase_shifted_trajectory(self, duration: float = 20.0,
										 motor_configs: Dict = None,
										 base_freq: float = 0.1) -> Dict:
		"""
		生成相位偏移轨迹 (各电机同频率但不同相位)
		
		Args:
			duration: 轨迹持续时间
			motor_configs: 电机配置字典，包含每个电机的运动范围或幅度
			base_freq: 基础频率 (Hz)
			
		Returns:
			相位偏移轨迹
		"""
		if motor_configs is None:
			motor_configs = {
				1: {"min_angle": -30.0, "max_angle": 4.0},
				2: {"min_angle": -120.0, "max_angle": 40.0},
				3: {"amplitude": 100.0},
				4: {"amplitude": 100.0},
				5: {"amplitude": 100.0}
			}
		
		dt = 0.01
		t = np.arange(0, duration, dt)
		
		all_positions = np.zeros((len(t), self.num_motors))
		all_velocities = np.zeros((len(t), self.num_motors))
		all_accelerations = np.zeros((len(t), self.num_motors))
		
		# 相位偏移 (根据实际电机数量动态计算)
		phase_shifts = [i * 2 * np.pi / self.num_motors for i in range(self.num_motors)]
		
		omega = 2 * np.pi * base_freq
		
		for motor_idx in range(self.num_motors):
			motor_id = motor_idx + 1
			config = motor_configs.get(motor_id, {})
			phase = omega * t + phase_shifts[motor_idx]
			
			if "min_angle" in config and "max_angle" in config:
				# 电机1和2: 使用指定范围
				min_angle_rad = np.radians(config["min_angle"])
				max_angle_rad = np.radians(config["max_angle"])
				center_rad = (min_angle_rad + max_angle_rad) / 2
				amplitude_rad = (max_angle_rad - min_angle_rad) / 2
				
				position = center_rad + amplitude_rad * np.sin(phase)
				velocity = amplitude_rad * omega * np.cos(phase)
				acceleration = -amplitude_rad * omega**2 * np.sin(phase)
				
			elif "amplitude" in config:
				# 电机3-5: 使用对称幅度
				amp_rad = np.radians(config["amplitude"])
				position = amp_rad * np.sin(phase)
				velocity = amp_rad * omega * np.cos(phase)
				acceleration = -amp_rad * omega**2 * np.sin(phase)
			
			else:
				# 默认配置
				position = velocity = acceleration = np.zeros_like(t)
			
			all_positions[:, motor_idx] = position
			all_velocities[:, motor_idx] = velocity
			all_accelerations[:, motor_idx] = acceleration
		
		return {
			'time': t,
			'positions': all_positions,
			'velocities': all_velocities,
			'accelerations': all_accelerations,
			'motor_configs': motor_configs,
			'base_freq': base_freq,
			'phase_shifts': phase_shifts,
			'trajectory_type': 'phase_shifted'
		}
	
	def generate_multi_harmonic_trajectory(self, duration: float = 25.0,
										  motor_configs: Dict = None,
										  fundamental_freq: float = 0.08) -> Dict:
		"""
		生成多谐波轨迹 (基频 + 多个谐波)
		
		Args:
			duration: 轨迹持续时间
			motor_configs: 电机配置字典，包含每个电机的运动范围或幅度
			fundamental_freq: 基频 (Hz)
			
		Returns:
			多谐波轨迹
		"""
		if motor_configs is None:
			motor_configs = {
				1: {"min_angle": -30.0, "max_angle": 4.0},
				2: {"min_angle": -120.0, "max_angle": 40.0},
				3: {"amplitude": 100.0},
				4: {"amplitude": 100.0},
				5: {"amplitude": 100.0}
			}
		
		dt = 0.01
		t = np.arange(0, duration, dt)
		
		all_positions = np.zeros((len(t), self.num_motors))
		all_velocities = np.zeros((len(t), self.num_motors))
		all_accelerations = np.zeros((len(t), self.num_motors))
		
		# 每个电机使用不同的谐波组合
		# 动态生成谐波配置，支持任意数量的电机
		harmonic_configs = []
		base_harmonics = [
			[1, 3, 5],      # 基频 + 3次 + 5次谐波
			[1, 2, 4],      # 基频 + 2次 + 4次谐波
			[1, 3, 7],      # 基频 + 3次 + 7次谐波
			[1, 2, 5],      # 基频 + 2次 + 5次谐波
			[1, 4, 6],      # 基频 + 4次 + 6次谐波
			[1, 2, 6]       # 基频 + 2次 + 6次谐波 (第6个电机)
		]
		
		for i in range(self.num_motors):
			harmonic_configs.append(base_harmonics[i % len(base_harmonics)])
		
		for motor_idx in range(self.num_motors):
			motor_id = motor_idx + 1
			config = motor_configs.get(motor_id, {})
			freq_set = harmonic_configs[motor_idx]
			position = np.zeros_like(t)
			velocity = np.zeros_like(t)
			acceleration = np.zeros_like(t)
			
			if "min_angle" in config and "max_angle" in config:
				# 电机1和2: 使用指定范围
				min_angle_rad = np.radians(config["min_angle"])
				max_angle_rad = np.radians(config["max_angle"])
				center_rad = (min_angle_rad + max_angle_rad) / 2
				amplitude_rad = (max_angle_rad - min_angle_rad) / 2
				
				for freq in freq_set:
					omega = 2 * np.pi * fundamental_freq * freq
					# 修正：每个谐波分量应该围绕中心振荡，而不是叠加中心
					pos_component = amplitude_rad * np.sin(omega * t) / len(freq_set)
					vel_component = amplitude_rad * omega * np.cos(omega * t) / len(freq_set)
					acc_component = -amplitude_rad * omega**2 * np.sin(omega * t) / len(freq_set)
					
					position += pos_component
					velocity += vel_component
					acceleration += acc_component
				
				# 在所有谐波叠加后再加上中心偏移
				position += center_rad
					
			elif "amplitude" in config:
				# 电机3-5: 使用对称幅度
				amp_rad = np.radians(config["amplitude"])
				
				for freq in freq_set:
					omega = 2 * np.pi * fundamental_freq * freq
					pos_component = amp_rad * np.sin(omega * t) / len(freq_set)
					vel_component = amp_rad * omega * np.cos(omega * t) / len(freq_set)
					acc_component = -amp_rad * omega**2 * np.sin(omega * t) / len(freq_set)
					
					position += pos_component
					velocity += vel_component
					acceleration += acc_component
			
			else:
				# 默认配置
				position = velocity = acceleration = np.zeros_like(t)
			# 移除这个重复的中心偏移，已经在上面处理了
			
			all_positions[:, motor_idx] = position
			all_velocities[:, motor_idx] = velocity
			all_accelerations[:, motor_idx] = acceleration
		
		return {
			'time': t,
			'positions': all_positions,
			'velocities': all_velocities,
			'accelerations': all_accelerations,
			'motor_configs': motor_configs,
			'fundamental_freq': fundamental_freq,
			'harmonic_configs': harmonic_configs,
			'trajectory_type': 'multi_harmonic'
		}
	
	def generate_complex_trajectory(self, duration: float = 30.0, 
								  motor_configs: Dict = None) -> Dict:
		"""
		生成复合激励轨迹 (多个电机同时运动，不同频率)
		
		Args:
			duration: 轨迹持续时间
			motor_configs: 电机配置字典，包含每个电机的运动范围或幅度
			
		Returns:
			复合轨迹
		"""
		if motor_configs is None:
			motor_configs = {
				1: {"min_angle": -30.0, "max_angle": 4.0},
				2: {"min_angle": -120.0, "max_angle": 40.0},
				3: {"amplitude": 100.0},
				4: {"amplitude": 100.0},
				5: {"amplitude": 100.0}
			}  # 电机1默认范围: -30° 到 0°
		
		dt = 0.01
		t = np.arange(0, duration, dt)
		
		all_positions = np.zeros((len(t), self.num_motors))
		all_velocities = np.zeros((len(t), self.num_motors))
		all_accelerations = np.zeros((len(t), self.num_motors))
		
		# 为每个电机分配不同的频率组合 - 动态支持任意数量电机
		base_frequencies = [
			[0.05, 0.15],  # m1: 低频 + 中频
			[0.08, 0.12],  # m2: 
			[0.06, 0.18],  # m3:
			[0.10, 0.20],  # m4:
			[0.04, 0.16],  # m5: 
			[0.07, 0.14]   # m6: 新增第6个电机
		]
		
		frequencies = []
		for i in range(self.num_motors):
			frequencies.append(base_frequencies[i % len(base_frequencies)])
		
		for motor_idx in range(self.num_motors):
			motor_id = motor_idx + 1
			config = motor_configs.get(motor_id, {})
			
			position = np.zeros_like(t)
			velocity = np.zeros_like(t)
			acceleration = np.zeros_like(t)
			
			if "min_angle" in config and "max_angle" in config:
				# 电机1和2: 使用指定范围
				min_angle_rad = np.radians(config["min_angle"])
				max_angle_rad = np.radians(config["max_angle"])
				center_rad = (min_angle_rad + max_angle_rad) / 2
				amplitude_rad = (max_angle_rad - min_angle_rad) / 2
				
				# 多频率组合，但在指定范围内
				position = center_rad * np.ones_like(t)  # 从中心位置开始
				
				for freq in frequencies[motor_idx]:
					omega = 2 * np.pi * freq
					weight = 1.0 / len(frequencies[motor_idx])
					
					position += weight * amplitude_rad * np.sin(omega * t)
					velocity += weight * amplitude_rad * omega * np.cos(omega * t)
					acceleration += -weight * amplitude_rad * omega**2 * np.sin(omega * t)
				
			elif "amplitude" in config:
				# 电机3-5: 使用对称幅度
				amp_rad = np.radians(config["amplitude"])
				
				for freq in frequencies[motor_idx]:
					omega = 2 * np.pi * freq
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
			'motor_configs': motor_configs,
			'frequencies': frequencies,
			'trajectory_type': 'complex'
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
		
		colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'brown', 'pink', 'gray']
		
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
	"""主函数 - 生成各种类型的轨迹"""
	print("=== IC ARM 动力学辨识轨迹生成器 (安全增强版) ===\n")
	
	# 从配置文件获取URDF路径
	from ic_arm_control.utils.config_loader import get_urdf_path
	urdf_path = get_urdf_path()
	print(f"使用URDF文件: {urdf_path}")
	
	# 控制是否绘图的标志
	enable_plotting = False  # 设置为False禁用绘图
	
	# 初始化轨迹生成器 - 使用URDF文件
	generator = TrajectoryGenerator(urdf_path=urdf_path)
	
	# 从URDF自动生成电机配置
	motor_configs = generator._generate_motor_configs_from_urdf()
	
	# 1. 生成单个电机轨迹
	print("1. 生成单个电机轨迹...")
	for motor_id in range(1, generator.num_motors + 1):
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
		filename = f"trajectory_motor_{motor_id}_single_safe.json"
		generator.save_trajectory(single_traj, filename)
		
		# 验证轨迹安全性
		validation = generator.validate_trajectory_safety(single_traj)
		print(f"   电机 {motor_id} 安全验证: {'✓ 安全' if validation['is_safe'] else '✗ 有风险'}")
		if not validation['is_safe']:
			for violation in validation['violations']:
				print(f"     - {violation['type']}: {violation}")
		
		# 打印统计信息
		if f'motor_{motor_id}' in validation['statistics']:
			stats = validation['statistics'][f'motor_{motor_id}']
			print(f"     最大速度: {stats['max_abs_vel']:.3f} rad/s")
			print(f"     位置范围: [{np.degrees(stats['pos_range'][0]):.1f}°, {np.degrees(stats['pos_range'][1]):.1f}°]")
	
	# 2. 生成新的非同频轨迹类型
	print("\n2. 生成扫频轨迹...")
	chirp_traj = generator.generate_chirp_trajectory(
		duration=20.0,
		motor_configs=motor_configs,
		freq_range=[0.05, 0.5]
	)  # 从0.05Hz扫到0.5Hz
	generator.save_trajectory(chirp_traj, "trajectory_chirp_sweep.json")
	# generator.plot_trajectory(chirp_traj, "Chirp Sweep Trajectory (0.05-0.5Hz)")
	
	print("\n3. 生成随机轨迹...")
	random_traj = generator.generate_random_trajectory(
		duration=25.0,
		motor_configs=motor_configs,
		bandwidth=0.3,  # 0.3Hz带宽
		seed=42
	)
	generator.save_trajectory(random_traj, "trajectory_random_bandlimited.json")
	# generator.plot_trajectory(random_traj, "Random Band-Limited Trajectory (0.3Hz BW)")
	
	print("\n4. 生成相位偏移轨迹...")
	phase_shifted_traj = generator.generate_phase_shifted_trajectory(
		duration=20.0,
		motor_configs=motor_configs,
		base_freq=0.1  # 0.1Hz基频，各电机相位偏移72°
	)
	generator.save_trajectory(phase_shifted_traj, "trajectory_phase_shifted.json")
	if enable_plotting:
		generator.plot_trajectory(phase_shifted_traj, "Phase-Shifted Trajectory (72° shifts)")
	
	print("\n5. 生成多谐波轨迹...")
	harmonic_traj = generator.generate_multi_harmonic_trajectory(
		duration=25.0,
		motor_configs=motor_configs,
		fundamental_freq=0.08  # 0.08Hz基频 + 不同谐波组合
	)
	generator.save_trajectory(harmonic_traj, "trajectory_multi_harmonic.json")
	if enable_plotting:
		generator.plot_trajectory(harmonic_traj, "Multi-Harmonic Trajectory")
	
	# 6. 生成不同参数的扫频轨迹
	print("\n6. 生成高频扫频轨迹...")
	chirp_high_traj = generator.generate_chirp_trajectory(
		duration=15.0,
		motor_configs=motor_configs,  # 使用统一配置
		freq_range=[0.1, 1.0]  # 高频扫频
	)
	generator.save_trajectory(chirp_high_traj, "trajectory_chirp_high_freq.json")
	
	# 7. 生成不同种子的随机轨迹
	print("\n7. 生成多个随机轨迹变种...")
	for seed in [123, 456, 789]:
		random_var_traj = generator.generate_random_trajectory(
			duration=20.0,
			motor_configs=motor_configs,
			bandwidth=0.25,
			seed=seed
		)
		generator.save_trajectory(random_var_traj, f"trajectory_random_seed_{seed}.json")
	
	# 8. 生成不同频率的相位偏移轨迹
	print("\n8. 生成不同频率的相位偏移轨迹...")
	for freq in [0.06, 0.12, 0.18]:
		phase_var_traj = generator.generate_phase_shifted_trajectory(
			duration=18.0,
			motor_configs=motor_configs,
			base_freq=freq
		)
		generator.save_trajectory(phase_var_traj, f"trajectory_phase_shifted_{freq:.2f}Hz.json")
	
	# 9. 生成原有的序列和同时轨迹
	print("\n9. 生成序列轨迹...")
	# 动态生成轨迹文件列表，基于实际检测到的电机数量
	trajectory_files = []
	for motor_id in range(1, generator.num_motors + 1):
		trajectory_files.append(f"trajectory_motor_{motor_id}_single_safe.json")
	
	concatenated_traj = generator.concatenate_trajectories_sequential(
		trajectory_files=trajectory_files,
		rest_duration=2.0
	)
	generator.save_trajectory(concatenated_traj, "trajectory_concatenated_sequential.json")
	
	print("\n10. 生成同时运动轨迹...")
	simultaneous_traj = generator.concatenate_trajectories_simultaneous(trajectory_files)
	generator.save_trajectory(simultaneous_traj, "trajectory_concatenated_simultaneous.json")
	
	# 10. 生成复合轨迹
	print("\n10. 生成复合激励轨迹...")
	complex_traj = generator.generate_complex_trajectory(
		duration=30.0,
		motor_configs=motor_configs
	)
	generator.save_trajectory(complex_traj, "trajectory_complex_multifreq.json")
	
	print("\n=== 轨迹生成完成 ===")
	print("生成的文件:")
	print("\n【单电机轨迹】")
	print("- trajectory_motor_1_single.json  (-30° to -4°)")
	print("- trajectory_motor_2_single.json  (-120° to +40°)")
	print("- trajectory_motor_3_single.json  (±100°)")
	print("- trajectory_motor_4_single.json  (±100°)")
	print("- trajectory_motor_5_single.json  (±100°)")
	
	print("\n【非同频轨迹】")
	print("- trajectory_chirp_sweep.json  (扫频 0.05-0.5Hz)")
	print("- trajectory_chirp_high_freq.json  (高频扫频 0.1-1.0Hz)")
	print("- trajectory_random_bandlimited.json  (随机 0.3Hz带宽)")
	print("- trajectory_random_seed_123.json  (随机变种1)")
	print("- trajectory_random_seed_456.json  (随机变种2)")
	print("- trajectory_random_seed_789.json  (随机变种3)")
	print("- trajectory_phase_shifted.json  (相位偏移 0.1Hz)")
	print("- trajectory_phase_shifted_0.06Hz.json  (相位偏移变种1)")
	print("- trajectory_phase_shifted_0.12Hz.json  (相位偏移变种2)")
	print("- trajectory_phase_shifted_0.18Hz.json  (相位偏移变种3)")
	print("- trajectory_multi_harmonic.json  (多谐波)")
	
	print("\n【组合轨迹】")
	print("- trajectory_concatenated_sequential.json  (1-5-4-3-2 序列)")
	print("- trajectory_concatenated_simultaneous.json  (所有电机同时)")
	print("- trajectory_complex_multifreq.json  (多频率复合轨迹)")
	
	print(f"\n总计生成了 {5 + 7 + 3 + 3} = 18 个轨迹文件！")
	print("\n这些轨迹涵盖了:")
	print("✓ 单电机运动")
	print("✓ 扫频激励 (线性变频)")
	print("✓ 随机激励 (带限白噪声)")
	print("✓ 相位偏移 (同频不同相)")
	print("✓ 多谐波 (基频+谐波)")
	print("✓ 序列运动")
	print("✓ 同时运动")
	print("✓ 多频率复合")

def test_safety_features():
	"""测试安全功能的专用函数"""
	print("=== 安全功能测试 ===\n")
	
	# 初始化带URDF的生成器（自动检测关节数量）
	urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/urdfs/ic_arm_8_dof/urdf/ic_arm_8_dof.urdf"
	generator = TrajectoryGenerator(urdf_path=urdf_path)
	
	print("\n1. 测试关节限制约束...")
	# 生成一个可能超出限制的轨迹
	unsafe_traj = generator.generate_single_motor_trajectory(
		motor_id=1, 
		min_angle=-50.0,  # 超出安全范围
		max_angle=20.0,   # 超出安全范围
		duration=5.0
	)
	
	# 验证原始轨迹
	validation = generator.validate_trajectory_safety(unsafe_traj)
	print(f"原始轨迹安全性: {'✓ 安全' if validation['is_safe'] else '✗ 有风险'}")
	
	# 应用安全约束
	safe_traj = generator.apply_safety_constraints_to_trajectory(unsafe_traj)
	safe_validation = generator.validate_trajectory_safety(safe_traj)
	print(f"约束后轨迹安全性: {'✓ 安全' if safe_validation['is_safe'] else '✗ 有风险'}")
	
	print("\n2. 测试速度限制...")
	for motor_id in [1, 2, 3]:
		traj = generator.generate_single_motor_trajectory(
			motor_id=motor_id,
			amplitude=30.0,
			duration=2.0,  # 短时间内大幅运动，测试速度限制
			num_cycles=3
		)
		validation = generator.validate_trajectory_safety(traj)
		stats = validation['statistics'].get(f'motor_{motor_id}', {})
		max_vel = stats.get('max_abs_vel', 0)
		print(f"电机 {motor_id} 最大速度: {max_vel:.3f} rad/s ({'✓' if max_vel <= 3.0 else '✗'})")
	
	print("\n3. 测试碰撞检测...")
	# 生成可能导致碰撞的轨迹（大角度joint2运动）
	collision_traj = generator.generate_single_motor_trajectory(
		motor_id=2,
		min_angle=-10.0,
		max_angle=150.0,  # 大角度可能导致碰撞
		duration=8.0
	)
	collision_validation = generator.validate_trajectory_safety(collision_traj, check_collisions=True)
	print(f"碰撞检测结果: {'✓ 无碰撞' if collision_validation['is_safe'] else '✗ 检测到潜在碰撞'}")
	
	print("\n4. 生成MuJoCo场景文件...")
	mujoco_xml = generator.create_mujoco_scene_with_obstacle(urdf_path)
	with open("/Users/lr-2002/project/instantcreation/IC_arm_control/robot_with_obstacle.xml", "w") as f:
		f.write(mujoco_xml)
	print("✓ MuJoCo场景文件已生成: robot_with_obstacle.xml")
	
	return generator

if __name__ == "__main__":
	import sys
	if len(sys.argv) > 1 and sys.argv[1] == "--test-safety":
		test_safety_features()
	else:
		main()
