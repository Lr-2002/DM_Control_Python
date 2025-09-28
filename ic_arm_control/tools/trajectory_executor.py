#!/usr/bin/env python3
"""
轨迹执行器 - 支持MuJoCo仿真和IC_ARM硬件两种执行模式
包含速度调节功能，通过插值实现轨迹播放速度控制
"""

import json
import time
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from ic_arm_control.utils.config_loader import get_joint_names

class TrajectoryExecutor:
	"""轨迹执行器 - 支持仿真和硬件两种模式"""
	
	def __init__(self, mode: str = "mujoco", speed_factor: float = 1.0):
		"""
		初始化轨迹执行器
		
		Args:
			mode: 执行模式 "mujoco" 或 "ic_arm"
			speed_factor: 播放速度倍数 (1.0=正常速度, >1.0=加速, <1.0=减速)
		"""
		self.mode = mode
		self.speed_factor = speed_factor
		self.joint_names = get_joint_names()
		self.num_joints = len(self.joint_names)
		
		print(f"轨迹执行器初始化: 模式={mode}, 关节数={self.num_joints}, 速度倍数={speed_factor}")
		
		# 根据模式初始化对应的执行器
		if mode == "ic_arm":
			self._init_ic_arm()
		elif mode == "mujoco":
			self._init_mujoco()
		else:
			raise ValueError(f"不支持的模式: {mode}")
	
	def _init_ic_arm(self):
		"""初始化IC_ARM硬件"""
		try:
			from ic_arm_control.control.IC_ARM import ICARM
			self.arm = ICARM(debug=False, gc=True)
			print("IC_ARM硬件初始化成功")
		except Exception as e:
			print(f"IC_ARM硬件初始化失败: {e}")
			raise
	
	def _init_mujoco(self):
		"""初始化MuJoCo仿真"""
		try:
			import mujoco
			import mujoco.viewer
			from ic_arm_control.utils.config_loader import get_urdf_path
			
			# 获取URDF路径
			urdf_path = get_urdf_path()
			print(f"加载URDF模型: {urdf_path}")
			
			# 检查URDF文件是否存在
			if not os.path.exists(urdf_path):
				print(f"URDF文件不存在: {urdf_path}")
				raise FileNotFoundError(f"URDF file not found: {urdf_path}")
			
			# 切换到URDF目录以便找到mesh文件
			original_cwd = os.getcwd()
			urdf_dir = os.path.dirname(urdf_path)
			os.chdir(urdf_dir)
			
			try:
				# 直接从URDF加载MuJoCo模型
				self.mj_model = mujoco.MjModel.from_xml_path(urdf_path)
				self.mj_data = mujoco.MjData(self.mj_model)
				print(f"✓ MuJoCo模型加载成功!")
				print(f"  关节数量: {self.mj_model.njnt}")
				print(f"  自由度: {self.mj_model.nv}")
				
				# 获取关节映射
				self.joint_indices = {}
				for i, joint_name in enumerate(self.joint_names):
					try:
						joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
						self.joint_indices[i] = joint_id
						print(f"  映射关节 {i}: {joint_name} -> ID {joint_id}")
					except Exception as e:
						print(f"  警告: 找不到关节 {joint_name}: {e}")
				
			finally:
				# 恢复原始工作目录
				os.chdir(original_cwd)
			
			print("✓ MuJoCo仿真模式初始化成功")
			
		except ImportError:
			print("MuJoCo未安装，使用模拟模式")
			self.mj_model = None
			self.mj_data = None
		except Exception as e:
			print(f"MuJoCo初始化失败: {e}")
			print("回退到模拟模式")
			self.mj_model = None
			self.mj_data = None
	
	def load_trajectory(self, trajectory_file: str) -> Dict:
		"""加载轨迹文件"""
		print(f"加载轨迹文件: {trajectory_file}")
		
		with open(trajectory_file, 'r') as f:
			trajectory = json.load(f)
		
		# 转换为numpy数组
		trajectory['time'] = np.array(trajectory['time'])
		trajectory['positions'] = np.array(trajectory['positions'])
		trajectory['velocities'] = np.array(trajectory['velocities'])
		trajectory['accelerations'] = np.array(trajectory['accelerations'])
		
		print(f"轨迹加载完成: {len(trajectory['time'])} 个数据点, 持续时间: {trajectory['time'][-1]:.2f}秒")
		return trajectory
	
	def set_speed_factor(self, speed_factor: float):
		"""设置播放速度倍数"""
		self.speed_factor = speed_factor
		print(f"播放速度设置为: {speed_factor}x")
	
	def _interpolate_trajectory(self, trajectory: Dict) -> Dict:
		"""根据速度倍数对轨迹进行插值调整"""
		if self.speed_factor == 1.0:
			return trajectory
		
		print(f"对轨迹进行速度调整: {self.speed_factor}x")
		
		# 调整时间轴
		original_time = trajectory['time']
		new_time = original_time / self.speed_factor
		
		# 调整速度和加速度（时间压缩，速度和加速度需要相应调整）
		new_velocities = trajectory['velocities'] * self.speed_factor
		new_accelerations = trajectory['accelerations'] * (self.speed_factor ** 2)
		
		return {
			'time': new_time,
			'positions': trajectory['positions'],
			'velocities': new_velocities,
			'accelerations': new_accelerations
		}
	
	def execute_trajectory(self, trajectory: Dict) -> bool:
		"""执行轨迹"""
		print(f"开始执行轨迹 (模式: {self.mode}, 速度: {self.speed_factor}x)")
		
		# 应用速度调节
		adjusted_trajectory = self._interpolate_trajectory(trajectory)
		
		if self.mode == "mujoco":
			return self._execute_mujoco(adjusted_trajectory)
		elif self.mode == "ic_arm":
			return self._execute_ic_arm(adjusted_trajectory)
		else:
			return False
	
	def _execute_mujoco(self, trajectory: Dict) -> bool:
		"""MuJoCo仿真执行"""
		print("MuJoCo仿真执行轨迹...")
		
		time_points = trajectory['time']
		positions = trajectory['positions']
		
		# 如果有MuJoCo模型，使用真实仿真
		if self.mj_model is not None and self.mj_data is not None:
			return self._execute_mujoco_sim(trajectory)
		
		# 否则使用模拟模式
		print("使用模拟模式执行轨迹...")
		start_time = time.time()
		
		for i, (target_time, target_positions) in enumerate(zip(time_points, positions)):
			# 等待到目标时间
			current_time = time.time() - start_time
			while current_time < target_time:
				time.sleep(0.001)
				current_time = time.time() - start_time
			
			# 确保位置数组长度匹配关节数量
			action = target_positions[:self.num_joints] if len(target_positions) >= self.num_joints else target_positions
			
			# 进度显示
			if i % 100 == 0:
				progress = (i / len(time_points)) * 100
				print(f"执行进度: {progress:.1f}% - 关节位置: {np.degrees(action)}")
		
		print("MuJoCo轨迹执行完成")
		return True
	
	def _execute_mujoco_sim(self, trajectory: Dict) -> bool:
		"""真实MuJoCo仿真执行（需要mjpython运行）"""
		try:
			import mujoco
			import mujoco.viewer
			
			print("启动MuJoCo viewer...")
			time_points = trajectory['time']
			positions = trajectory['positions']
			
			with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
				# 设置相机位置
				viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
				viewer.cam.distance = 1.5
				viewer.cam.azimuth = 45
				viewer.cam.elevation = -30
				
				print("MuJoCo Viewer已启动")
				print("  - 右键拖拽旋转视角")
				print("  - 滚轮缩放")
				print("  - 按ESC或关闭窗口退出")
				
				start_time = time.time()
				
				for i, (target_time, target_positions) in enumerate(zip(time_points, positions)):
					# 检查viewer是否还在运行
					if not viewer.is_running():
						print("Viewer已关闭，停止执行")
						break
					
					# 等待到目标时间
					current_time = time.time() - start_time
					while current_time < target_time:
						time.sleep(0.001)
						current_time = time.time() - start_time
						if not viewer.is_running():
							break
					
					if not viewer.is_running():
						break
					
					# 设置关节位置（使用关节映射）
					for joint_idx, mj_joint_id in self.joint_indices.items():
						if joint_idx < len(target_positions):
							self.mj_data.qpos[mj_joint_id] = target_positions[joint_idx]
					
					# 前向动力学
					mujoco.mj_forward(self.mj_model, self.mj_data)
					
					# 更新viewer
					viewer.sync()
					
					# 进度显示
					if i % 100 == 0:
						progress = (i / len(time_points)) * 100
						print(f"执行进度: {progress:.1f}% - 时间: {target_time:.2f}s")
			
			print("MuJoCo仿真执行完成")
			return True
			
		except Exception as e:
			print(f"MuJoCo仿真执行失败: {e}")
			import traceback
			traceback.print_exc()
			return False
	
	def _execute_ic_arm(self, trajectory: Dict) -> bool:
		"""IC_ARM硬件执行"""
		print("IC_ARM硬件执行轨迹...")
		
		if not self.arm:
			print("IC_ARM未初始化")
			return False
		
		# 转换为IC_ARM所需的轨迹点格式
		trajectory_points = self._convert_to_trajectory_points(trajectory)
		
		try:
			# 使用IC_ARM的轨迹执行功能
			return self.arm.execute_trajectory_points(trajectory_points, verbose=True)
		except Exception as e:
			print(f"IC_ARM轨迹执行失败: {e}")
			return False
	
	def _convert_to_trajectory_points(self, trajectory: Dict) -> List[List[float]]:
		"""将轨迹转换为IC_ARM所需的点格式"""
		time_points = trajectory['time']
		positions = trajectory['positions']
		
		trajectory_points = []
		zero_pad = [0, 0, 0]
		for i in range(len(time_points)):
			# 格式: [pos1_deg, pos2_deg, ..., pos6_deg, timestamp]
			positions_deg = np.degrees(positions[i][:self.num_joints])
			point = list(positions_deg) + zero_pad+  [time_points[i]]
			trajectory_points.append(point)
		
		return trajectory_points
	
	def __del__(self):
		"""清理资源"""
		if hasattr(self, 'arm') and self.arm and self.mode == "ic_arm":
			try:
				self.arm.disable_all_motors()
			except:
				pass

def main():
	"""主函数 - 演示轨迹执行"""
	import sys
	
	# 解析命令行参数
	if len(sys.argv) < 2:
		print("用法:")
		print("  python trajectory_executor.py <trajectory_file> [mode] [speed_factor]")
		print("  mjpython trajectory_executor.py <trajectory_file> [mode] [speed_factor]  # 使用MuJoCo viewer")
		print()
		print("参数:")
		print("  mode: mujoco 或 ic_arm (默认: mujoco)")
		print("  speed_factor: 播放速度倍数 (默认: 1.0)")
		print()
		print("注意:")
		print("  - 使用MuJoCo viewer需要用 mjpython 启动")
		print("  - 普通python只能使用模拟模式")
		return
	
	trajectory_file = sys.argv[1]
	mode = sys.argv[2] if len(sys.argv) > 2 else "mujoco"
	speed_factor = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
	
	try:
		# 创建执行器
		executor = TrajectoryExecutor(mode=mode, speed_factor=speed_factor)
		
		# 加载轨迹
		trajectory = executor.load_trajectory(trajectory_file)
		
		# 执行轨迹
		success = executor.execute_trajectory(trajectory)
		
		if success:
			print("轨迹执行成功!")
		else:
			print("轨迹执行失败!")
			
	except Exception as e:
		print(f"执行过程中出错: {e}")
		import traceback
		traceback.print_exc()

if __name__ == "__main__":
	main()