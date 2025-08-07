#!/usr/bin/env python3
"""
基于Pinocchio的相对位置标定工具
使用Pinocchio库进行精确的前向运动学计算
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import json
import time
from tqdm import tqdm

try:
	import pinocchio as pin
	PINOCCHIO_AVAILABLE = True
	print("Pinocchio库已加载")
except ImportError:
	PINOCCHIO_AVAILABLE = False
	print("警告: Pinocchio库未安装，将使用备用方法")

class PinocchioPoseIdentifier:
	def __init__(self, synchronized_data_path, urdf_path):
		"""
		初始化基于Pinocchio的相对位置标定工具
		
		Args:
			synchronized_data_path: 同步数据文件路径
			urdf_path: URDF文件路径
		"""
		self.data = pd.read_csv(synchronized_data_path)
		self.urdf_path = urdf_path
		print(f"加载数据: {len(self.data)} 行")
		
		if not PINOCCHIO_AVAILABLE:
			raise ImportError("需要安装Pinocchio库: pip install pin")
		
		# 加载URDF模型
		self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(urdf_path)
		self.data_model = self.model.createData()
		
		print(f"加载URDF模型: {self.model.name}")
		print(f"关节数量: {self.model.nq}")
		print(f"关节名称: {[self.model.names[i] for i in range(1, self.model.njoints)]}")
		
		# 找到关键frame的ID
		self.find_frame_ids()
		
		# 只需要关节零点偏移
		self.joint_offsets = np.zeros(5)  # 5个关节的零点偏移
		
		# 进度跟踪
		self.iteration_count = 0
		self.start_time = None
		self.progress_bar = None
	
	def find_frame_ids(self):
		"""找到基座和末端执行器的frame ID"""
		# 打印所有可用的frame
		print("\n可用的frames:")
		for i in range(self.model.nframes):
			frame = self.model.frames[i]
			print(f"  {i}: {frame.name} (parent: {self.model.names[frame.parent]})")
		
		# 找到正确的frame ID
		# 基座：base_link的根部
		# 末端执行器：最后一个link（l5）的末端点
		
		# 找到base_link frame
		self.base_frame_id = None
		self.ee_frame_id = None
		
		for i in range(self.model.nframes):
			frame = self.model.frames[i]
			if frame.name == 'base_root_marker':
				self.base_frame_id = i
			elif frame.name == 'l5':  # 最后一个link
				self.ee_frame_id = i
		
		if self.base_frame_id is None or self.ee_frame_id is None:
			print("警告: 未找到正确的frame，使用默认设置")
			self.base_frame_id = 1  # base_link frame
			self.ee_frame_id = 11   # l5 frame
		
		print(f"\n使用frame作为参考点:")
		print(f"  基座参考: Frame {self.base_frame_id} ({self.model.frames[self.base_frame_id].name})")
		print(f"  末端参考: Frame {self.ee_frame_id} ({self.model.frames[self.ee_frame_id].name})")
	
	def forward_kinematics_relative(self, joint_angles):
		"""
		使用Pinocchio计算末端执行器相对于基座的位置
		
		Args:
			joint_angles: 关节角度数组 [5]
		
		Returns:
			relative_pos: 末端执行器相对于基座的位置向量 [3]
		"""
		# 应用关节偏移
		q = joint_angles + self.joint_offsets
		
		# 确保关节角度数组长度正确
		if len(q) != self.model.nq:
			# 如果模型有更多关节（如固定关节），用零填充
			q_full = np.zeros(self.model.nq)
			q_full[:len(q)] = q
			q = q_full
		
		# 计算前向运动学
		pin.forwardKinematics(self.model, self.data_model, q)
		pin.updateFramePlacements(self.model, self.data_model)
		
		# 获取基座和末端执行器位置（使用frame）
		base_pos = self.data_model.oMf[self.base_frame_id].translation
		ee_pos = self.data_model.oMf[self.ee_frame_id].translation
		
		# 计算相对位置
		relative_pos = ee_pos - base_pos
		
		return relative_pos
	
	def extract_mocap_relative_position(self, data_row):
		"""
		提取动作捕捉系统中末端执行器相对于基座的位置
		
		Args:
			data_row: 数据行
			
		Returns:
			relative_mocap_pos: 末端执行器相对于基座的位置向量 (3,)
		"""
		# 基座标记点 (使用所有有效标记点的平均)
		base_markers = []
		for i in range(1, 6):  # M1, M2, M3, M4, M5 - 检查所有可能的标记点
			x = data_row[f'base_M{i}_X']
			y = data_row[f'base_M{i}_Y']
			z = data_row[f'base_M{i}_Z']
			if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
				base_markers.append([x, y, z])
		
		# 末端执行器标记点 (使用所有有效标记点的平均)
		ee_markers = []
		for i in range(1, 6):  # M1, M2, M3, M4, M5 - 检查所有可能的标记点
			x = data_row[f'ee_M{i}_X']
			y = data_row[f'ee_M{i}_Y']
			z = data_row[f'ee_M{i}_Z']
			if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
				ee_markers.append([x, y, z])
		
		if not base_markers or not ee_markers:
			return np.array([0, 0, 0])
		
		base_mocap_pos = np.mean(base_markers, axis=0)
		ee_mocap_pos = np.mean(ee_markers, axis=0)
		
		# 计算相对位置（转换单位 mm -> m）
		relative_mocap_pos = (ee_mocap_pos - base_mocap_pos) / 1000.0
		
		return relative_mocap_pos
	
	def objective_function(self, joint_offsets):
		"""
		相对位置优化目标函数
		
		Args:
			joint_offsets: 关节零点偏移 [5]
		
		Returns:
			residuals: 残差数组
		"""
		# 更新进度
		self.iteration_count += 1
		
		# 强制终止机制
		if hasattr(self, 'max_iterations') and self.iteration_count >= self.max_iterations:
			print(f"\n达到最大迭代次数 {self.max_iterations}，强制终止优化")
			return np.array([1e10])
		
		if self.progress_bar is not None:
			elapsed_time = time.time() - self.start_time
			self.progress_bar.set_description(f"Iter {self.iteration_count} | Time: {elapsed_time:.1f}s")
			if self.iteration_count <= self.max_iterations:
				self.progress_bar.update(1)
		
		# 更新关节偏移
		self.joint_offsets = joint_offsets
		
		residuals = []
		
		# 使用数据子集进行优化
		sample_step = max(1, len(self.data) // 1000)  # 最多使用1000个数据点
		if self.iteration_count == 1:  # 只在第一次迭代时打印
			print(f"使用采样步长: {sample_step}, 总数据点: {len(self.data)} -> {len(self.data)//sample_step}")
		
		for idx in range(0, len(self.data), sample_step):
			row = self.data.iloc[idx]
			
			# 提取关节角度
			joint_angles = np.array([
				row['m1_rad'], row['m2_rad'], row['m3_rad'],
				row['m4_rad'], row['m5_rad']
			])
			
			# 提取mocap相对位置
			relative_mocap = self.extract_mocap_relative_position(row)
			
			# 跳过无效数据
			if np.allclose(relative_mocap, 0):
				continue
			
			try:
				# 计算Pinocchio前向运动学相对位置
				relative_fk = self.forward_kinematics_relative(joint_angles)
				
				# 计算残差
				error = relative_mocap - relative_fk
				residuals.extend(error)
			except Exception as e:
				# 如果计算失败，跳过这个数据点
				if self.iteration_count == 1:
					print(f"警告: 前向运动学计算失败 (idx={idx}): {e}")
				continue
		
		return np.array(residuals)
	
	def identify_relative_pose(self):
		"""
		执行相对位置标定
		"""
		print("=== 开始基于Pinocchio的相对位置标定 ===")
		
		# 初始化进度跟踪
		self.iteration_count = 0
		self.start_time = time.time()
		
		# 初始参数估计（只有5个关节偏移）
		initial_params = np.zeros(5)
		
		# 设置参数边界（±90度）
		bounds = [(-np.pi/8, np.pi/8)] * 5
		
		print("开始优化...")
		print(f"数据集大小: {len(self.data)} 行")
		print("优化参数: 5个关节零点偏移")
		print("前向运动学: Pinocchio库")
		
		# 初始化进度条
		max_iterations = 300
		self.max_iterations = max_iterations
		self.progress_bar = tqdm(total=max_iterations, desc="Pinocchio Optimization", 
								unit="iter", ncols=100)
		
		try:
			# 使用最小二乘法优化
			result = least_squares(
				self.objective_function, 
				initial_params,
				bounds=([b[0] for b in bounds], [b[1] for b in bounds]),
				method='trf',
				max_nfev=max_iterations,
				ftol=1e-8,
				xtol=1e-8,
				gtol=1e-8
			)
		finally:
			# 关闭进度条
			if self.progress_bar is not None:
				self.progress_bar.close()
		
		if result.success:
			print("优化成功!")
			print(f"最终残差: {np.linalg.norm(result.fun):.6f}")
			
			# 保存结果
			calibration_result = {
				"method": "pinocchio_relative_pose",
				"urdf_path": self.urdf_path,
				"joint_offsets_rad": self.joint_offsets.tolist(),
				"joint_offsets_deg": np.degrees(self.joint_offsets).tolist(),
				"model_info": {
					"model_name": self.model.name,
					"n_joints": int(self.model.nq),
					"joint_names": [self.model.names[i] for i in range(1, min(6, self.model.njoints))],
					"base_frame_id": int(self.base_frame_id),
					"ee_frame_id": int(self.ee_frame_id)
				},
				"optimization_info": {
					"success": result.success,
					"final_cost": float(np.linalg.norm(result.fun)),
					"iterations": result.nfev,
					"message": result.message
				}
			}
			
			with open('pinocchio_pose_calibration.json', 'w') as f:
				json.dump(calibration_result, f, indent=2)
			
			print("\n=== Pinocchio相对位置标定结果 ===")
			print("关节零点偏移 (度):")
			for i, offset in enumerate(np.degrees(self.joint_offsets)):
				print(f"  关节 {i+1}: {offset:.3f}°")
			
			print(f"\n结果已保存到: pinocchio_pose_calibration.json")
			
			# 验证标定结果
			self.validate_calibration()
			
		else:
			print("优化失败!")
			print(f"原因: {result.message}")
			print(f"最终残差: {np.linalg.norm(result.fun):.6f}")
	
	def validate_calibration(self, sample_points=100):
		"""
		验证相对位置标定结果
		"""
		print("\n=== Pinocchio相对位置标定验证 ===")
		
		# 采样数据点
		step = max(1, len(self.data) // sample_points)
		sample_indices = range(0, len(self.data), step)
		
		errors = []
		mocap_relatives = []
		fk_relatives = []
		
		for idx in sample_indices:
			row = self.data.iloc[idx]
			
			# 提取关节角度
			joint_angles = np.array([
				row['m1_rad'], row['m2_rad'], row['m3_rad'],
				row['m4_rad'], row['m5_rad']
			])
			
			# 提取mocap相对位置
			relative_mocap = self.extract_mocap_relative_position(row)
			
			# 跳过无效数据
			if np.allclose(relative_mocap, 0):
				continue
			
			try:
				# 计算Pinocchio前向运动学相对位置
				relative_fk = self.forward_kinematics_relative(joint_angles)
				
				# 计算误差
				error = np.linalg.norm(relative_mocap - relative_fk)
				errors.append(error)
				mocap_relatives.append(relative_mocap)
				fk_relatives.append(relative_fk)
			except Exception:
				continue
		
		errors = np.array(errors)
		mocap_relatives = np.array(mocap_relatives)
		fk_relatives = np.array(fk_relatives)
		
		print(f"相对位置误差统计:")
		print(f"  平均误差: {np.mean(errors)*1000:.2f} mm")
		print(f"  最大误差: {np.max(errors)*1000:.2f} mm")
		print(f"  标准差: {np.std(errors)*1000:.2f} mm")
		print(f"  验证数据点: {len(errors)} 个")
		
		# 创建验证可视化
		self.create_validation_plots(mocap_relatives, fk_relatives, errors)
	
	def create_validation_plots(self, mocap_relatives, fk_relatives, errors):
		"""创建验证可视化图表"""
		fig, axes = plt.subplots(2, 3, figsize=(18, 12))
		fig.suptitle('Pinocchio Relative Position Calibration Validation', fontsize=16)
		
		# 1. 3D相对位置对比
		ax = axes[0, 0]
		ax.remove()
		ax = fig.add_subplot(2, 3, 1, projection='3d')
		
		ax.scatter(mocap_relatives[:, 0], mocap_relatives[:, 1], mocap_relatives[:, 2], 
				  c='red', s=20, alpha=0.6, label='MoCap Relative')
		ax.scatter(fk_relatives[:, 0], fk_relatives[:, 1], fk_relatives[:, 2], 
				  c='blue', s=15, alpha=0.8, marker='^', label='Pinocchio FK')
		
		ax.set_xlabel('X (m)')
		ax.set_ylabel('Y (m)')
		ax.set_zlabel('Z (m)')
		ax.set_title('3D Relative Position Comparison')
		ax.legend()
		
		# 2-4. XYZ方向对比
		directions = ['X', 'Y', 'Z']
		for i, direction in enumerate(directions):
			ax = axes[0, 1] if i == 0 else (axes[0, 2] if i == 1 else axes[1, 0])
			ax.scatter(mocap_relatives[:, i], fk_relatives[:, i], alpha=0.6)
			ax.plot([-1, 1], [-1, 1], 'r--', label='Perfect Match')
			ax.set_xlabel(f'MoCap {direction} (m)')
			ax.set_ylabel(f'Pinocchio {direction} (m)')
			ax.set_title(f'{direction} Direction Comparison')
			ax.legend()
			ax.grid(True)
		
		# 5. 误差分布
		axes[1, 1].hist(errors * 1000, bins=20, alpha=0.7, color='orange')
		axes[1, 1].set_xlabel('Position Error (mm)')
		axes[1, 1].set_ylabel('Frequency')
		axes[1, 1].set_title(f'Error Distribution\nMean: {np.mean(errors)*1000:.2f}mm')
		axes[1, 1].grid(True)
		
		# 6. 误差时间序列
		axes[1, 2].plot(errors * 1000, alpha=0.7)
		axes[1, 2].set_xlabel('Sample Index')
		axes[1, 2].set_ylabel('Position Error (mm)')
		axes[1, 2].set_title('Error Over Time')
		axes[1, 2].grid(True)
		
		plt.tight_layout()
		plt.savefig('pinocchio_pose_validation.png', dpi=300, bbox_inches='tight')
		plt.show()
		
		print("验证图表已保存为: pinocchio_pose_validation.png")

def main():
	# URDF文件路径（使用Pinocchio兼容版本）
	urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/ic_arm_urdf/urdf/ic1.1.2.urdf"
	
	# 创建基于Pinocchio的相对位置标定工具
	identifier = PinocchioPoseIdentifier('synchronized_data.csv', urdf_path)
	
	# 执行相对位置标定
	identifier.identify_relative_pose()

if __name__ == "__main__":
	main()
