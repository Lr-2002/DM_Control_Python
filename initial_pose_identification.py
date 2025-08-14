#!/usr/bin/env python3
"""
Initial Pose Identification Tool
利用同步的动作捕捉数据和机械臂关节角度数据进行初始位置辨识
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from scipy.spatial.transform import Rotation as R
import json
from urdf_parser import URDFParser
import time
from tqdm import tqdm

class InitialPoseIdentifier:
    def __init__(self, synchronized_data_path):
        """
        初始化位置辨识器
        
        Args:
            synchronized_data_path: 同步数据文件路径
        """
        self.data = pd.read_csv(synchronized_data_path)
        self.joint_offsets = np.zeros(5)  # 5个关节的零点偏移
        self.base_transform = np.eye(4)   # 基座坐标变换矩阵
        self.ee_transform = np.eye(4)     # 末端执行器坐标变换矩阵
        
        # 进度跟踪
        self.iteration_count = 0
        self.start_time = None
        self.progress_bar = None
        
        # 从URDF文件加载DH参数
        self.dh_params = self.load_dh_from_urdf()
        print("Loaded DH parameters from URDF:")
        print(f"  a (link lengths): {self.dh_params['a']}")
        print(f"  d (link offsets): {self.dh_params['d']}")
        print(f"  alpha (link twists): {self.dh_params['alpha']}")
        print(f"  theta_offset: {self.dh_params['theta_offset']}")
    
    def load_dh_from_urdf(self):
        """
        从URDF文件加载DH参数
        
        Returns:
            dict: DH参数字典
        """
        urdf_path = '/Users/lr-2002/project/instantcreation/IC_arm_control/ic_arm_urdf/urdf/ic1.1.2.urdf'
        parser = URDFParser(urdf_path)
        return parser.convert_to_dh_parameters()
        
    def forward_kinematics(self, joint_angles):
        """
        正向运动学计算
        
        Args:
            joint_angles: 关节角度数组 [rad]
            
        Returns:
            base_pos: 基座位置 (3,)
            ee_pos: 末端执行器位置 (3,)
        """
        # 添加关节偏移和URDF中的theta_offset
        q = joint_angles + self.joint_offsets + np.array(self.dh_params['theta_offset'])
        
        # DH变换矩阵
        T = np.eye(4)
        
        for i in range(5):
            # DH变换
            cos_q = np.cos(q[i])
            sin_q = np.sin(q[i])
            cos_alpha = np.cos(self.dh_params['alpha'][i])
            sin_alpha = np.sin(self.dh_params['alpha'][i])
            a = self.dh_params['a'][i]
            d = self.dh_params['d'][i]
            
            Ti = np.array([
                [cos_q, -sin_q * cos_alpha, sin_q * sin_alpha, a * cos_q],
                [sin_q, cos_q * cos_alpha, -cos_q * sin_alpha, a * sin_q],
                [0, sin_alpha, cos_alpha, d],
                [0, 0, 0, 1]
            ])
            
            T = T @ Ti
        
        # 基座位置 (第一个关节后的位置)
        T_base = np.eye(4)
        q0_total = q[0]  # 包含所有偏移的第一关节角度
        T_base = T_base @ np.array([
            [np.cos(q0_total), -np.sin(q0_total), 0, self.dh_params['a'][0] * np.cos(q0_total)],
            [np.sin(q0_total), np.cos(q0_total), 0, self.dh_params['a'][0] * np.sin(q0_total)],
            [0, 0, 1, self.dh_params['d'][0]],
            [0, 0, 0, 1]
        ])
        
        base_pos = (self.base_transform @ T_base)[:3, 3]
        ee_pos = (self.ee_transform @ T)[:3, 3]
        
        return base_pos, ee_pos
    
    def extract_mocap_positions(self, data_row):
        """
        提取动作捕捉标记点的平均位置
        
        Args:
            data_row: 数据行
            
        Returns:
            base_mocap_pos: 基座动作捕捉位置 (3,)
            ee_mocap_pos: 末端执行器动作捕捉位置 (3,)
        """
        # 基座标记点 (取前3个有效标记点的平均)
        base_markers = []
        for i in range(1, 4):  # M1, M2, M3
            x = data_row[f'base_M{i}_X']
            y = data_row[f'base_M{i}_Y'] 
            z = data_row[f'base_M{i}_Z']
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                base_markers.append([x, y, z])
        
        # 末端执行器标记点
        ee_markers = []
        for i in range(1, 4):  # M1, M2, M3
            x = data_row[f'ee_M{i}_X']
            y = data_row[f'ee_M{i}_Y']
            z = data_row[f'ee_M{i}_Z']
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                ee_markers.append([x, y, z])
        
        base_mocap_pos = np.mean(base_markers, axis=0) if base_markers else np.array([0, 0, 0])
        ee_mocap_pos = np.mean(ee_markers, axis=0) if ee_markers else np.array([0, 0, 0])
        
        # 转换单位 mm -> m
        return base_mocap_pos / 1000.0, ee_mocap_pos / 1000.0
    
    def objective_function(self, params):
        """
        优化目标函数
        
        Args:
            params: 优化参数 [joint_offsets(5), base_transform(6), ee_transform(6)]
                   变换参数格式: [tx, ty, tz, rx, ry, rz] (位移+欧拉角)
        
        Returns:
            residuals: 残差数组
        """
        # 更新进度
        self.iteration_count += 1
        
        # 强制终止机制
        if hasattr(self, 'max_iterations') and self.iteration_count >= self.max_iterations:
            print(f"\n达到最大迭代次数 {self.max_iterations}，强制终止优化")
            return np.array([1e10])  # 返回大残差值强制终止
        
        if self.progress_bar is not None:
            elapsed_time = time.time() - self.start_time
            self.progress_bar.set_description(f"Iter {self.iteration_count} | Time: {elapsed_time:.1f}s")
            if self.iteration_count <= self.max_iterations:
                self.progress_bar.update(1)
        
        # 解析参数
        self.joint_offsets = params[:5]
        
        # 基座变换参数
        base_trans = params[5:8]
        base_rot = params[8:11]
        self.base_transform = self._pose_to_matrix(base_trans, base_rot)
        
        # 末端执行器变换参数
        ee_trans = params[11:14]
        ee_rot = params[14:17]
        self.ee_transform = self._pose_to_matrix(ee_trans, ee_rot)
        
        residuals = []
        
        # 使用数据子集进行优化 (每50个点取1个，大幅提高速度)
        sample_step = max(1, len(self.data) // 200)  # 最多使用00个数据点
        if self.iteration_count == 1:  # 只在第一次迭代时打印
            print(f"使用采样步长: {sample_step}, 总数据点: {len(self.data)} -> {len(self.data)//sample_step}")
        
        for idx in range(0, len(self.data), sample_step):
            row = self.data.iloc[idx]
            
            # 获取关节角度
            joint_angles = np.array([
                row['m1_rad'], row['m2_rad'], row['m3_rad'], 
                row['m4_rad'], row['m5_rad']
            ])
            
            # 正向运动学计算
            base_fk, ee_fk = self.forward_kinematics(joint_angles)
            
            # 动作捕捉数据
            base_mocap, ee_mocap = self.extract_mocap_positions(row)
            
            # 计算残差
            if not np.any(np.isnan(base_mocap)):
                residuals.extend(base_fk - base_mocap)
            
            if not np.any(np.isnan(ee_mocap)):
                residuals.extend(ee_fk - ee_mocap)
        
        return np.array(residuals)
    
    def _pose_to_matrix(self, translation, rotation):
        """
        将位移和旋转转换为4x4变换矩阵
        
        Args:
            translation: [tx, ty, tz]
            rotation: [rx, ry, rz] (欧拉角)
        
        Returns:
            4x4变换矩阵
        """
        T = np.eye(4)
        T[:3, 3] = translation
        T[:3, :3] = R.from_euler('xyz', rotation).as_matrix()
        return T
    
    def identify_initial_pose(self):
        """
        执行初始位置辨识
        """
        print("=== 开始初始位置辨识 ===")
        
        # 初始化进度跟踪
        self.iteration_count = 0
        self.start_time = time.time()
        
        # 初始参数估计
        initial_params = np.zeros(17)  # 5个关节偏移 + 6个基座变换 + 6个末端变换
        
        # 设置参数边界
        bounds = [
            # 关节偏移边界 (±30度)
            (-np.pi/6, np.pi/6), (-np.pi/6, np.pi/6), (-np.pi/6, np.pi/6),
            (-np.pi/6, np.pi/6), (-np.pi/6, np.pi/6),
            # 基座位移边界 (±1m)
            (-1, 1), (-1, 1), (-1, 1),
            # 基座旋转边界 (±π)
            (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),
            # 末端位移边界 (±0.5m)
            (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5),
            # 末端旋转边界 (±π)
            (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)
        ]
        
        print("开始优化...")
        print(f"数据集大小: {len(self.data)} 行")
        
        # 初始化进度条
        max_iterations = 500  # 减少最大迭代次数
        self.max_iterations = max_iterations
        self.progress_bar = tqdm(total=max_iterations, desc="Optimization Progress", 
                                unit="iter", ncols=100)
        
        try:
            # 使用最小二乘法优化，添加更严格的终止条件
            result = least_squares(
                self.objective_function, 
                initial_params,
                bounds=([b[0] for b in bounds], [b[1] for b in bounds]),
                method='trf',
                max_nfev=max_iterations,
                ftol=1e-6,  # 函数值容差
                xtol=1e-6,  # 参数变化容差
                gtol=1e-6   # 梯度容差
            )
        finally:
            # 关闭进度条
            if self.progress_bar is not None:
                self.progress_bar.close()
        
        if result.success:
            print("优化成功!")
            print(f"最终残差: {np.linalg.norm(result.fun):.6f}")
            
            # 保存结果
            self.joint_offsets = result.x[:5]
            base_params = result.x[5:11]
            ee_params = result.x[11:17]
            
            self.base_transform = self._pose_to_matrix(base_params[:3], base_params[3:])
            self.ee_transform = self._pose_to_matrix(ee_params[:3], ee_params[3:])
            
            self.print_results()
            self.save_results()
            
        else:
            print("优化失败:", result.message)
    
    def print_results(self):
        """打印辨识结果"""
        print("\n=== 辨识结果 ===")
        print("关节零点偏移 (度):")
        for i, offset in enumerate(self.joint_offsets):
            print(f"  关节 {i+1}: {np.degrees(offset):.3f}°")
        
        print("\n基座坐标变换:")
        print(f"  位移: {self.base_transform[:3, 3]}")
        print(f"  旋转矩阵:\n{self.base_transform[:3, :3]}")
        
        print("\n末端执行器坐标变换:")
        print(f"  位移: {self.ee_transform[:3, 3]}")
        print(f"  旋转矩阵:\n{self.ee_transform[:3, :3]}")
    
    def save_results(self, filename='initial_pose_calibration.json'):
        """保存辨识结果到文件"""
        results = {
            'joint_offsets_deg': self.joint_offsets.tolist(),
            'joint_offsets_rad': np.degrees(self.joint_offsets).tolist(),
            'base_transform': self.base_transform.tolist(),
            'ee_transform': self.ee_transform.tolist(),
            'dh_parameters': self.dh_params
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n结果已保存到: {filename}")
    
    def validate_calibration(self):
        """验证标定结果"""
        print("\n=== 标定验证 ===")
        
        errors_base = []
        errors_ee = []
        
        # 使用所有数据进行验证
        for idx in range(0, len(self.data), 50):  # 每50个点取1个
            row = self.data.iloc[idx]
            
            joint_angles = np.array([
                row['m1_rad'], row['m2_rad'], row['m3_rad'], 
                row['m4_rad'], row['m5_rad']
            ])
            
            base_fk, ee_fk = self.forward_kinematics(joint_angles)
            base_mocap, ee_mocap = self.extract_mocap_positions(row)
            
            if not np.any(np.isnan(base_mocap)):
                error_base = np.linalg.norm(base_fk - base_mocap)
                errors_base.append(error_base)
            
            if not np.any(np.isnan(ee_mocap)):
                error_ee = np.linalg.norm(ee_fk - ee_mocap)
                errors_ee.append(error_ee)
        
        if errors_base:
            print(f"基座位置误差统计:")
            print(f"  平均误差: {np.mean(errors_base)*1000:.2f} mm")
            print(f"  最大误差: {np.max(errors_base)*1000:.2f} mm")
            print(f"  标准差: {np.std(errors_base)*1000:.2f} mm")
        
        if errors_ee:
            print(f"末端执行器位置误差统计:")
            print(f"  平均误差: {np.mean(errors_ee)*1000:.2f} mm")
            print(f"  最大误差: {np.max(errors_ee)*1000:.2f} mm")
            print(f"  标准差: {np.std(errors_ee)*1000:.2f} mm")
    
    def plot_comparison(self):
        """绘制对比图"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 采样数据进行可视化
        sample_indices = range(0, len(self.data), 100)
        
        base_fk_positions = []
        base_mocap_positions = []
        ee_fk_positions = []
        ee_mocap_positions = []
        
        for idx in sample_indices:
            row = self.data.iloc[idx]
            
            joint_angles = np.array([
                row['m1_rad'], row['m2_rad'], row['m3_rad'], 
                row['m4_rad'], row['m5_rad']
            ])
            
            base_fk, ee_fk = self.forward_kinematics(joint_angles)
            base_mocap, ee_mocap = self.extract_mocap_positions(row)
            
            if not np.any(np.isnan(base_mocap)):
                base_fk_positions.append(base_fk)
                base_mocap_positions.append(base_mocap)
            
            if not np.any(np.isnan(ee_mocap)):
                ee_fk_positions.append(ee_fk)
                ee_mocap_positions.append(ee_mocap)
        
        base_fk_positions = np.array(base_fk_positions)
        base_mocap_positions = np.array(base_mocap_positions)
        ee_fk_positions = np.array(ee_fk_positions)
        ee_mocap_positions = np.array(ee_mocap_positions)
        
        # 基座位置对比
        for i, axis in enumerate(['X', 'Y', 'Z']):
            axes[0, i].plot(base_fk_positions[:, i], 'b-', label='Forward Kinematics', alpha=0.7)
            axes[0, i].plot(base_mocap_positions[:, i], 'r--', label='Motion Capture', alpha=0.7)
            axes[0, i].set_title(f'Base Position {axis}')
            axes[0, i].set_ylabel('Position (m)')
            axes[0, i].legend()
            axes[0, i].grid(True)
        
        # 末端执行器位置对比
        for i, axis in enumerate(['X', 'Y', 'Z']):
            axes[1, i].plot(ee_fk_positions[:, i], 'b-', label='Forward Kinematics', alpha=0.7)
            axes[1, i].plot(ee_mocap_positions[:, i], 'r--', label='Motion Capture', alpha=0.7)
            axes[1, i].set_title(f'End Effector Position {axis}')
            axes[1, i].set_ylabel('Position (m)')
            axes[1, i].set_xlabel('Sample Index')
            axes[1, i].legend()
            axes[1, i].grid(True)
        
        plt.tight_layout()
        plt.savefig('/Users/lr-2002/project/instantcreation/IC_arm_control/pose_calibration_comparison.png', dpi=300)
        plt.show()

def main():
    """主函数"""
    # 加载同步数据
    sync_data_path = '/Users/lr-2002/project/instantcreation/IC_arm_control/synchronized_data.csv'
    
    try:
        identifier = InitialPoseIdentifier(sync_data_path)
        
        print(f"加载同步数据: {len(identifier.data)} 行")
        
        # 执行初始位置辨识
        identifier.identify_initial_pose()
        
        # 验证标定结果
        identifier.validate_calibration()
        
        # 绘制对比图
        identifier.plot_comparison()
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
