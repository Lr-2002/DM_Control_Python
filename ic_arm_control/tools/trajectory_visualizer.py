#!/usr/bin/env python3
"""
轨迹可视化工具
用于可视化和分析生成的轨迹文件
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

class TrajectoryVisualizer:
    """轨迹可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        self.motor_names = ['m1', 'm2', 'm3', 'm4', 'm5']
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def load_trajectory(self, filename: str) -> Dict:
        """加载轨迹文件"""
        try:
            with open(filename, 'r') as f:
                trajectory = json.load(f)
            
            # 转换为numpy数组
            trajectory['time'] = np.array(trajectory['time'])
            trajectory['positions'] = np.array(trajectory['positions'])
            trajectory['velocities'] = np.array(trajectory['velocities'])
            trajectory['accelerations'] = np.array(trajectory['accelerations'])
            
            print(f"✓ 成功加载轨迹: {filename}")
            print(f"  时间范围: {trajectory['time'][0]:.3f}s - {trajectory['time'][-1]:.3f}s")
            print(f"  数据点数: {len(trajectory['time'])}")
            
            return trajectory
            
        except Exception as e:
            print(f"✗ 加载轨迹失败: {filename}, 错误: {e}")
            return None
    
    def plot_single_trajectory(self, trajectory: Dict, title: str = "轨迹分析", 
                             save_path: Optional[str] = None, show_motors: Optional[List[int]] = None):
        """
        绘制单个轨迹的完整分析图
        
        Args:
            trajectory: 轨迹数据字典
            title: 图表标题
            save_path: 保存路径（可选）
            show_motors: 要显示的电机列表（1-5），None表示显示所有
        """
        if trajectory is None:
            print("轨迹数据为空，无法绘制")
            return
        
        t = trajectory['time']
        positions = trajectory['positions']
        velocities = trajectory['velocities']
        accelerations = trajectory['accelerations']
        
        # 确定要显示的电机
        num_motors = positions.shape[1]  # 获取实际电机数量
        if show_motors is None:
            # 自动检测有运动的电机
            show_motors = []
            for i in range(num_motors):
                if np.max(np.abs(positions[:, i])) > 1e-6:  # 有明显运动
                    show_motors.append(i)
        else:
            # 转换为0-based索引，并限制在实际电机数量范围内
            show_motors = [m - 1 for m in show_motors if 1 <= m <= num_motors]
        
        if not show_motors:
            print("没有检测到运动的电机")
            return
        
        # 创建子图
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 位置图
        ax1 = axes[0]
        for i in show_motors:
            motor_name = self.motor_names[i]
            color = self.colors[i % len(self.colors)]
            ax1.plot(t, np.degrees(positions[:, i]), 
                    label=f'{motor_name}', color=color, linewidth=2)
        
        ax1.set_ylabel('位置 (°)', fontsize=12)
        ax1.set_title('关节位置', fontsize=14)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 速度图
        ax2 = axes[1]
        for i in show_motors:
            motor_name = self.motor_names[i]
            color = self.colors[i % len(self.colors)]
            ax2.plot(t, np.degrees(velocities[:, i]), 
                    label=f'{motor_name}', color=color, linewidth=2)
        
        ax2.set_ylabel('速度 (°/s)', fontsize=12)
        ax2.set_title('关节速度', fontsize=14)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 加速度图
        ax3 = axes[2]
        for i in show_motors:
            motor_name = self.motor_names[i]
            color = self.colors[i % len(self.colors)]
            ax3.plot(t, np.degrees(accelerations[:, i]), 
                    label=f'{motor_name}', color=color, linewidth=2)
        
        ax3.set_xlabel('时间 (s)', fontsize=12)
        ax3.set_ylabel('加速度 (°/s²)', fontsize=12)
        ax3.set_title('关节加速度', fontsize=14)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图表已保存到: {save_path}")
        
        plt.show()
        
    def plot_motor_detail(self, trajectory: Dict, motor_id: int, title: str = None,
                         save_path: Optional[str] = None):
        """
        绘制单个电机的详细分析图
        
        Args:
            trajectory: 轨迹数据字典
            motor_id: 电机ID (1-5)
            title: 图表标题
            save_path: 保存路径（可选）
        """
        if trajectory is None:
            print("轨迹数据为空，无法绘制")
            return
        
        if not (1 <= motor_id <= 5):
            print(f"电机ID必须在1-5之间，当前: {motor_id}")
            return
        
        motor_idx = motor_id - 1
        motor_name = self.motor_names[motor_idx]
        
        if title is None:
            title = f"电机 {motor_name} 详细轨迹分析"
        
        t = trajectory['time']
        position = np.degrees(trajectory['positions'][:, motor_idx])
        velocity = np.degrees(trajectory['velocities'][:, motor_idx])
        acceleration = np.degrees(trajectory['accelerations'][:, motor_idx])
        
        # 检查是否有运动
        if np.max(np.abs(trajectory['positions'][:, motor_idx])) < 1e-6:
            print(f"警告: 电机 {motor_name} 没有明显运动")
        
        # 创建子图
        fig, axes = plt.subplots(4, 1, figsize=(14, 16))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        color = self.colors[motor_idx % len(self.colors)]
        
        # 位置图
        ax1 = axes[0]
        ax1.plot(t, position, color=color, linewidth=2.5)
        ax1.set_ylabel('位置 (°)', fontsize=12)
        ax1.set_title(f'{motor_name} 位置轨迹', fontsize=14)
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
        ax2.set_title(f'{motor_name} 速度轨迹', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        vel_stats = f"最大速度: {velocity.max():.1f}°/s, 最小速度: {velocity.min():.1f}°/s"
        ax2.text(0.02, 0.98, vel_stats, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 加速度图
        ax3 = axes[2]
        ax3.plot(t, acceleration, color=color, linewidth=2.5)
        ax3.set_ylabel('加速度 (°/s²)', fontsize=12)
        ax3.set_title(f'{motor_name} 加速度轨迹', fontsize=14)
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
        ax4.set_title(f'{motor_name} 相空间图 (位置 vs 速度)', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图表已保存到: {save_path}")
        
        plt.show()
        
    def compare_trajectories(self, trajectory_files: List[str], labels: Optional[List[str]] = None,
                           motor_id: int = 2, save_path: Optional[str] = None):
        """
        比较多个轨迹文件
        
        Args:
            trajectory_files: 轨迹文件列表
            labels: 轨迹标签列表
            motor_id: 要比较的电机ID (1-5)
            save_path: 保存路径（可选）
        """
        if not (1 <= motor_id <= 5):
            print(f"电机ID必须在1-5之间，当前: {motor_id}")
            return
        
        motor_idx = motor_id - 1
        motor_name = self.motor_names[motor_idx]
        
        if labels is None:
            labels = [f"轨迹{i+1}" for i in range(len(trajectory_files))]
        
        # 加载所有轨迹
        trajectories = []
        for i, filename in enumerate(trajectory_files):
            traj = self.load_trajectory(filename)
            if traj is not None:
                trajectories.append((traj, labels[i]))
        
        if not trajectories:
            print("没有成功加载任何轨迹")
            return
        
        # 创建比较图
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle(f'电机 {motor_name} 轨迹比较', fontsize=16, fontweight='bold')
        
        for i, (traj, label) in enumerate(trajectories):
            t = traj['time']
            position = np.degrees(traj['positions'][:, motor_idx])
            velocity = np.degrees(traj['velocities'][:, motor_idx])
            acceleration = np.degrees(traj['accelerations'][:, motor_idx])
            
            color = self.colors[i % len(self.colors)]
            
            # 位置比较
            axes[0].plot(t, position, label=label, color=color, linewidth=2)
            
            # 速度比较
            axes[1].plot(t, velocity, label=label, color=color, linewidth=2)
            
            # 加速度比较
            axes[2].plot(t, acceleration, label=label, color=color, linewidth=2)
        
        # 设置图表属性
        axes[0].set_ylabel('位置 (°)', fontsize=12)
        axes[0].set_title('位置比较', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_ylabel('速度 (°/s)', fontsize=12)
        axes[1].set_title('速度比较', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_xlabel('时间 (s)', fontsize=12)
        axes[2].set_ylabel('加速度 (°/s²)', fontsize=12)
        axes[2].set_title('加速度比较', fontsize=14)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 比较图表已保存到: {save_path}")
        
        plt.show()
        
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
        motor_name = self.motor_names[motor_idx]
        
        t = trajectory['time']
        position = trajectory['positions'][:, motor_idx]
        velocity = trajectory['velocities'][:, motor_idx]
        acceleration = trajectory['accelerations'][:, motor_idx]
        
        print(f"\n=== 电机 {motor_name} 连续性分析 ===")
        
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
    """主函数 - 演示可视化功能"""
    print("=== 轨迹可视化工具 ===\n")
    
    visualizer = TrajectoryVisualizer()
    
    # 查找可用的轨迹文件
    trajectory_files = []
    for file in os.listdir('.'):
        if file.startswith('trajectory_') and file.endswith('.json'):
            trajectory_files.append(file)
    
    if not trajectory_files:
        print("未找到轨迹文件")
        return
    
    print(f"找到 {len(trajectory_files)} 个轨迹文件:")
    for i, file in enumerate(trajectory_files):
        print(f"  {i+1}. {file}")
    
    # # 可视化2号电机的单独轨迹
    # motor2_file = 'trajectory_motor_2_single.json'
    # if motor2_file in trajectory_files:
    #     print(f"\n可视化 {motor2_file}...")
    #     trajectory = visualizer.load_trajectory(motor2_file)
    #     if trajectory:
    #         # 绘制详细分析
    #         visualizer.plot_motor_detail(trajectory, motor_id=2, 
    #                                    save_path='motor_2_detailed_analysis.png')
            
    #         # 分析连续性
    #         visualizer.analyze_continuity(trajectory, motor_id=2)
    
    # 可视化序列轨迹
    sequential_file = 'trajectory_random_bandlimited.json'
    if sequential_file in trajectory_files:
        print(f"\n可视化 {sequential_file}...")
        trajectory = visualizer.load_trajectory(sequential_file)
        if trajectory:
            visualizer.plot_single_trajectory(trajectory, 
                                            title="序列轨迹 (5-4-3-2)",
                                            save_path='sequential_trajectory_analysis.png')


if __name__ == "__main__":
    main()
