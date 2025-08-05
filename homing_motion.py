#!/usr/bin/env python3
"""
回零位驱动代码
使用插值在1秒内平滑回到零点位置
"""

from IC_ARM import ICARM
import time
import math
import numpy as np

class HomingMotion:
    def __init__(self):
        """初始化回零运动控制器"""
        self.ic_arm = ICARM()
        self.target_positions = [0.0, 0.0, 0.0, 0.0, 0.0]  # 目标零点位置（度）
        
    def get_current_positions_deg(self):
        """获取当前位置（度）"""
        positions = self.ic_arm.get_positions_only()
        return [positions[f'm{i+1}']['deg'] for i in range(5)]
    
    def linear_interpolation(self, start_pos, end_pos, duration, update_rate=50):
        """
        线性插值生成轨迹点
        
        Args:
            start_pos: 起始位置列表 [j1, j2, j3, j4, j5] (度)
            end_pos: 结束位置列表 [j1, j2, j3, j4, j5] (度)
            duration: 运动时间 (秒)
            update_rate: 更新频率 (Hz)
        
        Returns:
            trajectory: 轨迹点列表，每个点是 [j1, j2, j3, j4, j5, time]
        """
        num_points = int(duration * update_rate)
        trajectory = []
        
        for i in range(num_points + 1):
            t = i / num_points  # 归一化时间 [0, 1]
            
            # 线性插值
            current_pos = []
            for j in range(5):
                pos = start_pos[j] + t * (end_pos[j] - start_pos[j])
                current_pos.append(pos)
            
            time_stamp = t * duration
            trajectory.append(current_pos + [time_stamp])
        
        return trajectory
    
    def smooth_interpolation(self, start_pos, end_pos, duration, update_rate=50):
        """
        平滑插值（S曲线）生成轨迹点
        
        Args:
            start_pos: 起始位置列表 [j1, j2, j3, j4, j5] (度)
            end_pos: 结束位置列表 [j1, j2, j3, j4, j5] (度)
            duration: 运动时间 (秒)
            update_rate: 更新频率 (Hz)
        
        Returns:
            trajectory: 轨迹点列表，每个点是 [j1, j2, j3, j4, j5, time]
        """
        num_points = int(duration * update_rate)
        trajectory = []
        
        for i in range(num_points + 1):
            t = i / num_points  # 归一化时间 [0, 1]
            
            # S曲线插值 (3次多项式: 3t² - 2t³)
            s = 3 * t**2 - 2 * t**3
            
            # 应用插值
            current_pos = []
            for j in range(5):
                pos = start_pos[j] + s * (end_pos[j] - start_pos[j])
                current_pos.append(pos)
            
            time_stamp = t * duration
            trajectory.append(current_pos + [time_stamp])
        
        return trajectory
    
    def execute_trajectory(self, trajectory, verbose=True):
        """
        执行轨迹
        
        Args:
            trajectory: 轨迹点列表
            verbose: 是否打印详细信息
        """
        print("开始执行回零轨迹...")
        
        # 启用所有电机
        self.ic_arm.enable_all_motors()
        
        start_time = time.time()
        
        try:
            for i, point in enumerate(trajectory):
                target_positions = point[:5]  # 前5个是关节位置
                target_time = point[5]        # 第6个是时间戳
                
                # 等待到达目标时间
                while (time.time() - start_time) < target_time:
                    time.sleep(0.001)  # 1ms精度
                
                # 发送位置命令到各个电机
                for motor_idx, target_deg in enumerate(target_positions):
                    motor_id = motor_idx + 1
                    try:
                        # 设置角度，速度设为较高值以确保跟踪
                        self.ic_arm.mc.set_angle(motor_id, target_deg, 200)  # 200度/秒
                    except Exception as e:
                        if verbose:
                            print(f"警告: 电机 m{motor_id} 设置失败: {e}")
                
                # 打印进度
                if verbose and i % 10 == 0:  # 每10个点打印一次
                    progress = (i / len(trajectory)) * 100
                    current_pos = self.get_current_positions_deg()
                    print(f"进度: {progress:.1f}% | 目标: {[f'{p:.1f}' for p in target_positions]} | "
                          f"实际: {[f'{p:.1f}' for p in current_pos]}")
        
        except KeyboardInterrupt:
            print("\n运动被中断")
        
        finally:
            # 安全停止
            print("停止所有电机...")
            self.ic_arm.disable_all_motors()
        
        # 验证最终位置
        final_pos = self.get_current_positions_deg()
        print(f"\n回零完成!")
        print(f"最终位置: {[f'{p:.2f}°' for p in final_pos]}")
        
        # 计算误差
        errors = [abs(final_pos[i] - self.target_positions[i]) for i in range(5)]
        print(f"位置误差: {[f'{e:.2f}°' for e in errors]}")
        max_error = max(errors)
        print(f"最大误差: {max_error:.2f}°")
        
        return max_error < 2.0  # 如果最大误差小于2度认为成功
    
    def home_to_zero(self, duration=1.0, interpolation_type='smooth'):
        """
        回零主函数
        
        Args:
            duration: 运动时间 (秒)
            interpolation_type: 插值类型 ('linear' 或 'smooth')
        """
        print("=== IC ARM 回零运动 ===")
        
        # 获取当前位置
        print("读取当前位置...")
        current_pos = self.get_current_positions_deg()
        print(f"当前位置: {[f'{p:.2f}°' for p in current_pos]}")
        print(f"目标位置: {[f'{p:.2f}°' for p in self.target_positions]}")
        
        # 计算运动距离
        distances = [abs(current_pos[i] - self.target_positions[i]) for i in range(5)]
        max_distance = max(distances)
        print(f"最大运动距离: {max_distance:.2f}°")
        
        if max_distance < 1.0:
            print("已经接近零点位置，无需回零")
            return True
        
        # 生成轨迹
        print(f"生成{interpolation_type}插值轨迹，时长{duration}秒...")
        if interpolation_type == 'linear':
            trajectory = self.linear_interpolation(current_pos, self.target_positions, duration)
        else:
            trajectory = self.smooth_interpolation(current_pos, self.target_positions, duration)
        
        print(f"轨迹点数: {len(trajectory)}")
        
        # 执行轨迹
        success = self.execute_trajectory(trajectory)
        
        if success:
            print("✓ 回零成功!")
        else:
            print("✗ 回零精度不足，可能需要调整参数")
        
        return success
    
    def close(self):
        """关闭连接"""
        self.ic_arm.close()

def main():
    """主函数"""
    homing = HomingMotion()
    
    try:
        print("选择插值类型:")
        print("1. 线性插值 (linear)")
        print("2. 平滑插值 (smooth) - 推荐")
        
        choice = input("请选择 (1/2): ")
        
        if choice == '1':
            interpolation_type = 'linear'
        else:
            interpolation_type = 'smooth'
        
        # 询问运动时间
        duration_input = input("请输入运动时间 (秒，默认1.0): ")
        try:
            duration = float(duration_input) if duration_input else 1.0
        except ValueError:
            duration = 1.0
        
        print(f"\n使用{interpolation_type}插值，运动时间{duration}秒")
        
        # 执行回零
        success = homing.home_to_zero(duration=duration, interpolation_type=interpolation_type)
        
        if not success:
            retry = input("\n是否重试? (y/N): ")
            if retry.lower() == 'y':
                homing.home_to_zero(duration=duration, interpolation_type=interpolation_type)
    
    except Exception as e:
        print(f"错误: {e}")
    
    finally:
        homing.close()

if __name__ == "__main__":
    main()
