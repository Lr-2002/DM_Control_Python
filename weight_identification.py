#!/usr/bin/env python3
"""
URDF重量辨识工具
基于动力学方程和实际测量数据进行参数辨识
"""

import numpy as np
import matplotlib.pyplot as plt
from IC_ARM import ICARM
import time
import json
from scipy.optimize import minimize
from datetime import datetime

class WeightIdentifier:
    def __init__(self, urdf_path=None):
        """
        初始化重量辨识器
        
        Args:
            urdf_path: URDF文件路径
        """
        self.ic_arm = ICARM()
        self.urdf_path = urdf_path
        self.measurement_data = []
        
        # 初始质量估计值（从URDF读取或设置默认值）
        self.initial_masses = {
            'base_link': 0.297,  # kg
            'l1': 0.185,
            'l2': 0.156, 
            'l3': 0.089,
            'l4': 0.067,
            'l5': 0.045
        }
        
    def collect_static_data(self, positions_list, hold_time=3.0):
        """
        收集静态平衡数据
        
        Args:
            positions_list: 关节位置列表，每个位置是[j1,j2,j3,j4,j5]
            hold_time: 每个位置保持时间（秒）
        """
        print("开始收集静态平衡数据...")
        self.ic_arm.enable_all_motors()
        
        for i, target_pos in enumerate(positions_list):
            print(f"移动到位置 {i+1}/{len(positions_list)}: {target_pos}")
            
            # 移动到目标位置
            for j, pos in enumerate(target_pos):
                motor_id = j + 1
                self.ic_arm.mc.set_angle(motor_id, pos, 50)  # 50度/秒速度
            
            # 等待到达位置
            time.sleep(2.0)
            
            # 收集数据
            start_time = time.time()
            torques = []
            positions = []
            
            while time.time() - start_time < hold_time:
                # 读取当前位置和力矩
                current_pos = self.ic_arm.get_positions_only()
                current_torques = self.get_motor_torques()
                
                positions.append(current_pos)
                torques.append(current_torques)
                time.sleep(0.1)
            
            # 计算平均值
            avg_pos = np.mean(positions, axis=0)
            avg_torques = np.mean(torques, axis=0)
            
            self.measurement_data.append({
                'position': avg_pos.tolist(),
                'torques': avg_torques.tolist(),
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"位置: {avg_pos}")
            print(f"力矩: {avg_torques}")
            print("-" * 50)
        
        self.ic_arm.disable_all_motors()
        print("数据收集完成！")
        
    def get_motor_torques(self):
        """
        获取电机力矩（需要根据你的硬件接口调整）
        这里使用电流估算力矩
        """
        torques = []
        for motor_id in range(1, 6):
            try:
                # 刷新电机状态
                self.ic_arm.mc.refresh_motor_status(motor_id)
                # 获取电流（mA）
                current = self.ic_arm.mc.get_motor_current(motor_id)
                # 简单的电流到力矩转换（需要根据电机规格调整）
                torque = current * 0.001 * 0.1  # 假设转换系数
                torques.append(torque)
            except:
                torques.append(0.0)
        
        return np.array(torques)
    
    def gravity_torque_model(self, positions, masses):
        """
        重力力矩模型
        基于机械臂几何和质量计算重力力矩
        
        Args:
            positions: 关节角度 [j1, j2, j3, j4, j5]
            masses: 各link质量 [m1, m2, m3, m4, m5]
        
        Returns:
            gravity_torques: 重力力矩 [tau1, tau2, tau3, tau4, tau5]
        """
        q = np.array(positions)
        m = np.array(masses)
        g = 9.81  # 重力加速度
        
        # 简化的DH参数（需要根据实际机械臂调整）
        # 这里使用近似值，实际应该从URDF或CAD模型获取
        L = [0.1, 0.15, 0.12, 0.08, 0.05]  # 各段长度
        
        # 计算重力力矩（简化模型）
        tau = np.zeros(5)
        
        # 关节1（基座旋转，通常重力力矩为0）
        tau[0] = 0
        
        # 关节2-5的重力力矩计算
        for i in range(1, 5):
            # 累积质量效应
            for j in range(i, 5):
                # 计算质心到关节的距离
                r = sum(L[k] for k in range(i, j+1))
                # 重力力矩分量
                tau[i] += m[j] * g * r * np.cos(sum(q[k] for k in range(1, i+1)))
        
        return tau
    
    def objective_function(self, masses, measurement_data):
        """
        目标函数：最小化模型预测和实际测量的差异
        """
        total_error = 0
        
        for data in measurement_data:
            positions = data['position']
            measured_torques = np.array(data['torques'])
            
            # 计算模型预测的重力力矩
            predicted_torques = self.gravity_torque_model(positions, masses)
            
            # 计算误差
            error = np.sum((predicted_torques - measured_torques) ** 2)
            total_error += error
        
        return total_error
    
    def identify_masses(self):
        """
        使用优化算法辨识质量参数
        """
        if not self.measurement_data:
            print("错误：没有测量数据！请先收集数据。")
            return None
        
        print("开始质量参数辨识...")
        
        # 初始猜测值
        initial_guess = list(self.initial_masses.values())[1:]  # 排除base_link
        
        # 质量约束（正值，合理范围）
        bounds = [(0.01, 2.0) for _ in range(5)]  # 每个link质量在0.01-2.0kg之间
        
        # 优化
        result = minimize(
            self.objective_function,
            initial_guess,
            args=(self.measurement_data,),
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        if result.success:
            identified_masses = result.x
            print("辨识成功！")
            print("辨识结果：")
            link_names = ['l1', 'l2', 'l3', 'l4', 'l5']
            for i, (name, mass) in enumerate(zip(link_names, identified_masses)):
                print(f"{name}: {mass:.4f} kg (原值: {list(self.initial_masses.values())[i+1]:.4f} kg)")
            
            return {name: mass for name, mass in zip(link_names, identified_masses)}
        else:
            print("辨识失败：", result.message)
            return None
    
    def save_data(self, filename="weight_identification_data.json"):
        """保存测量数据"""
        with open(filename, 'w') as f:
            json.dump(self.measurement_data, f, indent=2)
        print(f"数据已保存到 {filename}")
    
    def load_data(self, filename="weight_identification_data.json"):
        """加载测量数据"""
        try:
            with open(filename, 'r') as f:
                self.measurement_data = json.load(f)
            print(f"数据已从 {filename} 加载")
            return True
        except FileNotFoundError:
            print(f"文件 {filename} 不存在")
            return False
    
    def plot_results(self):
        """绘制结果对比图"""
        if not self.measurement_data:
            print("没有数据可绘制")
            return
        
        positions = [data['position'] for data in self.measurement_data]
        measured_torques = [data['torques'] for data in self.measurement_data]
        
        plt.figure(figsize=(12, 8))
        
        for joint in range(5):
            plt.subplot(2, 3, joint + 1)
            torques_joint = [torques[joint] for torques in measured_torques]
            plt.plot(torques_joint, 'o-', label=f'关节{joint+1}')
            plt.title(f'关节{joint+1}力矩')
            plt.xlabel('测量点')
            plt.ylabel('力矩 (Nm)')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    """主函数：演示重量辨识流程"""
    identifier = WeightIdentifier()
    
    print("=== URDF重量辨识工具 ===")
    print("1. 收集新数据")
    print("2. 加载已有数据")
    print("3. 进行参数辨识")
    
    choice = input("请选择操作 (1/2/3): ")
    
    if choice == '1':
        # 定义测试位置（关节角度，单位：度）
        test_positions = [
            [0, 0, 0, 0, 0],           # 零位
            [0, 30, 0, 0, 0],          # 关节2抬起
            [0, 0, 45, 0, 0],          # 关节3抬起
            [0, 30, 45, 0, 0],         # 组合位置1
            [0, -30, 0, 0, 0],         # 关节2下压
            [0, 0, -45, 0, 0],         # 关节3下压
            [45, 0, 0, 0, 0],          # 基座旋转
            [0, 45, 90, 0, 0],         # 伸展位置
        ]
        
        identifier.collect_static_data(test_positions, hold_time=3.0)
        identifier.save_data()
        
    elif choice == '2':
        if identifier.load_data():
            identifier.plot_results()
        
    elif choice == '3':
        if identifier.load_data():
            identified_masses = identifier.identify_masses()
            if identified_masses:
                print("\n建议更新URDF文件中的质量参数：")
                for link, mass in identified_masses.items():
                    print(f'<mass value="{mass:.6f}" />  <!-- {link} -->')

if __name__ == "__main__":
    main()
