#!/usr/bin/env python3
"""
导出运动数据脚本
从CSV文件中读取关节位置、速度、加速度数据，格式化为动力学回归器所需的格式
"""

import numpy as np
import pandas as pd
from regressor import CalcDynamics

def load_motion_data(csv_file):
    """
    从CSV文件加载运动数据
    
    Args:
        csv_file: CSV文件路径
        
    Returns:
        q, dq, ddq: 位置、速度、加速度数组
    """
    print(f"正在加载运动数据: {csv_file}")
    
    # 读取CSV文件
    data = pd.read_csv(csv_file)
    
    print(f"数据形状: {data.shape}")
    print(f"列名: {list(data.columns)}")
    
    # 提取关节数据 - 假设列名包含position, velocity, acceleration
    # 根据实际CSV结构调整列名
    position_cols = [col for col in data.columns if ('pos' in col.lower() or 'position' in col.lower()) and 'actual'  in col.lower()]
    velocity_cols = [col for col in data.columns if ('vel' in col.lower() or 'velocity' in col.lower()) and 'actual' in col.lower()]
    acceleration_cols = [col for col in data.columns if ('acc' in col.lower() or 'acceleration' in col.lower()) and 'actual' in col.lower()]
    torque_cols = [col for col in data.columns if ('torque' in col.lower() or 'torq' in col.lower())]
    print(f"位置列: {position_cols}")
    print(f"速度列: {velocity_cols}")
    print(f"加速度列: {acceleration_cols}")
    print(f"力矩列: {torque_cols}")
    
    # 如果没有找到标准列名，尝试按索引提取（假设前5列是位置，接下来5列是速度，再接下来5列是加速度）
    if not position_cols or not velocity_cols or not acceleration_cols and not torque_cols:
        print("未找到标准列名，尝试按索引提取数据...")
        num_joints = 5
        
        # 假设数据结构：时间戳 + 5个关节位置 + 5个关节速度 + 5个关节加速度
        if data.shape[1] >= 16:  # 至少需要16列（时间 + 5*3）
            q = data.iloc[:, 1:6].values  # 位置：列1-5
            dq = data.iloc[:, 6:11].values  # 速度：列6-10
            ddq = data.iloc[:, 11:16].values  # 加速度：列11-15
            tau = data.iloc[:, 16:21].values  # 力矩：列16-21
        else:
            raise ValueError(f"CSV文件列数不足，期望至少16列，实际{data.shape[1]}列")
    else:
        # 使用找到的列名
        q = data[position_cols].values
        dq = data[velocity_cols].values
        ddq = data[acceleration_cols].values
        tau = data[torque_cols].values
    
    print(f"位置数据形状: {q.shape}")
    print(f"速度数据形状: {dq.shape}")
    print(f"加速度数据形状: {ddq.shape}")
    print(f"力矩数据形状: {tau.shape}")
    
    # 确保数据类型为float64
    q = q.astype(np.float64)
    dq = dq.astype(np.float64)
    ddq = ddq.astype(np.float64)
    tau = tau.astype(np.float64)
    
    return q, dq, ddq, tau

def export_regressor_data(csv_file, output_file=None):
    """
    导出回归器数据
    
    Args:
        csv_file: 输入CSV文件
        output_file: 输出文件（可选）
    """
    # 加载运动数据
    q, dq, ddq, tau = load_motion_data(csv_file)
    
    # 初始化动力学回归器
    regressor_lib_path = '/Users/lr-2002/project/instantcreation/IC_arm_control/urdfly/dyn_regress.dylib'
    calc_dynamics = CalcDynamics(regressor_lib_path)
    
    print(f"\n开始计算回归器矩阵...")
    print(f"数据点数: {len(q)}")
    
    # 计算回归器矩阵 - 逐个时间点处理
    all_regressors = []
    
    for i in range(len(q)):
        if i % 1000 == 0:  # 每1000个点显示进度
            print(f"处理进度: {i}/{len(q)} ({i/len(q)*100:.1f}%)")
        
        # 计算单个时间点的回归器矩阵
        regressor = calc_dynamics.calc(q[i], dq[i], ddq[i])
        all_regressors.append(regressor)
    
    # 转换为numpy数组
    regressor_matrix = np.vstack(all_regressors)
    
    print(f"回归器矩阵形状: {regressor_matrix.shape}")
    print(f"回归器矩阵统计:")
    print(f"  最小值: {regressor_matrix.min():.6f}")
    print(f"  最大值: {regressor_matrix.max():.6f}")
    print(f"  平均值: {regressor_matrix.mean():.6f}")
    print(f"  标准差: {regressor_matrix.std():.6f}")
    
    # 保存数据
    if output_file is None:
        output_file = csv_file.replace('.csv', '_regressor_data.npz')
    
    np.savez(output_file, 
             q=q, 
             dq=dq, 
             ddq=ddq, 
             regressor=regressor_matrix)
    
    print(f"\n数据已保存到: {output_file}")
    
    return q, dq, ddq, regressor_matrix

def analyze_motion_data(q, dq, ddq):
    """分析运动数据质量"""
    print("\n=== 运动数据分析 ===")
    
    num_joints = q.shape[1]
    
    for joint_idx in range(num_joints):
        joint_id = joint_idx + 1
        
        pos = q[:, joint_idx]
        vel = dq[:, joint_idx]
        acc = ddq[:, joint_idx]
        
        print(f"\n关节 {joint_id}:")
        print(f"  位置范围: [{np.degrees(pos.min()):.2f}°, {np.degrees(pos.max()):.2f}°]")
        print(f"  速度范围: [{np.degrees(vel.min()):.2f}°/s, {np.degrees(vel.max()):.2f}°/s]")
        print(f"  加速度范围: [{np.degrees(acc.min()):.2f}°/s², {np.degrees(acc.max()):.2f}°/s²]")
        
        # 检查是否有运动
        if np.max(np.abs(pos)) < 1e-6:
            print(f"  状态: 无运动")
        else:
            print(f"  状态: 有运动")

if __name__ == "__main__":
    # 输入CSV文件路径
    csv_file = "/Users/lr-2002/project/instantcreation/IC_arm_control/dynamics_data_20250820_170711.csv"
    
    try:
        # 导出回归器数据
        q, dq, ddq, regressor_matrix = export_regressor_data(csv_file)
        
        # 分析运动数据
        analyze_motion_data(q, dq, ddq)
        
        print("\n=== 导出完成 ===")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()