#!/usr/bin/env python3
"""
检查CSV数据格式和单位
"""

import numpy as np
import pandas as pd

def check_csv_format(csv_file):
    """检查CSV文件格式"""
    print(f"检查文件: {csv_file}")
    
    try:
        # 读取前几行查看结构
        data = pd.read_csv(csv_file, nrows=5)
        print(f"\n文件列数: {data.shape[1]}")
        print(f"列名: {list(data.columns)}")
        print(f"\n前5行数据:")
        print(data)
        
        # 读取完整数据
        full_data = pd.read_csv(csv_file)
        print(f"\n完整数据形状: {full_data.shape}")
        
        # 分析数据范围
        print(f"\n数据范围分析:")
        for i, col in enumerate(full_data.columns):
            if i == 0:  # 跳过时间戳
                continue
            values = full_data.iloc[:, i].values
            print(f"列{i} ({col}): [{values.min():.6f}, {values.max():.6f}]")
            
            if i <= 5:  # 位置数据
                print(f"  -> 角度范围: [{np.degrees(values.min()):.2f}°, {np.degrees(values.max()):.2f}°]")
            elif i <= 10:  # 速度数据  
                print(f"  -> 角速度范围: [{np.degrees(values.min()):.2f}°/s, {np.degrees(values.max()):.2f}°/s]")
            elif i <= 15:  # 加速度数据
                print(f"  -> 角加速度范围: [{np.degrees(values.min()):.2f}°/s², {np.degrees(values.max()):.2f}°/s²]")
            elif i <= 20:  # 力矩数据
                print(f"  -> 力矩范围: [{values.min():.3f}, {values.max():.3f}] Nm")
        
        return full_data
        
    except Exception as e:
        print(f"读取文件出错: {e}")
        return None

def analyze_data_units(data):
    """分析数据单位问题"""
    print(f"\n=== 数据单位分析 ===")
    
    # 检查是否需要单位转换
    if data.shape[1] >= 21:
        # 假设数据结构：时间戳 + 5位置 + 5速度 + 5加速度 + 5力矩
        q_cols = list(range(1, 6))
        dq_cols = list(range(6, 11))  
        ddq_cols = list(range(11, 16))
        tau_cols = list(range(16, 21))
        
        print("假设的数据结构:")
        print(f"位置列: {q_cols}")
        print(f"速度列: {dq_cols}")
        print(f"加速度列: {ddq_cols}")
        print(f"力矩列: {tau_cols}")
        
        # 检查位置数据是否合理
        for i, col_idx in enumerate(q_cols):
            values = data.iloc[:, col_idx].values
            range_rad = values.max() - values.min()
            range_deg = np.degrees(range_rad)
            
            print(f"\n关节{i+1}位置分析:")
            print(f"  原始范围: [{values.min():.6f}, {values.max():.6f}]")
            print(f"  假设弧度: [{np.degrees(values.min()):.2f}°, {np.degrees(values.max()):.2f}°]")
            
            # 检查是否可能是度数
            if abs(values.max()) > 10:  # 如果数值很大，可能是度数
                print(f"  假设度数: [{values.min():.2f}°, {values.max():.2f}°]")
                print(f"  转换弧度: [{np.radians(values.min()):.6f}, {np.radians(values.max()):.6f}]")
            
            # 检查是否可能是编码器计数
            if abs(values.max()) > 1000:
                print(f"  ⚠️  数值过大，可能是编码器计数或其他单位")
    
    return q_cols, dq_cols, ddq_cols, tau_cols

def suggest_data_loading(data, q_cols, dq_cols, ddq_cols, tau_cols):
    """建议正确的数据加载方式"""
    print(f"\n=== 数据加载建议 ===")
    
    # 分析位置数据特征
    pos_data = data.iloc[:, q_cols].values
    max_pos = np.max(np.abs(pos_data))
    
    if max_pos > 100:  # 很可能是度数
        print("建议1: 位置数据可能是度数，需要转换为弧度")
        print("  q = np.radians(data.iloc[:, 1:6].values)")
        
    elif max_pos > 10:  # 可能是编码器计数
        print("建议2: 位置数据可能是编码器计数，需要转换")
        print("  需要知道编码器分辨率进行转换")
        
    else:  # 可能已经是弧度
        print("建议3: 位置数据可能已经是弧度")
        print("  q = data.iloc[:, 1:6].values")
    
    # 检查速度数据
    vel_data = data.iloc[:, dq_cols].values
    max_vel = np.max(np.abs(vel_data))
    
    if max_vel < 0.01:  # 速度很小
        print("\n⚠️  速度数据异常小，可能存在问题:")
        print("  1. 数据采集时机器人静止")
        print("  2. 速度计算有误")
        print("  3. 单位转换问题")

if __name__ == "__main__":
    # 检查可用的CSV文件
    import os
    
    # 尝试找到一个可访问的CSV文件
    test_files = [
        "/Users/lr-2002/project/instantcreation/IC_arm_control/dynamics_data_20250820_170711.csv",
        "/Users/lr-2002/project/instantcreation/IC_arm_control/dynamic_infos/dynamics_data_20250808_111545.csv"
    ]
    
    for csv_file in test_files:
        if os.path.exists(csv_file):
            print(f"找到文件: {csv_file}")
            data = check_csv_format(csv_file)
            if data is not None:
                q_cols, dq_cols, ddq_cols, tau_cols = analyze_data_units(data)
                suggest_data_loading(data, q_cols, dq_cols, ddq_cols, tau_cols)
            break
    else:
        print("未找到可访问的CSV文件")
