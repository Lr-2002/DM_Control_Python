#!/usr/bin/env python3
"""
分析RMS拟合误差过大的原因
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minimum_param import MinimumParameterIdentification

def compare_datasets(old_file, new_file):
    """比较新旧数据集的差异"""
    print("=== 数据集对比分析 ===")
    
    identifier = MinimumParameterIdentification()
    
    # 加载两个数据集
    print("加载旧数据集...")
    q1, dq1, ddq1, tau1 = identifier.load_motion_data(old_file)
    
    print("加载新数据集...")
    q2, dq2, ddq2, tau2 = identifier.load_motion_data(new_file)
    
    print(f"\n数据点数对比:")
    print(f"旧数据集: {len(q1)} 点")
    print(f"新数据集: {len(q2)} 点")
    
    # 对比各关节的激励程度
    print(f"\n激励程度对比:")
    print(f"{'关节':<6} {'旧数据位置变化':<15} {'新数据位置变化':<15} {'旧数据速度变化':<15} {'新数据速度变化':<15}")
    print("-" * 75)
    
    for i in range(5):
        old_pos_range = np.degrees(q1[:, i].max() - q1[:, i].min())
        new_pos_range = np.degrees(q2[:, i].max() - q2[:, i].min())
        old_vel_range = np.degrees(dq1[:, i].max() - dq1[:, i].min())
        new_vel_range = np.degrees(dq2[:, i].max() - dq2[:, i].min())
        
        print(f"关节{i+1:<5} {old_pos_range:<15.2f} {new_pos_range:<15.2f} {old_vel_range:<15.2f} {new_vel_range:<15.2f}")
    
    # 对比力矩数据
    print(f"\n力矩数据对比:")
    print(f"{'关节':<6} {'旧数据力矩范围':<20} {'新数据力矩范围':<20} {'旧数据力矩标准差':<15} {'新数据力矩标准差':<15}")
    print("-" * 85)
    
    for i in range(5):
        old_tau_range = f"[{tau1[:, i].min():.2f}, {tau1[:, i].max():.2f}]"
        new_tau_range = f"[{tau2[:, i].min():.2f}, {tau2[:, i].max():.2f}]"
        old_tau_std = tau1[:, i].std()
        new_tau_std = tau2[:, i].std()
        
        print(f"关节{i+1:<5} {old_tau_range:<20} {new_tau_range:<20} {old_tau_std:<15.3f} {new_tau_std:<15.3f}")
    
    # 检查1号电机的特殊情况
    print(f"\n1号电机详细分析:")
    print(f"旧数据集 - 1号电机:")
    print(f"  位置范围: [{np.degrees(q1[:, 0].min()):.2f}°, {np.degrees(q1[:, 0].max()):.2f}°]")
    print(f"  速度范围: [{np.degrees(dq1[:, 0].min()):.2f}°/s, {np.degrees(dq1[:, 0].max()):.2f}°/s]")
    print(f"  力矩范围: [{tau1[:, 0].min():.3f}, {tau1[:, 0].max():.3f}] Nm")
    print(f"  力矩标准差: {tau1[:, 0].std():.3f} Nm")
    
    print(f"\n新数据集 - 1号电机:")
    print(f"  位置范围: [{np.degrees(q2[:, 0].min()):.2f}°, {np.degrees(q2[:, 0].max()):.2f}°]")
    print(f"  速度范围: [{np.degrees(dq2[:, 0].min()):.2f}°/s, {np.degrees(dq2[:, 0].max()):.2f}°/s]")
    print(f"  力矩范围: [{tau2[:, 0].min():.3f}, {tau2[:, 0].max():.3f}] Nm")
    print(f"  力矩标准差: {tau2[:, 0].std():.3f} Nm")
    
    return q1, dq1, ddq1, tau1, q2, dq2, ddq2, tau2

def analyze_torque_patterns(tau1, tau2):
    """分析力矩模式差异"""
    print(f"\n=== 力矩模式分析 ===")
    
    # 总力矩统计
    tau1_flat = tau1.flatten()
    tau2_flat = tau2.flatten()
    
    print(f"旧数据集力矩统计:")
    print(f"  总体范围: [{tau1_flat.min():.3f}, {tau1_flat.max():.3f}] Nm")
    print(f"  总体标准差: {tau1_flat.std():.3f} Nm")
    print(f"  零力矩比例: {np.sum(np.abs(tau1_flat) < 0.01) / len(tau1_flat) * 100:.1f}%")
    
    print(f"\n新数据集力矩统计:")
    print(f"  总体范围: [{tau2_flat.min():.3f}, {tau2_flat.max():.3f}] Nm")
    print(f"  总体标准差: {tau2_flat.std():.3f} Nm")
    print(f"  零力矩比例: {np.sum(np.abs(tau2_flat) < 0.01) / len(tau2_flat) * 100:.1f}%")
    
    # 检查1号电机力矩异常
    print(f"\n1号电机力矩分析:")
    print(f"旧数据 - 1号电机力矩分布:")
    tau1_m1 = tau1[:, 0]
    positive_ratio1 = np.sum(tau1_m1 > 0) / len(tau1_m1) * 100
    negative_ratio1 = np.sum(tau1_m1 < 0) / len(tau1_m1) * 100
    print(f"  正力矩比例: {positive_ratio1:.1f}%")
    print(f"  负力矩比例: {negative_ratio1:.1f}%")
    
    print(f"\n新数据 - 1号电机力矩分布:")
    tau2_m1 = tau2[:, 0]
    positive_ratio2 = np.sum(tau2_m1 > 0) / len(tau2_m1) * 100
    negative_ratio2 = np.sum(tau2_m1 < 0) / len(tau2_m1) * 100
    print(f"  正力矩比例: {positive_ratio2:.1f}%")
    print(f"  负力矩比例: {negative_ratio2:.1f}%")
    
    if negative_ratio2 > 95:
        print(f"  ⚠️  1号电机几乎全是负力矩，可能存在:")
        print(f"     - 重力补偿问题")
        print(f"     - 控制器设置问题")
        print(f"     - 传感器偏置")

def analyze_identification_quality(old_file, new_file):
    """分析辨识质量差异"""
    print(f"\n=== 辨识质量分析 ===")
    
    identifier = MinimumParameterIdentification()
    
    # 对比辨识结果
    datasets = [
        ("旧数据集", old_file),
        ("新数据集", new_file)
    ]
    
    for name, data_file in datasets:
        print(f"\n{name}辨识结果:")
        try:
            q, dq, ddq, tau = identifier.load_motion_data(data_file)
            
            # 使用较少数据点进行快速分析
            if len(q) > 2000:
                q = q[:2000]
                dq = dq[:2000]
                ddq = ddq[:2000]
                tau = tau[:2000]
            
            A_N = identifier.build_regressor_matrix(q, dq, ddq)
            tau_N = tau.flatten()
            
            # 使用correlation方法快速辨识
            theta_b = identifier.identify_base_parameters(A_N, tau_N, method='correlation')
            
            # 计算拟合质量
            A_b = A_N[:, identifier.base_columns]
            tau_pred = A_b @ theta_b
            residual = tau_N - tau_pred
            rms_error = np.sqrt(np.mean(residual**2))
            relative_error = rms_error / np.std(tau_N) * 100
            
            print(f"  基参数数量: {len(identifier.base_columns)}")
            print(f"  RMS误差: {rms_error:.6f} Nm")
            print(f"  相对误差: {relative_error:.2f}%")
            print(f"  力矩标准差: {np.std(tau_N):.3f} Nm")
            print(f"  条件数: {np.linalg.cond(A_b):.2e}")
            
            # 分析残差分布
            residual_std = np.std(residual)
            print(f"  残差标准差: {residual_std:.6f} Nm")
            print(f"  最大残差: {np.max(np.abs(residual)):.6f} Nm")
            
        except Exception as e:
            print(f"  辨识失败: {e}")

def suggest_improvements():
    """提出改进建议"""
    print(f"\n=== 改进建议 ===")
    print("1. 1号电机问题:")
    print("   - 检查重力补偿设置")
    print("   - 调整控制器参数 (kp, kd)")
    print("   - 验证力矩传感器校准")
    
    print("\n2. 数据质量改进:")
    print("   - 增加正向力矩的运动")
    print("   - 平衡各关节的激励程度")
    print("   - 检查轨迹执行精度")
    
    print("\n3. 辨识方法优化:")
    print("   - 增加正则化参数")
    print("   - 使用加权最小二乘")
    print("   - 过滤异常数据点")

def main():
    """主分析函数"""
    old_file = "/Users/lr-2002/project/instantcreation/IC_arm_control/dynamics_data_20250820_170711.csv"
    new_file = "/Users/lr-2002/project/instantcreation/IC_arm_control/dynamics_data_20250820_204957.csv"
    
    # 数据集对比
    q1, dq1, ddq1, tau1, q2, dq2, ddq2, tau2 = compare_datasets(old_file, new_file)
    
    # 力矩模式分析
    analyze_torque_patterns(tau1, tau2)
    
    # 辨识质量分析
    analyze_identification_quality(old_file, new_file)
    
    # 改进建议
    suggest_improvements()

if __name__ == "__main__":
    main()
