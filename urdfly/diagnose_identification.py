#!/usr/bin/env python3
"""
诊断参数辨识问题
分析数据质量、回归器矩阵条件数、激励轨迹等因素
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minimum_param import MinimumParameterIdentification
import seaborn as sns

def diagnose_data_quality(data_file, max_points=1000):
    """
    诊断数据质量
    
    Args:
        data_file: 数据文件路径
        max_points: 分析的数据点数
    """
    print("=== 数据质量诊断 ===")
    
    # 加载数据
    identifier = MinimumParameterIdentification()
    q, dq, ddq, tau = identifier.load_motion_data(data_file)
    
    if len(q) > max_points:
        q = q[:max_points]
        dq = dq[:max_points] 
        ddq = ddq[:max_points]
        tau = tau[:max_points]
    
    print(f"分析数据点数: {len(q)}")
    
    # 1. 检查数据范围和分布
    print(f"\n1. 数据范围分析:")
    for i in range(5):
        print(f"关节{i+1}:")
        print(f"  位置范围: [{np.degrees(q[:, i].min()):.2f}°, {np.degrees(q[:, i].max()):.2f}°]")
        print(f"  速度范围: [{np.degrees(dq[:, i].min()):.2f}°/s, {np.degrees(dq[:, i].max()):.2f}°/s]") 
        print(f"  加速度范围: [{np.degrees(ddq[:, i].min()):.2f}°/s², {np.degrees(ddq[:, i].max()):.2f}°/s²]")
        print(f"  力矩范围: [{tau[:, i].min():.3f}, {tau[:, i].max():.3f}] Nm")
    
    # 2. 检查数据激励程度
    print(f"\n2. 激励程度分析:")
    for i in range(5):
        pos_std = np.std(q[:, i])
        vel_std = np.std(dq[:, i])
        acc_std = np.std(ddq[:, i])
        tau_std = np.std(tau[:, i])
        
        print(f"关节{i+1}:")
        print(f"  位置标准差: {np.degrees(pos_std):.2f}°")
        print(f"  速度标准差: {np.degrees(vel_std):.2f}°/s")
        print(f"  加速度标准差: {np.degrees(acc_std):.2f}°/s²")
        print(f"  力矩标准差: {tau_std:.3f} Nm")
        
        # 检查是否有足够的激励
        if pos_std < np.radians(5):
            print(f"  ⚠️  位置激励不足 (< 5°)")
        if vel_std < np.radians(10):
            print(f"  ⚠️  速度激励不足 (< 10°/s)")
        if acc_std < np.radians(20):
            print(f"  ⚠️  加速度激励不足 (< 20°/s²)")
    
    # 3. 检查数据中的异常值
    print(f"\n3. 异常值检测:")
    for i in range(5):
        # 使用3σ准则检测异常值
        tau_mean = np.mean(tau[:, i])
        tau_std = np.std(tau[:, i])
        outliers = np.abs(tau[:, i] - tau_mean) > 3 * tau_std
        outlier_ratio = np.sum(outliers) / len(tau) * 100
        
        print(f"关节{i+1}: {outlier_ratio:.1f}% 异常值")
        if outlier_ratio > 5:
            print(f"  ⚠️  异常值比例过高")
    
    # 4. 检查数据连续性
    print(f"\n4. 数据连续性检查:")
    for i in range(5):
        # 检查速度和位置的一致性（数值微分）
        dt = 0.01  # 假设采样时间为10ms
        dq_numerical = np.gradient(q[:, i]) / dt
        dq_error = np.mean(np.abs(dq[:, i] - dq_numerical))
        
        # 检查加速度和速度的一致性
        ddq_numerical = np.gradient(dq[:, i]) / dt
        ddq_error = np.mean(np.abs(ddq[:, i] - ddq_numerical))
        
        print(f"关节{i+1}:")
        print(f"  速度一致性误差: {np.degrees(dq_error):.2f}°/s")
        print(f"  加速度一致性误差: {np.degrees(ddq_error):.2f}°/s²")
        
        if dq_error > np.radians(5):
            print(f"  ⚠️  速度数据可能不一致")
        if ddq_error > np.radians(10):
            print(f"  ⚠️  加速度数据可能不一致")
    
    return q, dq, ddq, tau

def diagnose_regressor_matrix(q, dq, ddq, tau):
    """
    诊断回归器矩阵问题
    
    Args:
        q, dq, ddq, tau: 运动数据
    """
    print(f"\n=== 回归器矩阵诊断 ===")
    
    # 构建回归器矩阵
    identifier = MinimumParameterIdentification()
    A_N = identifier.build_regressor_matrix(q, dq, ddq)
    tau_N = tau.flatten()
    
    print(f"回归器矩阵形状: {A_N.shape}")
    
    # 1. 矩阵条件数分析
    print(f"\n1. 矩阵条件数分析:")
    cond_num = np.linalg.cond(A_N)
    print(f"回归器矩阵条件数: {cond_num:.2e}")
    
    if cond_num > 1e12:
        print("  ⚠️  矩阵条件数过大，可能存在数值问题")
    elif cond_num > 1e8:
        print("  ⚠️  矩阵条件数较大，建议使用正则化")
    else:
        print("  ✓  矩阵条件数正常")
    
    # 2. 矩阵秩分析
    print(f"\n2. 矩阵秩分析:")
    rank = np.linalg.matrix_rank(A_N)
    print(f"矩阵秩: {rank}/{A_N.shape[1]}")
    print(f"秩亏损: {A_N.shape[1] - rank}")
    
    if rank < A_N.shape[1]:
        print("  ⚠️  矩阵不满秩，存在线性相关的列")
    
    # 3. 奇异值分析
    print(f"\n3. 奇异值分析:")
    U, s, Vt = np.linalg.svd(A_N, full_matrices=False)
    print(f"最大奇异值: {s[0]:.2e}")
    print(f"最小奇异值: {s[-1]:.2e}")
    print(f"奇异值比: {s[0]/s[-1]:.2e}")
    
    # 找到小奇异值
    small_sv_threshold = s[0] * 1e-8
    small_sv_count = np.sum(s < small_sv_threshold)
    print(f"小奇异值数量 (< {small_sv_threshold:.2e}): {small_sv_count}")
    
    # 4. 列相关性分析
    print(f"\n4. 列相关性分析:")
    # 计算列之间的最大相关系数
    corr_matrix = np.corrcoef(A_N.T)
    np.fill_diagonal(corr_matrix, 0)  # 忽略对角线
    max_corr = np.max(np.abs(corr_matrix))
    print(f"列间最大相关系数: {max_corr:.4f}")
    
    if max_corr > 0.95:
        print("  ⚠️  存在高度相关的列")
    
    # 5. 力矩向量分析
    print(f"\n5. 力矩向量分析:")
    print(f"力矩向量范围: [{tau_N.min():.3f}, {tau_N.max():.3f}]")
    print(f"力矩向量标准差: {tau_N.std():.3f}")
    print(f"力矩向量均值: {tau_N.mean():.3f}")
    
    # 检查是否有零力矩
    zero_torque_ratio = np.sum(np.abs(tau_N) < 1e-6) / len(tau_N) * 100
    print(f"近零力矩比例: {zero_torque_ratio:.1f}%")
    
    if zero_torque_ratio > 10:
        print("  ⚠️  零力矩比例过高，可能影响辨识")
    
    return A_N, tau_N, s

def diagnose_identification_methods(A_N, tau_N):
    """
    诊断不同辨识方法的结果
    
    Args:
        A_N: 回归器矩阵
        tau_N: 力矩向量
    """
    print(f"\n=== 辨识方法诊断 ===")
    
    identifier = MinimumParameterIdentification()
    methods = ['qr', 'svd', 'correlation']
    
    for method in methods:
        print(f"\n--- 方法: {method} ---")
        try:
            theta_b = identifier.identify_base_parameters(A_N, tau_N, method=method, regularization=1e-6)
            
            # 计算拟合质量
            A_b = A_N[:, identifier.base_columns]
            tau_pred = A_b @ theta_b
            residual = tau_N - tau_pred
            rms_error = np.sqrt(np.mean(residual**2))
            relative_error = rms_error / np.std(tau_N) * 100
            
            print(f"基参数数量: {len(identifier.base_columns)}")
            print(f"RMS误差: {rms_error:.6f}")
            print(f"相对误差: {relative_error:.2f}%")
            print(f"条件数: {np.linalg.cond(A_b):.2e}")
            
            # 检查参数合理性
            param_range = np.max(theta_b) - np.min(theta_b)
            param_std = np.std(theta_b)
            print(f"参数范围: {param_range:.3f}")
            print(f"参数标准差: {param_std:.3f}")
            
            # 检查是否有异常大的参数
            large_params = np.sum(np.abs(theta_b) > 1000)
            if large_params > 0:
                print(f"  ⚠️  {large_params}个参数值异常大 (>1000)")
            
        except Exception as e:
            print(f"方法失败: {e}")

def suggest_improvements(A_N, tau_N, s):
    """
    提出改进建议
    
    Args:
        A_N: 回归器矩阵
        tau_N: 力矩向量
        s: 奇异值
    """
    print(f"\n=== 改进建议 ===")
    
    suggestions = []
    
    # 1. 数据质量建议
    cond_num = np.linalg.cond(A_N)
    if cond_num > 1e10:
        suggestions.append("1. 增加正则化参数 (regularization=1e-4 或更大)")
        suggestions.append("2. 使用更丰富的激励轨迹")
    
    # 2. 奇异值建议
    sv_ratio = s[0] / s[-1]
    if sv_ratio > 1e12:
        suggestions.append("3. 矩阵病态严重，考虑:")
        suggestions.append("   - 数据预处理 (归一化)")
        suggestions.append("   - 减少参数数量")
        suggestions.append("   - 使用岭回归")
    
    # 3. 力矩数据建议
    tau_std = np.std(tau_N)
    if tau_std < 0.1:
        suggestions.append("4. 力矩变化太小，建议:")
        suggestions.append("   - 使用更大幅度的运动")
        suggestions.append("   - 增加负载变化")
    
    # 4. 数据量建议
    if A_N.shape[0] < A_N.shape[1] * 10:
        suggestions.append("5. 数据量可能不足，建议:")
        suggestions.append("   - 增加数据点数")
        suggestions.append("   - 使用更长的轨迹")
    
    if suggestions:
        for suggestion in suggestions:
            print(suggestion)
    else:
        print("数据质量良好，无明显问题")

def main():
    """主诊断函数"""
    data_file = "/Users/lr-2002/project/instantcreation/IC_arm_control/dynamics_data_20250820_170711.csv"
    
    # 1. 数据质量诊断
    q, dq, ddq, tau = diagnose_data_quality(data_file, max_points=10000)
    
    # 2. 回归器矩阵诊断
    A_N, tau_N, s = diagnose_regressor_matrix(q, dq, ddq, tau)
    
    # 3. 辨识方法诊断
    diagnose_identification_methods(A_N, tau_N)
    
    # 4. 改进建议
    suggest_improvements(A_N, tau_N, s)

if __name__ == "__main__":
    main()
