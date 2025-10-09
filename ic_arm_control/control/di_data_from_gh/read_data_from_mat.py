#!/usr/bin/env python3
"""
读取MAT文件数据
从GitHub论文数据中读取激励轨迹优化结果
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

def read_mat_file(mat_file_path):
    """
    读取MAT文件并返回数据结构
    
    Args:
        mat_file_path: MAT文件路径
        
    Returns:
        dict: 包含MAT文件中所有变量的字典
    """
    mat_data = scipy.io.loadmat(mat_file_path)
    
    print(f"=== 读取MAT文件: {os.path.basename(mat_file_path)} ===\n")
    print("文件中包含的变量:")
    
    data_dict = {}
    for key in mat_data.keys():
        if not key.startswith('__'):
            value = mat_data[key]
            data_dict[key] = value
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            
            if value.size < 50:
                print(f"    数据: {value}")
            print()
    
    return data_dict

def fourier_series_traj(t, q0, A, B, w, N):
    """
    计算截断傅里叶级数轨迹（积分形式）
    
    Args:
        t: 时间数组
        q0: 初始偏移 (n_joints,)
        A: 正弦系数 (n_joints, N)
        B: 余弦系数 (n_joints, N)
        w: 基础频率
        N: 谐波数量
        
    Returns:
        q: 位置
        qd: 速度
        q2d: 加速度
    """
    n_joints = len(q0)
    n_time = len(t)
    
    q = np.tile(q0.reshape(-1, 1), (1, n_time))
    qd = np.zeros((n_joints, n_time))
    q2d = np.zeros((n_joints, n_time))
    
    for k in range(1, N + 1):
        wk = w * k
        sin_term = np.sin(wk * t)
        cos_term = np.cos(wk * t)
        
        for joint_idx in range(n_joints):
            q[joint_idx, :] += A[joint_idx, k-1] / wk * sin_term - B[joint_idx, k-1] / wk * cos_term
            qd[joint_idx, :] += A[joint_idx, k-1] * cos_term + B[joint_idx, k-1] * sin_term
            q2d[joint_idx, :] += -A[joint_idx, k-1] * wk * sin_term + B[joint_idx, k-1] * wk * cos_term
    
    return q, qd, q2d

def polynomial_traj(t, C):
    """
    计算五阶多项式轨迹
    
    Args:
        t: 时间数组
        C: 多项式系数矩阵 (n_joints, 6)
        
    Returns:
        qp: 位置
        qpd: 速度
        qp2d: 加速度
    """
    n_joints = C.shape[0]
    n_time = len(t)
    
    qp = np.zeros((n_joints, n_time))
    qpd = np.zeros((n_joints, n_time))
    qp2d = np.zeros((n_joints, n_time))
    
    for joint_idx in range(n_joints):
        qp[joint_idx, :] = (C[joint_idx, 0] + 
                            C[joint_idx, 1] * t + 
                            C[joint_idx, 2] * t**2 + 
                            C[joint_idx, 3] * t**3 + 
                            C[joint_idx, 4] * t**4 + 
                            C[joint_idx, 5] * t**5)
        
        qpd[joint_idx, :] = (C[joint_idx, 1] + 
                             2 * C[joint_idx, 2] * t + 
                             3 * C[joint_idx, 3] * t**2 + 
                             4 * C[joint_idx, 4] * t**3 + 
                             5 * C[joint_idx, 5] * t**4)
        
        qp2d[joint_idx, :] = (2 * C[joint_idx, 2] + 
                              6 * C[joint_idx, 3] * t + 
                              12 * C[joint_idx, 4] * t**2 + 
                              20 * C[joint_idx, 5] * t**3)
    
    return qp, qpd, qp2d

def generate_trajectory(a, b, c_pol, traj_par):
    """
    生成混合轨迹：傅里叶级数 + 五阶多项式
    
    Args:
        a: 正弦系数矩阵 (n_joints, N)
        b: 余弦系数矩阵 (n_joints, N)
        c_pol: 多项式系数矩阵 (n_joints, 6)
        traj_par: 轨迹参数结构体
        
    Returns:
        t: 时间数组
        q: 关节角度轨迹 (n_joints, n_timesteps)
        qd: 关节速度轨迹
        q2d: 关节加速度轨迹
    """
    params = traj_par[0, 0]
    T = float(params['T'][0, 0])
    wf = float(params['wf'][0, 0])
    N = int(params['N'][0, 0])
    q0 = params['q0'].flatten().astype(np.float64)
    
    # 使用250Hz控制频率重新采样
    control_freq = 250.0  # Hz
    dt = 1.0 / control_freq
    n_points = int(T * control_freq) + 1
    t = np.linspace(0, T, n_points)
    
    qh, qhd, qh2d = fourier_series_traj(t, q0, a, b, wf, N)
    qp, qpd, qp2d = polynomial_traj(t, c_pol)
    
    q = qh + qp
    qd = qhd + qpd
    q2d = qh2d + qp2d
    
    return t, q, qd, q2d

def plot_trajectories(t, q):
    """
    绘制所有关节的轨迹
    
    Args:
        t: 时间数组
        q: 关节角度轨迹 (n_joints, n_timesteps)
    """
    plt.figure(figsize=(12, 8))
    
    n_joints = q.shape[0]
    colors = plt.cm.tab10(np.linspace(0, 1, n_joints))
    
    for joint_idx in range(n_joints):
        plt.plot(t, q[joint_idx, :], label=f'Joint {joint_idx+1}', 
                color=colors[joint_idx], linewidth=2)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Joint Angle (rad)', fontsize=12)
    plt.title('Excitation Trajectory - All Joints', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def save_trajectory_for_executor(t, q, qd, q2d, mat_file_path):
    """
    保存轨迹为trajectory_executor可执行的JSON格式
    
    Args:
        t: 时间数组 (n_timesteps,)
        q: 关节角度轨迹 (n_joints, n_timesteps)
        qd: 关节速度轨迹 (n_joints, n_timesteps)
        q2d: 关节加速度轨迹 (n_joints, n_timesteps)
        mat_file_path: 原始MAT文件路径
    """
    # 转置数组以匹配trajectory_executor的格式 (n_timesteps, n_joints)
    positions = q.T
    velocities = qd.T
    accelerations = q2d.T
    
    # 构建轨迹字典
    trajectory = {
        'time': t.tolist(),
        'positions': positions.tolist(),
        'velocities': velocities.tolist(),
        'accelerations': accelerations.tolist(),
        'metadata': {
            'source_file': os.path.basename(mat_file_path),
            'source_type': 'matlab_optimization',
            'generation_time': datetime.now().isoformat(),
            'num_joints': q.shape[0],
            'num_points': len(t),
            'duration': float(t[-1]),
            'control_frequency': 250.0,
            'description': 'Excitation trajectory for dynamics identification (Fourier + Polynomial)'
        }
    }
    
    # 生成输出文件名
    mat_basename = os.path.splitext(os.path.basename(mat_file_path))[0]
    output_dir = os.path.dirname(mat_file_path)
    output_filename = os.path.join(output_dir, f"{mat_basename}_trajectory_250hz.json")
    
    # 保存为JSON文件
    with open(output_filename, 'w') as f:
        json.dump(trajectory, f, indent=2)
    
    print(f"\n=== 轨迹已保存 ===")
    print(f"文件: {output_filename}")
    print(f"格式: JSON (trajectory_executor兼容)")
    print(f"关节数: {q.shape[0]}")
    print(f"数据点: {len(t)}")
    print(f"时长: {t[-1]:.2f}秒")
    print(f"采样频率: 250Hz")
    print(f"\n可以使用以下命令执行轨迹:")
    print(f"  python trajectory_executor.py {output_filename} mujoco")
    print(f"  或")
    print(f"  python trajectory_executor.py {output_filename} ic_arm")

def main():
    mat_file_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/ic_arm_control/control/di_data_from_gh/ptrnSrch_N7T25QR.mat"

    data = read_mat_file(mat_file_path)
    
    print("\n=== 数据详细信息 ===")
    for key, value in data.items():
        print(f"\n变量名: {key}")
        print(f"  形状: {value.shape}")
        print(f"  数据类型: {value.dtype}")
        
        if value.dtype.names:
            print(f"  结构体字段: {value.dtype.names}")
            struct_data = value[0, 0]
            for field_name in value.dtype.names:
                field_value = struct_data[field_name]
                print(f"    {field_name}: shape={field_value.shape}, dtype={field_value.dtype}")
                if field_value.size <= 10:
                    print(f"      值: {field_value.flatten()}")
        else:
            print(f"  最小值: {np.min(value)}")
            print(f"  最大值: {np.max(value)}")
            print(f"  平均值: {np.mean(value)}")
            
            if value.ndim == 2:
                print(f"  维度: {value.shape[0]} x {value.shape[1]}")
                if value.shape[0] <= 20 and value.shape[1] <= 20:
                    print(f"  完整数据:\n{value}")
            elif value.ndim == 1:
                print(f"  长度: {len(value)}")
                if len(value) <= 20:
                    print(f"  完整数据: {value}")
    
    print("\n=== 生成并绘制轨迹 ===")
    t, q, qd, q2d = generate_trajectory(data['a'], data['b'], data['c_pol'], data['traj_par'])
    print(f"生成轨迹: {q.shape[0]} 个关节, {q.shape[1]} 个时间点")
    print(f"时间范围: {t[0]:.2f}s 到 {t[-1]:.2f}s")
    
    for joint_idx in range(q.shape[0]):
        print(f"Joint {joint_idx+1}: 范围 [{np.min(q[joint_idx, :]):.3f}, {np.max(q[joint_idx, :]):.3f}] rad")
    
    plot_trajectories(t, q)
    
    # 保存轨迹为trajectory_executor可执行的格式
    save_trajectory_for_executor(t, q, qd, q2d, mat_file_path)
    
    return data

if __name__ == "__main__":
    data = main()