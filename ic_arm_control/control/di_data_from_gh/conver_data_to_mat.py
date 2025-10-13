#!/usr/bin/env python3
"""
将动力学辨识数据转换为MAT格式
按照MATLAB格式保存: t, q, dq, ddq, tau
"""

import numpy as np
import pandas as pd
import scipy.io
import os
from pathlib import Path


def convert_csv_to_mat(csv_file_path, output_dir=None):
    """
    将预处理后的CSV数据转换为MAT格式
    
    Args:
        csv_file_path: 预处理后的CSV文件路径
        output_dir: 输出目录，默认为CSV文件所在目录
    """
    print(f"\n转换文件: {Path(csv_file_path).name}")
    
    # 读取CSV数据
    df = pd.read_csv(csv_file_path)
    print(f"  数据形状: {df.shape}")
    
    # 提取时间数组
    t = df['time'].values
    
    # 提取关节数据 (6个关节)
    n_joints = 6
    n_samples = len(df)
    
    # 初始化矩阵 (n_samples x n_joints)
    q = np.zeros((n_samples, n_joints))
    dq = np.zeros((n_samples, n_joints))
    ddq = np.zeros((n_samples, n_joints))
    tau = np.zeros((n_samples, n_joints))
    
    # 填充数据
    for j in range(n_joints):
        joint_id = j + 1
        q[:, j] = df[f'q{joint_id}'].values
        dq[:, j] = df[f'dq{joint_id}'].values
        ddq[:, j] = df[f'ddq{joint_id}'].values
        tau[:, j] = df[f'tau{joint_id}'].values
    
    print(f"  时间范围: [{t[0]:.3f}, {t[-1]:.3f}] s")
    print(f"  关节数据形状: {q.shape}")
    
    # 生成输出文件名
    if output_dir is None:
        output_dir = Path(csv_file_path).parent
    
    csv_name = Path(csv_file_path).stem
    # 移除 "_filtered" 后缀
    if csv_name.endswith('_filtered'):
        csv_name = csv_name[:-9]
    
    mat_filename = f"{csv_name}.mat"
    mat_filepath = Path(output_dir) / mat_filename
    
    # 保存为MAT格式
    mat_data = {
        't': t,
        'q': q,
        'dq': dq,
        'ddq': ddq,
        'tau': tau
    }
    
    scipy.io.savemat(mat_filepath, mat_data)
    
    print(f"  保存为: {mat_filepath}")
    print(f"  变量: t({t.shape}), q{q.shape}, dq{dq.shape}, ddq{ddq.shape}, tau{tau.shape}")
    
    return mat_filepath


def convert_all_processed_data():
    """转换所有预处理后的数据"""
    
    print("=" * 80)
    print("将动力学辨识数据转换为MAT格式")
    print("=" * 80)
    
    # 预处理数据目录
    processed_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control/ic_arm_control/control/di_data_from_gh/processed_data"
    
    # 输出目录
    output_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control/ic_arm_control/control/di_data_from_gh/mat_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有预处理后的CSV文件
    csv_files = []
    for file_path in Path(processed_dir).glob("*_filtered.csv"):
        csv_files.append(str(file_path))
    
    if not csv_files:
        print("❌ 未找到预处理后的CSV文件")
        print(f"请确保在 {processed_dir} 目录下有 *_filtered.csv 文件")
        return
    
    print(f"找到 {len(csv_files)} 个预处理文件:")
    for csv_file in csv_files:
        print(f"  - {Path(csv_file).name}")
    
    # 转换每个文件
    converted_files = []
    for csv_file in csv_files:
        try:
            mat_file = convert_csv_to_mat(csv_file, output_dir)
            converted_files.append(mat_file)
        except Exception as e:
            print(f"❌ 转换失败 {Path(csv_file).name}: {e}")
    
    # 总结
    print(f"\n{'='*80}")
    print(f"转换完成!")
    print(f"{'='*80}")
    print(f"成功转换 {len(converted_files)} 个文件")
    print(f"输出目录: {output_dir}")
    print(f"\n转换后的MAT文件:")
    for mat_file in converted_files:
        print(f"  - {Path(mat_file).name}")
    
    print(f"\n这些MAT文件包含以下变量:")
    print(f"  - t: 时间数组 (n_samples,)")
    print(f"  - q: 关节位置 (n_samples, 6)")
    print(f"  - dq: 关节速度 (n_samples, 6)")
    print(f"  - ddq: 关节加速度 (n_samples, 6)")
    print(f"  - tau: 关节力矩 (n_samples, 6)")
    
    print(f"\n可以在MATLAB中使用:")
    print(f"  load('filename.mat')")
    print(f"  % 数据将自动加载为变量 t, q, dq, ddq, tau")


def main():
    """主函数"""
    convert_all_processed_data()


if __name__ == "__main__":
    main()