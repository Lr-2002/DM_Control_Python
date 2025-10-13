#!/usr/bin/env python3
"""
Step 1: Data Preprocessing
按照 shamilmamedov/dynamic_calibration 的逻辑进行数据预处理

流程：
1. 读取原始日志数据 (q, dq, tau)
2. 滤波处理 (零相位滤波器)
3. 估计加速度 (中心差分法)
4. 保存处理后的数据
"""

import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt
from pathlib import Path


class DataPreprocessor:
    """数据预处理器 - 实现 filterData.m 的功能"""
    
    def __init__(self, cutoff_freq=10.0, fs=250.0, filter_order=4):
        """
        初始化滤波器参数
        
        Args:
            cutoff_freq: 截止频率 (Hz)
            fs: 采样频率 (Hz)
            filter_order: 滤波器阶数
        """
        self.cutoff_freq = cutoff_freq
        self.fs = fs
        self.filter_order = filter_order
        
        # 设计 Butterworth 低通滤波器
        nyquist = 0.5 * fs
        normal_cutoff = cutoff_freq / nyquist
        self.b, self.a = butter(filter_order, normal_cutoff, btype='low')
        
        print(f"滤波器参数:")
        print(f"  截止频率: {cutoff_freq} Hz")
        print(f"  采样频率: {fs} Hz")
        print(f"  滤波器阶数: {filter_order}")
    
    def zero_phase_filter(self, data):
        """
        零相位滤波 - 使用 filtfilt (前向-后向滤波)
        
        Args:
            data: 原始数据 (1D array)
            
        Returns:
            filtered_data: 滤波后的数据
        """
        if len(data) < 3 * self.filter_order:
            print(f"警告: 数据长度 {len(data)} 太短，跳过滤波")
            return data
        
        # 零相位滤波 - 避免引入相位延迟
        filtered_data = filtfilt(self.b, self.a, data)
        return filtered_data
    
    def estimate_acceleration_central_diff(self, velocity, time_array):
        """
        使用中心差分法估计加速度 (基于实际时间戳)
        
        Args:
            velocity: 速度数据 (1D array)
            time_array: 时间戳数组 (1D array)
            
        Returns:
            acceleration: 加速度估计值
        """
        n = len(velocity)
        acceleration = np.zeros(n)
        
        # 中心差分法 (更准确) - 使用实际时间间隔
        for i in range(1, n - 1):
            dt_forward = time_array[i + 1] - time_array[i]
            dt_backward = time_array[i] - time_array[i - 1]
            dt_total = time_array[i + 1] - time_array[i - 1]
            
            # 中心差分
            acceleration[i] = (velocity[i + 1] - velocity[i - 1]) / dt_total
        
        # 边界处理 - 使用前向/后向差分
        dt_0 = time_array[1] - time_array[0]
        acceleration[0] = (velocity[1] - velocity[0]) / dt_0
        
        dt_end = time_array[-1] - time_array[-2]
        acceleration[-1] = (velocity[-1] - velocity[-2]) / dt_end
        
        return acceleration
    
    def process_trajectory_data(self, q_raw, dq_raw, tau_raw, time_array, 
                                filter_position=True, 
                                filter_velocity=True,
                                filter_torque=True,
                                filter_acceleration=True):
        """
        处理轨迹数据 - 完整的 filterData.m 流程
        
        Args:
            q_raw: 原始位置数据 (n_samples, n_joints)
            dq_raw: 原始速度数据 (n_samples, n_joints)
            tau_raw: 原始力矩数据 (n_samples, n_joints)
            time_array: 时间戳数组 (n_samples,)
            filter_position: 是否滤波位置
            filter_velocity: 是否滤波速度
            filter_torque: 是否滤波力矩
            filter_acceleration: 是否滤波加速度
            
        Returns:
            q_filtered: 滤波后的位置
            dq_filtered: 滤波后的速度
            ddq_filtered: 估计并滤波后的加速度
            tau_filtered: 滤波后的力矩
        """
        n_samples, n_joints = q_raw.shape
        
        # 计算平均时间步长用于显示
        avg_dt = np.mean(np.diff(time_array))
        
        print(f"\n数据预处理:")
        print(f"  数据点数: {n_samples}")
        print(f"  关节数: {n_joints}")
        print(f"  平均时间步长: {avg_dt:.6f} s ({1/avg_dt:.1f} Hz)")
        print(f"  时间范围: {time_array[0]:.3f} - {time_array[-1]:.3f} s")
        print(f"  总时长: {time_array[-1] - time_array[0]:.3f} s")
        
        # 1. 位置滤波 (根据传感器质量决定)
        if filter_position:
            print("  滤波位置数据...")
            q_filtered = np.zeros_like(q_raw)
            for j in range(n_joints):
                q_filtered[:, j] = self.zero_phase_filter(q_raw[:, j])
        else:
            print("  跳过位置滤波")
            q_filtered = q_raw.copy()
        
        # 2. 速度滤波 (根据传感器质量决定)
        if filter_velocity:
            print("  滤波速度数据...")
            dq_filtered = np.zeros_like(dq_raw)
            for j in range(n_joints):
                dq_filtered[:, j] = self.zero_phase_filter(dq_raw[:, j])
        else:
            print("  跳过速度滤波")
            dq_filtered = dq_raw.copy()
        
        # 3. 力矩滤波 (通常噪声较大，必须滤波)
        if filter_torque:
            print("  滤波力矩数据...")
            tau_filtered = np.zeros_like(tau_raw)
            for j in range(n_joints):
                tau_filtered[:, j] = self.zero_phase_filter(tau_raw[:, j])
        else:
            print("  跳过力矩滤波")
            tau_filtered = tau_raw.copy()
        
        # 4. 加速度估计 (使用中心差分法，基于实际时间戳)
        print("  估计加速度 (中心差分法，基于时间戳)...")
        ddq_raw = np.zeros_like(dq_filtered)
        for j in range(n_joints):
            ddq_raw[:, j] = self.estimate_acceleration_central_diff(dq_filtered[:, j], time_array)
        
        # 5. 加速度滤波 (必须滤波)
        if filter_acceleration:
            print("  滤波加速度数据...")
            ddq_filtered = np.zeros_like(ddq_raw)
            for j in range(n_joints):
                ddq_filtered[:, j] = self.zero_phase_filter(ddq_raw[:, j])
        else:
            print("  跳过加速度滤波")
            ddq_filtered = ddq_raw.copy()
        
        print("  ✓ 数据预处理完成")
        
        return q_filtered, dq_filtered, ddq_filtered, tau_filtered


def load_converted_data(csv_file):
    """
    从转换后的CSV文件加载数据
    
    Args:
        csv_file: 转换后的CSV文件路径
        
    Returns:
        q, dq, tau, time_array: 位置、速度、力矩和时间数组
    """
    print(f"\n加载转换后的数据: {Path(csv_file).name}")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"文件不存在: {csv_file}")
    
    # 读取数据
    df = pd.read_csv(csv_file)
    
    print(f"  数据点数: {len(df)}")
    print(f"  列数: {len(df.columns)}")
    
    # 提取数据 (前6个关节)
    n_joints = 6
    n_samples = len(df)
    
    q = np.zeros((n_samples, n_joints))
    dq = np.zeros((n_samples, n_joints))
    tau = np.zeros((n_samples, n_joints))
    
    for j in range(n_joints):
        motor_id = j + 1
        q[:, j] = df[f'm{motor_id}_pos_actual'].values
        dq[:, j] = df[f'm{motor_id}_vel_actual'].values
        tau[:, j] = df[f'm{motor_id}_torque'].values
    
    # 获取时间数组
    if 'time' in df.columns:
        time_array = df['time'].values
    else:
        # 假设 500Hz (convert_log_data.py 中的设置)
        time_array = np.arange(n_samples) * 0.002
    
    avg_dt = np.mean(np.diff(time_array))
    print(f"  提取数据: {n_samples} 样本, {n_joints} 关节")
    print(f"  平均时间步长: {avg_dt:.6f} s ({1/avg_dt:.1f} Hz)")
    
    return q, dq, tau, time_array


def save_processed_data(output_file, q, dq, ddq, tau, time_array):
    """
    保存处理后的数据
    
    Args:
        output_file: 输出文件路径
        q, dq, ddq, tau: 处理后的数据
        time_array: 时间数组
    """
    n_samples, n_joints = q.shape
    
    # 构建 DataFrame
    data_dict = {'time': time_array}
    
    for j in range(n_joints):
        joint_id = j + 1
        data_dict[f'q{joint_id}'] = q[:, j]
        data_dict[f'dq{joint_id}'] = dq[:, j]
        data_dict[f'ddq{joint_id}'] = ddq[:, j]
        data_dict[f'tau{joint_id}'] = tau[:, j]
    
    df = pd.DataFrame(data_dict)
    df.to_csv(output_file, index=False)
    
    print(f"\n保存处理后的数据: {output_file}")
    print(f"  形状: {df.shape}")
    print(f"  列: {list(df.columns)}")


def main():
    """主函数 - Step 1: 数据预处理"""
    
    print("=" * 80)
    print("Step 1: 数据预处理 (Data Preprocessing)")
    print("按照 shamilmamedov/dynamic_calibration 的 filterData.m 逻辑")
    print("=" * 80)
    
    # 输入转换后的数据文件
    converted_files = [
        "/Users/lr-2002/project/instantcreation/IC_arm_control/dyn_ana/dynamics_20251011_131131_ic_arm_control.csv",
        "/Users/lr-2002/project/instantcreation/IC_arm_control/dyn_ana/dynamics_20251011_131249_ic_arm_control.csv",
        "/Users/lr-2002/project/instantcreation/IC_arm_control/dyn_ana/dynamics_20251011_131436_ic_arm_control.csv",
    ]
    
    # 输出目录
    output_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control/ic_arm_control/control/di_data_from_gh/processed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建数据预处理器
    preprocessor = DataPreprocessor(
        cutoff_freq=10.0,  # 截止频率 (需要根据数据调整)
        fs=100.0,          # 采样频率 (convert_log_data.py 使用 500Hz)
        filter_order=4     # 滤波器阶数
    )
    
    # 处理每个转换后的文件
    processed_files = []
    
    for i, csv_file in enumerate(converted_files):
        print(f"\n{'='*80}")
        print(f"处理文件 {i+1}/{len(converted_files)}: {Path(csv_file).name}")
        print(f"{'='*80}")
        
        try:
            # 1. 加载转换后的数据
            q_raw, dq_raw, tau_raw, time_array = load_converted_data(csv_file)
            
            # 2. 数据预处理
            q_filtered, dq_filtered, ddq_filtered, tau_filtered = preprocessor.process_trajectory_data(
                q_raw, dq_raw, tau_raw, time_array,
                filter_position=True,      # 滤波位置
                filter_velocity=True,      # 滤波速度
                filter_torque=True,        # 滤波力矩 (必须)
                filter_acceleration=True   # 滤波加速度 (必须)
            )
            
            # 3. 保存处理后的数据
            file_name = Path(csv_file).stem  # 去掉扩展名
            output_file = os.path.join(output_dir, f"{file_name}_filtered.csv")
            save_processed_data(output_file, q_filtered, dq_filtered, ddq_filtered, tau_filtered, time_array)
            
            processed_files.append(output_file)
            
            # 4. 数据质量检查
            print(f"\n数据质量检查:")
            print(f"  位置范围: [{np.min(q_filtered):.4f}, {np.max(q_filtered):.4f}] rad")
            print(f"  速度范围: [{np.min(dq_filtered):.4f}, {np.max(dq_filtered):.4f}] rad/s")
            print(f"  加速度范围: [{np.min(ddq_filtered):.4f}, {np.max(ddq_filtered):.4f}] rad/s²")
            print(f"  力矩范围: [{np.min(tau_filtered):.4f}, {np.max(tau_filtered):.4f}] Nm")
            
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结
    print(f"\n{'='*80}")
    print(f"Step 1 完成!")
    print(f"{'='*80}")
    print(f"处理了 {len(processed_files)} 个日志文件")
    print(f"输出目录: {output_dir}")
    print(f"\n处理后的文件:")
    for f in processed_files:
        print(f"  - {f}")
    
    print(f"\n下一步: 运行 step2_parameter_estimation.py 进行参数估计")


if __name__ == "__main__":
    main()
