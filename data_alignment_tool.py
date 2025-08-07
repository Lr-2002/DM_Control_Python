#!/usr/bin/env python3
"""
动作捕捉数据与机械臂数据对齐工具
处理 base1.ly, ee.ly 和 icarm_positions CSV 文件的时间同步问题
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re

class DataAlignmentTool:
    def __init__(self):
        self.mocap_base_data = None
        self.mocap_ee_data = None
        self.arm_data = None
        
    def parse_ly_file(self, filepath):
        """解析 .ly 动作捕捉文件"""
        print(f"解析文件: {filepath}")
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # 找到数据开始行
        data_start_line = None
        for i, line in enumerate(lines):
            if line.strip().startswith('Frame#'):
                data_start_line = i + 2  # 跳过标题行和单位行
                break
        
        if data_start_line is None:
            raise ValueError("无法找到数据开始行")
        
        # 解析数据
        data_rows = []
        for line in lines[data_start_line:]:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 3:  # 至少包含Frame#, Time, Timestamp
                    try:
                        frame = int(parts[0])
                        time = float(parts[1])
                        timestamp = int(parts[2])
                        
                        # 提取标记点坐标 (只取前几个有效的标记点)
                        markers = []
                        marker_start = 3
                        for i in range(5):  # 最多5个标记点
                            idx = marker_start + i * 12  # 每个标记点12个值
                            if idx + 2 < len(parts):
                                try:
                                    x = float(parts[idx]) if parts[idx] else np.nan
                                    y = float(parts[idx + 1]) if parts[idx + 1] else np.nan
                                    z = float(parts[idx + 2]) if parts[idx + 2] else np.nan
                                    markers.extend([x, y, z])
                                except ValueError:
                                    markers.extend([np.nan, np.nan, np.nan])
                            else:
                                markers.extend([np.nan, np.nan, np.nan])
                        
                        row = [frame, time, timestamp] + markers
                        data_rows.append(row)
                    except (ValueError, IndexError):
                        continue
        
        # 创建DataFrame
        columns = ['Frame', 'Time', 'Timestamp']
        for i in range(5):
            columns.extend([f'M{i+1}_X', f'M{i+1}_Y', f'M{i+1}_Z'])
        
        df = pd.DataFrame(data_rows, columns=columns)
        print(f"解析完成，共 {len(df)} 行数据")
        return df
    
    def load_csv_data(self, filepath):
        """加载机械臂CSV数据"""
        print(f"加载CSV文件: {filepath}")
        df = pd.read_csv(filepath)
        print(f"加载完成，共 {len(df)} 行数据")
        return df
    
    def analyze_timing_differences(self):
        """分析时间差异"""
        print("\n=== 时间同步分析 ===")
        
        if self.mocap_base_data is not None:
            base_start_ts = self.mocap_base_data['Timestamp'].iloc[0]
            base_end_ts = self.mocap_base_data['Timestamp'].iloc[-1]
            base_duration = (base_end_ts - base_start_ts) / 1000  # 转换为秒
            print(f"Base动作捕捉数据:")
            print(f"  开始时间戳: {base_start_ts}")
            print(f"  结束时间戳: {base_end_ts}")
            print(f"  持续时间: {base_duration:.3f} 秒")
            print(f"  数据点数: {len(self.mocap_base_data)}")
            print(f"  采样率: {len(self.mocap_base_data) / base_duration:.1f} Hz")
        
        if self.mocap_ee_data is not None:
            ee_start_ts = self.mocap_ee_data['Timestamp'].iloc[0]
            ee_end_ts = self.mocap_ee_data['Timestamp'].iloc[-1]
            ee_duration = (ee_end_ts - ee_start_ts) / 1000
            print(f"\nEE动作捕捉数据:")
            print(f"  开始时间戳: {ee_start_ts}")
            print(f"  结束时间戳: {ee_end_ts}")
            print(f"  持续时间: {ee_duration:.3f} 秒")
            print(f"  数据点数: {len(self.mocap_ee_data)}")
            print(f"  采样率: {len(self.mocap_ee_data) / ee_duration:.1f} Hz")
        
        if self.arm_data is not None:
            arm_start_ts = self.arm_data['unix_timestamp'].iloc[0]
            arm_end_ts = self.arm_data['unix_timestamp'].iloc[-1]
            arm_duration = (arm_end_ts - arm_start_ts) / 1000
            print(f"\n机械臂数据:")
            print(f"  开始时间戳: {arm_start_ts}")
            print(f"  结束时间戳: {arm_end_ts}")
            print(f"  持续时间: {arm_duration:.3f} 秒")
            print(f"  数据点数: {len(self.arm_data)}")
            print(f"  采样率: {len(self.arm_data) / arm_duration:.1f} Hz")
        
        # 检查时间重叠
        if all([self.mocap_base_data is not None, self.mocap_ee_data is not None, self.arm_data is not None]):
            print(f"\n=== 时间重叠分析 ===")
            mocap_start = min(base_start_ts, ee_start_ts)
            mocap_end = max(base_end_ts, ee_end_ts)
            
            print(f"动作捕捉时间范围: {mocap_start} - {mocap_end}")
            print(f"机械臂时间范围: {arm_start_ts} - {arm_end_ts}")
            
            overlap_start = max(mocap_start, arm_start_ts)
            overlap_end = min(mocap_end, arm_end_ts)
            
            if overlap_start < overlap_end:
                overlap_duration = (overlap_end - overlap_start) / 1000
                print(f"重叠时间范围: {overlap_start} - {overlap_end}")
                print(f"重叠持续时间: {overlap_duration:.3f} 秒")
            else:
                print("警告: 数据时间范围没有重叠!")
    
    def synchronize_data(self, time_window_ms=50):
        """同步数据到相同的时间基准 - 以mocap为基准，插值电机数据"""
        print(f"\n=== 数据同步 (时间窗口: {time_window_ms}ms) ===")
        print("策略: 以动作捕捉数据为基准，插值电机数据到mocap时间戳")
        
        if not all([self.mocap_base_data is not None, self.mocap_ee_data is not None, self.arm_data is not None]):
            print("错误: 需要加载所有三个数据文件")
            return None
        
        # 找到公共时间范围
        mocap_start = min(self.mocap_base_data['Timestamp'].iloc[0], self.mocap_ee_data['Timestamp'].iloc[0])
        mocap_end = max(self.mocap_base_data['Timestamp'].iloc[-1], self.mocap_ee_data['Timestamp'].iloc[-1])
        arm_start = self.arm_data['unix_timestamp'].iloc[0]
        arm_end = self.arm_data['unix_timestamp'].iloc[-1]
        
        sync_start = max(mocap_start, arm_start)
        sync_end = min(mocap_end, arm_end)
        
        if sync_start >= sync_end:
            print("错误: 没有公共时间范围")
            return None
        
        print(f"同步时间范围: {sync_start} - {sync_end}")
        
        # 过滤到公共时间范围
        base_sync = self.mocap_base_data[
            (self.mocap_base_data['Timestamp'] >= sync_start) & 
            (self.mocap_base_data['Timestamp'] <= sync_end)
        ].copy()
        
        ee_sync = self.mocap_ee_data[
            (self.mocap_ee_data['Timestamp'] >= sync_start) & 
            (self.mocap_ee_data['Timestamp'] <= sync_end)
        ].copy()
        
        arm_sync = self.arm_data[
            (self.arm_data['unix_timestamp'] >= sync_start) & 
            (self.arm_data['unix_timestamp'] <= sync_end)
        ].copy()
        
        print(f"过滤后数据量: base={len(base_sync)}, ee={len(ee_sync)}, arm={len(arm_sync)}")
        
        # 以动作捕捉数据的时间戳为基准进行插值
        synchronized_data = []
        
        # 确保base和ee数据时间戳一致
        if len(base_sync) != len(ee_sync):
            print("警告: base和ee数据长度不一致，使用较短的长度")
            min_len = min(len(base_sync), len(ee_sync))
            base_sync = base_sync.iloc[:min_len]
            ee_sync = ee_sync.iloc[:min_len]
        
        # 为每个mocap时间戳找到对应的电机数据
        for i in range(len(base_sync)):
            base_row = base_sync.iloc[i]
            ee_row = ee_sync.iloc[i]
            target_ts = base_row['Timestamp']  # 以mocap时间戳为基准
            
            # 找到最接近的电机数据点
            arm_idx = np.argmin(np.abs(arm_sync['unix_timestamp'] - target_ts))
            arm_ts_diff = abs(arm_sync.iloc[arm_idx]['unix_timestamp'] - target_ts)
            
            # 只有在时间差小于窗口时才包含数据
            if arm_ts_diff <= time_window_ms:
                arm_row = arm_sync.iloc[arm_idx]
                
                sync_row = {
                    'timestamp': target_ts,
                    'mocap_frame': base_row['Frame'],
                    'datetime': pd.to_datetime(target_ts, unit='ms').strftime('%Y-%m-%d %H:%M:%S'),
                    # 机械臂数据
                    'm1_deg': arm_row['m1_deg'], 'm1_rad': arm_row['m1_rad'],
                    'm2_deg': arm_row['m2_deg'], 'm2_rad': arm_row['m2_rad'],
                    'm3_deg': arm_row['m3_deg'], 'm3_rad': arm_row['m3_rad'],
                    'm4_deg': arm_row['m4_deg'], 'm4_rad': arm_row['m4_rad'],
                    'm5_deg': arm_row['m5_deg'], 'm5_rad': arm_row['m5_rad'],
                    'arm_timestamp_diff_ms': arm_ts_diff,
                }
                
                # 添加base动作捕捉数据
                for col in base_sync.columns:
                    if col not in ['Frame', 'Time', 'Timestamp']:
                        sync_row[f'base_{col}'] = base_row[col]
                
                # 添加ee动作捕捉数据
                for col in ee_sync.columns:
                    if col not in ['Frame', 'Time', 'Timestamp']:
                        sync_row[f'ee_{col}'] = ee_row[col]
                
                synchronized_data.append(sync_row)
        
        sync_df = pd.DataFrame(synchronized_data)
        print(f"同步完成，共 {len(sync_df)} 行数据 (约{len(sync_df)/90:.1f}秒，90Hz)")
        
        if len(sync_df) > 0:
            avg_diff = sync_df['arm_timestamp_diff_ms'].mean()
            max_diff = sync_df['arm_timestamp_diff_ms'].max()
            print(f"电机数据时间差统计: 平均={avg_diff:.1f}ms, 最大={max_diff:.1f}ms")
        
        return sync_df
    
    def save_synchronized_data(self, sync_df, output_path):
        """保存同步后的数据"""
        if sync_df is not None:
            sync_df.to_csv(output_path, index=False)
            print(f"同步数据已保存到: {output_path}")
    
    def plot_timing_comparison(self):
        """绘制时间对比图"""
        plt.figure(figsize=(15, 10))
        
        # 子图1: 时间戳对比
        plt.subplot(2, 2, 1)
        if self.mocap_base_data is not None:
            plt.plot(self.mocap_base_data['Frame'], self.mocap_base_data['Timestamp'], 
                    'b-', label='Base MoCap', alpha=0.7)
        if self.mocap_ee_data is not None:
            plt.plot(self.mocap_ee_data['Frame'], self.mocap_ee_data['Timestamp'], 
                    'r-', label='EE MoCap', alpha=0.7)
        if self.arm_data is not None:
            plt.plot(range(len(self.arm_data)), self.arm_data['unix_timestamp'], 
                    'g-', label='Arm Data', alpha=0.7)
        plt.xlabel('Data Point Index')
        plt.ylabel('Timestamp')
        plt.title('Timestamp Comparison')
        plt.legend()
        plt.grid(True)
        
        # 子图2: 采样间隔
        plt.subplot(2, 2, 2)
        if self.mocap_base_data is not None and len(self.mocap_base_data) > 1:
            base_intervals = np.diff(self.mocap_base_data['Timestamp'])
            plt.hist(base_intervals, bins=50, alpha=0.7, label='Base MoCap')
        if self.mocap_ee_data is not None and len(self.mocap_ee_data) > 1:
            ee_intervals = np.diff(self.mocap_ee_data['Timestamp'])
            plt.hist(ee_intervals, bins=50, alpha=0.7, label='EE MoCap')
        if self.arm_data is not None and len(self.arm_data) > 1:
            arm_intervals = np.diff(self.arm_data['unix_timestamp'])
            plt.hist(arm_intervals, bins=50, alpha=0.7, label='Arm Data')
        plt.xlabel('Sampling Interval (ms)')
        plt.ylabel('Frequency')
        plt.title('Sampling Interval Distribution')
        plt.legend()
        plt.grid(True)
        
        # 子图3: 机械臂关节角度
        plt.subplot(2, 2, 3)
        if self.arm_data is not None:
            time_rel = (self.arm_data['unix_timestamp'] - self.arm_data['unix_timestamp'].iloc[0]) / 1000
            plt.plot(time_rel, self.arm_data['m1_deg'], label='Joint 1')
            plt.plot(time_rel, self.arm_data['m2_deg'], label='Joint 2')
            plt.plot(time_rel, self.arm_data['m3_deg'], label='Joint 3')
            plt.plot(time_rel, self.arm_data['m4_deg'], label='Joint 4')
            plt.plot(time_rel, self.arm_data['m5_deg'], label='Joint 5')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.title('Robot Arm Joint Angles')
        plt.legend()
        plt.grid(True)
        
        # 子图4: 动作捕捉标记点位置 (以第一个标记点为例)
        plt.subplot(2, 2, 4)
        if self.mocap_base_data is not None:
            time_rel = (self.mocap_base_data['Timestamp'] - self.mocap_base_data['Timestamp'].iloc[0]) / 1000
            plt.plot(time_rel, self.mocap_base_data['M1_X'], 'b-', label='Base M1_X')
            plt.plot(time_rel, self.mocap_base_data['M1_Y'], 'b--', label='Base M1_Y')
            plt.plot(time_rel, self.mocap_base_data['M1_Z'], 'b:', label='Base M1_Z')
        if self.mocap_ee_data is not None:
            time_rel = (self.mocap_ee_data['Timestamp'] - self.mocap_ee_data['Timestamp'].iloc[0]) / 1000
            plt.plot(time_rel, self.mocap_ee_data['M1_X'], 'r-', label='EE M1_X')
            plt.plot(time_rel, self.mocap_ee_data['M1_Y'], 'r--', label='EE M1_Y')
            plt.plot(time_rel, self.mocap_ee_data['M1_Z'], 'r:', label='EE M1_Z')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (mm)')
        plt.title('Motion Capture Marker Positions')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('/Users/lr-2002/project/instantcreation/IC_arm_control/data_timing_analysis.png', dpi=300)
        plt.show()

def main():
    """主函数"""
    tool = DataAlignmentTool()
    
    # 文件路径
    base_file = '/Users/lr-2002/project/instantcreation/IC_arm_control/base1.ly'
    ee_file = '/Users/lr-2002/project/instantcreation/IC_arm_control/ee.ly'
    csv_file = '/Users/lr-2002/project/instantcreation/IC_arm_control/icarm_positions_20250807_140250.csv'
    
    try:
        # 加载数据
        print("=== 加载数据文件 ===")
        tool.mocap_base_data = tool.parse_ly_file(base_file)
        tool.mocap_ee_data = tool.parse_ly_file(ee_file)
        tool.arm_data = tool.load_csv_data(csv_file)
        
        # 分析时间差异
        tool.analyze_timing_differences()
        
        # 同步数据
        synchronized_data = tool.synchronize_data(time_window_ms=50)
        
        if synchronized_data is not None:
            # 保存同步数据
            output_file = '/Users/lr-2002/project/instantcreation/IC_arm_control/synchronized_data.csv'
            tool.save_synchronized_data(synchronized_data, output_file)
            
            print(f"\n=== 同步数据统计 ===")
            print(f"同步数据点数: {len(synchronized_data)}")
            print(f"数据列数: {len(synchronized_data.columns)}")
            print(f"时间范围: {synchronized_data['timestamp'].min()} - {synchronized_data['timestamp'].max()}")
            
        # 绘制对比图
        print("\n=== 生成可视化图表 ===")
        tool.plot_timing_comparison()
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
