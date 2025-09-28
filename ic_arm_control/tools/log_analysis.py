import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class MotorLogAnalyzer:
    """分析电机控制日志数据，包括轨迹对比和误差分析"""
    
    def __init__(self, log_dir: str, alignment_threshold_ms: float = 1.0):
        """
        初始化日志分析器
        
        Args:
            log_dir: 日志文件夹路径
            alignment_threshold_ms: 时间戳对齐阈值（毫秒）
        """
        self.log_dir = log_dir
        self.alignment_threshold_ms = alignment_threshold_ms
        self.target_data = None
        self.state_data = None
        self.aligned_data = None
        self.motor_count = 9  # 初始值，会在load_data中动态调整
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载目标轨迹和实际状态数据
        
        Returns:
            target_data, state_data: 目标数据和状态数据的DataFrame
        """
        target_file = os.path.join(self.log_dir, 'joint_commands.csv')
        state_file = os.path.join(self.log_dir, 'motor_states.csv')
        
        if not os.path.exists(target_file) or not os.path.exists(state_file):
            raise FileNotFoundError(f"日志文件不存在: {target_file} 或 {state_file}")
        
        # 读取目标数据 - 处理格式错误的问题
        print(f"正在读取目标数据文件: {target_file}")
        
        # 先检查文件的实际格式
        with open(target_file, 'r') as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip()
            print(f"CSV头部: {first_line}")
            print(f"第一行数据: {second_line[:100]}...")
        
        # 检查是否是格式错误的CSV（数据都在第一列）
        if ' ' in second_line and ',' not in second_line.split(',')[0]:
            print("检测到CSV格式错误：数据被写入了单个列中")
            print("尝试修复CSV格式...")
            self._fix_malformed_csv(target_file)
        
        # 检查CSV文件的字段数匹配问题
        with open(target_file, 'r') as f:
            header_line = f.readline().strip()
            data_line = f.readline().strip()
            header_fields = len(header_line.split(','))
            data_fields = len(data_line.split(','))
            
            print(f"CSV字段数检查: 头部={header_fields}, 数据={data_fields}")
            
            if header_fields != data_fields:
                print(f"警告: CSV字段数不匹配! 头部有{header_fields}个字段，数据有{data_fields}个字段")
                print("这会导致pandas读取错误，尝试修复...")
                
                # 使用数据行的字段数来读取，忽略头部不匹配的问题
                try:
                    self.target_data = pd.read_csv(target_file, header=None, skiprows=1)
                    # 手动设置列名（只使用前面匹配的部分）
                    header_names = header_line.split(',')
                    if len(header_names) < len(self.target_data.columns):
                        # 如果数据列更多，为额外的列生成名称
                        extra_cols = len(self.target_data.columns) - len(header_names)
                        for i in range(extra_cols):
                            header_names.append(f'extra_col_{i+1}')
                    
                    self.target_data.columns = header_names[:len(self.target_data.columns)]
                    
                    # 转换数据类型：时间戳为字符串，其他为数值
                    self.target_data['timestamp'] = self.target_data['timestamp'].astype(str)
                    for col in self.target_data.columns:
                        if col != 'timestamp':
                            self.target_data[col] = pd.to_numeric(self.target_data[col], errors='coerce')
                    
                    print(f"修复后的数据形状: {self.target_data.shape}")
                    
                except Exception as e:
                    print(f"修复失败: {e}")
                    # 回退到原始方法
                    self.target_data = pd.read_csv(target_file, dtype={'timestamp': str})
            else:
                # 字段数匹配，正常读取
                self.target_data = pd.read_csv(target_file, dtype={'timestamp': str})
        
        print(f"目标数据形状: {self.target_data.shape}")
        print(f"目标数据列名: {list(self.target_data.columns)}")
        
        # 检查时间戳列的内容
        if 'timestamp' in self.target_data.columns:
            print(f"原始目标数据时间戳示例: {self.target_data['timestamp'].iloc[0]}")
            print(f"时间戳类型: {type(self.target_data['timestamp'].iloc[0])}")
            
            # 如果时间戳是数值，说明格式有问题
            if isinstance(self.target_data['timestamp'].iloc[0], (int, float)):
                print("警告: 时间戳列包含数值而不是时间字符串，CSV格式可能有问题")
                # 尝试从列名中提取时间戳
                self._extract_timestamp_from_malformed_data()
            else:
                self.target_data['timestamp'] = pd.to_datetime(self.target_data['timestamp'], errors='coerce')
                print(f"解析后目标数据时间戳示例: {self.target_data['timestamp'].iloc[0]}")
        else:
            print("错误: 找不到timestamp列")
        
        # 读取状态数据
        self.state_data = pd.read_csv(state_file)
        print(f"原始状态数据时间戳示例: {self.state_data['timestamp'].iloc[0]}")
        self.state_data['timestamp'] = pd.to_datetime(self.state_data['timestamp'], errors='coerce')
        print(f"解析后状态数据时间戳示例: {self.state_data['timestamp'].iloc[0]}")
        
        print(f"加载数据完成:")
        print(f"  目标数据: {len(self.target_data)} 条记录")
        print(f"  状态数据: {len(self.state_data)} 条记录")
        
        # 检查时间戳有效性
        target_min = self.target_data['timestamp'].min()
        target_max = self.target_data['timestamp'].max()
        state_min = self.state_data['timestamp'].min()
        state_max = self.state_data['timestamp'].max()
        
        print(f"  目标数据时间范围: {target_min} - {target_max}")
        print(f"  状态数据时间范围: {state_min} - {state_max}")
        
        # 检查是否有无效时间戳
        epoch_time = pd.Timestamp('1970-01-01 00:00:00')
        target_invalid = (self.target_data['timestamp'] == epoch_time).sum()
        state_invalid = (self.state_data['timestamp'] == epoch_time).sum()
        
        if target_invalid > 0:
            print(f"  警告: 目标数据中有 {target_invalid} 条无效时间戳 (1970-01-01)")
        if state_invalid > 0:
            print(f"  警告: 状态数据中有 {state_invalid} 条无效时间戳 (1970-01-01)")
        
        # 检查时间范围重叠
        if target_max < state_min or state_max < target_min:
            print("  警告: 目标数据和状态数据的时间范围没有重叠！")
        
        # 过滤掉无效时间戳
        if target_invalid > 0:
            self.target_data = self.target_data[self.target_data['timestamp'] != epoch_time]
            print(f"  已过滤目标数据中的无效时间戳，剩余 {len(self.target_data)} 条记录")
        
        if state_invalid > 0:
            self.state_data = self.state_data[self.state_data['timestamp'] != epoch_time]
            print(f"  已过滤状态数据中的无效时间戳，剩余 {len(self.state_data)} 条记录")
        
        # 动态检测实际的电机数量
        self._detect_motor_count()
        
        return self.target_data, self.state_data
    
    def _detect_motor_count(self):
        """根据CSV数据动态检测电机数量"""
        # 从目标数据中检测电机数量
        target_position_cols = [col for col in self.target_data.columns if col.startswith('target_position_motor_')]
        target_motor_count = len(target_position_cols)
        
        # 从状态数据中检测电机数量  
        state_position_cols = [col for col in self.state_data.columns if col.startswith('position_motor_')]
        state_motor_count = len(state_position_cols)
        
        print(f"检测到的电机数量:")
        print(f"  目标数据: {target_motor_count} 个电机")
        print(f"  状态数据: {state_motor_count} 个电机")
        
        # 使用较小的数量以确保数据对齐时不会出错
        detected_motor_count = min(target_motor_count, state_motor_count)
        
        if detected_motor_count != self.motor_count:
            print(f"  调整电机数量: {self.motor_count} -> {detected_motor_count}")
            self.motor_count = detected_motor_count
        else:
            print(f"  确认电机数量: {self.motor_count}")
        
        # 验证检测结果
        if self.motor_count == 0:
            print("警告: 未检测到有效的电机数据列!")
        elif self.motor_count < 6:
            print(f"警告: 检测到的电机数量({self.motor_count})少于预期的6个")
        
        return self.motor_count
    
    def _fix_malformed_csv(self, csv_file_path: str):
        """修复格式错误的CSV文件"""
        print("正在修复CSV格式...")
        backup_file = csv_file_path + '.backup'
        
        # 备份原文件
        import shutil
        shutil.copy2(csv_file_path, backup_file)
        print(f"原文件已备份到: {backup_file}")
        
        # 读取并修复
        with open(csv_file_path, 'r') as f:
            lines = f.readlines()
        
        fixed_lines = []
        for i, line in enumerate(lines):
            line = line.strip()
            if i == 0:  # 头部行
                fixed_lines.append(line)
            else:
                # 数据行：如果包含空格但逗号很少，说明格式错误
                if ' ' in line and line.count(',') < 10:
                    # 尝试用空格分割并重新用逗号连接
                    parts = line.split()
                    if len(parts) > 1:
                        # 第一部分是时间戳，其余是数据
                        timestamp = parts[0]
                        data_parts = parts[1:]
                        fixed_line = timestamp + ',' + ','.join(data_parts)
                        fixed_lines.append(fixed_line)
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
        
        # 写回修复后的文件
        with open(csv_file_path, 'w') as f:
            for line in fixed_lines:
                f.write(line + '\n')
        
        print("CSV格式修复完成")
    
    def _extract_timestamp_from_malformed_data(self):
        """从格式错误的数据中提取时间戳"""
        print("尝试从格式错误的数据中提取时间戳...")
        
        # 如果数据都在第一列，尝试解析
        if len(self.target_data.columns) == 1:
            # 数据可能都在列名中
            column_name = self.target_data.columns[0]
            if 'T' in column_name and ':' in column_name:
                # 列名包含时间戳
                timestamp_str = column_name.split()[0]  # 取第一部分作为时间戳
                print(f"从列名中提取的时间戳: {timestamp_str}")
                
                # 创建新的DataFrame结构
                # 这里需要更复杂的解析逻辑
                print("警告: 数据格式严重错误，需要手动修复CSV文件")
    
    def align_data(self) -> pd.DataFrame:
        """
        按照时间戳对齐目标和状态数据
        规则：先有target，然后很快（<1ms）有对应的state
        
        Returns:
            aligned_data: 对齐后的数据
        """
        if self.target_data is None or self.state_data is None:
            raise ValueError("请先调用load_data()加载数据")
        
        aligned_records = []
        threshold_ns = self.alignment_threshold_ms * 1e6  # 转换为纳秒
        
        for _, target_row in self.target_data.iterrows():
            target_time = target_row['timestamp']
            
            # 查找在阈值时间内的状态数据
            time_diff = (self.state_data['timestamp'] - target_time).dt.total_seconds() * 1000
            valid_states = self.state_data[(time_diff >= 0) & (time_diff <= self.alignment_threshold_ms)]
            
            if len(valid_states) == 0:
                print(f"警告: 时间戳 {target_time} 没有找到对应的状态数据")
                continue
            
            # 选择时间最接近的状态数据
            closest_state = valid_states.iloc[0]
            time_delay = (closest_state['timestamp'] - target_time).total_seconds() * 1000
            
            if time_delay > self.alignment_threshold_ms:
                raise ValueError(f"时间戳对齐失败: 延迟 {time_delay:.3f}ms 超过阈值 {self.alignment_threshold_ms}ms")
            
            # 构建对齐记录
            aligned_record = {
                'timestamp': target_time,
                'time_delay_ms': time_delay
            }
            
            # 添加每个电机的目标和实际数据
            for motor_id in range(1, self.motor_count + 1):
                aligned_record[f'target_position_motor_{motor_id}'] = target_row[f'target_position_motor_{motor_id}']
                aligned_record[f'actual_position_motor_{motor_id}'] = closest_state[f'position_motor_{motor_id}']
                aligned_record[f'target_velocity_motor_{motor_id}'] = target_row[f'target_velocity_motor_{motor_id}']
                aligned_record[f'actual_velocity_motor_{motor_id}'] = closest_state[f'velocity_motor_{motor_id}']
                aligned_record[f'target_torque_motor_{motor_id}'] = target_row[f'target_torque_motor_{motor_id}']
                aligned_record[f'actual_torque_motor_{motor_id}'] = closest_state[f'torque_motor_{motor_id}']
            
            aligned_records.append(aligned_record)
        
        self.aligned_data = pd.DataFrame(aligned_records)
        print(f"数据对齐完成: {len(self.aligned_data)} 条对齐记录")
        
        # 检查是否有对齐的数据
        if len(self.aligned_data) == 0:
            print("警告: 没有成功对齐的数据！")
            print("可能的原因:")
            print("1. 时间戳格式不正确或为无效值")
            print("2. 目标数据和状态数据的时间范围不匹配")
            print("3. 时间对齐阈值太小")
            return self.aligned_data
        
        print(f"平均时间延迟: {self.aligned_data['time_delay_ms'].mean():.3f}ms")
        
        return self.aligned_data
    
    def calculate_error_statistics(self) -> Dict:
        """计算误差统计信息"""
        if self.aligned_data is None or len(self.aligned_data) == 0:
            return {'error': '没有对齐的数据'}
        
        stats = {}
        
        for motor_id in range(1, self.motor_count + 1):
            target_pos_col = f'target_position_motor_{motor_id}'
            actual_pos_col = f'actual_position_motor_{motor_id}'
            target_vel_col = f'target_velocity_motor_{motor_id}'
            actual_vel_col = f'actual_velocity_motor_{motor_id}'
            
            if target_pos_col in self.aligned_data.columns and actual_pos_col in self.aligned_data.columns:
                # 位置误差
                pos_error = self.aligned_data[actual_pos_col] - self.aligned_data[target_pos_col]
                
                # 速度误差
                vel_error = None
                if target_vel_col in self.aligned_data.columns and actual_vel_col in self.aligned_data.columns:
                    vel_error = self.aligned_data[actual_vel_col] - self.aligned_data[target_vel_col]
                
                motor_stats = {
                    'position_error': {
                        'mean': float(pos_error.mean()),
                        'std': float(pos_error.std()),
                        'rms': float(np.sqrt(np.mean(pos_error**2))),
                        'max_abs': float(pos_error.abs().max())
                    }
                }
                
                if vel_error is not None:
                    motor_stats['velocity_error'] = {
                        'mean': float(vel_error.mean()),
                        'std': float(vel_error.std()),
                        'rms': float(np.sqrt(np.mean(vel_error**2))),
                        'max_abs': float(vel_error.abs().max())
                    }
                
                stats[f'motor_{motor_id}'] = motor_stats
        
        return stats
    
    def plot_motor_trajectories(self, motor_ids: List[int] = None, save_plots: bool = True) -> None:
        """
        绘制电机轨迹对比图
        
        Args:
            motor_ids: 要绘制的电机ID列表，None表示绘制所有电机
            save_plots: 是否保存图片
        """
        if self.aligned_data is None:
            raise ValueError("请先调用align_data()对齐数据")
        
        if motor_ids is None:
            motor_ids = list(range(1, self.motor_count + 1))
        
        # 转换时间戳为相对时间（秒）
        start_time = self.aligned_data['timestamp'].iloc[0]
        time_seconds = (self.aligned_data['timestamp'] - start_time).dt.total_seconds()
        
        # 为每个电机创建子图 - 只显示位置
        fig, axes = plt.subplots(len(motor_ids), 1, figsize=(12, 3 * len(motor_ids)))
        if len(motor_ids) == 1:
            axes = [axes]  # 确保axes是列表格式
        
        for i, motor_id in enumerate(motor_ids):
            # Position trajectory only
            axes[i].plot(time_seconds, self.aligned_data[f'target_position_motor_{motor_id}'], 
                        'b-', label='Target Position', linewidth=2)
            axes[i].plot(time_seconds, self.aligned_data[f'actual_position_motor_{motor_id}'], 
                        'r--', label='Actual Position', linewidth=2)
            axes[i].set_title(f'Motor {motor_id} - Position Trajectory')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Position (rad)')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            plot_file = os.path.join(self.log_dir, 'motor_trajectories.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"轨迹图已保存: {plot_file}")
        
        plt.show()
    
    def plot_error_analysis(self, save_plots: bool = True) -> None:
        """绘制误差分析图"""
        if self.aligned_data is None or len(self.aligned_data) == 0:
            print("警告: 没有对齐的数据，跳过误差分析图")
            return
        
        # 计算误差统计
        error_stats = self.calculate_error_statistics()
        if 'error' in error_stats:
            print(f"警告: {error_stats['error']}")
            return
        
        # 创建误差分析图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Motor Error Analysis', fontsize=16)
        
        motors_plotted = 0
        for motor_id in range(1, self.motor_count + 1):
            if motors_plotted >= 6:  # 最多显示6个电机
                break
                
            motor_key = f'motor_{motor_id}'
            if motor_key not in error_stats:
                continue
                
            row = motors_plotted // 3
            col = motors_plotted % 3
            ax = axes[row, col]
            
            # 计算位置误差
            target_pos_col = f'target_position_motor_{motor_id}'
            actual_pos_col = f'actual_position_motor_{motor_id}'
            
            if target_pos_col in self.aligned_data.columns and actual_pos_col in self.aligned_data.columns:
                pos_error = self.aligned_data[actual_pos_col] - self.aligned_data[target_pos_col]
                
                # 绘制误差时间序列
                time_indices = range(len(pos_error))
                ax.plot(time_indices, pos_error, 'b-', alpha=0.7, linewidth=0.5)
                ax.set_title(f'Motor {motor_id} Position Error')
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Position Error (rad)')
                ax.grid(True, alpha=0.3)
                
                # 添加统计信息
                stats = error_stats[motor_key]['position_error']
                ax.text(0.02, 0.98, f'RMS: {stats["rms"]:.4f}\nMax: {stats["max_abs"]:.4f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            motors_plotted += 1
        
        # 隐藏未使用的子图
        for i in range(motors_plotted, 6):
            row = i // 3
            col = i % 3
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_plots:
            plot_file = os.path.join(self.log_dir, 'error_analysis.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"误差分析图已保存: {plot_file}")
        
        plt.show()
    
    def debug_data_statistics(self) -> None:
        """
        Debug function to check data statistics and identify constant values
        """
        if self.aligned_data is None:
            raise ValueError("请先调用align_data()对齐数据")
        
        print("\n=== 数据统计调试信息 ===")
        for motor_id in range(1, self.motor_count + 1):
            target_pos = self.aligned_data[f'target_position_motor_{motor_id}']
            actual_pos = self.aligned_data[f'actual_position_motor_{motor_id}']
            
            target_range = target_pos.max() - target_pos.min()
            actual_range = actual_pos.max() - actual_pos.min()
            
            print(f"\nMotor {motor_id}:")
            print(f"  Target position range: {target_range:.6f} rad ({np.degrees(target_range):.3f}°)")
            print(f"  Actual position range: {actual_range:.6f} rad ({np.degrees(actual_range):.3f}°)")
            print(f"  Target position: min={target_pos.min():.6f}, max={target_pos.max():.6f}")
            print(f"  Actual position: min={actual_pos.min():.6f}, max={actual_pos.max():.6f}")
            
            if actual_range < 1e-6:
                print(f"  WARNING: Motor {motor_id} actual position appears constant!")
                print(f"  First 5 actual values: {actual_pos.head().tolist()}")
    
    def analyze_tracking_performance(self) -> Dict:
        """
        分析跟踪性能和误差
        
        Returns:
            analysis_results: 分析结果字典
        """
        if self.aligned_data is None:
            raise ValueError("请先调用align_data()对齐数据")
        
        analysis_results = {}
        
        for motor_id in range(1, self.motor_count + 1):
            # 计算位置误差
            target_pos = self.aligned_data[f'target_position_motor_{motor_id}']
            actual_pos = self.aligned_data[f'actual_position_motor_{motor_id}']
            position_error = actual_pos - target_pos
            
            # 计算速度误差
            target_vel = self.aligned_data[f'target_velocity_motor_{motor_id}']
            actual_vel = self.aligned_data[f'actual_velocity_motor_{motor_id}']
            velocity_error = actual_vel - target_vel
            
            # 统计分析
            motor_analysis = {
                'position_error': {
                    'mean': position_error.mean(),
                    'std': position_error.std(),
                    'max_abs': abs(position_error).max(),
                    'rms': np.sqrt(np.mean(position_error**2))
                },
                'velocity_error': {
                    'mean': velocity_error.mean(),
                    'std': velocity_error.std(),
                    'max_abs': abs(velocity_error).max(),
                    'rms': np.sqrt(np.mean(velocity_error**2))
                }
            }
            
            # PID调优建议
            pos_rms = motor_analysis['position_error']['rms']
            vel_rms = motor_analysis['velocity_error']['rms']
            pos_overshoot = (position_error > 0).sum() / len(position_error)
            
            tuning_advice = []
            if pos_rms > 0.01:  # 位置误差较大
                if pos_overshoot > 0.7:
                    tuning_advice.append("位置超调严重，建议减小Kp或增大Kd")
                else:
                    tuning_advice.append("位置跟踪误差大，建议增大Kp")
            
            if vel_rms > 0.1:  # 速度误差较大
                tuning_advice.append("速度跟踪误差大，建议调整Kd")
            
            if abs(motor_analysis['position_error']['mean']) > 0.005:
                tuning_advice.append("存在稳态误差，建议增加积分项Ki")
            
            if not tuning_advice:
                tuning_advice.append("跟踪性能良好，无需调整")
            
            motor_analysis['tuning_advice'] = tuning_advice
            analysis_results[f'motor_{motor_id}'] = motor_analysis
        
        return analysis_results
    
    def generate_report(self, analysis_results: Dict, save_report: bool = True) -> str:
        """
        生成分析报告
        
        Args:
            analysis_results: 分析结果
            save_report: 是否保存报告
        
        Returns:
            report_text: 报告文本
        """
        report_lines = []
        report_lines.append("# 电机控制性能分析报告")
        report_lines.append(f"\n## 基本信息")
        report_lines.append(f"- 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"- 日志路径: {self.log_dir}")
        report_lines.append(f"- 数据点数: {len(self.aligned_data)}")
        report_lines.append(f"- 平均时间延迟: {self.aligned_data['time_delay_ms'].mean():.3f}ms")
        
        report_lines.append(f"\n## 各电机性能分析")
        
        for motor_id in range(1, self.motor_count + 1):
            motor_key = f'motor_{motor_id}'
            if motor_key not in analysis_results:
                continue
                
            motor_data = analysis_results[motor_key]
            report_lines.append(f"\n### 电机 {motor_id}")
            
            # 位置误差分析
            pos_err = motor_data['position_error']
            report_lines.append(f"\n**位置跟踪误差:**")
            report_lines.append(f"- 平均误差: {pos_err['mean']:.6f} rad ({np.degrees(pos_err['mean']):.3f}°)")
            report_lines.append(f"- 标准差: {pos_err['std']:.6f} rad ({np.degrees(pos_err['std']):.3f}°)")
            report_lines.append(f"- 最大绝对误差: {pos_err['max_abs']:.6f} rad ({np.degrees(pos_err['max_abs']):.3f}°)")
            report_lines.append(f"- RMS误差: {pos_err['rms']:.6f} rad ({np.degrees(pos_err['rms']):.3f}°)")
            
            # 速度误差分析
            vel_err = motor_data['velocity_error']
            report_lines.append(f"\n**速度跟踪误差:**")
            report_lines.append(f"- 平均误差: {vel_err['mean']:.6f} rad/s")
            report_lines.append(f"- 标准差: {vel_err['std']:.6f} rad/s")
            report_lines.append(f"- 最大绝对误差: {vel_err['max_abs']:.6f} rad/s")
            report_lines.append(f"- RMS误差: {vel_err['rms']:.6f} rad/s")
            
            # 调优建议
            report_lines.append(f"\n**PID调优建议:**")
            for advice in motor_data['tuning_advice']:
                report_lines.append(f"- {advice}")
        
        # 总体建议
        report_lines.append(f"\n## 总体建议")
        
        # 计算所有电机的平均性能
        all_pos_rms = [analysis_results[f'motor_{i}']['position_error']['rms'] 
                       for i in range(1, self.motor_count + 1) 
                       if f'motor_{i}' in analysis_results]
        all_vel_rms = [analysis_results[f'motor_{i}']['velocity_error']['rms'] 
                       for i in range(1, self.motor_count + 1) 
                       if f'motor_{i}' in analysis_results]
        
        avg_pos_rms = np.mean(all_pos_rms)
        avg_vel_rms = np.mean(all_vel_rms)
        
        if avg_pos_rms > 0.01:
            report_lines.append("- 整体位置跟踪精度需要改善，建议优先调整位置环参数")
        if avg_vel_rms > 0.1:
            report_lines.append("- 整体速度跟踪精度需要改善，建议优先调整速度环参数")
        if avg_pos_rms <= 0.01 and avg_vel_rms <= 0.1:
            report_lines.append("- 整体控制性能良好，系统运行稳定")
        
        report_text = "\n".join(report_lines)
        
        if save_report:
            report_file = os.path.join(self.log_dir, 'analysis_report.md')
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"分析报告已保存: {report_file}")
        
        return report_text

    def run_complete_analysis(self) -> Dict:
        """
        运行完整的分析流程
        
        Returns:
            analysis_report: 包含所有分析结果的字典
        """
        print("开始完整分析流程...")
        
        # 1. 加载数据
        self.load_data()
        
        # 检查数据是否为空
        if len(self.target_data) == 0 or len(self.state_data) == 0:
            print("错误: 加载的数据为空，无法进行分析")
            return {'error': '数据为空'}
        
        # 2. 对齐数据
        self.align_data()
        
        # 检查对齐数据是否为空
        if len(self.aligned_data) == 0:
            print("错误: 数据对齐失败，无法进行后续分析")
            return {
                'error': '数据对齐失败',
                'data_summary': {
                    'target_records': len(self.target_data),
                    'state_records': len(self.state_data),
                    'aligned_records': 0,
                    'time_range': f"{self.target_data['timestamp'].min()} - {self.target_data['timestamp'].max()}" if len(self.target_data) > 0 else "N/A"
                }
            }
        
        # 3. 计算误差统计
        error_stats = self.calculate_error_statistics()
        
        # 4. 绘制轨迹对比图
        self.plot_motor_trajectories()
        
        # 5. 绘制误差分析图
        self.plot_error_analysis()
        
        # 6. 生成报告
        report = {
            'data_summary': {
                'target_records': len(self.target_data),
                'state_records': len(self.state_data),
                'aligned_records': len(self.aligned_data),
                'time_range': f"{self.target_data['timestamp'].min()} - {self.target_data['timestamp'].max()}"
            },
            'error_statistics': error_stats,
            'plots_saved': True
        }
        
        print("完整分析流程完成!")
        return report

def find_latest_log_dir(base_log_dir: str = "logs") -> str:
    """
    查找最新的日志目录
    
    Args:
        base_log_dir: 基础日志目录路径
        
    Returns:
        latest_log_dir: 最新日志目录的完整路径
    """
    import glob
    
    # 查找所有日志目录（格式：YYYYMMDD_HHMMSS）
    log_pattern = os.path.join(base_log_dir, "????????_??????")
    log_dirs = glob.glob(log_pattern)
    
    if not log_dirs:
        raise FileNotFoundError(f"在 {base_log_dir} 中没有找到日志目录")
    
    # 按目录名排序，最新的在最后
    log_dirs.sort()
    latest_log_dir = log_dirs[-1]
    
    print(f"自动选择最新日志目录: {os.path.basename(latest_log_dir)}")
    return latest_log_dir

def main(log_dir: str = None):
    """
    主函数
    
    Args:
        log_dir: 指定的日志目录，如果为None则自动选择最新的
    """
    if log_dir is None:
        log_dir = find_latest_log_dir()
    
    print(f"分析日志目录: {log_dir}")
    
    # 创建分析器
    analyzer = MotorLogAnalyzer(log_dir, alignment_threshold_ms=15)
    
    # 运行完整分析
    report = analyzer.run_complete_analysis()
    
    # 打印完整报告
    print("\n=== 完整分析报告 ===")
    print(report)


if __name__ == "__main__":
    import sys
    
    # 支持命令行参数
    if len(sys.argv) > 1:
        # 使用指定的日志目录
        specified_log_dir = sys.argv[1]
        print(f"使用指定的日志目录: {specified_log_dir}")
        main(specified_log_dir)
    else:
        # 自动选择最新的日志目录
        print("未指定日志目录，自动选择最新的...")
        main()