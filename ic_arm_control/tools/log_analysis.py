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
        self.motor_count = 9  # 根据CSV数据结构确定
        
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
        
        # 读取目标数据
        self.target_data = pd.read_csv(target_file)
        self.target_data['timestamp'] = pd.to_datetime(self.target_data['timestamp'])
        
        # 读取状态数据
        self.state_data = pd.read_csv(state_file)
        self.state_data['timestamp'] = pd.to_datetime(self.state_data['timestamp'])
        
        print(f"加载数据完成:")
        print(f"  目标数据: {len(self.target_data)} 条记录")
        print(f"  状态数据: {len(self.state_data)} 条记录")
        print(f"  时间范围: {self.target_data['timestamp'].min()} - {self.target_data['timestamp'].max()}")
        
        return self.target_data, self.state_data
    
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
        print(f"平均时间延迟: {self.aligned_data['time_delay_ms'].mean():.3f}ms")
        
        return self.aligned_data
    
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
    
    def run_complete_analysis(self, motor_ids: List[int] = None) -> str:
        """
        运行完整的分析流程
        
        Args:
            motor_ids: 要分析的电机ID列表
        
        Returns:
            report_text: 分析报告
        """
        print("开始日志分析...")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 对齐数据
        self.align_data()
        
        # 2.5. 调试数据统计
        self.debug_data_statistics()
        
        # 3. 绘制轨迹图
        self.plot_motor_trajectories(motor_ids)
        
        # 4. 分析性能
        analysis_results = self.analyze_tracking_performance()
        
        # 5. 生成报告
        report = self.generate_report(analysis_results)
        
        print("\n分析完成！")
        return report


def find_latest_log_dir(base_log_dir: str = "/Users/lr-2002/project/instantcreation/IC_arm_control/logs") -> str:
    """
    自动找到最新的日志目录
    
    Args:
        base_log_dir: 日志基础目录
    
    Returns:
        latest_log_dir: 最新的日志目录路径
    """
    import glob
    
    if not os.path.exists(base_log_dir):
        raise FileNotFoundError(f"日志基础目录不存在: {base_log_dir}")
    
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