#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹跟踪性能分析工具
分析IC ARM机器人的轨迹跟踪性能，包括误差、滞后、响应等指标
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from scipy import signal
from scipy.stats import pearsonr
import os
import sys
from typing import Dict, List, Tuple, Optional
import argparse

class TrajectoryAnalyzer:
    """轨迹跟踪性能分析器"""
    
    def __init__(self, data_file: str = None, data_df: pd.DataFrame = None):
        """
        初始化分析器
        Args:
            data_file: CSV数据文件路径
            data_df: 或者直接传入DataFrame
        """
        if data_file is not None:
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"数据文件不存在: {data_file}")
            self.data = pd.read_csv(data_file)
            self.data_file = data_file
        elif data_df is not None:
            self.data = data_df.copy()
            self.data_file = "DataFrame"
        else:
            raise ValueError("必须提供data_file或data_df参数")
        
        self.motor_ids = self._detect_motors()
        self.analysis_results = {}
        
        print(f"✓ 数据加载成功: {len(self.data)} 个数据点")
        print(f"✓ 检测到电机: {self.motor_ids}")
    
    def _detect_motors(self) -> List[int]:
        """自动检测数据中包含的电机"""
        motors = []
        for col in self.data.columns:
            if '_pos_target' in col:
                motor_id = col.split('_')[0][1:]  # 从 'm1_pos_target' 提取 '1'
                if motor_id.isdigit():
                    motors.append(int(motor_id))
        return sorted(motors)
    
    def analyze_tracking_performance(self, motor_id: int) -> Dict:
        """
        分析单个电机的轨迹跟踪性能
        
        Args:
            motor_id: 电机ID (1-5)
            
        Returns:
            包含各种性能指标的字典
        """
        if motor_id not in self.motor_ids:
            raise ValueError(f"电机 {motor_id} 的数据不存在")
        
        motor_name = f'm{motor_id}'
        
        # 提取数据
        time = self.data['time'].values
        pos_target = self.data[f'{motor_name}_pos_target'].values
        pos_actual = self.data[f'{motor_name}_pos_actual'].values
        vel_target = self.data[f'{motor_name}_vel_target'].values
        vel_actual = self.data[f'{motor_name}_vel_actual'].values
        torque = self.data[f'{motor_name}_torque'].values
        
        # 转换为度数便于理解
        pos_target_deg = np.degrees(pos_target)
        pos_actual_deg = np.degrees(pos_actual)
        vel_target_deg = np.degrees(vel_target)
        vel_actual_deg = np.degrees(vel_actual)
        
        # 1. 位置跟踪误差分析
        pos_error = pos_actual - pos_target
        pos_error_deg = np.degrees(pos_error)
        
        # 2. 速度跟踪误差分析
        vel_error = vel_actual - vel_target
        vel_error_deg = np.degrees(vel_error)
        
        # 3. 滞后分析（相关性和时间延迟）
        lag_analysis = self._analyze_lag(pos_target, pos_actual, time)
        
        # 4. 系统响应分析
        response_analysis = self._analyze_response(pos_target_deg, pos_actual_deg, time)
        
        # 5. 频域分析
        freq_analysis = self._analyze_frequency_response(pos_target, pos_actual, time)
        
        # 6. 稳态和动态性能
        steady_dynamic = self._analyze_steady_dynamic(pos_target_deg, pos_actual_deg, vel_target_deg, vel_actual_deg)
        
        # 7. 能耗分析
        energy_analysis = self._analyze_energy(torque, vel_actual, time)
        
        results = {
            'motor_id': motor_id,
            'data_points': len(time),
            'duration': time[-1] - time[0],
            
            # 位置跟踪性能
            'position_tracking': {
                'rmse_deg': np.sqrt(np.mean(pos_error_deg**2)),
                'mae_deg': np.mean(np.abs(pos_error_deg)),
                'max_error_deg': np.max(np.abs(pos_error_deg)),
                'std_error_deg': np.std(pos_error_deg),
                'mean_error_deg': np.mean(pos_error_deg),  # 系统偏差
                'correlation': pearsonr(pos_target_deg, pos_actual_deg)[0] if len(pos_target_deg) > 1 else 0.0,
            },
            
            # 速度跟踪性能
            'velocity_tracking': {
                'rmse_deg_s': np.sqrt(np.mean(vel_error_deg**2)),
                'mae_deg_s': np.mean(np.abs(vel_error_deg)),
                'max_error_deg_s': np.max(np.abs(vel_error_deg)),
                'correlation': pearsonr(vel_target_deg, vel_actual_deg)[0] if len(vel_target_deg) > 1 else 0.0,
            },
            
            # 滞后分析
            'lag_analysis': lag_analysis,
            
            # 系统响应
            'response_analysis': response_analysis,
            
            # 频域分析
            'frequency_analysis': freq_analysis,
            
            # 稳态和动态性能
            'steady_dynamic': steady_dynamic,
            
            # 能耗分析
            'energy_analysis': energy_analysis,
        }
        
        self.analysis_results[motor_id] = results
        return results
    
    def _analyze_lag(self, target: np.ndarray, actual: np.ndarray, time: np.ndarray) -> Dict:
        """分析滞后"""
        try:
            # 确保数据为一维数组
            target = np.asarray(target).flatten()
            actual = np.asarray(actual).flatten()
            time = np.asarray(time).flatten()
            
            # 简化的相关性分析
            if len(target) != len(actual):
                min_len = min(len(target), len(actual))
                target = target[:min_len]
                actual = actual[:min_len]
                time = time[:min_len]
            
            # 直接计算相关系数
            correlation_coeff = pearsonr(target, actual)[0] if len(target) > 1 else 0.0
            
            # 简化的滞后分析：找到最大变化的时间点
            target_diff = np.diff(target)
            actual_diff = np.diff(actual)
            
            # 找到最大变化的时间点
            max_target_change_idx = np.argmax(np.abs(target_diff))
            max_actual_change_idx = np.argmax(np.abs(actual_diff))
            
            # 计算时间延迟
            dt = np.mean(np.diff(time)) if len(time) > 1 else 0.001
            time_lag_samples = max_actual_change_idx - max_target_change_idx
            time_lag_seconds = time_lag_samples * dt
            
        except Exception as e:
            print(f"滞后分析失败: {e}")
            correlation_coeff = 0.0
            time_lag_seconds = 0.0
        
        return {
            'time_lag_ms': time_lag_seconds * 1000,
            'correlation_coefficient': correlation_coeff,
            'max_correlation': abs(correlation_coeff) if correlation_coeff is not None else 0.0
        }
    
    def _analyze_response(self, target_deg: np.ndarray, actual_deg: np.ndarray, time: np.ndarray) -> Dict:
        """分析系统响应特性"""
        # 寻找阶跃响应（大的位置变化）
        target_diff = np.diff(target_deg)
        step_indices = np.where(np.abs(target_diff) > 5.0)[0]  # 5度以上的变化
        
        if len(step_indices) == 0:
            return {'step_responses': [], 'avg_rise_time_ms': None, 'avg_settling_time_ms': None, 'avg_overshoot_percent': None}
        
        step_responses = []
        rise_times = []
        settling_times = []
        
        for idx in step_indices[:5]:  # 分析前5个阶跃响应
            if idx + 100 < len(actual_deg):  # 确保有足够的数据
                step_start = idx
                step_end = min(idx + 100, len(actual_deg) - 1)
                
                target_step = target_deg[step_start:step_end]
                actual_step = actual_deg[step_start:step_end]
                time_step = time[step_start:step_end] - time[step_start]
                
                # 计算上升时间（10%-90%）
                final_value = target_step[-1]
                initial_value = target_step[0]
                value_range = final_value - initial_value
                
                if abs(value_range) > 1.0:  # 足够大的变化
                    ten_percent = initial_value + 0.1 * value_range
                    ninety_percent = initial_value + 0.9 * value_range
                    
                    # 找到10%和90%的时间点
                    try:
                        if value_range > 0:
                            t10_idx = np.where(actual_step >= ten_percent)[0][0]
                            t90_idx = np.where(actual_step >= ninety_percent)[0][0]
                        else:
                            t10_idx = np.where(actual_step <= ten_percent)[0][0]
                            t90_idx = np.where(actual_step <= ninety_percent)[0][0]
                        
                        rise_time = time_step[t90_idx] - time_step[t10_idx]
                        rise_times.append(rise_time * 1000)  # 转换为毫秒
                        
                        # 计算稳定时间（进入±2%误差范围）
                        steady_threshold = abs(value_range) * 0.02
                        error = np.abs(actual_step - final_value)
                        settling_idx = np.where(error <= steady_threshold)[0]
                        if len(settling_idx) > 0:
                            settling_time = time_step[settling_idx[0]]
                            settling_times.append(settling_time * 1000)
                        
                        step_responses.append({
                            'initial_value': initial_value,
                            'final_value': final_value,
                            'rise_time_ms': rise_time * 1000,
                            'overshoot_percent': (np.max(actual_step) - final_value) / abs(value_range) * 100
                        })
                    except (IndexError, ValueError):
                        continue
        
        return {
            'step_responses': step_responses,
            'avg_rise_time_ms': np.mean(rise_times) if rise_times else None,
            'avg_settling_time_ms': np.mean(settling_times) if settling_times else None,
            'avg_overshoot_percent': np.mean([r['overshoot_percent'] for r in step_responses]) if step_responses else None
        }
    
    def _analyze_frequency_response(self, target: np.ndarray, actual: np.ndarray, time: np.ndarray) -> Dict:
        """频域分析"""
        dt = np.mean(np.diff(time))
        fs = 1.0 / dt
        
        # 计算功率谱密度
        f_target, psd_target = signal.welch(target, fs, nperseg=min(1024, len(target)//4))
        f_actual, psd_actual = signal.welch(actual, fs, nperseg=min(1024, len(actual)//4))
        
        # 计算传递函数（频率响应）
        f_cross, psd_cross = signal.csd(target, actual, fs, nperseg=min(1024, len(target)//4))
        
        # 主要频率成分
        dominant_freq_target = f_target[np.argmax(psd_target)]
        dominant_freq_actual = f_actual[np.argmax(psd_actual)]
        
        return {
            'dominant_freq_target_hz': dominant_freq_target,
            'dominant_freq_actual_hz': dominant_freq_actual,
            'bandwidth_hz': fs / 2,  # 奈奎斯特频率
            'freq_response_available': True
        }
    
    def _analyze_steady_dynamic(self, pos_target_deg: np.ndarray, pos_actual_deg: np.ndarray, 
                               vel_target_deg: np.ndarray, vel_actual_deg: np.ndarray) -> Dict:
        """稳态和动态性能分析"""
        # 识别稳态区域（速度接近0的区域）
        steady_mask = np.abs(vel_target_deg) < 1.0  # 速度小于1度/秒
        dynamic_mask = ~steady_mask
        
        steady_results = {}
        dynamic_results = {}
        
        if np.any(steady_mask):
            steady_pos_error = pos_actual_deg[steady_mask] - pos_target_deg[steady_mask]
            steady_results = {
                'steady_state_error_deg': np.mean(steady_pos_error),
                'steady_state_std_deg': np.std(steady_pos_error),
                'steady_state_points': np.sum(steady_mask)
            }
        
        if np.any(dynamic_mask):
            dynamic_pos_error = pos_actual_deg[dynamic_mask] - pos_target_deg[dynamic_mask]
            dynamic_results = {
                'dynamic_error_rms_deg': np.sqrt(np.mean(dynamic_pos_error**2)),
                'dynamic_error_max_deg': np.max(np.abs(dynamic_pos_error)),
                'dynamic_points': np.sum(dynamic_mask)
            }
        
        return {
            'steady_state': steady_results,
            'dynamic': dynamic_results
        }
    
    def _analyze_energy(self, torque: np.ndarray, velocity: np.ndarray, time: np.ndarray) -> Dict:
        """能耗分析"""
        # 功率 = 力矩 × 角速度
        power = torque * velocity
        
        # 能量 = 功率 × 时间
        dt = np.mean(np.diff(time))
        energy = np.sum(np.abs(power)) * dt
        
        return {
            'total_energy_j': energy,
            'avg_power_w': np.mean(np.abs(power)),
            'max_power_w': np.max(np.abs(power)),
            'rms_torque_nm': np.sqrt(np.mean(torque**2))
        }
    
    def generate_analysis_report(self, motor_id: int, save_path: str = None) -> str:
        """生成分析报告"""
        if motor_id not in self.analysis_results:
            self.analyze_tracking_performance(motor_id)
        
        results = self.analysis_results[motor_id]
        
        report = f"""
=== IC ARM 轨迹跟踪性能分析报告 ===
电机ID: {motor_id}
数据文件: {self.data_file}
数据点数: {results['data_points']}
持续时间: {results['duration']:.2f}s

【位置跟踪性能】
- 均方根误差 (RMSE): {f"{results['position_tracking']['rmse_deg']:.3f}°" if results['position_tracking'].get('rmse_deg') is not None else 'N/A'}
- 平均绝对误差 (MAE): {f"{results['position_tracking']['mae_deg']:.3f}°" if results['position_tracking'].get('mae_deg') is not None else 'N/A'}
- 最大误差: {f"{results['position_tracking']['max_error_deg']:.3f}°" if results['position_tracking'].get('max_error_deg') is not None else 'N/A'}
- 系统偏差: {f"{results['position_tracking']['mean_error_deg']:.3f}°" if results['position_tracking'].get('mean_error_deg') is not None else 'N/A'}
- 误差标准差: {f"{results['position_tracking']['std_error_deg']:.3f}°" if results['position_tracking'].get('std_error_deg') is not None else 'N/A'}

【速度跟踪性能】
- 均方根误差 (RMSE): {f"{results['velocity_tracking']['rmse_deg_s']:.3f}°/s" if results['velocity_tracking'].get('rmse_deg_s') is not None else 'N/A'}
- 平均绝对误差 (MAE): {f"{results['velocity_tracking']['mae_deg_s']:.3f}°/s" if results['velocity_tracking'].get('mae_deg_s') is not None else 'N/A'}
- 最大误差: {f"{results['velocity_tracking']['max_error_deg_s']:.3f}°/s" if results['velocity_tracking'].get('max_error_deg_s') is not None else 'N/A'}
- 相关系数: {f"{results['velocity_tracking']['correlation']:.4f}" if results['velocity_tracking'].get('correlation') is not None else 'N/A'}

【滞后分析】
- 时间滞后: {f"{results['lag_analysis']['time_lag_ms']:.2f}ms" if results['lag_analysis'].get('time_lag_ms') is not None else 'N/A'}
- 相关系数: {f"{results['lag_analysis']['correlation_coefficient']:.4f}" if results['lag_analysis'].get('correlation_coefficient') is not None else 'N/A'}

【系统响应】
- 平均上升时间: {f"{results['response_analysis']['avg_rise_time_ms']:.1f}ms" if results['response_analysis']['avg_rise_time_ms'] is not None else 'N/A'}
- 平均稳定时间: {f"{results['response_analysis']['avg_settling_time_ms']:.1f}ms" if results['response_analysis']['avg_settling_time_ms'] is not None else 'N/A'}
- 平均超调: {f"{results['response_analysis']['avg_overshoot_percent']:.1f}%" if results['response_analysis']['avg_overshoot_percent'] is not None else 'N/A'}

【稳态性能】
- 稳态误差: {f"{results['steady_dynamic']['steady_state'].get('steady_state_error_deg'):.3f}°" if results['steady_dynamic']['steady_state'].get('steady_state_error_deg') is not None else 'N/A'}
- 稳态标准差: {f"{results['steady_dynamic']['steady_state'].get('steady_state_std_deg'):.3f}°" if results['steady_dynamic']['steady_state'].get('steady_state_std_deg') is not None else 'N/A'}

【动态性能】
- 动态误差RMS: {f"{results['steady_dynamic']['dynamic'].get('dynamic_error_rms_deg'):.3f}°" if results['steady_dynamic']['dynamic'].get('dynamic_error_rms_deg') is not None else 'N/A'}
- 动态最大误差: {f"{results['steady_dynamic']['dynamic'].get('dynamic_error_max_deg'):.3f}°" if results['steady_dynamic']['dynamic'].get('dynamic_error_max_deg') is not None else 'N/A'}

【能耗分析】
- 总能耗: {f"{results['energy_analysis']['total_energy_j']:.3f}J" if results['energy_analysis'].get('total_energy_j') is not None else 'N/A'}
- 平均功率: {f"{results['energy_analysis']['avg_power_w']:.3f}W" if results['energy_analysis'].get('avg_power_w') is not None else 'N/A'}
- 最大功率: {f"{results['energy_analysis']['max_power_w']:.3f}W" if results['energy_analysis'].get('max_power_w') is not None else 'N/A'}
- RMS力矩: {f"{results['energy_analysis']['rms_torque_nm']:.3f}N·m" if results['energy_analysis'].get('rms_torque_nm') is not None else 'N/A'}

【性能评级】
{self._generate_performance_rating(results)}
        """
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✓ 分析报告已保存到: {save_path}")
        
        return report
    
    def _generate_performance_rating(self, results: Dict) -> str:
        """生成性能评级"""
        pos_rmse = results['position_tracking']['rmse_deg']
        vel_corr = results['velocity_tracking']['correlation']
        lag_ms = abs(results['lag_analysis']['time_lag_ms'])
        
        # 位置精度评级
        if pos_rmse < 0.5:
            pos_rating = "优秀"
        elif pos_rmse < 1.0:
            pos_rating = "良好"
        elif pos_rmse < 2.0:
            pos_rating = "一般"
        else:
            pos_rating = "需改进"
        
        # 速度跟踪评级
        if vel_corr > 0.95:
            vel_rating = "优秀"
        elif vel_corr > 0.90:
            vel_rating = "良好"
        elif vel_corr > 0.80:
            vel_rating = "一般"
        else:
            vel_rating = "需改进"
        
        # 响应速度评级
        if lag_ms < 10:
            lag_rating = "优秀"
        elif lag_ms < 50:
            lag_rating = "良好"
        elif lag_ms < 100:
            lag_rating = "一般"
        else:
            lag_rating = "需改进"
        
        return f"""- 位置跟踪精度: {pos_rating} (RMSE: {pos_rmse:.3f}°)
- 速度跟踪性能: {vel_rating} (相关性: {vel_corr:.3f})
- 系统响应速度: {lag_rating} (滞后: {lag_ms:.1f}ms)"""
    
    def plot_comprehensive_analysis(self, motor_id: int, save_path: str = None):
        """绘制综合分析图表"""
        if motor_id not in self.motor_ids:
            raise ValueError(f"电机 {motor_id} 的数据不存在")
        
        motor_name = f'm{motor_id}'
        
        # 提取数据
        time = self.data['time'].values
        pos_target = np.degrees(self.data[f'{motor_name}_pos_target'].values)
        pos_actual = np.degrees(self.data[f'{motor_name}_pos_actual'].values)
        vel_target = np.degrees(self.data[f'{motor_name}_vel_target'].values)
        vel_actual = np.degrees(self.data[f'{motor_name}_vel_actual'].values)
        torque = self.data[f'{motor_name}_torque'].values
        
        # 创建综合分析图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'电机 {motor_id} 轨迹跟踪综合分析', fontsize=16, fontweight='bold')
        
        # 1. 位置跟踪
        axes[0, 0].plot(time, pos_target, 'b-', label='目标位置', linewidth=2)
        axes[0, 0].plot(time, pos_actual, 'r--', label='实际位置', linewidth=1.5)
        axes[0, 0].set_ylabel('位置 (°)')
        axes[0, 0].set_title('位置跟踪')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 位置误差
        pos_error = pos_actual - pos_target
        axes[0, 1].plot(time, pos_error, 'g-', linewidth=1.5)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0, 1].set_ylabel('位置误差 (°)')
        axes[0, 1].set_title(f'位置误差 (RMSE: {np.sqrt(np.mean(pos_error**2)):.3f}°)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 速度跟踪
        axes[0, 2].plot(time, vel_target, 'b-', label='目标速度', linewidth=2)
        axes[0, 2].plot(time, vel_actual, 'r--', label='实际速度', linewidth=1.5)
        axes[0, 2].set_ylabel('速度 (°/s)')
        axes[0, 2].set_title('速度跟踪')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 速度误差
        vel_error = vel_actual - vel_target
        axes[1, 0].plot(time, vel_error, 'orange', linewidth=1.5)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 0].set_ylabel('速度误差 (°/s)')
        axes[1, 0].set_xlabel('时间 (s)')
        axes[1, 0].set_title(f'速度误差 (RMSE: {np.sqrt(np.mean(vel_error**2)):.3f}°/s)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 力矩
        axes[1, 1].plot(time, torque, 'purple', linewidth=1.5)
        axes[1, 1].set_ylabel('力矩 (N·m)')
        axes[1, 1].set_xlabel('时间 (s)')
        axes[1, 1].set_title(f'电机力矩 (RMS: {np.sqrt(np.mean(torque**2)):.3f}N·m)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 误差分布直方图
        axes[1, 2].hist(pos_error, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 2].axvline(x=np.mean(pos_error), color='red', linestyle='--', 
                          label=f'均值: {np.mean(pos_error):.3f}°')
        axes[1, 2].axvline(x=np.mean(pos_error) + np.std(pos_error), color='orange', 
                          linestyle='--', alpha=0.7, label=f'±1σ: {np.std(pos_error):.3f}°')
        axes[1, 2].axvline(x=np.mean(pos_error) - np.std(pos_error), color='orange', 
                          linestyle='--', alpha=0.7)
        axes[1, 2].set_xlabel('位置误差 (°)')
        axes[1, 2].set_ylabel('频次')
        axes[1, 2].set_title('位置误差分布')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 分析图表已保存到: {save_path}")
        
        plt.show()


def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(description='IC ARM轨迹跟踪性能分析工具')
    parser.add_argument('data_file', help='CSV数据文件路径')
    parser.add_argument('--motor', '-m', type=int, default=5, help='要分析的电机ID (默认: 5)')
    parser.add_argument('--report', '-r', help='保存分析报告的路径')
    parser.add_argument('--plot', '-p', help='保存分析图表的路径')
    
    args = parser.parse_args()
    
    try:
        # 创建分析器
        analyzer = TrajectoryAnalyzer(data_file=args.data_file)
        
        # 执行分析
        print(f"\n开始分析电机 {args.motor} 的轨迹跟踪性能...")
        results = analyzer.analyze_tracking_performance(args.motor)
        
        # 生成报告
        report_path = args.report or args.data_file.replace('.csv', f'_analysis_motor_{args.motor}.txt')
        report = analyzer.generate_analysis_report(args.motor, report_path)
        print(report)
        
        # 绘制分析图表
        plot_path = args.plot or args.data_file.replace('.csv', f'_analysis_motor_{args.motor}.png')
        analyzer.plot_comprehensive_analysis(args.motor, plot_path)
        
        print(f"\n✓ 分析完成!")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
