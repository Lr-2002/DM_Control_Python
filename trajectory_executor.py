#!/usr/bin/env python3
"""
轨迹执行器 - 用于在IC ARM上执行轨迹并采集动力学数据
增强调试友好性：详细错误处理、类型检查、日志输出
"""

import json
import time
import numpy as np
import pandas as pd
import traceback
import logging
from typing import Dict, List, Optional, Any, Union
import threading
from collections import deque
import matplotlib.pyplot as plt
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TrajectoryExecutor')

# 调试工具函数
def debug_print(msg: str, level: str = 'INFO'):
    """Debug print with timestamp"""
    timestamp = time.strftime('%H:%M:%S')
    print(f"[{timestamp}] [EXECUTOR-{level}] {msg}")

def safe_call(func, *args, **kwargs):
    """Safe function call with detailed error reporting"""
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        error_msg = f"Error in {func.__name__}: {str(e)}"
        debug_print(error_msg, 'ERROR')
        debug_print(f"Traceback: {traceback.format_exc()}", 'ERROR')
        return None, str(e)

def validate_trajectory_data(trajectory: Dict) -> bool:
    """Validate trajectory data structure"""
    required_keys = ['time', 'positions', 'velocities', 'accelerations']
    
    for key in required_keys:
        if key not in trajectory:
            debug_print(f"Missing required key in trajectory: {key}", 'ERROR')
            return False
        
        data = trajectory[key]
        if not isinstance(data, (list, np.ndarray)):
            debug_print(f"Trajectory {key} must be list or array, got {type(data)}", 'ERROR')
            return False
        
        if len(data) == 0:
            debug_print(f"Trajectory {key} is empty", 'ERROR')
            return False
    
    # Check data consistency
    time_len = len(trajectory['time'])
    for key in required_keys[1:]:  # Skip 'time'
        if len(trajectory[key]) != time_len:
            debug_print(f"Trajectory {key} length {len(trajectory[key])} != time length {time_len}", 'ERROR')
            return False
    
    debug_print("✓ 轨迹数据验证通过")
    return True

# 尝试导入重构后IC_ARM类
ICARMClass = None
USE_REFACTORED = False

try:
    from IC_ARM_refactored import ICARM
    ICARMClass = ICARM
    USE_REFACTORED = True
    ICARM_AVAILABLE = True
    print("IC ARM重构库已加载")
except ImportError:
    try:
        from IC_ARM import ICARM
        ICARMClass = ICARM
        USE_REFACTORED = False
        ICARM_AVAILABLE = True
        print("IC ARM原始库已加载")
    except ImportError:
        ICARM_AVAILABLE = False
        print("警告: IC ARM库未找到，将使用模拟模式")

# 禁用rerun功能
RERUN_AVAILABLE = False

class TrajectoryExecutor:
    def __init__(self, use_hardware=True, sample_rate=100, debug=True):
        """
        初始化轨迹执行器
        
        Args:
            use_hardware: 是否使用真实硬件
            sample_rate: 采样频率 (Hz)
            debug: 是否启用调试模式
        """
        self.debug = debug
        debug_print("=== 初始化TrajectoryExecutor ===")
        
        try:
            # 验证参数
            if not isinstance(use_hardware, bool):
                raise ValueError(f"use_hardware must be bool, got {type(use_hardware)}")
            if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
                raise ValueError(f"sample_rate must be positive number, got {sample_rate}")
            
            self.use_hardware = use_hardware
            self.sample_rate = sample_rate
            self.dt = 1.0 / sample_rate
            
            debug_print(f"参数设置: use_hardware={use_hardware}, sample_rate={sample_rate}Hz, dt={self.dt:.4f}s")
            
            # 初始化硬件连接
            if use_hardware and ICARM_AVAILABLE:
                debug_print("尝试连接硬件...")
                try:
                    # 使用导入的IC_ARM类
                    debug_print(f"使用IC_ARM类: {ICARMClass.__name__}")
                    
                    # 创建实例（传递debug参数）
                    if USE_REFACTORED:
                        self.arm = ICARMClass(debug=debug)
                    else:
                        self.arm = ICARMClass()
                    
                    self.use_refactored = USE_REFACTORED
                    
                    if USE_REFACTORED:
                        debug_print("✓ 使用重构后的IC_ARM接口")
                    else:
                        debug_print("✓ 使用原始IC_ARM接口")
                    
                    # 启用电机
                    debug_print("启用所有电机...")
                    result, error = safe_call(self.arm.enable_all_motors)
                    if error:
                        debug_print(f"启用电机失败: {error}", 'ERROR')
                        raise Exception(f"Motor enable failed: {error}")
                    
                    debug_print("✓ IC ARM硬件已连接并启用")
                    
                except Exception as e:
                    debug_print(f"硬件连接失败: {e}", 'ERROR')
                    debug_print(f"详细错误: {traceback.format_exc()}", 'ERROR')
                    debug_print("回退到模拟模式", 'WARNING')
                    self.use_hardware = False
                    self.arm = None
                    self.use_refactored = False
            else:
                self.arm = None
                self.use_refactored = False
                if not ICARM_AVAILABLE:
                    debug_print("✗ IC_ARM库不可用，使用模拟模式", 'WARNING')
                else:
                    debug_print("使用模拟模式")
            
            debug_print(f"✓ TrajectoryExecutor初始化完成 (硬件: {self.use_hardware}, 重构: {self.use_refactored})")
            
        except Exception as e:
            debug_print(f"✗ TrajectoryExecutor初始化失败: {e}", 'ERROR')
            debug_print(f"详细错误: {traceback.format_exc()}", 'ERROR')
            raise
        
        # 数据采集缓冲区
        self.data_buffer = {
            'time': deque(),
            'positions': deque(),
            'velocities': deque(),
            'accelerations': deque(),
            'target_positions': deque(),
            'target_velocities': deque(),
            'target_accelerations': deque(),
            'motor_currents': deque(),
            'motor_torques': deque()
        }
        
       
        # 安全限制
        self.max_position_error = np.radians(10)  # 最大位置误差 (弧度)
        self.max_velocity = np.radians(180)       # 最大速度 (弧度/秒)
        self.max_acceleration = np.radians(360)   # 最大加速度 (弧度/秒²)
        
        # 执行状态
        self.is_executing = False
        self.emergency_stop = False
        
    def load_trajectory(self, trajectory_file: str) -> Dict:
        """加载轨迹文件"""
        with open(trajectory_file, 'r') as f:
            trajectory = json.load(f)
        
        # 转换列表为numpy数组
        trajectory['time'] = np.array(trajectory['time'])
        trajectory['positions'] = np.array(trajectory['positions'])
        trajectory['velocities'] = np.array(trajectory['velocities'])
        trajectory['accelerations'] = np.array(trajectory['accelerations'])
        
        print(f"轨迹已加载: {trajectory_file}")
        print(f"  持续时间: {trajectory['time'][-1]:.2f}秒")
        print(f"  数据点数: {len(trajectory['time'])}")
        
        return trajectory
    
    def validate_trajectory(self, trajectory: Dict) -> bool:
        """验证轨迹安全性"""
        positions = trajectory['positions']
        velocities = trajectory['velocities']
        accelerations = trajectory['accelerations']
        
        # 检查速度限制
        max_vel = np.max(np.abs(velocities))
        if max_vel > self.max_velocity:
            print(f"警告: 最大速度 {np.degrees(max_vel):.1f}°/s 超过限制 {np.degrees(self.max_velocity):.1f}°/s")
            return False
        
        # 检查加速度限制
        max_acc = np.max(np.abs(accelerations))
        if max_acc > self.max_acceleration:
            print(f"警告: 最大加速度 {np.degrees(max_acc):.1f}°/s² 超过限制 {np.degrees(self.max_acceleration):.1f}°/s²")
            return False
        
        print("轨迹验证通过")
        return True
    
    def get_current_state(self) -> Optional[Dict[str, Union[np.ndarray, List[float]]]]:
        """获取当前机器人状态（增强调试版本）"""
        if self.debug:
            debug_print("获取当前机器人状态...")
        
        if not self.use_hardware or not self.arm:
            if self.debug:
                debug_print("使用模拟状态")
            return self._get_simulated_state()
        
        try:
            if self.use_refactored:
                if self.debug:
                    debug_print("  使用重构后的统一接口...")
                
                # 使用重构后的统一接口
                state, error = safe_call(self.arm.get_complete_state)
                if error:
                    debug_print(f"重构接口获取状态失败: {error}", 'ERROR')
                    return None
                
                # 验证返回的状态
                if not isinstance(state, dict):
                    debug_print(f"重构接口返回的状态不是字典: {type(state)}", 'ERROR')
                    return None
                
                required_keys = ['positions', 'velocities', 'accelerations', 'currents', 'torques']
                for key in required_keys:
                    if key not in state:
                        debug_print(f"重构接口状态缺少关键字: {key}", 'ERROR')
                        return None
                
                result_state = {
                    'positions': state['positions'],      # rad
                    'velocities': state['velocities'],    # rad/s  
                    'accelerations': state['accelerations'], # rad/s²
                    'currents': state['currents'],        # A
                    'torques': state['torques']           # N·m
                }
                
                if self.debug:
                    debug_print(f"  ✓ 重构接口状态获取成功")
                    for key, value in result_state.items():
                        if hasattr(value, 'shape'):
                            debug_print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                        else:
                            debug_print(f"    {key}: {type(value)}")
                
                return result_state
                
            else:
                if self.debug:
                    debug_print("  使用原始接口，手动构建状态...")
                
                # 使用原始接口，手动构建状态
                positions_deg, error = safe_call(self.arm.get_positions_only)
                if error:
                    debug_print(f"获取位置失败: {error}", 'ERROR')
                    return None
                
                if not isinstance(positions_deg, list) or len(positions_deg) != 5:
                    debug_print(f"位置数据格式错误: {type(positions_deg)}, 长度: {len(positions_deg) if hasattr(positions_deg, '__len__') else 'N/A'}", 'ERROR')
                    return None
                
                try:
                    positions_rad = [np.radians(pos['deg']) for pos in positions_deg]
                except (KeyError, TypeError) as e:
                    debug_print(f"位置数据解析失败: {e}, 数据: {positions_deg}", 'ERROR')
                    return None
                
                # 获取力矩数据
                currents_dict, error = safe_call(self.arm.read_all_currents)
                if error:
                    debug_print(f"获取力矩失败: {error}", 'ERROR')
                    currents_dict = {}
                
                # 转换为列表格式
                currents = []
                torques = []
                motor_names = ['m1', 'm2', 'm3', 'm4', 'm5']
                
                for motor_name in motor_names:
                    if motor_name in currents_dict:
                        torque_val = currents_dict[motor_name]
                        if isinstance(torque_val, (int, float)):
                            torques.append(float(torque_val))
                            # 估算电流（假设力矩常数为0.1）
                            currents.append(float(torque_val) / 0.1)
                        else:
                            debug_print(f"电机 {motor_name} 力矩值类型错误: {type(torque_val)}", 'WARNING')
                            torques.append(0.0)
                            currents.append(0.0)
                    else:
                        debug_print(f"电机 {motor_name} 力矩数据缺失", 'WARNING')
                        torques.append(0.0)
                        currents.append(0.0)
                
                result_state = {
                    'positions': np.array(positions_rad, dtype=np.float64),
                    'velocities': np.zeros(5, dtype=np.float64),  # 原始接口没有直接的速度读取
                    'accelerations': np.zeros(5, dtype=np.float64),  # 原始接口没有直接的加速度读取
                    'currents': np.array(currents, dtype=np.float64),
                    'torques': np.array(torques, dtype=np.float64)
                }
                
                if self.debug:
                    debug_print(f"  ✓ 原始接口状态获取成功")
                
                return result_state
                
        except Exception as e:
            debug_print(f"获取状态失败: {e}", 'ERROR')
            debug_print(f"详细错误: {traceback.format_exc()}", 'ERROR')
            return None
    
    def _get_simulated_state(self) -> Dict[str, np.ndarray]:
        """获取模拟状态"""
        return {
            'positions': np.zeros(5, dtype=np.float64),
            'velocities': np.zeros(5, dtype=np.float64),
            'accelerations': np.zeros(5, dtype=np.float64),
            'currents': np.zeros(5, dtype=np.float64),
            'torques': np.zeros(5, dtype=np.float64)
        }
    
    def send_position_command(self, target_positions: List[float]):
        """发送位置命令到机器人"""
        if self.use_hardware and self.arm:
            try:
                # 使用重构后的接口或原始接口
                if self.use_refactored:
                    # 使用重构后的统一接口
                    self.arm.set_joint_positions(target_positions)
                else:
                    # 使用原始接口的controlMIT方法
                    motor_names = ['m1', 'm2', 'm3', 'm4', 'm5']
                    
                    for i, target_pos_rad in enumerate(target_positions):
                        if i < len(motor_names):
                            motor_name = motor_names[i]
                            motor = self.arm.motors[motor_name]
                            
                            # 获取电机参数
                            from IC_ARM import motor_config
                            motor_params = motor_config[motor_name]
                            kp = motor_params['kp']
                            kd = motor_params['kd']
                            torque = motor_params['torque']
                            
                            # 使用MIT控制模式发送位置命令
                            self.arm.mc.controlMIT(motor, kp, kd, target_pos_rad, 0.0, torque)
                
            except Exception as e:
                print(f"发送命令失败: {e}")
                self.emergency_stop = True
    
    def execute_trajectory_simple(self, trajectory: Dict, record_data=True) -> Optional[pd.DataFrame]:
        """
        使用串行模式执行轨迹并采集数据（适配CAN半双工系统）
        
        执行流程：发送位置命令 → 等待稳定 → 读取状态 → 保存数据 → 下一个命令
        
        Args:
            trajectory: 轨迹字典
            record_data: 是否记录数据
            
        Returns:
            采集的数据DataFrame，如果失败返回None
        """
        if self.debug:
            debug_print("=== 开始串行轨迹执行 ===")
        
        # 验证轨迹
        if not validate_trajectory_data(trajectory):
            debug_print("轨迹数据验证失败，取消执行", 'ERROR')
            return None
        
        debug_print("开始执行轨迹...")
        input('Press Enter to continue...')
        
        self.is_executing = True
        self.emergency_stop = False
        
        # 清空数据缓冲区
        if record_data:
            for key in self.data_buffer:
                self.data_buffer[key].clear()
            debug_print("数据缓冲区已清空")
        
        # 准备轨迹数据
        time_points = trajectory['time']
        target_positions = trajectory['positions']  # 弧度
        target_velocities = trajectory['velocities']
        target_accelerations = trajectory['accelerations']
        
        debug_print(f"轨迹点数: {len(time_points)}, 持续时间: {time_points[-1]:.2f}s")
        
        if not self.use_hardware or not self.arm:
            debug_print("模拟模式，跳过实际执行", 'WARNING')
            return self._simulate_trajectory_execution(trajectory, record_data)
        
        try:
            start_time = time.time()
            debug_print("开始串行执行轨迹点...")
            
            # 串行执行每个轨迹点
            for i in range(len(time_points)):
                if self.emergency_stop:
                    debug_print("紧急停止被触发，终止执行", 'WARNING')
                    break
                
                current_time = time.time() - start_time
                target_time = time_points[i]
                target_pos_rad = target_positions[i]
                
                # 优化时间同步：只在必要时等待，减少延迟
                if current_time < target_time:
                    sleep_time = target_time - current_time
                    # 只在sleep时间较大时才等待，减少微小sleep引起的延迟
                    if sleep_time > 0.001:  # 只在超过1ms时才等待
                        if self.debug and i % 100 == 0:  # 减少调试输出频率
                            debug_print(f"  [{i}] 等待 {sleep_time:.3f}s")
                        time.sleep(sleep_time)
                    # 对于微小的时间差，直接跳过，提高实时性
                
                # 步骤 1: 发送位置命令
                if self.debug and i % 200 == 0:  # 进一步减少调试输出频率
                    debug_print(f"  [{i}/{len(time_points)}] 位置: {[f'{np.degrees(p):.1f}' for p in target_pos_rad]}°")
                
                result, error = safe_call(self.send_position_command, target_pos_rad)
                if error:
                    debug_print(f"发送位置命令失败: {error}", 'ERROR')
                    self.emergency_stop = True
                    break
                
                # # 步骤 2: 等待系统稳定（关键：避免CAN冲突）
                # time.sleep(0.005)  # 5ms稳定时间
                
                # 步骤 3: 读取当前状态（快速刷新）
                if record_data:
                    if self.use_refactored:
                        # 重构版本：使用快速状态刷新（平衡性能和数据正确性）
                        self.arm._refresh_all_states_fast()
                        
                        # 保持数据正确性：使用copy避免数据被后续更新覆盖
                        current_state = {
                            'positions': self.arm.q.copy(),
                            'velocities': self.arm.dq.copy(),
                            'accelerations': self.arm.ddq.copy(),
                            'torques': self.arm.tau.copy(),
                            'currents': self.arm.currents.copy()
                        }
                        
                        # 调试输出（每500个点输出一次，减少IO延迟）
                        if self.debug and i % 500 == 0:
                            debug_print(f"    [{i}] 位置: {[f'{p:.3f}' for p in current_state['positions'][:2]]}... 速度: {[f'{v:.3f}' for v in current_state['velocities'][:2]]}...", 'DEBUG')
                    else:
                        # 原始版本：安全读取状态
                        current_state, error = safe_call(self.get_current_state)
                        if error or current_state is None:
                            debug_print(f"读取状态失败: {error}", 'WARNING')
                            current_state = self._get_simulated_state()
                    
                    # 步骤 4: 保存数据
                    self._save_trajectory_point_data(
                        i, current_time, target_pos_rad, target_velocities[i], 
                        target_accelerations[i], current_state
                    )
            
            debug_print(f"✓ 轨迹执行完成，共执行 {len(time_points)} 个点")
            
            # 获取最终位置
            final_state = self.get_current_state()
            if final_state:
                final_positions = [np.degrees(pos) for pos in final_state['positions']]
                debug_print(f"最终位置: {[f'{p:.1f}' for p in final_positions]}°")
            
        except Exception as e:
            debug_print(f"执行过程中出错: {e}", 'ERROR')
            debug_print(f"详细错误: {traceback.format_exc()}", 'ERROR')
            self.emergency_stop = True
        
        finally:
            self.is_executing = False
            debug_print("轨迹执行结束")
        
        # 返回采集的数据
        if record_data and len(self.data_buffer['time']) > 0:
            return self._convert_buffer_to_dataframe()
        else:
            debug_print("无数据采集或数据为空", 'WARNING')
            return None
    
    def _save_trajectory_point_data(self, index: int, current_time: float, 
                                   target_pos: np.ndarray, target_vel: np.ndarray, 
                                   target_acc: np.ndarray, current_state: Dict):
        """超快速数据保存，最小化copy和调试输出"""
        try:
            # 保存时间和目标数据（已经是copy的）
            self.data_buffer['time'].append(current_time)
            self.data_buffer['target_positions'].append(target_pos.copy())
            self.data_buffer['target_velocities'].append(target_vel.copy())
            self.data_buffer['target_accelerations'].append(target_acc.copy())
            
            # 保存实际数据（已经是copy的，再次copy保证数据安全）
            self.data_buffer['positions'].append(current_state['positions'].copy())
            self.data_buffer['velocities'].append(current_state['velocities'].copy())
            self.data_buffer['accelerations'].append(current_state['accelerations'].copy())
            self.data_buffer['motor_currents'].append(current_state['currents'].copy())
            self.data_buffer['motor_torques'].append(current_state['torques'].copy())
            
            # 大幅减少调试输出频率（从每50个点→每1000个点）
            # if self.debug and index % 1000 == 0:
            #     debug_print(f"    数据点 {index} 已保存")
                
        except Exception as e:
            # 只在真正出错时才输出
            if index % 1000 == 0:  # 减少错误输出频率
                debug_print(f"保存数据点 {index} 失败: {e}", 'ERROR')
    
    def _simulate_trajectory_execution(self, trajectory: Dict, record_data: bool) -> Optional[pd.DataFrame]:
        """模拟轨迹执行（用于测试）"""
        debug_print("模拟轨迹执行...")
        
        if not record_data:
            debug_print("模拟模式不记录数据")
            return None
        
        time_points = trajectory['time']
        target_positions = trajectory['positions']
        target_velocities = trajectory['velocities']
        target_accelerations = trajectory['accelerations']
        
        # 清空数据缓冲区
        for key in self.data_buffer:
            self.data_buffer[key].clear()
        
        # 模拟数据采集
        for i in range(len(time_points)):
            current_state = self._get_simulated_state()
            # 模拟一些变化
            current_state['positions'] = target_positions[i] + np.random.normal(0, 0.01, 5)
            
            self._save_trajectory_point_data(
                i, time_points[i], target_positions[i], 
                target_velocities[i], target_accelerations[i], current_state
            )
        
        debug_print(f"模拟数据采集完成，共 {len(time_points)} 个点")
        return self._convert_buffer_to_dataframe()
    
    def _convert_buffer_to_dataframe(self) -> pd.DataFrame:
        """将数据缓冲区转换为DataFrame"""
        if self.debug:
            debug_print("转换数据缓冲区为DataFrame...")
        
        try:
            data_dict = {}
            
            # 时间
            data_dict['time'] = list(self.data_buffer['time'])
            
            # 位置、速度、加速度数据
            for motor_idx in range(5):
                motor_name = f'm{motor_idx + 1}'
                
                # 实际数据
                data_dict[f'{motor_name}_pos_actual'] = [pos[motor_idx] for pos in self.data_buffer['positions']]
                data_dict[f'{motor_name}_vel_actual'] = [vel[motor_idx] for vel in self.data_buffer['velocities']]
                data_dict[f'{motor_name}_acc_actual'] = [acc[motor_idx] for acc in self.data_buffer['accelerations']]
                data_dict[f'{motor_name}_current'] = [curr[motor_idx] for curr in self.data_buffer['motor_currents']]
                data_dict[f'{motor_name}_torque'] = [torque[motor_idx] for torque in self.data_buffer['motor_torques']]
                
                # 目标数据
                data_dict[f'{motor_name}_pos_target'] = [pos[motor_idx] for pos in self.data_buffer['target_positions']]
                data_dict[f'{motor_name}_vel_target'] = [vel[motor_idx] for vel in self.data_buffer['target_velocities']]
                data_dict[f'{motor_name}_acc_target'] = [acc[motor_idx] for acc in self.data_buffer['target_accelerations']]
            
            df = pd.DataFrame(data_dict)
            debug_print(f"✓ 数据转换完成，共 {len(df)} 个数据点")
            return df
            
        except Exception as e:
            debug_print(f"数据转换失败: {e}", 'ERROR')
            debug_print(f"详细错误: {traceback.format_exc()}", 'ERROR')
            return pd.DataFrame()
    
    def _get_simulated_state(self) -> Dict:
        """获取模拟状态数据（用于测试和fallback）"""
        return {
            'positions': np.zeros(5, dtype=np.float64),
            'velocities': np.zeros(5, dtype=np.float64),
            'accelerations': np.zeros(5, dtype=np.float64),
            'currents': np.zeros(5, dtype=np.float64),
            'torques': np.zeros(5, dtype=np.float64)
        }
    
    def _collect_data_during_execution(self, time_points, target_positions, target_velocities, target_accelerations):
        """在轨迹执行过程中采集数据的线程函数"""
        start_time = time.time()
        last_positions = None
        last_velocities = None
        
        for i, target_time in enumerate(time_points):
            if self.emergency_stop or not self.is_executing:
                break
            
            # 等待到目标时间
            current_time = time.time() - start_time
            while current_time < target_time and not self.emergency_stop:
                time.sleep(0.001)
                current_time = time.time() - start_time
            
            # 获取当前状态
            current_state = self.get_current_state()
            if current_state is None:
                continue
            
            # 计算实际速度和加速度
            actual_positions = np.array(current_state['positions'])
            
            if last_positions is not None:
                actual_velocities = (actual_positions - last_positions) / self.dt
            else:
                actual_velocities = np.zeros_like(actual_positions)
            
            if last_velocities is not None:
                actual_accelerations = (actual_velocities - last_velocities) / self.dt
            else:
                actual_accelerations = np.zeros_like(actual_velocities)
            
            # 记录数据
            self.data_buffer['time'].append(current_time)
            self.data_buffer['positions'].append(actual_positions.copy())
            self.data_buffer['velocities'].append(actual_velocities.copy())
            self.data_buffer['accelerations'].append(actual_accelerations.copy())
            self.data_buffer['target_positions'].append(target_positions[i].copy())
            self.data_buffer['target_velocities'].append(target_velocities[i].copy())
            self.data_buffer['target_accelerations'].append(target_accelerations[i].copy())
            self.data_buffer['motor_currents'].append(current_state['currents'])
            self.data_buffer['motor_torques'].append(current_state['torques'])
            
            # 更新历史状态
            last_positions = actual_positions
            last_velocities = actual_velocities
    
    def execute_trajectory(self, trajectory: Dict, record_data=True) -> Optional[pd.DataFrame]:
        """执行轨迹的主接口，使用简化版本"""
        return self.execute_trajectory_simple(trajectory, record_data)
        
        # 转换数据为DataFrame
        if record_data and len(self.data_buffer['time']) > 0:
            data_dict = {}
            
            # 时间
            data_dict['time'] = list(self.data_buffer['time'])
            
            # 位置、速度、加速度数据
            for motor_idx in range(5):
                motor_name = f'm{motor_idx + 1}'
                
                data_dict[f'{motor_name}_pos_actual'] = [pos[motor_idx] for pos in self.data_buffer['positions']]
                data_dict[f'{motor_name}_vel_actual'] = [vel[motor_idx] for vel in self.data_buffer['velocities']]
                data_dict[f'{motor_name}_acc_actual'] = [acc[motor_idx] for acc in self.data_buffer['accelerations']]
                
                data_dict[f'{motor_name}_pos_target'] = [pos[motor_idx] for pos in self.data_buffer['target_positions']]
                data_dict[f'{motor_name}_vel_target'] = [vel[motor_idx] for vel in self.data_buffer['target_velocities']]
                data_dict[f'{motor_name}_acc_target'] = [acc[motor_idx] for acc in self.data_buffer['target_accelerations']]
                
                data_dict[f'{motor_name}_current'] = [curr[motor_idx] for curr in self.data_buffer['motor_currents']]
                data_dict[f'{motor_name}_torque'] = [torq[motor_idx] for torq in self.data_buffer['motor_torques']]
            
            df = pd.DataFrame(data_dict)
            print(f"数据采集完成，共 {len(df)} 个数据点")
            return df
        
        return None
    
    def save_data(self, data: pd.DataFrame, filename: str):
        """保存采集的数据"""
        data.to_csv(filename, index=False)
        print(f"数据已保存到: {filename}")
    
    def plot_results(self, data: pd.DataFrame, motor_id: int = 1):
        """Plot execution results"""
        motor_name = f'm{motor_id}'
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 12))
        
        time = data['time']
        
        # Position comparison
        axes[0].plot(time, np.degrees(data[f'{motor_name}_pos_target']), 'b-', label='Target Position', linewidth=2)
        axes[0].plot(time, np.degrees(data[f'{motor_name}_pos_actual']), 'r--', label='Actual Position', linewidth=1)
        axes[0].set_ylabel('Position (deg)')
        axes[0].set_title(f'Motor {motor_id} Execution Results')
        axes[0].legend()
        axes[0].grid(True)
        
        # Velocity comparison
        axes[1].plot(time, np.degrees(data[f'{motor_name}_vel_target']), 'b-', label='Target Velocity', linewidth=2)
        axes[1].plot(time, np.degrees(data[f'{motor_name}_vel_actual']), 'r--', label='Actual Velocity', linewidth=1)
        axes[1].set_ylabel('Velocity (deg/s)')
        axes[1].legend()
        axes[1].grid(True)
        
        # Acceleration comparison
        axes[2].plot(time, np.degrees(data[f'{motor_name}_acc_target']), 'b-', label='Target Acceleration', linewidth=2)
        axes[2].plot(time, np.degrees(data[f'{motor_name}_acc_actual']), 'r--', label='Actual Acceleration', linewidth=1)
        axes[2].set_ylabel('Acceleration (deg/s²)')
        axes[2].legend()
        axes[2].grid(True)
        
        # Torque
        axes[3].plot(time, data[f'{motor_name}_torque'], 'g-', label='Motor Torque', linewidth=2)
        axes[3].set_ylabel('Torque (N·m)')
        axes[3].set_xlabel('Time (s)')
        axes[3].legend()
        axes[3].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'arm') and self.arm and self.use_hardware:
            try:
                self.arm.disable_all_motors()
            except:
                pass

def main():
    """主函数 - 演示轨迹执行"""
    print("=== IC ARM 轨迹执行器 ===\n")
    
    # 创建执行器
    executor = TrajectoryExecutor(use_hardware=True, sample_rate=100)
    
    # 加载并执行单个电机轨迹
    trajectory_file = "trajectory_motor_5_single.json"
    
    try:
        # 加载轨迹
        trajectory = executor.load_trajectory(trajectory_file)
        
        # 执行轨迹并采集数据
        print(f"执行轨迹: {trajectory_file}")
        data = executor.execute_trajectory(trajectory, record_data=True)
        
        if data is not None:
            # 保存数据
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            data_filename = f"dynamics_data_{timestamp}.csv"
            executor.save_data(data, data_filename)
            
            # 绘制结果
            executor.plot_results(data, motor_id=5)
            
            print(f"\n执行完成!")
            print(f"数据文件: {data_filename}")
        else:
            print("执行失败，未采集到数据")
    
    except FileNotFoundError:
        print(f"轨迹文件未找到: {trajectory_file}")
        print("请先运行 trajectory_generator.py 生成轨迹文件")
    
    except Exception as e:
        print(f"执行过程中出错: {e}")

if __name__ == "__main__":
    main()
