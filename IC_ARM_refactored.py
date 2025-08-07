#!/usr/bin/env python3
"""
IC_ARM 重构版本 - 提供清晰的读写分离和统一状态管理
增强调试友好性：类型检查、详细报错、日志输出
"""

import time
import math
import numpy as np
import traceback
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import serial
# DM_CAN imports
from DM_CAN import DM_Motor_Type, MotorControl, Motor

# motor_config = {
#     'm1': {'type': DM_Motor_Type.DM4340, 'id': 0x01, 'master_id': 0x00, 'kp': 60, 'kd': 1.5, 'torque': 0},
#     'm2': {'type': DM_Motor_Type.DM4340, 'id': 0x02, 'master_id': 0x00, 'kp': 45, 'kd': 1.5, 'torque': 0.},
#     'm3': {'type': DM_Motor_Type.DM4340, 'id': 0x03, 'master_id': 0x00, 'kp': 40, 'kd': 1.5, 'torque': 0.5},
#     'm4': {'type': DM_Motor_Type.DM4340, 'id': 0x04, 'master_id': 0x00, 'kp': 38, 'kd': 1.5, 'torque': 0.0},
#     'm5': {'type': DM_Motor_Type.DM4340, 'id': 0x05, 'master_id': 0x00, 'kp': 35, 'kd': 1.5, 'torque': 0.5},
# }
motor_config = {
    'm1': {'type': DM_Motor_Type.DM4340, 'id': 0x01, 'master_id': 0x00, 'kp': 0, 'kd': 0, 'torque': 0},
    'm2': {'type': DM_Motor_Type.DM4340, 'id': 0x02, 'master_id': 0x00, 'kp': 45, 'kd': 1.5, 'torque': 0},
    'm3': {'type': DM_Motor_Type.DM4340, 'id': 0x03, 'master_id': 0x00, 'kp': 40, 'kd': 1.5, 'torque': 0},
    'm4': {'type': DM_Motor_Type.DM4340, 'id': 0x04, 'master_id': 0x00, 'kp': 38, 'kd': 1.5, 'torque': 0},
    'm5': {'type': DM_Motor_Type.DM4340, 'id': 0x05, 'master_id': 0x00, 'kp': 35, 'kd': 1.5, 'torque': 0},
}

# ===== 辅助函数定义 =====

def debug_print(message: str, level: str = 'INFO'):
    """调试输出函数"""
    timestamp = time.strftime('%H:%M:%S')
    print(f"[{timestamp}] [{level}] {message}")

def safe_call(func, *args, **kwargs) -> Tuple[Any, Optional[str]]:
    """安全函数调用，返回(结果, 错误信息)"""
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        error_msg = f"{func.__name__}() 失败: {str(e)}"
        debug_print(f"安全调用失败: {error_msg}", 'ERROR')
        debug_print(f"详细错误: {traceback.format_exc()}", 'ERROR')
        return None, error_msg

def validate_type(value: Any, expected_type: Union[type, Tuple[type, ...]], name: str) -> bool:
    """验证变量类型"""
    if not isinstance(value, expected_type):
        # 处理tuple类型的情况
        if isinstance(expected_type, tuple):
            type_names = ' or '.join([t.__name__ for t in expected_type])
        else:
            type_names = expected_type.__name__
        debug_print(f"类型验证失败: {name} 期望 {type_names}, 实际 {type(value).__name__}", 'ERROR')
        return False
    return True

def validate_array(array: np.ndarray, expected_shape: Tuple, name: str) -> bool:
    """验证numpy数组形状"""
    if not isinstance(array, np.ndarray):
        debug_print(f"数组验证失败: {name} 不是numpy数组, 类型: {type(array)}", 'ERROR')
        return False
    if array.shape != expected_shape:
        debug_print(f"数组形状验证失败: {name} 期望 {expected_shape}, 实际 {array.shape}", 'ERROR')
        return False
    return True

class ICARM:
    def __init__(self, port='/dev/cu.usbmodem00000000050C1', baudrate=921600, debug=True):
        """Initialize IC ARM with refactored interface and debug support"""
        self.debug = debug
        debug_print("=== 初始化IC_ARM_Refactored ===")
        
        try:
            # Hardware setup
            debug_print(f"设置硬件连接: {port}, {baudrate}")
            self.port = port
            self.baudrate = baudrate
            
            # 验证参数
            if not validate_type(port, str, 'port'):
                raise ValueError(f"Invalid port type: {type(port)}")
            if not validate_type(baudrate, int, 'baudrate'):
                raise ValueError(f"Invalid baudrate type: {type(baudrate)}")
            
            # 初始化串口
            debug_print("初始化串口连接...")
            self.serial_device = serial.Serial(port, baudrate, timeout=0.1)
            self.mc = MotorControl(self.serial_device)
            debug_print("✓ 串口连接成功")
            
            # Motor configuration
            self.motor_config = motor_config  # 添加motor_config属性
            self.motor_names = ['m1', 'm2', 'm3', 'm4', 'm5']
            self.motors = {}
            
            # Initialize motors
            debug_print("初始化电机...")
            for motor_name in self.motor_names:
                try:
                    config = motor_config[motor_name]
                    debug_print(f"  配置电机 {motor_name}: {config}")
                    motor = Motor(config['type'], config['id'], config['master_id'])
                    self.motors[motor_name] = motor
                    self.mc.addMotor(motor)
                    debug_print(f"  ✓ 电机 {motor_name} 初始化成功")
                except Exception as e:
                    debug_print(f"  ✗ 电机 {motor_name} 初始化失败: {e}", 'ERROR')
                    raise
            
            # State variables (all in radians and SI units) - 内部维护的状态变量
            debug_print("初始化状态变量...")
            self.q = np.zeros(5, dtype=np.float64)        # Joint positions (rad)
            self.dq = np.zeros(5, dtype=np.float64)       # Joint velocities (rad/s)
            self.ddq = np.zeros(5, dtype=np.float64)      # Joint accelerations (rad/s²)
            self.tau = np.zeros(5, dtype=np.float64)      # Joint torques (N·m)
            self.currents = np.zeros(5, dtype=np.float64) # Joint currents (A)
            
            # History for numerical differentiation
            self.q_prev = np.zeros(5, dtype=np.float64)
            self.dq_prev = np.zeros(5, dtype=np.float64)
            self.last_update_time = time.time()
            
            # 验证状态变量
            self._validate_internal_state()
            debug_print("✓ 状态变量初始化成功")
            
            # Read motor info and initialize states
            debug_print("读取电机信息并初始化状态...")
            self._read_motor_info()
            self._refresh_all_states()
            debug_print("✓ IC_ARM_Refactored 初始化完成")
            
        except Exception as e:
            debug_print(f"✗ 初始化失败: {e}", 'ERROR')
            debug_print(f"详细错误: {traceback.format_exc()}", 'ERROR')
            raise
    
    def _validate_internal_state(self):
        """验证内部状态变量的完整性"""
        state_vars = {
            'q': self.q,
            'dq': self.dq, 
            'ddq': self.ddq,
            'tau': self.tau,
            'currents': self.currents,
            'q_prev': self.q_prev,
            'dq_prev': self.dq_prev
        }
        
        for name, var in state_vars.items():
            if not validate_array(var, (5,), name):
                raise ValueError(f"Invalid state variable {name}")
        
        debug_print("✓ 内部状态变量验证通过")
    
    # ========== LOW-LEVEL MOTOR READ FUNCTIONS ==========
    # 只使用DM_CAN实际提供的API：getPosition, getVelocity, getTorque
    
    def _read_motor_position_raw(self, motor_name: str) -> float:
        """Read position from a single motor (refresh first)"""
        if self.debug:
            debug_print(f"读取电机 {motor_name} 位置...")
        
        # 验证参数
        if not validate_type(motor_name, str, 'motor_name'):
            return 0.0
        
        if motor_name not in self.motors:
            debug_print(f"电机 {motor_name} 不存在于电机列表中: {list(self.motors.keys())}", 'ERROR')
            return 0.0
        
        try:
            motor = self.motors[motor_name]
            if self.debug:
                debug_print(f"  刷新电机 {motor_name} 状态...")
            
            # 安全调用刷新状态
            result, error = safe_call(self.mc.refresh_motor_status, motor)
            if error:
                debug_print(f"刷新电机 {motor_name} 状态失败: {error}", 'ERROR')
                return 0.0
            
            # 读取位置
            position = motor.getPosition()
            
            # 验证返回值
            if not validate_type(position, (int, float, np.float32, np.float64), f'{motor_name}_position'):
                debug_print(f"电机 {motor_name} 返回的位置类型错误: {type(position)}, 值: {position}", 'ERROR')
                return 0.0
            
            position = float(position)
            if self.debug:
                debug_print(f"  ✓ 电机 {motor_name} 位置: {position:.4f} rad")
            
            return position
            
        except Exception as e:
            debug_print(f"读取电机 {motor_name} 位置失败: {e}", 'ERROR')
            debug_print(f"详细错误: {traceback.format_exc()}", 'ERROR')
            return 0.0
    
    def _read_motor_velocity_raw(self, motor_name: str) -> float:
        """Read velocity from a single motor (refresh first)"""
        if self.debug:
            debug_print(f"读取电机 {motor_name} 速度...")
        
        # 验证参数
        if not validate_type(motor_name, str, 'motor_name'):
            return 0.0
        
        if motor_name not in self.motors:
            debug_print(f"电机 {motor_name} 不存在", 'ERROR')
            return 0.0
        
        try:
            motor = self.motors[motor_name]
            
            # 安全调用刷新状态
            result, error = safe_call(self.mc.refresh_motor_status, motor)
            if error:
                debug_print(f"刷新电机 {motor_name} 状态失败: {error}", 'ERROR')
                return 0.0
            
            # 读取速度
            velocity = motor.getVelocity()
            
            # 验证返回值
            if not validate_type(velocity, (int, float, np.float32, np.float64), f'{motor_name}_velocity'):
                debug_print(f"电机 {motor_name} 返回的速度类型错误: {type(velocity)}, 值: {velocity}", 'ERROR')
                return 0.0
            
            velocity = float(velocity)
            if self.debug:
                debug_print(f"  ✓ 电机 {motor_name} 速度: {velocity:.4f} rad/s")
            
            return velocity
            
        except Exception as e:
            debug_print(f"读取电机 {motor_name} 速度失败: {e}", 'ERROR')
            debug_print(f"详细错误: {traceback.format_exc()}", 'ERROR')
            return 0.0
    
    def _read_motor_torque_raw(self, motor_name: str) -> float:
        """Read torque from a single motor (refresh first)"""
        if self.debug:
            debug_print(f"读取电机 {motor_name} 力矩...")
        
        # 验证参数
        if not validate_type(motor_name, str, 'motor_name'):
            return 0.0
        
        if motor_name not in self.motors:
            debug_print(f"电机 {motor_name} 不存在", 'ERROR')
            return 0.0
        
        try:
            motor = self.motors[motor_name]
            
            # 安全调用刷新状态
            result, error = safe_call(self.mc.refresh_motor_status, motor)
            if error:
                debug_print(f"刷新电机 {motor_name} 状态失败: {error}", 'ERROR')
                return 0.0
            
            # 读取力矩
            torque = motor.getTorque()
            
            # 验证返回值
            if not validate_type(torque, (int, float, np.float32, np.float64), f'{motor_name}_torque'):
                debug_print(f"电机 {motor_name} 返回的力矩类型错误: {type(torque)}, 值: {torque}", 'ERROR')
                return 0.0
            
            torque = float(torque)
            if self.debug:
                debug_print(f"  ✓ 电机 {motor_name} 力矩: {torque:.4f} N·m")
            
            return torque
            
        except Exception as e:
            debug_print(f"读取电机 {motor_name} 力矩失败: {e}", 'ERROR')
            debug_print(f"详细错误: {traceback.format_exc()}", 'ERROR')
            return 0.0
    
   
    # ========== BATCH READ FUNCTIONS ==========
    
    def _read_all_positions_raw(self):
        """Read positions from all motors"""
        positions = np.zeros(5)
        motor_names = ['m1', 'm2', 'm3', 'm4', 'm5']
        
        for i, motor_name in enumerate(motor_names):
            if motor_name in self.motors:
                positions[i] = self._read_motor_position_raw(motor_name)
        
        return positions
    
    def _read_all_velocities_raw(self):
        """Read velocities from all motors"""
        velocities = np.zeros(5)
        motor_names = ['m1', 'm2', 'm3', 'm4', 'm5']
        
        for i, motor_name in enumerate(motor_names):
            if motor_name in self.motors:
                velocities[i] = self._read_motor_velocity_raw(motor_name)
        
        return velocities
    
    def _read_all_torques_raw(self):
        """Read torques from all motors"""
        torques = np.zeros(5)
        motor_names = ['m1', 'm2', 'm3', 'm4', 'm5']
        
        for i, motor_name in enumerate(motor_names):
            if motor_name in self.motors:
                torques[i] = self._read_motor_torque_raw(motor_name)
        
        return torques
    
    def _read_all_currents_raw(self):
        """Estimate currents from torques (DM_CAN doesn't provide direct current reading)"""
        currents = np.zeros(5)
        torques = self._read_all_torques_raw()
        
        # 估算电流（使用力矩常数，需要校准）
        for i in range(5):
            currents[i] = torques[i] / 0.1  # 假设力矩常数为0.1 N·m/A
        
        return currents
    
    # ========== STATE UPDATE FUNCTIONS ==========
    
    def _refresh_all_states(self):
        """Refresh all motor states and update internal variables"""
        current_time = time.time()
        dt = current_time - self.last_update_time
        
        # Read current positions and copy to internal state
        new_positions = self._read_all_positions_raw()
        self.q = new_positions.copy()  # 复制位置信息到内部变量
        
        # Read current velocities from hardware and copy to internal state
        hardware_velocities = self._read_all_velocities_raw()
        
        # Calculate velocities using numerical differentiation as backup
        if dt > 0 and hasattr(self, 'q_prev'):
            calculated_velocities = (self.q - self.q_prev) / dt
            
            # Use hardware velocities if available, otherwise use calculated
            self.dq = hardware_velocities.copy()  # 复制速度信息到内部变量
            
            # Calculate accelerations using numerical differentiation
            if hasattr(self, 'dq_prev'):
                self.ddq = (self.dq - self.dq_prev) / dt  # 复制加速度信息到内部变量
        else:
            self.dq = hardware_velocities.copy()
        
        # Read torques and copy to internal state
        self.tau = self._read_all_torques_raw().copy()  # 复制力矩信息到内部变量
        
        # Read currents and store (optional, for completeness)
        self.currents = self._read_all_currents_raw().copy()  # 复制电流信息到内部变量
        
        # Update history for next iteration
        self.q_prev = self.q.copy()
        self.dq_prev = self.dq.copy()
        self.last_update_time = current_time
        
        # Debug: print update confirmation (可选)
        # print(f"State updated at {current_time:.3f}: pos={self.q[:2]}, vel={self.dq[:2]}, tau={self.tau[:2]}")
    
    def _refresh_all_states_fast(self):
        """快速状态刷新，减少调试输出和不必要的操作"""
        current_time = time.time()
        dt = current_time - self.last_update_time
        
        # 快速读取位置（无调试输出）
        for i, motor_name in enumerate(['m1', 'm2', 'm3', 'm4', 'm5']):
            if motor_name in self.motors:
                motor = self.motors[motor_name]
                # 快速刷新状态
                self.mc.refresh_motor_status(motor)
                # 直接读取位置
                self.q[i] = float(motor.getPosition())
                # 直接读取速度
                self.dq[i] = float(motor.getVelocity())
                # 直接读取力矩
                self.tau[i] = float(motor.getTorque())
        
        # 计算加速度（数值微分）
        if dt > 0 and hasattr(self, 'dq_prev'):
            self.ddq = (self.dq - self.dq_prev) / dt
        
        # 估算电流
        self.currents = self.tau / 0.1  # 简单估算
        
        # 更新历史
        self.q_prev = self.q.copy()
        self.dq_prev = self.dq.copy()
        self.last_update_time = current_time
    
    # ========== PUBLIC READ INTERFACES ==========
    
    def get_joint_positions(self, refresh=True):
        """Get joint positions in radians - 返回内部维护的位置状态"""
        if refresh:
            self._refresh_all_states()  # 更新内部状态变量
        return self.q.copy()  # 返回内部维护的位置副本
    
    def get_joint_velocities(self, refresh=True):
        """Get joint velocities in rad/s - 返回内部维护的速度状态"""
        if refresh:
            self._refresh_all_states()  # 更新内部状态变量
        return self.dq.copy()  # 返回内部维护的速度副本
    
    def get_joint_accelerations(self, refresh=True):
        """Get joint accelerations in rad/s² - 返回内部维护的加速度状态"""
        if refresh:
            self._refresh_all_states()  # 更新内部状态变量
        return self.ddq.copy()  # 返回内部维护的加速度副本
    
    def get_joint_torques(self, refresh=True):
        """Get joint torques in N·m - 返回内部维护的力矩状态"""
        if refresh:
            self._refresh_all_states()  # 更新内部状态变量
        return self.tau.copy()  # 返回内部维护的力矩副本
    
    def get_joint_currents(self, refresh=True):
        """Get joint currents in A - 返回内部维护的电流状态"""
        if refresh:
            self._refresh_all_states()  # 更新内部状态变量
        return self.currents.copy()  # 返回内部维护的电流副本
    
    def get_complete_state(self) -> Dict[str, Union[np.ndarray, float]]:
        """Get complete robot state with debug support"""
        if self.debug:
            debug_print("获取完整机器人状态...")
        
        try:
            # 刷新所有状态
            if self.debug:
                debug_print("  刷新所有状态...")
            self._refresh_all_states()
            
            # 验证内部状态
            self._validate_internal_state()
            
            # 构建状态字典
            state = {
                'positions': self.q.copy(),      # rad
                'velocities': self.dq.copy(),    # rad/s
                'accelerations': self.ddq.copy(), # rad/s²
                'torques': self.tau.copy(),      # N·m
                'currents': self.currents.copy(), # A
                'timestamp': self.last_update_time
            }
            
            # 验证返回的状态
            if self.debug:
                debug_print("  验证返回状态...")
                for key, value in state.items():
                    if key == 'timestamp':
                        if not validate_type(value, (int, float), f'state.{key}'):
                            raise ValueError(f"Invalid timestamp type: {type(value)}")
                    else:
                        if not validate_array(value, (5,), f'state.{key}'):
                            raise ValueError(f"Invalid state array {key}")
                        debug_print(f"    {key}: {value[:2]}... (shape: {value.shape}, dtype: {value.dtype})")
            
            if self.debug:
                debug_print("✓ 完整状态获取成功")
            
            return state
            
        except Exception as e:
            debug_print(f"获取完整状态失败: {e}", 'ERROR')
            debug_print(f"详细错误: {traceback.format_exc()}", 'ERROR')
            
            # 返回安全的默认状态
            debug_print("返回安全的默认状态", 'WARNING')
            return {
                'positions': np.zeros(5, dtype=np.float64),
                'velocities': np.zeros(5, dtype=np.float64),
                'accelerations': np.zeros(5, dtype=np.float64),
                'torques': np.zeros(5, dtype=np.float64),
                'currents': np.zeros(5, dtype=np.float64),
                'timestamp': time.time()
            }
    
    # ========== CONVENIENCE READ FUNCTIONS ==========
    
    def get_positions_degrees(self, refresh=True):
        """Get joint positions in degrees"""
        positions_rad = self.get_joint_positions(refresh)
        return np.degrees(positions_rad)
    
    def get_velocities_degrees(self, refresh=True):
        """Get joint velocities in deg/s"""
        velocities_rad = self.get_joint_velocities(refresh)
        return np.degrees(velocities_rad)
    
    def get_single_joint_state(self, joint_index, refresh=True):
        """Get state of a single joint (0-4)"""
        if refresh:
            self._refresh_all_states()
        
        if 0 <= joint_index < 5:
            return {
                'position': self.q[joint_index],
                'velocity': self.dq[joint_index],
                'acceleration': self.ddq[joint_index],
                'torque': self.tau[joint_index]
            }
        else:
            raise ValueError("Joint index must be 0-4")
    
    # ========== LOW-LEVEL WRITE FUNCTIONS ==========
    
    def _send_motor_command_raw(self, motor, position_rad, velocity_rad_s=0.0, torque_nm=0.0):
        """Send command to a single motor (lowest level)"""
        try:
            motor_name = None
            for name, m in self.motors.items():
                if m == motor:
                    motor_name = name
                    break
            
            if motor_name:
                config = self.motor_config[motor_name]
                kp = config['kp']
                kd = config['kd']
                self.mc.controlMIT(motor, kp, kd, position_rad, velocity_rad_s, torque_nm)
                return True
            else:
                print("Motor not found in configuration")
                return False
        except Exception as e:
            print(f"Failed to send command to motor: {e}")
            return False
    
    # ========== PUBLIC WRITE INTERFACES ==========
    
    def set_joint_position(self, joint_index, position_rad, velocity_rad_s=0.0, torque_nm=0.0):
        """Set position of a single joint"""
        if 0 <= joint_index < 5:
            motor_name = self.motor_names[joint_index]
            if motor_name in self.motors:
                return self._send_motor_command_raw(
                    self.motors[motor_name], 
                    position_rad, 
                    velocity_rad_s, 
                    torque_nm
                )
        return False
    
    def set_joint_positions(self, positions_rad, velocities_rad_s=None, torques_nm=None):
        """Set positions of all joints"""
        if velocities_rad_s is None:
            velocities_rad_s = np.zeros(5)
        if torques_nm is None:
            torques_nm = np.zeros(5)
        
        success = True
        for i in range(min(5, len(positions_rad))):
            result = self.set_joint_position(
                i, 
                positions_rad[i], 
                velocities_rad_s[i], 
                torques_nm[i]
            )
            success = success and result
        
        return success
    
    def set_joint_positions_degrees(self, positions_deg, velocities_deg_s=None, torques_nm=None):
        """Set positions of all joints in degrees"""
        positions_rad = np.radians(positions_deg)
        velocities_rad_s = np.radians(velocities_deg_s) if velocities_deg_s is not None else None
        return self.set_joint_positions(positions_rad, velocities_rad_s, torques_nm)
    
    # ========== MOTOR CONTROL FUNCTIONS ==========
    
    def enable_motor(self, joint_index):
        """Enable a single motor"""
        if 0 <= joint_index < 5:
            motor_name = self.motor_names[joint_index]
            if motor_name in self.motors and not self.motors[motor_name].isEnable:
                try:
                    self.mc.enable(self.motors[motor_name])
                    self.motors[motor_name].isEnable = True
                    print(f"Motor {motor_name} enabled")
                    return True
                except Exception as e:
                    print(f"Failed to enable motor {motor_name}: {e}")
        return False
    
    def disable_motor(self, joint_index):
        """Disable a single motor"""
        if 0 <= joint_index < 5:
            motor_name = self.motor_names[joint_index]
            if motor_name in self.motors and self.motors[motor_name].isEnable:
                try:
                    self.mc.disable(self.motors[motor_name])
                    self.motors[motor_name].isEnable = False
                    print(f"Motor {motor_name} disabled")
                    return True
                except Exception as e:
                    print(f"Failed to disable motor {motor_name}: {e}")
        return False
    
    def enable(self):
        return self.enable_all_motors()

    def disable(self):
        return self.disable_all_motors()

    def enable_all_motors(self):
        """Enable all motors"""
        print("Enabling all motors...")
        success = True
        for i in range(5):
            result = self.enable_motor(i)
            success = success and result
        
        if success:
            print("Waiting for motors to stabilize...")
            time.sleep(2)
        return success
    
    def disable_all_motors(self):
        """Disable all motors"""
        print("Disabling all motors...")
        success = True
        for i in range(5):
            result = self.disable_motor(i)
            success = success and result
        return success
    
    def emergency_stop(self):
        """Emergency stop - disable all motors immediately"""
        print("EMERGENCY STOP!")
        return self.disable_all_motors()
    
    # ========== INFORMATION FUNCTIONS ==========
    
    def _read_motor_info(self):
        """Read and display motor information"""
        print("\n" + "="*80)
        print("MOTOR INFORMATION TABLE")
        print("="*80)
        print(f"{'Motor':<8} {'ID':<4} {'PMAX':<12} {'VMAX':<12} {'TMAX':<12} {'Status':<10}")
        print("-"*80)
        
        for motor_name, motor in self.motors.items():
            try:
                pmax = self.mc.read_motor_param(motor, DM_variable.PMAX)
                vmax = self.mc.read_motor_param(motor, DM_variable.VMAX)
                tmax = self.mc.read_motor_param(motor, DM_variable.TMAX)
                status = "OK"
            except Exception as e:
                pmax = vmax = tmax = "ERROR"
                status = "FAIL"
                
            print(f"{motor_name:<8} {motor.SlaveID:<4} {pmax:<12} {vmax:<12} {tmax:<12} {status:<10}")
        
        print("="*80)
        print()
    
    def print_current_state(self):
        """Print current robot state"""
        state = self.get_complete_state()
        
        print("\n" + "="*80)
        print("CURRENT ROBOT STATE")
        print("="*80)
        print(f"{'Joint':<8} {'Pos(deg)':<12} {'Vel(deg/s)':<12} {'Acc(deg/s²)':<15} {'Torque(Nm)':<12}")
        print("-"*80)
        
        for i in range(5):
            print(f"m{i+1:<7} {np.degrees(state['positions'][i]):<12.2f} "
                  f"{np.degrees(state['velocities'][i]):<12.2f} "
                  f"{np.degrees(state['accelerations'][i]):<15.2f} "
                  f"{state['torques'][i]:<12.3f}")
        
        print("="*80)
        print(f"Timestamp: {state['timestamp']:.3f}")
        print()
    
    # ========== TRAJECTORY EXECUTION ==========
    
    def execute_trajectory_points(self, trajectory_points, verbose=True):
        """Execute a trajectory given as a list of points"""
        if not trajectory_points:
            print("Empty trajectory")
            return False
        
        print(f"Executing trajectory with {len(trajectory_points)} points...")
        self.enable_all_motors()
        
        start_time = time.time()
        
        try:
            for i, point in enumerate(trajectory_points):
                if len(point) < 6:  # Need 5 positions + 1 timestamp
                    print(f"Invalid point at index {i}: {point}")
                    continue
                
                target_positions_deg = point[:5]
                target_time = point[5]
                
                # Wait for target time
                while (time.time() - start_time) < target_time:
                    time.sleep(0.001)
                
                # Send commands
                self.set_joint_positions_degrees(target_positions_deg)
                
                # Progress reporting
                if verbose and i % 10 == 0:
                    progress = (i / len(trajectory_points)) * 100
                    current_pos = self.get_positions_degrees()
                    print(f"Progress: {progress:.1f}% | Target: {[f'{p:.1f}' for p in target_positions_deg]} | "
                          f"Actual: {[f'{p:.1f}' for p in current_pos]}")
        
        except KeyboardInterrupt:
            print("\nTrajectory interrupted")
        except Exception as e:
            print(f"Trajectory execution error: {e}")
        finally:
            self.disable_all_motors()
        
        final_pos = self.get_positions_degrees()
        print(f"Trajectory execution completed. Final position: {[f'{p:.2f}°' for p in final_pos]}")
        return final_pos
    
    # ========== CLEANUP ==========
    
    def close(self):
        """Close the connection and cleanup"""
        try:
            self.disable_all_motors()
            self.serial_device.close()
            print("ICARM connection closed")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor"""
        try:
            self.close()
        except:
            pass

# ========== EXAMPLE USAGE ==========
if __name__ == "__main__":
    # Example usage
    arm = ICARM()
    
    try:
        # Test single joint movement
        print("Testing single joint movement...")
        arm.enable_all_motors()
        
        # Move joint 0 to 30 degrees
        # arm.set_joint_positions_degrees([30, 0, 0, 0, 0])
        # time.sleep(2)
        
        # Read state again
        arm.print_current_state()
        
        # # Return to zero
        # arm.set_joint_positions_degrees([0, 0, 0, 0, 0])
        time.sleep(2)
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        arm.close()
