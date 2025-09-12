"""
IC_ARM 重构版本 - 使用unified_motor_control作为底层电机控制接口
提供高级机械臂控制功能：轨迹规划、安全检查、重力补偿等
"""

import time
import math
import numpy as np
import traceback
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
# 使用新的统一电机控制系统
from Python.unified_motor_control import MotorManager, MotorInfo, MotorType
from Python.damiao import Motor_Control, DmActData, DM_Motor_Type, Control_Mode, limit_param as dm_limit
from Python.ht_motor import HTMotorManager
from Python.src import usb_class
from usb_hw_wrapper import USBHardwareWrapper

# ===== 全局配置 =====
NUM_MOTORS = 6  # 基础电机数量（达妙电机）
NUM_MOTORS_WITH_HT = 8  # 包含HT电机的总数量（6个达妙 + 2个HT）

# 电机配置现在由unified_motor_control系统管理，不再需要这些配置

# ===== 辅助函数定义 =====

def debug_print(msg: str, level: str = 'INFO'):
    """Debug print with timestamp"""

    timestamp = time.strftime('%H:%M:%S') + f".{int(time.time() * 1000) % 1000:03d}"
    print(f"[{timestamp}] [IC_ARM-{level}] {msg}")


def safe_call(func, *args, **kwargs) -> Tuple[Any, Optional[str]]:
    """安全函数调用，返回(结果, 错误信息)"""
    try:
        result = func(*args, **kwargs)
        time.sleep(0.0002)
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
    def __init__(self, device_sn="F561E08C892274DB09496BCC1102DBC5", debug=False, gc=False, use_ht=False):
        """Initialize IC ARM with unified motor control system"""
        self.debug = debug
        self.use_ht = use_ht
        debug_print("=== 初始化IC_ARM_Unified ===")
 
        try:
            # 初始化统一电机控制系统
            debug_print("初始化统一电机控制系统...")
            
            # 创建USB硬件接口
            usb_hw = usb_class(1000000, 5000000, device_sn)
            usb_hw = USBHardwareWrapper(usb_hw)
            # 创建电机管理器
            self.motor_manager = MotorManager(usb_hw)
            
            # 配置达妙电机数据
            # damiao_data = [
            #   DmActData(DM_Motor_Type.DM10010L, Control_Mode.MIT_MODE, 0x01, 0x11, 100, 3),
            #   DmActData(DM_Motor_Type.DM4340, Control_Mode.MIT_MODE, 0x02, 0x12, 55, 1.5),
            #   DmActData(DM_Motor_Type.DM6248, Control_Mode.MIT_MODE, 0x03, 0x13, 65, 1.8),
            #   DmActData(DM_Motor_Type.DM4340, Control_Mode.MIT_MODE, 0x04, 0x14, 70, 1.9),
            #   DmActData(DM_Motor_Type.DM4340, Control_Mode.MIT_MODE, 0x05, 0x15, 50, 1.8),
            #   DmActData(DM_Motor_Type.DM4340, Control_Mode.MIT_MODE, 0x06, 0x16, 55, 1.5),
            # ]
            damiao_data = [
                DmActData(DM_Motor_Type.DM10010L, Control_Mode.MIT_MODE, 0x01, 0x11, 0, 0),
                DmActData(DM_Motor_Type.DM4340, Control_Mode.MIT_MODE, 0x02, 0x12, 0, 0),
                DmActData(DM_Motor_Type.DM6248, Control_Mode.MIT_MODE, 0x03, 0x13, 0, 0),
                DmActData(DM_Motor_Type.DM4340, Control_Mode.MIT_MODE, 0x04, 0x14, 0, 0),
                DmActData(DM_Motor_Type.DM4340, Control_Mode.MIT_MODE, 0x05, 0x15, 0, 0),
                DmActData(DM_Motor_Type.DM4310, Control_Mode.MIT_MODE, 0x06, 0x16, 0, 0),
            ]
            
            # 添加达妙电机协议
            motor_control = Motor_Control(usb_hw=usb_hw, data_ptr=damiao_data)
            self.motor_manager.add_damiao_protocol(motor_control)
            
            # 如果需要HT电机，添加HT协议
            if use_ht:
                ht_manager = HTMotorManager(usb_hw)
                self.motor_manager.add_ht_protocol(ht_manager)
            
            # 添加具体电机到管理器
            if use_ht:
                # 混合配置：前6个达妙电机 + 后2个HT电机
                debug_print("配置混合电机系统（达妙+HT）...")
                
                # 添加6个达妙电机 (1-6)
                for i in range(6):
                    motor_info = MotorInfo(
                        motor_id=i+1,
                        motor_type=MotorType.DAMIAO,
                        can_id=damiao_data[i].can_id,
                        name=f"m{i+1}",
                        kp=damiao_data[i].kp,
                        kd=damiao_data[i].kd,
                        limits=dm_limit[damiao_data[i].motorType.value]
                    )
                    success = self.motor_manager.add_motor(i+1, 'damiao', motor_info, can_id=damiao_data[i].can_id)
                    if success:
                        debug_print(f"  ✓ 达妙电机 m{i+1} 添加成功")
                    else:
                        debug_print(f"  ✗ 达妙电机 m{i+1} 添加失败", 'ERROR')
                
                # 添加2个HT电机 (7-8)
                for i in range(2):
                    motor_info = MotorInfo(
                        motor_id=i + 7,
                        motor_type=MotorType.HIGH_TORQUE,
                        can_id=0x8094,  # HT电机使用固定的发送ID
                        name=f"ht_{i+7}",
                        kp=0,
                        kd=0,
                        limits=[12.5, 50.0, 20.0],  # HT电机扭矩限制更高
                    )
                    success = self.motor_manager.add_motor(i + 7, "ht", motor_info, ht_motor_id=i + 7)
                    if success:
                        debug_print(f"  ✓ HT电机 ht_{i+7} 添加成功")
                    else:
                        debug_print(f"  ✗ HT电机 ht_{i+7} 添加失败", 'ERROR')
            else:
                # 纯达妙电机配置
                debug_print("配置纯达妙电机系统...")
                for i in range(6):
                    motor_info = MotorInfo(
                        motor_id=i+1,
                        motor_type=MotorType.DAMIAO,
                        can_id=damiao_data[i].can_id,
                        name=f"m{i+1}",
                        kp=damiao_data[i].kp,
                        kd=damiao_data[i].kd,
                        limits=dm_limit[damiao_data[i].motorType.value]
                    )
                    success = self.motor_manager.add_motor(i+1, 'damiao', motor_info, can_id=damiao_data[i].can_id)
                    if success:
                        debug_print(f"  ✓ 达妙电机 m{i+1} 添加成功")
                    else:
                        debug_print(f"  ✗ 达妙电机 m{i+1} 添加失败", 'ERROR')
            
            # State variables (all in radians and SI units) - 内部维护的状态变量
            debug_print("初始化状态变量...")
            # 根据是否使用HT电机决定状态变量大小
            motor_count = NUM_MOTORS_WITH_HT if use_ht else NUM_MOTORS
            self.motor_count = motor_count  # 保存实际电机数量
            
            self.q = np.zeros(motor_count, dtype=np.float64)        # Joint positions (rad)
            self.dq = np.zeros(motor_count, dtype=np.float64)       # Joint velocities (rad/s)
            self.ddq = np.zeros(motor_count, dtype=np.float64)      # Joint accelerations (rad/s²)
            self.tau = np.zeros(motor_count, dtype=np.float64)      # Joint torques (N·m)
            self.currents = np.zeros(motor_count, dtype=np.float64) # Joint currents (A)
            
            # History for numerical differentiation
            self.q_prev = np.zeros(motor_count, dtype=np.float64)
            self.dq_prev = np.zeros(motor_count, dtype=np.float64)
            self.last_update_time = time.time()
            
            # 验证状态变量
            self._validate_internal_state()
            debug_print("✓ 状态变量初始化成功")
            
            # 使能所有电机并初始化状态
            debug_print("使能电机并初始化状态...")
            self.enable()
            self._refresh_all_states()
            debug_print("✓ IC_ARM_Unified 初始化完成")
            
            # 重力补偿初始化
            self.gc_flag = gc 
            if self.gc_flag:

                from minimum_gc import MinimumGravityCompensation as GC
                self.gc = GC()
                debug_print('✓ 重力补偿系统已启动')
        except Exception as e:
            debug_print(f"✗ 初始化失败: {e}", 'ERROR')
            debug_print(f"详细错误: {traceback.format_exc()}", 'ERROR')
            raise
    
    def _validate_internal_state(self):
        """验证内部状态变量的完整性"""
        expected_shape = (self.motor_count,)
        
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
            if not validate_array(var, expected_shape, name):
                raise ValueError(f"Invalid state variable {name}")
        
        debug_print("✓ 内部状态变量验证通过")
    
    # ========== LOW-LEVEL MOTOR READ FUNCTIONS ==========
    # 使用unified_motor_control接口读取电机状态
    
    def _read_motor_position_raw(self, motor_id: int) -> float:
        """Read position from a single motor using unified interface"""
        if self.debug:
            debug_print(f"读取电机 {motor_id} 位置...")
        
        try:
            motor = self.motor_manager.get_motor(motor_id)
            if motor is None:
                debug_print(f"电机 {motor_id} 不存在", 'ERROR')
                return 0.0
            
            state = motor.get_state()
            position = state['position']
            
            if self.debug:
                debug_print(f"  ✓ 电机 {motor_id} 位置: {position:.4f} rad")
            
            return position
            
        except Exception as e:
            debug_print(f"读取电机 {motor_id} 位置失败: {e}", 'ERROR')
            return 0.0
    
    def _read_motor_velocity_raw(self, motor_id: int) -> float:
        """Read velocity from a single motor using unified interface"""
        if self.debug:
            debug_print(f"读取电机 {motor_id} 速度...")
        
        try:
            motor = self.motor_manager.get_motor(motor_id)
            if motor is None:
                debug_print(f"电机 {motor_id} 不存在", 'ERROR')
                return 0.0
            
            state = motor.get_state()
            velocity = state['velocity']
            
            if self.debug:
                debug_print(f"  ✓ 电机 {motor_id} 速度: {velocity:.4f} rad/s")
            
            return velocity
            
        except Exception as e:
            debug_print(f"读取电机 {motor_id} 速度失败: {e}", 'ERROR')
            return 0.0
    
    def _read_motor_torque_raw(self, motor_id: int) -> float:
        """Read torque from a single motor using unified interface"""
        if self.debug:
            debug_print(f"读取电机 {motor_id} 力矩...")
        
        try:
            motor = self.motor_manager.get_motor(motor_id)
            if motor is None:
                debug_print(f"电机 {motor_id} 不存在", 'ERROR')
                return 0.0
            
            state = motor.get_state()
            torque = state['torque']
            
            if self.debug:
                debug_print(f"  ✓ 电机 {motor_id} 力矩: {torque:.4f} N·m")
            
            return torque
            
        except Exception as e:
            debug_print(f"读取电机 {motor_id} 力矩失败: {e}", 'ERROR')
            return 0.0
    
   
    # ========== BATCH READ FUNCTIONS ==========
    
    def _read_all_positions_raw(self):
        """Read positions from all motors using unified interface"""
        positions = np.zeros(self.motor_count)
        
        for i in range(self.motor_count):
            positions[i] = self._read_motor_position_raw(i+1)  # motor_id starts from 1
        
        return positions
    
    def _read_all_velocities_raw(self):
        """Read velocities from all motors using unified interface"""
        velocities = np.zeros(self.motor_count)
        
        for i in range(self.motor_count):
            velocities[i] = self._read_motor_velocity_raw(i+1)  # motor_id starts from 1
        
        return velocities
    
    def _read_all_torques_raw(self):
        """Read torques from all motors using unified interface"""
        torques = np.zeros(self.motor_count)
        
        for i in range(self.motor_count):
            torques[i] = self._read_motor_torque_raw(i+1)  # motor_id starts from 1
        
        return torques
    
    def _read_all_currents_raw(self):
        """Estimate currents from torques (DM_CAN doesn't provide direct current reading)"""
        currents = np.zeros(self.motor_count)
        torques = self._read_all_torques_raw()
        
        # 估算电流（使用力矩常数，需要校准）
        for i in range(self.motor_count):
            currents[i] = torques[i] / 0.1  # 假设力矩常数为0.1 N·m/A
        
        return currents
    
    # ========== STATE UPDATE FUNCTIONS ==========
    
    def _refresh_all_states(self):
        """Refresh all motor states using unified motor control system"""
        current_time = time.time()
        dt = current_time - self.last_update_time
        
        # 使用统一电机控制系统更新所有状态
        self.motor_manager.update_all_states()
        
        # 读取所有电机状态
        for i in range(self.motor_count):
            motor = self.motor_manager.get_motor(i+1)
            if motor:
                state = motor.get_state()
                self.q[i] = state['position']
                self.dq[i] = state['velocity']
                self.tau[i] = state['torque']
        
        # Calculate accelerations using numerical differentiation
        if dt > 0 and hasattr(self, 'dq_prev'):
            self.ddq = (self.dq - self.dq_prev) / dt
        
        # 估算电流
        self.currents = self.tau / 0.1  # 简单估算
        
        # Update history for next iteration
        self.q_prev = self.q.copy()
        self.dq_prev = self.dq.copy()
        self.last_update_time = current_time
    
    def _refresh_all_states_fast(self):
        """快速状态刷新，使用统一电机控制系统"""
        current_time = time.time()
        dt = current_time - self.last_update_time
        
        # 使用统一系统快速更新状态
        self.motor_manager.update_all_states()
        
        # 快速读取所有电机状态
        for i in range(self.motor_count):
            motor = self.motor_manager.get_motor(i+1)
            if motor:
                state = motor.get_state()
                self.q[i] = state['position']
                self.dq[i] = state['velocity']
                self.tau[i] = state['torque']
        
        # 计算加速度（数值微分）
        if dt > 0 and hasattr(self, 'dq_prev'):
            self.ddq = (self.dq - self.dq_prev) / dt
        
        # 估算电流
        self.currents = self.tau / 0.1  # 简单估算
        
        # 更新历史
        self.q_prev = self.q.copy()
        self.dq_prev = self.dq.copy()
        self.last_update_time = current_time
    
    def _refresh_all_states_ultra_fast(self):
        """超快速状态刷新，使用统一电机控制系统的优化接口"""
        current_time = time.time()
        dt = current_time - self.last_update_time
        
        # 使用统一系统的快速更新
        self.motor_manager.update_all_states()
        
        # 超快速读取所有电机状态
        for i in range(self.motor_count):
            motor = self.motor_manager.get_motor(i+1)
            if motor:
                state = motor.get_state()
                self.q[i] = state['position']
                self.dq[i] = state['velocity']
                self.tau[i] = state['torque']
        
        # 最简化的加速度计算
        if dt > 0 and hasattr(self, 'dq_prev'):
            self.ddq = (self.dq - self.dq_prev) / dt
        
        # 简化电流估算
        self.currents = self.tau * 10.0  # 快速估算，避免除法
        
        # 最小化历史更新
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
                        if not validate_array(value, (self.motor_count,), f'state.{key}'):
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
                'positions': np.zeros(self.motor_count, dtype=np.float64),
                'velocities': np.zeros(self.motor_count, dtype=np.float64),
                'accelerations': np.zeros(self.motor_count, dtype=np.float64),
                'torques': np.zeros(self.motor_count, dtype=np.float64),
                'currents': np.zeros(self.motor_count, dtype=np.float64),
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
    
    def _send_motor_command_raw(self, motor_id, position_rad=0.0, velocity_rad_s=0.0, torque_nm=0.0):
        """Send command to a single motor using unified interface"""
        try:
            motor = self.motor_manager.get_motor(motor_id)
            if motor is None:
                debug_print(f"Motor {motor_id} not found", 'ERROR')
                return False
            
            # 使用电机的默认kp, kd参数
            motor_info = motor.motor_info
            kp = motor_info.kp
            kd = motor_info.kd
            
            if self.debug:
                debug_print(f"Sending command to motor {motor_id}: kp={kp}, kd={kd}, position={position_rad}, velocity={velocity_rad_s}, torque={torque_nm}")
            
            # 使用统一电机控制系统发送命令
            success = motor.set_command(position_rad, velocity_rad_s, kp, kd, torque_nm)
            return success
            
        except Exception as e:
            debug_print(f"Failed to send command to motor {motor_id}: {e}", 'ERROR')
            return False
    
    # ========== PUBLIC WRITE INTERFACES ==========
    


    def set_joint_position(self, joint_index, position_rad, velocity_rad_s=0.0, torque_nm=0.0):
        """Set position of a single joint using unified interface"""
        if joint_index < self.motor_count:
            motor_id = joint_index + 1  # motor_id starts from 1
            if self.debug:
                debug_print(f"Setting joint {joint_index} position to {position_rad} rad, velocity to {velocity_rad_s} rad/s, torque to {torque_nm} Nm")
            return self._send_motor_command_raw(
                motor_id, 
                position_rad, 
                velocity_rad_s, 
                torque_nm
            )
        return False

    def set_joint_torque(self, torques_nm):
        """Set torques of all joints using unified interface"""
        if torques_nm is None:
            torques_nm = np.zeros(self.motor_count)
        
        success = True
        for i in range(min(self.motor_count, len(torques_nm))):
            if self.debug:
                debug_print(f"Setting joint {i} torque to {torques_nm[i]} Nm")
            result = self._send_motor_command_raw(
                i + 1,  # motor_id starts from 1
                position_rad=0,
                velocity_rad_s=0,
                torque_nm=torques_nm[i]
            )
            success = success and result
        
        if not success: 
            debug_print('Joint torque setting failed', 'ERROR')
        return success   

    def set_joint_positions_with_gc(self, positions_rad, velocities_rad_s=None):
        tau = self.cal_gravity()[0]
        return self.set_joint_positions(positions_rad, velocities_rad_s, tau)

    def set_joint_positions(self, positions_rad, velocities_rad_s=None, torques_nm=None):
        """Set positions of all joints"""
        if velocities_rad_s is None:
            velocities_rad_s = np.zeros(self.motor_count)
        if torques_nm is None:
            torques_nm = np.zeros(self.motor_count)
        
        success = True
        # breakpoint()
        for i in range(min(self.motor_count, len(positions_rad))):
            result = self.set_joint_position(
                i, 
                positions_rad[i], 
                velocities_rad_s[i], 
                torques_nm[i]
            )
            success = success and result
        
            if not success: 
                print('------ run error')
        return success
    
    def set_joint_positions_degrees(self, positions_deg, velocities_deg_s=None, torques_nm=None):
        """Set positions of all joints in degrees"""
        positions_rad = np.radians(positions_deg)
        velocities_rad_s = np.radians(velocities_deg_s) if velocities_deg_s is not None else None
        return self.set_joint_positions(positions_rad, velocities_rad_s, torques_nm)
    
    # ========== MOTOR CONTROL FUNCTIONS ==========
    
    def enable_motor(self, joint_index):
        """Enable a single motor using unified interface"""
        motor_id = joint_index + 1  # motor_id starts from 1
        try:
            success = self.motor_manager.enable_motor(motor_id)
            if success:
                debug_print(f"Motor {motor_id} enabled")
            else:
                debug_print(f"Failed to enable motor {motor_id}", 'ERROR')
            return success
        except Exception as e:
            debug_print(f"Failed to enable motor {motor_id}: {e}", 'ERROR')
            return False
    
    def disable_motor(self, joint_index):
        """Disable a single motor using unified interface"""
        motor_id = joint_index + 1  # motor_id starts from 1
        try:
            success = self.motor_manager.disable_motor(motor_id)
            if success:
                debug_print(f"Motor {motor_id} disabled")
            else:
                debug_print(f"Failed to disable motor {motor_id}", 'ERROR')
            return success
        except Exception as e:
            debug_print(f"Failed to disable motor {motor_id}: {e}", 'ERROR')
            return False
    
    def enable(self):
        return self.enable_all_motors()

    def disable(self):
        return self.disable_all_motors()

    def enable_all_motors(self):
        """Enable all motors using unified interface"""
        debug_print("Enabling all motors...")
        try:
            success = self.motor_manager.enable_all()
            if success:
                debug_print("All motors enabled successfully")
                debug_print("Waiting for motors to stabilize...")
                time.sleep(2)
            else:
                debug_print("Failed to enable all motors", 'ERROR')
            return success
        except Exception as e:
            debug_print(f"Failed to enable all motors: {e}", 'ERROR')
            return False
    
    def disable_all_motors(self):
        """Disable all motors using unified interface"""
        debug_print("Disabling all motors...")
        try:
            success = self.motor_manager.disable_all()
            if success:
                debug_print("All motors disabled successfully")
            else:
                debug_print("Failed to disable all motors", 'ERROR')
            return success
        except Exception as e:
            debug_print(f"Failed to disable all motors: {e}", 'ERROR')
            return False
    
    def emergency_stop(self):
        """Emergency stop - disable all motors immediately"""
        print("EMERGENCY STOP!")
        return self.disable_all_motors()
    
    def home_to_zero(self, speed: float = 0.5, timeout: float = 30.0, frequency=100) -> bool:
        """
        让所有电机平滑地回到零位
        
        Args:
            speed: 回零速度 (rad/s)，默认0.5 rad/s
            timeout: 超时时间 (秒)，默认30秒
            
        Returns:
            bool: 是否成功回零
        """
        debug_print("开始执行回零操作...")
        
        
        try:
            # 获取当前位置
            current_positions = self.get_joint_positions()
            if current_positions is None:
                debug_print("无法获取当前位置", 'ERROR')
                return False
            
            debug_print(f"当前位置: {[f'{np.degrees(pos):.1f}°' for pos in current_positions]}")
            
            # 计算需要移动的距离和时间
            max_distance = max(abs(pos) for pos in current_positions)
            estimated_time = max_distance / speed
            
            debug_print(f"最大移动距离: {np.degrees(max_distance):.1f}°")
            debug_print(f'speed is {speed}')
            debug_print(f"预计回零时间: {estimated_time:.1f}秒")
            
            if estimated_time > timeout:
                debug_print(f"预计时间超过超时限制 ({timeout}s)，建议增加速度或超时时间", 'WARNING')
            
            # 生成平滑轨迹到零位
            num_steps = max(10, int(estimated_time * frequency))  # 至少10步，或按100Hz计算
            dt = estimated_time / num_steps
            
            debug_print(f"生成轨迹: {num_steps}步，步长{dt:.3f}s")
            
            # 预生成轨迹点用于可视化
            trajectory_points = []
            time_points = []
            
            for i in range(num_steps + 1):
                progress = i / num_steps
                smooth_progress = 0.5 * (1 - np.cos(np.pi * progress))  # 余弦插值
                target_positions = current_positions * (1 - smooth_progress)
                trajectory_points.append(target_positions.copy())
                time_points.append(i * dt)
            
            # 可视化轨迹
            self._plot_trajectory_preview(trajectory_points, time_points, current_positions)
            
            # 询问用户是否继续执行
            response = input('轨迹预览完成，是否继续执行? (y/n): ').lower().strip()
            if response != 'y':
                debug_print("用户取消轨迹执行")
                return False
            
            start_time = time.time()
            
            for i in range(num_steps + 1):
                # 检查超时
                if time.time() - start_time > timeout:
                    debug_print("回零操作超时", 'ERROR')
                    return False
                
                # 计算插值位置 (使用平滑的余弦插值)
                progress = i / num_steps
                smooth_progress = 0.5 * (1 - np.cos(np.pi * progress))  # 余弦插值，起始和结束速度为0
                
                target_positions = current_positions * (1 - smooth_progress)
                
                # 发送位置命令
                if self.gc_flag:
                    success = self.set_joint_positions_with_gc(target_positions)
                else:
                    success = self.set_joint_positions(target_positions)
                if not success:
                    debug_print(f"发送位置命令失败 (步骤 {i})", 'ERROR')
                    return False
                
                # 显示进度
                if i % (num_steps // 10) == 0 or i == num_steps:
                    current_pos = self.get_joint_positions(refresh=False)
                    if current_pos is not None:
                        max_error = max(abs(pos) for pos in current_pos)
                        debug_print(f"回零进度: {progress*100:.0f}%, 最大偏差: {np.degrees(max_error):.2f}°")
                
                # 等待下一步
                if i < num_steps:
                    time.sleep(dt)
            
            # 验证回零结果
            # time.sleep(0.5)  # 等待稳定
            final_positions = self.get_joint_positions()
            
            if final_positions is not None:
                max_error = max(abs(pos) for pos in final_positions)
                debug_print(f"回零完成! 最终位置: {[f'{np.degrees(pos):.2f}°' for pos in final_positions]}")
                debug_print(f"最大误差: {np.degrees(max_error):.2f}°")
                
                # 判断是否成功回零 (误差小于1度认为成功)
                if max_error < np.radians(3):
                    debug_print("✓ 回零成功!", 'INFO')
                    return True
                else:
                    debug_print(f"回零精度不足，最大误差: {np.degrees(max_error):.2f}°", 'WARNING')
                    return False
            else:
                debug_print("无法验证回零结果", 'ERROR')
                return False
                
        except Exception as e:
            debug_print(f"回零操作失败: {e}", 'ERROR')
            import traceback
            traceback.print_exc()
            return False
    
    def _plot_trajectory_preview(self, trajectory_points, time_points, start_positions):
        """
        可视化轨迹预览
        
        Args:
            trajectory_points: 轨迹点列表，每个点是5个关节的位置数组
            time_points: 时间点列表
            start_positions: 起始位置
        """
        try:
            import matplotlib.pyplot as plt
            
            # 转换为numpy数组便于处理
            trajectory_array = np.array(trajectory_points)  # shape: (num_steps, 5)
            
            # 创建子图
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('IC ARM 回零轨迹预览', fontsize=16)
            
            motor_names = self.motor_names
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink']
            
            # 绘制每个关节的位置轨迹
            for i in range(min(NUM_MOTORS, self.motor_count)):  # 使用原有的NUM_MOTORS保持可视化兼容性
                row = i // 3
                col = i % 3
                ax = axes[row, col]
                
                # 位置轨迹（度）
                positions_deg = np.degrees(trajectory_array[:, i])
                ax.plot(time_points, positions_deg, color=colors[i], linewidth=2, label=f'{motor_names[i]} pos')
                
                # 标记起始和结束点
                ax.plot(time_points[0], np.degrees(start_positions[i]), 'ro', markersize=8, label='start point')
                ax.plot(time_points[-1], 0, 'go', markersize=8, label='target point(0°)')
                
                ax.set_xlabel('time s')
                ax.set_ylabel('pos degress')
                ax.set_title(f'{motor_names[i]} ')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # 添加数值信息
                start_deg = np.degrees(start_positions[i])
                ax.text(0.02, 0.98, f'start: {start_deg:.1f}°\nend: 0.0°\nchagne: {-start_deg:.1f}°', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # 第6个子图：显示所有关节的综合信息
            ax_summary = axes[1, 2]
            
            # 计算每个时间点的总偏差
            total_deviation = np.sqrt(np.sum(trajectory_array**2, axis=1))
            ax_summary.plot(time_points, np.degrees(total_deviation), 'black', linewidth=3, label='total error')
            ax_summary.set_xlabel('time (s)')
            ax_summary.set_ylabel('total error (degress)')
            ax_summary.set_title('overview')
            ax_summary.grid(True, alpha=0.3)
            ax_summary.legend()
            
            # 添加进度信息
            max_deviation = np.degrees(np.max(total_deviation))
            ax_summary.text(0.02, 0.98, f'max err: {max_deviation:.1f}°\n time: {time_points[-1]:.1f}s\npoints: {len(time_points)}', 
                           transform=ax_summary.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # plt.tight_layout()
            # plt.show(block=False)  # 非阻塞显示
            
            # 打印轨迹摘要
            debug_print("=== 轨迹预览摘要 ===")
            debug_print(f"轨迹时长: {time_points[-1]:.2f}s")
            debug_print(f"轨迹点数: {len(time_points)}")
            debug_print(f"更新频率: {len(time_points)/time_points[-1]:.1f} Hz")
            
            for i, name in enumerate(motor_names):
                start_deg = np.degrees(start_positions[i])
                debug_print(f"{name}: {start_deg:6.1f}° → 0.0° (变化: {-start_deg:6.1f}°)")
            
            max_total_dev = np.degrees(np.max(total_deviation))
            debug_print(f"最大总偏差: {max_total_dev:.1f}°")
            debug_print("==================")
            
        except ImportError:
            debug_print("matplotlib未安装，跳过轨迹可视化", 'WARNING')
            debug_print("可以通过 pip install matplotlib 安装matplotlib来启用可视化功能", 'INFO')
            
            # 提供文本版本的轨迹预览
            debug_print("=== 文本版轨迹预览 ===")
            debug_print(f"轨迹时长: {time_points[-1]:.2f}s")
            debug_print(f"轨迹点数: {len(time_points)}")
            
            motor_names = self.motor_names
            for i, name in enumerate(motor_names):
                start_deg = np.degrees(start_positions[i])
                debug_print(f"{name}: {start_deg:6.1f}° → 0.0° (变化: {-start_deg:6.1f}°)")
            debug_print("=====================")
            
        except Exception as e:
            debug_print(f"轨迹可视化失败: {e}", 'ERROR')

    def set_zero_position(self) -> bool:
        """
        将当前位置设置为零位 (软件零位)
        注意: 这不会改变电机的硬件零位，只是软件层面的零位偏移
        
        Returns:
            bool: 是否成功设置零位
        """
        debug_print("设置当前位置为软件零位...")
        
        try:
            current_positions = self.get_joint_positions()
            if current_positions is None:
                debug_print("无法获取当前位置", 'ERROR')
                return False
            
            # 这里可以实现软件零位偏移逻辑
            # 由于DM电机的特性，我们主要通过记录偏移量来实现
            debug_print(f"当前位置已记录为零位: {[f'{np.degrees(pos):.2f}°' for pos in current_positions]}")
            debug_print("注意: 这是软件零位，重启后需要重新设置", 'WARNING')
            
            return True
            
        except Exception as e:
            debug_print(f"设置零位失败: {e}", 'ERROR')
            return False

    @property
    def motors(self):
        return self.motor_manager.motors

    def set_all_zero_positions(self) -> bool:
        """
        设置所有关节的零点位置
        
        Returns:
            bool: 是否成功设置所有关节零点
        """
        debug_print("设置所有关节零点位置...")
        
        try:
            # 获取当前所有关节位置
            current_positions = self.get_positions_degrees()
            if current_positions is None or len(current_positions) == 0:
                debug_print("无法获取当前位置", 'ERROR')
                return False
            
            # 显示当前位置信息
            for i, name in enumerate(MOTOR_LIST):
                if i < len(current_positions):
                    debug_print(f"{name}: 当前位置 {current_positions[i]:.2f}° 将设为零点")
            
            # 使用MotorManager的set_all_zero方法设置所有电机的零点
            success = self.motor_manager.set_all_zero()
            
            if success:
                debug_print(f"✓ 所有关节零点设置成功")
            else:
                debug_print(f"⚠ 部分关节零点设置失败", 'WARNING')
            
            return success
            
        except Exception as e:
            debug_print(f"设置所有零点失败: {e}", 'ERROR')
            return False

    def set_single_zero_position(self, motor_name: str) -> bool:
        """
        设置单个关节的零点位置
        
        Args:
            motor_name: 电机名称 (m1, m2, m3, m4, m5)
            
        Returns:
            bool: 是否成功设置零点
        """
        debug_print(f"设置 {motor_name} 零点位置...")
        
        try:
            # 获取电机ID
            motor_id = None
            for i, name in enumerate(MOTOR_LIST):
                if name == motor_name:
                    motor_id = i + 1  # 电机ID从1开始
                    break
            
            if motor_id is None:
                debug_print(f"无效的电机名称: {motor_name}", 'ERROR')
                return False
            
            # 获取电机对象
            motor = self.motor_manager.get_motor(motor_id)
            if motor is None:
                debug_print(f"无法获取电机 {motor_name} (电机ID: {motor_id})", 'ERROR')
                return False
            
            # 获取当前位置
            state = motor.get_state()
            current_pos = state['position']
            if current_pos is None:
                debug_print(f"{motor_name}: 无法获取当前位置", 'ERROR')
                return False
            
            # 显示当前位置
            debug_print(f"{motor_name}: 当前位置 {np.degrees(current_pos):.2f}° 将设为零点")
            
            # 设置零点
            success = motor.set_zero()
            
            if success:
                debug_print(f"✓ {motor_name} 零点设置成功")
                debug_print("注意: 这是软件零位，重启后需要重新设置", 'WARNING')
            else:
                debug_print(f"⚠ {motor_name} 零点设置失败", 'ERROR')
            
            return success
            
        except Exception as e:
            debug_print(f"设置 {motor_name} 零点失败: {e}", 'ERROR')
            return False
    def cal_gravity_full(self):
        return self.gc.calculate_torque(self.q, self.dq, self.ddq)

    def cal_gravity_coriolis(self):
        return self.gc.calculate_coriolis_torque(self.q, self.dq)
    def cal_gravity(self):
        """计算重力补偿力矩"""
        if self.gc_flag:
            self._refresh_all_states_ultra_fast()
            return self.gc.get_gravity_compensation_torque(self.q)
        else:
            return np.zeros(self.motor_count)

    def start_gravity_compensation_mode(self, duration=None, update_rate=100):
        """
        启动重力补偿模式
        
        Args:
            duration: 运行时长(秒)，None为无限运行
            update_rate: 更新频率(Hz)
        """
        if not self.gc_flag:
            print("❌ 重力补偿未启用，请在初始化时设置gc=True")
            return False
        
        print("=== 启动重力补偿模式 ===")
        print(f"更新频率: {update_rate} Hz")
        print(f"运行时长: {'无限制' if duration is None else f'{duration}秒'}")
        print("按 Ctrl+C 停止")
        
        # 切换到重力补偿控制参数
        self._switch_to_gravity_compensation_mode()
        
        dt = 1.0 / update_rate
        start_time = time.time()
        
        try:
            while True:
                loop_start = time.time()
                
                # 更新状态
                self._refresh_all_states()
                
                # 计算重力补偿力矩
                tau_compensation = self.cal_gravity()
                if tau_compensation.ndim > 1:
                    tau_compensation = tau_compensation.flatten()
                
                # 应用重力补偿力矩到各电机
                self.set_joint_torque(
                    tau_compensation # 重力补偿力矩
                )
                
                # 显示状态
                elapsed = time.time() - start_time
                # if int(elapsed * 10) % 10 == 0:  # 每0.1秒显示一次
                    # pos_str = " ".join([f"{np.degrees(p):6.1f}°" for p in self.q])
                    # vel_str = " ".join([f"{np.degrees(v):6.1f}°/s" for v in self.dq])
                tau_str = " ".join([f"{t:6.2f}" for t in tau_compensation])
                tau_real = self.get_joint_torques()
                tau_real_str = " ".join([f"{t:6.2f}" for t in tau_real])
                print(f"\n期望力矩: [{tau_str}]\n实际力矩: [{tau_real_str}]\n[{elapsed:6.1f}s]", 
                      end="", flush=True)
                
                # 检查运行时长
                if duration is not None and elapsed >= duration:
                    break
                
                # 控制循环频率
                loop_time = time.time() - loop_start
                if loop_time < dt:
                    time.sleep(dt - loop_time)
                    
        except KeyboardInterrupt:
            print("\n用户中断重力补偿")
        except Exception as e:
            print(f"\n重力补偿模式出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 恢复原始控制参数
            self._restore_normal_mode()
            print("\n重力补偿模式结束")
        
        return True
    
    def _switch_to_gravity_compensation_mode(self):
        """切换到重力补偿模式的控制参数"""
        print("切换到重力补偿控制参数...")
        
        # for motor_name, config in motor_config_gc.items():
        #     if motor_name in self.motors:
        #         motor = self.motors[motor_name]
        #         # 应用重力补偿模式的参数
        #         motor.set_torque_control(
        #             position=0.0,
        #             velocity=0.0, 
        #             kp=config['kp'],      # 0
        #             kd=config['kd'],      # 0
        #             torque=config['torque']  # 0
        #         )
        self.motor_config = motor_config_gc
        time.sleep(0.1)  # 等待参数生效
        print("✓ 已切换到重力补偿模式")
    
    def _restore_normal_mode(self):
        """恢复正常控制模式的参数"""
        print("恢复正常控制参数...")
        
        # for motor_name, config in motor_config.items():
        #     if motor_name in self.motors:
        #         motor = self.motors[motor_name]
        #         # 恢复正常模式的参数
        #         motor.set_torque_control(
        #             position=0.0,
        #             velocity=0.0,
        #             kp=config['kp'],
        #             kd=config['kd'], 
        #             torque=config['torque']
        #         )
        self.motor_config = motor_config
        time.sleep(0.1)  # 等待参数生效
        print("✓ 已恢复正常控制模式")

    def pseudo_gravity_compensation(self, update_rate=50.0, duration=None, 
                                   kp_scale=1.0, kd_scale=1.0, enable_logging=True):
        """
        伪重力补偿：实时读取关节角度并设置为位置目标
        
        这个方法会持续运行一个控制循环：
        1. 读取当前关节位置
        2. 将当前位置设置为新的目标位置
        3. 通过PD控制器保持当前姿态，抵抗外力（如重力）
        
        Args:
            update_rate: 控制循环频率 (Hz)，建议20-100Hz
            duration: 运行时长 (秒)，None为无限制
            kp_scale: Kp增益缩放因子，用于调整位置控制强度
            kd_scale: Kd增益缩放因子，用于调整阻尼
            enable_logging: 是否启用详细日志
            
        Returns:
            bool: 是否正常结束（True）或异常退出（False）
        """
        debug_print("=== 启动伪重力补偿模式 ===")
        debug_print(f"控制频率: {update_rate:.1f} Hz")
        debug_print(f"运行时长: {'无限制' if duration is None else f'{duration:.1f}s'}")
        debug_print(f"Kp缩放: {kp_scale:.2f}, Kd缩放: {kd_scale:.2f}")
        debug_print("按 Ctrl+C 停止补偿")
        
        try:
            # 验证参数
            validate_type(update_rate, (int, float), 'update_rate')
            validate_type(duration, (int, float, type(None)), 'duration')
            validate_type(kp_scale, (int, float), 'kp_scale')
            validate_type(kd_scale, (int, float), 'kd_scale')
            
            if update_rate <= 0 or update_rate > 200:
                raise ValueError(f"update_rate应在(0, 200]范围内，当前: {update_rate}")
            if duration is not None and duration <= 0:
                raise ValueError(f"duration应为正数或None，当前: {duration}")
            if kp_scale <= 0 or kd_scale <= 0:
                raise ValueError(f"kp_scale和kd_scale应为正数，当前: kp={kp_scale}, kd={kd_scale}")
            
            # 计算控制周期
            dt = 1.0 / update_rate
            
            # 启用所有电机
            debug_print("启用所有电机...")
            self.enable_all_motors()
            time.sleep(0.1)
            
            # 读取初始位置
            debug_print("读取初始位置...")
            self._refresh_all_states()
            initial_positions = self.q.copy()
            debug_print(f"初始位置 (度): {np.degrees(initial_positions)}")
            
            # 初始化统计变量
            loop_count = 0
            start_time = time.time()
            last_log_time = start_time
            max_position_change = np.zeros(self.motor_count)
            total_position_change = np.zeros(self.motor_count)
            
            # 主控制循环
            debug_print("开始重力补偿控制循环...")
            
            while True:
                loop_start_time = time.time()
                
                # 检查运行时间
                if duration is not None and (loop_start_time - start_time) >= duration:
                    debug_print(f"达到预设运行时长 {duration:.1f}s，正常结束")
                    break
                
                try:
                    # 1. 快速读取当前位置（避免过多调试输出）
                    self._refresh_all_states_ultra_fast()
                    current_positions = self.q.copy()
                    
                    # 2. 将当前位置设置为目标位置
                    # 使用当前配置的PD参数，但可以通过缩放因子调整
                    for i, motor_name in enumerate(self.motor_names):
                        motor = self.motors[motor_name]
                        config = self.motor_config[motor_name]
                        
                        # 应用缩放因子
                        kp = config['kp'] * kp_scale
                        kd = config['kd'] * kd_scale
                        torque_ff = config['torque']  # 前馈力矩保持不变
                        
                        # 发送位置命令（目标位置=当前位置）
                        self.mc.controlMIT(motor, current_positions[i], 0.0, kp, kd, torque_ff)
                    
                    # 3. 统计和日志
                    loop_count += 1
                    
                    # 计算位置变化
                    if loop_count > 1:
                        position_change = np.abs(current_positions - initial_positions)
                        max_position_change = np.maximum(max_position_change, position_change)
                        total_position_change += position_change
                    
                    # 定期日志输出
                    if enable_logging and (loop_start_time - last_log_time) >= 2.0:  # 每2秒输出一次
                        elapsed = loop_start_time - start_time
                        actual_freq = loop_count / elapsed if elapsed > 0 else 0
                        
                        debug_print(f"补偿运行中... 时间: {elapsed:.1f}s, 频率: {actual_freq:.1f}Hz")
                        debug_print(f"当前位置 (度): {np.degrees(current_positions)}")
                        debug_print(f"最大偏移 (度): {np.degrees(max_position_change)}")
                        
                        last_log_time = loop_start_time
                    
                    # 4. 控制循环时序
                    loop_duration = time.time() - loop_start_time
                    sleep_time = dt - loop_duration
                    
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    elif enable_logging and loop_count % 100 == 0:  # 偶尔警告时序问题
                        debug_print(f"警告: 控制循环超时 {loop_duration*1000:.1f}ms > {dt*1000:.1f}ms", 'WARNING')
                
                except KeyboardInterrupt:
                    debug_print("用户中断，停止重力补偿")
                    break
                except Exception as e:
                    debug_print(f"控制循环异常: {e}", 'ERROR')
                    if enable_logging:
                        debug_print(f"详细错误: {traceback.format_exc()}", 'ERROR')
                    # 继续运行，不因单次异常而退出
                    continue
            
            # 输出最终统计
            total_time = time.time() - start_time
            avg_freq = loop_count / total_time if total_time > 0 else 0
            avg_position_change = total_position_change / max(loop_count - 1, 1)
            
            debug_print("=== 重力补偿统计 ===")
            debug_print(f"运行时长: {total_time:.2f}s")
            debug_print(f"控制循环次数: {loop_count}")
            debug_print(f"平均频率: {avg_freq:.1f} Hz")
            debug_print(f"目标频率: {update_rate:.1f} Hz")
            debug_print(f"频率达成率: {(avg_freq/update_rate)*100:.1f}%")
            debug_print(f"最大位置偏移 (度): {np.degrees(max_position_change)}")
            debug_print(f"平均位置偏移 (度): {np.degrees(avg_position_change)}")
            debug_print("===================")
            
            return True
            
        except KeyboardInterrupt:
            debug_print("用户中断重力补偿")
            return True
        except Exception as e:
            debug_print(f"重力补偿失败: {e}", 'ERROR')
            debug_print(f"详细错误: {traceback.format_exc()}", 'ERROR')
            return False
        finally:
            # 安全清理
            try:
                debug_print("清理资源...")
                # 可选择是否禁用电机，通常保持启用状态
                # self.disable_all_motors()
                debug_print("重力补偿模式结束")
            except Exception as e:
                debug_print(f"清理资源时出错: {e}", 'ERROR')

    def monitor_positions_continuous(self, update_rate=10.0, duration=None, 
                                    save_csv=False, csv_filename=None):
        """
        连续监控电机位置
        
        Args:
            update_rate: 更新频率 (Hz)
            duration: 监控时长 (秒)，None为无限制
            save_csv: 是否保存CSV文件
            csv_filename: CSV文件名
        """
        import time
        import csv
        from datetime import datetime
        
        print(f"开始连续位置监控...")
        print(f"更新频率: {update_rate} Hz")
        print(f"监控时长: {duration if duration else '无限制'} 秒")
        print(f"CSV保存: {'启用' if save_csv else '禁用'}")
        print("按 Ctrl+C 停止监控\n")
        
        # 准备CSV文件
        csv_file = None
        csv_writer = None
        if save_csv:
            if csv_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_filename = f"position_monitor_{timestamp}.csv"
            
            csv_file = open(csv_filename, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            # 写入表头
            headers = ['timestamp', 'time_s'] + [f'm{i+1}_pos_deg' for i in range(self.motor_count)] + [f'm{i+1}_vel_deg_s' for i in range(self.motor_count)]
            csv_writer.writerow(headers)
            print(f"CSV文件: {csv_filename}")
        
        start_time = time.time()
        update_interval = 1.0 / update_rate
        
        try:
            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # 检查是否超过监控时长
                if duration and elapsed_time >= duration:
                    print(f"\n监控时长达到 {duration} 秒，自动停止")
                    break
                
                # 获取当前状态
                try:
                    positions = self.get_positions_degrees(refresh=True)
                    velocities = self.get_velocities_degrees(refresh=False)  # 使用已刷新的数据
                    
                    # 显示位置信息
                    pos_str = " ".join([f"{pos:6.1f}°" for pos in positions])
                    vel_str = " ".join([f"{vel:6.1f}°/s" for vel in velocities])
                    
                    print(f"\r[{elapsed_time:6.1f}s] 位置: [{pos_str}] 速度: [{vel_str}]", end="", flush=True)
                    
                    if self.gc_flag:
                        tau = self.cal_gravity()
                        # tau = self.cal_gravity_coriolis()
                        print(tau)
                    # 保存到CSV
                    if save_csv and csv_writer:
                        timestamp = datetime.now().isoformat()
                        row = [timestamp, elapsed_time] + list(positions) + list(velocities)
                        csv_writer.writerow(row)
                        csv_file.flush()  # 确保数据写入
                    
                except Exception as e:
                    print(f"\n读取状态时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # 等待下次更新
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print(f"\n\n用户中断监控 (Ctrl+C)")
        except Exception as e:
            print(f"\n监控过程中出错: {e}")
        finally:
            if csv_file:
                csv_file.close()
                print(f"CSV文件已保存: {csv_filename}")
            print("监控结束")
    
    def get_velocities_degrees(self, refresh=True):
        """
        获取所有关节速度 (度/秒)
        
        Args:
            refresh: 是否刷新状态
            
        Returns:
            np.ndarray: 关节速度数组 (度/秒)
        """
        if refresh:
            self._refresh_all_states()
        
        return np.degrees(self.dq)

    # ========== INFORMATION FUNCTIONS ==========
    
    def _read_motor_info(self):
        """Read and display motor information"""
        print("="*80)
        print(f"{'Motor':<8} {'ID':<4} {'Position':<12} {'Velocity':<12} {'Torque':<12} {'Status':<10}")
        print("-"*80)
        
        for motor_id, motor in self.motor_manager.motors.items():
            try:
                # Update motor state
                motor.update_state()
                
                # Get motor information
                position = motor.get_position()
                velocity = motor.get_velocity()
                torque = motor.get_torque()
                status = "OK"
            except Exception as e:
                position = velocity = torque = "ERROR"
                status = "FAIL"
                print(e)
            
            time.sleep(0.001)
            print(f"{motor.info.name:<8} {motor_id:<4} {position:<12.4f} {velocity:<12.4f} {torque:<12.4f} {status:<10}")
        
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
        
        for i in range(self.motor_count):
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
            # No need to close serial_device in unified motor control system
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
    arm = ICARM(debug=False)
    # arm.connect()
    try:
        # Test single joint movement
        print("Testing single joint movement...")
        arm.enable_all_motors()
        succes = arm.home_to_zero(speed=0.3, timeout=30.0)       
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
