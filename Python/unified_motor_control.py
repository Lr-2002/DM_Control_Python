#!/usr/bin/env python3
"""
统一电机控制接口
支持Damiao、High Torque和舵机的统一控制
"""

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Any, Union
import time
import numpy as np
from dataclasses import dataclass

# 导入必要的模块
try:
    # 当从Python目录内部导入时
    from damiao import Control_Mode_Code
    from src import can_value_type
except ImportError:
    # 当从上层目录导入时
    from Python.damiao import Control_Mode_Code
    from Python.src import can_value_type


class MotorType(IntEnum):
    """电机类型枚举"""
    DAMIAO = 1
    HIGH_TORQUE = 2
    SERVO = 3


@dataclass
class MotorInfo:
    """电机信息配置"""
    motor_id: int
    motor_type: MotorType
    can_id: int
    master_id: Optional[int] = None
    name: str = ""
    
    # 控制参数
    kp: float = 0.0
    kd: float = 0.0
    torque_offset: float = 0.0
    
    # 限制参数 [position_limit, velocity_limit, torque_limit]
    limits: List[float] = None
    
    def __post_init__(self):
        if self.limits is None:
            self.limits = [12.5, 50.0, 10.0]  # 默认限制
        if not self.name:
            self.name = f"motor_{self.motor_id}"


@dataclass
class MotorFeedback:
    """电机反馈数据"""
    position: float = 0.0      # 位置 (rad)
    velocity: float = 0.0      # 速度 (rad/s)
    torque: float = 0.0        # 力矩 (N·m)
    error_code: int = 0        # 错误代码
    timestamp: float = 0.0     # 时间戳


class MotorProtocol(ABC):
    """电机通信协议抽象基类"""
    
    def __init__(self, usb_hw):
        self.usb_hw = usb_hw
        self.motors = {}  # motor_id -> motor_instance
        
    @abstractmethod
    def add_motor(self, motor_id: int, **config) -> bool:
        """添加电机到协议中"""
        pass
    
    @abstractmethod
    def enable_motor(self, motor_id: int) -> bool:
        """使能电机"""
        pass
    
    @abstractmethod
    def disable_motor(self, motor_id: int) -> bool:
        """失能电机"""
        pass
    
    @abstractmethod
    def set_command(self, motor_id: int, pos: float, vel: float, 
                   kp: float, kd: float, tau: float) -> bool:
        """设置电机命令"""
        pass
    
    @abstractmethod
    def send_commands(self) -> bool:
        """发送所有设置的命令"""
        pass
    
    @abstractmethod
    def read_feedback(self, motor_id: int) -> MotorFeedback:
        """读取电机反馈"""
        pass
    
    @abstractmethod
    def set_zero_position(self, motor_id: int) -> bool:
        """设置零位"""
        pass
    
    @abstractmethod
    def get_limits(self, motor_id: int) -> Tuple[float, float, float]:
        """获取电机限制参数"""
        pass


class DamiaoProtocol(MotorProtocol):
    """达妙电机协议实现"""
    
    def __init__(self, usb_hw, motor_control_instance):
        super().__init__(usb_hw)
        self.motor_control = motor_control_instance  # 现有的Motor_Control实例
        self.pending_commands = {}  # motor_id -> command
        
    def add_motor(self, motor_id: int, **config) -> bool:
        """添加达妙电机"""
        try:
            can_id = config['can_id']
            # 从现有的Motor_Control中获取电机实例
            dm_motor = self.motor_control.getMotor(can_id)
            if dm_motor:
                self.motors[motor_id] = dm_motor
                print(f"Damiao motor {motor_id} (CAN ID: {can_id}) added successfully")
                return True
            else:
                print(f"Failed to find Damiao motor with CAN ID {can_id}")
                return False
        except Exception as e:
            print(f"Failed to add Damiao motor {motor_id}: {e}")
            return False
    
    def enable_motor(self, motor_id: int) -> bool:
        """达妙电机使能"""
        if motor_id not in self.motors:
            return False
        try:
            motor = self.motors[motor_id]
            # 切换到MIT控制模式
            self.motor_control.switchControlMode(motor, Control_Mode_Code.MIT)
            time.sleep(0.01)  # 等待模式切换
            
            # 发送使能命令
            for _ in range(3):
                self.motor_control.control_cmd(motor.GetCanId() + motor.GetMotorMode(), 0xFC)
                time.sleep(0.002)
            return True
        except Exception as e:
            print(f"Failed to enable Damiao motor {motor_id}: {e}")
            return False
    
    def disable_motor(self, motor_id: int) -> bool:
        """达妙电机失能"""
        if motor_id not in self.motors:
            return False
        try:
            motor = self.motors[motor_id]
            # 发送失能命令
            for _ in range(3):
                self.motor_control.control_cmd(motor.GetCanId() + motor.GetMotorMode(), 0xFD)
                time.sleep(0.002)
            return True
        except Exception as e:
            print(f"Failed to disable Damiao motor {motor_id}: {e}")
            return False
    
    def set_command(self, motor_id: int, pos: float, vel: float, 
                   kp: float, kd: float, tau: float) -> bool:
        """设置达妙电机命令（立即发送）"""
        if motor_id not in self.motors:
            return False
        
        try:
            motor = self.motors[motor_id]
            # 直接调用现有的control_mit方法
            self.motor_control.control_mit(motor, kp, kd, pos, vel, tau)
            return True
        except Exception as e:
            print(f"Failed to set command for Damiao motor {motor_id}: {e}")
            return False
    
    def send_commands(self) -> bool:
        """达妙电机命令已在set_command中发送"""
        return True
    
    def read_feedback(self, motor_id: int) -> MotorFeedback:
        """读取达妙电机反馈"""
        if motor_id not in self.motors:
            return MotorFeedback()  # 电机不存在时返回空反馈
        
        try:
            motor = self.motors[motor_id]
            # 刷新电机状态

            self.motor_control.refresh_motor_status(motor)
            
            return MotorFeedback(
                position=motor.Get_Position(),
                velocity=motor.Get_Velocity(),
                torque=motor.Get_tau(),
                error_code=0,  # 达妙电机暂时没有错误码，默认为0
                timestamp=time.time()
            )
        except Exception as e:
            print(f"Failed to read feedback from Damiao motor {motor_id}: {e}")
            return MotorFeedback(  # 异常时返回错误状态的反馈
                position=0.0,
                velocity=0.0,
                torque=0.0,
                error_code=-1,  # 错误码-1表示读取失败
                timestamp=time.time()
            )
    
    def set_zero_position(self, motor_id: int) -> bool:
        """设置达妙电机零位"""
        if motor_id not in self.motors:
            return False
        
        try:
            motor = self.motors[motor_id]
            self.motor_control.set_zero_position(motor)
            return True
        except Exception as e:
            print(f"Failed to set zero position for Damiao motor {motor_id}: {e}")
            return False
    
    def get_limits(self, motor_id: int) -> Tuple[float, float, float]:
        """获取达妙电机限制参数"""
        if motor_id not in self.motors:
            return (12.5, 50.0, 10.0)  # 默认值
        
        motor = self.motors[motor_id]
        limits = motor.get_limit_param()
        return tuple(limits)


class HTProtocol(MotorProtocol):
    """High Torque电机协议实现"""
    
    def __init__(self, usb_hw, ht_manager_instance):
        super().__init__(usb_hw)
        self.ht_manager = ht_manager_instance  # 现有的HTMotorManager实例
        self.pending_commands = {}  # motor_id -> (pos, vel, kp, kd, tau)
        
    def add_motor(self, motor_id: int, **config) -> bool:
        """添加HT电机"""
        try:
            ht_motor_id = config['ht_motor_id']
            # 通过HTMotorManager添加电机
            self.ht_manager.add_motor(ht_motor_id)
            ht_motor = self.ht_manager.get_motor(ht_motor_id)
            if ht_motor:
                self.motors[motor_id] = ht_motor
                print(f"HT motor {motor_id} (HT ID: {ht_motor_id}) added successfully")
                return True
            else:
                print(f"Failed to find HT motor with ID {ht_motor_id}")
                return False
        except Exception as e:
            print(f"Failed to add HT motor {motor_id}: {e}")
            return False
    
    def enable_motor(self, motor_id: int) -> bool:
        """HT电机使能（HT电机不支持单独enable，通过控制命令激活）"""
        if motor_id not in self.motors:
            return False
        try:
            # HT电机通过发送控制命令来激活，这里可以发送一个零位命令来激活
            motor = self.motors[motor_id]
            # 发送一个小的位置命令来激活电机
            return True
        except Exception as e:
            print(f"Failed to enable HT motor {motor_id}: {e}")
            return False
    
    def disable_motor(self, motor_id: int) -> bool:
        """HT电机失能"""
        if motor_id not in self.motors:
            return False
        try:
            # HT电机的制动通过manager处理
            motor = self.motors[motor_id]
            # 发送单个电机制动命令
            ht_motor_id = motor.motor_id
            self.ht_manager._send_raw((0x80 | ht_motor_id) << 8 | ht_motor_id, 
                                    [0x01, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x00, 0x00])
            return True
        except Exception as e:
            print(f"Failed to disable HT motor {motor_id}: {e}")
            # 如果单个电机制动失败，尝试全部制动
            try:
                self.ht_manager.brake()
                return True
            except:
                return False
    
    def set_command(self, motor_id: int, pos: float, vel: float, 
                   kp: float, kd: float, tau: float) -> bool:
        """设置HT电机命令（缓存，等待批量发送）"""
        if motor_id not in self.motors:
            return False
        
        self.pending_commands[motor_id] = (pos, vel, kp, kd, tau)
        return True
    
    def send_commands(self) -> bool:
        """批量发送HT电机命令"""
        if not self.pending_commands:
            return True
        
        try:
            # 构建批量命令数组
            motor_ids = sorted(self.pending_commands.keys())
            pos_list = []
            vel_list = []
            kp_list = []
            kd_list = []
            torque_list = []
            
            for motor_id in motor_ids:
                pos, vel, kp, kd, tau = self.pending_commands[motor_id]
                pos_list.append(pos)
                vel_list.append(vel)
                kp_list.append(kp)
                kd_list.append(kd)
                torque_list.append(tau)
            
            # 使用HTMotorManager的批量控制方法
            self.ht_manager.mit_control(
                pos_list=pos_list,
                vel_list=vel_list,
                kp_list=kp_list,
                kd_list=kd_list,
                torque_list=torque_list
            )
            
            # 清空待发送命令
            self.pending_commands.clear()
            return True
            
        except Exception as e:
            print(f"Failed to send HT batch commands: {e}")
            return False
    
    def read_feedback(self, motor_id: int) -> MotorFeedback:
        """读取HT电机反馈"""
        if motor_id not in self.motors:
            return MotorFeedback()
        
        try:
            motor = self.motors[motor_id]
            # 刷新电机状态
            self.ht_manager.refresh_motor_status()
            time.sleep(0.001)  # 等待反馈
            
            return MotorFeedback(
                position=motor.position,
                velocity=motor.velocity, 
                torque=motor.torque,
                error_code=motor.error,
                timestamp=time.time()
            )
        except Exception as e:
            print(f"Failed to read feedback from HT motor {motor_id}: {e}")
            return MotorFeedback()
    
    def set_zero_position(self, motor_id: int) -> bool:
        """设置HT电机零位"""
        if motor_id not in self.motors:
            return False
        try:
            motor = self.motors[motor_id]
            # HT电机设置零位
            ht_motor_id = motor.motor_id
            self.ht_manager._send_raw((0x80 | ht_motor_id) << 8 | ht_motor_id, 
                                    [0x40, 0x01, 0x04, 0x64, 0x20, 0x63, 0x0a])
            return True
        except Exception as e:
            print(f"Failed to set zero position for HT motor {motor_id}: {e}")
            return False
    
    def get_limits(self, motor_id: int) -> Tuple[float, float, float]:
        """获取HT电机限制参数"""
        # HT电机的默认限制参数
        return (12.5, 50.0, 20.0)


class UnifiedMotor:
    """统一电机接口"""
    
    def __init__(self, motor_id: int, protocol: MotorProtocol, motor_info: MotorInfo):
        self.motor_id = motor_id
        self.protocol = protocol
        self.info = motor_info
        self.feedback = MotorFeedback()
        
    # 细粒度接口
    def get_position(self) -> float:
        """获取位置"""
        return self.feedback.position
    
    def get_velocity(self) -> float:
        """获取速度"""
        return self.feedback.velocity
    
    def get_torque(self) -> float:
        """获取力矩"""
        return self.feedback.torque
    
    def get_error_code(self) -> int:
        """获取错误码"""
        return self.feedback.error_code
    
    # 完整状态接口
    def get_state(self) -> Dict[str, float]:
        """获取完整状态"""
        return {
            'position': self.feedback.position,
            'velocity': self.feedback.velocity,
            'torque': self.feedback.torque,
            'error_code': self.feedback.error_code,
            'timestamp': self.feedback.timestamp
        }
    
    def update_state(self) -> bool:
        """更新状态"""
        try:
            self.feedback = self.protocol.read_feedback(self.motor_id)
            return True
        except Exception as e:
            print(f"Failed to update state for motor {self.motor_id}: {e}")
            return False
    
    # 控制接口
    def set_command(self, pos: float, vel: float, kp: float, kd: float, tau: float) -> bool:
        """设置MIT控制命令"""
        return self.protocol.set_command(self.motor_id, pos, vel, kp, kd, tau)
    
    def enable(self) -> bool:
        """使能电机"""
        return self.protocol.enable_motor(self.motor_id)
    
    def disable(self) -> bool:
        """失能电机"""
        return self.protocol.disable_motor(self.motor_id)
    
    def set_zero(self) -> bool:
        """设置零位"""
        return self.protocol.set_zero_position(self.motor_id)


class CANFrameDispatcher:
    """CAN帧分发器 - 解决多协议共享USB硬件的回调冲突"""
    
    def __init__(self, usb_hw):
        self.usb_hw = usb_hw
        self.handlers = {}  # protocol_name -> handler_function
        # 设置统一的回调函数
        self.usb_hw.setFrameCallback(self._unified_callback)
    
    def register_handler(self, protocol_name: str, handler_func):
        """注册协议处理函数"""
        self.handlers[protocol_name] = handler_func
        print(f"Registered CAN frame handler for {protocol_name}")
    
    def _unified_callback(self, frame: can_value_type):
        """统一的CAN帧回调函数"""
        can_id = frame.head.id
        
        # 根据CAN ID范围分发到不同的协议处理器
        try:
            # 达妙电机: ID范围通常是 0x01-0x06, 0x11-0x16 等
            if can_id <= 0x100:  # 达妙电机ID范围
                if 'damiao' in self.handlers:
                    self.handlers['damiao'](frame)
            
            # HT电机: ID范围通常是 0x700, 0x800 等
            elif can_id >= 0x700:  # HT电机ID范围
                if 'ht' in self.handlers:
                    self.handlers['ht'](frame)
            
            # 其他协议可以在这里添加
            
        except Exception as e:
            print(f"Error in CAN frame dispatch: {e}")


class MotorManager:
    """统一电机管理器"""
    
    def __init__(self, usb_hw):
        self.usb_hw = usb_hw
        self.protocols = {}  # protocol_type -> protocol_instance
        self.motors = {}     # motor_id -> UnifiedMotor
        
        # 创建CAN帧分发器
        self.can_dispatcher = CANFrameDispatcher(usb_hw)
        
    def add_damiao_protocol(self, motor_control_instance):
        """添加达妙电机协议"""
        protocol = DamiaoProtocol(self.usb_hw, motor_control_instance)
        self.protocols['damiao'] = protocol
        # 注册达妙电机的CAN帧处理函数
        self.can_dispatcher.register_handler('damiao', motor_control_instance.canframeCallback)
    
    def add_ht_protocol(self, ht_manager_instance):
        """添加HT电机协议"""
        protocol = HTProtocol(self.usb_hw, ht_manager_instance)
        self.protocols['ht'] = protocol
        # 注册HT电机的CAN帧处理函数
        self.can_dispatcher.register_handler('ht', ht_manager_instance.can_frame_callback)
    
    def add_motor(self, motor_id: int, motor_type: str, motor_info: MotorInfo, **config) -> bool:
        """添加电机"""
        if motor_type not in self.protocols:
            print(f"Protocol {motor_type} not available")
            return False
        
        protocol = self.protocols[motor_type]
        
        # 添加到协议中
        if not protocol.add_motor(motor_id, **config):
            return False
        
        # 创建统一电机实例
        unified_motor = UnifiedMotor(motor_id, protocol, motor_info)
        self.motors[motor_id] = unified_motor
        
        return True
    
    def get_motor(self, motor_id: int) -> Optional[UnifiedMotor]:
        """获取电机实例"""
        return self.motors.get(motor_id)
    
    def update_all_states(self) -> bool:
        """更新所有电机状态"""
        success = True
        for motor in self.motors.values():
            if not motor.update_state():
                success = False
        return success
    
    def send_all_commands(self) -> bool:
        """发送所有协议的命令"""
        success = True
        for protocol in self.protocols.values():
            if not protocol.send_commands():
                success = False
        return success
    
    def enable_all(self) -> bool:
        """使能所有电机"""
        success = True
        for motor in self.motors.values():
            if not motor.enable():
                success = False
        return success
    
    def disable_all(self) -> bool:
        """失能所有电机"""
        success = True
        for motor in self.motors.values():
            if not motor.disable():
                success = False
        return success
    
    def set_all_zero(self) -> bool:
        """设置所有电机零位"""
        success = True
        for motor in self.motors.values():
            if not motor.set_zero():
                success = False
        return success
    
    # 批量控制接口
    def control_mit_batch(self, motor_ids: List[int], positions: List[float], 
                         velocities: List[float], kps: List[float], 
                         kds: List[float], torques: List[float]) -> bool:
        """批量MIT控制"""
        success = True
        
        # 设置所有命令
        for i, motor_id in enumerate(motor_ids):
            if motor_id in self.motors:
                motor = self.motors[motor_id]
                if not motor.set_command(positions[i], velocities[i], 
                                       kps[i], kds[i], torques[i]):
                    success = False
        
        # 发送所有命令
        if not self.send_all_commands():
            success = False
        
        return success