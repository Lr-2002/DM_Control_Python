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
from dataclasses import dataclass, field
import struct
import sys
import os

import pysnooper

# 添加当前目录到Python路径，确保能找到本地模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ic_arm_control.control.src import can_value_type
from ic_arm_control.control.motor_info import *
from ic_arm_control.control.damiao import Motor as DM_Motor
from ic_arm_control.control.ht_motor import HTMotor as HT_motor
from ic_arm_control.control.servo_motor import ServoMotor as Servo_Motor


class MotorProtocol(ABC):
    """电机通信协议抽象基类"""

    def __init__(self, usb_hw):
        self.usb_hw = usb_hw
        self.motors = {}  # motor_id -> motor_instance
        self.motor_infos = {}  # motor_id -> motor_info

    @abstractmethod
    def add_motor(self, motor_info: MotorInfo) -> bool:
        """添加电机到协议中"""
        self.motor_infos[motor_info.motor_id] = motor_info
        return True

    @abstractmethod
    def enable_motor(self, motor_id: int) -> bool:
        """使能电机"""
        pass

    @abstractmethod
    def disable_motor(self, motor_id: int) -> bool:
        """失能电机"""
        pass

    @abstractmethod
    def set_command(
        self, motor_id: int, pos: float, vel: float, kp: float, kd: float, tau: float
    ) -> bool:
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

    def add_motor(self, motor_info: MotorInfo) -> bool:
        """添加达妙电机"""
        super().add_motor(motor_info)
        motor= DM_Motor(
                motor_info.motor_index,
                Control_Mode.MIT_MODE,
                motor_info.can_id,
                motor_info.master_id,
            )
        self.motors[motor_info.motor_id] = motor
        self.motor_control.addMotor(
            motor
        )
        return True

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
                self.motor_control.control_cmd(
                    motor.GetCanId() + motor.GetMotorMode(), 0xFC
                )
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
                self.motor_control.control_cmd(
                    motor.GetCanId() + motor.GetMotorMode(), 0xFD
                )
                time.sleep(0.002)
            return True
        except Exception as e:
            print(f"Failed to disable Damiao motor {motor_id}: {e}")
            return False

    def set_command(
        self, motor_id: int, pos: float, vel: float, kp: float, kd: float, tau: float
    ) -> bool:
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
        motor = self.motors[motor_id]
        # 刷新电机状态

        self.motor_control.refresh_motor_status(motor)

        return MotorFeedback(
            position=motor.Get_Position(),
            velocity=motor.Get_Velocity(),
            torque=motor.Get_tau(),
            # error_code=0,  # 达妙电机暂时没有错误码，默认为0
            # timestamp=time.time(),
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


class ServoProtocol(MotorProtocol):
    """伺服电机协议实现 - 上层控制接口"""

    def __init__(self, usb_hw, servo_manager_instance=None):
        super().__init__(usb_hw)

        # 使用底层ServoMotorManager实例
        if servo_manager_instance is None:
            from servo_motor import ServoMotorManager

            self.servo_manager = ServoMotorManager(usb_hw, as_sub_module=True)
        else:
            self.servo_manager = servo_manager_instance

        # 电机ID映射 (上层ID -> 底层ServoMotor对象)
        self.motor_mapping = {}

    def add_motor(self, motor_info: MotorInfo) -> bool:
        """添加伺服电机"""

        super().add_motor(motor_info)
        servo =Servo_Motor(
                self.usb_hw,
                motor_info.motor_id,
                motor_info.can_id,
                motor_info.master_id,
            )
        self.servo_manager.add_servo(
            # servo,
            motor_info.motor_id,
            motor_info.can_id,
            motor_info.master_id,
            motor=servo
        )
        self.motor_mapping[motor_info.motor_id] = servo
        self.motors[motor_info.motor_id] = servo
        return True

    def enable_motor(self, motor_id: int) -> bool:
        """使能电机"""
        if motor_id not in self.motor_mapping:
            return False

        servo = self.motor_mapping[motor_id]
        return servo.enable()

    def disable_motor(self, motor_id: int) -> bool:
        """失能电机"""
        if motor_id not in self.motor_mapping:
            return False

        servo = self.motor_mapping[motor_id]
        return servo.disable()

    def set_command(
        self, motor_id: int, pos: float, vel: float, kp: float, kd: float, tau: float
    ) -> bool:
        """设置电机命令"""
        if motor_id not in self.motor_mapping:
            return False

        servo = self.motor_mapping[motor_id]
        # 舵机协议中kp, kd, tau参数不使用，只使用pos和vel
        return servo.set_position(pos, vel)

    def send_commands(self):
        """发送所有缓存的命令"""
        # 舵机协议中命令是立即发送的，这里不需要额外处理
        pass

    def read_feedback(self, motor_id: int):
        """读取电机反馈"""
        servo = self.motor_mapping[motor_id]

        # 请求读取状态
        # servo.read_status()

        # 返回当前状态
        return MotorFeedback(
            position=servo.position,
            velocity=servo.velocity,
            torque=servo.torque,
            # error_code=0,
            # timestamp=time.time()
        )
    def set_zero_position(self, motor_id: int) -> bool:
        """设置当前位置为零位"""
        # 舵机通常不支持软件零位设置
        return False

    def get_limits(self, motor_id: int):
        """获取电机限制参数"""
        if motor_id not in self.motor_mapping:
            return [3.14159, 10.0, 5.0]  # 默认限制

        servo = self.motor_mapping[motor_id]
        return [servo.max_position, servo.max_velocity, servo.max_torque]

    def set_limits(
        self,
        motor_id: int,
        max_pos: float = None,
        min_pos: float = None,
        max_vel: float = None,
        max_torque: float = None,
    ) -> bool:
        """设置电机限制参数"""
        if motor_id not in self.motor_mapping:
            return False

        return self.servo_manager.set_limits(
            motor_id, max_pos, min_pos, max_vel, max_torque
        )

    def emergency_stop(self) -> bool:
        """紧急停止所有舵机"""
        return self.servo_manager.emergency_stop()

    def get_servo_manager(self):
        """获取底层舵机管理器实例"""
        return self.servo_manager


class HTProtocol(MotorProtocol):
    """High Torque电机协议实现"""

    def __init__(self, usb_hw, ht_manager_instance):
        super().__init__(usb_hw)
        self.ht_manager = ht_manager_instance  # 现有的HTMotorManager实例
        self.pending_commands = {}  # motor_id -> (pos, vel, kp, kd, tau)
        self.motor_infos = {}

    def add_motor(self, motor_info: MotorInfo) -> bool:
        """添加HT电机"""
        super().add_motor(motor_info)
        motor = self.ht_manager.add_motor(motor_info.motor_id)
        self.motors[motor_info.motor_id] = motor
        

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
            self.ht_manager._send_raw(
                (0x80 | ht_motor_id) << 8 | ht_motor_id,
                [0x01, 0x00, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00],
            )
            return True
        except Exception as e:
            print(f"Failed to disable HT motor {motor_id}: {e}")
            # 如果单个电机制动失败，尝试全部制动
            try:
                self.ht_manager.brake()
                return True
            except:
                return False

    def set_command(
        self, motor_id: int, pos: float, vel: float, kp: float, kd: float, tau: float
    ) -> bool:
        """设置HT电机命令（缓存，等待批量发送）"""
        if motor_id not in self.motors:
            return False

        self.pending_commands[motor_id] = (pos, vel, kp, kd, tau)
        if len(self.pending_commands) == len(self.motors):
            self.send_commands()
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
                torque_list=torque_list,
            )

            # 清空待发送命令
            self.pending_commands.clear()
            return True

        except Exception as e:
            print(f"Failed to send HT batch commands: {e}")
            return False

    def read_feedback(self, motor_id: int) -> MotorFeedback:
        """读取HT电机反馈"""
        motor = self.motors[motor_id]
        # 刷新电机状态
        self.ht_manager.refresh_motor_status()

        return MotorFeedback(
            position=motor.position,
            velocity=motor.velocity,
            torque=motor.torque,
            # error_code=motor.error,
            # timestamp=time.time(),
        )
    def set_zero_position(self, motor_id: int) -> bool:
        """设置HT电机零位"""
        if motor_id not in self.motors:
            return False
        try:
            motor = self.motors[motor_id]
            # HT电机设置零位
            ht_motor_id = motor.motor_id
            self.ht_manager._send_raw(
                (0x80 | ht_motor_id) << 8 | ht_motor_id,
                [0x40, 0x01, 0x04, 0x64, 0x20, 0x63, 0x0A],
            )
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
        self.motor_info = motor_info
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
            "position": self.feedback.position,
            "velocity": self.feedback.velocity,
            "torque": self.feedback.torque,
            "error_code": self.feedback.error_code,
            "timestamp": self.feedback.timestamp,
        }

    @pysnooper.snoop()
    def update_state(self) -> bool:
        """更新状态"""
        print("update_state", self.motor_id)
        self.feedback = self.protocol.read_feedback(self.motor_id)
        return True
    # 控制接口
    def set_command(
        self, pos: float, vel: float, kp: float, kd: float, tau: float
    ) -> bool:
        """设置MIT控制命令"""
        print("set_command", self.motor_id, pos, vel, kp, kd, tau)
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
        self.hexify = lambda a: [hex(x) for x in a]

    def register_handler(self, protocol_name: str, handler_func):
        """注册协议处理函数"""
        self.handlers[protocol_name] = handler_func
        print(f"Registered CAN frame handler for {protocol_name}")

    def _unified_callback(self, frame: can_value_type):
        """统一的CAN帧回调函数"""
        can_id = frame.head.id
        # print(f"[RECV] from [{hex(frame.head.id)}]: {self.hexify(frame.data)}")
        # 根据CAN ID范围分发到不同的协议处理器
        # 达妙电机: ID范围通常是 0x01-0x06, 0x11-0x16 等
        if can_id <= 0x16:  # 达妙电机ID范围
            if "damiao" in self.handlers:
                self.handlers["damiao"](frame)
        elif can_id <=0x19:
            if 'servo' in self.handlers:
                self.handlers['servo'](frame)
        # HT电机: ID范围通常是 0x700, 0x800 等
        elif can_id >= 0x700:  # HT电机ID范围
            if "ht" in self.handlers:
                self.handlers["ht"](frame)


class MotorManager:
    """统一电机管理器"""

    def __init__(self, usb_hw):
        self.usb_hw = usb_hw
        self.protocols = {}  # protocol_type -> protocol_instance
        self.motors = {}  # motor_id -> UnifiedMotor
        print("totally inited ", len(self.motors), "motors ")
        # 创建CAN帧分发器
        self.can_dispatcher = CANFrameDispatcher(usb_hw)
        self.motor_cnt = 0
        self.motor_infos = {}

    def add_protocol(self, protocol_instance, protocol_type=None):
        """添加协议 - 自动识别协议类型并加载其中的电机"""
        if not protocol_type:
            protocol_type = self._identify_protocol_type(protocol_instance)
        
        if protocol_type == "damiao":
            self.protocols["damiao"] = protocol_instance
            self.can_dispatcher.register_handler(
                "damiao", protocol_instance.motor_control.canframeCallback
            )
            # 加载达妙电机
            self._load_motors_from_protocol(protocol_instance, "damiao")
            return True
            
        elif protocol_type == "ht":
            if "ht" not in self.protocols:
                self.protocols["ht"] = protocol_instance 
                self.can_dispatcher.register_handler(
                    "ht", protocol_instance.ht_manager.can_frame_callback
                )
                # 加载HT电机
                self._load_motors_from_protocol(protocol_instance, "ht")
                return True
            return False
            
        elif protocol_type == "servo":
            if "servo" not in self.protocols:
                self.protocols["servo"] = protocol_instance
                # 根据servo协议的实际结构注册回调
                if hasattr(protocol_instance, 'servo_manager'):
                    callback = getattr(protocol_instance.servo_manager, 'can_frame_callback', lambda x: None)
                else:
                    callback = getattr(protocol_instance, 'can_frame_callback', lambda x: None)
                self.can_dispatcher.register_handler("servo", callback)
                # 加载舵机
                self._load_motors_from_protocol(protocol_instance, "servo")
                return True
            return False
        else:
            print(f"Unknown protocol type: {type(protocol_instance).__name__}")
            return False
    
    def _load_motors_from_protocol(self, protocol_instance, protocol_type):
        """从protocol实例中加载电机到manager"""
        if hasattr(protocol_instance, 'motors') and protocol_instance.motors:
            for motor_id, info in protocol_instance.motor_infos.items():
                self.motor_infos[motor_id] = info
            for motor_id, motor_instance in protocol_instance.motors.items():
                self.motor_cnt += 1
                motor_info =None
                # motor_instance
                # 创建基本的MotorInfo（可能需要根据实际情况调整）
                # motor_info = MotorInfo(
                #     motor_id=motor_id,
                #     motor_type=getattr(MotorType, protocol_type.upper(), MotorType.DAMIAO),
                #     can_id=getattr(motor_instance, 'can_id', motor_id)
                # )
                self.add_motor(motor_id, protocol_type, motor_info)
                print(f"Loaded motor {motor_id} from {protocol_type} protocol")
    
    def _identify_protocol_type(self, protocol_instance):
        """识别协议类型"""
        class_name = type(protocol_instance).__name__
        
        # 通过类名识别
        if "Motor_Control" in class_name or hasattr(protocol_instance, 'addMotor'):
            return "damiao"
        elif "HTMotor" in class_name or hasattr(protocol_instance, 'mit_control'):
            return "ht"
        elif "Servo" in class_name or hasattr(protocol_instance, 'add_servo'):
            return "servo"
        else:
            return "unknown"

    def get_motor_info(self, motor_id):
        return self.motor_infos[motor_id]
    # 向后兼容的方法
    def add_damiao_protocol(self, motor_control_instance):
        """添加达妙电机协议 - 向后兼容"""
        return self.add_protocol(motor_control_instance, 'damiao')

    def add_ht_protocol(self, ht_manager_instance):
        """添加HT电机协议 - 向后兼容"""
        return self.add_protocol(ht_manager_instance, 'ht')

    def add_servo_protocol(self, servo_manager_instance):
        """添加伺服电机协议 - 向后兼容"""
        return self.add_protocol(servo_manager_instance, 'servo')

    def add_motor(
        self, motor_id: int, motor_type: str, motor_info: MotorInfo, **config
    ) -> bool:
        """从protocol中加载电机到manager"""
        if motor_type not in self.protocols:
            print(f"Protocol {motor_type} not available")
            return False

        protocol = self.protocols[motor_type]
        
        # 创建统一电机实例
        unified_motor = UnifiedMotor(motor_id, protocol, motor_info)
        self.motors[motor_id] = unified_motor

        return True

    def get_motor(self, motor_id: int) -> Optional[UnifiedMotor]:
        """获取电机实例"""
        return self.motors.get(motor_id)

    # @pysnooper.snoop()
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
    def control_mit_batch(
        self,
        motor_ids: List[int],
        positions: List[float],
        velocities: List[float],
        kps: List[float],
        kds: List[float],
        torques: List[float],
    ) -> bool:
        """批量MIT控制"""
        success = True

        # 设置所有命令
        for i, motor_id in enumerate(motor_ids):
            if motor_id in self.motors:
                motor = self.motors[motor_id]
                if not motor.set_command(
                    positions[i], velocities[i], kps[i], kds[i], torques[i]
                ):
                    success = False

        # 发送所有命令
        if not self.send_all_commands():
            success = False

        return success
