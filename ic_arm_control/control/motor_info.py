from enum import IntEnum

from dataclasses import dataclass
from typing import Optional, List
class DM_Motor_Type(IntEnum):
    DM3507 = 0
    DM4310 = 1
    DM4310_48V = 2
    DM4340 = 3
    DM4340_48V = 4
    DM6006 = 5
    DM6248 = 6
    DM8006 = 7
    DM8009 = 8
    DM10010L = 9
    DM10010 = 10
    DMH3510 = 11
    DMH6215 = 12
    DMS3519 = 13
    DMG6220 = 14
    Num_Of_Motor = 15
    HT4438 = 16

class Control_Mode(IntEnum):
    MIT_MODE = 0x000
    POS_VEL_MODE = 0x100
    VEL_MODE = 0x200
    POS_FORCE_MODE = 0x300

class Control_Mode_Code(IntEnum):
    MIT = 1
    POS_VEL = 2
    VEL = 3
    POS_FORCE = 4

limit_param = [
    [12.566, 50, 5],   # DM3507         check 
    [12.5, 30, 10],   # DM4310          check
    [12.5, 50, 10],   # DM4310_48V
    [12.5, 10, 28],   # DM4340          check
    [12.5, 20, 28],   # DM4340_48V      check
    [12.5, 45, 12],   # DM6006          check
    [12.566, 20, 120],   # DM6248       check
    [12.5, 45, 20],   # DM8006          check
    [12.5, 45, 54],   # DM8009          check
    [12.5, 25, 200],  # DM10010L        check
    [12.5, 20, 200],  # DM10010         check
    [12.5, 280, 1],   # DMH3510         check
    [12.5, 45, 10],   # DMH6215
    [12.5, 2000, 2],    # DMS3519         check
    [12.5, 45, 10]    # DMG6220         check
]



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
    motor_index: Optional[DM_Motor_Type] = None
    can_id: int =0
    master_id: Optional[int] = None
    
    # 控制参数
    kp: float = 0.0
    kd: float = 0.0
    torque_offset: float = 0.0
    
    # 限制参数 [position_limit, velocity_limit, torque_limit]
    limits: List[float] = None
    
    name: str = ""
    def __post_init__(self):
        if self.limits is None :
            if self.motor_index:
                self.limits = limit_param[self.motor_index]

            else:
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



