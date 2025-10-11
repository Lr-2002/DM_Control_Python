import sys
import pysnooper
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ic_arm_control.control.src import usb_class, can_value_type
from enum import IntEnum
import time
import struct
from typing import Optional, List
import threading
import signal
from dataclasses import dataclass
from ic_arm_control.control.motor_info import *
@dataclass
class DmActData:
    motorType: DM_Motor_Type  # 是哪款电机
    mode: Control_Mode        # 电机处于哪种控制模式
    can_id: int
    mst_id: int
    kp: int = 0
    kd: int = 0

class DM_REG(IntEnum):
    UV_Value = 0
    KT_Value = 1
    OT_Value = 2
    OC_Value = 3
    ACC = 4
    DEC = 5
    MAX_SPD = 6
    MST_ID = 7
    ESC_ID = 8
    TIMEOUT = 9
    CTRL_MODE = 10
    Damp = 11
    Inertia = 12
    hw_ver = 13
    sw_ver = 14
    SN = 15
    NPP = 16
    Rs = 17
    LS = 18
    Flux = 19
    Gr = 20
    PMAX = 21
    VMAX = 22
    TMAX = 23
    I_BW = 24
    KP_ASR = 25
    KI_ASR = 26
    KP_APR = 27
    KI_APR = 28
    OV_Value = 29
    GREF = 30
    Deta = 31
    V_BW = 32
    IQ_c1 = 33
    VL_c1 = 34
    can_br = 35
    sub_ver = 36
    u_off = 50
    v_off = 51
    k1 = 52
    k2 = 53
    m_off = 54
    dir = 55
    p_m = 80
    xout = 81

class ValueUnion:
    def __init__(self):
        self.floatValue = 0.0
        self.intValue = 0  # 若以后用到，可扩展

class ValueType:
    def __init__(self):
        self.value = ValueUnion()
        self.isFloat = False

# 电机类
class Motor:
    def __init__(self, motor_type: DM_Motor_Type, ctrl_mode: Control_Mode, can_id: int, master_id: int):
        self.Motor_Type = motor_type
        self.mode = ctrl_mode
        self.Can_id = can_id
        self.Master_id = master_id
        if motor_type.value < len(limit_param):
            self.limit_param = limit_param[motor_type.value]
        else:
            raise ValueError(f"Invalid motor type: {motor_type}")
        self.param_map: dict[int, ValueType] = {}
        self.last_time_ = time.monotonic()
        self.delta_time_= 0
        self.state_q = 0
        self.state_dq = 0 
        self.state_tau = 0

    def updateTimeInterval(self) -> float:
        now = time.monotonic()
        self.delta_time_ = now - self.last_time_
        self.last_time_ = now
        return self.delta_time_  # 单位为秒
    
    def getTimeInterval(self): 
        return self.delta_time_


    def receive_data(self, q: float, dq: float, tau: float):
        self.state_q = q
        self.state_dq = dq
        self.state_tau = tau

    def set_param(self, key: int, value):
        v = ValueType()
        if type(value) is int:
            v.value.uint32Value = value
            v.isFloat = False
        elif type(value) is float:
            v.value.floatValue = value
            v.isFloat = True
        self.param_map[key] = v

    def get_param_as_float(self, key: int) -> float:
        v = self.param_map.get(key)
        if v is not None:
            if v.isFloat:
                return v.value.floatValue
        return 0.0

    def get_param_as_uint32(self, key: int) -> int:
        v = self.param_map.get(key)
        if v is not None:
            if not v.isFloat:
                return v.value.uint32Value
        return 0
    
    def is_have_param(self, key: int) -> bool:
        return key in self.param_map

    def GetMotorType(self):
        return self.Motor_Type
    def GetMotorMode(self):
        return self.mode   

    def get_limit_param(self):
        return self.limit_param  # 获取电机限制参数   

    def GetMasterId(self):
        return self.Master_id  # 获取反馈ID    

    def GetCanId(self):
        return self.Can_id  # 获取电机CAN ID

    def Get_Position(self):
        return self.state_q

    def Get_Velocity(self):
        return self.state_dq

    def Get_tau(self):
        return self.state_tau

    def set_mode(self, value: Control_Mode):
        self.mode = value

class DmMotorManager:
    def __init__(self,usb_hw=None, nom_baud: int=0, dat_baud: int=5000000, sn: str ='null', data_ptr: list=[], use_ht=False):
        assert usb_hw or (nom_baud != 0), 'the usb or nom should be one right '
        self.data_ptr_ = data_ptr
        self.motors: dict[int, Motor] = {}
        self.read_write_save = threading.Event()# 初始为未设置状态
        self.read_write_save.clear()
        
        # 遍历该bus下所有电机
        # for act_data in self.data_ptr_:
        #     motor = Motor(act_data.motorType, act_data.mode, act_data.can_id, act_data.mst_id)
        #     self.addMotor(motor)
        if usb_hw:
            self.usb_hw = usb_hw
        else: 
            self.usb_hw = usb_class(nom_baud, dat_baud,sn)
        time.sleep(0.5)
                # ERR状态映射
        self.err_status = {
            0x0: "失能",
            0x1: "使能",
            0x8: "超压",
            0x9: "欠压",
            0xA: "过电流",
            0xB: "MOS过温",
            0xC: "电机线圈过温",
            0xD: "通讯丢失",
            0xE: "过载"
        }
        # 注意：回调函数将由MotorManager的CANFrameDispatcher统一管理
        # self.usb_hw.setFrameCallback(lambda val: self.canframeCallback(val, None))
        time.sleep(0.2)

        self.enable_all()  # 使能该接口下的所有电机
        print("**********Motor_Control init success**********\n")

    # def __del__(self):
    #       print("Enter ~Motor_Control")
    #       if self.getUSBHw().getDeviceHandle() is not None:
    #           self.disable_all()  # 使能该接口下的所有电机
    #           self.usb_hw.close()
    def add_motor(self, motor_id):
        pass

    def add_ht_motor(self, motor_id):

        assert hasattr(self, 'ht_manager'), "You should import ht motor first "
        self.ht_manager.add_motor(motor_id)
    def __enter__(self):
        # 必须返回 self 才能赋值给 usb2
        print("__enter__  Motor_Control")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("__exit__  Motor_Control")
        if self.getUSBHw().getDeviceHandle() is not None:
            self.disable_all()  # 失能该接口下的所有电机
            self.usb_hw.close()

    def close(self):
        print("close  Motor_Control")
        if self.getUSBHw().getDeviceHandle() is not None:
            self.disable_all()  # 使能该接口下的所有电机
            self.usb_hw.close()

    @staticmethod
    def is_in_ranges(number: int) -> bool:
        return (7 <= number <= 10) or (13 <= number <= 16) or (35 <= number <= 36)

    @staticmethod
    def float_to_uint32(value: float) -> int:
        return int(value)

    @staticmethod
    def uint32_to_float(value: int) -> float:
        return float(value)

    @staticmethod
    def uint8_to_float(data: List[int]) -> float:
        """
        参数 `data` 应为长度为 4 的整数列表（0-255）。
        例如：data = [0xDB, 0x0F, 0x49, 0x40] 表示 float 3.1415927
        """
        if len(data) != 4:
            raise ValueError("data must be a list of 4 bytes")
        byte_data = bytes(data)
        return struct.unpack('<f', byte_data)[0]
    
    def getMotor(self, id: int) -> Optional[Motor]:
        motor = self.motors.get(id)
        if motor is not None:
            return motor
        else:
            print(f"[Error] In getMotor, no motor with id {id} is registered.")
            return None

    def getUSBHw(self) -> Optional[usb_class]:
        return self.usb_hw
    
    def addMotor(self, DM_Motor:Motor):
        self.motors[DM_Motor.GetCanId()] = DM_Motor
        self.motors[DM_Motor.GetMasterId()] = DM_Motor

    def enable_all(self):
        # data=[4,0,0,0]
        # for motor in self.motors.values():
        #         for _ in range(5):
        #             self.write_motor_param(motor,0x23,data)
        #             time.sleep(0.002)  # 2000 microseconds = 2 milliseconds
        #             self.write_motor_param(motor,0x23,data)
        #             time.sleep(0.002)
        # for motor in self.motors.values():
        #     self.read_motor_param(motor,10)
        # for motor in self.motors.values():
        #     parm=motor.get_param_as_uint32(10)
        #     print(f"id: {motor.GetCanId()} mode is {parm}", file=sys.stderr)

        # for motor in self.motors.values():
        #     self.switchControlMode(motor,Control_Mode_Code.MIT)
        # for motor in self.motors.values():
        #     self.read_motor_param(motor,10)
        # for motor in self.motors.values():
        #     parm=motor.get_param_as_uint32(10)
        #     print(f"id: {motor.GetCanId()} mode is {parm}", file=sys.stderr)

        for motor in self.motors.values():
            self.switchControlMode(motor,Control_Mode_Code.MIT)
        for motor in self.motors.values():
            self.read_motor_param(motor,10)
        for motor in self.motors.values():
            parm=motor.get_param_as_uint32(10)
            print(f"id: {motor.GetCanId()} mode is {parm}", file=sys.stderr)

        for motor in self.motors.values():
            for _ in range(5):
                self.control_cmd(motor.GetCanId() + motor.GetMotorMode(), 0xFC)
                time.sleep(0.002)  

    def disable_all(self):
        for key, motor in self.motors.items():  # motors 是一个字典
            for _ in range(5):
                self.control_cmd(motor.GetCanId() + motor.GetMotorMode(), 0xFD)
                time.sleep(0.002)  # 2000 微秒 = 2 毫秒

    def read_motor_param(self, DM_Motor, RID: int) -> float:
        self.read_write_save.set()  # 表示正在进行参数读取

        can_id = DM_Motor.GetCanId()
        id_low = can_id & 0xFF
        id_high = (can_id >> 8) & 0xFF

        mydata = bytes([id_low, id_high, 0x33, RID, 0x00, 0x00, 0x00, 0x00])
        self.usb_hw.fdcanFrameSend(mydata, 0x7FF)
        time.sleep(0.002)
        return 0.0

    def save_motor_param(self, DM_Motor):
        id = DM_Motor.GetCanId()
        mode = DM_Motor.GetMotorMode()
        self.control_cmd(id + mode, 0xFD)  # 失能
        time.sleep(0.01)  # 10000 微秒 = 10 毫秒

        self.read_write_save.set()  # 标志位

        id_low = id & 0xFF
        id_high = (id >> 8) & 0xFF

        mydata = bytes([id_low, id_high, 0xAA, 0x01, 0x00, 0x00, 0x00, 0x00])
        self.usb_hw.fdcanFrameSend(mydata, 0x7FF)

        time.sleep(0.1)  # 100000 微秒 = 100 毫秒

    def refresh_motor_status(self, motor):
        id_low = motor.GetCanId() & 0xFF
        id_high = (motor.GetCanId() >> 8) & 0xFF

        mydata = bytes([id_low, id_high, 0xCC, 0x00])
        self.usb_hw.fdcanFrameSend(mydata, 0x7FF)

    def control_cmd(self, id: int, cmd: int):
        mydata = bytes([0xFF] * 7 + [cmd])
        self.usb_hw.fdcanFrameSend(mydata, id)

    def write_motor_param(self, DM_Motor, RID: int, data: list):
        self.read_write_save.set()

        id = DM_Motor.GetCanId()
        id_low = id & 0xFF
        id_high = (id >> 8) & 0xFF

        mydata = bytes([id_low, id_high, 0x55, RID, data[0], data[1], data[2], data[3]])
        self.usb_hw.fdcanFrameSend(mydata, 0x7FF)
        time.sleep(0.002)

    def set_zero_position(self, DM_Motor):
        self.control_cmd(DM_Motor.GetCanId() + DM_Motor.GetMotorMode(), 0xFE)
        time.sleep(0.002)

    # @pysnooper.snoop()
    def control_mit(self, DM_Motor, kp: float, kd: float, q: float, dq: float, tau: float):
        float_to_uint = lambda x, xmin, xmax, bits: int((x - xmin) / (xmax - xmin) * ((1 << bits) - 1))

        id = DM_Motor.GetCanId()
        if id not in self.motors:
            print(f"[Error] In control_mit, no motor with id {DM_Motor.GetCanId()} is registered.")
            sys.exit(-1)

        m = self.motors[id]
        kp_uint = float_to_uint(kp, 0, 500, 12)
        kd_uint = float_to_uint(kd, 0, 5, 12)

        limit_param_cmd = m.get_limit_param()

        q_uint = float_to_uint(q, -limit_param_cmd[0], limit_param_cmd[0], 16)
        dq_uint = float_to_uint(dq, -limit_param_cmd[1], limit_param_cmd[1], 12)
        tau_uint = float_to_uint(tau, -limit_param_cmd[2], limit_param_cmd[2], 12)

        can_id = id + Control_Mode.MIT_MODE

        data = [0] * 8
        data[0] = (q_uint >> 8) & 0xff
        data[1] = q_uint & 0xff
        data[2] = dq_uint >> 4
        data[3] = ((dq_uint & 0xf) << 4) | ((kp_uint >> 8) & 0xf)
        data[4] = kp_uint & 0xff
        data[5] = kd_uint >> 4
        data[6] = ((kd_uint & 0xf) << 4) | ((tau_uint >> 8) & 0xf)
        data[7] = tau_uint & 0xff

        self.usb_hw.fdcanFrameSend(data, can_id)

    def control_pos_vel(self, DM_Motor, pos: float, vel: float):
        id_ = DM_Motor.GetCanId()
        if id_ not in self.motors:
            print(f"[Error] In control_pos_vel, no motor with id {id_} is registered.", file=sys.stderr)
            sys.exit(-1)  # 终止程序，返回非 0 表示错误

        can_id = id_ + Control_Mode.POS_VEL_MODE  # 需要保证 self.POS_VEL_MODE 定义了

        # 把 float 按 4 字节小端序转换为 bytes
        pbuf = struct.pack('<f', pos)
        vbuf = struct.pack('<f', vel)

        # 组合数据成长度8的list
        mydata = list(pbuf + vbuf)  # bytes 拼接后转成列表

        self.usb_hw.fdcanFrameSend(mydata, can_id)

    def control_vel(self, DM_Motor, vel: float):
        id_ = DM_Motor.GetCanId()
        if id_ not in self.motors:
            print(f"[Error] In control_vel, no motor with id {id_} is registered.", file=sys.stderr)
            sys.exit(-1)

        can_id = id_ + Control_Mode.VEL_MODE

        # 把 float 按 4 字节小端序转换成 bytes
        vbuf = struct.pack('<f', vel)

        # 只取4个字节，转换成列表
        mydata = list(vbuf)

        self.usb_hw.fdcanFrameSend(mydata, can_id)

    def receive_param(self, data: bytes):
        canID = (data[1] << 8) | data[0]
        RID = data[3]
        if canID not in self.motors:
            print(f"[Error] In receive_param, no motor with id {canID} is registered.", file=sys.stderr)
            sys.exit(-1)

        if self.is_in_ranges(RID):
            data_uint32 = (data[7] << 24) | (data[6] << 16) | (data[5] << 8) | data[4]
            self.motors[canID].set_param(RID, data_uint32)

            if RID == 10:
                if data_uint32 == 1:
                    self.motors[canID].set_mode(Control_Mode.MIT_MODE)
                elif data_uint32 == 2:
                    self.motors[canID].set_mode(Control_Mode.POS_VEL_MODE)
                elif data_uint32 == 3:
                    self.motors[canID].set_mode(Control_Mode.VEL_MODE)
                elif data_uint32 == 4:
                    self.motors[canID].set_mode(Control_Mode.POS_FORCE_MODE)
        else:
            data_float = self.uint8_to_float(data[4:8])  # 取4个字节，转float
            self.motors[canID].set_param(RID, data_float)
    
    def switchControlMode(self, DM_Motor, mode:Control_Mode_Code):
        write_data = bytes([mode, 0x00, 0x00, 0x00])
        RID = 10
        self.write_motor_param(DM_Motor, RID, write_data)

        can_id = DM_Motor.GetCanId()
        if can_id not in self.motors:
            print(f"[Error] In switchControlMode, no motor with id {can_id} is registered.", file=sys.stderr)
            sys.exit(-1)
            return False

        return True

    def change_motor_param(self, DM_Motor, RID, data):
        if self.is_in_ranges(RID):
            # 传入的应转换为整型表示
            data_uint32 = self.float_to_uint32(data)
            data_bytes = data_uint32.to_bytes(4, byteorder='little')
            self.write_motor_param(DM_Motor, RID, data_bytes)
        else:
            data_bytes = struct.pack('f', data)
            self.write_motor_param(DM_Motor, RID, data_bytes)

        can_id = DM_Motor.GetCanId()
        if can_id not in self.motors:
            print(f"[Error] In change_motor_param, no motor with id {can_id} is registered.", file=sys.stderr)
            sys.exit(-1)
            return False

        return True

    def changeMotorLimit(self, DM_Motor, P_MAX, Q_MAX, T_MAX):
        motor_type = DM_Motor.GetMotorType()
        self.limit_param[motor_type] = [P_MAX, Q_MAX, T_MAX]

    """ 
    can_value_type在usb_class类里面是这样定义的
        class can_head_type:
            def __init__(self):
                self.id = 0
                self.time_stamp = 0
                self.reserve = [0, 0, 0]
                self.fram_type = 0
                self.can_type = 0
                self.id_type = 0
                self.dir = 0
                self.dlc = 0  

        class can_value_type:
            def __init__(self):
                self.head = can_head_type()
                self.data = [0] * 64
    """
    def canframeCallback(self, value:can_value_type):
        uint_to_float = lambda x, xmin, xmax, bits: ((float(x) / ((1 << bits) - 1)) * (xmax - xmin)) + xmin

        canID = value.head.id
        # HT电机的帧将由CANFrameDispatcher处理，这里只处理达妙电机的帧
        
        # 解析D[0]: ID|ERR<<4
        # 低4位是ID，高4位是ERR
        motor_id = value.data[0] & 0x0F  # 低4位
        err_code = (value.data[0] >> 4) & 0x0F  # 高4位
        

        
        if err_code != 0 and err_code != 1:
            err_msg = self.err_status.get(err_code, f"未知状态(0x{err_code:X})")
            print(f'[ERROR] canID {canID} (motor_id={motor_id}): ERR={err_code:X} ({err_msg}), D[0]=0x{value.data[0]:02X}')
            raise RuntimeError(f'CAN [{motor_id}] Error: {err_code:X} ({err_msg})')
        
        if self.read_write_save.is_set() and canID in self.motors:
            if value.data[2] in (0x33, 0x55, 0xAA):
                if value.data[2] in (0x33, 0x55):
                    #print(value.data[4])
                    self.receive_param(bytes(value.data))
                    self.read_write_save.clear()
                self.read_write_save.clear()
        else:
            q_uint = (value.data[1] << 8) | value.data[2]
            dq_uint = (value.data[3] << 4) | (value.data[4] >> 4)
            tau_uint = ((value.data[4] & 0xf) << 8) | value.data[5]

            if canID not in self.motors:
                return

            m = self.motors[canID-16]
            limit_param_receive = m.get_limit_param()
            receive_q = uint_to_float(q_uint, -limit_param_receive[0], limit_param_receive[0], 16)
            receive_dq = uint_to_float(dq_uint, -limit_param_receive[1], limit_param_receive[1], 12)
            receive_tau = uint_to_float(tau_uint, -limit_param_receive[2], limit_param_receive[2], 12)

            m.receive_data(receive_q, receive_dq, receive_tau)

            interval=m.updateTimeInterval()
            
            #print(f"motor id is: {canID}: {interval}", file=sys.stderr)
            
            
           
running =threading.Event()
running.set()  # 初始为 True

# Ctrl+C 信号处理函数
def signal_handler(signum, frame):
    running.clear()
    sys.stderr.write(f"\nInterrupt signal ({signum}) received.\n")
    sys.stderr.flush()



running =threading.Event()
running.set()  # 初始为 True

# Ctrl+C 信号处理函数
def signal_handler(signum, frame):
    running.clear()
    sys.stderr.write(f"\nInterrupt signal ({signum}) received.\n")
    sys.stderr.flush()

# 注册 SIGINT 处理（即 Ctrl+C）
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    try:
        init_data1= []
        canid1=0x01
        mstid1=0x11
        canid2=0x02
        mstid2=0x12
        canid3=0x03
        mstid3=0x13
        canid4=0x04
        mstid4=0x14
        canid5=0x05
        mstid5=0x15
        canid6=0x06
        mstid6=0x16
        canid7=0x8094
        mstid7=0x700

        id_list = [canid1, canid2, canid3, canid4, canid5, canid6]
        init_data1.append(DmActData(
                    motorType=DM_Motor_Type.DM10010L,  # 或者具体类型，如 DM_Motor_Type.DM4310
                    mode=Control_Mode.MIT_MODE,        # 如 Control_Mode.MIT_MODE
                    can_id=canid1,
                    mst_id=mstid1))
        init_data1.append(DmActData(
                    motorType=DM_Motor_Type.DM4340,  # 或者具体类型，如 DM_Motor_Type.DM4310
                    mode=Control_Mode.MIT_MODE,        # 如 Control_Mode.MIT_MODE
                    can_id=canid2,
                    mst_id=mstid2))
        init_data1.append(DmActData(
                    motorType=DM_Motor_Type.DM6248,  # 或者具体类型，如 DM_Motor_Type.DM4310
                    mode=Control_Mode.MIT_MODE,        # 如 Control_Mode.MIT_MODE
                    can_id=canid3,
                    mst_id=mstid3))
        init_data1.append(DmActData(
                    motorType=DM_Motor_Type.DM4340,  # 或者具体类型，如 DM_Motor_Type.DM4310
                    mode=Control_Mode.MIT_MODE,        # 如 Control_Mode.MIT_MODE
                    can_id=canid4,
                    mst_id=mstid4))
        init_data1.append(DmActData(
                    motorType=DM_Motor_Type.DM4340,  # 或者具体类型，如 DM_Motor_Type.DM4310
                    mode=Control_Mode.MIT_MODE,        # 如 Control_Mode.MIT_MODE
                    can_id=canid5,
                    mst_id=mstid5))
        init_data1.append(DmActData(
                    motorType=DM_Motor_Type.DM4340,  # 或者具体类型，如 DM_Motor_Type.DM4310
                    mode=Control_Mode.MIT_MODE,        # 如 Control_Mode.MIT_MODE
                    can_id=canid6,
                    mst_id=mstid6))
        # init_data1.append(DmActData(
        #             motorType=DM_Motor_Type.HT4438,
        #             mode=Control_Mode.MIT_MODE,
        #             can_id=canid7,
        #             mst_id=mstid7))
        #with Motor_Control(1000000, 5000000,"14AA044B241402B10DDBDAFE448040BB",init_data1) as control\
        #       ,Motor_Control(1000000, 5000000, "AA96DF2EC013B46B1BE4613798544085", init_data2) as control2:
        with Motor_Control(1000000, 5000000,"F561E08C892274DB09496BCC1102DBC5",init_data1, use_ht=True) as control:
            control.add_ht_motor(7)
            control.add_ht_motor(8)
        #control=Motor_Control(1000000, 5000000,"14AA044B241402B10DDBDAFE448040BB",init_data1) 
            while running.is_set():
                    desired_duration = 0.001  # 秒
                    current_time = time.perf_counter()
                    # kp: float, kd: float, q: float, dq: float, tau: float)
                    # control.control_mit(control.getMotor(canid1), 0.0, 0.0, 0.0, 0.0, 0.0)
                    # control.control_mit(control.getMotor(canid2), 0.0, 0.0, 0.0, 0.0, 0.0)
                    # control.control_mit(control.getMotor(canid3), 0.0, 0.0, 0.0, 0.0, 0.0)
                    # control.control_mit(control.getMotor(canid4), 0.0, 0.0, 0.0, 0.0, 0.0)
                    # control.control_mit(control.getMotor(canid5), 0.0, 0.0, 0.0, 0.0, 0.0)
                    #control.control_mit(control.getMotor(canid6), 0.0, 0.0, 0.0, 0.0, 0.0)
                    # control.control_mit(control.getMotor(canid7), 0.0, 0.0, 0.0, 0.0, 0.0)
                    # control.control_mit(control.getMotor(canid8), 0.0, 0.0, 0.0, 0.0, 0.0)
                    # control.control_mit(control.getMotor(canid9), 0.0, 0.0, 0.0, 0.0, 0.0)
                    
                    # control.control_vel(control.getMotor(canid1), 2.0)
                    for id in id_list: 
                        control.refresh_motor_status(control.getMotor(id))
                        pos = control.getMotor(id).Get_Position()
                        vel = control.getMotor(id).Get_Velocity()
                        tau = control.getMotor(id).Get_tau()
                        interval = control.getMotor(id).getTimeInterval()

                        print(f"canid is: {id} pos: {pos} vel: {vel} effort: {tau} time(s): {interval}", file=sys.stderr)
                    control.ht_manager.refresh_motor_status()
                    for id, motor in control.ht_manager.motors.items():
                        pos = control.ht_manager.get_motor(id).Get_Position()
                        vel = control.ht_manager.get_motor(id).Get_Velocity()
                        tau = control.ht_manager.get_motor(id).Get_tau()

                        print(f"canid is: {id} pos: {pos} vel: {vel} effort: {tau}", file=sys.stderr)
                    #control2.control_vel(control2.getMotor(canid2), -3.0)
                    #control.enable_all()
                    sleep_till = current_time + desired_duration
                    now = time.perf_counter()
                    if sleep_till > now:
                        time.sleep(sleep_till - now)
                    
            print("The program exited safely.") 
    except Exception as e:
        print(f"Error: hardware interface exception: {e}", file=sys.stderr)
    finally:
        #control.close()
        #control2.close()
        pass
