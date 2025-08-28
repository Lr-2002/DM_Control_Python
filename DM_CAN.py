from time import sleep
import numpy as np
from enum import IntEnum
from struct import unpack
from struct import pack
from typing import List, Optional

class Motor:
    def __init__(self, MotorType, SlaveID, MasterID):
        """
        define Motor object 定义电机对象
        :param MotorType: Motor type 电机类型
        :param SlaveID: CANID 电机ID
        :param MasterID: MasterID 主机ID 建议不要设为0
        """
        self.Pd = float(0)
        self.Vd = float(0)
        self.state_q = float(0)
        self.state_dq = float(0)
        self.state_tau = float(0)
        self.SlaveID = SlaveID
        self.MasterID = MasterID
        self.MotorType = MotorType
        self.isEnable = False
        self.NowControlMode = Control_Type.MIT
        self.temp_param_dict = {}

    def recv_data(self, q: float, dq: float, tau: float):
        self.state_q = q
        self.state_dq = dq
        self.state_tau = tau

    def getPosition(self):
        """
        get the position of the motor 获取电机位置
        :return: the position of the motor 电机位置
        """
        return self.state_q

    def getVelocity(self):
        """
        get the velocity of the motor 获取电机速度
        :return: the velocity of the motor 电机速度
        """
        return self.state_dq

    def getTorque(self):
        """
        get the torque of the motor 获取电机力矩
        :return: the torque of the motor 电机力矩
        """
        return self.state_tau

    def getParam(self, RID):
        """
        get the parameter of the motor 获取电机内部的参数，需要提前读取
        :param RID: DM_variable 电机参数
        :return: the parameter of the motor 电机参数
        """
        if RID in self.temp_param_dict:
            return self.temp_param_dict[RID]
        else:
            return None


class MotorControl:
    send_data_frame = np.array(
        [0x55, 0xAA, 0x1e, 0x03, 0x01, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x00, 0, 0, 0, 0, 0x00, 0x08, 0x00,
         0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0x00], np.uint8)
    #                4310           4310_48        4340           4340_48
    Limit_Param = [[12.5, 30, 10], [12.5, 50, 10], [12.5, 8, 28], [12.5, 10, 28],
                   # 6006           8006           8009            10010L         10010
                   [12.5, 45, 20], [12.5, 45, 40], [12.5, 45, 54], [12.5, 25, 200], [12.5, 20, 200],
                   # H3510            DMG62150      DMH6220             DM6248
                   [12.5 , 280 , 1],[12.5 , 45 , 10],[12.5 , 45 , 10], [12.56, 20, 120]]

    def __init__(self, serial_device):
        """
        define MotorControl object 定义电机控制对象
        :param serial_device: serial object 串口对象
        """
        self.serial_ = serial_device
        self.motors_map = dict()
        self.data_save = bytes()  # save data
        if self.serial_.is_open:  # open the serial port
            print("Serial port is open")
            serial_device.close()
        self.serial_.open()

    def controlMIT(self, DM_Motor, kp: float, kd: float, q: float, dq: float, tau: float):
        """
        MIT Control Mode Function 达妙电机MIT控制模式函数
        :param DM_Motor: Motor object 电机对象
        :param kp: kp
        :param kd:  kd
        :param q:  position  期望位置
        :param dq:  velocity  期望速度
        :param tau: torque  期望力矩
        :return: None
        """
        if DM_Motor.SlaveID not in self.motors_map:
            print("controlMIT ERROR : Motor ID not found")
            return
        # print('recv control data like this ', DM_Motor, kp, kd, q, dq, tau)
        kp_uint = float_to_uint(kp, 0, 500, 12)
        kd_uint = float_to_uint(kd, 0, 5, 12)
        MotorType = DM_Motor.MotorType
        Q_MAX = self.Limit_Param[MotorType][0]
        DQ_MAX = self.Limit_Param[MotorType][1]
        TAU_MAX = self.Limit_Param[MotorType][2]
        q_uint = float_to_uint(q, -Q_MAX, Q_MAX, 16)
        dq_uint = float_to_uint(dq, -DQ_MAX, DQ_MAX, 12)
        tau_uint = float_to_uint(tau, -TAU_MAX, TAU_MAX, 12)
        data_buf = np.array([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], np.uint8)
        data_buf[0] = (q_uint >> 8) & 0xff
        data_buf[1] = q_uint & 0xff
        data_buf[2] = dq_uint >> 4
        data_buf[3] = ((dq_uint & 0xf) << 4) | ((kp_uint >> 8) & 0xf)
        data_buf[4] = kp_uint & 0xff
        data_buf[5] = kd_uint >> 4
        data_buf[6] = ((kd_uint & 0xf) << 4) | ((tau_uint >> 8) & 0xf)
        data_buf[7] = tau_uint & 0xff
        self.__send_data(DM_Motor.SlaveID, data_buf)
        self.recv()  # receive the data from serial port

    def control_delay(self, DM_Motor, kp: float, kd: float, q: float, dq: float, tau: float, delay: float):
        """
        MIT Control Mode Function with delay 达妙电机MIT控制模式函数带延迟
        :param DM_Motor: Motor object 电机对象
        :param kp: kp
        :param kd: kd
        :param q:  position  期望位置
        :param dq:  velocity  期望速度
        :param tau: torque  期望力矩
        :param delay: delay time 延迟时间 单位秒
        """
        self.controlMIT(DM_Motor, kp, kd, q, dq, tau)
        sleep(delay)

    def control_Pos_Vel(self, Motor, P_desired: float, V_desired: float):
        """
        control the motor in position and velocity control mode 电机位置速度控制模式
        :param Motor: Motor object 电机对象
        :param P_desired: desired position 期望位置
        :param V_desired: desired velocity 期望速度
        :return: None
        """
        if Motor.SlaveID not in self.motors_map:
            print("Control Pos_Vel Error : Motor ID not found")
            return
        motorid = 0x100 + Motor.SlaveID
        data_buf = np.array([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], np.uint8)
        P_desired_uint8s = float_to_uint8s(P_desired)
        V_desired_uint8s = float_to_uint8s(V_desired)
        data_buf[0:4] = P_desired_uint8s
        data_buf[4:8] = V_desired_uint8s
        self.__send_data(motorid, data_buf)
        # time.sleep(0.001)
        self.recv()  # receive the data from serial port

    def control_Vel(self, Motor, Vel_desired):
        """
        control the motor in velocity control mode 电机速度控制模式
        :param Motor: Motor object 电机对象
        :param Vel_desired: desired velocity 期望速度
        """
        if Motor.SlaveID not in self.motors_map:
            print("control_VEL ERROR : Motor ID not found")
            return
        motorid = 0x200 + Motor.SlaveID
        data_buf = np.array([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], np.uint8)
        Vel_desired_uint8s = float_to_uint8s(Vel_desired)
        data_buf[0:4] = Vel_desired_uint8s
        self.__send_data(motorid, data_buf)
        self.recv()  # receive the data from serial port

    def control_pos_force(self, Motor, Pos_des: float, Vel_des, i_des):
        """
        control the motor in EMIT control mode 电机力位混合模式
        :param Pos_des: desired position rad  期望位置 单位为rad
        :param Vel_des: desired velocity rad/s  期望速度 为放大100倍
        :param i_des: desired current rang 0-10000 期望电流标幺值放大10000倍
        电流标幺值：实际电流值除以最大电流值，最大电流见上电打印
        """
        if Motor.SlaveID not in self.motors_map:
            print("control_pos_vel ERROR : Motor ID not found")
            return
        motorid = 0x300 + Motor.SlaveID
        data_buf = np.array([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], np.uint8)
        Pos_desired_uint8s = float_to_uint8s(Pos_des)
        data_buf[0:4] = Pos_desired_uint8s
        Vel_uint = np.uint16(Vel_des)
        ides_uint = np.uint16(i_des)
        data_buf[4] = Vel_uint & 0xff
        data_buf[5] = Vel_uint >> 8
        data_buf[6] = ides_uint & 0xff
        data_buf[7] = ides_uint >> 8
        self.__send_data(motorid, data_buf)
        self.recv()  # receive the data from serial port

    def enable(self, Motor):
        """
        enable motor 使能电机
        最好在上电后几秒后再使能电机
        :param Motor: Motor object 电机对象
        """
        self.__control_cmd(Motor, np.uint8(0xFC))
        sleep(0.1)
        print('enable motor')
        Motor.isEnable=True
        self.recv()  # receive the data from serial port

    def enable_old(self, Motor ,ControlMode):
        """
        enable motor old firmware 使能电机旧版本固件，这个是为了旧版本电机固件的兼容性
        可恶的旧版本固件使能需要加上偏移量
        最好在上电后几秒后再使能电机
        :param Motor: Motor object 电机对象
        """
        data_buf = np.array([0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfc], np.uint8)
        enable_id = ((int(ControlMode)-1) << 2) + Motor.SlaveID
        self.__send_data(enable_id, data_buf)
        sleep(0.1)
        self.recv()  # receive the data from serial port

    def disable(self, Motor):
        """
        disable motor 失能电机
        :param Motor: Motor object 电机对象
        """
        self.__control_cmd(Motor, np.uint8(0xFD))
        sleep(0.1)
        Motor.isEnable=False
        self.recv()  # receive the data from serial port

    def set_zero_position(self, Motor):
        """
        set the zero position of the motor 设置电机0位
        :param Motor: Motor object 电机对象
        """
        self.__control_cmd(Motor, np.uint8(0xFE))
        sleep(0.1)
        self.recv()  # receive the data from serial port
    def recv_raw(self):
        ra = self.serial_.read_all()
        print(len(ra))
        print(ra)
        data_recv = b''.join([self.data_save, ra])
        packets = self.__extract_packets(data_recv)
        return packets


    def recv(self):
        # 把上次没有解析完的剩下的也放进来
        data_recv = b''.join([self.data_save, self.serial_.read_all()])
        packets = self.__extract_packets(data_recv)
        for packet in packets:
            data = packet[7:15]
            CANID = (packet[6] << 24) | (packet[5] << 16) | (packet[4] << 8) | packet[3]
            CMD = packet[1]
            self.__process_packet(data, CANID, CMD)

    def recv_set_param_data(self):
        data_recv = self.serial_.read_all()
        packets = self.__extract_packets(data_recv)
        for packet in packets:
            data = packet[7:15]
            CANID = (packet[6] << 24) | (packet[5] << 16) | (packet[4] << 8) | packet[3]
            CMD = packet[1]
            self.__process_set_param_packet(data, CANID, CMD)

    def __process_packet(self, data, CANID, CMD):
        if CMD == 0x11:
            if CANID != 0x00:
                if CANID in self.motors_map:
                    q_uint = np.uint16((np.uint16(data[1]) << 8) | data[2])
                    dq_uint = np.uint16((np.uint16(data[3]) << 4) | (data[4] >> 4))
                    tau_uint = np.uint16(((data[4] & 0xf) << 8) | data[5])
                    MotorType_recv = self.motors_map[CANID].MotorType
                    Q_MAX = self.Limit_Param[MotorType_recv][0]
                    DQ_MAX = self.Limit_Param[MotorType_recv][1]
                    TAU_MAX = self.Limit_Param[MotorType_recv][2]
                    recv_q = uint_to_float(q_uint, -Q_MAX, Q_MAX, 16)
                    recv_dq = uint_to_float(dq_uint, -DQ_MAX, DQ_MAX, 12)
                    recv_tau = uint_to_float(tau_uint, -TAU_MAX, TAU_MAX, 12)
                    self.motors_map[CANID].recv_data(recv_q, recv_dq, recv_tau)
            else:
                MasterID=data[0] & 0x0f
                if MasterID in self.motors_map:
                    q_uint = np.uint16((np.uint16(data[1]) << 8) | data[2])
                    dq_uint = np.uint16((np.uint16(data[3]) << 4) | (data[4] >> 4))
                    tau_uint = np.uint16(((data[4] & 0xf) << 8) | data[5])
                    MotorType_recv = self.motors_map[MasterID].MotorType
                    Q_MAX = self.Limit_Param[MotorType_recv][0]
                    DQ_MAX = self.Limit_Param[MotorType_recv][1]
                    TAU_MAX = self.Limit_Param[MotorType_recv][2]
                    recv_q = uint_to_float(q_uint, -Q_MAX, Q_MAX, 16)
                    recv_dq = uint_to_float(dq_uint, -DQ_MAX, DQ_MAX, 12)
                    recv_tau = uint_to_float(tau_uint, -TAU_MAX, TAU_MAX, 12)
                    self.motors_map[MasterID].recv_data(recv_q, recv_dq, recv_tau)


    def __process_set_param_packet(self, data, CANID, CMD):
        if CMD == 0x11 and (data[2] == 0x33 or data[2] == 0x55):
            masterid=CANID
            slaveId = ((data[1] << 8) | data[0])
            if CANID==0x00:  #防止有人把MasterID设为0稳一手
                masterid=slaveId

            if masterid not in self.motors_map:
                if slaveId not in self.motors_map:
                    return
                else:
                    masterid=slaveId

            RID = data[3]
            # 读取参数得到的数据
            if is_in_ranges(RID):
                #uint32类型
                num = uint8s_to_uint32(data[4], data[5], data[6], data[7])
                self.motors_map[masterid].temp_param_dict[RID] = num

            else:
                #float类型
                num = uint8s_to_float(data[4], data[5], data[6], data[7])
                self.motors_map[masterid].temp_param_dict[RID] = num


    def addMotor(self, Motor):
        """
        add motor to the motor control object 添加电机到电机控制对象
        :param Motor: Motor object 电机对象
        """
        self.motors_map[Motor.SlaveID] = Motor
        if Motor.MasterID != 0:
            self.motors_map[Motor.MasterID] = Motor
        return True

    def __control_cmd(self, Motor, cmd: np.uint8):
        data_buf = np.array([0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, cmd], np.uint8)
        self.__send_data(Motor.SlaveID, data_buf)
    def send_data(self, motor_id, data):
       
        self.__send_data(motor_id, data)
    def __send_data(self, motor_id, data):
        """
        send data to the motor 发送数据到电机
        :param motor_id:
        :param data:
        :return:
        """
        self.send_data_frame[13] = motor_id & 0xff
        self.send_data_frame[14] = (motor_id >> 8)& 0xff  #id high 8 bits
        self.send_data_frame[21:29] = data
        self.serial_.write(bytes(self.send_data_frame.T))

    def __read_RID_param(self, Motor, RID):
        can_id_l = Motor.SlaveID & 0xff #id low 8 bits
        can_id_h = (Motor.SlaveID >> 8)& 0xff  #id high 8 bits
        data_buf = np.array([np.uint8(can_id_l), np.uint8(can_id_h), 0x33, np.uint8(RID), 0x00, 0x00, 0x00, 0x00], np.uint8)
        self.__send_data(0x7FF, data_buf)

    def __write_motor_param(self, Motor, RID, data):
        can_id_l = Motor.SlaveID & 0xff #id low 8 bits
        can_id_h = (Motor.SlaveID >> 8)& 0xff  #id high 8 bits
        data_buf = np.array([np.uint8(can_id_l), np.uint8(can_id_h), 0x55, np.uint8(RID), 0x00, 0x00, 0x00, 0x00], np.uint8)
        if not is_in_ranges(RID):
            # data is float
            data_buf[4:8] = float_to_uint8s(data)
        else:
            # data is int
            data_buf[4:8] = data_to_uint8s(int(data))
        self.__send_data(0x7FF, data_buf)

    def switchControlMode(self, Motor, ControlMode):
        """
        switch the control mode of the motor 切换电机控制模式
        :param Motor: Motor object 电机对象
        :param ControlMode: Control_Type 电机控制模式 example:MIT:Control_Type.MIT MIT模式
        """
        max_retries = 20
        retry_interval = 0.1  #retry times
        RID = 10
        self.__write_motor_param(Motor, RID, np.uint8(ControlMode))
        for _ in range(max_retries):
            sleep(retry_interval)
            self.recv_set_param_data()
            if Motor.SlaveID in self.motors_map:
                if RID in self.motors_map[Motor.SlaveID].temp_param_dict:
                    if abs(self.motors_map[Motor.SlaveID].temp_param_dict[RID] - ControlMode) < 0.1:
                        return True
                    else:
                        return False
        return False

    def save_motor_param(self, Motor):
        """
        save the all parameter  to flash 保存所有电机参数
        :param Motor: Motor object 电机对象
        :return:
        """
        can_id_l = Motor.SlaveID & 0xff #id low 8 bits
        can_id_h = (Motor.SlaveID >> 8)& 0xff  #id high 8 bits
        data_buf = np.array([np.uint8(can_id_l), np.uint8(can_id_h), 0xAA, 0x00, 0x00, 0x00, 0x00, 0x00], np.uint8)
        self.disable(Motor)  # before save disable the motor
        self.__send_data(0x7FF, data_buf)
        sleep(0.001)

    def change_limit_param(self, Motor_Type, PMAX, VMAX, TMAX):
        """
        change the PMAX VMAX TMAX of the motor 改变电机的PMAX VMAX TMAX
        :param Motor_Type:
        :param PMAX: 电机的PMAX
        :param VMAX: 电机的VMAX
        :param TMAX: 电机的TMAX
        :return:
        """
        self.Limit_Param[Motor_Type][0] = PMAX
        self.Limit_Param[Motor_Type][1] = VMAX
        self.Limit_Param[Motor_Type][2] = TMAX

    def refresh_motor_status(self,Motor):
        """
        get the motor status 获得电机状态
        """
        can_id_l = Motor.SlaveID & 0xff #id low 8 bits
        can_id_h = (Motor.SlaveID >> 8) & 0xff  #id high 8 bits
        data_buf = np.array([np.uint8(can_id_l), np.uint8(can_id_h), 0xCC, 0x00, 0x00, 0x00, 0x00, 0x00], np.uint8)
        self.__send_data(0x7FF, data_buf)
        self.recv()  # receive the data from serial port

    def change_motor_param(self, Motor, RID, data):
        """
        change the RID of the motor 改变电机的参数
        :param Motor: Motor object 电机对象
        :param RID: DM_variable 电机参数
        :param data: 电机参数的值
        :return: True or False ,True means success, False means fail
        """
        max_retries = 20
        retry_interval = 0.05  #retry times

        self.__write_motor_param(Motor, RID, data)
        for _ in range(max_retries):
            self.recv_set_param_data()
            if Motor.SlaveID in self.motors_map and RID in self.motors_map[Motor.SlaveID].temp_param_dict:
                if abs(self.motors_map[Motor.SlaveID].temp_param_dict[RID] - data) < 0.1:
                    return True
                else:
                    return False
            sleep(retry_interval)
        return False

    def read_motor_param(self, Motor, RID):
        """
        read only the RID of the motor 读取电机的内部信息例如 版本号等
        :param Motor: Motor object 电机对象
        :param RID: DM_variable 电机参数
        :return: 电机参数的值
        """
        max_retries = 20
        retry_interval = 0.05  #retry times
        self.__read_RID_param(Motor, RID)
        for _ in range(max_retries):
            sleep(retry_interval)
            self.recv_set_param_data()
            if Motor.SlaveID in self.motors_map:
                if RID in self.motors_map[Motor.SlaveID].temp_param_dict:
                    return self.motors_map[Motor.SlaveID].temp_param_dict[RID]
        return None

    # -------------------------------------------------
    # Extract packets from the serial data
    def __extract_packets(self, data):
        frames = []
        header = 0xAA
        tail = 0x55
        frame_length = 16
        i = 0
        remainder_pos = 0

        while i <= len(data) - frame_length:
            if data[i] == header and data[i + frame_length - 1] == tail:
                frame = data[i:i + frame_length]
                frames.append(frame)
                i += frame_length
                remainder_pos = i
            else:
                i += 1
        self.data_save = data[remainder_pos:]
        return frames


def LIMIT_MIN_MAX(x, min, max):
    if x <= min:
        x = min
    elif x > max:
        x = max


def float_to_uint(x: float, x_min: float, x_max: float, bits):
    LIMIT_MIN_MAX(x, x_min, x_max)
    span = x_max - x_min
    data_norm = (x - x_min) / span
    return np.uint16(data_norm * ((1 << bits) - 1))


def uint_to_float(x: np.uint16, min: float, max: float, bits):
    span = max - min
    data_norm = float(x) / ((1 << bits) - 1)
    temp = data_norm * span + min
    return np.float32(temp)


def float_to_uint8s(value):
    # Pack the float into 4 bytes
    packed = pack('f', value)
    # Unpack the bytes into four uint8 values
    return unpack('4B', packed)


def data_to_uint8s(value):
    # Check if the value is within the range of uint32
    if isinstance(value, int) and (0 <= value <= 0xFFFFFFFF):
        # Pack the uint32 into 4 bytes
        packed = pack('I', value)
    else:
        raise ValueError("Value must be an integer within the range of uint32")

    # Unpack the bytes into four uint8 values
    return unpack('4B', packed)


def is_in_ranges(number):
    """
    check if the number is in the range of uint32
    :param number:
    :return:
    """
    if (7 <= number <= 10) or (13 <= number <= 16) or (35 <= number <= 36):
        return True
    return False


def uint8s_to_uint32(byte1, byte2, byte3, byte4):
    # Pack the four uint8 values into a single uint32 value in little-endian order
    packed = pack('<4B', byte1, byte2, byte3, byte4)
    # Unpack the packed bytes into a uint32 value
    return unpack('<I', packed)[0]


def uint8s_to_float(byte1, byte2, byte3, byte4):
    # Pack the four uint8 values into a single float value in little-endian order
    packed = pack('<4B', byte1, byte2, byte3, byte4)
    # Unpack the packed bytes into a float value
    return unpack('<f', packed)[0]


def print_hex(data):
    hex_values = [f'{byte:02X}' for byte in data]
    print(' '.join(hex_values))


def get_enum_by_index(index, enum_class):
    try:
        return enum_class(index)
    except ValueError:
        return None


class DM_Motor_Type(IntEnum):
    DM4310 = 0
    DM4310_48V = 1
    DM4340 = 2
    DM4340_48V = 3
    DM6006 = 4
    DM8006 = 5
    DM8009 = 6
    DM10010L = 7
    DM10010 = 8
    DMH3510 = 9
    DMH6215 = 10
    DMG6220 = 11
    DM6248 = 12


class DM_variable(IntEnum):
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


class Control_Type(IntEnum):
    MIT = 1
    POS_VEL = 2
    VEL = 3
    Torque_Pos = 4


class ServoController:
    """
    舵机控制类 - 集成到DM_CAN系统中
    支持3个舵机的位置和速度读写控制
    协议更新:
    - 0x06: 写入3个舵机角度 [a1, a2, b1, b2, c1, c2, xx, xx]
    - 0x07: 读取舵机A,B的位置和速度 [pa1, pa2, va1, va2, pb1, pb2, vb1, vb2]
    - 0x08: 读取舵机C的位置和速度 [pc1, pc2, vc1, vc2, xx, xx, xx, xx]
    """
    
    def __init__(self, motor_control: MotorControl):
        """
        初始化舵机控制器
        
        Args:
            motor_control: MotorControl对象，用于串口通信
        """
        self.mc = motor_control
        self.servo_positions = [0, 0, 0]  # 3个舵机的当前位置缓存
        self.servo_velocities = [0, 0, 0]  # 3个舵机的当前速度缓存
        self.write_id = 0x06  # 写位置命令ID
        self.read_ab_id = 0x07   # 读取舵机A,B命令ID
        self.read_c_id = 0x08    # 读取舵机C命令ID
        
    def set_servo_position(self, servo_index: int, position: int) -> bool:
        """
        设置单个舵机位置
        
        Args:
            servo_index: 舵机索引 (0-2)
            position: 目标位置值
            
        Returns:
            bool: 设置是否成功
        """
        if not (0 <= servo_index < 3):
            print(f"舵机索引超出范围: {servo_index}, 应该在0-2之间")
            return False
            
        # 更新缓存位置
        self.servo_positions[servo_index] = position
        
        # 发送所有舵机位置
        return self.set_all_servo_positions(self.servo_positions)
    
    def set_all_servo_positions(self, positions: List[int]) -> bool:
        """
        设置所有舵机位置
        
        Args:
            positions: 3个舵机的位置列表 [a, b, c]
            
        Returns:
            bool: 设置是否成功
        """
        if len(positions) < 3:
            print(f"位置数据不足: 需要3个，提供了{len(positions)}个")
            return False
            
        try:
            # 构造8字节数据包：[a1, a2, b1, b2, c1, c2, xx, xx]
            data = []
            for i in range(3):
                pos = int(positions[i])
                # 将位置值分解为高低字节
                high_byte = (pos >> 8) & 0xFF
                low_byte = pos & 0xFF
                data.extend([high_byte, low_byte])
            
            # 添加2字节填充
            data.extend([0x00, 0x00])
            
            # 发送位置命令
            print('舵机位置设置: ', data)
            self.mc.send_data(self.write_id, data)
            
            # 更新缓存
            self.servo_positions = positions[:3]
            
            print(f"舵机位置设置成功: {positions[:3]}")
            return True
            
        except Exception as e:
            print(f"设置舵机位置失败: {e}")
            return False
    
    def get_servo_positions(self) -> Optional[List[int]]:
        """
        读取所有舵机当前位置
        
        Returns:
            List[int]: 3个舵机的位置列表 [a, b, c]，失败返回None
        """
        try:
            positions = []
            
            # 读取舵机A和B的位置 (0x07)
            read_data = [0, 0, 0, 0, 0, 0, 0, 0]  # 8字节空数据
            self.mc.send_data(self.read_ab_id, read_data)
            sleep(0.1)  # 等待响应
            
            raw_data_ab = self.mc.recv_raw()
            sleep(0.1)
            print('ab recv raw data is ', raw_data_ab)
            if raw_data_ab and len(raw_data_ab) > 0 and len(raw_data_ab[0]) >= 15:
                data_ab = raw_data_ab[0][7:15]  # [pa1, pa2, va1, va2, pb1, pb2, vb1, vb2]
                
                # 解析舵机A位置
                pos_a = data_ab[0] * 256 + data_ab[1]
                positions.append(pos_a)
                
                # 解析舵机B位置
                pos_b = data_ab[4] * 256 + data_ab[5]
                positions.append(pos_b)
                
                # 更新速度缓存
                vel_a = data_ab[2] * 256 + data_ab[3]
                vel_b = data_ab[6] * 256 + data_ab[7]
                self.servo_velocities[0] = vel_a
                self.servo_velocities[1] = vel_b
            else:
                print("读取舵机A,B失败")
                return None
            
            # 读取舵机C的位置 (0x08)
            self.mc.send_data(self.read_c_id, read_data)
            sleep(0.1)  # 等待响应
            
            raw_data_c = self.mc.recv_raw()
            print('c recv raw data is ', raw_data_c)
            if raw_data_c and len(raw_data_c) > 0 and len(raw_data_c[0]) >= 15:
                data_c = raw_data_c[0][7:15]  # [pc1, pc2, vc1, vc2, xx, xx, xx, xx]
                
                # 解析舵机C位置
                pos_c = data_c[0] * 256 + data_c[1]
                positions.append(pos_c)
                
                # 更新速度缓存
                vel_c = data_c[2] * 256 + data_c[3]
                self.servo_velocities[2] = vel_c
            else:
                print("读取舵机C失败")
                return None
            
            # 更新位置缓存
            self.servo_positions = positions
            
            print(f"舵机位置读取成功: {positions}")
            return positions
                
        except Exception as e:
            print(f"读取舵机位置失败: {e}")
            return None
    
    def get_servo_velocities(self) -> Optional[List[int]]:
        """
        获取所有舵机当前速度
        
        Returns:
            List[int]: 3个舵机的速度列表 [va, vb, vc]，失败返回None
        """
        # 先读取位置（同时会更新速度缓存）
        if self.get_servo_positions() is not None:
            return self.servo_velocities.copy()
        return None
    
    def get_servo_position(self, servo_index: int) -> Optional[int]:
        """
        获取单个舵机位置
        
        Args:
            servo_index: 舵机索引 (0-2)
            
        Returns:
            int: 舵机位置，失败返回None
        """
        if not (0 <= servo_index < 3):
            print(f"舵机索引超出范围: {servo_index}")
            return None
            
        positions = self.get_servo_positions()
        if positions:
            return positions[servo_index]
        return None
    
    def get_servo_velocity(self, servo_index: int) -> Optional[int]:
        """
        获取单个舵机速度
        
        Args:
            servo_index: 舵机索引 (0-2)
            
        Returns:
            int: 舵机速度，失败返回None
        """
        if not (0 <= servo_index < 3):
            print(f"舵机索引超出范围: {servo_index}")
            return None
            
        velocities = self.get_servo_velocities()
        if velocities:
            return velocities[servo_index]
        return None
    
    def move_servo_relative(self, servo_index: int, delta: int) -> bool:
        """
        相对移动舵机位置
        
        Args:
            servo_index: 舵机索引 (0-2)
            delta: 位置增量
            
        Returns:
            bool: 移动是否成功
        """
        if not (0 <= servo_index < 3):
            print(f"舵机索引超出范围: {servo_index}")
            return False
            
        current_pos = self.get_servo_position(servo_index)
        if current_pos is None:
            print(f"无法获取舵机{servo_index}当前位置")
            return False
            
        new_pos = current_pos + delta
        return self.set_servo_position(servo_index, new_pos)
    
    def get_cached_positions(self) -> List[int]:
        """
        获取缓存的舵机位置（不进行实际读取）
        
        Returns:
            List[int]: 缓存的3个舵机位置
        """
        return self.servo_positions.copy()
    
    def get_cached_velocities(self) -> List[int]:
        """
        获取缓存的舵机速度（不进行实际读取）
        
        Returns:
            List[int]: 缓存的3个舵机速度
        """
        return self.servo_velocities.copy()
    
    def demo_servo_control(self, cycles: int = 5):
        """
        舵机控制演示
        
        Args:
            cycles: 演示循环次数
        """
        print("开始舵机控制演示...")
        
        # 初始位置
        initial_positions = [3131, 156, 0, 0]
        delta = 200
        
        try:
            for cycle in range(cycles):
                print(f"\n=== 演示循环 {cycle + 1}/{cycles} ===")
                
                # 读取当前位置
                current_positions = self.get_servo_positions()
                if current_positions:
                    print(f"当前位置: {current_positions}")
                
                # 设置新位置（增加delta）
                new_positions = [pos + delta for pos in initial_positions]
                success = self.set_all_servo_positions(new_positions)
                if success:
                    print(f"移动到: {new_positions}")
                
                sleep(2)
                
                # 恢复初始位置
                success = self.set_all_servo_positions(initial_positions)
                if success:
                    print(f"恢复到: {initial_positions}")
                
                sleep(2)
                
        except KeyboardInterrupt:
            print("\n用户中断演示")
        except Exception as e:
            print(f"演示过程中出错: {e}")
        
        print("舵机控制演示完成")
