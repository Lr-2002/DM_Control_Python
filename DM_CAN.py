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
        """发送数据到电机 (兼容DM和HT_CAN)"""
        self.__send_data(motor_id, data)
    
    def send_ht_can_data(self, motor_id, data):
        """
        发送HT_CAN协议数据 - 使用DM的USB-CAN转接器格式
        
        Args:
            motor_id: 电机ID  
            data: 8字节数据列表
        """
        try:
            # 构造HT_CAN的CAN ID (0x8000 | motor_id)
            can_id = 0x8000 | motor_id
            
            # 确保数据长度为8字节
            if len(data) < 8:
                data += [0x00] * (8 - len(data))
            elif len(data) > 8:
                data = data[:8]
            
            # 调试输出
            print(f"发送HT_CAN: motor_id={motor_id}, can_id=0x{can_id:04X}")
            print(f"数据: {[hex(x) for x in data]}")
            
            # 复制DM的帧模板
            ht_frame = self.send_data_frame.copy()
            
            # 设置HT_CAN的CAN ID到DM格式的对应位置
            ht_frame[13] = can_id & 0xFF        # CAN ID 低字节
            ht_frame[14] = (can_id >> 8) & 0xFF # CAN ID 高字节
            
            # 设置数据到DM格式的数据位置 (Byte 21-28)
            ht_frame[21:29] = data
            
            # 调试输出完整帧
            print(f"完整帧: {[hex(x) for x in ht_frame]}")
            
            self.serial_.write(bytes(ht_frame))
            
        except Exception as e:
            print(f"HT_CAN数据发送失败: {e}")
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
        print('send data is ', ' '.join(f'{b:02X}' for b in self.send_data_frame))

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
            print(f"舵机控制演示失败: {e}")


class HT_Motor:
    """高擎电机对象类"""
    def __init__(self, motor_id: int, motor_type: str = "M4438_30"):
        self.motor_id = motor_id
        self.motor_type = motor_type
        self.position = 0.0
        self.velocity = 0.0
        self.torque = 0.0
        self.temperature = 0.0
        self.is_enabled = False
        
    def update_state(self, position: float, velocity: float, torque: float, temperature: float = 0.0):
        """更新电机状态"""
        self.position = position
        self.velocity = velocity
        self.torque = torque
        self.temperature = temperature


class HT_CAN_Controller:
    """
    高擎电机HT_CAN协议控制器
    专门用于4438_30等高擎电机的控制和监听
    """
    
    def __init__(self, motor_control: MotorControl):
        """
        初始化HT_CAN控制器
        
        Args:
            motor_control: MotorControl对象，用于串口通信
        """
        self.mc = motor_control
        self.motors = {}  # 存储电机对象
        
        # HT_CAN协议命令定义 (根据协议文档)
        # 读取状态: cmd=0x17, addr=0x01 (读取位置、速度、力矩)
        self.CMD_READ_STATE = 0x17
        self.ADDR_READ_STATE = 0x01
        
        # 普通模式控制: cmd1=0x07, cmd2=0x07
        self.CMD_NORMAL_MODE = [0x07, 0x07]
        
        # 力矩模式控制: cmd1=0x05, cmd2=0x13  
        self.CMD_TORQUE_MODE = [0x05, 0x13]
        
        # 协同控制模式: cmd1=0x07, cmd2=0x35
        self.CMD_COOP_MODE = [0x07, 0x35]
        
        # 电机停止: 0x01, 0x00, 0x00
        self.CMD_STOP = [0x01, 0x00, 0x00]
        
        # 电机刹车: 0x01, 0x00, 0x0f
        self.CMD_BRAKE = [0x01, 0x00, 0x0f]
        
        # 周期状态返回: 0x05, 0xb4
        self.CMD_TIMED_RETURN = [0x05, 0xb4]
        
        # 无限制标志
        self.NO_LIMIT = 0x8000
        
        # 4438_30电机参数
        self.MOTOR_PARAMS = {
            "M4438_30": {
                "max_position": 12.5,    # 最大位置 (rad)
                "max_velocity": 30.0,    # 最大速度 (rad/s)
                "max_torque": 10.0,      # 最大力矩 (Nm)
                "reduction_ratio": 30    # 减速比
            }
        }
    
    def add_motor(self, motor_id: int, motor_type: str = "M4438_30") -> bool:
        """
        添加电机到控制器
        
        Args:
            motor_id: 电机ID (1-127)
            motor_type: 电机型号
            
        Returns:
            bool: 添加是否成功
        """
        if not (1 <= motor_id <= 127):
            print(f"电机ID超出范围: {motor_id}, 应该在1-127之间")
            return False
            
        if motor_type not in self.MOTOR_PARAMS:
            print(f"不支持的电机型号: {motor_type}")
            return False
            
        self.motors[motor_id] = HT_Motor(motor_id, motor_type)
        print(f"添加电机成功: ID={motor_id}, 型号={motor_type}")
        return True
    
    def enable_motor(self, motor_id: int) -> bool:
        """
        使能电机 (HT_CAN协议中通过发送控制命令自动使能)
        这里通过读取状态来激活电机
        
        Args:
            motor_id: 电机ID
            
        Returns:
            bool: 使能是否成功
        """
        if motor_id not in self.motors:
            print(f"电机ID {motor_id} 未找到")
            return False
            
        try:
            # HT_CAN协议中没有专门的使能命令
            # 通过读取状态来激活电机通信
            success = self.read_motor_state(motor_id)
            
            if success:
                self.motors[motor_id].is_enabled = True
                print(f"电机 {motor_id} 使能成功")
                return True
            else:
                print(f"电机 {motor_id} 使能失败: 无法读取状态")
                return False
            
        except Exception as e:
            print(f"电机 {motor_id} 使能失败: {e}")
            return False
    
    def disable_motor(self, motor_id: int) -> bool:
        """
        停止电机 (根据HT_CAN协议: 0x01, 0x00, 0x00)
        
        Args:
            motor_id: 电机ID
            
        Returns:
            bool: 停止是否成功
        """
        if motor_id not in self.motors:
            print(f"电机ID {motor_id} 未找到")
            return False
            
        try:
            # 根据协议文档: 电机停止命令
            data = self.CMD_STOP + [0x00] * 5  # 补齐8字节
            
            self.mc.send_ht_can_data(motor_id, data)
            self.motors[motor_id].is_enabled = False
            print(f"电机 {motor_id} 已停止")
            return True
            
        except Exception as e:
            print(f"电机 {motor_id} 停止失败: {e}")
            return False
    
    def set_position(self, motor_id: int, position: float, torque: float = 1.0) -> bool:
        """
        位置控制 (普通模式: 0x07, 0x07)
        位置单位: 圈, 力矩单位: Nm
        
        Args:
            motor_id: 电机ID
            position: 目标位置 (圈)
            torque: 最大力矩 (Nm)
            
        Returns:
            bool: 设置是否成功
        """
        if motor_id not in self.motors:
            print(f"电机ID {motor_id} 未找到")
            return False
            
        motor = self.motors[motor_id]
        params = self.MOTOR_PARAMS[motor.motor_type]
        
        try:
            # 转换为协议格式
            # 位置: 单位0.0001圈, int16
            pos_int16 = int(position * 10000)  # 转换为0.0001圈单位
            pos_int16 = max(-32767, min(32767, pos_int16))  # int16范围限制
            
            # 力矩: 需要根据4438电机的转换公式 (暂用简化版本)
            tqe_int16 = int(torque * 1000)  # 简化转换，实际需要查表
            tqe_int16 = max(-32767, min(32767, tqe_int16))
            
            # 构造普通模式位置控制命令: 0x07, 0x07, pos1, pos2, val1, val2, tqe1, tqe2
            # 位置控制时速度设为无限制(0x8000)
            data = [
                self.CMD_NORMAL_MODE[0],  # 0x07
                self.CMD_NORMAL_MODE[1],  # 0x07
                pos_int16 & 0xFF,         # pos1 (低字节)
                (pos_int16 >> 8) & 0xFF,  # pos2 (高字节)
                self.NO_LIMIT & 0xFF,     # val1 (速度无限制)
                (self.NO_LIMIT >> 8) & 0xFF,  # val2
                tqe_int16 & 0xFF,         # tqe1 (力矩低字节)
                (tqe_int16 >> 8) & 0xFF   # tqe2 (力矩高字节)
            ]
            
            self.mc.send_ht_can_data(motor_id, data)
            print(f"电机 {motor_id} 位置设置: {position:.4f} 圈, 力矩限制: {torque:.3f} Nm")
            return True
            
        except Exception as e:
            print(f"电机 {motor_id} 位置设置失败: {e}")
            return False
    
    def set_velocity(self, motor_id: int, velocity: float, torque: float = 1.0) -> bool:
        """
        速度控制 (普通模式: 0x07, 0x07)
        速度单位: 转/秒, 力矩单位: Nm
        
        Args:
            motor_id: 电机ID
            velocity: 目标速度 (转/秒)
            torque: 最大力矩 (Nm)
            
        Returns:
            bool: 设置是否成功
        """
        if motor_id not in self.motors:
            print(f"电机ID {motor_id} 未找到")
            return False
            
        try:
            # 转换为协议格式
            # 速度: 单位0.00025转/秒, int16
            vel_int16 = int(velocity / 0.00025)  # 转换为协议单位
            vel_int16 = max(-32767, min(32767, vel_int16))
            
            # 力矩: 简化转换
            tqe_int16 = int(torque * 1000)
            tqe_int16 = max(-32767, min(32767, tqe_int16))
            
            # 构造普通模式速度控制命令: 0x07, 0x07, pos1, pos2, val1, val2, tqe1, tqe2
            # 速度控制时位置设为无限制(0x8000)
            data = [
                self.CMD_NORMAL_MODE[0],  # 0x07
                self.CMD_NORMAL_MODE[1],  # 0x07
                self.NO_LIMIT & 0xFF,     # pos1 (位置无限制)
                (self.NO_LIMIT >> 8) & 0xFF,  # pos2
                vel_int16 & 0xFF,         # val1 (速度低字节)
                (vel_int16 >> 8) & 0xFF,  # val2 (速度高字节)
                tqe_int16 & 0xFF,         # tqe1 (力矩低字节)
                (tqe_int16 >> 8) & 0xFF   # tqe2 (力矩高字节)
            ]
            
            self.mc.send_ht_can_data(motor_id, data)
            print(f"电机 {motor_id} 速度设置: {velocity:.3f} 转/秒, 力矩限制: {torque:.3f} Nm")
            return True
            
        except Exception as e:
            print(f"电机 {motor_id} 速度设置失败: {e}")
            return False
    
    def read_motor_state(self, motor_id: int) -> bool:
        """
        读取电机状态
        根据HT_CAN协议: cmd=0x17, addr=0x01
        
        Args:
            motor_id: 电机ID
            
        Returns:
            bool: 读取是否成功
        """
        if motor_id not in self.motors:
            print(f"电机ID {motor_id} 未找到")
            return False
            
        try:
            # 根据HT_CAN协议构造读取状态命令
            # cmd = 0x17, addr = 0x01
            data = [
                self.CMD_READ_STATE,   # 命令字: 0x17
                self.ADDR_READ_STATE,  # 地址: 0x01
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00  # 填充
            ]
            
            self.mc.send_ht_can_data(motor_id, data)
            print(f"发送电机 {motor_id} 状态读取命令 (cmd=0x17, addr=0x01)")
            
            # 接收并解析回复
            sleep(0.1)  # 等待回复
            raw_data = self.mc.recv_raw()
            
            if raw_data and len(raw_data) > 0:
                self._parse_motor_state(motor_id, raw_data[0])
                return True
            else:
                print(f"电机 {motor_id} 无状态回复")
                return False
                
        except Exception as e:
            print(f"电机 {motor_id} 状态读取失败: {e}")
            return False
    
    def scan_ht_motors(self, id_range=(1, 20)):
        """
        扫描HT电机ID
        
        Args:
            id_range: ID扫描范围 (start, end)
        """
        print(f"🔍 扫描HT电机ID范围: {id_range[0]}-{id_range[1]}")
        found_motors = []
        
        for motor_id in range(id_range[0], id_range[1] + 1):
            print(f"测试ID: {motor_id}", end=" ")
            
            # 发送状态读取命令
            data = [0x17, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
            self.mc.send_ht_can_data(motor_id, data)
            
            sleep(0.1)
            raw_data = self.mc.recv_raw()
            
            if raw_data and len(raw_data) > 0:
                print(f"✅ 发现HT电机!")
                found_motors.append(motor_id)
                # 解析响应数据
                print(f"   响应数据: {[hex(x) for x in raw_data[0]] if raw_data[0] else 'None'}")
            else:
                print(f"❌")
        
        if found_motors:
            print(f"\n🎯 找到 {len(found_motors)} 个HT电机: {found_motors}")
        else:
            print("\n⚠️  未找到任何HT电机响应")
            
        return found_motors
    
    def _parse_motor_state(self, motor_id: int, raw_data: bytes):
        """
        解析电机状态数据 (根据HT_CAN协议)
        返回格式: cmd=0x27, addr=0x01, pos1, pos2, vel1, vel2, tqe1, tqe2
        
        Args:
            motor_id: 电机ID
            raw_data: 原始数据包
        """
        try:
            if len(raw_data) >= 8:
                data = raw_data[2:8]  # 跳过cmd和addr，提取数据部分
                
                if len(data) >= 6:
                    # 解析int16格式数据 (小端模式)
                    pos_raw = (data[1] << 8) | data[0]  # pos1, pos2
                    vel_raw = (data[3] << 8) | data[2]  # vel1, vel2  
                    tqe_raw = (data[5] << 8) | data[4]  # tqe1, tqe2
                    
                    # 转换为有符号int16
                    if pos_raw > 32767: pos_raw -= 65536
                    if vel_raw > 32767: vel_raw -= 65536
                    if tqe_raw > 32767: tqe_raw -= 65536
                    
                    # 转换为实际单位
                    position = pos_raw * 0.0001  # 转换为圈
                    velocity = vel_raw * 0.00025  # 转换为转/秒
                    torque = tqe_raw * 0.001  # 简化转换，实际需要查表
                    
                    # 更新电机状态
                    self.motors[motor_id].update_state(position, velocity, torque)
                    
                    print(f"电机 {motor_id} 状态:")
                    print(f"  位置: {position:.4f} rad")
                    print(f"  速度: {velocity:.4f} rad/s")
                    print(f"  力矩: {torque:.4f} Nm")
                    
        except Exception as e:
            print(f"解析电机 {motor_id} 状态失败: {e}")
    
    def get_motor_position(self, motor_id: int) -> float:
        """获取电机位置"""
        if motor_id in self.motors:
            return self.motors[motor_id].position
        return 0.0
    
    def get_motor_velocity(self, motor_id: int) -> float:
        """获取电机速度"""
        if motor_id in self.motors:
            return self.motors[motor_id].velocity
        return 0.0
    
    def get_motor_torque(self, motor_id: int) -> float:
        """获取电机力矩"""
        if motor_id in self.motors:
            return self.motors[motor_id].torque
        return 0.0
    
    def monitor_motor(self, motor_id: int, duration: float = 10.0, interval: float = 0.5):
        """
        监听电机状态
        
        Args:
            motor_id: 电机ID
            duration: 监听时长 (秒)
            interval: 读取间隔 (秒)
        """
        if motor_id not in self.motors:
            print(f"电机ID {motor_id} 未找到")
            return
            
        print(f"开始监听电机 {motor_id} 状态，时长 {duration} 秒...")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            self.read_motor_state(motor_id)
            sleep(interval)
        
        print(f"电机 {motor_id} 监听结束")
    
    def set_torque(self, motor_id: int, torque: float) -> bool:
        """
        纯力矩控制 (力矩模式: 0x05, 0x13)
        
        Args:
            motor_id: 电机ID
            torque: 目标力矩 (Nm)
            
        Returns:
            bool: 设置是否成功
        """
        if motor_id not in self.motors:
            print(f"电机ID {motor_id} 未找到")
            return False
            
        try:
            # 力矩转换 (简化版本，实际需要查表)
            tqe_int16 = int(torque * 1000)
            tqe_int16 = max(-32767, min(32767, tqe_int16))
            
            # 构造力矩模式命令: 0x05, 0x13, tqe1, tqe2
            data = [
                self.CMD_TORQUE_MODE[0],  # 0x05
                self.CMD_TORQUE_MODE[1],  # 0x13
                tqe_int16 & 0xFF,         # tqe1 (低字节)
                (tqe_int16 >> 8) & 0xFF,  # tqe2 (高字节)
                0x00, 0x00, 0x00, 0x00    # 填充到8字节
            ]
            
            self.mc.send_ht_can_data(motor_id, data)
            print(f"电机 {motor_id} 力矩设置: {torque:.3f} Nm")
            return True
            
        except Exception as e:
            print(f"电机 {motor_id} 力矩设置失败: {e}")
            return False
    
    def brake_motor(self, motor_id: int) -> bool:
        """
        电机刹车 (0x01, 0x00, 0x0f)
        
        Args:
            motor_id: 电机ID
            
        Returns:
            bool: 刹车是否成功
        """
        if motor_id not in self.motors:
            print(f"电机ID {motor_id} 未找到")
            return False
            
        try:
            data = self.CMD_BRAKE + [0x00] * 5  # 补齐8字节
            
            self.mc.send_ht_can_data(motor_id, data)
            print(f"电机 {motor_id} 已刹车")
            return True
            
        except Exception as e:
            print(f"电机 {motor_id} 刹车失败: {e}")
            return False
    
    def set_timed_return(self, motor_id: int, period_ms: int) -> bool:
        """
        设置周期状态返回 (0x05, 0xb4)
        
        Args:
            motor_id: 电机ID
            period_ms: 周期时间 (毫秒), 0表示停止
            
        Returns:
            bool: 设置是否成功
        """
        if motor_id not in self.motors:
            print(f"电机ID {motor_id} 未找到")
            return False
            
        try:
            # 构造周期返回命令
            data = [
                self.CMD_TIMED_RETURN[0],  # 0x05
                self.CMD_TIMED_RETURN[1],  # 0xb4
                0x02, 0x00, 0x00,          # 固定参数
                period_ms & 0xFF,          # 周期低字节
                (period_ms >> 8) & 0xFF,   # 周期高字节
                0x00                       # 填充
            ]
            
            self.mc.send_ht_can_data(motor_id, data)
            if period_ms > 0:
                print(f"电机 {motor_id} 设置周期返回: {period_ms}ms")
            else:
                print(f"电机 {motor_id} 停止周期返回")
            return True
            
        except Exception as e:
            print(f"电机 {motor_id} 周期返回设置失败: {e}")
            return False
    
    def set_position_velocity_torque(self, motor_id: int, position: float, velocity: float, torque: float) -> bool:
        """
        协同控制模式 (0x07, 0x35)
        同时控制位置、速度、力矩
        
        Args:
            motor_id: 电机ID
            position: 目标位置 (圈)
            velocity: 目标速度 (转/秒)
            torque: 最大力矩 (Nm)
            
        Returns:
            bool: 设置是否成功
        """
        if motor_id not in self.motors:
            print(f"电机ID {motor_id} 未找到")
            return False
            
        try:
            # 转换为协议格式
            pos_int16 = int(position * 10000)  # 位置: 0.0001圈
            vel_int16 = int(velocity / 0.00025)  # 速度: 0.00025转/秒
            tqe_int16 = int(torque * 1000)  # 力矩简化转换
            
            # 限制范围
            pos_int16 = max(-32767, min(32767, pos_int16))
            vel_int16 = max(-32767, min(32767, vel_int16))
            tqe_int16 = max(-32767, min(32767, tqe_int16))
            
            # 构造协同控制命令: 0x07, 0x35, val1, val2, tqe1, tqe2, pos1, pos2
            data = [
                self.CMD_COOP_MODE[0],    # 0x07
                self.CMD_COOP_MODE[1],    # 0x35
                vel_int16 & 0xFF,         # val1 (速度低字节)
                (vel_int16 >> 8) & 0xFF,  # val2 (速度高字节)
                tqe_int16 & 0xFF,         # tqe1 (力矩低字节)
                (tqe_int16 >> 8) & 0xFF,  # tqe2 (力矩高字节)
                pos_int16 & 0xFF,         # pos1 (位置低字节)
                (pos_int16 >> 8) & 0xFF   # pos2 (位置高字节)
            ]
            
            self.mc.send_ht_can_data(motor_id, data)
            print(f"电机 {motor_id} 协同控制 - 位置: {position:.4f}圈, 速度: {velocity:.3f}转/秒, 力矩: {torque:.3f}Nm")
            return True
            
        except Exception as e:
            print(f"电机 {motor_id} 协同控制失败: {e}")
            return False


if __name__ == "__main__":
    pass
