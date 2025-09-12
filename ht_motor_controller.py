#!/usr/bin/env python3

import serial
import time
import numpy as np
import signal
import sys

class HTMotorController:
	def __init__(self, port="/dev/cu.usbmodem00000000050C1", baudrate=921600, timeout=0.1):
		"""
		HT电机控制器
		
		Args:
			port: 串口设备路径
			baudrate: 波特率
			timeout: 超时时间
		"""
		self.port = port
		self.baudrate = baudrate
		self.timeout = timeout
		self.ser = None
		self.motor_states = {}
		
		# 单位转换常数 (根据文档)
		self.POSITION_SCALE = 0.0001  # int16: 0.0001 turns/LSB
		self.VELOCITY_SCALE = 0.00025  # int16: 0.00025 rev/s/LSB
		self.TORQUE_K = 0.004587      # int16: k=0.004587
		self.TORQUE_D = -0.290788     # int16: d=-0.290788
		
		# 控制帧模板
		self.control_frame = np.array([
			0x55, 0xAA, 0x1e, 0x03, 0x01, 0x00, 0x00, 0x00, 
			0x0a, 0x00, 0x00, 0x00, 0x00,
			0, 0, 0, 0,  # motor_id
			0x00, 0x08, 0x00, 0x00,  # frame_type, id_acc, data_acc, len
			0, 0, 0, 0, 0, 0, 0, 0,  # data
			0x00
		], np.uint8)
		
		# 读取帧模板
		self.read_frame = np.array([
			0x55, 0xAA, 0x1e, 0x03, 0x01, 0x00, 0x00, 0x00,
			0x0a, 0x00, 0x00, 0x00, 0x01,
			0, 0, 0, 0,  # motor_id
			0x00, 0x08, 0x00, 0x00,  # frame_type, id_acc, data_acc, len
			0, 0, 0, 0, 0, 0, 0, 0,  # data
			0x00
		], np.uint8)
		
		self._setup_signal_handler()
	
	# 单位转换函数
	def position_to_raw(self, position_turns):
		"""位置转换: turns -> raw"""
		return int(position_turns / self.POSITION_SCALE)
	
	def raw_to_position(self, raw_value):
		"""位置转换: raw -> turns"""
		if raw_value > 32767: raw_value -= 65536  # 转换为有符号数
		return raw_value * self.POSITION_SCALE
	
	def velocity_to_raw(self, velocity_revs):
		"""速度转换: rev/s -> raw"""
		return int(velocity_revs / self.VELOCITY_SCALE)
	
	def raw_to_velocity(self, raw_value):
		"""速度转换: raw -> rev/s"""
		if raw_value > 32767: raw_value -= 65536  # 转换为有符号数
		return raw_value * self.VELOCITY_SCALE
	
	def torque_to_raw(self, torque_nm):
		"""力矩转换: Nm -> raw (torque = k * raw + d)"""
		return int((torque_nm - self.TORQUE_D) / self.TORQUE_K)
	
	def raw_to_torque(self, raw_value):
		"""力矩转换: raw -> Nm (torque = k * raw + d)"""
		if raw_value > 32767: raw_value -= 65536  # 转换为有符号数
		return self.TORQUE_K * raw_value + self.TORQUE_D
		
	def connect(self):
		"""连接串口"""
		try:
			if self.ser and self.ser.is_open:
				self.ser.close()
			
			self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
			print(f"Connected to {self.port} at {self.baudrate} baud")
			return True
		except Exception as e:
			print(f"Connection failed: {e}")
			return False
	
	def disconnect(self):
		"""断开串口连接"""
		if self.ser and self.ser.is_open:
			self.ser.close()
			print("Disconnected")
	
	def _setup_signal_handler(self):
		"""设置信号处理器"""
		def signal_handler(sig, frame):
			print('\nStopping all motors...')
			for motor_id in self.motor_states.keys():
				self.disable_motor(motor_id)
			self.disconnect()
			sys.exit(0)
		
		signal.signal(signal.SIGINT, signal_handler)
	
	def _send_data(self, motor_id, data, read_mode=False):
		"""
		发送数据到电机
		
		Args:
			motor_id: 电机ID
			data: 8字节数据列表
			read_mode: 是否为读取模式
		"""
		if not self.ser or not self.ser.is_open:
			print("Serial port not connected")
			return False
		
		# 选择帧模板
		frame = self.read_frame.copy() if read_mode else self.control_frame.copy()
		
		# 确保数据长度为8字节
		data = np.array(data)
		if data.shape != (8,):
			raw_data = np.zeros(8, dtype=np.uint8)
			raw_data[:min(len(data), 8)] = data[:8]
			data = raw_data
		
		# 设置电机ID和数据
		frame[13] = motor_id & 0xFF
		frame[14] = (motor_id >> 8) & 0xFF
		frame[21:29] = data
		
		# 发送数据
		self.ser.write(bytes(frame))
		
		# 调试输出
		hex_cmd = ' '.join(f'{b:02X}' for b in frame)
		# print(f"TX: {hex_cmd}")
		
		return True
	
	def controlMIT(self, motor_id, pos, vel, tqe):
		"""MIT控制模式 (使用原始值)"""
		print('controlMIT', hex(motor_id), pos, vel, tqe)
		pos1 = pos % 256
		pos2 = pos // 256
		vel1 = vel % 256
		vel2 = vel // 256
		tqe1 = tqe % 256
		tqe2 = tqe // 256
		self._send_data(motor_id, [0x07, 0x35, pos1, pos2, vel1, vel2, tqe1, tqe2])
		self._recv_data()
	
	def control_mit_real(self, motor_id, position_turns=None, velocity_revs=None, torque_nm=None):
		"""MIT控制模式 (使用真实单位)
		
		Args:
			motor_id: 电机ID
			position_turns: 目标位置 (转数), None表示不限制
			velocity_revs: 目标速度 (转/秒), None表示不限制  
			torque_nm: 目标力矩 (牛米), None表示不限制
		"""
		# 转换为原始值，None用0x8000表示不限制
		pos_raw = self.position_to_raw(position_turns) if position_turns is not None else 0x8000
		vel_raw = self.velocity_to_raw(velocity_revs) if velocity_revs is not None else 0x8000
		tqe_raw = self.torque_to_raw(torque_nm) if torque_nm is not None else 0x8000
		
		# 限制在int16范围内
		pos_raw = max(-32768, min(32767, pos_raw))
		vel_raw = max(-32768, min(32767, vel_raw))
		tqe_raw = max(-32768, min(32767, tqe_raw))
		
		print(f'MIT Control: pos={position_turns} turns, vel={velocity_revs} rev/s, tqe={torque_nm} Nm')
		self.controlMIT(motor_id, pos_raw, vel_raw, tqe_raw)

	def _recv_data(self):
		"""接收数据"""
		if not self.ser or not self.ser.is_open:
			return b''
		
		data = self.ser.read_all()
		if data:
			# print(f"RX: {' '.join(f'{b:02X}' for b in data)}")
			frames = self._extract_packets(data)
			for frame in frames:
				self._parse_motor_response(frame)
		
		return data
	

	def _extract_packets(self, data):
		"""解包接收到的数据帧"""
		frames = []
		header = 0xAA
		tail = 0x55
		frame_length = 16
		i = 0
		
		while i <= len(data) - frame_length:
			if data[i] == header and data[i + frame_length - 1] == tail:
				frame = data[i:i + frame_length]
				frames.append(frame)
				i += frame_length
			else:
				i += 1
		
		return frames
	

	def _parse_motor_response(self, frame):
		"""解析电机响应帧"""
		if len(frame) < 16:
			return
		
		# 帧格式: AA 11 08 canid[4] data[8] 55
		can_id = (frame[6] << 24) | (frame[5] << 16) | (frame[4] << 8) | frame[3]
		motor_id = frame[3]  # 电机ID在CAN ID的低字节
		cmd = frame[7]
		addr = frame[8]
		data = frame[9:15]  # a1, a2, b1, b2, c1, c2
		
		# 解析CMD格式
		reply_flag = (cmd >> 4) & 0x0F  # 高4位: 0010表示回复
		data_type = (cmd >> 2) & 0x03   # 2-3位: 数据类型
		data_count = cmd & 0x03         # 低2位: 数据数量
		
		# print(f"Motor ID: 0x{motor_id:02X}, CMD: 0x{cmd:02X}, Addr: 0x{addr:02X}")
		# print(f"  Reply: {reply_flag}, Type: {data_type}, Count: {data_count}")
		
		if reply_flag == 0x02:  # 回复帧
			if addr == 0x01:  # 状态响应
				# 解析数据根据类型和数量
				if data_type == 0x01 and data_count == 0x03:  # int16_t, 3个数据
					pos_raw = (data[1] << 8) | data[0]  # 小端模式
					vel_raw = (data[3] << 8) | data[2]
					tqe_raw = (data[5] << 8) | data[4]
					
					# 转换为有符号数
					if pos_raw > 32767: pos_raw -= 65536
					if vel_raw > 32767: vel_raw -= 65536
					if tqe_raw > 32767: tqe_raw -= 65536
					
					# 转换为实际单位
					position = self.raw_to_position(pos_raw)
					velocity = self.raw_to_velocity(vel_raw)
					torque = self.raw_to_torque(tqe_raw)
					
					# 更新电机状态
					if motor_id not in self.motor_states:
						self.motor_states[motor_id] = {}
					
					self.motor_states[motor_id].update({
						'position': position,
						'velocity': velocity,
						'torque': torque,
						'timestamp': time.time()
					})
					
					print(f"  q: {position:.4f} turns | dq: {velocity:.4f} rev/s | t: {torque:.3f} Nm")
			
			elif addr == 0x2B:  # PID参数响应 (P/D/I)
				if data_type == 0x01 and data_count == 0x03:  # int16_t, 3个数据
					kp_raw = (data[1] << 8) | data[0]  # 小端模式
					kd_raw = (data[3] << 8) | data[2]
					ki_raw = (data[5] << 8) | data[4]
					
					# 转换为有符号数
					if kp_raw > 32767: kp_raw -= 65536
					if kd_raw > 32767: kd_raw -= 65536
					if ki_raw > 32767: ki_raw -= 65536
					
					# 转换为实际值 (根据文档 int16: 1/32767 - 0.000030519)
					kp_actual = kp_raw / 32767.0
					kd_actual = kd_raw / 32767.0
					ki_actual = ki_raw / 32767.0

					print(f"  Kp raw: {kp_raw}, actual: {kp_actual:.6f}")
					print(f"  Kd raw: {kd_raw}, actual: {kd_actual:.6f}")
					print(f"  Ki raw: {ki_raw}, actual: {ki_actual:.6f}")
					
					# 更新电机参数状态
					if motor_id not in self.motor_states:
						self.motor_states[motor_id] = {}
					
					self.motor_states[motor_id].update({
						'kp_raw': kp_raw,
						'kd_raw': kd_raw,
						'ki_raw': ki_raw,
						'kp_actual': kp_actual,
						'kd_actual': kd_actual,
						'ki_actual': ki_actual,
						'param_timestamp': time.time()
					})
			else:
				print(f"  Unknown addr: 0x{addr:02X}")
				print(f"  Raw data: {' '.join(f'{b:02X}' for b in data)}")
		else:
			print(f"  Not a reply frame")
			print(f"  Raw data: {' '.join(f'{b:02X}' for b in data)}")
	
	def read_motor_state(self, motor_id):
		"""读取电机状态"""
		# print(f"Reading state from motor 0x{motor_id:04X}")
		self._send_data(motor_id, [0x17, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], read_mode=True)
		time.sleep(0.001)
		self._recv_data()
		
		return self.motor_states.get(motor_id, None)
	
	def position_control(self, motor_id, position, torque_limit, velocity=0x8000):
		"""位置控制 (原始值)"""
		pos1 = position & 0xFF
		pos2 = (position >> 8) & 0xFF
		vel1 = velocity & 0xFF
		vel2 = (velocity >> 8) & 0xFF
		tqe1 = torque_limit & 0xFF
		tqe2 = (torque_limit >> 8) & 0xFF
		
		print(f"Position control: motor=0x{motor_id:04X}, pos={position}, vel={velocity}, tqe={torque_limit}")
		self._send_data(motor_id, [0x07, 0x07, pos1, pos2, vel1, vel2, tqe1, tqe2])
		time.sleep(0.1)
		self._recv_data()
	
	def position_control_real(self, motor_id, position_turns, torque_limit_nm, velocity_revs=None):
		"""位置控制 (真实单位)
		
		Args:
			motor_id: 电机ID
			position_turns: 目标位置 (转数)
			torque_limit_nm: 力矩限制 (牛米)
			velocity_revs: 速度限制 (转/秒), None表示无限制
		"""
		pos_raw = self.position_to_raw(position_turns)
		tqe_raw = self.torque_to_raw(torque_limit_nm)
		vel_raw = self.velocity_to_raw(velocity_revs) if velocity_revs is not None else 0x8000
		
		# 限制范围
		pos_raw = max(-32768, min(32767, pos_raw))
		vel_raw = max(-32768, min(32767, vel_raw))
		tqe_raw = max(-32768, min(32767, tqe_raw))
		
		print(f"Position control: {position_turns} turns, vel_limit={velocity_revs} rev/s, tqe_limit={torque_limit_nm} Nm")
		self.position_control(motor_id, pos_raw, tqe_raw, vel_raw)
	
	def velocity_control(self, motor_id, velocity, torque_limit, position=0x8000):
		"""速度控制 (原始值)"""
		pos1 = position & 0xFF
		pos2 = (position >> 8) & 0xFF
		vel1 = velocity & 0xFF
		vel2 = (velocity >> 8) & 0xFF
		tqe1 = torque_limit & 0xFF
		tqe2 = (torque_limit >> 8) & 0xFF
		
		print(f"Velocity control: motor=0x{motor_id:04X}, vel={velocity}, pos={position}, tqe={torque_limit}")
		self._send_data(motor_id, [0x07, 0x07, pos1, pos2, vel1, vel2, tqe1, tqe2])
		time.sleep(0.1)
		self._recv_data()
	
	def velocity_control_real(self, motor_id, velocity_revs, torque_limit_nm, position_turns=None):
		"""速度控制 (真实单位)
		
		Args:
			motor_id: 电机ID
			velocity_revs: 目标速度 (转/秒)
			torque_limit_nm: 力矩限制 (牛米)
			position_turns: 位置限制 (转数), None表示无限制
		"""
		vel_raw = self.velocity_to_raw(velocity_revs)
		tqe_raw = self.torque_to_raw(torque_limit_nm)
		pos_raw = self.position_to_raw(position_turns) if position_turns is not None else 0x8000
		
		# 限制范围
		pos_raw = max(-32768, min(32767, pos_raw))
		vel_raw = max(-32768, min(32767, vel_raw))
		tqe_raw = max(-32768, min(32767, tqe_raw))
		
		print(f"Velocity control: {velocity_revs} rev/s, pos_limit={position_turns} turns, tqe_limit={torque_limit_nm} Nm")
		self.velocity_control(motor_id, vel_raw, tqe_raw, pos_raw)
	
	def torque_control(self, motor_id, torque):
		"""力矩控制 (原始值)"""
		tqe1 = torque & 0xFF
		tqe2 = (torque >> 8) & 0xFF
		
		print(f"Torque control: motor=0x{motor_id:04X}, tqe={torque}")
		self._send_data(motor_id, [0x05, 0x13, tqe1, tqe2, 0x00, 0x00, 0x00, 0x00])
		time.sleep(0.1)
		self._recv_data()
	
	def torque_control_real(self, motor_id, torque_nm):
		"""力矩控制 (真实单位)
		
		Args:
			motor_id: 电机ID
			torque_nm: 目标力矩 (牛米)
		"""
		tqe_raw = self.torque_to_raw(torque_nm)
		tqe_raw = max(-32768, min(32767, tqe_raw))
		
		print(f"Torque control: {torque_nm} Nm")
		self.torque_control(motor_id, tqe_raw)
	
	def brake_motor(self, motor_id):
		"""刹车电机"""
		print(f"Braking motor 0x{motor_id:04X}")
		self._send_data(motor_id, [0x01, 0x00, 0x0f, 0x14, 0x04, 0x00, 0x11, 0x0f])
		time.sleep(0.1)
		self._recv_data()
	
	def get_motor_state(self, motor_id):
		"""获取电机状态"""
		return self.motor_states.get(motor_id, None)
	
	def list_motors(self):
		"""列出所有已知电机"""
		return list(self.motor_states.keys())
	
	def scan_motors(self, id_range=(0x8001, 0x8010)):
		"""
		扫描电机ID范围
		
		Args:
			id_range: (start_id, end_id) 扫描范围
		"""
		print(f"Scanning motors from 0x{id_range[0]:04X} to 0x{id_range[1]:04X}")
		found_motors = []
		
		for motor_id in range(id_range[0], id_range[1] + 1):
			print(f"Testing motor 0x{motor_id:04X}...", end=" ")
			
			# 发送状态读取命令
			self._send_data(motor_id, [0x17, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], read_mode=True)
			time.sleep(0.01)
			
			# 检查是否有响应
			old_count = len(self.motor_states)
			self._recv_data()
			
			if len(self.motor_states) > old_count or motor_id in self.motor_states:
				found_motors.append(motor_id)
		
		print(f"Found {len(found_motors)} motors: {[hex(m) for m in found_motors]}")
		return found_motors
	def read_pid(self, motor_id):
		"""读取电机PID参数"""
		print(f"Reading PID from motor 0x{motor_id:04X}")
		self._send_data(motor_id, [0x17, 0x2b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], read_mode=True)
		time.sleep(0.001)
		self._recv_data()
		
		return self.motor_states.get(motor_id, None)

# 使用示例
if __name__ == "__main__":
	# 创建控制器
	controller = HTMotorController()
	
	# 连接
	if not controller.connect():
		exit(1)
	
	try:
		# 扫描电机
		motors = controller.scan_motors((0x8001, 0x8010))
		
		if motors:
			motor_id = motors[0]  # 使用第一个找到的电机
			controller.read_pid(motor_id)
			print(controller.motor_states)
			
			# 读取PID参数
			params = controller.read_pid(motor_id)
			if params:
				print(f"Motor parameters: {params}")
			
			# 读取状态
			state = controller.read_motor_state(motor_id)
			if state:
				print(f"Current state: {state}")
			
			# 位置控制示例 - 使用真实单位
			controller.control_mit_real(motor_id, position_turns=0.5, torque_nm=0.5)
			for i in range(10):
				controller.read_motor_state(motor_id)
				time.sleep(0.1)
			time.sleep(2)

			
			# 读取新状态
			controller.read_motor_state(motor_id)
			# 刹车
			controller.brake_motor(motor_id)
			time.sleep(1)
		
	except KeyboardInterrupt:
		print("\nStopped by user")
	
	finally:
		controller.disconnect()
