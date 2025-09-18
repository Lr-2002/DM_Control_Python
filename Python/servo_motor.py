import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from platform import java_ver
import struct
import time
import math
from enum import IntEnum
from typing import List, Optional, Dict
from src import usb_class, can_value_type

class ServoControlMode(IntEnum):
	"""舵机控制模式"""
	DISABLE = 0x00  # 失能
	ENABLE = 0x01   # 使能
	POSITION = 0x02 # 位置控制
	READ = 0x03     # 读取状态
	MID = 0x04      # 中位

class ServoMotor:
	"""
	舵机控制类，基于CAN通信协议
	参考HT电机的实现结构
	"""
	
	def __init__(self, usb_hw, motor_id: int, can_id: int, rx_id: int):
		"""
		初始化舵机
		
		Args:
			usb_hw: USB硬件接口
			motor_id: 电机ID
			can_id: CAN发送ID
			rx_id: CAN接收ID
		"""
		self.usb_hw = usb_hw
		self.motor_id = motor_id
		self.can_id = can_id
		self.rx_id = rx_id
		
		# 舵机状态
		self.enabled = False
		self.position = 0
		self.velocity = 0
		self.torque = 0
		
		# 舵机限制参数 (0-4095表示一圈)
		self.max_position = 4095     # 最大位置
		self.min_position = 0        # 最小位置
		self.max_velocity = 4095     # 最大速度
		self.max_torque = 4095       # 最大力矩
		
	def enable(self) -> bool:
		"""使能舵机"""
		cmd = [ServoControlMode.ENABLE, 0, 0, 0, 0, 0, 0, 0]
		success = self._send_command(cmd)
		if success:
			self.enabled = True
		return success
		
	def disable(self) -> bool:
		"""失能舵机"""
		cmd = [ServoControlMode.DISABLE, 0, 0, 0, 0, 0, 0, 0]
		success = self._send_command(cmd)
		if success:
			self.enabled = False
		return success
	def set_mid(self):
		cmd = [ServoControlMode.MID, 0, 0, 0, 0, 0, 0 , 0]
		success = self._send_command(cmd)
		if success:
			print(f'[motor {self.can_id}] set mid success')
		else:
			print(f'[motor {self.can_id}] set mid failed')


	def set_position(self, position: int, velocity: int = 100) -> bool:
		"""
		设置舵机位置
		
		Args:
			position: 目标位置 (0-4095，表示一圈)
			velocity: 运动速度 (0-4095)
		"""
		# 限制位置范围 0-4095
		position = max(min(position, 4095), 0)
		velocity = max(min(velocity, 4095), 1)
		
		# 构造命令: [0x02, pos_high, pos_low, vel_high, vel_low, 0, 0, 0]
		cmd = [
			ServoControlMode.POSITION,
			(position >> 8) & 0xFF,  # 位置高8位
			position & 0xFF,         # 位置低8位
			(velocity >> 8) & 0xFF,  # 速度高8位
			velocity & 0xFF,         # 速度低8位
			0, 0, 0                  # 填充
		]
		
		return self._send_command(cmd)
		
	def read_status(self) -> bool:
		"""读取舵机状态"""
		cmd = [ServoControlMode.READ, 0, 0, 0, 0, 0, 0, 0]
		
		mark = self._send_command(cmd)
		# time.sleep(0.1)
		return mark
		
	def get_position(self) -> int:
		"""获取当前位置 (0-4095)"""
		return self.position / 4095 * 2 * 3.14
		
	def get_velocity(self) -> int:
		"""获取当前速度 (0-4095)"""
		return self.velocity
		
	def get_torque(self) -> int:
		"""获取当前力矩 (0-4095)"""
		return self.torque
		
	def is_enabled(self) -> bool:
		"""检查是否使能"""
		return self.enabled
		
	def _send_command(self, cmd: List[int]) -> bool:
		"""
		发送命令到舵机
		
		Args:
			cmd: 8字节命令数据
		"""
		try:
		   
			self.usb_hw.fdcanFrameSend(cmd, self.can_id)
			time.sleep(0.005)
			return True
			
		except Exception as e:
			print(f"舵机 {self.motor_id} 发送命令失败: {e}")
			return False
			
	def process_feedback(self, frame) -> bool:
		"""
		处理舵机反馈数据
		
		Args:
			frame: CAN帧对象
		"""
		if frame.head.id != self.rx_id or frame.head.dlc < 6:
			return False
			
		# 解析反馈数据: [pos_high, pos_low, vel_high, vel_low, torque_high, torque_low, ...]
		pos_int =frame.data[0] * 256 + frame.data[1] 
		vel_int = frame.data[2] * 256 + frame.data[3]
		torque_int = frame.data[4] * 256 + frame.data[5]
		
		# 直接使用原始值 (0-4095)
		self.position = pos_int
		self.velocity = vel_int
		self.torque = torque_int
		# print(' the data for ',self.can_id, self.position, self.velocity, self.torque)          
		return True
			

class ServoMotorManager:
	"""
	舵机管理器，管理多个舵机
	参考HTMotorManager的实现结构
	"""
	
	def __init__(self, usb_hw, as_sub_module=True):
		"""
		初始化舵机管理器
		
		Args:
			usb_hw: USB硬件接口
			as_sub_module: 是否作为子模块使用
		"""
		self.usb_hw = usb_hw
		self.servos: Dict[int, ServoMotor] = {}  # 舵机字典
		
		# 设置CAN帧回调
		if not as_sub_module:
			self.usb_hw.setFrameCallback(self.can_frame_callback)
			
	def add_servo(self,  motor_id: int, can_id: int, rx_id: int, motor=None) -> ServoMotor:
		"""
		添加舵机
		
		Args:
			motor_id: 电机逻辑ID
			can_id: 发送CAN ID
			rx_id: 接收CAN ID
			
		Returns:
			ServoMotor: 创建的舵机对象
		"""
		if motor:
			servo=motor
		else:
			servo = ServoMotor(self.usb_hw, motor_id, can_id, rx_id)
		self.servos[motor_id] = servo
		return servo
		
	def get_servo(self, motor_id: int) -> Optional[ServoMotor]:
		"""
		获取舵机对象
		
		Args:
			motor_id: 电机ID
			
		Returns:
			ServoMotor: 舵机对象，如果不存在返回None
		"""
		return self.servos.get(motor_id)
		
	def enable_all(self) -> bool:
		"""使能所有舵机"""
		success = True
		for servo in self.servos.values():
			if not servo.enable():
				success = False
		return success
		
	def disable_all(self) -> bool:
		"""失能所有舵机"""
		success = True
		for servo in self.servos.values():
			if not servo.disable():
				success = False
		return success
		
	def set_positions(self, positions: Dict[int, float], velocities: Dict[int, float] = None) -> bool:
		"""
		批量设置舵机位置
		
		Args:
			positions: 位置字典 {motor_id: position}
			velocities: 速度字典 {motor_id: velocity}
		"""
		if velocities is None:
			velocities = {}
			
		success = True
		for motor_id, position in positions.items():
			if motor_id in self.servos:
				velocity = velocities.get(motor_id, 1.0)
				if not self.servos[motor_id].set_position(position, velocity):
					success = False
			else:
				print(f"舵机 {motor_id} 不存在")
				success = False
				
		return success
		
	def read_all_status(self) -> bool:
		"""读取所有舵机状态"""
		success = True
		for servo in self.servos.values():
			if not servo.read_status():
				success = False
		return success
		
	def get_all_positions(self) -> Dict[int, float]:
		"""获取所有舵机位置"""
		positions = {}
		self.read_all_status()
		for motor_id, servo in self.servos.items():
			positions[motor_id] = servo.get_position()
		if None in positions.values():
			return None
		return positions
		
	def get_all_velocities(self) -> Dict[int, float]:
		"""获取所有舵机速度"""
		velocities = {}
		for motor_id, servo in self.servos.items():
			velocities[motor_id] = servo.get_velocity()
		return velocities
		
	def get_all_torques(self) -> Dict[int, float]:
		"""获取所有舵机力矩"""
		torques = {}
		for motor_id, servo in self.servos.items():
			torques[motor_id] = servo.get_torque()
		return torques
		
	def can_frame_callback(self, frame) -> None:
		"""
		CAN帧回调函数
		
		Args:
			frame: CAN帧对象
		"""
		# 将帧分发给对应的舵机处理
		for servo in self.servos.values():
			if servo.process_feedback(frame):
				break  # 找到处理的舵机就退出
				
	def emergency_stop(self) -> bool:
		"""紧急停止所有舵机"""
		return self.disable_all()
		
	def set_limits(self, motor_id: int, max_pos: float = None, min_pos: float = None, 
				   max_vel: float = None, max_torque: float = None) -> bool:
		"""
		设置舵机限制参数
		
		Args:
			motor_id: 电机ID
			max_pos: 最大位置
			min_pos: 最小位置
			max_vel: 最大速度
			max_torque: 最大力矩
		"""
		if motor_id not in self.servos:
			return False
			
		servo = self.servos[motor_id]
		if max_pos is not None:
			servo.max_position = max_pos
		if min_pos is not None:
			servo.min_position = min_pos
		if max_vel is not None:
			servo.max_velocity = max_vel
		if max_torque is not None:
			servo.max_torque = max_torque
			
		return True

	# def set_mids(self, motor_id=None):
	# 	if motor_id:
	# 		self.servos[motor_id].set_mid()
	# 	else:
	# 		for servo in self.servos.values():
	# 			servo.set_mid()

if __name__ =='__main__':
	# 底层舵机测试代码
	import time
	from src.usb_class import usb_class

	from usb_hw_wrapper import USBHardwareWrapper
	usb_hw = usb_class(1000000, 5000000, "F561E08C892274DB09496BCC1102DBC5")
	usb_hw = USBHardwareWrapper(usb_hw)
	print("✓ 使用真实USB硬件接口")
	
   
	# 测试单个舵机
	print("=== 测试单个舵机 ===")
	servo = ServoMotor(usb_hw, motor_id=1, can_id=0x09, rx_id=0x19)

	print("\n=== 测试舵机管理器 ===")
	servo_manager = ServoMotorManager(usb_hw, as_sub_module=False)
	
	# 添加多个舵机
	print("1. 添加舵机:")
	servo1 = servo_manager.add_servo(1, 0x09, 0x19)
	print(f"   添加舵机1: ID=1, CAN=0x09, RX=0x19")
	
	# servo_manager.disable_all()
	while True:
		pos= servo_manager.get_all_positions()
		print(pos)
	# # 测试批量使能
	# print("2. 批量使能:")
	# servo_manager.enable_all()
	# input('time to go ? ')
	
	# print("3. 批量失能:")
	# servo_manager.disable_all()
	# input('time to go ? ')
	# 测试读取所有状态
	# print("4. 读取所有状态:")
	# servo_manager.read_all_status()   
	# # 测试批量位置设置
	print("5. 批量位置设置:")
	# 获取当前位置
	current_positions = servo_manager.get_all_positions()
	# print(f"   当前位置: {current_positions}")
	input('开始运动？ ')
	# 根据当前位置进行±500控制
	positions = {}
	velocities = {}
	for motor_id, current_pos in current_positions.items():
		# 在当前位置±500范围内设置目标位置
		target_pos = max(0, min(4095, current_pos + 500))  # 先向上500
		positions[motor_id] = target_pos
		velocities[motor_id] = 100
		print(f"   舵机{motor_id}: {current_pos} -> {target_pos}")
	
	servo_manager.set_positions(positions, velocities)
	input('观察运动，按回车继续...')
	
	# 回到原位置
	print("6. 回到原位置:")
	original_positions = servo_manager.get_all_positions()
	for motor_id, current_pos in original_positions.items():
		target_pos = max(0, min(4095, current_pos - 500))  # 向下500
		positions[motor_id] = target_pos
		print(f"   舵机{motor_id}: {current_pos} -> {target_pos}")
	
	servo_manager.set_positions(positions, velocities)
	input('观察运动，按回车继续...')

	#
	print("\n=== 测试完成 ===")
	if hasattr(usb_hw, 'sent_frames'):
		print(f"总共发送了 {len(usb_hw.sent_frames)} 个CAN帧")
	else:
		print("使用真实USB接口，无法统计发送帧数")
