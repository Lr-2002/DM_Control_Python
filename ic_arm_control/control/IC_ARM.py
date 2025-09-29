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

# Optional imports
try:
	import pysnooper
	HAS_PYSNOOPER = True
except ImportError:
	HAS_PYSNOOPER = False
# 使用新的统一电机控制系统
from ic_arm_control.control.unified_motor_control import (
	DamiaoProtocol,
	HTProtocol,
	MotorManager,
	MotorInfo,
	MotorType,
	ServoProtocol,
)
from ic_arm_control.control.damiao import (
	DmMotorManager,
	DM_Motor_Type,
	limit_param as dm_limit,
)
from ic_arm_control.control.ht_motor import HTMotorManager
from ic_arm_control.control.servo_motor import ServoMotorManager
from ic_arm_control.control.src import usb_class
from ic_arm_control.control.usb_hw_wrapper import USBHardwareWrapper
from ic_arm_control.control.async_logger import AsyncLogManager
from ic_arm_control.control.safety_monitor import SafetyMonitor
from ic_arm_control.control.buffer_control_thread import BufferControlThread

# 添加mlp_compensation和urdfly模块路径
import sys
from pathlib import Path
current_dir = Path(__file__).parent
mlp_compensation_dir = current_dir / "mlp_compensation"
urdfly_dir = current_dir / "urdfly"
if mlp_compensation_dir.exists() and str(mlp_compensation_dir) not in sys.path:
	sys.path.append(str(mlp_compensation_dir))
if urdfly_dir.exists() and str(urdfly_dir) not in sys.path:
	sys.path.append(str(urdfly_dir))

# 电机名称列表（排除servo电机）
MOTOR_LIST = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8']

def debug_print(msg: str, level: str = "INFO"):
	"""Debug print with timestamp"""

	timestamp = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
	print(f"[{timestamp}] [IC_ARM-{level}] {msg}")


def safe_call(func, *args, **kwargs) -> Tuple[Any, Optional[str]]:
	"""安全函数调用，返回(结果, 错误信息)"""
	try:
		result = func(*args, **kwargs)
		# time.sleep(0.0002)
		return result, None
	except Exception as e:
		error_msg = f"{func.__name__}() 失败: {str(e)}"
		debug_print(f"安全调用失败: {error_msg}", "ERROR")
		debug_print(f"详细错误: {traceback.format_exc()}", "ERROR")
		return None, error_msg


def validate_type(
	value: Any, expected_type: Union[type, Tuple[type, ...]], name: str
) -> bool:
	"""验证变量类型"""
	if not isinstance(value, expected_type):
		# 处理tuple类型的情况
		if isinstance(expected_type, tuple):
			type_names = " or ".join([t.__name__ for t in expected_type])
		else:
			type_names = expected_type.__name__
		debug_print(
			f"类型验证失败: {name} 期望 {type_names}, 实际 {type(value).__name__}",
			"ERROR",
		)
		return False
	return True


def validate_array(array: np.ndarray, expected_shape: Tuple, name: str) -> bool:
	"""验证numpy数组形状"""
	if not isinstance(array, np.ndarray):
		debug_print(f"数组验证失败: {name} 不是numpy数组, 类型: {type(array)}", "ERROR")
		return False
	if array.shape != expected_shape:
		debug_print(
			f"数组形状验证失败: {name} 期望 {expected_shape}, 实际 {array.shape}",
			"ERROR",
		)
		return False
	return True




class ICARM:
	def __init__(
		self, device_sn="F561E08C892274DB09496BCC1102DBC5", debug=False, gc=False,
		gc_type="dyn", enable_buffered_control=True, control_freq=300, gc_only= False
	):
		"""Initialize IC ARM with unified motor control system"""
		self.debug = debug
		self.use_ht = True
		self.enable_buffered_control = enable_buffered_control
		self.control_freq = control_freq
		self.gc_type = gc_type  # 存储重力补偿类型
		debug_print("=== 初始化IC_ARM_Unified ===")

		# 初始化统一电机控制系统
		usb_hw = usb_class(1000000, 5000000, device_sn)
		usb_hw = USBHardwareWrapper(usb_hw)
		self.motor_manager = MotorManager(usb_hw)

		# 电机配置数据
		
		self.motors_data = [
			MotorInfo(1, MotorType.DAMIAO, DM_Motor_Type.DM10010L, 0x01, 0x11, 250, 5),
			MotorInfo(2, MotorType.DAMIAO, DM_Motor_Type.DM6248, 0x02, 0x12, 120, 2),
			MotorInfo(3, MotorType.DAMIAO, DM_Motor_Type.DM6248, 0x03, 0x13, 120, 2),
			MotorInfo(4, MotorType.DAMIAO, DM_Motor_Type.DM4340, 0x04, 0x14, 40, 1),
			MotorInfo(5, MotorType.DAMIAO, DM_Motor_Type.DM4340, 0x05, 0x15, 40, 1),
			MotorInfo(6, MotorType.DAMIAO, DM_Motor_Type.DM4310, 0x06, 0x16, 30, 1),
			MotorInfo(7, MotorType.HIGH_TORQUE, None, 0x8094, 0x07, 8, 1.2),
			MotorInfo(8, MotorType.HIGH_TORQUE, None, 0x8094, 0x08, 8, 1.2),
			MotorInfo(9, MotorType.SERVO, None, 0x09, 0x19, 0, 0),
		]

		self.gc_only = gc_only
		if self.gc_only:
			self.motors_data = [
				MotorInfo(1, MotorType.DAMIAO, DM_Motor_Type.DM10010L, 0x01, 0x11, 0, 0),
				MotorInfo(2, MotorType.DAMIAO, DM_Motor_Type.DM6248, 0x02, 0x12, 0, 0),
				MotorInfo(3, MotorType.DAMIAO, DM_Motor_Type.DM6248, 0x03, 0x13, 0, 0),
				MotorInfo(4, MotorType.DAMIAO, DM_Motor_Type.DM4340, 0x04, 0x14, 0, 0),
				MotorInfo(5, MotorType.DAMIAO, DM_Motor_Type.DM4340, 0x05, 0x15, 0, 0),
				MotorInfo(6, MotorType.DAMIAO, DM_Motor_Type.DM4310, 0x06, 0x16, 0, 0.),
				MotorInfo(7, MotorType.HIGH_TORQUE, None, 0x8094, 0x07, 0, 0),
				MotorInfo(8, MotorType.HIGH_TORQUE, None, 0x8094, 0x08, 0, 0),
				MotorInfo(9, MotorType.SERVO, None, 0x09, 0x19, 0, 0),
			]

		# 创建协议管理器
		dm_protocol = DamiaoProtocol(usb_hw, DmMotorManager(usb_hw=usb_hw))
		ht_protocol = HTProtocol(usb_hw, HTMotorManager(usb_hw=usb_hw))
		servo_protocol = ServoProtocol(usb_hw, ServoMotorManager(usb_hw=usb_hw))

		self.protocols = {
			MotorType.DAMIAO: dm_protocol,
			MotorType.HIGH_TORQUE: ht_protocol,
			MotorType.SERVO: servo_protocol,
		}

		# 添加电机到对应协议
		for motor_data in self.motors_data:
			self.protocols[motor_data.motor_type].add_motor(motor_data)

		self.motor_manager.add_damiao_protocol(dm_protocol)
		self.motor_manager.add_ht_protocol(ht_protocol)
		self.motor_manager.add_servo_protocol(servo_protocol)
		# 初始化状态变量
		motor_count = len(self.motors_data)
		self.motor_count = motor_count

		self.q = np.zeros(motor_count, dtype=np.float64)
		self.dq = np.zeros(motor_count, dtype=np.float64)
		self.ddq = np.zeros(motor_count, dtype=np.float64)
		self.tau = np.zeros(motor_count, dtype=np.float64)
		self.currents = np.zeros(motor_count, dtype=np.float64)
		self.positions = np.zeros(self.motor_count)
		self.velocities = np.zeros(self.motor_count)
		self.torques = np.zeros(self.motor_count) 
		self.q_prev = np.zeros(motor_count, dtype=np.float64)
		self.dq_prev = np.zeros(motor_count, dtype=np.float64)
		self.last_update_time = time.time()
		self.zero_pos = np.zeros(motor_count, dtype=np.float64)
		self.zero_vel = np.zeros(motor_count, dtype=np.float64)
		self.zero_tau = np.zeros(motor_count, dtype=np.float64)
		self._validate_internal_state()

		# 使能所有电机并初始化状态
		self.enable()
		self._refresh_all_states()

		# 重力补偿初始化
		self.gc_flag = gc
		if self.gc_flag:
			debug_print(f"初始化重力补偿系统，类型: {gc_type}")

			if gc_type == "mlp":
				# 使用MLP重力补偿
				try:
					from mlp_gravity_integrator import MLPGravityCompensation
					# 模型路径相对于IC_ARM.py的位置
					model_path = current_dir / "mlp_compensation" / "mlp_gravity_model_improved.pkl"
					self.gc = MLPGravityCompensation(
						model_path=str(model_path),
						enable_enhanced=True,
						debug=debug,
						max_torques=[15.0, 12.0, 12.0, 4.0, 4.0, 3.0]
					)
					debug_print("✅ MLP重力补偿初始化成功")
				except Exception as e:
					debug_print(f"❌ MLP重力补偿初始化失败: {e}，回退到静态补偿")
					from utils.static_gc import StaticGravityCompensation
					self.gc = StaticGravityCompensation()
					self.gc_type = "static"
			elif gc_type == "dyn":
				# 使用动力学重力补偿
				try:
					from minimum_gc import MinimumGravityCompensation
					self.gc = MinimumGravityCompensation()
					debug_print("✅ 动力学重力补偿初始化成功")
				except Exception as e:
					debug_print(f"❌ 动力学重力补偿初始化失败: {e}，回退到静态补偿")
					from utils.static_gc import StaticGravityCompensation
					self.gc = StaticGravityCompensation()
					self.gc_type = "static"
			else:
				# 使用原有的静态重力补偿
				from utils.static_gc import StaticGravityCompensation
				self.gc = StaticGravityCompensation()
				debug_print("✅ 静态重力补偿初始化成功")
		else:
			self.gc = None
			debug_print("重力补偿未启用")

		# 初始化异步日志管理器
		self.logger = AsyncLogManager(
			log_dir="/Users/lr-2002/project/instantcreation/IC_arm_control/logs",
			log_name="ic_arm_control",
			save_csv=True
		)
		self.logger.start()
		debug_print("✓ 异步日志系统已启动")

		# 状态缓存机制 - 减少冗余USB通信
		self._state_cache = {
			'q': np.zeros(motor_count, dtype=np.float64),
			'dq': np.zeros(motor_count, dtype=np.float64),
			'tau': np.zeros(motor_count, dtype=np.float64),
			'currents': np.zeros(motor_count, dtype=np.float64),
			'timestamp': 0,
			'valid': False
		}
		self._currents_cached = None  # 懒加载缓存
		self._last_state_refresh = 0
		self._state_refresh_interval = 0.001  # 1ms最小刷新间隔
		
		# 初始化缓冲控制组件
		self.safety_monitor = SafetyMonitor(motor_count=self.motor_count)
		self.buffer_control_thread = None
		
		if self.enable_buffered_control:
			self.buffer_control_thread = BufferControlThread(self, control_freq=self.control_freq)
			debug_print(f"✓ 缓冲控制组件已初始化 (频率: {self.control_freq}Hz)")
		else:
			debug_print("缓冲控制未启用，使用传统控制模式")

	def _validate_internal_state(self):
		"""验证内部状态变量的完整性"""
		expected_shape = (self.motor_count,)

		state_vars = {
			"q": self.q,
			"dq": self.dq,
			"ddq": self.ddq,
			"tau": self.tau,
			"currents": self.currents,
			"q_prev": self.q_prev,
			"dq_prev": self.dq_prev,
		}

		for name, var in state_vars.items():
			if not validate_array(var, expected_shape, name):
				raise ValueError(f"Invalid state variable {name}")

		debug_print("✓ 内部状态变量验证通过")

	# ========== LOW-LEVEL MOTOR READ FUNCTIONS ==========
	# 使用unified_motor_control接口读取电机状态

	def _read_motor_state(self, motor_id: int) -> dict:
		"""Read state from a single motor using unified interface"""
		return self.motor_manager.get_motor(motor_id).get_state()


	# ========== BATCH READ FUNCTIONS ==========
	# @pysnooper.snoop()
	def _read_all_states(self, refresh=True, enable_logging=True):
		"""Read all motor states using unified interface - optimized version"""
		# 方案1: 使用批量更新状态
		if refresh:
			self.motor_manager.update_all_states()

		# 方案2: 优化的循环 - 减少函数调用和字典访问
		motors = self.motor_manager.motors
		for i in range(self.motor_count):
			motor_id = i + 1
			motor = motors[motor_id]
			feedback = motor.feedback
			self.q[i] = feedback.position
			self.dq[i] = feedback.velocity
			self.tau[i] = feedback.torque

		# 记录电机状态到日志（仅在启用时）
		if (enable_logging and hasattr(self, 'logger') and
			self.logger.is_running):
			log_success = self.logger.log_motor_states(self.q, self.dq, self.tau)
			if not log_success:
				# 静默处理日志失败，避免调试输出影响性能
				pass

		return self.q, self.dq, self.tau

	def _read_all_states_fast(self, refresh=True):
		"""Read all motor states without logging - optimized for maximum FPS"""
		# 方案1: 使用批量更新状态
		if refresh:
			self.motor_manager.update_all_states()

		# 方案2: 优化的循环 - 减少函数调用和字典访问
		motors = self.motor_manager.motors
		for i in range(self.motor_count):
			motor_id = i + 1
			motor = motors[motor_id]
			feedback = motor.feedback
			self.q[i] = feedback.position
			self.dq[i] = feedback.velocity
			self.tau[i] = feedback.torque

		return self.q, self.dq, self.tau

	def _read_all_states_cached(self):
		"""Read all motor states with caching - 最小化USB通信"""
		current_time = time.time()

		# 检查缓存是否有效
		if (self._state_cache['valid'] and
			current_time - self._last_state_refresh < self._state_refresh_interval):
			# 返回缓存的数据
			self.q = self._state_cache['q'].copy()
			self.dq = self._state_cache['dq'].copy()
			self.tau = self._state_cache['tau'].copy()
			return self.q, self.dq, self.tau

		# 需要刷新状态
		self.motor_manager.update_all_states()

		# 优化循环 - 减少函数调用
		motors = self.motor_manager.motors
		for i in range(self.motor_count):
			motor_id = i + 1
			motor = motors[motor_id]
			feedback = motor.feedback
			self.q[i] = feedback.position
			self.dq[i] = feedback.velocity
			self.tau[i] = feedback.torque

		# 更新缓存
		self._state_cache['q'] = self.q.copy()
		self._state_cache['dq'] = self.dq.copy()
		self._state_cache['tau'] = self.tau.copy()
		self._state_cache['timestamp'] = current_time
		self._state_cache['valid'] = True
		self._last_state_refresh = current_time

		return self.q, self.dq, self.tau

	# ========== STATE UPDATE FUNCTIONS ==========
	# @pysnooper.snoop()
	def _refresh_all_states(self):
		"""Refresh all motor states using unified motor control system"""

		# 使用标准读取方法，包含日志记录
		self.q, self.dq, self.tau = self._read_all_states()
		self.currents = self.tau * 10.0  # 优化: 使用乘法代替除法

		self.last_update_time = time.time()

		# 添加调试输出 - 显示日志记录
		# if hasattr(self, 'logger') and self.logger.is_running:
		# 	print(f"[LOG] 电机状态已记录到日志系统")

	def _refresh_all_states_fast(self):
		"""快速状态刷新 - 跳过日志记录"""
		self.q, self.dq, self.tau = self._read_all_states_fast()
		self.currents = self.tau * 10.0  # 优化: 使用乘法代替除法
		self.last_update_time = time.time()
		# 调试输出
		# print(f"[FAST] 跳过日志记录，完成快速状态刷新")

	def _refresh_all_states_ultra_fast(self):
		"""超快速状态刷新 - 使用缓存和跳过所有非必要操作"""
		# 使用缓存方法读取状态，最小化USB通信
		self.q, self.dq, self.tau = self._read_all_states_cached()
		# 重置电流缓存
		self._currents_cached = None
		self.last_update_time = time.time()
		# 调试输出
		# print(f"[ULTRA] 使用缓存机制，完成超快速状态刷新")

	def _refresh_all_states_cached(self):
		"""缓存状态刷新 - 用于高频控制循环"""
		# 使用缓存方法读取状态
		self.q, self.dq, self.tau = self._read_all_states_cached()
		self.last_update_time = time.time()
		# 调试输出
		print(f"[CACHED] 使用缓存机制，最小化USB通信")

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

		# 懒加载计算currents，如果还没有计算的话
		if not hasattr(self, '_currents_cached') or self._currents_cached is None:
			self._currents_cached = self.tau * 10.0  # 优化: 使用乘法代替除法
		return self._currents_cached.copy()  # 返回内部维护的电流副本

	def get_complete_state(self) -> Dict[str, Union[np.ndarray, float]]:
		"""Get complete robot state"""
		self._refresh_all_states()
		return {
			"positions": self.q.copy(),
			"velocities": self.dq.copy(),
			"accelerations": self.ddq.copy(),
			"torques": self.tau.copy(),
			"currents": self.currents.copy(),
			"timestamp": self.last_update_time,
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
				"position": self.q[joint_index],
				"velocity": self.dq[joint_index],
				"acceleration": self.ddq[joint_index],
				"torque": self.tau[joint_index],
			}
		else:
			raise ValueError("Joint index must be 0-4")

	# ========== LOW-LEVEL WRITE FUNCTIONS ==========

	def _send_motor_command(
		self, motor_id, position_rad=0.0, velocity_rad_s=0.0, torque_nm=0.0, kp=None, kd=None
	):
		"""Send command to a single motor using unified interface"""
		motor = self.motor_manager.get_motor(motor_id)
		motor_info = self.motor_manager.get_motor_info(motor_id)
		if motor is None:
			return False
		# if motor_id == 7 or motor_id==8 :
		#     print("set_command", motor_id, position_rad, velocity_rad_s, motor_info.kp, motor_info.kd, torque_nm)
		if kp is None:
			kp = motor_info.kp
		if kd is None:
			kd = motor_info.kd
		return motor.set_command(
			position_rad, velocity_rad_s, kp, kd, torque_nm
		)

	# ========== PUBLIC WRITE INTERFACES ==========

	def set_joint_position(
		self, joint_index, position_rad, velocity_rad_s=0.0, torque_nm=0.0
	):
		"""Set position of a single joint using unified interface"""
		if joint_index < self.motor_count:
			return self._send_motor_command(
				joint_index + 1, position_rad, velocity_rad_s, torque_nm
			)
		return False

	def set_joint_torque(self, torques_nm):
		"""Set torques of all joints using unified interface"""
		if torques_nm is None:
			torques_nm = np.zeros(self.motor_count)

		success = True
		for i in range(min(self.motor_count, len(torques_nm))):
			result = self._send_motor_command(i + 1, 0, 0, torques_nm[i])
			success = success and result

		return success

	# def set_joint_positions_with_gc(self, positions_rad, velocities_rad_s=None):
	# 	tau = self.cal_gravity()
	# 	return self.set_joint_positions(positions_rad, velocities_rad_s, tau)

	def set_joint_positions(
		self, positions_rad, velocities_rad_s=None, torques_nm=None, enable_logging=True
	):
		"""Set positions of all joints - 支持缓冲控制模式"""
		if self.gc_flag and not torques_nm:
			torques_nm = self.cal_gravity()
		if velocities_rad_s is None:
			velocities_rad_s = np.zeros(self.motor_count)
		if torques_nm is None:
			torques_nm = np.zeros(self.motor_count)

		# 记录关节命令到日志（仅在启用时）
		if (enable_logging and hasattr(self, 'logger') and
			self.logger.is_running):
			self.logger.log_joint_command(
				np.array(positions_rad),
				np.array(velocities_rad_s),
				np.array(torques_nm)
			)

		# 检查是否启用缓冲控制
		if (self.enable_buffered_control and
			self.buffer_control_thread and
			self.buffer_control_thread.is_running()):
			# 缓冲控制模式：通过控制线程发送，立即返回
			self.buffer_control_thread.set_target_command(
				positions=np.array(positions_rad),
				velocities=np.array(velocities_rad_s),
				torques=np.array(torques_nm)
			)
			return True  # 立即返回，不阻塞
		else:
			# 传统控制模式：直接发送到硬件
			return self._original_set_joint_positions(positions_rad, velocities_rad_s, torques_nm)
	
	def _original_set_joint_positions(self, positions_rad, velocities_rad_s, torques_nm):
		"""原始的关节位置设置方法 - 直接发送到硬件"""
		success = True
		for i in range(min(self.motor_count, len(positions_rad))):
			result = self.set_joint_position(
				i, positions_rad[i], velocities_rad_s[i], torques_nm[i]
			)
			success = success and result
		return success

	# @pysnooper.snoop()
	def set_joint_positions_degrees(
		self, positions_deg, velocities_deg_s=None, torques_nm=None, enable_logging=True
	):
		"""Set positions of all joints in degrees"""
		positions_rad = np.radians(positions_deg)
		velocities_rad_s = (
			np.radians(velocities_deg_s) if velocities_deg_s is not None else None
		)
		return self.set_joint_positions(positions_rad, velocities_rad_s, torques_nm, enable_logging)

	# ========== MOTOR CONTROL FUNCTIONS ==========

	def enable_motor(self, joint_index):
		"""Enable a single motor using unified interface"""
		motor_id = joint_index + 1  # motor_id starts from 1
		try:
			success = self.motor_manager.enable_motor(motor_id)
			if success:
				debug_print(f"Motor {motor_id} enabled")
			else:
				debug_print(f"Failed to enable motor {motor_id}", "ERROR")
			return success
		except Exception as e:
			debug_print(f"Failed to enable motor {motor_id}: {e}", "ERROR")
			return False

	def disable_motor(self, joint_index):
		"""Disable a single motor using unified interface"""
		motor_id = joint_index + 1  # motor_id starts from 1
		try:
			success = self.motor_manager.disable_motor(motor_id)
			if success:
				debug_print(f"Motor {motor_id} disabled")
			else:
				debug_print(f"Failed to disable motor {motor_id}", "ERROR")
			return success
		except Exception as e:
			debug_print(f"Failed to disable motor {motor_id}: {e}", "ERROR")
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
				
				# 启动缓冲控制线程（如果启用）
				if self.enable_buffered_control:
					if not hasattr(self, 'buffer_control_thread') or self.buffer_control_thread is None:
						self.buffer_control_thread = BufferControlThread(self, control_freq=self.control_freq)
						debug_print("✓ 缓冲控制线程已创建")
					
					if not self.buffer_control_thread.is_running():
						self.buffer_control_thread.start()
						debug_print("✓ 缓冲控制线程已启动")
			else:
				debug_print("Failed to enable all motors", "ERROR")
			return success
		except Exception as e:
			debug_print(f"Failed to enable all motors: {e}", "ERROR")
			return False

	def disable_all_motors(self):
		"""Disable all motors using unified interface"""
		debug_print("Disabling all motors...")
		
		# 停止缓冲控制线程（如果运行中）
		if (hasattr(self, 'buffer_control_thread') and 
			self.buffer_control_thread and 
			self.buffer_control_thread.is_running()):
			self.buffer_control_thread.stop()
			debug_print("✓ 缓冲控制线程已停止")
		
		try:
			success = self.motor_manager.disable_all()
			if success:
				debug_print("All motors disabled successfully")
			else:
				debug_print("Failed to disable all motors", "ERROR")
			return success
		except Exception as e:
			debug_print(f"Failed to disable all motors: {e}", "ERROR")
			return False

	def emergency_stop(self):
		"""Emergency stop - disable all motors immediately"""
		print("EMERGENCY STOP!")
		
		# 立即设置安全监控器的紧急停止标志
		if hasattr(self, 'safety_monitor'):
			self.safety_monitor.set_emergency_stop(True)
			
		return self.disable_all_motors()

	# ========== 缓冲控制管理方法 ==========
	
	def enable_buffered_control_mode(self):
		"""启用缓冲控制模式"""
		if not self.enable_buffered_control:
			print("⚠️  缓冲控制未在初始化时启用")
			return False
			
		if not self.buffer_control_thread:
			self.buffer_control_thread = BufferControlThread(self, control_freq=self.control_freq)
			
		if not self.buffer_control_thread.is_running():
			success = self.buffer_control_thread.start()
			if success:
				debug_print("✅ 缓冲控制模式已启用")
			return success
		else:
			debug_print("缓冲控制模式已在运行")
			return True
	
	def disable_buffered_control_mode(self):
		"""禁用缓冲控制模式"""
		if (self.buffer_control_thread and 
			self.buffer_control_thread.is_running()):
			success = self.buffer_control_thread.stop()
			if success:
				debug_print("✅ 缓冲控制模式已禁用")
			return success
		else:
			debug_print("缓冲控制模式未运行")
			return True
	
	def get_buffered_control_status(self) -> dict:
		"""获取缓冲控制状态"""
		status = {
			'enabled': self.enable_buffered_control,
			'running': False,
			'statistics': None,
			'safety_status': None
		}
		
		if self.buffer_control_thread:
			status['running'] = self.buffer_control_thread.is_running()
			if status['running']:
				status['statistics'] = self.buffer_control_thread.get_statistics()
		
		if hasattr(self, 'safety_monitor'):
			status['safety_status'] = self.safety_monitor.get_safety_status()
			
		return status
	
	def reset_emergency_stop(self):
		"""重置紧急停止状态"""
		if hasattr(self, 'safety_monitor'):
			self.safety_monitor.set_emergency_stop(False)
			self.safety_monitor.reset_safety_violations()
			debug_print("✅ 紧急停止状态已重置")
			return True
		return False

	def home_to_zero(
		self, speed: float = 0.5, timeout: float = 30.0, frequency=100
	) -> bool:
		"""
		让机械臂主要关节平滑地回到零位（前8个电机，排除servo电机）

		Args:
			speed: 回零速度 (rad/s)，默认0.5 rad/s
			timeout: 超时时间 (秒)，默认30秒
			frequency: 控制频率 (Hz)，默认100Hz

		Returns:
			bool: 是否成功回零
		"""
		debug_print("开始执行回零操作（前8个电机，保持servo电机不动）...")

		try:
			# 获取所有电机的当前位置
			all_positions = self.get_joint_positions()
			if all_positions is None:
				debug_print("无法获取当前位置", "ERROR")
				return False

			# 只对前8个电机进行回零操作
			num_control_motors = len(MOTOR_LIST)  # 8个电机
			current_positions = all_positions[:num_control_motors]
			
			debug_print(f"控制电机: {MOTOR_LIST}")
			debug_print(
				f"当前位置: {[f'{MOTOR_LIST[i]}={np.degrees(current_positions[i]):.1f}°' for i in range(len(current_positions))]}"
			)
			
			# 如果有第9个电机（servo），显示但不控制
			if len(all_positions) > num_control_motors:
				servo_pos = all_positions[num_control_motors]
				debug_print(f"servo电机(m9)当前位置: {np.degrees(servo_pos):.1f}° (保持不动)")
			
			debug_print(f"回零操作将控制前{num_control_motors}个电机，servo电机保持当前位置")

			# 计算需要移动的距离和时间
			max_distance = max(abs(pos) for pos in current_positions)
			estimated_time = max_distance / speed

			debug_print(f"最大移动距离: {np.degrees(max_distance):.1f}°")
			debug_print(f"speed is {speed}")
			debug_print(f"预计回零时间: {estimated_time:.1f}秒")

			if estimated_time > timeout:
				debug_print(
					f"预计时间超过超时限制 ({timeout}s)，建议增加速度或超时时间",
					"WARNING",
				)

			# 生成平滑轨迹到零位
			num_steps = max(
				10, int(estimated_time * frequency)
			)  # 至少10步，或按100Hz计算
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

			# # 可视化轨迹
			# self._plot_trajectory_preview(
			#     trajectory_points, time_points, current_positions
			# )

			# 询问用户是否继续执行
			response = input("轨迹预览完成，是否继续执行? (y/n): ").lower().strip()
			if response != "y":
				debug_print("用户取消轨迹执行")
				return False

			start_time = time.time()

			for i in range(num_steps + 1):
				# 检查超时
				if time.time() - start_time > timeout:
					debug_print("回零操作超时", "ERROR")
					return False

				# 计算插值位置 (使用平滑的余弦插值)
				progress = i / num_steps
				smooth_progress = 0.5 * (
					1 - np.cos(np.pi * progress)
				)  # 余弦插值，起始和结束速度为0

				# 计算前8个电机的目标位置
				target_positions_control = current_positions * (1 - smooth_progress)
				
				# 构造完整的目标位置数组（包括servo电机的当前位置）
				if len(all_positions) > num_control_motors:
					# 保持servo电机在当前位置
					target_positions_full = np.zeros(len(all_positions))
					target_positions_full[:num_control_motors] = target_positions_control
					target_positions_full[num_control_motors:] = all_positions[num_control_motors:]
				else:
					target_positions_full = target_positions_control

				# 发送位置命令
				if self.gc_flag:
					success = self.set_joint_positions_with_gc(target_positions_full)
				else:
					success = self.set_joint_positions(target_positions_full)
				if not success:
					debug_print(f"发送位置命令失败 (步骤 {i})", "ERROR")
					return False

				# 显示进度
				current_pos = self.get_joint_positions(refresh=False)
				if current_pos is not None:
					# 只检查前8个电机的误差
					control_pos = current_pos[:num_control_motors]
					max_error = max(abs(pos) for pos in control_pos)
					debug_print(
						f"回零进度: {progress * 100:.0f}%, 最大偏差: {np.degrees(max_error):.2f}° (前{num_control_motors}个电机)"
					)

				# 等待下一步
				if i < num_steps:
					time.sleep(dt)

			# 验证回零结果
			# time.sleep(0.5)  # 等待稳定
			final_all_positions = self.get_joint_positions()

			if final_all_positions is not None:
				# 只验证前8个电机的回零结果
				final_control_positions = final_all_positions[:num_control_motors]
				max_error = max(abs(pos) for pos in final_control_positions)
				
				debug_print(
					f"回零完成! 控制电机最终位置: {[f'{MOTOR_LIST[i]}={np.degrees(final_control_positions[i]):.2f}°' for i in range(len(final_control_positions))]}"
				)
				
				# 显示servo电机位置（如果存在）
				if len(final_all_positions) > num_control_motors:
					servo_final_pos = final_all_positions[num_control_motors]
					debug_print(f"servo电机(m9)最终位置: {np.degrees(servo_final_pos):.2f}° (未控制)")
				
				debug_print(f"控制电机最大误差: {np.degrees(max_error):.2f}°")

				# 判断是否成功回零 (误差小于3度认为成功)
				if max_error < np.radians(3):
					debug_print("✓ 回零成功!", "INFO")
					return True
				else:
					debug_print(
						f"回零精度不足，最大误差: {np.degrees(max_error):.2f}°",
						"WARNING",
					)
					return False
			else:
				debug_print("无法验证回零结果", "ERROR")
				return False

		except Exception as e:
			debug_print(f"回零操作失败: {e}", "ERROR")
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
			fig.suptitle("IC ARM 回零轨迹预览", fontsize=16)

			motor_names = self.motor_names
			colors = ["red", "blue", "green", "orange", "purple", "pink"]

			# 绘制每个关节的位置轨迹
			for i in range(self.motor_count):  # 使用原有的NUM_MOTORS保持可视化兼容性
				row = i // 3
				col = i % 3
				ax = axes[row, col]

				# 位置轨迹（度）
				positions_deg = np.degrees(trajectory_array[:, i])
				ax.plot(
					time_points,
					positions_deg,
					color=colors[i],
					linewidth=2,
					label=f"{motor_names[i]} pos",
				)

				# 标记起始和结束点
				ax.plot(
					time_points[0],
					np.degrees(start_positions[i]),
					"ro",
					markersize=8,
					label="start point",
				)
				ax.plot(
					time_points[-1], 0, "go", markersize=8, label="target point(0°)"
				)

				ax.set_xlabel("time s")
				ax.set_ylabel("pos degress")
				ax.set_title(f"{motor_names[i]} ")
				ax.grid(True, alpha=0.3)
				ax.legend()

				# 添加数值信息
				start_deg = np.degrees(start_positions[i])
				ax.text(
					0.02,
					0.98,
					f"start: {start_deg:.1f}°\nend: 0.0°\nchagne: {-start_deg:.1f}°",
					transform=ax.transAxes,
					verticalalignment="top",
					bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
				)

			# 第6个子图：显示所有关节的综合信息
			ax_summary = axes[1, 2]

			# 计算每个时间点的总偏差
			total_deviation = np.sqrt(np.sum(trajectory_array**2, axis=1))
			ax_summary.plot(
				time_points,
				np.degrees(total_deviation),
				"black",
				linewidth=3,
				label="total error",
			)
			ax_summary.set_xlabel("time (s)")
			ax_summary.set_ylabel("total error (degress)")
			ax_summary.set_title("overview")
			ax_summary.grid(True, alpha=0.3)
			ax_summary.legend()

			# 添加进度信息
			max_deviation = np.degrees(np.max(total_deviation))
			ax_summary.text(
				0.02,
				0.98,
				f"max err: {max_deviation:.1f}°\n time: {time_points[-1]:.1f}s\npoints: {len(time_points)}",
				transform=ax_summary.transAxes,
				verticalalignment="top",
				bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
			)

			# plt.tight_layout()
			# plt.show(block=False)  # 非阻塞显示

			# 打印轨迹摘要
			debug_print("=== 轨迹预览摘要 ===")
			debug_print(f"轨迹时长: {time_points[-1]:.2f}s")
			debug_print(f"轨迹点数: {len(time_points)}")
			debug_print(f"更新频率: {len(time_points) / time_points[-1]:.1f} Hz")

			for i, name in enumerate(motor_names):
				start_deg = np.degrees(start_positions[i])
				debug_print(
					f"{name}: {start_deg:6.1f}° → 0.0° (变化: {-start_deg:6.1f}°)"
				)

			max_total_dev = np.degrees(np.max(total_deviation))
			debug_print(f"最大总偏差: {max_total_dev:.1f}°")
			debug_print("==================")

		except ImportError:
			debug_print("matplotlib未安装，跳过轨迹可视化", "WARNING")
			debug_print(
				"可以通过 pip install matplotlib 安装matplotlib来启用可视化功能", "INFO"
			)

			# 提供文本版本的轨迹预览
			debug_print("=== 文本版轨迹预览 ===")
			debug_print(f"轨迹时长: {time_points[-1]:.2f}s")
			debug_print(f"轨迹点数: {len(time_points)}")

			motor_names = self.motor_names
			for i, name in enumerate(motor_names):
				start_deg = np.degrees(start_positions[i])
				debug_print(
					f"{name}: {start_deg:6.1f}° → 0.0° (变化: {-start_deg:6.1f}°)"
				)
			debug_print("=====================")

		except Exception as e:
			debug_print(f"轨迹可视化失败: {e}", "ERROR")

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
				debug_print("无法获取当前位置", "ERROR")
				return False

			# 这里可以实现软件零位偏移逻辑
			# 由于DM电机的特性，我们主要通过记录偏移量来实现
			debug_print(
				f"当前位置已记录为零位: {[f'{np.degrees(pos):.2f}°' for pos in current_positions]}"
			)
			debug_print("注意: 这是软件零位，重启后需要重新设置", "WARNING")

			return True

		except Exception as e:
			debug_print(f"设置零位失败: {e}", "ERROR")
			return False

	@property
	def motors(self):
		return self.motor_manager.motors
	
	@property
	def motor_names(self):
		"""返回所有电机的名称列表（除了servo）"""
		# 返回前8个电机的名称（排除第9个servo电机）
		return [f"m{i+1}" for i in range(8)]

	def set_all_zero_positions(self) -> bool:
		"""
		设置所有关节的零点位置（排除servo电机）

		Returns:
			bool: 是否成功设置所有关节零点
		"""
		debug_print("设置所有关节零点位置（排除servo电机）...")

		try:
			# 获取当前所有关节位置
			current_positions = self.get_positions_degrees()
			if current_positions is None or len(current_positions) == 0:
				debug_print("无法获取当前位置", "ERROR")
				return False

			# 显示当前位置信息（只显示前8个电机）
			for i, name in enumerate(MOTOR_LIST):
				if i < len(current_positions):
					debug_print(
						f"{name}: 当前位置 {current_positions[i]:.2f}° 将设为零点"
					)

			# 只设置前8个电机的零点（排除servo电机）
			success_count = 0
			total_motors = len(MOTOR_LIST)
			
			for i in range(total_motors):
				motor_id = i + 1  # 电机ID从1开始
				motor = self.motor_manager.get_motor(motor_id)
				if motor is not None:
					try:
						if motor.set_zero():
							success_count += 1
							debug_print(f"✓ {MOTOR_LIST[i]} 零点设置成功")
						else:
							debug_print(f"⚠ {MOTOR_LIST[i]} 零点设置失败", "WARNING")
					except Exception as e:
						debug_print(f"⚠ {MOTOR_LIST[i]} 零点设置异常: {e}", "WARNING")
				else:
					debug_print(f"⚠ 无法获取电机 {MOTOR_LIST[i]}", "WARNING")

			success = success_count == total_motors
			if success:
				debug_print(f"✓ 所有关节零点设置成功 ({success_count}/{total_motors})")
			else:
				debug_print(f"⚠ 部分关节零点设置失败 ({success_count}/{total_motors})", "WARNING")

			return success

		except Exception as e:
			debug_print(f"设置所有零点失败: {e}", "ERROR")
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
				debug_print(f"无效的电机名称: {motor_name}", "ERROR")
				return False

			# 获取电机对象
			motor = self.motor_manager.get_motor(motor_id)
			if motor is None:
				debug_print(f"无法获取电机 {motor_name} (电机ID: {motor_id})", "ERROR")
				return False

			# 获取当前位置
			state = motor.get_state()
			current_pos = state["position"]
			if current_pos is None:
				debug_print(f"{motor_name}: 无法获取当前位置", "ERROR")
				return False

			# 显示当前位置
			debug_print(
				f"{motor_name}: 当前位置 {np.degrees(current_pos):.2f}° 将设为零点"
			)

			# 设置零点
			success = motor.set_zero()

			if success:
				debug_print(f"✓ {motor_name} 零点设置成功")
				debug_print("注意: 这是软件零位，重启后需要重新设置", "WARNING")
			else:
				debug_print(f"⚠ {motor_name} 零点设置失败", "ERROR")

			return success

		except Exception as e:
			debug_print(f"设置 {motor_name} 零点失败: {e}", "ERROR")
			return False

	def cal_gravity_full(self):
		return self.gc.calculate_torque(self.q, self.dq, self.ddq)

	def cal_gravity_coriolis(self):
		return self.gc.calculate_coriolis_torque(self.q, self.dq)

	def cal_gravity(self):
		"""计算重力补偿力矩"""
		if not self.gc_flag:
			return np.zeros(self.motor_count)

		if self.gc_type == "mlp":
			return self.cal_gravity_mlp()
		elif self.gc_type == "dyn":
			return self.cal_gravity_dyn()
		else:
			# 原有的静态重力补偿逻辑
			self._refresh_all_states_ultra_fast()
			return self.gc.get_gravity_compensation_torque(self.q)

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
					tau_compensation  # 重力补偿力矩
				)

				# 显示状态
				elapsed = time.time() - start_time
				# if int(elapsed * 10) % 10 == 0:  # 每0.1秒显示一次
				# pos_str = " ".join([f"{np.degrees(p):6.1f}°" for p in self.q])
				# vel_str = " ".join([f"{np.degrees(v):6.1f}°/s" for v in self.dq])
				tau_str = " ".join([f"{t:6.2f}" for t in tau_compensation])
				tau_real = self.get_joint_torques()
				tau_real_str = " ".join([f"{t:6.2f}" for t in tau_real])
				print(
					f"\n期望力矩: [{tau_str}]\n实际力矩: [{tau_real_str}]\n[{elapsed:6.1f}s]",
					end="",
					flush=True,
				)

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

	def cal_gravity_mlp(self):
		"""使用MLP计算重力补偿力矩"""
		if not self.gc_flag or self.gc_type != "mlp":
			return np.zeros(self.motor_count)

		try:
			self._refresh_all_states_ultra_fast()
			# MLP重力补偿只需要位置信息
			positions = self.q[:6]  # 前6个关节
			compensation_torque = self.gc.get_gravity_compensation_torque(positions)

			# 扩展到所有电机（保持与原有接口兼容）
			full_compensation = np.zeros(self.motor_count)
			full_compensation[:6] = compensation_torque

			return full_compensation
		except Exception as e:
			debug_print(f"MLP重力补偿计算失败: {e}", "ERROR")
			return np.zeros(self.motor_count)

	def cal_gravity_dyn(self):
		"""使用动力学模型计算重力补偿力矩"""
		if not self.gc_flag or self.gc_type != "dyn":
			return np.zeros(self.motor_count)

		try:
			self._refresh_all_states_ultra_fast()
			# 动力学重力补偿需要6个关节的位置、速度、加速度信息
			positions = self.q[:6]  # 前6个关节
			velocities = self.dq[:6]  # 前6个关节速度
			accelerations = self.ddq[:6]  # 前6个关节加速度

			# 使用MinimumGravityCompensation计算重力力矩
			compensation_torque = self.gc.calculate_gravity_torque(positions)

			# 扩展到所有电机（保持与原有接口兼容）
			full_compensation = np.zeros(self.motor_count)
			full_compensation[:6] = compensation_torque

			return full_compensation
		except Exception as e:
			debug_print(f"动力学重力补偿计算失败: {e}", "ERROR")
			return np.zeros(self.motor_count)

	def switch_to_mlp_gravity_compensation(self):
		"""切换到MLP重力补偿模式"""
		if not self.gc_flag:
			debug_print("重力补偿未启用", "ERROR")
			return False

		try:
			from mlp_gravity_integrator import MLPGravityCompensation
			model_path = Path(__file__).parent / "mlp_compensation" / "mlp_gravity_model_improved.pkl"

			self.gc = MLPGravityCompensation(
				model_path=str(model_path),
				enable_enhanced=True,
				debug=self.debug,
				max_torques=[15.0, 12.0, 12.0, 4.0, 4.0, 3.0]
			)
			self.gc_type = "mlp"
			debug_print("✅ 已切换到MLP重力补偿模式")
			return True
		except Exception as e:
			debug_print(f"切换到MLP重力补偿失败: {e}", "ERROR")
			return False

	def switch_to_dyn_gravity_compensation(self):
		"""切换到动力学重力补偿模式"""
		if not self.gc_flag:
			debug_print("重力补偿未启用", "ERROR")
			return False

		try:
			from minimum_gc import MinimumGravityCompensation
			self.gc = MinimumGravityCompensation()
			self.gc_type = "dyn"
			debug_print("✅ 已切换到动力学重力补偿模式")
			return True
		except Exception as e:
			debug_print(f"切换到动力学重力补偿失败: {e}", "ERROR")
			return False

	def switch_to_static_gravity_compensation(self):
		"""切换到静态重力补偿模式"""
		if not self.gc_flag:
			debug_print("重力补偿未启用", "ERROR")
			return False

		try:
			from utils.static_gc import StaticGravityCompensation
			self.gc = StaticGravityCompensation()
			self.gc_type = "static"
			debug_print("✅ 已切换到静态重力补偿模式")
			return True
		except Exception as e:
			debug_print(f"切换到静态重力补偿失败: {e}", "ERROR")
			return False

	def get_gravity_compensation_performance(self):
		"""获取重力补偿性能统计"""
		if not self.gc_flag or self.gc_type != "mlp":
			return None

		try:
			return self.gc.get_performance_stats()
		except Exception as e:
			debug_print(f"获取性能统计失败: {e}", "ERROR")
			return None

	def print_gravity_compensation_summary(self):
		"""打印重力补偿性能摘要"""
		if not self.gc_flag:
			print("重力补偿未启用")
			return

		print(f"=== 重力补偿系统状态 ===")
		print(f"类型: {self.gc_type}")
		print(f"状态: {'启用' if self.gc_flag else '禁用'}")

		if self.gc_type == "mlp":
			try:
				self.gc.print_performance_summary()
			except Exception as e:
				print(f"MLP性能统计获取失败: {e}")
		elif self.gc_type == "dyn":
			try:
				param_info = self.gc.get_parameter_info()
				print(f"动力学重力补偿信息:")
				print(f"  基参数数量: {param_info['num_base_params']}")
				print(f"  参数范围: [{param_info['param_range'][0]:.6f}, {param_info['param_range'][1]:.6f}]")
				print(f"  参数标准差: {param_info['param_std']:.6f}")
			except Exception as e:
				print(f"动力学重力补偿信息获取失败: {e}")

	def pseudo_gravity_compensation(
		self,
		update_rate=50.0,
		duration=None,
		kp_scale=1.0,
		kd_scale=1.0,
		enable_logging=True,
	):
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
			validate_type(update_rate, (int, float), "update_rate")
			validate_type(duration, (int, float, type(None)), "duration")
			validate_type(kp_scale, (int, float), "kp_scale")
			validate_type(kd_scale, (int, float), "kd_scale")

			if update_rate <= 0 or update_rate > 200:
				raise ValueError(f"update_rate应在(0, 200]范围内，当前: {update_rate}")
			if duration is not None and duration <= 0:
				raise ValueError(f"duration应为正数或None，当前: {duration}")
			if kp_scale <= 0 or kd_scale <= 0:
				raise ValueError(
					f"kp_scale和kd_scale应为正数，当前: kp={kp_scale}, kd={kd_scale}"
				)

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
						kp = config["kp"] * kp_scale
						kd = config["kd"] * kd_scale
						torque_ff = config["torque"]  # 前馈力矩保持不变

						# 发送位置命令（目标位置=当前位置）
						self.mc.controlMIT(
							motor, current_positions[i], 0.0, kp, kd, torque_ff
						)

					# 3. 统计和日志
					loop_count += 1

					# 计算位置变化
					if loop_count > 1:
						position_change = np.abs(current_positions - initial_positions)
						max_position_change = np.maximum(
							max_position_change, position_change
						)
						total_position_change += position_change

					# 定期日志输出
					if (
						enable_logging and (loop_start_time - last_log_time) >= 2.0
					):  # 每2秒输出一次
						elapsed = loop_start_time - start_time
						actual_freq = loop_count / elapsed if elapsed > 0 else 0

						debug_print(
							f"补偿运行中... 时间: {elapsed:.1f}s, 频率: {actual_freq:.1f}Hz"
						)
						debug_print(f"当前位置 (度): {np.degrees(current_positions)}")
						debug_print(f"最大偏移 (度): {np.degrees(max_position_change)}")

						last_log_time = loop_start_time

					# 4. 控制循环时序
					loop_duration = time.time() - loop_start_time
					sleep_time = dt - loop_duration

					if sleep_time > 0:
						time.sleep(sleep_time)
					elif enable_logging and loop_count % 100 == 0:  # 偶尔警告时序问题
						debug_print(
							f"警告: 控制循环超时 {loop_duration * 1000:.1f}ms > {dt * 1000:.1f}ms",
							"WARNING",
						)

				except KeyboardInterrupt:
					debug_print("用户中断，停止重力补偿")
					break
				except Exception as e:
					debug_print(f"控制循环异常: {e}", "ERROR")
					if enable_logging:
						debug_print(f"详细错误: {traceback.format_exc()}", "ERROR")
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
			debug_print(f"频率达成率: {(avg_freq / update_rate) * 100:.1f}%")
			debug_print(f"最大位置偏移 (度): {np.degrees(max_position_change)}")
			debug_print(f"平均位置偏移 (度): {np.degrees(avg_position_change)}")
			debug_print("===================")

			return True

		except KeyboardInterrupt:
			debug_print("用户中断重力补偿")
			return True
		except Exception as e:
			debug_print(f"重力补偿失败: {e}", "ERROR")
			debug_print(f"详细错误: {traceback.format_exc()}", "ERROR")
			return False
		finally:
			# 安全清理
			try:
				debug_print("清理资源...")
				# 可选择是否禁用电机，通常保持启用状态
				# self.disable_all_motors()
				debug_print("重力补偿模式结束")
			except Exception as e:
				debug_print(f"清理资源时出错: {e}", "ERROR")

	# @pysnooper.snoop()
	def monitor_positions_continuous(
		self, update_rate=10.0, duration=None, save_csv=False, csv_filename=None
	):
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

			csv_file = open(csv_filename, "w", newline="")
			csv_writer = csv.writer(csv_file)
			# 写入表头
			headers = (
				["timestamp", "time_s"]
				+ [f"m{i + 1}_pos_deg" for i in range(self.motor_count)]
				+ [f"m{i + 1}_vel_deg_s" for i in range(self.motor_count)]
			)
			csv_writer.writerow(headers)
			print(f"CSV文件: {csv_filename}")

		start_time = time.time()
		update_interval = 1.0 / update_rate

		try:
			while True:
				current_time = time.time()
				elapsed_time = current_time - start_time

				# # 检查是否超过监控时长
				# if duration and elapsed_time >= duration:
				#     print(f"\n监控时长达到 {duration} 秒，自动停止")
				#     break

				# 获取当前状态
				# self._refresh_all_states_ultra_fast()
				self._refresh_all_states()
				try:
					positions = self.get_positions_degrees(refresh=False)
					velocities = self.get_velocities_degrees(
						refresh=False
					)  # 使用已刷新的数据

					# 显示位置信息
					# print(positions, velocities)
					pos_str = " ".join([f"{pos:6.1f}°" for pos in positions])
					vel_str = " ".join([f"{vel:6.1f}°/s" for vel in velocities])

					print(
						f"\r[{elapsed_time:6.1f}s] 位置: [{pos_str}] 速度: [{vel_str}]",
						end="",
						flush=True,
					)

					if self.gc_flag:
						tau = self.cal_gravity()
						# tau = self.cal_gravity_coriolis()
						print(tau)


				except Exception as e:
					print(f"\n读取状态时出错: {e}")
					import traceback

					traceback.print_exc()
					continue

				# 等待下次更新
				# time.sleep(update_interval)

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
		print("=" * 80)
		print(
			f"{'Motor':<8} {'ID':<4} {'Position':<12} {'Velocity':<12} {'Torque':<12} {'Status':<10}"
		)
		print("-" * 80)

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
			print(
				f"{motor.info.name:<8} {motor_id:<4} {position:<12.4f} {velocity:<12.4f} {torque:<12.4f} {status:<10}"
			)

		print("=" * 80)
		print()

	def print_current_state(self):
		"""Print current robot state"""
		state = self.get_complete_state()

		print("\n" + "=" * 80)
		print("CURRENT ROBOT STATE")
		print("=" * 80)
		print(
			f"{'Joint':<8} {'Pos(deg)':<12} {'Vel(deg/s)':<12} {'Acc(deg/s²)':<15} {'Torque(Nm)':<12}"
		)
		print("-" * 80)

		for i in range(self.motor_count):
			print(
				f"m{i + 1:<7} {np.degrees(state['positions'][i]):<12.2f} "
				f"{np.degrees(state['velocities'][i]):<12.2f} "
				f"{np.degrees(state['accelerations'][i]):<15.2f} "
				f"{state['torques'][i]:<12.3f}"
			)

		print("=" * 80)
		print(f"Timestamp: {state['timestamp']:.3f}")
		print()

	# ========== TRAJECTORY EXECUTION ==========

	def _generate_linear_interpolation(self, start_positions_deg, end_positions_deg, duration_sec):
		"""
		Generate linear interpolation trajectory between two positions
		
		Args:
			start_positions_deg: Starting positions in degrees
			end_positions_deg: Ending positions in degrees  
			duration_sec: Duration of interpolation in seconds
			frequency_hz: Control frequency in Hz
			
		Returns:
			List of trajectory points [pos1, pos2, ..., posN, timestamp]
		"""
		frequency_hz = self.control_freq
		start_positions_deg = np.array(start_positions_deg)
		end_positions_deg = np.array(end_positions_deg)
		
		num_steps = int(duration_sec * frequency_hz)
		if num_steps < 1:
			num_steps = 1
			
		trajectory_points = []
		
		for i in range(num_steps + 1):  # +1 to include the final point
			t = i / num_steps  # Normalized time from 0 to 1
			
			# Linear interpolation
			current_positions = start_positions_deg + t * (end_positions_deg - start_positions_deg)
			timestamp = i / frequency_hz
			
			# Create trajectory point [pos1, pos2, ..., posN, timestamp]
			point = list(current_positions) + [timestamp]
			trajectory_points.append(point)
			
		return trajectory_points

	def execute_trajectory_points(self, trajectory_points, verbose=True,
							   smooth_start=True, smooth_end=True,
							   transition_duration=4.0, enable_logging=True):
		"""
		Execute a trajectory given as a list of points with smooth transitions

		Args:
			trajectory_points: List of points, each point is [pos1_deg, pos2_deg, ..., posN_deg, timestamp]
			verbose: Whether to print progress information
			smooth_start: Whether to add smooth transition from current position to trajectory start
			smooth_end: Whether to add smooth transition from trajectory end to zero position
			transition_duration: Duration of smooth transitions in seconds
			enable_logging: Whether to enable logging during execution (default: False for performance)

		Returns:
			bool: True if execution successful, False otherwise
		"""
		if not trajectory_points:
			if verbose:
				print("Empty trajectory")
			return False

		# 启用所有电机
		if verbose:
			print("启用所有电机...")
		self.enable_all_motors()

		success = True
		complete_trajectory = []

		try:
			# 临时禁用日志记录以提高性能
			logging_backup = hasattr(self, 'logger') and self.logger.is_running
			if not enable_logging and logging_backup:
				if verbose:
					print("临时禁用日志记录以提高性能...")
				# 记录当前日志状态，但不在轨迹执行中记录

			# 获取当前位置（使用快速读取）
			current_positions_deg = self.get_positions_degrees()
			if verbose:
				print(f"当前位置: {[f'{p:.1f}°' for p in current_positions_deg]}")

			# 获取轨迹起始位置
			first_point = trajectory_points[0]
			trajectory_start_positions = first_point[:-1]  # 除时间戳外的位置

			# 1. 添加平滑开始轨迹（从当前位置到轨迹起始位置）
			start_trajectory = []
			if smooth_start:
				if verbose:
					print(f"生成平滑开始轨迹: {transition_duration}秒")
					print(f"从 {[f'{p:.1f}°' for p in current_positions_deg]} 到 {[f'{p:.1f}°' for p in trajectory_start_positions]}")

				start_trajectory = self._generate_linear_interpolation(
					current_positions_deg, trajectory_start_positions, transition_duration
				)
				complete_trajectory.extend(start_trajectory)

			# 2. 添加主轨迹（调整时间戳）
			time_offset = transition_duration if smooth_start else 0.0
			for point in trajectory_points:
				adjusted_point = point[:-1] + [point[-1] + time_offset]  # 调整时间戳
				complete_trajectory.append(adjusted_point)

			# 3. 添加平滑结束轨迹（从轨迹结束位置到零点）
			end_trajectory = []
			if smooth_end:
				last_point = trajectory_points[-1]
				trajectory_end_positions = last_point[:-1]  # 除时间戳外的位置
				zero_positions = [0.0] * len(trajectory_end_positions)  # 零点位置

				if verbose:
					print(f"生成平滑结束轨迹: {transition_duration}秒")
					print(f"从 {[f'{p:.1f}°' for p in trajectory_end_positions]} 到 {[f'{p:.1f}°' for p in zero_positions]}")

				# 计算结束轨迹的时间偏移
				final_time_offset = time_offset + trajectory_points[-1][-1]
				end_trajectory = self._generate_linear_interpolation(
					trajectory_end_positions, zero_positions, transition_duration
				)

				# 调整结束轨迹的时间戳（跳过第一个点避免重复）
				for i, point in enumerate(end_trajectory[1:], 1):  # 跳过第一个点
					adjusted_point = point[:-1] + [point[-1] + final_time_offset]
					complete_trajectory.append(adjusted_point)

			# 执行完整轨迹
			total_points = len(complete_trajectory)
			if verbose:
				print(f"执行完整轨迹: {total_points} 个轨迹点")
				print(f"  - 平滑开始: {len(start_trajectory) if smooth_start else 0} 点")
				print(f"  - 主轨迹: {len(trajectory_points)} 点")
				print(f"  - 平滑结束: {len(end_trajectory)-1 if smooth_end else 0} 点")
				if not enable_logging:
					print("  - 日志记录: 已禁用（高性能模式）")

			start_time = time.time()
			log_counter = 0

			for i, point in enumerate(complete_trajectory):
				# 提取位置和时间
				target_positions_deg = point[:-1]
				target_time = point[-1]

				# 发送位置命令（在轨迹执行期间禁用日志记录）
				self.set_joint_positions_degrees(target_positions_deg, enable_logging=True)

				# 有限的状态读取和日志记录（避免性能问题）
				# if enable_logging and log_counter % 20 == 0:  # 每20个点记录一次
				self._read_all_states(enable_logging=True)
				# else:
				# 	# 使用快速状态读取，跳过日志记录
				# 	self._read_all_states_fast()

				log_counter += 1

				# 进度报告
				if verbose and i % 50 == 0:
					progress = (i / total_points) * 100
					elapsed = time.time() - start_time
					avg_fps = i / elapsed if elapsed > 0 else 0
					timestamp = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
					print(f"[{timestamp}]执行进度: {progress:.1f}% ({i}/{total_points}) - 平均FPS: {avg_fps:.1f}")

		except KeyboardInterrupt:
			if verbose:
				print("\n轨迹执行被用户中断")
			success = False
		except Exception as e:
			import traceback
			if verbose:
				print(f"轨迹执行错误: {e}")
				traceback.print_exc()
			success = False
		finally:
			if verbose:
				print("禁用所有电机...")
			self.disable_all_motors()

			# 恢复日志记录状态
			if not enable_logging and logging_backup:
				if verbose:
					print("恢复日志记录...")
				# 日志记录会在下次状态读取时自动恢复

		if verbose:
			final_pos = self.get_positions_degrees()
			print(f"轨迹执行完成. 最终位置: {[f'{p:.1f}°' for p in final_pos]}")

		return success

	# ========== CLEANUP ==========

	def close(self):
		"""Close the connection and cleanup"""
		try:
			# 停止缓冲控制线程
			if (hasattr(self, 'buffer_control_thread') and 
				self.buffer_control_thread and 
				self.buffer_control_thread.is_running()):
				self.buffer_control_thread.stop()
				debug_print("✓ 缓冲控制线程已关闭")
			
			# 停止日志系统
			if hasattr(self, 'logger') and self.logger.is_running:
				self.logger.stop()
				debug_print("✓ 日志系统已关闭")
				
			# 禁用所有电机
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

	def test_fps_performance(self, duration=5.0, method='ultra_fast'):
		"""
		测试不同状态刷新方法的FPS性能

		Args:
			duration: 测试时长(秒)
			method: 测试方法 ('normal', 'fast', 'ultra_fast', 'cached')

		Returns:
			dict: 性能统计信息
		"""
		import time

		print(f"=== FPS性能测试 ===")
		print(f"测试方法: {method}")
		print(f"测试时长: {duration}秒")
		print("请确保电机已启用并安全...")

		method_map = {
			'normal': self._refresh_all_states,
			'fast': self._refresh_all_states_fast,
			'ultra_fast': self._refresh_all_states_ultra_fast,
			'cached': self._refresh_all_states_cached
		}

		if method not in method_map:
			print(f"无效的测试方法: {method}")
			return None

		refresh_func = method_map[method]

		# 预热
		print("预热中...")
		for _ in range(10):
			refresh_func()
		time.sleep(0.1)

		# 开始测试
		print("开始性能测试...")
		start_time = time.time()
		count = 0

		try:
			while time.time() - start_time < duration:
				refresh_func()
				count += 1

		except KeyboardInterrupt:
			print("测试被用户中断")

		end_time = time.time()
		actual_duration = end_time - start_time
		fps = count / actual_duration

		print(f"=== 测试结果 ===")
		print(f"测试方法: {method}")
		print(f"实际时长: {actual_duration:.3f}秒")
		print(f"调用次数: {count}")
		print(f"平均FPS: {fps:.1f} Hz")
		print(f"平均调用间隔: {1000.0/fps:.2f} ms")
		print("================")

		return {
			'method': method,
			'duration': actual_duration,
			'count': count,
			'fps': fps,
			'avg_interval_ms': 1000.0/fps
		}

	def benchmark_all_methods(self, duration=3.0):
		"""
		基准测试所有状态刷新方法

		Args:
			duration: 每个方法的测试时长(秒)

		Returns:
			dict: 所有方法的性能比较结果
		"""
		methods = ['normal', 'fast', 'ultra_fast', 'cached']
		results = {}

		print(f"=== 全面的FPS性能基准测试 ===")
		print(f"每个方法测试时长: {duration}秒")
		print("========================")

		for method in methods:
			result = self.test_fps_performance(duration, method)
			if result:
				results[method] = result
			print()  # 空行分隔

		# 性能比较
		print("=== 性能比较 ===")
		if results:
			best_method = max(results.items(), key=lambda x: x[1]['fps'])
			worst_method = min(results.items(), key=lambda x: x[1]['fps'])

			print(f"最快方法: {best_method[0]} ({best_method[1]['fps']:.1f} Hz)")
			print(f"最慢方法: {worst_method[0]} ({worst_method[1]['fps']:.1f} Hz)")
			print(f"性能提升: {best_method[1]['fps']/worst_method[1]['fps']:.1f}x")

			for method, result in results.items():
				print(f"{method:12s}: {result['fps']:6.1f} Hz ({result['avg_interval_ms']:5.2f} ms)")

		return results


# ========== EXAMPLE USAGE ==========
if __name__ == "__main__":
	# Example usage
	arm = ICARM(debug=False, gc=True, control_freq=300.0, gc_only=True)
	# arm.connect()
	try:
		# Test single joint movement
		# print("Testing single joint movement...")
		# arm.enable_all_motors()
		# succes = arm.home_to_zero(speed=0.3, timeout=30.0)
		# # Move joint 0 to 30 degrees
		# # arm.set_joint_positions_degrees([30, 0, 0, 0, 0])
		# # time.sleep(2)
		# arm.switch_to_dyn_gravity_compensation()
		while True:
			
			tau = arm.cal_gravity()
			pos = arm.get_joint_positions()
			arm.set_joint_torque(np.array(tau)*0.8)
			# arm.set_joint_positions(positions_rad=pos, torques_nm=tau)
			# print('all info is ', arm._read_all_states(refresh=False))
			print('predicted tau is ', tau)
   
		# # Read state again
		# arm.print_current_state()

		# # # Return to zero
		# # arm.set_joint_positions_degrees([0, 0, 0, 0, 0])
		# time.sleep(2)

	except KeyboardInterrupt:
		print("Interrupted by user")
	finally:
		arm.close()
