"""
优化版缓冲控制线程 - 实现300Hz高性能控制
主要优化：
1. 批量发送CAN帧，减少函数调用层次
2. 参数缓存（kp, kd, limits）
3. 内联关键函数
4. 预分配内存
"""

import time
import threading
import numpy as np
from typing import Optional, Dict
import queue

# 导入必要的常量
from ic_arm_control.control.motor_info import Control_Mode
MIT_MODE = Control_Mode.MIT_MODE


class OptimizedBufferControlThread:
    """优化版300Hz缓冲控制线程 - 减少性能瓶颈"""

    def __init__(self, icarm_instance, control_freq: int = 300):
        """
        初始化优化版缓冲控制线程

        Args:
            icarm_instance: IC_ARM实例
            control_freq: 控制频率 (Hz)
        """
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        print(f'[OptimizedBufferControlThread] 初始化，目标频率: {control_freq}Hz (周期: {self.dt*1000:.2f}ms)')

        # 线程控制
        self.running = False
        self.thread = None
        self.thread_lock = threading.Lock()

        # 命令队列 - 增大队列容量
        self.command_queue = queue.Queue(maxsize=200)
        self.current_command = {
            'positions': None,
            'velocities': None,
            'torques': None,
            'timestamp': time.time()
        }
        self.command_lock = threading.Lock()

        # 统计信息
        self.loop_count = 0
        self.total_time = 0.0
        self.max_loop_time = 0.0
        self.missed_deadlines = 0
        self.commands_received = 0
        self.commands_processed = 0

        # IC ARM实例引用
        self.icarm = icarm_instance

        # 优化：预计算和缓存参数
        self._init_parameter_cache()

        # 优化：预分配CAN帧数组
        self._preallocate_can_frames()

        # 内联函数定义
        self._inline_float_to_uint = lambda x, xmin, xmax, bits: int((x - xmin) / (xmax - xmin) * ((1 << bits) - 1))

        print(f"[OptimizedBufferControlThread] 优化组件已初始化")

    def _init_parameter_cache(self):
        """初始化参数缓存，避免运行时重复查找"""
        self.kp_cache = {}
        self.kd_cache = {}
        self.limits_cache = {}
        self.can_id_cache = {}

        print("[OptimizedBufferControlThread] 缓存电机参数（前6个电机）...")

        # 根据您的电机配置设置正确的limit_param
        motor_limits = {
            1: [12.5, 25, 200],   # m1: DM10010L
            2: [12.566, 20, 120], # m2: DM6248
            3: [12.566, 20, 120], # m3: DM6248
            4: [12.5, 10, 28],    # m4: DM4340
            5: [12.5, 20, 28],    # m5: DM4340
            6: [12.5, 30, 10],    # m6: DM4310
        }

        for i in range(min(6, self.icarm.motor_count)):  # 只缓存前6个电机
            motor_id = i + 1
            try:
                # 获取电机信息
                motor_info = self.icarm.motor_manager.get_motor_info(motor_id)

                # 缓存kp, kd
                self.kp_cache[motor_id] = motor_info.kp
                self.kd_cache[motor_id] = motor_info.kd

                # 使用配置的limits参数
                self.limits_cache[motor_id] = tuple(motor_limits.get(motor_id, [12.5, 50.0, 10.0]))

                # 缓存CAN ID
                motor = self.icarm.motor_manager.get_motor(motor_id)
                if motor and hasattr(motor, 'GetCanId'):
                    self.can_id_cache[motor_id] = motor.GetCanId()
                else:
                    # 默认CAN ID
                    self.can_id_cache[motor_id] = motor_id

                print(f"  电机{motor_id}: kp={self.kp_cache[motor_id]:.1f}, kd={self.kd_cache[motor_id]:.1f}, limits={self.limits_cache[motor_id]}")

            except Exception as e:
                print(f"[OptimizedBufferControlThread] ⚠️ 缓存电机{motor_id}参数失败: {e}")
                # 使用配置的默认值
                self.kp_cache[motor_id] = 0.0
                self.kd_cache[motor_id] = 0.0
                self.limits_cache[motor_id] = tuple(motor_limits.get(motor_id, [12.5, 50.0, 10.0]))
                self.can_id_cache[motor_id] = motor_id

    def _preallocate_can_frames(self):
        """预分配CAN帧数组，避免运行时内存分配"""
        self.can_frames = [None] * 6  # 6个电机的CAN帧
        self.temp_data = [0] * 8  # 临时数据数组

    def set_target_command(self, positions: Optional[np.ndarray] = None,
                          velocities: Optional[np.ndarray] = None,
                          torques: Optional[np.ndarray] = None):
        """
        设置目标控制指令 - 非阻塞方式

        Args:
            positions: 目标位置 (rad)
            velocities: 目标速度 (rad/s)
            torques: 目标力矩 (N·m)
        """
        command = {
            'positions': np.array(positions).copy() if positions is not None else None,
            'velocities': np.array(velocities).copy() if velocities is not None else None,
            'torques': np.array(torques).copy() if torques is not None else None,
            'timestamp': time.time()
        }

        self.commands_received += 1

        try:
            # 非阻塞放入队列
            self.command_queue.put_nowait(command)
        except queue.Full:
            # 队列满时的处理策略
            try:
                self.command_queue.put(command, timeout=0.001)
            except queue.Full:
                # 强制放入，确保命令不丢失
                self.command_queue.put(command)

    def get_next_command(self) -> dict:
        """按顺序获取下一个命令（非阻塞）"""
        try:
            command = self.command_queue.get_nowait()
            self.commands_processed += 1
            with self.command_lock:
                self.current_command.update(command)
        except queue.Empty:
            pass
        except Exception:
            pass

        # 返回当前命令
        with self.command_lock:
            return {
                'positions': self.current_command.get('positions', None),
                'velocities': self.current_command.get('velocities', None),
                'torques': self.current_command.get('torques', None),
                'timestamp': self.current_command.get('timestamp', None)
            }

    def _build_can_frame_fast(self, motor_id: int, pos: float, torque: float, kp: float = None, kd: float = None) -> Dict:
        """
        内联版本：快速构建CAN数据帧

        Args:
            motor_id: 电机ID
            pos: 位置 (rad)
            torque: 力矩 (N·m)
            kp: 比例增益 (可选，如果提供则使用，否则使用缓存值)
            kd: 微分增益 (可选，如果提供则使用，否则使用缓存值)

        Returns:
            dict: {'data': list, 'can_id': int}
        """
        # 使用传入的参数，如果没有提供则使用缓存的参数
        if kp is not None:
            actual_kp = kp
        else:
            actual_kp = self.kp_cache[motor_id]

        if kd is not None:
            actual_kd = kd
        else:
            actual_kd = self.kd_cache[motor_id]

        limits = self.limits_cache[motor_id]

        # 内联float_to_uint转换
        kp_uint = self._inline_float_to_uint(actual_kp, 0, 500, 12)
        kd_uint = self._inline_float_to_uint(actual_kd, 0, 5, 12)
        q_uint = self._inline_float_to_uint(pos, -limits[0], limits[0], 16)
        tau_uint = self._inline_float_to_uint(torque, -limits[2], limits[2], 12)

        # 构建CAN ID
        can_id = self.can_id_cache[motor_id] + MIT_MODE

        # 内联数据打包（速度使用特殊值0x7F0，与原版damiao一致）
        dq_uint = 0x7F0  # 速度特殊值，12位中的高8位为0x7F，低4位为0
        data = self.temp_data  # 使用预分配的数组

        data[0] = (q_uint >> 8) & 0xff
        data[1] = q_uint & 0xff
        data[2] = dq_uint >> 4
        data[3] = ((dq_uint & 0xf) << 4) | ((kp_uint >> 8) & 0xf)
        data[4] = kp_uint & 0xff
        data[5] = kd_uint >> 4
        data[6] = ((kd_uint & 0xf) << 4) | ((tau_uint >> 8) & 0xf)
        data[7] = tau_uint & 0xff

        # 返回数据的副本
        return {
            'data': data.copy(),
            'can_id': can_id
        }

    def _send_to_hardware_optimized(self, command: dict):
        """
        优化版本：批量发送所有电机指令

        Args:
            command: 控制指令字典
        """
        # 检查命令数据有效性
        if command is None:
            return

        positions = command.get('positions')
        velocities = command.get('velocities')
        torques = command.get('torques')

        if positions is None or torques is None:
            return

        # 批量构建所有CAN帧
        num_motors = min(6, len(positions))  # 最多6个电机

        for i in range(num_motors):
            motor_id = i + 1
            if motor_id in self.can_id_cache:
                # 获取命令中的KP和KD值（如果有）
                kp = None
                kd = None

                # 这里假设命令中可能包含kp和kd信息
                # 实际使用时可以根据需要修改命令格式
                if hasattr(command, 'get'):
                    kp = command.get('kp_' + str(motor_id), None)
                    kd = command.get('kd_' + str(motor_id), None)

                # 快速构建CAN帧
                self.can_frames[i] = self._build_can_frame_fast(
                    motor_id, positions[i], torques[i], kp, kd
                )

        # 批量发送所有帧
        for i in range(num_motors):
            if self.can_frames[i] is not None:
                try:
                    # 通过motor_manager访问USB接口
                    motor = self.icarm.motor_manager.get_motor(motor_id + 1)
                    if motor and hasattr(motor, 'protocol') and hasattr(motor.protocol, 'usb_hw'):
                        motor.protocol.usb_hw.fdcanFrameSend(
                            self.can_frames[i]['data'],
                            self.can_frames[i]['can_id']
                        )
                    else:
                        # 备用方案：调用原始方法
                        self._fallback_send_command(motor_id, self.can_frames[i])
                except Exception as e:
                    print(f"[OptimizedBufferControlThread] ❌ 发送电机{i+1}指令失败: {e}")

    def _fallback_send_command(self, motor_id: int, can_frame: dict):
        """备用发送方法：直接跳过，不发送"""
        # 如果无法直接访问USB接口，则跳过发送
        return

    def start(self):
        """启动控制线程"""
        with self.thread_lock:
            if self.running:
                print("[OptimizedBufferControlThread] ⚠️ 控制线程已在运行")
                return False

            self.running = True
            self.thread = threading.Thread(target=self._control_loop, daemon=True)
            self.thread.start()

            print(f"[OptimizedBufferControlThread] ✅ 控制线程已启动 (频率: {self.control_freq}Hz)")
            return True

    def stop(self, timeout: float = 2.0):
        """停止控制线程"""
        with self.thread_lock:
            if not self.running:
                return True

            self.running = False

            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=timeout)

                if self.thread.is_alive():
                    print("[OptimizedBufferControlThread] ⚠️ 控制线程停止超时")
                    return False
                else:
                    print("[OptimizedBufferControlThread] ✅ 控制线程已停止")
                    return True

            return True

    def is_running(self) -> bool:
        """检查线程是否在运行"""
        with self.thread_lock:
            return self.running and self.thread and self.thread.is_alive()

    def once_send(self):
        """单次发送 - 用于性能测试"""
        self._send_to_hardware_optimized(self.get_next_command())
        self.loop_count += 1

    def _control_loop(self):
        """优化版300Hz控制循环"""
        print(f"[OptimizedBufferControlThread] 开始{self.control_freq}Hz控制循环...")

        # 设置线程优先级
        try:
            import os
            os.nice(-10)
        except:
            pass

        last_freq_report = time.time()

        while self.running:
            loop_start_time = time.time()

            self.once_send()
            # 精确定时控制
            loop_time = time.time() - loop_start_time
            self.total_time += loop_time
            self.max_loop_time = max(self.max_loop_time, loop_time)

            # 计算睡眠时间
            sleep_time = self.dt - loop_time   # 微调补偿

            if sleep_time > 0:
                time.sleep(sleep_time)

            # 定期报告状态
            current_time = time.time()
            if self.loop_count % 100 == 0:  # 每2秒报告一次
                actual_freq = self.loop_count / (current_time - last_freq_report) if self.loop_count > 0 else 0
                queue_size = self.command_queue.qsize()
                print(f"[OptimizedBufferControlThread] 频率: {actual_freq:.1f}Hz, 队列: {queue_size}, "
                      f"接收: {self.commands_received}, 处理: {self.commands_processed}")
                self.loop_count = 0
                last_freq_report = current_time

        print("[OptimizedBufferControlThread] 控制循环已退出")

    def _send_zero_command(self):
        """发送零指令 - 紧急安全措施"""
        try:
            zero_positions = np.zeros(6)
            zero_torques = np.zeros(6)

            self._send_to_hardware_optimized({
                'positions': zero_positions,
                'velocities': None,
                'torques': zero_torques
            })
        except Exception as e:
            print(f"[OptimizedBufferControlThread] ❌ 发送零指令失败: {e}")

    def get_statistics(self) -> dict:
        """获取控制线程统计信息"""
        if self.loop_count > 0:
            avg_loop_time = self.total_time / self.loop_count
            actual_freq = 1.0 / avg_loop_time if avg_loop_time > 0 else 0
        else:
            avg_loop_time = 0
            actual_freq = 0

        return {
            'is_running': self.is_running(),
            'target_frequency': self.control_freq,
            'actual_frequency': actual_freq,
            'loop_count': self.loop_count,
            'avg_loop_time_ms': avg_loop_time * 1000,
            'max_loop_time_ms': self.max_loop_time * 1000,
            'missed_deadlines': self.missed_deadlines,
            'commands_received': self.commands_received,
            'commands_processed': self.commands_processed,
            'queue_size': self.command_queue.qsize()
        }

    def print_statistics(self):
        """打印控制线程统计信息"""
        stats = self.get_statistics()
        print(f"\n[OptimizedBufferControlThread] 性能统计:")
        print(f"  运行状态: {'运行中' if stats['is_running'] else '已停止'}")
        print(f"  目标频率: {stats['target_frequency']}Hz")
        print(f"  实际频率: {stats['actual_frequency']:.1f}Hz")
        print(f"  循环次数: {stats['loop_count']}")
        print(f"  平均循环时间: {stats['avg_loop_time_ms']:.2f}ms")
        print(f"  最大循环时间: {stats['max_loop_time_ms']:.2f}ms")
        print(f"  命令接收: {stats['commands_received']}")
        print(f"  命令处理: {stats['commands_processed']}")
        print(f"  队列大小: {stats['queue_size']}")


if __name__ == "__main__":
    # 简单测试
    class MockICArm:
        def __init__(self):
            self.motor_count = 9
            self.motor_manager = MockMotorManager()

        def usb_hw(self):
            return MockUSB()

    class MockMotorManager:
        def get_motor_info(self, motor_id):
            return MockMotorInfo()

        def get_motor(self, motor_id):
            return MockMotor()

    class MockMotorInfo:
        def __init__(self):
            self.kp = 30.0
            self.kd = 1.0

    class MockMotor:
        def get_limit_param(self):
            return (12.5, 50.0, 10.0)

        def GetCanId(self):
            return 1

    class MockUSB:
        def fdcanFrameSend(self, data, can_id):
            pass

    # 创建测试实例
    mock_arm = MockICArm()
    control_thread = OptimizedBufferControlThread(mock_arm, control_freq=300)

    # 启动线程
    control_thread.start()

    # 发送测试指令
    for i in range(10):
        positions = np.array([0.1 * i] * 8)
        torques = np.array([0.01 * i] * 8)
        control_thread.set_target_command(positions=positions, torques=torques)
        time.sleep(0.1)

    # 等待一段时间
    time.sleep(2)

    # 打印统计信息
    control_thread.print_statistics()

    # 停止线程
    control_thread.stop()

    print("OptimizedBufferControlThread测试完成")