"""
提供300Hz固定频率的机械臂控制循环
"""

import time
import threading
import numpy as np
from typing import Optional, Callable
import queue
import pysnooper 

class BufferControlThread:
    """500Hz缓冲控制线程 - 提供固定频率的实时控制"""

    def __init__(self, icarm_instance, control_freq: int = 500):
        """
        初始化缓冲控制线程
        
        Args:
            icarm_instance: IC_ARM实例
            control_freq: 控制频率 (Hz)
        """
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        print(' buffer control thread dt is ', self.dt)
        # 线程控制
        self.running = False
        self.thread = None
        self.thread_lock = threading.Lock()
        
        # 命令队列 - 线程安全的命令缓冲（增大队列容量）
        self.command_queue = queue.Queue(maxsize=100)  # 增大到100个命令缓冲
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
        self.commands_received = 0  # 接收到的命令总数
        self.commands_processed = 0  # 处理的命令总数
        self.icarm =icarm_instance
        # 预计算常用数组，减少运行时分配
        self._zero_velocities = np.zeros(self.icarm.motor_count)
        self._zero_torques = np.zeros(self.icarm.motor_count)
        
        print(f"[BufferControlThread] 控制线程已初始化，频率: {control_freq}Hz, 周期: {self.dt*1000:.2f}ms")
    
    def set_target_command(self, positions: Optional[np.ndarray] = None,
                          velocities: Optional[np.ndarray] = None,
                          torques: Optional[np.ndarray] = None):
        """
        设置目标控制指令 - 通过队列发送，立即返回，不阻塞
        
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
        
        # 统计接收到的命令
        self.commands_received += 1
        
        try:
            # 非阻塞放入队列
            self.command_queue.put_nowait(command)
        except queue.Full:
            # 队列满了，使用阻塞方式等待，确保命令不丢失
            # 设置短暂超时避免死锁
            try:
                self.command_queue.put(command, timeout=0.001)  # 1ms超时
            except queue.Full:
                # 如果还是满，说明控制线程可能有问题，记录警告但不丢弃命令
                print(f"[BufferControlThread] ⚠️ 队列持续满载，可能存在性能问题")
                # 强制等待放入，确保命令不丢失
                self.command_queue.put(command)
    
    def get_next_command(self) -> dict:
        """按顺序获取下一个命令（非阻塞）"""
        try:
            # 只获取一个命令，按顺序执行
            command = self.command_queue.get_nowait()
            self.commands_processed += 1  # 统计处理的命令
            with self.command_lock:
                self.current_command.update(command)
        except queue.Empty:
            # 没有新命令，继续使用当前命令
            pass
        except Exception as e:
            # 静默处理异常，避免影响控制循环
            pass
        
        # 返回当前命令
        with self.command_lock:
            return {
                'positions': self.current_command.get('positions', None),
                'velocities': self.current_command.get('velocities', None),
                'torques': self.current_command.get('torques', None),
                'timestamp': self.current_command.get('timestamp', None)
            }
    
    def start(self):
        """启动控制线程"""
        with self.thread_lock:
            if self.running:
                print("[BufferControlThread] ⚠️  控制线程已在运行")
                return False
                
            self.running = True
            self.thread = threading.Thread(target=self._control_loop, daemon=True)
            self.thread.start()
            
            print(f"[BufferControlThread] ✅ 控制线程已启动 (频率: {self.control_freq}Hz, 周期: {self.dt*1000:.2f}ms)")
            return True
    
    def stop(self, timeout: float = 2.0):
        """
        停止控制线程
        
        Args:
            timeout: 等待线程停止的超时时间 (秒)
        """
        with self.thread_lock:
            if not self.running:
                print("[BufferControlThread] 控制线程未运行")
                return True
                
            self.running = False
            
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=timeout)
                
                if self.thread.is_alive():
                    print("[BufferControlThread] ⚠️  控制线程停止超时")
                    return False
                else:
                    print("[BufferControlThread] ✅ 控制线程已停止")
                    return True
            
            return True
    
    def is_running(self) -> bool:
        """检查线程是否在运行"""
        with self.thread_lock:
            return self.running and self.thread and self.thread.is_alive()
    # @pysnooper.snoop()
    def once_send(self):
        self._send_to_hardware(self.get_next_command())
        self.loop_count += 1


    # @pysnooper.snoop()
    def _control_loop(self):
        """500Hz控制循环 - 核心控制逻辑"""
        print(f"[BufferControlThread] 开始500Hz控制循环...")
        
        # 设置线程优先级 (尽力而为)
        try:
            import os
            os.nice(-10)  # 提高线程优先级
        except:
            pass
        last_time = time.time()
        while self.running:
            loop_start_time = time.time()
            
            try:
                self.once_send()
            except Exception as e:
                print(f"[BufferControlThread] ❌ 控制循环错误: {e}")
                # 发生错误时发送零指令确保安全
                self._send_zero_command()
            
            # 6. 精确定时控制
            loop_time = time.time() - loop_start_time
            self.total_time += loop_time
            self.max_loop_time = max(self.max_loop_time, loop_time)
            
            # 计算需要睡眠的时间
            sleep_time = self.dt - loop_time -0.0007
            
            if sleep_time > 0:
                # print('the sleep time is ', sleep_time)
                time.sleep(sleep_time)
            # else:
            #     # 错过了截止时间
                
            #         print(f"[BufferControlThread] ⚠️  错过截止时间: {self.missed_deadlines}次, 当前循环时间: {loop_time*1000:.2f}ms")
            # if self.loop_count % 10 == 0 :
            #     print('the frequency is ', 10/(time.time() - last_time))
            #     last_time = time.time()
            if self.loop_count % 500 == 0 :
                queue_size = self.command_queue.qsize()
                print(time.strftime("%H:%M:%S"), 'BufferControlThread', 'freq:', f'{500/(time.time() - last_time):.1f}Hz', 
                      f'queue:{queue_size}', f'recv:{self.commands_received}', f'proc:{self.commands_processed}')
                last_time = time.time()
        print("[BufferControlThread] 控制循环已退出")
        print('[BufferControlThread] 总共执行时间为', self.total_time)
        print('[BufferControlThread] 总共步长为', self.loop_count)
    
    @pysnooper.snoop()
    def _send_to_hardware(self, command: dict):
        """
        发送指令到硬件层 - 优化版本
        
        Args:
            command: 控制指令字典
        """
        assert command['positions'] is not None and command['velocities'] is not None and command['torques'] is not None
        try:
            # 直接调用底层方法，减少函数调用开销
            self.icarm._original_set_joint_positions(
                command['positions'], command['velocities'], command['torques']
            )
                
        except Exception as e:
            print(f"[BufferControlThread] ❌ 硬件发送失败: {e}")
            raise
    
    def _send_zero_command(self):
        """发送零指令 - 紧急安全措施"""
        try:
            zero_positions = np.zeros(self.icarm.motor_count)
            zero_velocities = np.zeros(self.icarm.motor_count)
            zero_torques = np.zeros(self.icarm.motor_count)
            
            self._send_to_hardware({
                'positions': zero_positions,
                'velocities': zero_velocities,
                'torques': zero_torques
            })
        except Exception as e:
            print(f"[BufferControlThread] ❌ 发送零指令失败: {e}")
    
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
            'deadline_miss_rate': self.missed_deadlines / max(self.loop_count, 1) * 100
        }
    
    def print_statistics(self):
        """打印控制线程统计信息"""
        stats = self.get_statistics()
        print(f"\n[BufferControlThread] 统计信息:")
        print(f"  运行状态: {'运行中' if stats['is_running'] else '已停止'}")
        print(f"  目标频率: {stats['target_frequency']}Hz")
        print(f"  实际频率: {stats['actual_frequency']:.1f}Hz")
        print(f"  循环次数: {stats['loop_count']}")
        print(f"  平均循环时间: {stats['avg_loop_time_ms']:.2f}ms")
        print(f"  最大循环时间: {stats['max_loop_time_ms']:.2f}ms")
        print(f"  错过截止时间: {stats['missed_deadlines']}次 ({stats['deadline_miss_rate']:.1f}%)")


if __name__ == "__main__":
    # 简单测试
    class MockICArm:
        def __init__(self):
            self.motor_count = 9
            
        def _original_set_joint_positions(self, pos, vel, torque):
            print(f"Mock发送指令: pos={pos is not None}, vel={vel is not None}, torque={torque is not None}")
    
    # 创建测试实例
    mock_arm = MockICArm()
    control_thread = BufferControlThread(mock_arm, control_freq=100)  # 降低频率用于测试
    
    # 启动线程
    control_thread.start()
    
    # 发送一些测试指令
    for i in range(5):
        positions = np.array([0.1 * i] * 9)
        control_thread.set_target_command(positions=positions)
        time.sleep(0.5)
    
    # 等待一段时间
    time.sleep(2)
    
    # 打印统计信息
    control_thread.print_statistics()
    
    # 停止线程
    control_thread.stop()
    
    print("BufferControlThread测试完成")
