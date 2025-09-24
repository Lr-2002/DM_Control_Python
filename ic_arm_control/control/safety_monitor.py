"""
SafetyMonitor - 机械臂安全监控器
提供实时安全检查和限制功能
"""

import numpy as np
import time
from typing import Tuple, Optional
import threading


class SafetyMonitor:
    """机械臂安全监控器 - 实时安全检查和保护"""
    
    def __init__(self, motor_count: int = 9):
        """
        初始化安全监控器
        
        Args:
            motor_count: 电机数量
        """
        self.motor_count = motor_count
        
        # 安全限制参数 (可根据实际硬件调整)
        self.joint_limits_rad = np.array([
            [-np.pi, np.pi],      # m1
            [-np.pi, np.pi],      # m2  
            [-np.pi, np.pi],      # m3
            [-np.pi, np.pi],      # m4
            [-np.pi, np.pi],      # m5
            [-np.pi, np.pi],      # m6
            [-np.pi, np.pi],      # m7
            [-np.pi, np.pi],      # m8
            [-np.pi*10, np.pi*10] # m9 (servo, 更大范围)
        ])
        
        # 速度限制 (rad/s)
        self.velocity_limits = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 10.0])
        
        # 加速度限制 (rad/s²)
        self.acceleration_limits = np.array([10.0] * motor_count)
        
        # 力矩限制 (N·m)
        self.torque_limits = np.array([50.0] * motor_count)
        
        # 安全状态标志
        self.emergency_stop_flag = False
        self.safety_violations = 0
        self.last_check_time = time.time()
        
        # 线程安全锁
        self._lock = threading.Lock()
        
        print(f"[SafetyMonitor] 安全监控器已初始化，监控{motor_count}个电机")
    
    def set_emergency_stop(self, stop: bool = True):
        """设置紧急停止标志"""
        with self._lock:
            self.emergency_stop_flag = stop
            if stop:
                print("[SafetyMonitor] ⚠️  紧急停止已激活")
            else:
                print("[SafetyMonitor] ✅ 紧急停止已解除")
    
    def is_emergency_stopped(self) -> bool:
        """检查是否处于紧急停止状态"""
        with self._lock:
            return self.emergency_stop_flag
    
    def check_joint_limits(self, positions: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        检查关节限位
        
        Args:
            positions: 关节位置数组 (rad)
            
        Returns:
            (is_safe, safe_positions): 是否安全，安全的位置
        """
        if positions is None or len(positions) == 0:
            return False, np.zeros(self.motor_count)
            
        positions = np.array(positions)
        safe_positions = positions.copy()
        is_safe = True
        
        for i in range(min(len(positions), self.motor_count)):
            min_limit, max_limit = self.joint_limits_rad[i]
            
            if positions[i] < min_limit:
                safe_positions[i] = min_limit
                is_safe = False
                print(f"[SafetyMonitor] ⚠️  关节{i+1}超出下限: {np.degrees(positions[i]):.1f}° < {np.degrees(min_limit):.1f}°")
                
            elif positions[i] > max_limit:
                safe_positions[i] = max_limit
                is_safe = False
                print(f"[SafetyMonitor] ⚠️  关节{i+1}超出上限: {np.degrees(positions[i]):.1f}° > {np.degrees(max_limit):.1f}°")
        
        return is_safe, safe_positions
    
    def check_velocity_limits(self, velocities: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        检查速度限制
        
        Args:
            velocities: 关节速度数组 (rad/s)
            
        Returns:
            (is_safe, safe_velocities): 是否安全，安全的速度
        """
        if velocities is None or len(velocities) == 0:
            return True, np.zeros(self.motor_count)
            
        velocities = np.array(velocities)
        safe_velocities = velocities.copy()
        is_safe = True
        
        for i in range(min(len(velocities), self.motor_count)):
            max_vel = self.velocity_limits[i]
            
            if abs(velocities[i]) > max_vel:
                safe_velocities[i] = np.sign(velocities[i]) * max_vel
                is_safe = False
                print(f"[SafetyMonitor] ⚠️  关节{i+1}速度超限: {np.degrees(abs(velocities[i])):.1f}°/s > {np.degrees(max_vel):.1f}°/s")
        
        return is_safe, safe_velocities
    
    def check_torque_limits(self, torques: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        检查力矩限制
        
        Args:
            torques: 关节力矩数组 (N·m)
            
        Returns:
            (is_safe, safe_torques): 是否安全，安全的力矩
        """
        if torques is None or len(torques) == 0:
            return True, np.zeros(self.motor_count)
            
        torques = np.array(torques)
        safe_torques = torques.copy()
        is_safe = True
        
        for i in range(min(len(torques), self.motor_count)):
            max_torque = self.torque_limits[i]
            
            if abs(torques[i]) > max_torque:
                safe_torques[i] = np.sign(torques[i]) * max_torque
                is_safe = False
                print(f"[SafetyMonitor] ⚠️  关节{i+1}力矩超限: {abs(torques[i]):.2f}N·m > {max_torque:.2f}N·m")
        
        return is_safe, safe_torques
    
    def check_command_safety(self, positions: Optional[np.ndarray] = None, 
                           velocities: Optional[np.ndarray] = None,
                           torques: Optional[np.ndarray] = None) -> Tuple[bool, dict]:
        """
        综合安全检查
        
        Args:
            positions: 目标位置 (rad)
            velocities: 目标速度 (rad/s) 
            torques: 目标力矩 (N·m)
            
        Returns:
            (is_safe, safe_command): 是否安全，安全的指令字典
        """
        with self._lock:
            # 检查紧急停止
            if self.emergency_stop_flag:
                return False, {
                    'positions': np.zeros(self.motor_count),
                    'velocities': np.zeros(self.motor_count),
                    'torques': np.zeros(self.motor_count),
                    'reason': 'emergency_stop'
                }
        
        overall_safe = True
        safe_command = {}
        
        # 检查位置限制
        if positions is not None:
            pos_safe, safe_pos = self.check_joint_limits(positions)
            safe_command['positions'] = safe_pos
            overall_safe &= pos_safe
        else:
            safe_command['positions'] = None
            
        # 检查速度限制
        if velocities is not None:
            vel_safe, safe_vel = self.check_velocity_limits(velocities)
            safe_command['velocities'] = safe_vel
            overall_safe &= vel_safe
        else:
            safe_command['velocities'] = None
            
        # 检查力矩限制
        if torques is not None:
            torque_safe, safe_torque = self.check_torque_limits(torques)
            safe_command['torques'] = safe_torque
            overall_safe &= torque_safe
        else:
            safe_command['torques'] = None
        
        # 更新违规计数
        if not overall_safe:
            self.safety_violations += 1
            
        self.last_check_time = time.time()
        
        return overall_safe, safe_command
    
    def get_safety_status(self) -> dict:
        """获取安全状态信息"""
        with self._lock:
            return {
                'emergency_stop': self.emergency_stop_flag,
                'safety_violations': self.safety_violations,
                'last_check_time': self.last_check_time,
                'joint_limits': self.joint_limits_rad.tolist(),
                'velocity_limits': self.velocity_limits.tolist(),
                'torque_limits': self.torque_limits.tolist()
            }
    
    def reset_safety_violations(self):
        """重置安全违规计数"""
        with self._lock:
            self.safety_violations = 0
            print("[SafetyMonitor] 安全违规计数已重置")


if __name__ == "__main__":
    # 测试安全监控器
    safety = SafetyMonitor(motor_count=9)
    
    # 测试正常指令
    positions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    velocities = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    torques = np.array([10.0] * 9)
    
    is_safe, safe_cmd = safety.check_command_safety(positions, velocities, torques)
    print(f"正常指令安全检查: {is_safe}")
    
    # 测试超限指令
    bad_positions = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])  # 超出限制
    is_safe, safe_cmd = safety.check_command_safety(bad_positions, velocities, torques)
    print(f"超限指令安全检查: {is_safe}")
    
    # 测试紧急停止
    safety.set_emergency_stop(True)
    is_safe, safe_cmd = safety.check_command_safety(positions, velocities, torques)
    print(f"紧急停止状态检查: {is_safe}")
    
    print("SafetyMonitor测试完成")
