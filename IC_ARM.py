import math 
from DM_CAN import *
import serial
import time
import rerun as rr
import numpy as np
motor_config = {
    'm1': {'type': DM_Motor_Type.DM4340, 'id': 0x01, 'master_id': 0x00, 'kp': 40, 'kd': 1.5, 'torque': 5},
    'm2': {'type': DM_Motor_Type.DM4340, 'id': 0x02, 'master_id': 0x00, 'kp': 45, 'kd': 1.5, 'torque': 0.},
    'm3': {'type': DM_Motor_Type.DM4340, 'id': 0x03, 'master_id': 0x00, 'kp': 40, 'kd': 1.5, 'torque': 0.5},
    'm4': {'type': DM_Motor_Type.DM4340, 'id': 0x04, 'master_id': 0x00, 'kp': 38, 'kd': 1.5, 'torque': 0.0},
    'm5': {'type': DM_Motor_Type.DM4340, 'id': 0x05, 'master_id': 0x00, 'kp': 35, 'kd': 1.5, 'torque': 0.5},
    }


class ICARM:
    def __init__(self, serial_port='/dev/cu.usbmodem00000000050C1', baudrate=921600):
        """Initialize the ICARM system with all motors"""
        # Initialize serial connection
        self.serial_device = serial.Serial(serial_port, baudrate, timeout=0.5)
        
        # Initialize motor control
        self.mc = MotorControl(self.serial_device)
        
        # Initialize all motors using motor_config parameters
        self.motors = {}
        for motor_name, config in motor_config.items():
            motor = Motor(config['type'], config['id'], config['master_id'])
            self.motors[motor_name] = motor
        
        # Add all motors to motor control
        for motor_name, motor in self.motors.items():
            self.mc.addMotor(motor)
            
        print(f"ICARM initialized with {len(self.motors)} motors using motor_config parameters")
    
    def read_all_motor_info(self):
        """Read all motor information and print as a table"""
        print("\n" + "="*80)
        print("MOTOR INFORMATION TABLE")
        print("="*80)
        print(f"{'Motor':<8} {'ID':<4} {'PMAX':<12} {'VMAX':<12} {'TMAX':<12} {'Status':<10}")
        print("-"*80)
        
        for motor_name, motor in self.motors.items():
            try:
                # Read motor parameters
                pmax = self.mc.read_motor_param(motor, DM_variable.PMAX)
                vmax = self.mc.read_motor_param(motor, DM_variable.VMAX)
                tmax = self.mc.read_motor_param(motor, DM_variable.TMAX)
                status = "OK"
            except Exception as e:
                pmax = vmax = tmax = "ERROR"
                status = "FAIL"
                
            print(f"{motor_name:<8} {motor.SlaveID:<4} {pmax:<12} {vmax:<12} {tmax:<12} {status:<10}")
        
        print("="*80)
        print()
    
    def read_all_angles(self):
        """Read all motor angles (positions) and print as a table"""
        print("\n" + "="*60)
        print("MOTOR ANGLES (POSITIONS)")
        print("="*60)
        print(f"{'Motor':<8} {'ID':<4} {'Angle (rad)':<15} {'Angle (deg)':<15} {'Status':<10}")
        print("-"*60)
        
        angles = {}
        for motor_name, motor in self.motors.items():
            try:
                angle_rad = motor.getPosition()
                print(angle_rad)
                angle_deg = math.degrees(angle_rad)
                angles[motor_name] = angle_rad
                status = "OK"
            except Exception as e:
                angle_rad = angle_deg = "ERROR"
                status = "FAIL"
                
            print(f"{motor_name:<8} {motor.SlaveID:<4} {angle_rad:<15} {angle_deg:<15} {status:<10}")
        
        print("="*60)
        print()
        return angles
    
    def read_all_velocities(self):
        """Read all motor velocities and print as a table"""
        print("\n" + "="*60)
        print("MOTOR VELOCITIES")
        print("="*60)
        print(f"{'Motor':<8} {'ID':<4} {'Velocity (rad/s)':<18} {'Velocity (rpm)':<15} {'Status':<10}")
        print("-"*60)
        
        velocities = {}
        for motor_name, motor in self.motors.items():
            try:
                vel_rad_s = motor.getVelocity()
                vel_rpm = vel_rad_s * 60 / (2 * math.pi)  # Convert rad/s to rpm
                velocities[motor_name] = vel_rad_s
                status = "OK"
            except Exception as e:
                vel_rad_s = vel_rpm = "ERROR"
                status = "FAIL"
                
            print(f"{motor_name:<8} {motor.SlaveID:<4} {vel_rad_s:<18} {vel_rpm:<15} {status:<10}")
        
        print("="*60)
        print()
        return velocities
    
    def read_all_currents(self):
        """Read all motor currents (torques) and print as a table"""
        print("\n" + "="*60)
        print("MOTOR CURRENTS (TORQUES)")
        print("="*60)
        print(f"{'Motor':<8} {'ID':<4} {'Torque (Nm)':<15} {'Status':<10}")
        print("-"*60)
        
        currents = {}
        for motor_name, motor in self.motors.items():
            try:
                torque = motor.getTorque()
                currents[motor_name] = torque
                status = "OK"
            except Exception as e:
                torque = "ERROR"
                status = "FAIL"
                
            print(f"{motor_name:<8} {motor.SlaveID:<4} {torque:<15} {status:<10}")
        
        print("="*60)
        print()
        return currents
    
    def enable_all_motors(self):
        """Enable all motors"""
        print("\nEnabling all motors...")
        for motor_name, motor in self.motors.items():
            try:
                self.mc.enable(motor)
                motor.isEnable = True
                print(f"Motor {motor_name} enabled successfully")
            except Exception as e:
                print(f"Failed to enable motor {motor_name}: {e}")
        print("Waiting for motors to stabilize...")
        time.sleep(1)  # Wait for motors to stabilize
    
    def disable_all_motors(self):
        """Disable all motors"""
        print("\nDisabling all motors...")
        for motor_name, motor in self.motors.items():
            try:
                self.mc.disable(motor)
                motor.isEnable = False
                print(f"Motor {motor_name} disabled successfully")
            except Exception as e:
                print(f"Failed to disable motor {motor_name}: {e}")
    
    def test_motor_communication(self):
        """Test motor communication by enabling motors and reading positions"""
        print("\n" + "="*70)
        print("TESTING MOTOR COMMUNICATION")
        print("="*70)
        
        # Enable all motors first
        self.enable_all_motors()
        
        # Try to send a small position command to activate feedback
        # print("\nSending small position commands to activate feedback...")
        # for motor_name, motor in self.motors.items():
        #     try:
        #         # Send a very small position command (0.01 rad) to activate the motor
        #         self.mc.controlMIT(motor, 10, 1, 0.01, 0, 0)
        #         time.sleep(0.1)
        #     except Exception as e:
        #         print(f"Failed to send command to {motor_name}: {e}")
        
        # time.sleep(0.5)  # Wait for response
        
        # Now read positions
        print("\nReading positions after enabling and commanding:")
        angles = self.read_all_angles()
        
        # # Return to zero position
        # print("\nReturning to zero position...")
        # for motor_name, motor in self.motors.items():
        #     try:
        #         self.mc.controlMIT(motor, 10, 1, 0, 0, 0)
        #         time.sleep(0.1)
        #     except Exception as e:
        #         print(f"Failed to return {motor_name} to zero: {e}")
        
        # time.sleep(0.5)
        
        return angles
    
    # ========== CONTROL API FUNCTIONS (Basic Implementation) ==========
    
    def enable_motor(self, motor_name):
        """Enable a specific motor"""
        if motor_name in self.motors:
            motor = self.motors[motor_name]
            # TODO: Implement motor enable logic
            print(f"Enabling motor {motor_name}")
        else:
            print(f"Motor {motor_name} not found")
    
    def disable_motor(self, motor_name):
        """Disable a specific motor"""
        if motor_name in self.motors:
            motor = self.motors[motor_name]
            # TODO: Implement motor disable logic
            print(f"Disabling motor {motor_name}")
        else:
            print(f"Motor {motor_name} not found")
    
    def set_position(self, motor_name, position, velocity=None, torque=None):
        """Set motor position with optional velocity and torque limits"""
        if motor_name in self.motors:
            motor = self.motors[motor_name]
            # TODO: Implement position control
            print(f"Setting {motor_name} position to {position}")
        else:
            print(f"Motor {motor_name} not found")
    
    def set_velocity(self, motor_name, velocity, torque=None):
        """Set motor velocity with optional torque limit"""
        if motor_name in self.motors:
            motor = self.motors[motor_name]
            # TODO: Implement velocity control
            print(f"Setting {motor_name} velocity to {velocity}")
        else:
            print(f"Motor {motor_name} not found")
    
    def set_torque(self, motor_name, torque):
        """Set motor torque"""
        if motor_name in self.motors:
            motor = self.motors[motor_name]
            # TODO: Implement torque control
            print(f"Setting {motor_name} torque to {torque}")
        else:
            print(f"Motor {motor_name} not found")
    
    def get_motor_state(self, motor_name):
        """Get current motor state (position, velocity, torque)"""
        if motor_name in self.motors:
            motor = self.motors[motor_name]
            # TODO: Implement state reading
            print(f"Reading {motor_name} state")
            return {'position': 0, 'velocity': 0, 'torque': 0}  # Placeholder
        else:
            print(f"Motor {motor_name} not found")
            return None
    
    def emergency_stop(self):
        """Emergency stop all motors"""
        print("EMERGENCY STOP - Disabling all motors")
        for motor_name in self.motors:
            self.disable_motor(motor_name)
    
    def get_positions_only(self):
        """Get only the positions of all motors without printing"""
        positions = {}
        for motor_name, motor in self.motors.items():
            try:
                # First refresh motor status to get latest data
                self.mc.refresh_motor_status(motor)
                # Then read the position
                angle_rad = motor.getPosition()
                angle_deg = math.degrees(angle_rad)
                positions[motor_name] = {
                    'rad': angle_rad,
                    'deg': angle_deg,
                    'id': motor.SlaveID
                }
            except Exception as e:
                positions[motor_name] = {
                    'rad': None,
                    'deg': None,
                    'id': motor.SlaveID,
                    'error': str(e)
                }
        return positions
    
    def monitor_positions_continuous(self, update_rate=10.0, duration=None):
        """Continuously monitor and display motor positions using rerun
        
        Args:
            update_rate: Updates per second (Hz)
            duration: Duration in seconds (None for infinite)
        """
        # Initialize rerun
        rr.init("ICARM_Position_Monitor", spawn=True)
        
        print("Starting continuous position monitoring...")
        print("Press Ctrl+C to stop")
        
        # Enable all motors first
        self.enable_all_motors()
        
        start_time = time.time()
        update_interval = 1.0 / update_rate
        
        try:
            while True:
                current_time = time.time()
                
                # Check duration limit
                if duration and (current_time - start_time) > duration:
                    break
                
                # Get positions
                positions = self.get_positions_only()
                print(positions)
                # Log to rerun
                rr.set_time_seconds("timestamp", current_time - start_time)
                
                # Create arrays for plotting
                motor_names = list(positions.keys())
                angles_deg = []
                motor_ids = []
                
                for motor_name in motor_names:
                    pos_data = positions[motor_name]
                    if pos_data['deg'] is not None:
                        angles_deg.append(pos_data['deg'])
                        motor_ids.append(pos_data['id'])
                        
                        # Log individual motor position
                        rr.log(f"motors/{motor_name}/position_deg", rr.Scalars(float(pos_data['deg'])))
                    else:
                        angles_deg.append(0.0)
                        motor_ids.append(pos_data['id'])
                
                # Log combined data as time series
                if angles_deg:
                    for i, motor_name in enumerate(motor_names):
                        rr.log(f"overview/motor_{motor_name}_deg", rr.Scalars(float(angles_deg[i])))
                
                # Print to console (compact format)
                pos_str = " | ".join([f"{name}: {positions[name]['deg']:.2f}°" 
                                     for name in motor_names if positions[name]['deg'] is not None])
                print(f"\r{pos_str}", end="", flush=True)
                
                # Wait for next update
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\n\nStopping position monitoring...")
        finally:
            # Disable motors for safety
            self.disable_all_motors()
            print("Position monitoring stopped.")
    
    def set_all_zero_positions(self):
        """Set current positions of all motors as new zero positions
        
        This function should be called when the arm is in the desired zero position.
        All motors will be disabled first, then their current positions will be set as zero.
        """
        print("Setting current positions as zero for all motors...")
        print("WARNING: This will change the zero reference for all motors!")
        
        # Confirm with user
        confirm = input("Are you sure you want to set current positions as zero? (y/N): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return False
        
        # First, read and display current positions
        print("\nCurrent positions before setting zero:")
        current_positions = self.get_positions_only()
        for motor_name, pos_data in current_positions.items():
            if pos_data['deg'] is not None:
                print(f"{motor_name}: {pos_data['deg']:.2f}°")
        
        # Disable all motors first (required for setting zero position)
        print("\nDisabling all motors...")
        self.disable_all_motors()
        
        # Set zero position for each motor
        success_count = 0
        for motor_name, motor in self.motors.items():
            try:
                print(f"Setting zero position for {motor_name}...")
                self.mc.set_zero_position(motor)
                success_count += 1
                print(f"✓ {motor_name} zero position set successfully")
            except Exception as e:
                print(f"✗ Failed to set zero position for {motor_name}: {e}")
        
        print(f"\nZero position setting completed: {success_count}/{len(self.motors)} motors successful")
        
        # Verify new positions (should be close to zero)
        print("\nVerifying new zero positions:")
        time.sleep(1)  # Wait a moment for the changes to take effect
        new_positions = self.get_positions_only()
        for motor_name, pos_data in new_positions.items():
            if pos_data['deg'] is not None:
                print(f"{motor_name}: {pos_data['deg']:.2f}°")
        
        return success_count == len(self.motors)
    
    def set_single_zero_position(self, motor_name):
        """Set current position of a single motor as new zero position
        
        Args:
            motor_name: Name of the motor (e.g., 'm1', 'm2', etc.)
        """
        if motor_name not in self.motors:
            print(f"Error: Motor {motor_name} not found")
            return False
        
        motor = self.motors[motor_name]
        
        print(f"Setting current position as zero for {motor_name}...")
        
        # Read current position
        current_pos = self.get_positions_only()[motor_name]
        if current_pos['deg'] is not None:
            print(f"Current position: {current_pos['deg']:.2f}°")
        
        # Confirm with user
        confirm = input(f"Set current position as zero for {motor_name}? (y/N): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return False
        
        # Disable motor first
        print(f"Disabling {motor_name}...")
        self.disable_motor(motor_name)
        
        try:
            # Set zero position
            self.mc.set_zero_position(motor)
            print(f"✓ {motor_name} zero position set successfully")
            
            # Verify
            time.sleep(1)
            new_pos = self.get_positions_only()[motor_name]
            if new_pos['deg'] is not None:
                print(f"New position: {new_pos['deg']:.2f}°")
            
            return True
        except Exception as e:
            print(f"✗ Failed to set zero position for {motor_name}: {e}")
            return False
    
    def get_current_positions_deg(self):
        """获取当前位置（度）"""
        positions = self.get_positions_only()
        return [positions[f'm{i+1}']['deg'] for i in range(5)]
    
    def linear_interpolation(self, start_pos, end_pos, duration, update_rate=50):
        """
        线性插值生成轨迹点
        
        Args:
            start_pos: 起始位置列表 [j1, j2, j3, j4, j5] (度)
            end_pos: 结束位置列表 [j1, j2, j3, j4, j5] (度)
            duration: 运动时间 (秒)
            update_rate: 更新频率 (Hz)
        
        Returns:
            trajectory: 轨迹点列表，每个点是 [j1, j2, j3, j4, j5, time]
        """
        num_points = int(duration * update_rate)
        trajectory = []
        
        for i in range(num_points + 1):
            t = i / num_points  # 归一化时间 [0, 1]
            
            # 线性插值
            current_pos = []
            for j in range(5):
                pos = start_pos[j] + t * (end_pos[j] - start_pos[j])
                current_pos.append(pos)
            
            time_stamp = t * duration
            trajectory.append(current_pos + [time_stamp])
        
        return trajectory
    
    def smooth_interpolation(self, start_pos, end_pos, duration, update_rate=50):
        """
        平滑插值（S曲线）生成轨迹点
        
        Args:
            start_pos: 起始位置列表 [j1, j2, j3, j4, j5] (度)
            end_pos: 结束位置列表 [j1, j2, j3, j4, j5] (度)
            duration: 运动时间 (秒)
            update_rate: 更新频率 (Hz)
        
        Returns:
            trajectory: 轨迹点列表，每个点是 [j1, j2, j3, j4, j5, time]
        """
        num_points = int(duration * update_rate)
        trajectory = []
        
        for i in range(num_points + 1):
            t = i / num_points  # 归一化时间 [0, 1]
            
            # S曲线插值 (3次多项式: 3t² - 2t³)
            s = 3 * t**2 - 2 * t**3
            
            # 应用插值
            current_pos = []
            for j in range(5):
                pos = start_pos[j] + s * (end_pos[j] - start_pos[j])
                current_pos.append(pos)
            
            time_stamp = t * duration
            trajectory.append(current_pos + [time_stamp])
        
        return trajectory
    
    def execute_trajectory(self, trajectory, verbose=True):
        """
        执行轨迹
        
        Args:
            trajectory: 轨迹点列表
            verbose: 是否打印详细信息
        """
        print("开始执行轨迹...")
        
        # 启用所有电机
        self.enable_all_motors()
        
        start_time = time.time()
        
        try:
            for i, point in enumerate(trajectory):
                target_positions = point[:5]  # 前5个是关节位置
                target_time = point[5]        # 第6个是时间戳
                
                # 等待到达目标时间
                while (time.time() - start_time) < target_time:
                    time.sleep(0.001)  # 1ms精度
                
                # 发送位置命令到各个电机
                for motor_idx, target_deg in enumerate(target_positions):
                    motor_name = f'm{motor_idx + 1}'
                    if motor_name in self.motors:
                        motor = self.motors[motor_name]
                        try:
                            # 转换为弧度
                            target_rad = math.radians(target_deg)
                            # 使用每个电机的个性化参数
                            motor_params = motor_config[motor_name]
                            kp = motor_params['kp']
                            kd = motor_params['kd']
                            torque = motor_params['torque']
                            # 使用MIT控制模式：位置控制
                            self.mc.controlMIT(motor, kp, kd, target_rad, 0.0, torque)
                        except Exception as e:
                            if verbose:
                                print(f"警告: 电机 {motor_name} 设置失败: {e}")
                
                # 打印进度
                if verbose and i % 10 == 0:  # 每10个点打印一次
                    progress = (i / len(trajectory)) * 100
                    current_pos = self.get_current_positions_deg()
                    print(f"进度: {progress:.1f}% | 目标: {[f'{p:.1f}' for p in target_positions]} | "
                          f"实际: {[f'{p:.1f}' for p in current_pos]}")
        
        except KeyboardInterrupt:
            print("\n运动被中断")
        
        finally:
            # 安全停止
            print("停止所有电机...")
            self.disable_all_motors()
        
        # 验证最终位置
        final_pos = self.get_current_positions_deg()
        print(f"\n轨迹执行完成!")
        print(f"最终位置: {[f'{p:.2f}°' for p in final_pos]}")
        
        return final_pos
    
    def home_to_zero(self, duration=1.0, interpolation_type='smooth'):
        """
        回零主函数
        
        Args:
            duration: 运动时间 (秒)
            interpolation_type: 插值类型 ('linear' 或 'smooth')
        
        Returns:
            success: 是否成功回零
        """
        print("=== IC ARM 回零运动 ===")
        
        target_positions = [0.0, 0.0, 0.0, 0.0, 0.0]  # 目标零点位置（度）
        
        # 获取当前位置
        print("读取当前位置...")
        current_pos = self.get_current_positions_deg()
        print(f"当前位置: {[f'{p:.2f}°' for p in current_pos]}")
        print(f"目标位置: {[f'{p:.2f}°' for p in target_positions]}")
        
        # 计算运动距离
        distances = [abs(current_pos[i] - target_positions[i]) for i in range(5)]
        max_distance = max(distances)
        print(f"最大运动距离: {max_distance:.2f}°")
        
        if max_distance < 1.0:
            print("已经接近零点位置，无需回零")
            return True
        
        # 生成轨迹
        print(f"生成{interpolation_type}插值轨迹，时长{duration}秒...")
        if interpolation_type == 'linear':
            trajectory = self.linear_interpolation(current_pos, target_positions, duration)
        else:
            trajectory = self.smooth_interpolation(current_pos, target_positions, duration)
        
        print(f"轨迹点数: {len(trajectory)}")
        
        # 执行轨迹
        final_pos = self.execute_trajectory(trajectory)
        
        # 计算误差
        errors = [abs(final_pos[i] - target_positions[i]) for i in range(5)]
        print(f"位置误差: {[f'{e:.2f}°' for e in errors]}")
        max_error = max(errors)
        print(f"最大误差: {max_error:.2f}°")
        
        success = max_error < 2.0  # 如果最大误差小于2度认为成功
        
        if success:
            print("✓ 回零成功!")
        else:
            print("✗ 回零精度不足，可能需要调整参数")
        
        return success
    
    def move_to_position(self, target_positions, duration=2.0, interpolation_type='smooth'):
        """
        移动到指定位置
        
        Args:
            target_positions: 目标位置列表 [j1, j2, j3, j4, j5] (度)
            duration: 运动时间 (秒)
            interpolation_type: 插值类型 ('linear' 或 'smooth')
        
        Returns:
            success: 是否成功到达目标位置
        """
        print(f"=== 移动到目标位置 ===")
        
        # 获取当前位置
        current_pos = self.get_current_positions_deg()
        print(f"当前位置: {[f'{p:.2f}°' for p in current_pos]}")
        print(f"目标位置: {[f'{p:.2f}°' for p in target_positions]}")
        
        # 计算运动距离
        distances = [abs(current_pos[i] - target_positions[i]) for i in range(5)]
        max_distance = max(distances)
        print(f"最大运动距离: {max_distance:.2f}°")
        
        # 生成轨迹
        print(f"生成{interpolation_type}插值轨迹，时长{duration}秒...")
        if interpolation_type == 'linear':
            trajectory = self.linear_interpolation(current_pos, target_positions, duration)
        else:
            trajectory = self.smooth_interpolation(current_pos, target_positions, duration)
        
        print(f"轨迹点数: {len(trajectory)}")
        
        # 执行轨迹
        final_pos = self.execute_trajectory(trajectory)
        
        # 计算误差
        errors = [abs(final_pos[i] - target_positions[i]) for i in range(5)]
        print(f"位置误差: {[f'{e:.2f}°' for e in errors]}")
        max_error = max(errors)
        print(f"最大误差: {max_error:.2f}°")
        
        success = max_error < 3.0  # 如果最大误差小于3度认为成功
        
        if success:
            print("✓ 运动成功!")
        else:
            print("✗ 运动精度不足，可能需要调整参数")
        
        return success
    
    def close(self):
        """Close serial connection and cleanup"""
        print("Closing ICARM connection")
        if hasattr(self, 'serial_device') and self.serial_device.is_open:
            self.serial_device.close()

# Example usage (uncomment to test)
if __name__ == "__main__":
    arm = ICARM()
    
    try:
        # Read motor information
        arm.read_all_motor_info()
        
        # Test motor communication (enable motors and try to get real positions)
        print("\n" + "="*50)
        print("TESTING MOTOR COMMUNICATION...")
        print("="*50)
        
        # Test communication
        test_angles = arm.test_motor_communication()
        
        # Read all motor angles (should show real positions now)
        print("\nFinal position readings:")
        angles = arm.read_all_angles()
        
        # Read all motor velocities
        velocities = arm.read_all_velocities()
        
        # Read all motor currents (torques)
        currents = arm.read_all_currents()
        
        # Disable all motors for safety
        arm.disable_all_motors()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        # Make sure to disable motors even if there's an error
        try:
            arm.disable_all_motors()
        except:
            pass
    
    finally:
        arm.close()
