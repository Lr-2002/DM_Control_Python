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
                angles_rad = []
                angles_deg = []
                motor_ids = []
                
                for motor_name in motor_names:
                    pos_data = positions[motor_name]
                    if pos_data['rad'] is not None:
                        angles_rad.append(pos_data['rad'])
                        angles_deg.append(pos_data['deg'])
                        motor_ids.append(pos_data['id'])
                        
                        # Log individual motor position
                        rr.log(f"motors/{motor_name}/position_rad", rr.Scalars(float(pos_data['rad'])))
                        rr.log(f"motors/{motor_name}/position_deg", rr.Scalars(float(pos_data['deg'])))
                    else:
                        angles_rad.append(0.0)
                        angles_deg.append(0.0)
                        motor_ids.append(pos_data['id'])
                
                # Log combined data as time series
                if angles_rad:
                    for i, motor_name in enumerate(motor_names):
                        rr.log(f"overview/motor_{motor_name}_rad", rr.Scalars(float(angles_rad[i])))
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
