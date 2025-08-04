#!/usr/bin/env python3
"""
Simple function to rotate m5 motor by 30 degrees
"""

import math
import time
from IC_ARM import ICARM

def rotate_m5_30_degrees():
    """Simple function to rotate m5 motor by 30 degrees"""
    print("Rotating m5 motor by 30 degrees...")
    
    # Initialize ICARM
    arm = ICARM()
    
    try:
        # Get m5 motor
        motor = arm.motors['m5']
        
        # Enable motor
        print("Enabling m5 motor...")
        arm.mc.enable(motor)
        time.sleep(0.5)
        
        # Get current position
        arm.mc.refresh_motor_status(motor)
        current_pos_rad = motor.getPosition()
        current_pos_deg = math.degrees(current_pos_rad)
        
        # Calculate target position (current + 30 degrees)
        target_pos_deg = current_pos_deg + 30.0
        target_pos_rad = math.radians(target_pos_deg)
        
        print(f"Current position: {current_pos_deg:.2f}°")
        print(f"Target position: {target_pos_deg:.2f}°")
        
        # Control parameters
        kp = 25.0  # Position gain
        kd = 1.0   # Damping gain
        
        # Control loop
        start_time = time.time()
        timeout = 10.0  # 10 second timeout
        tolerance = math.radians(1.0)  # 1 degree tolerance
        
        while True:
            # Check timeout
            if (time.time() - start_time) > timeout:
                print("Timeout reached!")
                break
            
            # Get current position
            arm.mc.refresh_motor_status(motor)
            current_pos_rad = motor.getPosition()
            
            # Calculate error
            error = target_pos_rad - current_pos_rad
            
            # Check if reached target
            if abs(error) < tolerance:
                final_pos_deg = math.degrees(current_pos_rad)
                print(f"Target reached! Final position: {final_pos_deg:.2f}°")
                break
            
            # Calculate desired velocity
            max_vel = 1.0  # rad/s
            desired_vel = max_vel * math.tanh(error / math.radians(10))
            
            # Send control command
            arm.mc.controlMIT(motor, kp, kd, target_pos_rad, desired_vel, 0.0)
            
            # Show progress
            current_deg = math.degrees(current_pos_rad)
            error_deg = math.degrees(error)
            print(f"\rPosition: {current_deg:.1f}° (error: {error_deg:.1f}°)", end="", flush=True)
            
            time.sleep(0.02)  # 50Hz
        
        print("\nRotation completed!")
        
        # Hold position briefly
        for _ in range(25):
            arm.mc.controlMIT(motor, kp, kd, target_pos_rad, 0.0, 0.0)
            time.sleep(0.02)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Always disable motor and close connection
        try:
            arm.mc.disable(motor)
            print("Motor disabled.")
        except:
            pass
        arm.close()

if __name__ == "__main__":
    rotate_m5_30_degrees()
