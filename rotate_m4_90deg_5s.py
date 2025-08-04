#!/usr/bin/env python3
"""
Function to rotate m4 motor by 90 degrees in 5 seconds
"""

import math
import time
from IC_ARM import ICARM

def rotate_m4_90_degrees_in_5s():
    """Rotate m4 motor by 90 degrees in exactly 5 seconds"""
    print("Rotating m4 motor by 90 degrees in 5 seconds...")
    
    # Initialize ICARM
    arm = ICARM()
    
    try:
        # Get m4 motor
        motor = arm.motors['m4']
        
        # Enable motor
        print("Enabling m4 motor...")
        arm.mc.enable(motor)
        time.sleep(0.5)
        
        # Get current position
        arm.mc.refresh_motor_status(motor)
        start_pos_rad = motor.getPosition()
        start_pos_deg = math.degrees(start_pos_rad)
        
        # Calculate target position (current + 90 degrees)
        target_pos_deg = start_pos_deg +90 
        target_pos_rad = math.radians(target_pos_deg)
        
        print(f"Start position: {start_pos_deg:.2f}°")
        print(f"Target position: {target_pos_deg:.2f}°")
        print("Starting 5-second rotation...")
        
        # Control parameters
        kp = 54  # Position gain
        kd = 0.0   # Damping gain
        
        # Time-based control
        input('start?')
        start_time = time.time()
        duration = 1.0  # 5 seconds
        
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Check if 5 seconds have passed
            if elapsed_time >= duration:
                print(f"\n5 seconds completed!")
                break
            
            # Calculate desired position based on time (linear interpolation)
            progress = elapsed_time / duration  # 0 to 1
            # Use smooth S-curve for better motion profile
            smooth_progress = 3 * progress**2 - 2 * progress**3  # Smooth S-curve
            
            desired_pos_rad = start_pos_rad + (target_pos_rad - start_pos_rad) * smooth_progress
            
            # Get current position
            arm.mc.refresh_motor_status(motor)
            current_pos_rad = motor.getPosition()
            
            # Calculate error
            error = desired_pos_rad - current_pos_rad
            
            # Calculate desired velocity (derivative of desired position)
            if elapsed_time < duration:
                # Velocity from S-curve derivative
                velocity_factor = 6 * progress * (1 - progress) / duration
                desired_vel = (target_pos_rad - start_pos_rad) * velocity_factor
            else:
                desired_vel = 0.0
            
            # Send control command
            arm.mc.controlMIT(motor, kp, kd, desired_pos_rad, desired_vel, 0.0)
            
            # Show progress
            current_deg = math.degrees(current_pos_rad)
            desired_deg = math.degrees(desired_pos_rad)
            progress_percent = progress * 100
            
            print(f"\rTime: {elapsed_time:.1f}s ({progress_percent:.0f}%) | "
                  f"Current: {current_deg:.1f}° | Target: {desired_deg:.1f}°", 
                  end="", flush=True)
            
            time.sleep(0.02)  # 50Hz control loop
        
        # Final position check
        arm.mc.refresh_motor_status(motor)
        final_pos_rad = motor.getPosition()
        final_pos_deg = math.degrees(final_pos_rad)
        actual_rotation = final_pos_deg - start_pos_deg
        
        print(f"\nRotation completed!")
        print(f"Final position: {final_pos_deg:.2f}°")
        print(f"Actual rotation: {actual_rotation:.2f}° (target: 90.0°)")
        print(f"Error: {abs(actual_rotation - 90.0):.2f}°")
        
        # Hold final position briefly
        for _ in range(25):  # Hold for 0.5 seconds
            arm.mc.controlMIT(motor, kp, kd, target_pos_rad, 0.0, 0.0)
            time.sleep(0.02)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Always disable motor and close connection
        try:
            arm.mc.disable(motor)
            print("m4 motor disabled.")
        except:
            pass
        arm.close()

if __name__ == "__main__":
    rotate_m4_90_degrees_in_5s()
