#!/usr/bin/env python3
# Test script for HT motors using the integrated damiao.py

import sys
import os
import math
import time
import signal
from damiao import Motor_Control, DmActData, DM_Motor_Type, Control_Mode, HTMotor

# Flag to control the main loop
running = True
motor_control = None

# Signal handler for Ctrl+C
def signal_handler(signum, frame):
    global running, motor_control
    print("\nInterrupt signal received. Exiting...")
    running = False
    
    # Properly close the motor control if it exists
    if motor_control is not None:
        try:
            print("Disabling motors...")
            motor_control.disable_all()
            print("Motors disabled")
        except Exception as e:
            print(f"Error disabling motors: {e}")
    
    # Force exit the program
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

def generate_sine_trajectory(t, amplitude=math.pi/4, frequency=0.5):
    """
    Generate a sine wave trajectory
    
    Args:
        t: Time in seconds
        amplitude: Amplitude in radians
        frequency: Frequency in Hz
    
    Returns:
        position: Position in radians
        velocity: Velocity in rad/s
    """
    position = amplitude * math.sin(2 * math.pi * frequency * t)
    velocity = amplitude * 2 * math.pi * frequency * math.cos(2 * math.pi * frequency * t)
    return position, velocity

def main():
    global motor_control
    
    # Define motor IDs to use
    motor_ids = [7, 8]  # Using motors with IDs 7 and 8
    
    # Create motor data
    motor_data = []
    for motor_id in motor_ids:
        motor_data.append(DmActData(
            motorType=DM_Motor_Type.HT4438_30,
            mode=Control_Mode.HT_MIT_MODE,
            can_id=motor_id,
            mst_id=0  # Not used for HT motors
        ))
    
    # Initialize motor control
    print("Initializing motor control...")
    motor_control = Motor_Control(
        nom_baud=1000000,  # 1 Mbps nominal baud rate
        dat_baud=5000000,  # 5 Mbps data baud rate
        sn="F561E08C892274DB09496BCC1102DBC5",  # USB2FDCAN device serial number
        data_ptr=motor_data
    )
    
    print("Motor control initialized")
    
    # MIT control parameters
    kp_list = [40, 40]  # Position gains for each motor
    kd_list = [0.5, 0.5]  # Velocity gains for each motor
    
    # Main control loop
    try:
        start_time = time.time()
        
        while running:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Generate trajectory for each motor with different phase offsets
            pos_list = []
            vel_list = []
            torque_list = []
            
            for i, motor_id in enumerate(motor_ids):
                # Generate sine wave with phase offset for each motor
                phase_offset = i * (2 * math.pi / len(motor_ids))
                position, velocity = generate_sine_trajectory(elapsed_time + phase_offset)
                
                print(f"Motor {motor_id} - Position: {position:.4f} rad, Velocity: {velocity:.4f} rad/s")
                
                # Add values to lists
                pos_list.append(position)
                vel_list.append(velocity)
                torque_list.append(0.0)  # No feed-forward torque
            
            # Send MIT control command with lists
            motor_control.control_ht_mit_multi(
                pos_list=pos_list,
                vel_list=vel_list,
                torque_list=torque_list,
                kp_list=kp_list,
                kd_list=kd_list
            )
            
            # Sleep to maintain control frequency
            time.sleep(0.01)  # 100 Hz control loop
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Disable all motors before exiting
        if motor_control is not None:
            print("Disabling motors...")
            motor_control.disable_all()
            print("Motors disabled")
        
        print("Program exited safely.")

if __name__ == "__main__":
    main()
