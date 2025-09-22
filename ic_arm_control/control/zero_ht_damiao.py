#!/usr/bin/env python3
# Script to set zero position for HT motors using the integrated damiao.py

import sys
import os
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
    
    try:
        # Set zero position for all HT motors
        print("Setting zero position for all HT motors...")
        motor_control.set_zero_position_all_ht()
        
        # Wait for a moment to ensure the command is processed
        time.sleep(2)
        
        print("Zero position set successfully!")
        print("Please restart the motors for the changes to take effect.")
        
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
