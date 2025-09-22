import sys
import os
import math
import time
import struct
import signal
import threading
from enum import IntEnum
from typing import Optional, List, Dict
from dataclasses import dataclass

# Add the current directory to the path to find the src module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src import usb_class, can_value_type
from ht_motor import HTMotor, HTMotorManager

# Flag to control the main loop
running = True
usb_hw = None

# Signal handler for Ctrl+C
def signal_handler(signum, frame):
    global running, usb_hw
    print("\nInterrupt signal received. Exiting...")
    running = False
    
    # Properly close the USB connection if it exists
    if usb_hw is not None:
        try:
            print("Stopping CAN capture...")
            usb_hw.USB_CMD_STOP_CAP()
            global motor_manager
            motor_manager.disable_all()
            print("CAN capture stopped")
        except Exception as e:
            print(f"Error stopping CAN capture: {e}")
    
    # Force exit the program
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Initialize USB connection
usb_hw = usb_class(1000000, 5000000, "F561E08C892274DB09496BCC1102DBC5")

# Start CAN capture
result = usb_hw.USB_CMD_START_CAP()
if result != 0:  # Assuming 0 is ACK_PACK in Python binding
    print(f"Failed to start CAN capture, error code: {result}")
    sys.exit(1)

print("CAN capture started successfully")
print("======================================")
print("HT Motor (4438_30) MIT Control Demo")
print("Press Ctrl+C to exit")
print("======================================")

# Create motor manager
motor_manager = HTMotorManager(usb_hw)

# Define motor IDs to use
motor_ids = [7, 8 ]  # Using motors with IDs 1, 2, and 3

# Add motors to the manager
for motor_id in motor_ids:
    motor_manager.add_motor(motor_id)

# Initialize all motors
print("Initializing motors...")
if not motor_manager.init_all():
    print("Failed to initialize all motors")
    usb_hw.USB_CMD_STOP_CAP()
    sys.exit(1)

print("Motors initialized successfully")

# Function to generate a sine wave trajectory
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

# MIT control parameters
kp = 40.0  # Position gain
kd = 2.0   # Velocity gain
# Main control loop
kp_list = [40, 40 ]
kd_list = [0.5, 0.5]
# kp_list = [0, 0 ]
# kd_list = [0, 0]
try:
    start_time = time.time()
    
    while running:
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Generate trajectory for each motor with different phase offsets
        pos_list = []
        vel_list = []
        torque_list = []
        motor_manager._send_raw(0x8094, [0x00]*20+ [0x00, 0x00, 0xff, 0xff])       
        # motor_manager._send_raw(0x8007, [0xff, 0xff])       
        # motor_manager._send_raw(0x8008, [0xff, 0xff])       
        time.sleep(0.001)

        # for i, motor_id in enumerate(motor_ids):
        #     # Generate sine wave with phase offset for each motor
        #     phase_offset = i * (2 * math.pi / len(motor_ids))
        #     position, velocity = generate_sine_trajectory(elapsed_time + phase_offset)
        #
        #     # For initial testing, use zero position/velocity
        #     # position, velocity = 0, 0
        #
        #     # Add values to lists
        #     print('target pos is ', position)
        #     print('target vel is ', velocity)
        #     pos_list.append(position)
        #     vel_list.append(velocity)
        #     torque_list.append(0.0)  # No feed-forward torque
        #
        #     # kp_list.append(kp)
        #     # kd_list.append(kd)
        #
        # # Send MIT control command with lists
        # motor_manager.mit_control(
        #     pos_list=pos_list,
        #     vel_list=vel_list,
        #     torque_list=torque_list,
        #     kp_list=kp_list,
        #     kd_list=kd_list
        # )
        #
        
        # Sleep to maintain control frequency
        # time.sleep(0.005)  # 100 Hz control loop
        
except Exception as e:
    print(f"Error: {e}")
finally:
    # Disable all motors before exiting
    print("Disabling motors...")
    motor_manager.disable_all()
    
    # Stop CAN capture if not already stopped by signal handler
    if running:  # If we didn't exit via Ctrl+C
        usb_hw.USB_CMD_STOP_CAP()

        print("CAN capture stopped")
    print("Program exited safely.")
