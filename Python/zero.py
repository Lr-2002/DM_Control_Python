import sys
import os
import time
import signal

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
print("HT Motor Zero Position Setting")
print("Press Ctrl+C to exit")
print("======================================")

# Create motor manager
motor_manager = HTMotorManager(usb_hw)

# Define motor IDs to use
motor_ids = [7, 8]  # Using motors with IDs 7 and 8

# Add motors to the manager
for motor_id in motor_ids:
    motor_manager.add_motor(motor_id)

try:
    # Set zero position for all motors
    print("Setting zero position for all motors...")
    motor_manager.set_zero_position()
    
    # Wait for a moment to ensure the command is processed
    time.sleep(2)
    
    print("Zero position set successfully!")
    print("Please restart the motors for the changes to take effect.")
    
except Exception as e:
    print(f"Error: {e}")
finally:
    # Stop CAN capture if not already stopped by signal handler
    if running:  # If we didn't exit via Ctrl+C
        usb_hw.USB_CMD_STOP_CAP()
        print("CAN capture stopped")
    
    print("Program exited safely.")
