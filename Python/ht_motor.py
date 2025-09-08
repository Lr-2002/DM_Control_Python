import struct
import time
import math
from enum import IntEnum
import numpy as np
from typing import List, Tuple, Optional
# from damiao import DM_Motor_Type

# Motor control modes
class ControlMode(IntEnum):
    IDLE = 0
    POSITION = 10
    SPEED = 11
    TORQUE = 12
    MIT = 20


def convert_to_hex(data: list):
    if isinstance(data, int):
        return f"{data:02x}"
    return " ".join(f"{byte:02x}" for byte in data)


class HTMotor:
    """
    HT Motor (4438_30) control class using CAN-FD protocol
    """

    def __init__(self, usb_hw, motor_id: int = 1, source_id: int = 0):
        """
        Initialize the HT motor controller

        Args:
            usb_hw: USB hardware interface for CAN communication
            motor_id: Motor ID (destination address, 0-127)
            source_id: Source ID (0-127)
        """
        self.usb_hw = usb_hw
        self.motor_id = int(motor_id % 256)  # Ensure 7-bit ID
        self.source_id = int(source_id % 256)  # Ensure 7-bit ID
        # print('the id is ', self.motor_id, self.source_id)

        # Set default values
        self.position = 0.0  # Position in radians
        self.velocity = 0.0  # Velocity in rad/s
        self.torque = 0.0  # Torque in Nm
        self.error = 0  # Error code

        # Constants for torque conversion (4438 motor with 32:1 reduction)
        self.torque_k = 0.004855  # Slope for int16 torque
        self.torque_d = -0.083  # Offset for int16 torque

        # Conversion factors
        self.RAD_TO_TURN = 1.0 / (2.0 * math.pi)  # Convert radians to turns
        self.TURN_TO_RAD = 2.0 * math.pi  # Convert turns to radians

        # MIT mode parameters
        self.kp = 0.0
        self.kd = 0.0
        self.target_pos = 0.0
        self.target_vel = 0.0
        self.target_torque = 0.0

        # Initialize the motor
        self.initialized = False

    def init(self) -> bool:
        """Initialize the motor"""
        try:
            # Enable the motor
            self.enable()
            time.sleep(0.1)

            # Set to MIT mode
            self.set_mode(ControlMode.MIT)
            time.sleep(0.1)

            # Read initial state
            self.read_state()

            self.initialized = True
            return True
        except Exception as e:
            print(f"Error initializing motor: {e}")
            return False

    def enable(self) -> None:
        """Enable the motor"""
        pass

    def disable(self) -> None:
        """Disable the motor"""
        pass

    def set_mode(self, mode: ControlMode) -> None:
        """Set the motor control mode"""
        pass

    def read_state(self) -> None:
        """Read the motor state (position, velocity, torque)"""
        # Create command to read registers 0x20-0x22 (position, velocity, torque)
        # Using mode 2 format (0x18 = read + int32 + mode2, 0x03 = 3 values, 0x20 = start register)
        cmd = [
            0xff,
            0xff,
            # 0x00,
        ]

        # Send with reply expected (set highest bit of source ID)
        self._send_command(cmd, expect_reply=True)

        # Note: The actual data will be processed in the callback function

    def mit_control(
        self, position: float, velocity: float, kp: float, kd: float, torque_ff: float
    ) -> None:
        """
        MIT mode control

        Args:
            position: Target position in radians
            velocity: Target velocity in rad/s
            kp: Position gain
            kd: Velocity gain
            torque_ff: Feed-forward torque in Nm
        """
        if not self.initialized:
            print("Motor not initialized")
            return

        # Save MIT parameters
        self.target_pos = position
        self.target_vel = velocity
        self.kp = kp
        self.kd = kd
        self.target_torque = torque_ff

        # Convert to motor units
        pos_turns = position * self.RAD_TO_TURN
        vel_turns = velocity * self.RAD_TO_TURN

        # Convert to int16 format
        pos_int = int(pos_turns / 0.0001)  # LSB = 0.0001 turns
        vel_int = int(vel_turns / 0.00025)  # LSB = 0.00025 turns/s
        kp_int = int(kp * 10)  # LSB = 0.1
        kd_int = int(kd * 10)  # LSB = 0.1

        # Convert torque from Nm to motor units (reverse the torque equation)
        torque_int = int((torque_ff - self.torque_d) / self.torque_k)

        # Clamp values to int16 range
        pos_int = max(min(pos_int, 32767), -32768)
        vel_int = max(min(vel_int, 32767), -32768)
        kp_int = max(min(kp_int, 32767), -32768)
        kd_int = max(min(kd_int, 32767), -32768)
        torque_int = max(min(torque_int, 32767), -32768)

        # Pack as little-endian int16 values
        pos_bytes = struct.pack("<h", pos_int)
        vel_bytes = struct.pack("<h", vel_int)
        kp_bytes = struct.pack("<h", kp_int)
        kd_bytes = struct.pack("<h", kd_int)
        torque_bytes = struct.pack("<h", torque_int)

        # Create MIT control command (using mode 3 format for one-to-many)
        # For ID 0x8093 (MIT control with PD gains)
        data = bytearray()
        data.extend(pos_bytes)  # Position (2 bytes)
        data.extend(vel_bytes)  # Velocity (2 bytes)
        data.extend(torque_bytes)  # Torque (2 bytes)
        data.extend(kp_bytes)  # Kp (2 bytes)
        data.extend(kd_bytes)  # Kd (2 bytes)

        # Add padding to fill the rest of the motor slots (if needed)
        # Add status query command at the end (0x14, 0x04, 0x00)
        data.extend([0x00, 0x00, 0xFF, 0xFF])

        self._send_raw(0x8094, list(data))

    def process_can_frame(self, frame) -> None:
        """
        Process received CAN frame

        Args:
            frame: CAN frame object with head.id, head.dlc, and data[] properties
        """
        # Check if this frame is for us
        frame_id = frame.head.id

        # Check if this is a motor status frame (0x700 + motor_id or 0x800 + motor_id)
        if frame_id == 0x700 or frame_id == 0x800:
            # Process motor status data
            self._process_motor_status(frame)
            return

        # Check if this is a reply to our command
        if (frame_id >> 8) == self.motor_id and (frame_id & 0xFF) == self.source_id:
            # Process reply data
            self._process_reply(frame)
            return

    def _process_motor_status(self, frame) -> None:
        """Process motor status frame (0x700/0x800 series)

        For HT motor, the frame contains:
        - byte 0: error code
        - bytes 1-2: position (int16, little-endian)
        - bytes 3-4: velocity (int16, little-endian)
        - bytes 5-6: torque (int16, little-endian)
        """
        # Extract data from the frame
        if frame.head.dlc < 7:  # Need at least 7 bytes for all data
            return  # Not enough data

        # Extract error code (first byte)
        self.error = frame.data[0]

        # Extract position, velocity, and torque as int16 values (little-endian)
        pos_int = struct.unpack("<h", bytes([frame.data[2], frame.data[3]]))[0]
        vel_int = struct.unpack("<h", bytes([frame.data[4], frame.data[5]]))[0]
        torque_int = struct.unpack("<h", bytes([frame.data[6], frame.data[7]]))[0]

        # Convert to physical units
        # print('pos_int is ', pos_int)
        self.position = pos_int * 0.0001 * self.TURN_TO_RAD  # Convert to radians
        self.velocity = vel_int * 0.00025 * self.TURN_TO_RAD  # Convert to rad/s
        self.torque = torque_int * self.torque_k + self.torque_d  # Convert to Nm

        # Print status
        # print(
        #     f"Motor {self.motor_id} Status: Error={self.error}, "
        #     f"Pos={self.position:.4f} rad, "
        #     f"Vel={self.velocity:.4f} rad/s, "
        #     f"Torque={self.torque:.4f} Nm"
        # )
        # print('data is ', convert_to_hex(frame.data))

    def _process_reply(self, frame) -> None:
        """Process reply to a command"""
        if frame.head.dlc < 3:
            return  # Not enough data

        # Check the command type (first byte)
        cmd = frame.data[0]
        cmd_type = (cmd >> 4) & 0xF  # Extract high 4 bits

        if cmd_type != 0x2:  # 0x2 = reply
            return

        data_type = (cmd >> 2) & 0x3  # Extract bits 2-3

        if data_type == 0x2:  # int32
            # Process int32 data
            self._process_int32_reply(frame)
        elif data_type == 0x1:  # int16
            # Process int16 data
            self._process_int16_reply(frame)

    def _process_int32_reply(self, frame) -> None:
        """Process int32 reply data"""
        # Check if this is mode 2 format
        if (frame.data[0] & 0x3) == 0:
            # Mode 2: cmd, num, addr, data...
            num_values = frame.data[1]
            start_addr = frame.data[2]

            # Process each value
            for i in range(num_values):
                if 3 + i * 4 + 3 >= frame.head.dlc:
                    break  # Not enough data

                # Extract int32 value (little-endian)
                value = struct.unpack(
                    "<i",
                    bytes(
                        [
                            frame.data[3 + i * 4],
                            frame.data[3 + i * 4 + 1],
                            frame.data[3 + i * 4 + 2],
                            frame.data[3 + i * 4 + 3],
                        ]
                    ),
                )[0]

                # Process based on register address
                self._update_value_from_register(start_addr + i, value, "int32")
        else:
            # Mode 1: cmd, addr, data...
            num_values = frame.data[0] & 0x3
            if num_values == 0:
                num_values = 3  # Special case

            start_addr = frame.data[1]

            # Process each value
            for i in range(num_values):
                if 2 + i * 4 + 3 >= frame.head.dlc:
                    break  # Not enough data

                # Extract int32 value (little-endian)
                value = struct.unpack(
                    "<i",
                    bytes(
                        [
                            frame.data[2 + i * 4],
                            frame.data[2 + i * 4 + 1],
                            frame.data[2 + i * 4 + 2],
                            frame.data[2 + i * 4 + 3],
                        ]
                    ),
                )[0]

                # Process based on register address
                self._update_value_from_register(start_addr + i, value, "int32")
    def GetMotorType(self):
        return DM_Motor_Type.HT4438

    def _process_int16_reply(self, frame) -> None:
        """Process int16 reply data"""
        # Check if this is mode 2 format
        if (frame.data[0] & 0x3) == 0:
            # Mode 2: cmd, num, addr, data...
            num_values = frame.data[1]
            start_addr = frame.data[2]

            # Process each value
            for i in range(num_values):
                if 3 + i * 2 + 1 >= frame.head.dlc:
                    break  # Not enough data

                # Extract int16 value (little-endian)
                value = struct.unpack(
                    "<h", bytes([frame.data[3 + i * 2], frame.data[3 + i * 2 + 1]])
                )[0]

                # Process based on register address
                self._update_value_from_register(start_addr + i, value, "int16")
        else:
            # Mode 1: cmd, addr, data...
            num_values = frame.data[0] & 0x3
            if num_values == 0:
                num_values = 3  # Special case

            start_addr = frame.data[1]

            # Process each value
            for i in range(num_values):
                if 2 + i * 2 + 1 >= frame.head.dlc:
                    break  # Not enough data

                # Extract int16 value (little-endian)
                value = struct.unpack(
                    "<h", bytes([frame.data[2 + i * 2], frame.data[2 + i * 2 + 1]])
                )[0]

                # Process based on register address
                self._update_value_from_register(start_addr + i, value, "int16")

    def _update_value_from_register(
        self, register: int, value: int, data_type: str
    ) -> None:
        """Update internal state based on register value"""

    def _send_command(self, cmd: List[int], expect_reply: bool = False) -> None:
        """
        Send a command to the motor

        Args:
            cmd: Command bytes
            expect_reply: Whether to expect a reply
        """
        # Create CAN ID
        # Set highest bit of source ID to request reply
        can_id = ((0x80 | self.source_id) << 8) | self.motor_id

        # Send the command
        self._send_raw(can_id, cmd)

    def _send_raw(self, can_id: int, data: List[int]) -> None:
        """
        Send raw CAN frame

        Args:
            can_id: CAN ID
            data: Data bytes
        """
        try:
            # print('data is ', convert_to_hex(data))
            # print('can_id is ', convert_to_hex(can_id))
            self.usb_hw.fdcanFrameSend(data, can_id)
        except Exception as e:
            print(f"Error sending CAN frame: {e}")
    def Get_Position(self):
        return self.position
    def Get_Velocity(self):
        return self.velocity
    def Get_tau(self):
        return self.torque

class HTMotorManager:
    """
    Manager for multiple HT motors
    """

    def __init__(self, usb_hw, as_sub_module=False):
        """
        Initialize the HT motor manager

        Args:
            usb_hw: USB hardware interface for CAN communication
        """
        self.usb_hw = usb_hw
        self.motors = {}  # Dictionary of motors by ID
        self.last_data = [0]*20 
        self.read_extra = [0x00, 0x00, 0xff, 0xff]
        # Set up callback for CAN frames
        # if not as_sub_module:
        #     self.usb_hw.setFrameCallback(self.can_frame_callback)

    def add_motor(self, motor_id: int, source_id: int = 0) -> HTMotor:
        """
        Add a motor to the manager

        Args:
            motor_id: Motor ID
            source_id: Source ID

        Returns:
            HTMotor: The created motor object
        """
        motor = HTMotor(self.usb_hw, motor_id, source_id)
        self.motors[motor_id] = motor
        return motor

    def get_motor(self, motor_id: int) -> Optional[HTMotor]:
        """
        Get a motor by ID

        Args:
            motor_id: Motor ID

        Returns:
            HTMotor: The motor object, or None if not found
        """
        if motor_id >= 10:
            motor_id = int(motor_id / 256)

        return self.motors.get(motor_id)

    def mit_control(
        self,
        pos_list: List[float] = None,
        vel_list: List[float] = None,
        torque_list: List[float] = None,
        kp_list: List[float] = None,
        kd_list: List[float] = None,
    ) -> None:
        """
        Send MIT control command to all motors in the format:
        [p1, v1, t1, kp1, kd1, p2, v2, t2, kp2, kd2, placeholder, 0xff, 0xff]

        Args:
            pos_list: List of target positions in radians for each motor
            vel_list: List of target velocities in rad/s for each motor
            torque_list: List of feed-forward torques in Nm for each motor
            kp_list: List of position gains for each motor
            kd_list: List of velocity gains for each motor
        """
        # Constants for conversion
        RAD_TO_TURN = 1.0 / (2.0 * math.pi)  # Convert radians to turns

        # Torque conversion constants for 4438 motor with 32:1 reduction
        torque_k = 0.004855  # Slope for int16 torque
        torque_d = -0  # Offset for int16 torque

        # Get sorted list of motor IDs
        motor_ids = sorted(self.motors.keys())
        num_motors = len(motor_ids)

        # Initialize default values if lists are not provided
        if pos_list is None:
            pos_list = [0.0] * num_motors
        if vel_list is None:
            vel_list = [0.0] * num_motors
        if torque_list is None:
            torque_list = [0.0] * num_motors
        if kp_list is None:
            kp_list = [0.0] * num_motors
        if kd_list is None:
            kd_list = [0.0] * num_motors

        # Ensure all lists have the correct length
        if len(pos_list) < num_motors:
            pos_list.extend([0.0] * (num_motors - len(pos_list)))
        if len(vel_list) < num_motors:
            vel_list.extend([0.0] * (num_motors - len(vel_list)))
        if len(torque_list) < num_motors:
            torque_list.extend([0.0] * (num_motors - len(torque_list)))
        if len(kp_list) < num_motors:
            kp_list.extend([0.0] * (num_motors - len(kp_list)))
        if len(kd_list) < num_motors:
            kd_list.extend([0.0] * (num_motors - len(kd_list)))

        # Create data array for all motors
        data = bytearray()

        # Add data for each motor in the format [p, v, t, kp, kd]
        for i, motor_id in enumerate(motor_ids):
            # Get values for this motor
            position = pos_list[i]
            velocity = vel_list[i]
            torque_ff = torque_list[i]
            kp = kp_list[i]
            kd = kd_list[i]

            # Convert to motor units
            pos_turns = position * RAD_TO_TURN
            vel_turns = velocity * RAD_TO_TURN

            # Convert to int16 format
            pos_int = int(pos_turns / 0.0001)  # LSB = 0.0001 turns
            vel_int = int(vel_turns / 0.00025)  # LSB = 0.00025 turns/s
            kp_int = int(kp * 10)  # LSB = 0.1
            kd_int = int(kd * 10)  # LSB = 0.1

            # Convert torque from Nm to motor units (reverse the torque equation)
            torque_int = int((torque_ff - torque_d) / torque_k)

            # Clamp values to int16 range
            pos_int = max(min(pos_int, 32767), -32768)
            vel_int = max(min(vel_int, 32767), -32768)
            kp_int = max(min(kp_int, 32767), -32768)
            kd_int = max(min(kd_int, 32767), -32768)
            torque_int = max(min(torque_int, 32767), -32768)

            # Pack as little-endian int16 values
            pos_bytes = struct.pack("<h", pos_int)
            vel_bytes = struct.pack("<h", vel_int)
            kp_bytes = struct.pack("<h", kp_int)
            kd_bytes = struct.pack("<h", kd_int)
            torque_bytes = struct.pack("<h", torque_int)

            # Add data for this motor
            data.extend(pos_bytes)  # Position (2 bytes)
            data.extend(vel_bytes)  # Velocity (2 bytes)
            data.extend(torque_bytes)  # Torque (2 bytes)
            data.extend(kp_bytes)  # Kp (2 bytes)
            data.extend(kd_bytes)  # Kd (2 bytes)

        # Add the final 0xFF 0xFF at the end
        data.extend([0x00, 0x00, 0xFF, 0xFF])

        # Send with ID 0x8094 (MIT control)
        # print(f"Sending MIT control: {convert_to_hex(list(data))}")
        self._send_raw(0x8094, list(data))

    def _send_raw(self, can_id: int, data: List[int]) -> None:
        """Send raw CAN frame"""
        try:
            # print('data is ', convert_to_hex(data))
            # print('can_id is ', convert_to_hex(can_id))
            # print('sending data to ', convert_to_hex(can_id), convert_to_hex(data))
            self.usb_hw.fdcanFrameSend(data, can_id)
        except Exception as e:
            print(f"Error sending CAN frame: {e}")

    def init_all(self) -> bool:
        """
        Initialize all motors

        Returns:
            bool: True if all motors initialized successfully
        """
        success = True
        for motor in self.motors.values():
            if not motor.init():
                success = False
        return success

    def disable_all(self) -> None:
        """Disable all motors"""
        self.brake()
        # for motor in self.motors.values():
        #     motor.disable()

    def can_frame_callback(self, frame) -> None:
        """
        Callback for received CAN frames

        Args:
            frame: CAN frame object
        """
        # Extract motor ID from the frame
        frame_id = frame.head.id
        # Check if this is a motor status frame (0x700 + motor_id or 0x800 + motor_id)
        if frame_id == 0x700:
            # print("Motor status frame received")
            self.get_motor(frame_id).process_can_frame(frame)
        elif frame_id == 0x800:
            # print("Motor reply frame received")
            self.get_motor(frame_id).process_can_frame(frame)

    def brake(self):
        for id, motor in self.motors.items():
            self._send_raw((0x80 | id) << 8 | id, [0x01, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x00, 0x00])
    
    def set_zero_position(self):
        for id, motor in self.motors.items():
            self._send_raw((0x80 | id) << 8 | id, [0x40, 0x01, 0x04, 0x64, 0x20, 0x63, 0x0a])

        print('zero set, please restart ')
    def refresh_motor_status(self):
        self._send_raw(0x8094, self.last_data + self.read_extra)
