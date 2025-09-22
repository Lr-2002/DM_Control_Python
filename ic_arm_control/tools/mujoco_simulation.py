#!/usr/bin/env python3
"""
MuJoCo Simulation for IC ARM
Reads joint angles from IC_ARM.py and syncs them with URDF model
"""

import os
import time
import numpy as np
import mujoco as mj
import mujoco.viewer as viewer
from ic_arm_control.control.IC_ARM import ICARM
import threading
import signal
import time
import os
import sys
import math

class MuJoCoICARMSimulation:
	def __init__(self, urdf_path=None, update_rate=30.0):
		"""
		Initialize MuJoCo simulation with IC ARM synchronization
		
		Args:
			urdf_path: Path to URDF file (default: auto-detect)
			update_rate: Update frequency in Hz
		"""
		self.update_rate = update_rate
		self.running = False
		
		# Auto-detect URDF path if not provided
		if urdf_path is None:
			current_dir = os.path.dirname(os.path.abspath(__file__))
			# urdf_path = os.path.join(current_dir, 'ic_arm_urdf', 'urdf', 'ic1.1.2.urdf')
			# urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/robot/urdf/robot.urdf"
			urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/robot_8dof/urdf/robot_8dof.urdf"
		self.urdf_path = urdf_path
		
		# Initialize IC ARM connection
		try:
			print("Initializing IC ARM connection...")
			self.ic_arm = ICARM()
			print("IC ARM connected successfully!")
		except Exception as e:
			print(f"Warning: Could not connect to IC ARM: {e}")
			print("Simulation will run without real hardware sync")
			self.ic_arm = None
		
		# Load MuJoCo model
		self.load_mujoco_model()
		
		# Joint mapping from IC ARM motors to MuJoCo joints
		self.joint_mapping = {
			'm1': 'joint1',  # Motor 1 -> Joint 1
			'm2': 'joint2',  # Motor 2 -> Joint 2
			'm3': 'joint3',  # Motor 3 -> Joint 3
			'm4': 'joint4',  # Motor 4 -> Joint 4
			'm5': 'joint5',   # Motor 5 -> Joint 5
			"m6": 'joint6'
		}
		
		# Get joint indices in MuJoCo model
		self.joint_indices = {}
		for motor_name, joint_name in self.joint_mapping.items():
			try:
				joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
				self.joint_indices[motor_name] = joint_id
				print(f"Mapped {motor_name} -> {joint_name} (ID: {joint_id})")
			except Exception as e:
				print(f"Warning: Could not find joint {joint_name}: {e}")
		
		print(f"MuJoCo simulation initialized with {len(self.joint_indices)} joints")
	
	def load_mujoco_model(self):
		"""Load URDF model into MuJoCo"""
		try:
			print(f"Loading URDF model from: {self.urdf_path}")
			
			# Check if URDF file exists
			if not os.path.exists(self.urdf_path):
				raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")
			
			# Change to URDF directory so mesh files can be found
			original_cwd = os.getcwd()
			urdf_dir = os.path.dirname(self.urdf_path)
			os.chdir(urdf_dir)
			
			try:
				# Load model from URDF
				self.model = mj.MjModel.from_xml_path(self.urdf_path)
				self.data = mj.MjData(self.model)
			finally:
				# Always restore original working directory
				os.chdir(original_cwd)

			print(f"Model loaded successfully!")
			print(f"Number of joints: {self.model.njnt}")
			print(f"Number of DOFs: {self.model.nv}")
			
			# Print joint names for debugging
			print("\nJoint names in model:")
			for i in range(self.model.njnt):
				joint_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, i)
				print(f"  {i}: {joint_name}")
			
		except Exception as e:
			print(f"Error loading URDF model: {e}")
			print("Creating a simple fallback model...")
			
			# Create a simple fallback model if URDF loading fails
			xml_string = """
			<mujoco>
				<worldbody>
					<body name="base">
						<geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
						<joint name="j1" type="hinge" axis="0 0 1"/>
						<body name="link1" pos="0.2 0 0">
							<geom type="cylinder" size="0.05 0.1" rgba="0 1 0 1"/>
							<joint name="j2" type="hinge" axis="0 1 0"/>
							<body name="link2" pos="0.15 0 0">
								<geom type="cylinder" size="0.04 0.08" rgba="0 0 1 1"/>
								<joint name="j3" type="hinge" axis="1 0 0"/>
								<body name="link3" pos="0.12 0 0">
									<geom type="cylinder" size="0.03 0.06" rgba="1 1 0 1"/>
									<joint name="j4" type="hinge" axis="0 1 0"/>
									<body name="link4" pos="0.08 0 0">
										<geom type="cylinder" size="0.02 0.04" rgba="1 0 1 1"/>
										<joint name="j5" type="hinge" axis="0 0 1"/>
									</body>
								</body>
							</body>
						</body>
					</body>
				</worldbody>
			</mujoco>
			"""
			self.model = mj.MjModel.from_xml_string(xml_string)
			self.data = mj.MjData(self.model)

	
	def read_ic_arm_positions(self):
		"""Read current joint positions from IC ARM"""
		if self.ic_arm is None:
			# Return dummy data if no hardware connection
			return {
				'm1': {'rad': 0.0, 'deg': 0.0},
				'm2': {'rad': 0.0, 'deg': 0.0},
				'm3': {'rad': 0.0, 'deg': 0.0},
				'm4': {'rad': 0.0, 'deg': 0.0},
				'm5': {'rad': 0.0, 'deg': 0.0},
				'm6': {'rad': 0.0, 'deg': 0.0}
			}
		
		try:
			# 直接读取每个电机的位置，确保获取最新数据
			positions = {}
			motor_names = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']
			
			for motor_name in motor_names:
				try:
					# 直接调用 _read_motor_position_raw 确保获取最新位置
					rad_value = self.ic_arm._read_motor_position_raw(motor_name)
					deg_value = float(np.degrees(rad_value))
					positions[motor_name] = {'rad': rad_value, 'deg': deg_value}
					print(f"  {motor_name}: {deg_value:.2f}° ({rad_value:.4f} rad)")
				except Exception as e:
					print(f"  Error reading {motor_name}: {e}")
					positions[motor_name] = {'rad': 0.0, 'deg': 0.0}
			
			return positions
		except Exception as e:
			print(f"Error reading IC ARM positions: {e}")
			return {}
	
	def sync_positions_to_mujoco(self, ic_arm_positions):
		"""Sync IC ARM positions to MuJoCo simulation"""
		print(f"Syncing {len(ic_arm_positions)} positions to MuJoCo...")
		for motor_name, joint_idx in self.joint_indices.items():
			if motor_name in ic_arm_positions:
				position_data = ic_arm_positions[motor_name]
				
				# Check if position data is valid
				if isinstance(position_data, dict) and 'rad' in position_data:
					angle_rad = position_data['rad']
					
					# Skip if angle is None or invalid
					if angle_rad is not None:
						try:
							# Set joint position in MuJoCo
							old_pos = self.data.qpos[joint_idx]
							self.data.qpos[joint_idx] = angle_rad
							print(f"  {motor_name} -> joint_idx={joint_idx}: {old_pos:.4f} -> {angle_rad:.4f}")
						except Exception as e:
							print(f"Error setting joint {motor_name}: {e}")
					else:
						print(f"  {motor_name}: angle_rad is None")
				else:
					print(f"  {motor_name}: invalid position data format")
			else:
				print(f"  {motor_name}: not found in ic_arm_positions")
	
	def update_simulation(self):
		"""Update simulation with current IC ARM positions"""
		# Read positions from IC ARM
		ic_arm_positions = self.read_ic_arm_positions()
		
		# Sync positions to MuJoCo
		self.sync_positions_to_mujoco(ic_arm_positions)
		
		# Update kinematics to reflect new joint positions
		mj.mj_kinematics(self.model, self.data)
		
		# Forward simulation one step (for physics)
		mj.mj_forward(self.model, self.data)
		
		return ic_arm_positions
	
	def print_status(self, ic_arm_positions):
		"""Print current status"""
		print("\n" + "="*60)
		print("IC ARM -> MuJoCo Sync Status")
		print("="*60)
		print(f"{'Motor':<8} {'IC ARM (rad)':<15} {'IC ARM (deg)':<15} {'MuJoCo (rad)':<15}")
		print("-"*60)
		
		for motor_name, joint_idx in self.joint_indices.items():
			if motor_name in ic_arm_positions:
				position_data = ic_arm_positions[motor_name]
				
				if isinstance(position_data, dict):
					ic_rad = position_data.get('rad', 'N/A')
					ic_deg = position_data.get('deg', 'N/A')
					
					try:
						mj_rad = self.data.qpos[joint_idx]
					except:
						mj_rad = 'N/A'
					
					print(f"{motor_name:<8} {ic_rad:<15} {ic_deg:<15} {mj_rad:<15}")
				else:
					print(f"{motor_name:<8} {'ERROR':<15} {'ERROR':<15} {'ERROR':<15}")
		
		print("="*60)
	
	def run_headless(self, duration=None, print_interval=2.0):
		"""Run simulation without viewer (headless mode)"""
		print(f"Starting headless simulation at {self.update_rate} Hz")
		print("Press Ctrl+C to stop")
		
		self.running = True
		start_time = time.time()
		last_print_time = start_time
		
		try:
			while self.running:
				current_time = time.time()
				
				# Update simulation
				ic_arm_positions = self.update_simulation()
				
				# Print status periodically
				if current_time - last_print_time >= print_interval:
					self.print_status(ic_arm_positions)
					last_print_time = current_time
				
				# Check duration limit
				if duration and (current_time - start_time) >= duration:
					break
				
				# Sleep to maintain update rate
				time.sleep(1.0 / self.update_rate)
				
		except KeyboardInterrupt:
			print("\nSimulation stopped by user")
		finally:
			self.running = False
	
	def run_with_viewer(self):
		"""Run simulation with MuJoCo viewer"""
		print(f"Starting simulation with viewer at {self.update_rate} Hz")
		print("Close viewer window or press Ctrl+C to stop")
		
		self.running = True
		last_update_time = time.time()
		update_interval = 1.0 / self.update_rate
		
		try:
			# Launch viewer
			with viewer.launch_passive(self.model, self.data) as v:
				# Enable coordinate frame visualization
				v.opt.frame = mj.mjtFrame.mjFRAME_WORLD  # Show world frame and body frames
				v.opt.label = mj.mjtLabel.mjLABEL_BODY  # Show body labels
				
				# Set camera position and orientation
				# Camera position: (1, 0, 1) looking at (0, 0, 0)
				cam_pos = np.array([1.0, 0.0, -1.0])
				cam_target = np.array([0.0, 0.0, 0.0])
				
				# Calculate camera orientation
				forward = cam_target - cam_pos  # Direction from camera to target
				forward = forward / np.linalg.norm(forward)  # Normalize
				
				# Set camera parameters
				v.cam.lookat[:] = cam_target  # Point to look at
				v.cam.distance = np.linalg.norm(cam_pos - cam_target)  # Distance from target
				
				# Calculate azimuth and elevation angles
				azimuth = np.arctan2(forward[1], forward[0]) * 180 / np.pi
				elevation = np.arcsin(forward[2]) * 180 / np.pi
				
				v.cam.azimuth = azimuth
				v.cam.elevation = -elevation  # Negative because MuJoCo uses different convention
				
				print("MuJoCo Viewer Controls:")
				print("  - Press 'F' to toggle coordinate frames")
				print("  - Press 'L' to toggle labels")
				print("  - Right-click and drag to rotate view")
				print("  - Scroll to zoom")
				print(f"Camera set to position: {cam_pos}, looking at: {cam_target}")
				
				while v.is_running() and self.running:
					current_time = time.time()
					
					# Update simulation at specified rate
					if current_time - last_update_time >= update_interval:
						try:
							self.update_simulation()
							last_update_time = current_time
						except Exception as e:
							print(f"Error in simulation update: {e}")
							break
					
					# Sync viewer (this should be called frequently)
					v.sync()
					
					# Small sleep to prevent excessive CPU usage
					time.sleep(0.001)  # 1ms
					
		except KeyboardInterrupt:
			print("\nReceived interrupt signal, shutting down...")
		except Exception as e:
			print(f"Viewer error: {e}")
		finally:
			self.running = False
	
	def cleanup(self):
		"""Clean up resources"""
		self.running = False
		if hasattr(self, 'ic_arm') and self.ic_arm:
			try:
				# Disable all motors for safety
				self.ic_arm.disable_all_motors()
			except:
				pass
		print("Simulation cleaned up")

def signal_handler(signum, frame):
	"""Handle Ctrl+C gracefully"""
	print("\nReceived interrupt signal, shutting down...")
	sys.exit(0)

def main():
	"""Main function"""
	import argparse
	
	parser = argparse.ArgumentParser(description='MuJoCo IC ARM Simulation')
	parser.add_argument('--headless', action='store_true', 
					   help='Run without viewer (headless mode)')
	parser.add_argument('--duration', type=float, default=None,
					   help='Duration to run simulation (seconds)')
	parser.add_argument('--rate', type=float, default=30.0,
					   help='Update rate in Hz (default: 30)')
	parser.add_argument('--urdf', type=str, default=None,
					   help='Path to URDF file')
	
	args = parser.parse_args()
	
	# Set up signal handler
	signal.signal(signal.SIGINT, signal_handler)
	
	try:
		# Create simulation
		sim = MuJoCoICARMSimulation(urdf_path=args.urdf, update_rate=args.rate)
		
		if args.headless:
			sim.run_headless(duration=args.duration)
		else:
			sim.run_with_viewer()
			
	except Exception as e:
		print(f"Error running simulation: {e}")
	finally:
		if 'sim' in locals():
			sim.cleanup()

if __name__ == "__main__":
	main()