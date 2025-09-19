#!/usr/bin/env python3
"""
MuJoCo Trajectory Simulation for IC ARM
Loads and visualizes trajectory files in MuJoCo simulation
"""

import os
import time
import json
import numpy as np
import mujoco as mj
import mujoco.viewer as viewer
import argparse
import sys
from trajectory_generator import TrajectoryGenerator

class MuJoCoTrajectorySimulation:
    def __init__(self, urdf_path=None, update_rate=100.0):
        """
        Initialize MuJoCo trajectory simulation
        
        Args:
            urdf_path: Path to URDF file (default: auto-detect)
            update_rate: Update frequency in Hz
        """
        self.update_rate = update_rate
        self.running = False
        self.trajectory = None
        self.trajectory_index = 0
        self.start_time = None
        self.paused = False
        
        # Initialize trajectory generator for safety validation
        self.trajectory_generator = None
        
        # Auto-detect URDF path if not provided
        if urdf_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Use the updated URDF with safety constraints
            # urdf_path = os.path.join(current_dir, "robot_8dof", "urdf", "robot_8dof_updated.urdf")
            urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/robot_8dof/urdf/robot_8dof_updated.urdf"
            # Fallback to obstacle scene if available
            obstacle_scene = os.path.join(current_dir, "robot_with_obstacle.xml")
            if os.path.exists(obstacle_scene):
                urdf_path = obstacle_scene
                print("Using MuJoCo scene with obstacle for collision visualization")
        self.urdf_path = urdf_path
        
        # Load MuJoCo model
        self.load_mujoco_model()
        
        # Joint mapping from trajectory to MuJoCo joints
        # Now supporting 6 DOF trajectories
        self.joint_mapping = {
            0: 'joint1',  # Motor 1 -> Joint 1
            1: 'joint2',  # Motor 2 -> Joint 2
            2: 'joint3',  # Motor 3 -> Joint 3
            3: 'joint4',  # Motor 4 -> Joint 4
            4: 'joint5',  # Motor 5 -> Joint 5
            5: 'joint6',  # Motor 6 -> Joint 6
        }
        
        # Get joint indices in MuJoCo model
        self.joint_indices = {}
        for motor_idx, joint_name in self.joint_mapping.items():
            try:
                joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
                self.joint_indices[motor_idx] = joint_id
                print(f"Mapped motor_{motor_idx+1} -> {joint_name} (ID: {joint_id})")
            except Exception as e:
                print(f"Warning: Could not find joint {joint_name}: {e}")
        
        print(f"MuJoCo trajectory simulation initialized with {len(self.joint_indices)} joints")
        
        # Initialize trajectory generator for safety validation
        try:
            # Use the actual URDF file for trajectory generator, not the obstacle scene
            urdf_for_validation = self.urdf_path
            if "robot_with_obstacle.xml" in self.urdf_path:
                urdf_for_validation = "/Users/lr-2002/project/instantcreation/IC_arm_control/robot_8dof/urdf/robot_8dof_updated.urdf"
            
            self.trajectory_generator = TrajectoryGenerator(urdf_path=urdf_for_validation)
            print("✓ Trajectory safety validator initialized")
        except Exception as e:
            print(f"Warning: Could not initialize trajectory validator: {e}")
            self.trajectory_generator = None
    
    def load_mujoco_model(self):
        """Load URDF model into MuJoCo"""
        try:
            print(f"Loading URDF model from: {self.urdf_path}")
            
            if not os.path.exists(self.urdf_path):
                raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")
            
            # Change to URDF directory for mesh loading
            original_cwd = os.getcwd()
            urdf_dir = os.path.dirname(self.urdf_path)
            os.chdir(urdf_dir)
            
            try:
                self.model = mj.MjModel.from_xml_path(self.urdf_path)
                self.data = mj.MjData(self.model)
                
                # Add 1m cylindrical obstacle to the loaded model
                self._add_obstacle_to_model()
                
            finally:
                os.chdir(original_cwd)

            print(f"Model loaded successfully!")
            print(f"Number of joints: {self.model.njnt}")
            print(f"Number of DOFs: {self.model.nv}")
            print("✓ Added 1m cylindrical obstacle in -y direction")
            
        except Exception as e:
            print(f"Error loading URDF model: {e}")
            print("Creating a simple fallback model...")
            
            # Create simple fallback model with 6 joints and 1m cylindrical obstacle
            xml_string = """
            <mujoco>
                <worldbody>
                    <!-- Ground plane -->
                    <geom name="floor" pos="0 0 -0.1" size="2 2 0.05" type="box" rgba="0.8 0.8 0.8 1"/>
                    
                    <!-- 1m long cylindrical obstacle in -y direction -->
                    <body name="obstacle" pos="0 -0.5 0.5">
                        <geom name="cylinder_obstacle" 
                              type="cylinder" 
                              size="0.05 0.5" 
                              rgba="1 0 0 0.7"
                              pos="0 0 0"/>
                    </body>
                    
                    <!-- Robot arm -->
                    <body name="base">
                        <geom type="box" size="0.1 0.1 0.1" rgba="0.8 0.8 0.8 1"/>
                        <joint name="joint1" type="hinge" axis="0 0 1"/>
                        <body name="link1" pos="0.2 0 0">
                            <geom type="cylinder" size="0.05 0.1" rgba="1 0 0 1"/>
                            <joint name="joint2" type="hinge" axis="0 1 0"/>
                            <body name="link2" pos="0.15 0 0">
                                <geom type="cylinder" size="0.04 0.08" rgba="0 1 0 1"/>
                                <joint name="joint3" type="hinge" axis="1 0 0"/>
                                <body name="link3" pos="0.12 0 0">
                                    <geom type="cylinder" size="0.03 0.06" rgba="0 0 1 1"/>
                                    <joint name="joint4" type="hinge" axis="0 1 0"/>
                                    <body name="link4" pos="0.08 0 0">
                                        <geom type="cylinder" size="0.02 0.04" rgba="1 1 0 1"/>
                                        <joint name="joint5" type="hinge" axis="0 0 1"/>
                                        <body name="link5" pos="0.06 0 0">
                                            <geom type="cylinder" size="0.015 0.03" rgba="1 0 1 1"/>
                                            <joint name="joint6" type="hinge" axis="1 0 0"/>
                                        </body>
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
    
    def _add_obstacle_to_model(self):
        """Add cylindrical obstacle to existing MuJoCo model"""
        try:
            # Create XML string with obstacle
            obstacle_xml = """
            <mujoco>
                <worldbody>
                    <!-- 1m long cylindrical obstacle in -y direction -->
                    <body name="obstacle" pos="0 -0.5 0.5">
                        <geom name="cylinder_obstacle" 
                              type="cylinder" 
                              size="0.05 0.5" 
                              rgba="1 0 0 0.7"
                              pos="0 0 0"/>
                    </body>
                </worldbody>
            </mujoco>
            """
            
            # Get current model XML
            current_xml = mj.mj_saveLastXML(self.model)
            
            # Parse current XML and add obstacle
            import xml.etree.ElementTree as ET
            
            # Parse current model XML
            root = ET.fromstring(current_xml)
            worldbody = root.find('worldbody')
            
            # Parse obstacle XML
            obstacle_root = ET.fromstring(obstacle_xml)
            obstacle_body = obstacle_root.find('worldbody/body')
            
            # Add obstacle to worldbody
            if worldbody is not None and obstacle_body is not None:
                worldbody.append(obstacle_body)
                
                # Convert back to string and reload model
                modified_xml = ET.tostring(root, encoding='unicode')
                self.model = mj.MjModel.from_xml_string(modified_xml)
                self.data = mj.MjData(self.model)
                
        except Exception as e:
            print(f"Warning: Could not add obstacle to model: {e}")
            # Continue without obstacle if addition fails

    def load_trajectory(self, trajectory_file):
        """Load trajectory from JSON file with safety validation"""
        try:
            print(f"Loading trajectory from: {trajectory_file}")
            
            if not os.path.exists(trajectory_file):
                raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")
            
            with open(trajectory_file, 'r') as f:
                self.trajectory = json.load(f)
            
            # Validate trajectory data
            required_keys = ['time', 'positions']
            for key in required_keys:
                if key not in self.trajectory:
                    raise ValueError(f"Missing required key '{key}' in trajectory")
            
            # Convert to numpy arrays for easier manipulation
            self.trajectory['time'] = np.array(self.trajectory['time'])
            self.trajectory['positions'] = np.array(self.trajectory['positions'])
            
            # Add velocities if not present (compute from positions)
            if 'velocities' not in self.trajectory:
                print("Computing velocities from positions...")
                dt = np.diff(self.trajectory['time'])
                dt = np.append(dt, dt[-1])  # Extend to same length
                velocities = np.diff(self.trajectory['positions'], axis=0) / dt[:-1, np.newaxis]
                velocities = np.vstack([velocities[0], velocities])  # Duplicate first row
                self.trajectory['velocities'] = velocities
            else:
                self.trajectory['velocities'] = np.array(self.trajectory['velocities'])
            
            # Add accelerations if not present
            if 'accelerations' not in self.trajectory:
                print("Computing accelerations from velocities...")
                dt = np.diff(self.trajectory['time'])
                dt = np.append(dt, dt[-1])  # Extend to same length
                accelerations = np.diff(self.trajectory['velocities'], axis=0) / dt[:-1, np.newaxis]
                accelerations = np.vstack([accelerations[0], accelerations])  # Duplicate first row
                self.trajectory['accelerations'] = accelerations
            else:
                self.trajectory['accelerations'] = np.array(self.trajectory['accelerations'])
            
            print(f"✓ Trajectory loaded successfully!")
            print(f"  Duration: {self.trajectory['time'][-1]:.2f}s")
            print(f"  Points: {len(self.trajectory['time'])}")
            print(f"  Motors: {self.trajectory['positions'].shape[1]}")
            
            # Perform safety validation if trajectory generator is available
            if self.trajectory_generator is not None:
                print("Validating trajectory safety...")
                validation = self.trajectory_generator.validate_trajectory_safety(
                    self.trajectory, check_collisions=True
                )
                
                if validation['is_safe']:
                    print("✓ Trajectory passes all safety checks")
                else:
                    print("⚠ Trajectory has safety concerns:")
                    for violation in validation['violations']:
                        print(f"  - {violation}")
                    
                    # Show statistics
                    for motor_key, stats in validation['statistics'].items():
                        motor_id = motor_key.split('_')[1]
                        max_vel = stats['max_abs_vel']
                        pos_range = stats['pos_range']
                        print(f"  Motor {motor_id}: max_vel={max_vel:.3f} rad/s, "
                              f"range=[{np.degrees(pos_range[0]):.1f}°, {np.degrees(pos_range[1]):.1f}°]")
            
            # Reset trajectory playback
            self.trajectory_index = 0
            self.start_time = None
            
            return True
            
        except Exception as e:
            print(f"Error loading trajectory: {e}")
            return False
    
    def get_trajectory_position_at_time(self, current_time):
        """Get trajectory position at given time using interpolation"""
        if self.trajectory is None:
            return None
        
        time_points = self.trajectory['time']
        positions = self.trajectory['positions']
        
        # Clamp time to trajectory bounds
        if current_time <= time_points[0]:
            return positions[0]
        elif current_time >= time_points[-1]:
            return positions[-1]
        
        # Linear interpolation
        idx = np.searchsorted(time_points, current_time)
        if idx == 0:
            return positions[0]
        elif idx >= len(time_points):
            return positions[-1]
        
        # Interpolate between idx-1 and idx
        t0, t1 = time_points[idx-1], time_points[idx]
        p0, p1 = positions[idx-1], positions[idx]
        
        alpha = (current_time - t0) / (t1 - t0)
        return p0 + alpha * (p1 - p0)
    
    def update_simulation(self, current_time=None):
        """Update simulation with trajectory positions"""
        if self.trajectory is None:
            return
        
        if current_time is None:
            if self.start_time is None:
                self.start_time = time.time()
            current_time = time.time() - self.start_time
        
        # Get target positions from trajectory
        target_positions = self.get_trajectory_position_at_time(current_time)
        
        if target_positions is not None:
            # Set joint positions in MuJoCo
            for motor_idx, joint_idx in self.joint_indices.items():
                if motor_idx < len(target_positions):
                    self.data.qpos[joint_idx] = target_positions[motor_idx]
        
        # Update kinematics and physics
        mj.mj_kinematics(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        
        return current_time, target_positions
    
    def print_status(self, current_time, target_positions):
        """Print current trajectory status"""
        if self.trajectory is None or target_positions is None:
            return
        
        total_time = self.trajectory['time'][-1]
        progress = min(100.0, (current_time / total_time) * 100.0)
        
        print(f"\rTime: {current_time:6.2f}s / {total_time:6.2f}s ({progress:5.1f}%) | "
              f"Positions: {[f'{np.degrees(p):6.1f}°' for p in target_positions[:5]]}", end='')
    
    def run_trajectory_headless(self, trajectory_file, speed_factor=1.0, print_interval=0.5):
        """Run trajectory simulation without viewer"""
        if not self.load_trajectory(trajectory_file):
            return False
        
        print(f"Starting headless trajectory playback at {speed_factor}x speed")
        print("Press Ctrl+C to stop")
        
        self.running = True
        self.start_time = time.time()
        last_print_time = 0
        
        try:
            while self.running:
                real_time = time.time() - self.start_time
                trajectory_time = real_time * speed_factor
                
                # Update simulation
                current_time, target_positions = self.update_simulation(trajectory_time)
                
                # Print status periodically
                if real_time - last_print_time >= print_interval:
                    self.print_status(trajectory_time, target_positions)
                    last_print_time = real_time
                
                # Check if trajectory is complete
                if trajectory_time >= self.trajectory['time'][-1]:
                    print(f"\n✓ Trajectory playback completed!")
                    break
                
                # Sleep to maintain real-time playback
                time.sleep(1.0 / self.update_rate)
                
        except KeyboardInterrupt:
            print("\nTrajectory playback stopped by user")
        finally:
            self.running = False
        
        return True
    
    def run_trajectory_with_viewer(self, trajectory_file, speed_factor=1.0):
        """Run trajectory simulation with MuJoCo viewer"""
        if not self.load_trajectory(trajectory_file):
            return False
        
        print(f"Starting trajectory playback with viewer at {speed_factor}x speed")
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  R - Restart trajectory")
        print("  +/- - Increase/Decrease speed")
        print("  ESC or close window - Exit")
        
        self.running = True
        self.paused = False
        self.start_time = None
        pause_start_time = None
        total_pause_time = 0
        
        try:
            with viewer.launch_passive(self.model, self.data) as v:
                # Set up viewer
                v.opt.frame = mj.mjtFrame.mjFRAME_WORLD
                v.opt.label = mj.mjtLabel.mjLABEL_BODY
                
                # Set camera position
                v.cam.lookat[:] = [0.0, 0.0, 0.0]
                v.cam.distance = 1.5
                v.cam.azimuth = 45
                v.cam.elevation = -30
                
                print("Viewer ready. Press SPACE to start trajectory playback...")
                
                while v.is_running() and self.running:
                    current_real_time = time.time()
                    
                    # Handle keyboard input
                    if hasattr(v, 'key_pressed'):
                        # This is a simplified key handling - actual implementation may vary
                        pass
                    
                    if not self.paused:
                        if self.start_time is None:
                            self.start_time = current_real_time
                        
                        # Calculate trajectory time accounting for pauses
                        elapsed_real_time = current_real_time - self.start_time - total_pause_time
                        trajectory_time = elapsed_real_time * speed_factor
                        
                        # Update simulation
                        current_time, target_positions = self.update_simulation(trajectory_time)
                        
                        # Print status
                        if hasattr(self, '_last_status_time'):
                            if current_real_time - self._last_status_time > 0.1:  # Update every 100ms
                                self.print_status(trajectory_time, target_positions)
                                self._last_status_time = current_real_time
                        else:
                            self._last_status_time = current_real_time
                        
                        # Check if trajectory is complete
                        if trajectory_time >= self.trajectory['time'][-1]:
                            print(f"\n✓ Trajectory playback completed! Press R to restart or ESC to exit.")
                            self.paused = True
                    
                    else:
                        if pause_start_time is None:
                            pause_start_time = current_real_time
                    
                    # Sync viewer
                    v.sync()
                    time.sleep(0.001)  # Small sleep to prevent excessive CPU usage
                    
        except KeyboardInterrupt:
            print("\nTrajectory playback interrupted")
        except Exception as e:
            print(f"Viewer error: {e}")
        finally:
            self.running = False
        
        return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='MuJoCo IC ARM Trajectory Simulation')
    parser.add_argument('trajectory_file', 
                       help='Path to trajectory JSON file')
    parser.add_argument('--headless', action='store_true', 
                       help='Run without viewer (headless mode)')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Playback speed factor (default: 1.0)')
    parser.add_argument('--rate', type=float, default=100.0,
                       help='Update rate in Hz (default: 100)')
    parser.add_argument('--urdf', type=str, default="/Users/lr-2002/project/instantcreation/IC_arm_control/robot_8dof/urdf/robot_8dof_updated.urdf",
                       help='Path to URDF file')
    
    args = parser.parse_args()
    
    try:
        # Create simulation
        sim = MuJoCoTrajectorySimulation(urdf_path=args.urdf, update_rate=args.rate)
        
        if args.headless: success = sim.run_trajectory_headless(args.trajectory_file, speed_factor=args.speed)
        else:
            success = sim.run_trajectory_with_viewer(args.trajectory_file, speed_factor=args.speed)
        
        if not success:
            sys.exit(1)
            
    except Exception as e:
        print(f"Error running trajectory simulation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
