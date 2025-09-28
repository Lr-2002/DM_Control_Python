"""
LeRobot Integration Dataset Collection System
Based on Hugging Face LeRobot architecture for robot arm data collection
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime

# Import IC_ARM control system
from ic_arm_control.control.IC_ARM import IC_ARM
from ic_arm_control.control.unified_motor_control import MotorManager, MotorType

@dataclass
class EpisodeData:
    """Single episode data container"""
    episode_id: str
    timestamp: float
    actions: List[np.ndarray]  # Motor commands
    observations: Dict[str, List[np.ndarray]]  # Sensor readings including angles
    rewards: Optional[List[float]] = None
    dones: Optional[List[bool]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class DatasetConfig:
    """Dataset collection configuration"""
    dataset_name: str
    collection_rate: float = 100.0  # Hz
    max_episodes: Optional[int] = None
    episode_duration: Optional[float] = None  # seconds
    save_format: str = "hdf5"  # hdf5, npz, or json
    compression: str = "gzip"
    include_images: bool = False
    include_torque: bool = True
    include_velocity: bool = True

class LeRobotDatasetCollector:
    """
    Main dataset collection class following LeRobot architecture
    """

    def __init__(self, config: DatasetConfig, ic_arm: IC_ARM):
        self.config = config
        self.ic_arm = ic_arm
        self.dataset_path = Path(f"./datasets/{config.dataset_name}")
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        self.logger = logging.getLogger("lerobot_collector")
        self.logger.setLevel(logging.INFO)

        # Current episode data
        self.current_episode = None
        self.episode_count = 0
        self.is_collecting = False

        # Initialize data buffers
        self._init_buffers()

    def _init_buffers(self):
        """Initialize data buffers for collection"""
        self.buffers = {
            'actions': [],
            'observations': {
                'joint_angles': [],
                'joint_velocities': [],
                'joint_torques': [],
                'end_effector_pos': [],
                'end_effector_quat': [],
            }
        }

        if self.config.include_images:
            self.buffers['observations']['camera_images'] = []

    def start_episode(self, episode_id: Optional[str] = None) -> str:
        """Start a new data collection episode"""
        if self.is_collecting:
            self.logger.warning("Episode already in progress, stopping current episode")
            self.stop_episode()

        if episode_id is None:
            episode_id = f"episode_{self.episode_count:06d}_{int(time.time())}"

        self.current_episode = EpisodeData(
            episode_id=episode_id,
            timestamp=time.time(),
            actions=[],
            observations={
                'joint_angles': [],
                'joint_velocities': [],
                'joint_torques': [],
                'end_effector_pos': [],
                'end_effector_quat': [],
            },
            metadata={
                'collection_rate': self.config.collection_rate,
                'start_time': datetime.now().isoformat(),
                'motor_info': self._get_motor_info()
            }
        )

        if self.config.include_images:
            self.current_episode.observations['camera_images'] = []

        self.is_collecting = True
        self.logger.info(f"Started episode: {episode_id}")
        return episode_id

    def stop_episode(self) -> Optional[EpisodeData]:
        """Stop current episode and return collected data"""
        if not self.is_collecting:
            self.logger.warning("No episode in progress")
            return None

        self.is_collecting = False
        self.episode_count += 1

        # Add completion metadata
        if self.current_episode:
            self.current_episode.metadata['end_time'] = datetime.now().isoformat()
            self.current_episode.metadata['duration'] = time.time() - self.current_episode.timestamp
            self.current_episode.metadata['num_frames'] = len(self.current_episode.actions)

        self.logger.info(f"Stopped episode: {self.current_episode.episode_id}")
        return self.current_episode

    def collect_frame(self) -> bool:
        """Collect single frame of data"""
        if not self.is_collecting:
            self.logger.warning("No episode in progress")
            return False

        try:
            # Get current joint angles
            joint_angles = self.ic_arm.get_all_joint_angles()
            joint_velocities = self.ic_arm.get_all_joint_velocities()
            joint_torques = self.ic_arm.get_all_joint_torques()

            # Get end effector state
            end_effector_pos, end_effector_quat = self.ic_arm.get_end_effector_pose()

            # Get current action (last command)
            current_action = self.ic_arm.get_last_commands()

            # Store in current episode
            self.current_episode.actions.append(current_action)
            self.current_episode.observations['joint_angles'].append(joint_angles)
            self.current_episode.observations['joint_velocities'].append(joint_velocities)
            self.current_episode.observations['joint_torques'].append(joint_torques)
            self.current_episode.observations['end_effector_pos'].append(end_effector_pos)
            self.current_episode.observations['end_effector_quat'].append(end_effector_quat)

            # Collect images if enabled
            if self.config.include_images:
                images = self._collect_camera_images()
                self.current_episode.observations['camera_images'].append(images)

            return True

        except Exception as e:
            self.logger.error(f"Error collecting frame: {e}")
            return False

    def _collect_camera_images(self) -> Dict[str, np.ndarray]:
        """Collect camera images (placeholder implementation)"""
        # This would integrate with actual camera systems
        return {}

    def _get_motor_info(self) -> Dict[str, Any]:
        """Get motor configuration information"""
        motor_info = {}
        try:
            for motor_name in self.ic_arm.MOTOR_LIST:
                motor = self.ic_arm.motor_manager.get_motor(motor_name)
                if motor:
                    motor_info[motor_name] = {
                        'type': motor.motor_type.value,
                        'limits': getattr(motor, 'limits', {}),
                        ' calibration': getattr(motor, 'calibration', {})
                    }
        except Exception as e:
            self.logger.error(f"Error getting motor info: {e}")
        return motor_info

    def save_episode(self, episode: EpisodeData) -> str:
        """Save episode data to disk"""
        episode_dir = self.dataset_path / episode.episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        if self.config.save_format == "hdf5":
            return self._save_hdf5(episode, episode_dir)
        elif self.config.save_format == "npz":
            return self._save_npz(episode, episode_dir)
        elif self.config.save_format == "json":
            return self._save_json(episode, episode_dir)
        else:
            raise ValueError(f"Unsupported save format: {self.config.save_format}")

    def _save_hdf5(self, episode: EpisodeData, episode_dir: Path) -> str:
        """Save episode in HDF5 format"""
        import h5py

        file_path = episode_dir / "episode_data.hdf5"

        with h5py.File(file_path, 'w') as f:
            # Save metadata
            metadata_group = f.create_group('metadata')
            for key, value in episode.metadata.items():
                metadata_group.attrs[key] = str(value)

            # Save actions
            actions = np.array(episode.actions)
            f.create_dataset('actions', data=actions, compression=self.config.compression)

            # Save observations
            obs_group = f.create_group('observations')
            for obs_key, obs_values in episode.observations.items():
                if obs_values:
                    obs_data = np.array(obs_values)
                    obs_group.create_dataset(obs_key, data=obs_data, compression=self.config.compression)

        self.logger.info(f"Saved episode to {file_path}")
        return str(file_path)

    def _save_npz(self, episode: EpisodeData, episode_dir: Path) -> str:
        """Save episode in NPZ format"""
        file_path = episode_dir / "episode_data.npz"

        save_dict = {
            'actions': np.array(episode.actions),
            'metadata': json.dumps(episode.metadata)
        }

        for obs_key, obs_values in episode.observations.items():
            if obs_values:
                save_dict[f'obs_{obs_key}'] = np.array(obs_values)

        np.savez_compressed(file_path, **save_dict)
        self.logger.info(f"Saved episode to {file_path}")
        return str(file_path)

    def _save_json(self, episode: EpisodeData, episode_dir: Path) -> str:
        """Save episode in JSON format"""
        file_path = episode_dir / "episode_data.json"

        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {
            'episode_id': episode.episode_id,
            'timestamp': episode.timestamp,
            'actions': [action.tolist() for action in episode.actions],
            'observations': {},
            'metadata': episode.metadata
        }

        for obs_key, obs_values in episode.observations.items():
            serializable_data['observations'][obs_key] = [
                obs.tolist() if hasattr(obs, 'tolist') else obs
                for obs in obs_values
            ]

        with open(file_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)

        self.logger.info(f"Saved episode to {file_path}")
        return str(file_path)

    def collect_continuous(self, duration: Optional[float] = None) -> List[str]:
        """Collect data continuously for specified duration"""
        if duration is None:
            duration = self.config.episode_duration

        if duration is None:
            raise ValueError("Duration must be specified")

        episode_id = self.start_episode()
        saved_files = []

        start_time = time.time()
        last_frame_time = start_time
        frame_interval = 1.0 / self.config.collection_rate

        try:
            while time.time() - start_time < duration:
                current_time = time.time()

                if current_time - last_frame_time >= frame_interval:
                    success = self.collect_frame()
                    if success:
                        last_frame_time = current_time

                # Small sleep to prevent CPU overload
                time.sleep(0.001)

        except KeyboardInterrupt:
            self.logger.info("Collection interrupted by user")

        finally:
            episode = self.stop_episode()
            if episode:
                file_path = self.save_episode(episode)
                saved_files.append(file_path)

        return saved_files

    def create_dataset_index(self) -> str:
        """Create dataset index file with all episodes info"""
        index_data = {
            'dataset_name': self.config.dataset_name,
            'total_episodes': self.episode_count,
            'collection_config': asdict(self.config),
            'episodes': [],
            'created_at': datetime.now().isoformat()
        }

        # Scan for all episodes
        for episode_dir in self.dataset_path.iterdir():
            if episode_dir.is_dir():
                episode_id = episode_dir.name
                index_data['episodes'].append({
                    'episode_id': episode_id,
                    'path': str(episode_dir)
                })

        index_path = self.dataset_path / "dataset_index.json"
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)

        self.logger.info(f"Created dataset index at {index_path}")
        return str(index_path)