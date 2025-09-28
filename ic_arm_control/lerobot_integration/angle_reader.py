"""
LeRobot Angle Reader Module
Specialized for reading and processing joint angles from IC_ARM system
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from collections import deque

@dataclass
class AngleData:
    """Container for angle measurement data"""
    timestamp: float
    joint_angles: np.ndarray
    joint_velocities: np.ndarray
    joint_accelerations: Optional[np.ndarray] = None
    end_effector_pos: Optional[np.ndarray] = None
    end_effector_quat: Optional[np.ndarray] = None
    raw_data: Optional[Dict[str, Any]] = None

class AngleReader:
    """
    High-frequency angle reader for precise joint angle measurements
    """

    def __init__(self,
                 sample_rate: float = 1000.0,
                 buffer_size: int = 1000,
                 smoothing_window: int = 5,
                 enable_acceleration: bool = True):
        """
        Initialize angle reader

        Args:
            sample_rate: Sampling frequency in Hz
            buffer_size: Size of circular buffer for storing angle history
            smoothing_window: Window size for velocity/acceleration smoothing
            enable_acceleration: Whether to compute joint accelerations
        """
        self.sample_rate = sample_rate
        self.sample_interval = 1.0 / sample_rate
        self.buffer_size = buffer_size
        self.smoothing_window = smoothing_window
        self.enable_acceleration = enable_acceleration

        # Initialize data buffers
        self.angle_buffer = deque(maxlen=buffer_size)
        self.velocity_buffer = deque(maxlen=buffer_size)
        self.acceleration_buffer = deque(maxlen=buffer_size) if enable_acceleration else None
        self.timestamp_buffer = deque(maxlen=buffer_size)

        # Initialize logger
        self.logger = logging.getLogger("angle_reader")
        self.logger.setLevel(logging.INFO)

        # Kalman filter parameters for each joint
        self.kalman_filters = {}
        self._init_kalman_filters()

        # Joint limits (in radians)
        self.joint_limits = [
            (-np.pi, np.pi),      # Joint 1
            (-np.pi/2, np.pi/2),  # Joint 2
            (-np.pi, np.pi),      # Joint 3
            (-np.pi, np.pi),      # Joint 4
            (-np.pi/2, np.pi/2),  # Joint 5
            (-np.pi, np.pi),      # Joint 6
            (-np.pi, np.pi),      # Joint 7
            (-np.pi, np.pi),      # Joint 8
        ]

    def _init_kalman_filters(self):
        """Initialize Kalman filters for each joint"""
        class SimpleKalmanFilter:
            def __init__(self, process_variance=1e-5, measurement_variance=1e-4):
                self.process_variance = process_variance
                self.measurement_variance = measurement_variance
                self.state = 0.0
                self.covariance = 1.0

            def update(self, measurement):
                # Prediction
                predicted_state = self.state
                predicted_covariance = self.covariance + self.process_variance

                # Update
                kalman_gain = predicted_covariance / (predicted_covariance + self.measurement_variance)
                self.state = predicted_state + kalman_gain * (measurement - predicted_state)
                self.covariance = (1 - kalman_gain) * predicted_covariance

                return self.state

        for i in range(8):  # 8 joints
            self.kalman_filters[f'joint_{i+1}'] = SimpleKalmanFilter()

    def read_angles(self, ic_arm) -> Optional[AngleData]:
        """
        Read current joint angles from IC_ARM system

        Args:
            ic_arm: IC_ARM instance

        Returns:
            AngleData object with current angle measurements
        """
        try:
            timestamp = time.time()

            # Get raw angles from IC_ARM
            raw_angles = ic_arm.get_all_joint_angles()
            if raw_angles is None:
                self.logger.error("Failed to read angles from IC_ARM")
                return None

            # Apply Kalman filtering
            filtered_angles = np.array([
                self.kalman_filters[f'joint_{i+1}'].update(angle)
                for i, angle in enumerate(raw_angles)
            ])

            # Apply joint limits
            filtered_angles = self._apply_joint_limits(filtered_angles)

            # Calculate velocities
            velocities = self._calculate_velocities(filtered_angles, timestamp)

            # Calculate accelerations if enabled
            accelerations = None
            if self.enable_acceleration:
                accelerations = self._calculate_accelerations(velocities, timestamp)

            # Get end effector position
            end_effector_pos, end_effector_quat = ic_arm.get_end_effector_pose()

            # Create angle data object
            angle_data = AngleData(
                timestamp=timestamp,
                joint_angles=filtered_angles,
                joint_velocities=velocities,
                joint_accelerations=accelerations,
                end_effector_pos=end_effector_pos,
                end_effector_quat=end_effector_quat,
                raw_data={
                    'raw_angles': raw_angles,
                    'filtered_angles': filtered_angles,
                    'kalman_states': [kf.state for kf in self.kalman_filters.values()]
                }
            )

            # Update buffers
            self._update_buffers(angle_data)

            return angle_data

        except Exception as e:
            self.logger.error(f"Error reading angles: {e}")
            return None

    def _apply_joint_limits(self, angles: np.ndarray) -> np.ndarray:
        """Apply joint limits to angle measurements"""
        limited_angles = np.copy(angles)
        for i, (min_limit, max_limit) in enumerate(self.joint_limits):
            if i < len(limited_angles):
                limited_angles[i] = np.clip(limited_angles[i], min_limit, max_limit)
        return limited_angles

    def _calculate_velocities(self, angles: np.ndarray, timestamp: float) -> np.ndarray:
        """Calculate joint velocities using numerical differentiation"""
        if len(self.angle_buffer) == 0:
            return np.zeros_like(angles)

        # Get previous angle and timestamp
        prev_angles = self.angle_buffer[-1]
        prev_timestamp = self.timestamp_buffer[-1]

        # Calculate time difference
        dt = timestamp - prev_timestamp
        if dt <= 0:
            return np.zeros_like(angles)

        # Calculate velocities
        velocities = (angles - prev_angles) / dt

        # Apply smoothing if enough data available
        if len(self.velocity_buffer) >= self.smoothing_window:
            # Simple moving average smoothing
            recent_velocities = list(self.velocity_buffer)[-self.smoothing_window:]
            recent_velocities.append(velocities)
            velocities = np.mean(recent_velocities, axis=0)

        return velocities

    def _calculate_accelerations(self, velocities: np.ndarray, timestamp: float) -> np.ndarray:
        """Calculate joint accelerations"""
        if len(self.velocity_buffer) == 0:
            return np.zeros_like(velocities)

        # Get previous velocity and timestamp
        prev_velocities = self.velocity_buffer[-1]
        prev_timestamp = self.timestamp_buffer[-1]

        # Calculate time difference
        dt = timestamp - prev_timestamp
        if dt <= 0:
            return np.zeros_like(velocities)

        # Calculate accelerations
        accelerations = (velocities - prev_velocities) / dt

        # Apply smoothing
        if len(self.acceleration_buffer) >= self.smoothing_window:
            recent_accelerations = list(self.acceleration_buffer)[-self.smoothing_window:]
            recent_accelerations.append(accelerations)
            accelerations = np.mean(recent_accelerations, axis=0)

        return accelerations

    def _update_buffers(self, angle_data: AngleData):
        """Update data buffers with new measurements"""
        self.angle_buffer.append(angle_data.joint_angles)
        self.velocity_buffer.append(angle_data.joint_velocities)
        self.timestamp_buffer.append(angle_data.timestamp)

        if angle_data.joint_accelerations is not None and self.acceleration_buffer is not None:
            self.acceleration_buffer.append(angle_data.joint_accelerations)

    def get_angle_history(self, window_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get angle history from buffer

        Args:
            window_size: Number of recent samples to return

        Returns:
            Tuple of (angles, timestamps) arrays
        """
        if window_size is None:
            window_size = min(len(self.angle_buffer), self.buffer_size)

        angles = np.array(list(self.angle_buffer)[-window_size:])
        timestamps = np.array(list(self.timestamp_buffer)[-window_size:])

        return angles, timestamps

    def get_velocity_history(self, window_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get velocity history from buffer"""
        if window_size is None:
            window_size = min(len(self.velocity_buffer), self.buffer_size)

        velocities = np.array(list(self.velocity_buffer)[-window_size:])
        timestamps = np.array(list(self.timestamp_buffer)[-window_size:])

        return velocities, timestamps

    def get_acceleration_history(self, window_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get acceleration history from buffer"""
        if self.acceleration_buffer is None:
            return np.array([]), np.array([])

        if window_size is None:
            window_size = min(len(self.acceleration_buffer), self.buffer_size)

        accelerations = np.array(list(self.acceleration_buffer)[-window_size:])
        timestamps = np.array(list(self.timestamp_buffer)[-window_size:])

        return accelerations, timestamps

    def get_angle_statistics(self, window_size: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Get statistical summary of recent angle measurements

        Args:
            window_size: Number of samples to analyze

        Returns:
            Dictionary with statistics for each joint
        """
        if len(self.angle_buffer) < 2:
            return {}

        angles, timestamps = self.get_angle_history(window_size)
        velocities, _ = self.get_velocity_history(window_size)

        stats = {}
        for i in range(angles.shape[1]):
            joint_angles = angles[:, i]
            joint_velocities = velocities[:, i] if len(velocities) > 0 else np.array([])

            stats[f'joint_{i+1}'] = {
                'mean': float(np.mean(joint_angles)),
                'std': float(np.std(joint_angles)),
                'min': float(np.min(joint_angles)),
                'max': float(np.max(joint_angles)),
                'range': float(np.ptp(joint_angles)),
                'velocity_mean': float(np.mean(joint_velocities)) if len(joint_velocities) > 0 else 0.0,
                'velocity_std': float(np.std(joint_velocities)) if len(joint_velocities) > 0 else 0.0,
            }

        return stats

    def detect_joint_limits(self, angles: np.ndarray, threshold: float = 0.1) -> List[bool]:
        """
        Detect if joints are near their limits

        Args:
            angles: Current joint angles
            threshold: Distance from limit to trigger detection (radians)

        Returns:
            List of booleans indicating if each joint is near its limit
        """
        near_limits = []
        for i, angle in enumerate(angles):
            if i < len(self.joint_limits):
                min_limit, max_limit = self.joint_limits[i]
                near_min = abs(angle - min_limit) < threshold
                near_max = abs(angle - max_limit) < threshold
                near_limits.append(near_min or near_max)
            else:
                near_limits.append(False)
        return near_limits

    def reset_buffers(self):
        """Reset all data buffers"""
        self.angle_buffer.clear()
        self.velocity_buffer.clear()
        self.timestamp_buffer.clear()
        if self.acceleration_buffer is not None:
            self.acceleration_buffer.clear()

        # Reset Kalman filters
        for kalman_filter in self.kalman_filters.values():
            kalman_filter.state = 0.0
            kalman_filter.covariance = 1.0

        self.logger.info("Angle reader buffers reset")

    def export_angle_data(self, filepath: str, format: str = 'csv'):
        """
        Export angle data to file

        Args:
            filepath: Output file path
            format: Export format ('csv', 'json', 'hdf5')
        """
        if len(self.angle_buffer) == 0:
            self.logger.warning("No angle data to export")
            return

        angles, timestamps = self.get_angle_history()
        velocities, _ = self.get_velocity_history()
        accelerations, _ = self.get_acceleration_history()

        if format == 'csv':
            import pandas as pd
            data = []
            for i in range(len(timestamps)):
                row = {'timestamp': timestamps[i]}
                for j in range(angles.shape[1]):
                    row[f'joint_{j+1}_angle'] = angles[i, j]
                    row[f'joint_{j+1}_velocity'] = velocities[i, j] if i < len(velocities) else 0.0
                    if accelerations.size > 0 and i < len(accelerations):
                        row[f'joint_{j+1}_acceleration'] = accelerations[i, j]
                data.append(row)

            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)

        elif format == 'json':
            data = {
                'timestamps': timestamps.tolist(),
                'angles': angles.tolist(),
                'velocities': velocities.tolist(),
                'accelerations': accelerations.tolist() if accelerations.size > 0 else []
            }
            import json
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

        self.logger.info(f"Angle data exported to {filepath}")