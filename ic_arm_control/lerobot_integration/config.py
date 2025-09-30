"""
LeRobot Integration Configuration
Configuration settings for dataset collection and angle reading
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json

@dataclass
class CollectionConfig:
    """Dataset collection configuration"""
    # Collection parameters
    sample_rate: float = 500.0  # Hz - 统一500Hz采样率
    max_episodes: Optional[int] = None
    episode_duration: float = 30.0  # seconds

    # Data format settings
    save_format: str = "hdf5"  # hdf5, npz, json
    compression: str = "gzip"

    # Data types to collect
    include_images: bool = False
    include_torque: bool = True
    include_velocity: bool = True
    include_acceleration: bool = True
    include_end_effector: bool = True

    # Filtering and processing
    enable_kalman_filter: bool = True
    smoothing_window: int = 5
    velocity_smoothing: bool = True
    acceleration_smoothing: bool = True

    # Safety parameters
    enable_safety_checks: bool = True
    joint_limit_threshold: float = 0.1  # radians
    max_velocity: float = 2.0  # rad/s
    max_acceleration: float = 5.0  # rad/s²

    # Storage settings
    dataset_path: str = "./datasets"
    auto_create_directories: bool = True
    backup_enabled: bool = False
    backup_interval: int = 10  # episodes

    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    environment_info: Dict[str, str] = field(default_factory=dict)

@dataclass
class AngleReaderConfig:
    """Angle reader specific configuration"""
    # Sampling parameters
    sample_rate: float = 500.0  # Hz - 500Hz控制频率
    buffer_size: int = 1000

    # Filtering
    enable_kalman_filter: bool = True
    kalman_process_variance: float = 1e-5
    kalman_measurement_variance: float = 1e-4
    smoothing_window: int = 5

    # Calculations
    enable_acceleration: bool = True
    velocity_calculation_method: str = "numerical"  # numerical, kalman
    acceleration_calculation_method: str = "numerical"

    # Joint settings
    joint_limits_enabled: bool = True
    custom_joint_limits: Optional[Dict[str, tuple]] = None

    # Export settings
    default_export_format: str = "csv"
    auto_export_on_stop: bool = False
    export_directory: str = "./angle_exports"

@dataclass
class LeRobotIntegrationConfig:
    """Main integration configuration"""
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    angle_reader: AngleReaderConfig = field(default_factory=AngleReaderConfig)

    # Integration settings
    enable_logging: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Performance settings
    max_memory_usage_mb: int = 1024
    enable_threading: bool = True
    thread_pool_size: int = 4

    # Control frequency settings
    control_frequency: float = 500.0  # Hz - 统一控制频率

    # Debug settings
    debug_mode: bool = False
    debug_save_raw_data: bool = False
    debug_print_interval: float = 1.0  # seconds

class ConfigManager:
    """Configuration file manager"""

    def __init__(self, config_file: str = "lerobot_config.json"):
        self.config_file = config_file
        self.config = LeRobotIntegrationConfig()

    def load_config(self) -> LeRobotIntegrationConfig:
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_dict = json.load(f)

                # Update configuration
                if 'collection' in config_dict:
                    self.config.collection = CollectionConfig(**config_dict['collection'])

                if 'angle_reader' in config_dict:
                    self.config.angle_reader = AngleReaderConfig(**config_dict['angle_reader'])

                # Update other settings
                for key, value in config_dict.items():
                    if key not in ['collection', 'angle_reader'] and hasattr(self.config, key):
                        setattr(self.config, key, value)

                print(f"Configuration loaded from {self.config_file}")

            except Exception as e:
                print(f"Error loading config: {e}")
                print("Using default configuration")

        return self.config

    def save_config(self, config: Optional[LeRobotIntegrationConfig] = None):
        """Save configuration to file"""
        if config is None:
            config = self.config

        try:
            config_dict = {
                'collection': config.collection.__dict__,
                'angle_reader': config.angle_reader.__dict__,
                'enable_logging': config.enable_logging,
                'log_level': config.log_level,
                'log_file': config.log_file,
                'max_memory_usage_mb': config.max_memory_usage_mb,
                'enable_threading': config.enable_threading,
                'thread_pool_size': config.thread_pool_size,
                'debug_mode': config.debug_mode,
                'debug_save_raw_data': config.debug_save_raw_data,
                'debug_print_interval': config.debug_print_interval
            }

            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)

            print(f"Configuration saved to {self.config_file}")

        except Exception as e:
            print(f"Error saving config: {e}")

# Default configurations for different use cases
DEFAULT_CONFIGS = {
    'basic': CollectionConfig(
        sample_rate=500.0,
        episode_duration=30.0,
        save_format="hdf5",
        include_images=False,
        include_torque=True,
        include_velocity=True,
        include_acceleration=False,
        enable_kalman_filter=True
    ),

    'high_frequency': CollectionConfig(
        sample_rate=500.0,
        episode_duration=60.0,
        save_format="hdf5",
        include_images=False,
        include_torque=True,
        include_velocity=True,
        include_acceleration=True,
        enable_kalman_filter=True,
        smoothing_window=3
    ),

    'control_500hz': CollectionConfig(
        sample_rate=500.0,
        episode_duration=60.0,
        save_format="hdf5",
        include_images=False,
        include_torque=True,
        include_velocity=True,
        include_acceleration=True,
        enable_kalman_filter=True,
        smoothing_window=2,
        max_velocity=3.0,  # 提高速度限制以适应高频控制
        max_acceleration=10.0  # 提高加速度限制
    ),

    'full_dataset': CollectionConfig(
        sample_rate=500.0,
        episode_duration=120.0,
        save_format="hdf5",
        include_images=True,
        include_torque=True,
        include_velocity=True,
        include_acceleration=True,
        enable_kalman_filter=True,
        compression="gzip"
    ),

    'debug': CollectionConfig(
        sample_rate=50.0,
        episode_duration=10.0,
        save_format="json",
        include_images=False,
        include_torque=True,
        include_velocity=True,
        include_acceleration=True,
        enable_kalman_filter=False
    )
}

DEFAULT_ANGLE_CONFIGS = {
    'basic': AngleReaderConfig(
        sample_rate=500.0,
        buffer_size=500,
        enable_kalman_filter=True,
        smoothing_window=3,
        enable_acceleration=False
    ),

    'high_precision': AngleReaderConfig(
        sample_rate=1000.0,
        buffer_size=1000,
        enable_kalman_filter=True,
        kalman_process_variance=1e-6,
        kalman_measurement_variance=1e-5,
        smoothing_window=3,
        enable_acceleration=True
    ),

    'real_time': AngleReaderConfig(
        sample_rate=500.0,
        buffer_size=100,
        enable_kalman_filter=True,
        smoothing_window=1,
        enable_acceleration=False,
        auto_export_on_stop=True
    ),

    'control_500hz': AngleReaderConfig(
        sample_rate=500.0,
        buffer_size=500,
        enable_kalman_filter=True,
        smoothing_window=2,
        enable_acceleration=True,
        kalman_process_variance=1e-5,
        kalman_measurement_variance=1e-4
    )
}

def get_default_config(config_name: str = 'basic') -> LeRobotIntegrationConfig:
    """Get a default configuration by name"""
    config = LeRobotIntegrationConfig()

    if config_name in DEFAULT_CONFIGS:
        config.collection = DEFAULT_CONFIGS[config_name]

    if config_name in DEFAULT_ANGLE_CONFIGS:
        config.angle_reader = DEFAULT_ANGLE_CONFIGS[config_name]

    return config

def create_config_template(config_file: str = "lerobot_config_template.json"):
    """Create a configuration template file"""
    config = get_default_config('basic')
    manager = ConfigManager(config_file)
    manager.save_config(config)
    print(f"Configuration template created at {config_file}")