"""
LeRobot Integration Package
Dataset collection and angle reading system for IC_ARM
"""

from .dataset_collector import (
    LeRobotDatasetCollector,
    DatasetConfig,
    EpisodeData
)

from .angle_reader import (
    AngleReader,
    AngleData
)

from .config import (
    ConfigManager,
    LeRobotIntegrationConfig,
    CollectionConfig,
    AngleReaderConfig,
    get_default_config,
    DEFAULT_CONFIGS,
    DEFAULT_ANGLE_CONFIGS
)

__version__ = "1.0.0"
__author__ = "LeRobot Integration Team"

__all__ = [
    # Dataset collection
    "LeRobotDatasetCollector",
    "DatasetConfig",
    "EpisodeData",

    # Angle reading
    "AngleReader",
    "AngleData",

    # Configuration
    "ConfigManager",
    "LeRobotIntegrationConfig",
    "CollectionConfig",
    "AngleReaderConfig",
    "get_default_config",
    "DEFAULT_CONFIGS",
    "DEFAULT_ANGLE_CONFIGS",
]