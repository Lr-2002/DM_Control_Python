# LeRobot Integration for IC_ARM

A comprehensive dataset collection and angle reading system for IC_ARM, following the architecture of Hugging Face's LeRobot project.

## Features

### Dataset Collection
- **High-frequency data collection** at configurable rates (50-2000 Hz)
- **Multiple data formats**: HDF5, NPZ, JSON
- **Comprehensive data collection**: joint angles, velocities, accelerations, torques, end-effector positions
- **Episode management**: automatic episode creation, saving, and indexing
- **Data compression**: gzip compression for efficient storage

### Angle Reading
- **Kalman filtering**: noise reduction for precise angle measurements
- **High-frequency sampling**: up to 2000 Hz for real-time applications
- **Velocity and acceleration calculation**: numerical differentiation with smoothing
- **Joint limit detection**: safety monitoring with configurable thresholds
- **Statistical analysis**: real-time statistics and data export

### Configuration Management
- **Flexible configuration**: JSON-based configuration system
- **Predefined configurations**: basic, high-frequency, full-dataset, debug profiles
- **Runtime configuration**: dynamic configuration updates

## Installation

1. Install dependencies:
```bash
pip install numpy h5py pandas
```

2. Ensure IC_ARM system is properly initialized and accessible.

## Quick Start

### Basic Dataset Collection
```python
from ic_arm_control.control.IC_ARM import IC_ARM
from lerobot_integration import LeRobotDatasetCollector, DatasetConfig

# Initialize IC_ARM
ic_arm = IC_ARM()
ic_arm.initialize()

# Create dataset configuration
config = DatasetConfig(
    dataset_name="my_dataset",
    collection_rate=100.0,
    episode_duration=30.0,
    save_format="hdf5"
)

# Create collector and start collection
collector = LeRobotDatasetCollector(config, ic_arm)
episode_id = collector.start_episode()

# Collect data
for _ in range(1000):  # 1000 frames
    collector.collect_frame()
    time.sleep(0.01)  # Control collection rate

# Stop and save
episode = collector.stop_episode()
saved_file = collector.save_episode(episode)
```

### High-Frequency Angle Reading
```python
from ic_arm_control.control.IC_ARM import IC_ARM
from lerobot_integration import AngleReader

# Initialize IC_ARM
ic_arm = IC_ARM()
ic_arm.initialize()

# Create angle reader
angle_reader = AngleReader(
    sample_rate=1000.0,
    buffer_size=2000,
    enable_acceleration=True
)

# Read angles
angle_data = angle_reader.read_angles(ic_arm)
if angle_data:
    print(f"Joint angles: {angle_data.joint_angles}")
    print(f"Joint velocities: {angle_data.joint_velocities}")
    print(f"Joint accelerations: {angle_data.joint_accelerations}")
```

### Configuration Management
```python
from lerobot_integration import ConfigManager, get_default_config

# Use predefined configuration
config = get_default_config('high_frequency')

# Or load from file
config_manager = ConfigManager("my_config.json")
config = config_manager.load_config()
```

## Configuration Options

### Dataset Collection Config
- `sample_rate`: Data collection frequency (Hz)
- `episode_duration`: Duration of each episode (seconds)
- `save_format`: Output format (hdf5, npz, json)
- `compression`: Compression algorithm (gzip, lzf, none)
- `include_images`: Include camera images in dataset
- `include_torque`: Include joint torque data
- `include_velocity`: Include joint velocity data
- `include_acceleration`: Include joint acceleration data

### Angle Reader Config
- `sample_rate`: Angle sampling frequency (Hz)
- `buffer_size`: Circular buffer size
- `enable_kalman_filter`: Enable Kalman filtering
- `smoothing_window`: Window size for smoothing
- `enable_acceleration`: Calculate joint accelerations

## Directory Structure

```
lerobot_integration/
├── __init__.py              # Package initialization
├── dataset_collector.py      # Main dataset collection system
├── angle_reader.py          # High-frequency angle reading
├── config.py                # Configuration management
├── example_usage.py         # Usage examples
└── README.md               # This file
```

## Data Storage Format

### HDF5 Format
```python
# Episode data structure
episode_data.hdf5
├── metadata/               # Episode metadata
│   ├── episode_id
│   ├── start_time
│   ├── end_time
│   └── collection_rate
├── actions                 # Motor commands
├── observations/           # Sensor data
│   ├── joint_angles
│   ├── joint_velocities
│   ├── joint_torques
│   ├── end_effector_pos
│   └── end_effector_quat
└── camera_images (optional)  # Camera data
```

### JSON Format
```json
{
  "episode_id": "episode_000001",
  "timestamp": 1234567890.0,
  "actions": [[0.1, 0.2, ...], [0.15, 0.25, ...]],
  "observations": {
    "joint_angles": [[0.0, 0.1, ...], [0.05, 0.15, ...]],
    "joint_velocities": [[0.0, 0.0, ...], [0.01, 0.02, ...]]
  },
  "metadata": {
    "start_time": "2024-01-01T10:00:00",
    "end_time": "2024-01-01T10:00:30",
    "duration": 30.0,
    "num_frames": 1000
  }
}
```

## Safety Features

### Joint Limit Detection
```python
# Check if joints are near limits
near_limits = angle_reader.detect_joint_limits(angles)
for i, near_limit in enumerate(near_limits):
    if near_limit:
        print(f"Warning: Joint {i+1} near limit!")
```

### Velocity Monitoring
```python
# Monitor joint velocities
max_velocities = np.abs(angle_data.joint_velocities)
max_velocity_limit = 2.0  # rad/s
if np.any(max_velocities > max_velocity_limit):
    print("Warning: High joint velocity detected!")
```

## Examples

Run the included examples:
```bash
python lerobot_integration/example_usage.py
```

Examples included:
- Basic dataset collection
- High-frequency angle reading
- Continuous collection integration
- Configuration management
- Safety monitoring

## Performance Optimization

### Memory Management
- Configurable buffer sizes
- Circular buffers for efficient memory usage
- Optional data compression

### Threading
- Multi-threaded data collection support
- Thread pool for parallel processing
- Asynchronous data saving

## Troubleshooting

### Common Issues

1. **IC_ARM initialization fails**
   - Ensure IC_ARM hardware is connected
   - Check motor configurations
   - Verify USB connections

2. **Data collection is slow**
   - Reduce sample rate
   - Disable unnecessary data types
   - Use threading for parallel processing

3. **Memory usage is high**
   - Reduce buffer sizes
   - Enable compression
   - Use streaming mode

### Debug Mode
Enable debug mode for detailed logging:
```python
config.debug_mode = True
config.debug_save_raw_data = True
```

## License

This project is part of the IC_ARM control system and follows the same license terms.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review the example usage
- Contact the development team