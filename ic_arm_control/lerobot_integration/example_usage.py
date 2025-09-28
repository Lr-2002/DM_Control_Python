"""
LeRobot Integration Example Usage
Demonstrates how to use the dataset collection and angle reading system
"""

import time
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ic_arm_control.control.IC_ARM import IC_ARM
from lerobot_integration.dataset_collector import LeRobotDatasetCollector, DatasetConfig, EpisodeData
from lerobot_integration.angle_reader import AngleReader, AngleData
from lerobot_integration.config import ConfigManager, get_default_config

def example_basic_dataset_collection():
    """Example: Basic dataset collection"""
    print("=== Basic Dataset Collection Example ===")

    # Initialize IC_ARM
    ic_arm = IC_ARM()
    ic_arm.initialize()

    # Create dataset configuration
    config = DatasetConfig(
        dataset_name="basic_arm_data",
        collection_rate=100.0,
        episode_duration=10.0,
        save_format="hdf5",
        include_torque=True,
        include_velocity=True
    )

    # Create dataset collector
    collector = LeRobotDatasetCollector(config, ic_arm)

    try:
        # Start an episode
        episode_id = collector.start_episode()
        print(f"Started episode: {episode_id}")

        # Collect data for 10 seconds
        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < 10.0:
            success = collector.collect_frame()
            if success:
                frame_count += 1

            # Small delay to control collection rate
            time.sleep(0.01)

        print(f"Collected {frame_count} frames")

        # Stop and save episode
        episode = collector.stop_episode()
        if episode:
            saved_file = collector.save_episode(episode)
            print(f"Episode saved to: {saved_file}")

    finally:
        # Clean up
        ic_arm.cleanup()

def example_high_frequency_angle_reading():
    """Example: High-frequency angle reading"""
    print("\n=== High-Frequency Angle Reading Example ===")

    # Initialize IC_ARM
    ic_arm = IC_ARM()
    ic_arm.initialize()

    # Create angle reader with high frequency
    angle_reader = AngleReader(
        sample_rate=1000.0,
        buffer_size=2000,
        smoothing_window=3,
        enable_acceleration=True
    )

    try:
        print("Starting angle reading for 5 seconds...")
        start_time = time.time()
        reading_count = 0

        while time.time() - start_time < 5.0:
            # Read angles
            angle_data = angle_reader.read_angles(ic_arm)

            if angle_data:
                reading_count += 1

                # Print statistics every second
                if reading_count % 1000 == 0:
                    stats = angle_reader.get_angle_statistics(100)
                    print(f"Joint 1 angle: {angle_data.joint_angles[0]:.3f} rad, "
                          f"velocity: {angle_data.joint_velocities[0]:.3f} rad/s")

            # Control sample rate
            time.sleep(0.001)

        print(f"Completed {reading_count} angle readings")

        # Get final statistics
        final_stats = angle_reader.get_angle_statistics(500)
        print("\nFinal statistics:")
        for joint_name, stats in final_stats.items():
            print(f"{joint_name}: Mean={stats['mean']:.3f}, "
                  f"Std={stats['std']:.3f}, "
                  f"Range={stats['range']:.3f}")

        # Export angle data
        angle_reader.export_angle_data("angle_data_export.csv", format="csv")
        print("Angle data exported to angle_data_export.csv")

    finally:
        # Clean up
        ic_arm.cleanup()

def example_continuous_collection_with_angle_reader():
    """Example: Continuous dataset collection with integrated angle reading"""
    print("\n=== Continuous Collection with Angle Reader Integration ===")

    # Initialize IC_ARM
    ic_arm = IC_ARM()
    ic_arm.initialize()

    # Create configuration with angle reader integration
    config = get_default_config('high_frequency')
    config.collection.dataset_name = "integrated_collection"
    config.collection.episode_duration = 30.0

    # Create components
    collector = LeRobotDatasetCollector(config.collection, ic_arm)
    angle_reader = AngleReader(
        sample_rate=config.angle_reader.sample_rate,
        buffer_size=config.angle_reader.buffer_size,
        enable_acceleration=config.angle_reader.enable_acceleration
    )

    try:
        print("Starting integrated collection for 30 seconds...")
        start_time = time.time()
        episode_id = collector.start_episode()

        angle_data_buffer = []

        while time.time() - start_time < 30.0:
            # Read angles at high frequency
            angle_data = angle_reader.read_angles(ic_arm)

            if angle_data:
                angle_data_buffer.append(angle_data)

                # Collect dataset frame at lower frequency
                if len(angle_data_buffer) % 10 == 0:  # Collect at 100Hz if reading at 1000Hz
                    collector.collect_frame()

                # Print progress every 5 seconds
                if len(angle_data_buffer) % 5000 == 0:
                    elapsed = time.time() - start_time
                    print(f"Collected {len(angle_data_buffer)} angle readings, "
                          f"{len(collector.current_episode.actions)} dataset frames, "
                          f"elapsed: {elapsed:.1f}s")

        # Stop collection
        episode = collector.stop_episode()
        if episode:
            saved_file = collector.save_episode(episode)
            print(f"Dataset episode saved to: {saved_file}")

        # Export angle data
        angle_reader.export_angle_data("integrated_angle_data.csv", format="csv")
        print("Angle data exported to integrated_angle_data.csv")

        # Create dataset index
        index_file = collector.create_dataset_index()
        print(f"Dataset index created: {index_file}")

    finally:
        # Clean up
        ic_arm.cleanup()

def example_config_management():
    """Example: Configuration management"""
    print("\n=== Configuration Management Example ===")

    # Create config manager
    config_manager = ConfigManager("example_config.json")

    # Load configuration
    config = config_manager.load_config()
    print(f"Loaded configuration with sample rate: {config.collection.sample_rate}Hz")

    # Modify configuration
    config.collection.dataset_name = "example_dataset"
    config.collection.sample_rate = 200.0
    config.angle_reader.sample_rate = 1000.0

    # Save configuration
    config_manager.save_config(config)
    print("Modified configuration saved")

    # Use predefined configurations
    high_freq_config = get_default_config('high_frequency')
    print(f"High frequency config sample rate: {high_freq_config.collection.sample_rate}Hz")

def example_safety_monitoring():
    """Example: Safety monitoring with angle reading"""
    print("\n=== Safety Monitoring Example ===")

    # Initialize IC_ARM
    ic_arm = IC_ARM()
    ic_arm.initialize()

    # Create angle reader with safety settings
    angle_reader = AngleReader(
        sample_rate=500.0,
        buffer_size=1000,
        enable_acceleration=True
    )

    try:
        print("Monitoring joint limits and velocities...")
        start_time = time.time()

        while time.time() - start_time < 15.0:
            angle_data = angle_reader.read_angles(ic_arm)

            if angle_data:
                # Check joint limits
                near_limits = angle_reader.detect_joint_limits(angle_data.joint_angles)

                # Check velocities
                max_velocities = np.abs(angle_data.joint_velocities)
                max_velocity_limit = 2.0  # rad/s

                # Safety warnings
                for i, (near_limit, velocity) in enumerate(zip(near_limits, max_velocities)):
                    joint_name = f"Joint {i+1}"
                    if near_limit:
                        print(f"WARNING: {joint_name} near limit! "
                              f"Angle: {angle_data.joint_angles[i]:.3f} rad")

                    if velocity > max_velocity_limit:
                        print(f"WARNING: {joint_name} high velocity! "
                              f"Velocity: {velocity:.3f} rad/s")

                # Check accelerations
                if angle_data.joint_accelerations is not None:
                    max_accelerations = np.abs(angle_data.joint_accelerations)
                    max_accel_limit = 5.0  # rad/s²

                    for i, acceleration in enumerate(max_accelerations):
                        if acceleration > max_accel_limit:
                            print(f"WARNING: Joint {i+1} high acceleration! "
                                  f"Acceleration: {acceleration:.3f} rad/s²")

            time.sleep(0.002)

    finally:
        ic_arm.cleanup()

def main():
    """Run all examples"""
    print("LeRobot Integration Examples")
    print("=" * 50)

    examples = [
        ("Basic Dataset Collection", example_basic_dataset_collection),
        ("High-Frequency Angle Reading", example_high_frequency_angle_reading),
        ("Continuous Collection Integration", example_continuous_collection_with_angle_reader),
        ("Configuration Management", example_config_management),
        ("Safety Monitoring", example_safety_monitoring),
    ]

    for example_name, example_func in examples:
        try:
            example_func()
            print(f"\n✓ {example_name} completed successfully\n")
        except Exception as e:
            print(f"\n✗ {example_name} failed: {e}\n")

if __name__ == "__main__":
    main()