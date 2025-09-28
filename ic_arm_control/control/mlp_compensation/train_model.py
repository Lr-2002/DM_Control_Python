#!/usr/bin/env python3
"""
Training script for MLP gravity compensation model
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from mlp_gravity_compensation import LightweightMLPGravityCompensation


def load_training_data_from_datasets(dataset_dir):
    """
    Load and merge training data from multiple dataset directories

    Args:
        dataset_dir: Path to datasets directory containing multiple timestamped folders

    Returns:
        positions: Joint positions (N, 6)
        velocities: Joint velocities (N, 6)
        torques: Target torques (N, 6)
    """
    dataset_path = Path(dataset_dir)
    all_positions = []
    all_velocities = []
    all_torques = []

    # Find all timestamped directories
    timestamp_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    print(f"Found {len(timestamp_dirs)} dataset directories")

    for timestamp_dir in timestamp_dirs:
        print(f"Loading data from {timestamp_dir.name}")

        motor_states_file = timestamp_dir / 'motor_states.csv'
        if not motor_states_file.exists():
            print(f"⚠️ No motor_states.csv in {timestamp_dir.name}")
            continue

        try:
            # Read motor states
            motor_states = pd.read_csv(motor_states_file)

            # Extract data for first 6 joints
            n_joints = 6
            position_cols = [f'position_motor_{i+1}' for i in range(n_joints)]
            velocity_cols = [f'velocity_motor_{i+1}' for i in range(n_joints)]
            torque_cols = [f'torque_motor_{i+1}' for i in range(n_joints)]

            positions = motor_states[position_cols].values
            velocities = motor_states[velocity_cols].values
            torques = motor_states[torque_cols].values

            # Remove invalid data
            valid_mask = ~(np.isnan(positions).any(axis=1) |
                          np.isnan(velocities).any(axis=1) |
                          np.isnan(torques).any(axis=1))

            positions = positions[valid_mask]
            velocities = velocities[valid_mask]
            torques = torques[valid_mask]

            # Remove torque outliers (more conservative for gravity compensation)
            torque_mask = np.abs(torques) < 10.0  # Reduced from 20.0 to 10.0
            positions = positions[torque_mask.all(axis=1)]
            velocities = velocities[torque_mask.all(axis=1)]
            torques = torques[torque_mask.all(axis=1)]

            # Additional filtering: remove near-zero torque data (not useful for gravity compensation)
            torque_magnitude = np.linalg.norm(torques, axis=1)
            significant_mask = torque_magnitude > 0.1  # Only data with meaningful torque
            positions = positions[significant_mask]
            velocities = velocities[significant_mask]
            torques = torques[significant_mask]

            all_positions.append(positions)
            all_velocities.append(velocities)
            all_torques.append(torques)

            print(f"  Loaded {len(positions)} valid samples from {timestamp_dir.name}")

        except Exception as e:
            print(f"❌ Error loading {timestamp_dir.name}: {e}")
            continue

    # Concatenate all data
    if all_positions:
        positions = np.vstack(all_positions)
        velocities = np.vstack(all_velocities)
        torques = np.vstack(all_torques)
        print(f"Total merged dataset: {len(positions)} samples")
        return positions, velocities, torques
    else:
        raise ValueError("No valid data found in any dataset directory")


def load_training_data(log_dir):
    """
    Legacy function for backward compatibility
    """
    return load_training_data_from_datasets(log_dir)


def main():
    """Main training function"""
    print("=== Training MLP Gravity Compensation Model ===")

    # Configuration
    dataset_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control/datasets"
    model_path = "mlp_gravity_model_new.pkl"

    try:
        # Load training data from multiple datasets
        print(f"Loading data from {dataset_dir}")
        positions, velocities, torques = load_training_data_from_datasets(dataset_dir)
        print(f"Loaded {len(positions)} valid data points")

        # Data statistics
        print("\n=== Data Statistics ===")
        print(f"Position range: [{np.min(positions):.3f}, {np.max(positions):.3f}] rad")
        print(f"Velocity range: [{np.min(velocities):.3f}, {np.max(velocities):.3f}] rad/s")
        print(f"Torque range: [{np.min(torques):.3f}, {np.max(torques):.3f}] Nm")

        # Initialize and train MLP model
        mlp_system = LightweightMLPGravityCompensation(
            hidden_layer_sizes=(150, 75, 30),  # Deeper network for better performance
            max_iter=1000,  # More iterations
            random_state=42
        )

        # Train model
        print("\n=== Training MLP Model ===")
        mlp_system.train(positions, torques, velocities)

        # Print performance summary
        print("\n" + mlp_system.get_performance_summary())

        # Save model
        mlp_system.save_model(model_path)

        # Performance test
        print("\n=== Performance Test ===")
        n_tests = 1000
        test_positions = positions[:n_tests]
        test_velocities = velocities[:n_tests]

        import time
        start_time = time.time()
        predictions = mlp_system.predict(test_positions, test_velocities)
        avg_time = (time.time() - start_time) / n_tests * 1000

        print(f"Average computation time: {avg_time:.3f} ms")
        print(f"Frequency: {1000/avg_time:.1f} Hz")

        if 1000/avg_time > 300:
            print("✅ Suitable for 300Hz operation")
        else:
            print("⚠️ May not achieve 300Hz")

        # Additional validation
        print("\n=== Additional Validation ===")
        # Test on a few random positions
        test_indices = np.random.choice(len(positions), min(5, len(positions)), replace=False)
        for idx in test_indices:
            pos = positions[idx:idx+1]
            vel = velocities[idx:idx+1]
            true_torque = torques[idx]
            pred_torque = mlp_system.predict(pos, vel)[0]

            error = np.linalg.norm(true_torque - pred_torque)
            print(f"Sample {idx}: True=[{', '.join(f'{t:.2f}' for t in true_torque)}], "
                  f"Pred=[{', '.join(f'{p:.2f}' for p in pred_torque)}], "
                  f"Error={error:.3f} Nm")

        print("\n✅ MLP gravity compensation model training completed!")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()