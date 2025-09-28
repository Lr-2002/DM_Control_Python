#!/usr/bin/env python3
"""
Training script for MLP gravity compensation model
"""

import numpy as np
import pandas as pd
import os
from mlp_gravity_compensation import LightweightMLPGravityCompensation


def load_training_data(log_dir):
    """
    Load training data from log directory

    Args:
        log_dir: Path to log directory containing motor_states.csv

    Returns:
        positions: Joint positions (N, 6)
        velocities: Joint velocities (N, 6)
        torques: Target torques (N, 6)
    """
    # Read motor states
    motor_states_file = os.path.join(log_dir, 'motor_states.csv')
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

    # Remove torque outliers
    torque_mask = np.abs(torques) < 20.0
    positions = positions[torque_mask.all(axis=1)]
    velocities = velocities[torque_mask.all(axis=1)]
    torques = torques[torque_mask.all(axis=1)]

    return positions, velocities, torques


def main():
    """Main training function"""
    print("=== Training MLP Gravity Compensation Model ===")

    # Configuration
    log_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control/logs/20250926_175734"
    model_path = "mlp_gravity_model.pkl"

    # Load training data
    print(f"Loading data from {log_dir}")
    positions, velocities, torques = load_training_data(log_dir)
    print(f"Loaded {len(positions)} valid data points")

    # Initialize and train MLP model
    mlp_system = LightweightMLPGravityCompensation(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=42
    )

    # Train model
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

    print("\n✅ MLP gravity compensation model training completed!")


if __name__ == "__main__":
    main()