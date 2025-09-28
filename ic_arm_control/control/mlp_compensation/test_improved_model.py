#!/usr/bin/env python3
"""
Direct test of the improved model with the same dataset used for training
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import sys
sys.path.append('.')
from mlp_gravity_compensation import LightweightMLPGravityCompensation


def load_enhanced_training_data(dataset_dir):
    """Load data with the same filtering as the improved model training"""
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

            # Enhanced filtering for Joint 1 (same as training)
            joint1_torques = torques[:, 0]
            joint1_mask = np.abs(joint1_torques) < 8.0

            positions = positions[joint1_mask]
            velocities = velocities[joint1_mask]
            torques = torques[joint1_mask]

            # Additional filtering for all joints
            torque_mask = np.abs(torques) < 8.0
            positions = positions[torque_mask.all(axis=1)]
            velocities = velocities[torque_mask.all(axis=1)]
            torques = torques[torque_mask.all(axis=1)]

            # Remove near-zero torque data
            torque_magnitude = np.linalg.norm(torques, axis=1)
            significant_mask = torque_magnitude > 0.05

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
        raise ValueError("No valid data found")


def main():
    """Test the improved model with filtered data"""
    print("=== Testing Improved Model with Filtered Data ===")

    # Load filtered data
    dataset_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control/datasets"
    positions, velocities, torques = load_enhanced_training_data(dataset_dir)

    # Load improved model
    with open("mlp_gravity_model_improved.pkl", 'rb') as f:
        model_data = pickle.load(f)

    # Create model instance
    mlp_system = LightweightMLPGravityCompensation(
        hidden_layer_sizes=model_data['hidden_layer_sizes'],
        max_iter=model_data['max_iter'],
        random_state=model_data['random_state']
    )

    mlp_system.mlps = model_data['mlps']
    mlp_system.input_scaler = model_data['input_scaler']
    mlp_system.output_scaler = model_data['output_scaler']
    mlp_system.train_scores = model_data['train_scores']
    mlp_system.val_scores = model_data.get('val_scores', [])
    mlp_system.is_trained = model_data['is_trained']
    mlp_system.train_enhanced = model_data['enhanced_training']
    mlp_system.enhanced_feature_dim = model_data['enhanced_feature_dim']

    print(f"Model trained with enhanced features: {mlp_system.train_enhanced}")
    print(f"Enhanced feature dimension: {mlp_system.enhanced_feature_dim}")

    # Test with enhanced features
    enhanced_features = mlp_system.enhance_features(positions, velocities)
    predictions = mlp_system.predict_enhanced(enhanced_features)

    # Calculate metrics
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']

    print(f"\n=== Improved Model Performance on Filtered Data ===")
    print(f"Test samples: {len(positions)}")

    for i, joint_name in enumerate(joint_names):
        r2 = r2_score(torques[:, i], predictions[:, i])
        rmse = np.sqrt(mean_squared_error(torques[:, i], predictions[:, i]))
        mae = mean_absolute_error(torques[:, i], predictions[:, i])

        print(f"{joint_name}: R² = {r2:.4f}, RMSE = {rmse:.4f} Nm, MAE = {mae:.4f} Nm")

    # Performance test
    import time
    n_tests = 1000
    test_enhanced = enhanced_features[:n_tests]

    start_time = time.time()
    test_predictions = mlp_system.predict_enhanced(test_enhanced)
    avg_time = (time.time() - start_time) / n_tests * 1000

    print(f"\n=== Performance ===")
    print(f"Average prediction time: {avg_time:.3f} ms")
    print(f"Prediction frequency: {1000/avg_time:.1f} Hz")


if __name__ == "__main__":
    main()