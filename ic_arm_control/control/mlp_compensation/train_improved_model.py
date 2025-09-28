#!/usr/bin/env python3
"""
Improved MLP training with enhanced preprocessing for Joint 1 performance issues
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from mlp_gravity_compensation import LightweightMLPGravityCompensation
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pickle


def load_training_data_from_datasets(dataset_dir):
    """
    Load and merge training data from multiple dataset directories
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

            # Enhanced filtering for Joint 1
            # Remove torque outliers (more aggressive for Joint 1)
            joint1_torques = torques[:, 0]
            joint1_mask = np.abs(joint1_torques) < 8.0  # Reduced from 10.0 to 8.0

            # Apply mask to all joints
            positions = positions[joint1_mask]
            velocities = velocities[joint1_mask]
            torques = torques[joint1_mask]

            # Additional filtering: remove extreme torque data for all joints
            torque_mask = np.abs(torques) < 8.0
            positions = positions[torque_mask.all(axis=1)]
            velocities = velocities[torque_mask.all(axis=1)]
            torques = torques[torque_mask.all(axis=1)]

            # Remove near-zero torque data (not useful for gravity compensation)
            torque_magnitude = np.linalg.norm(torques, axis=1)
            significant_mask = torque_magnitude > 0.05  # Reduced threshold

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


def enhance_features(positions, velocities):
    """
    Enhance input features with additional engineering for better Joint 1 modeling
    """
    enhanced_features = []

    # Original features
    enhanced_features.append(positions)
    enhanced_features.append(velocities)

    # Joint 1 specific features
    joint1_pos = positions[:, 0:1]
    joint1_vel = velocities[:, 0:1]

    # Non-linear features for Joint 1
    enhanced_features.append(joint1_pos ** 2)  # Position squared
    enhanced_features.append(joint1_vel ** 2)  # Velocity squared
    enhanced_features.append(joint1_pos * joint1_vel)  # Cross term

    # Sin/cos features for Joint 1 (helpful for rotational joints)
    enhanced_features.append(np.sin(joint1_pos))
    enhanced_features.append(np.cos(joint1_pos))

    # Velocity direction for Joint 1
    enhanced_features.append(np.sign(joint1_vel))

    # Concatenate all features
    return np.concatenate(enhanced_features, axis=1)


def train_improved_model():
    """Train improved MLP model with enhanced preprocessing"""
    print("=== Training Improved MLP Gravity Compensation Model ===")

    # Configuration
    dataset_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control/datasets"
    model_path = "mlp_gravity_model_improved.pkl"

    try:
        # Load training data
        print(f"Loading data from {dataset_dir}")
        positions, velocities, torques = load_training_data_from_datasets(dataset_dir)
        print(f"Loaded {len(positions)} valid data points")

        # Data statistics
        print("\n=== Enhanced Data Statistics ===")
        print(f"Position range: [{np.min(positions):.3f}, {np.max(positions):.3f}] rad")
        print(f"Velocity range: [{np.min(velocities):.3f}, {np.max(velocities):.3f}] rad/s")
        print(f"Torque range: [{np.min(torques):.3f}, {np.max(torques):.3f}] Nm")

        # Enhanced feature engineering
        print("\n=== Feature Enhancement ===")
        enhanced_features = enhance_features(positions, velocities)
        print(f"Enhanced features shape: {enhanced_features.shape}")
        print(f"Original features: 12, Enhanced features: {enhanced_features.shape[1]}")

        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(
            enhanced_features, torques,
            test_size=0.2,
            random_state=42
        )

        # Create enhanced MLP system
        print("\n=== Initializing Enhanced MLP Model ===")
        mlp_system = LightweightMLPGravityCompensation(
            hidden_layer_sizes=(200, 100, 50, 25),  # Deeper network
            max_iter=2000,  # More iterations
            random_state=42
        )

        # Enhanced training with custom preprocessing
        print("\n=== Training Enhanced MLP Model ===")

        # Custom training with enhanced features
        mlp_system.train_enhanced = True
        mlp_system.enhanced_feature_dim = enhanced_features.shape[1]

        # Train with enhanced features
        mlp_system.mlps = []
        mlp_system.input_scaler = RobustScaler()  # More robust to outliers
        mlp_system.output_scaler = RobustScaler()

        # Scale enhanced features
        X_train_scaled = mlp_system.input_scaler.fit_transform(X_train)
        y_train_scaled = mlp_system.output_scaler.fit_transform(y_train)

        # Train separate MLP for each joint
        joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']
        train_scores = []
        val_scores = []

        for i in range(6):
            print(f"\nTraining {joint_names[i]} MLP...")

            # Special handling for Joint 1
            if i == 0:
                # Use deeper network for Joint 1
                from sklearn.neural_network import MLPRegressor
                mlp = MLPRegressor(
                    hidden_layer_sizes=(300, 150, 75, 30),
                    max_iter=3000,
                    random_state=42,
                    alpha=0.0005,
                    learning_rate_init=0.0005,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=50,
                    verbose=True
                )
            else:
                # Standard network for other joints
                from sklearn.neural_network import MLPRegressor
                mlp = MLPRegressor(
                    hidden_layer_sizes=(200, 100, 50),
                    max_iter=2000,
                    random_state=42,
                    alpha=0.001,
                    learning_rate_init=0.001,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=30,
                    verbose=False
                )

            # Train the MLP
            mlp.fit(X_train_scaled, y_train_scaled[:, i])

            # Store the trained MLP
            mlp_system.mlps.append(mlp)

            # Calculate scores
            train_score = mlp.score(X_train_scaled, y_train_scaled[:, i])
            train_scores.append(train_score)

            print(f"{joint_names[i]} Training R²: {train_score:.4f}")

        mlp_system.train_scores = train_scores
        mlp_system.val_scores = val_scores
        mlp_system.is_trained = True

        # Print performance summary
        print("\n=== Enhanced Model Performance ===")
        for i, joint_name in enumerate(joint_names):
            print(f"{joint_name}: R² = {train_scores[i]:.4f}")

        # Test performance
        print("\n=== Test Performance ===")
        X_test_scaled = mlp_system.input_scaler.transform(X_test)
        y_pred_scaled = np.column_stack([mlp.predict(X_test_scaled) for mlp in mlp_system.mlps])
        y_pred = mlp_system.output_scaler.inverse_transform(y_pred_scaled)

        from sklearn.metrics import r2_score, mean_squared_error
        for i, joint_name in enumerate(joint_names):
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
            print(f"{joint_name}: Test R² = {r2:.4f}, RMSE = {rmse:.4f} Nm")

        # Save enhanced model
        print(f"\n=== Saving Enhanced Model ===")
        model_data = {
            'mlps': mlp_system.mlps,
            'input_scaler': mlp_system.input_scaler,
            'output_scaler': mlp_system.output_scaler,
            'train_scores': mlp_system.train_scores,
            'val_scores': mlp_system.val_scores,
            'is_trained': mlp_system.is_trained,
            'enhanced_training': True,
            'enhanced_feature_dim': enhanced_features.shape[1],
            'hidden_layer_sizes': (200, 100, 50, 25),
            'max_iter': 2000,
            'random_state': 42
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ Enhanced model saved to {model_path}")

        # Performance test
        print("\n=== Performance Test ===")
        n_tests = 1000
        test_enhanced_features = enhance_features(positions[:n_tests], velocities[:n_tests])

        import time
        start_time = time.time()
        predictions = mlp_system.predict_enhanced(test_enhanced_features)
        avg_time = (time.time() - start_time) / n_tests * 1000

        print(f"Average computation time: {avg_time:.3f} ms")
        print(f"Frequency: {1000/avg_time:.1f} Hz")

        if 1000/avg_time > 300:
            print("✅ Suitable for 300Hz operation")
        else:
            print("⚠️ May not achieve 300Hz")

        print("\n✅ Enhanced MLP gravity compensation model training completed!")
        return True

    except Exception as e:
        print(f"❌ Enhanced training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    train_improved_model()