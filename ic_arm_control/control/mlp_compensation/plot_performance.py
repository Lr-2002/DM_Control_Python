#!/usr/bin/env python3
"""
Visualization script for MLP gravity compensation model performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from mlp_gravity_compensation import LightweightMLPGravityCompensation


def load_training_data(log_dir):
    """Load training data from log directory"""
    motor_states_file = os.path.join(log_dir, 'motor_states.csv')
    motor_states = pd.read_csv(motor_states_file)

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


def plot_training_performance(mlp_system):
    """Plot training performance metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # R² scores per joint
    joints = range(1, 7)
    train_scores = mlp_system.train_scores
    val_scores = mlp_system.val_scores

    x = np.arange(len(joints))
    width = 0.35

    ax1.bar(x - width/2, train_scores, width, label='Train R²', alpha=0.8)
    ax1.bar(x + width/2, val_scores, width, label='Validation R²', alpha=0.8)
    ax1.set_xlabel('Joint')
    ax1.set_ylabel('R² Score')
    ax1.set_title('R² Scores per Joint')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'J{i}' for i in joints])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.2, 1.0)

    # Average scores
    avg_train = np.mean(train_scores)
    avg_val = np.mean(val_scores)
    ax2.bar(['Average Train', 'Average Val'], [avg_train, avg_val], color=['skyblue', 'lightcoral'])
    ax2.set_ylabel('R² Score')
    ax2.set_title('Average Performance')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)

    # Performance distribution
    all_scores = train_scores + val_scores
    labels = [f'J{i}_train' for i in joints] + [f'J{i}_val' for i in joints]
    colors = ['skyblue'] * 6 + ['lightcoral'] * 6

    bars = ax3.bar(labels, all_scores, color=colors)
    ax3.set_ylabel('R² Score')
    ax3.set_title('All Joint Performance')
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=45)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.2, 1.0)

    # Performance summary
    performance_text = f"Model Performance Summary\n\n"
    performance_text += f"Hidden Layers: {mlp_system.hidden_layer_sizes}\n"
    performance_text += f"Max Iterations: {mlp_system.max_iter}\n\n"
    performance_text += f"Best Joint: Joint 1 (R² = {max(val_scores):.4f})\n"
    performance_text += f"Worst Joint: Joint 6 (R² = {min(val_scores):.4f})\n\n"
    performance_text += f"Average Train R²: {avg_train:.4f}\n"
    performance_text += f"Average Val R²: {avg_val:.4f}\n\n"

    # Performance rating
    if avg_val > 0.8:
        rating = "Excellent"
    elif avg_val > 0.6:
        rating = "Good"
    elif avg_val > 0.4:
        rating = "Fair"
    else:
        rating = "Poor"

    performance_text += f"Overall Rating: {rating}"

    ax4.text(0.1, 0.9, performance_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Performance Summary')

    plt.tight_layout()
    plt.savefig('mlp_training_performance.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_prediction_accuracy(mlp_system, positions, torques, velocities, n_samples=1000):
    """Plot prediction accuracy with actual vs predicted"""
    # Sample data for plotting
    if len(positions) > n_samples:
        indices = np.random.choice(len(positions), n_samples, replace=False)
        positions_sample = positions[indices]
        torques_sample = torques[indices]
        velocities_sample = velocities[indices]
    else:
        positions_sample = positions
        torques_sample = torques
        velocities_sample = velocities

    # Get predictions
    predictions = mlp_system.predict(positions_sample, velocities_sample)

    # Create subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i in range(6):
        ax = axes[i]
        actual = torques_sample[:, i]
        predicted = predictions[:, i]

        # Scatter plot
        ax.scatter(actual, predicted, alpha=0.6, s=10)

        # Perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')

        # Calculate metrics
        r2 = r2_score(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)

        # Set labels and title
        ax.set_xlabel(f'Actual Torque (Nm)')
        ax.set_ylabel(f'Predicted Torque (Nm)')
        ax.set_title(f'Joint {i+1}\nR² = {r2:.4f}, RMSE = {rmse:.4f}')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig('mlp_prediction_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_time_series_comparison(mlp_system, positions, torques, velocities, n_points=500):
    """Plot time series comparison for a few joints"""
    # Select first n_points for time series plot
    positions_ts = positions[:n_points]
    torques_ts = torques[:n_points]
    velocities_ts = velocities[:n_points]

    # Get predictions
    predictions_ts = mlp_system.predict(positions_ts, velocities_ts)

    # Create time array
    time_points = np.arange(len(positions_ts))

    # Plot for joints with good performance
    good_joints = [0, 1, 3, 4]  # Joints 1, 2, 4, 5 (0-indexed)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, joint_idx in enumerate(good_joints):
        ax = axes[idx]
        actual = torques_ts[:, joint_idx]
        predicted = predictions_ts[:, joint_idx]

        ax.plot(time_points, actual, 'b-', alpha=0.7, label='Actual', linewidth=1)
        ax.plot(time_points, predicted, 'r-', alpha=0.7, label='Predicted', linewidth=1)

        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Torque (Nm)')
        ax.set_title(f'Joint {joint_idx+1} - Time Series')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mlp_time_series.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main plotting function"""
    print("=== MLP Model Performance Visualization ===")

    # Load trained model
    model_path = "mlp_gravity_model.pkl"
    try:
        mlp_system = LightweightMLPGravityCompensation()
        mlp_system.load_model(model_path)
        print("✅ Model loaded successfully")
    except FileNotFoundError:
        print("❌ Model file not found. Please train the model first.")
        return

    # Load data
    log_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control/logs/20250926_175734"
    positions, velocities, torques = load_training_data(log_dir)
    print(f"✅ Loaded {len(positions)} data points")

    print("\n=== Generating Performance Plots ===")

    # Plot 1: Training performance
    print("1. Training performance metrics...")
    plot_training_performance(mlp_system)

    # Plot 2: Prediction accuracy
    print("2. Prediction accuracy scatter plots...")
    plot_prediction_accuracy(mlp_system, positions, torques, velocities)

    # Plot 3: Time series comparison
    print("3. Time series comparison...")
    plot_time_series_comparison(mlp_system, positions, torques, velocities)

    print("\n✅ All plots generated successfully!")
    print("Generated files:")
    print("- mlp_training_performance.png")
    print("- mlp_prediction_accuracy.png")
    print("- mlp_time_series.png")


if __name__ == "__main__":
    main()