#!/usr/bin/env python3
"""
MLP Gravity Compensation Model Evaluation Script
Comprehensive evaluation of the trained model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import time

# Add current directory to path for imports
import sys
sys.path.append('.')
from mlp_gravity_compensation import LightweightMLPGravityCompensation
from train_model import load_training_data_from_datasets


def load_model(model_path):
    """Load trained model"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    mlp_system = LightweightMLPGravityCompensation(
        hidden_layer_sizes=model_data['hidden_layer_sizes'],
        max_iter=model_data['max_iter'],
        random_state=model_data['random_state']
    )

    mlp_system.mlps = model_data['mlps']
    mlp_system.input_scaler = model_data['input_scaler']
    mlp_system.output_scaler = model_data['output_scaler']
    mlp_system.train_scores = model_data['train_scores']
    mlp_system.val_scores = model_data['val_scores']
    mlp_system.is_trained = model_data['is_trained']

    return mlp_system


def comprehensive_evaluation(mlp_system, positions, velocities, torques):
    """
    Comprehensive model evaluation

    Args:
        mlp_system: Trained MLP model
        positions: Joint positions
        velocities: Joint velocities
        torques: True torques
    """
    print("=== Comprehensive Model Evaluation ===")

    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        np.concatenate([positions, velocities], axis=1),
        torques,
        test_size=0.2,
        random_state=42
    )

    # Make predictions
    print("Making predictions...")
    start_time = time.time()
    y_pred = mlp_system.predict(X_test[:, :6], X_test[:, 6:])
    prediction_time = time.time() - start_time

    # Overall metrics
    print("\n=== Overall Performance ===")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.4f} Nm")
    print(f"MAE: {mae:.4f} Nm")
    print(f"R² Score: {r2:.4f}")
    print(f"Prediction time for {len(X_test)} samples: {prediction_time:.4f} s")
    print(f"Average prediction time: {prediction_time/len(X_test)*1000:.3f} ms")

    # Per-joint metrics
    print("\n=== Per-Joint Performance ===")
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']

    for i in range(6):
        rmse_joint = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        mae_joint = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2_joint = r2_score(y_test[:, i], y_pred[:, i])

        print(f"{joint_names[i]}:")
        print(f"  RMSE: {rmse_joint:.4f} Nm")
        print(f"  MAE: {mae_joint:.4f} Nm")
        print(f"  R²: {r2_joint:.4f}")

    # Error distribution analysis
    print("\n=== Error Analysis ===")
    errors = y_test - y_pred
    error_magnitudes = np.linalg.norm(errors, axis=1)

    print(f"Error magnitude statistics:")
    print(f"  Mean: {np.mean(error_magnitudes):.4f} Nm")
    print(f"  Std: {np.std(error_magnitudes):.4f} Nm")
    print(f"  Min: {np.min(error_magnitudes):.4f} Nm")
    print(f"  Max: {np.max(error_magnitudes):.4f} Nm")
    print(f"  Median: {np.median(error_magnitudes):.4f} Nm")

    # Percentiles
    percentiles = [50, 75, 90, 95, 99]
    print(f"\nError percentiles:")
    for p in percentiles:
        val = np.percentile(error_magnitudes, p)
        print(f"  {p}th percentile: {val:.4f} Nm")

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'prediction_time': prediction_time,
        'per_joint_metrics': {
            'rmse': [np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])) for i in range(6)],
            'mae': [np.sqrt(mean_absolute_error(y_test[:, i], y_pred[:, i])) for i in range(6)],
            'r2': [r2_score(y_test[:, i], y_pred[:, i]) for i in range(6)]
        },
        'error_stats': {
            'mean': np.mean(error_magnitudes),
            'std': np.std(error_magnitudes),
            'min': np.min(error_magnitudes),
            'max': np.max(error_magnitudes),
            'median': np.median(error_magnitudes)
        }
    }


def performance_test(mlp_system, n_tests=10000):
    """Test model performance under heavy load"""
    print(f"\n=== Performance Test ({n_tests} predictions) ===")

    # Generate random test data
    n_joints = 6
    test_positions = np.random.uniform(-np.pi, np.pi, (n_tests, n_joints))
    test_velocities = np.random.uniform(-2, 2, (n_tests, n_joints))

    # Benchmark prediction speed
    start_time = time.time()
    predictions = mlp_system.predict(test_positions, test_velocities)
    total_time = time.time() - start_time

    avg_time = total_time / n_tests * 1000  # Convert to ms
    frequency = 1000 / avg_time if avg_time > 0 else float('inf')

    print(f"Total time for {n_tests} predictions: {total_time:.4f} s")
    print(f"Average prediction time: {avg_time:.6f} ms")
    print(f"Prediction frequency: {frequency:.1f} Hz")

    # Test suitability for real-time control
    target_frequencies = [100, 200, 300, 500, 1000]
    print(f"\nReal-time suitability:")
    for target_freq in target_frequencies:
        required_time = 1000 / target_freq  # ms per prediction
        suitable = avg_time < required_time
        status = "✅ Suitable" if suitable else "❌ Not suitable"
        print(f"  {target_freq} Hz: {status} (requires {required_time:.3f} ms, we have {avg_time:.3f} ms)")

    return {
        'avg_time_ms': avg_time,
        'frequency_hz': frequency,
        'total_time_s': total_time
    }


def generate_visualizations(mlp_system, positions, velocities, torques, output_dir="."):
    """Generate evaluation visualizations"""
    print(f"\n=== Generating Visualizations ===")

    # Make predictions for visualization
    sample_size = min(1000, len(positions))
    sample_indices = np.random.choice(len(positions), sample_size, replace=False)

    sample_positions = positions[sample_indices]
    sample_velocities = velocities[sample_indices]
    sample_torques = torques[sample_indices]

    predictions = mlp_system.predict(sample_positions, sample_velocities)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('MLP Gravity Compensation Model Performance', fontsize=16)

    joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']

    for i, ax in enumerate(axes.flat):
        if i < 6:
            # Scatter plot: True vs Predicted
            true_vals = sample_torques[:, i]
            pred_vals = predictions[:, i]

            ax.scatter(true_vals, pred_vals, alpha=0.6, s=20)

            # Perfect prediction line
            min_val = min(np.min(true_vals), np.min(pred_vals))
            max_val = max(np.max(true_vals), np.max(pred_vals))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

            # Calculate R² for this subset
            r2 = r2_score(true_vals, pred_vals)
            rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))

            ax.set_xlabel(f'True Torque (Nm)')
            ax.set_ylabel(f'Predicted Torque (Nm)')
            ax.set_title(f'{joint_names[i]}\nR² = {r2:.3f}, RMSE = {rmse:.3f}')
            ax.grid(True, alpha=0.3)
            ax.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/mlp_evaluation_scatter.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved scatter plot to {output_dir}/mlp_evaluation_scatter.png")

    # Error distribution plot
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle('Prediction Error Distribution', fontsize=16)

    for i, ax in enumerate(axes2.flat):
        if i < 6:
            errors = sample_torques[:, i] - predictions[:, i]
            ax.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2)

            mean_error = np.mean(errors)
            std_error = np.std(errors)

            ax.set_xlabel(f'Prediction Error (Nm)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{joint_names[i]}\nMean = {mean_error:.3f}, Std = {std_error:.3f}')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/mlp_error_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved error distribution plot to {output_dir}/mlp_error_distribution.png")

    plt.close('all')


def main():
    """Main evaluation function"""
    print("=== MLP Gravity Compensation Model Evaluation ===")

    # Configuration
    dataset_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control/datasets"
    model_path = "mlp_gravity_model_new.pkl"
    output_dir = "."

    try:
        # Load data
        print(f"Loading data from {dataset_dir}")
        positions, velocities, torques = load_training_data_from_datasets(dataset_dir)
        print(f"Loaded {len(positions)} samples for evaluation")

        # Load model
        print(f"Loading model from {model_path}")
        mlp_system = load_model(model_path)

        # Comprehensive evaluation
        eval_results = comprehensive_evaluation(mlp_system, positions, velocities, torques)

        # Performance test
        perf_results = performance_test(mlp_system, n_tests=10000)

        # Generate visualizations
        generate_visualizations(mlp_system, positions, velocities, torques, output_dir)

        # Summary
        print(f"\n=== Evaluation Summary ===")
        print(f"Model: {model_path}")
        print(f"Dataset: {dataset_dir}")
        print(f"Samples: {len(positions)}")
        print(f"Overall RMSE: {eval_results['rmse']:.4f} Nm")
        print(f"Overall R²: {eval_results['r2']:.4f}")
        print(f"Prediction frequency: {perf_results['frequency_hz']:.1f} Hz")

        # Joint with worst performance
        worst_joint_idx = np.argmin(eval_results['per_joint_metrics']['r2'])
        worst_joint_r2 = eval_results['per_joint_metrics']['r2'][worst_joint_idx]
        print(f"Worst performing joint: Joint {worst_joint_idx + 1} (R² = {worst_joint_r2:.4f})")

        print(f"\n✅ Evaluation completed successfully!")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()