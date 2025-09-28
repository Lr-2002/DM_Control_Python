#!/usr/bin/env python3
"""
Final Model Comparison Script
Compare the original model with the improved model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time
import sys
sys.path.append('.')
from train_model import load_training_data_from_datasets
from mlp_gravity_compensation import LightweightMLPGravityCompensation


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
    mlp_system.val_scores = model_data.get('val_scores', [])
    mlp_system.is_trained = model_data['is_trained']

    # Set enhanced training flag if present
    if 'enhanced_training' in model_data:
        mlp_system.train_enhanced = model_data['enhanced_training']
        mlp_system.enhanced_feature_dim = model_data.get('enhanced_feature_dim', 18)

    return mlp_system


def evaluate_model_performance(mlp_system, positions, velocities, torques, model_name="Model"):
    """Evaluate model performance"""
    print(f"\n=== {model_name} Performance ===")

    # Test on full dataset
    start_time = time.time()

    # Check if model uses enhanced features
    if hasattr(mlp_system, 'train_enhanced') and mlp_system.train_enhanced:
        # Use enhanced features
        enhanced_features = mlp_system.enhance_features(positions, velocities)
        predictions = mlp_system.predict_enhanced(enhanced_features)
    else:
        # Use standard features
        predictions = mlp_system.predict(positions, velocities)

    prediction_time = time.time() - start_time

    # Calculate metrics
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']
    results = {}

    print(f"Prediction time for {len(positions)} samples: {prediction_time:.4f} s")
    print(f"Average prediction time: {prediction_time/len(positions)*1000:.3f} ms")

    for i, joint_name in enumerate(joint_names):
        r2 = r2_score(torques[:, i], predictions[:, i])
        rmse = np.sqrt(mean_squared_error(torques[:, i], predictions[:, i]))
        mae = mean_absolute_error(torques[:, i], predictions[:, i])

        results[joint_name] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        }

        print(f"{joint_name}: R² = {r2:.4f}, RMSE = {rmse:.4f} Nm, MAE = {mae:.4f} Nm")

    return results, prediction_time


def compare_models():
    """Compare original and improved models"""
    print("=== MLP Model Comparison ===")

    # Load test data
    dataset_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control/datasets"
    positions, velocities, torques = load_training_data_from_datasets(dataset_dir)
    print(f"Loaded {len(positions)} samples for comparison")

    # Load models
    try:
        original_model = load_model("mlp_gravity_model_new.pkl")
        print("✅ Loaded original model")
    except FileNotFoundError:
        print("❌ Original model not found")
        original_model = None

    try:
        improved_model = load_model("mlp_gravity_model_improved.pkl")
        print("✅ Loaded improved model")
    except FileNotFoundError:
        print("❌ Improved model not found")
        improved_model = None

    # Evaluate models
    results = {}

    if original_model:
        orig_results, orig_time = evaluate_model_performance(
            original_model, positions, velocities, torques, "Original Model"
        )
        results['original'] = orig_results

    if improved_model:
        imp_results, imp_time = evaluate_model_performance(
            improved_model, positions, velocities, torques, "Improved Model"
        )
        results['improved'] = imp_results

    # Generate comparison table
    if original_model and improved_model:
        print(f"\n=== Model Improvement Summary ===")
        joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']

        print(f"{'Joint':<10} {'Orig R²':<10} {'Imp R²':<10} {'Improvement':<15} {'Orig RMSE':<12} {'Imp RMSE':<12}")
        print("-" * 80)

        for joint_name in joint_names:
            orig_r2 = results['original'][joint_name]['r2']
            imp_r2 = results['improved'][joint_name]['r2']
            orig_rmse = results['original'][joint_name]['rmse']
            imp_rmse = results['improved'][joint_name]['rmse']

            r2_improvement = imp_r2 - orig_r2
            rmse_improvement = orig_rmse - imp_rmse

            print(f"{joint_name:<10} {orig_r2:<10.4f} {imp_r2:<10.4f} {r2_improvement:+15.4f} {orig_rmse:<12.4f} {imp_rmse:<12.4f}")

        # Key improvements
        print(f"\n=== Key Improvements ===")
        joint1_improvement = results['improved']['Joint 1']['r2'] - results['original']['Joint 1']['r2']
        print(f"Joint 1 R² improvement: {joint1_improvement:+.4f}")
        print(f"Joint 1 went from {results['original']['Joint 1']['r2']:.4f} to {results['improved']['Joint 1']['r2']:.4f}")

        # Worst joint improvement
        worst_orig_idx = np.argmin([results['original'][j]['r2'] for j in joint_names])
        worst_imp_idx = np.argmin([results['improved'][j]['r2'] for j in joint_names])

        print(f"Original worst joint: {joint_names[worst_orig_idx]} (R² = {results['original'][joint_names[worst_orig_idx]]['r2']:.4f})")
        print(f"Improved worst joint: {joint_names[worst_imp_idx]} (R² = {results['improved'][joint_names[worst_imp_idx]]['r2']:.4f})")

    # Performance comparison
    print(f"\n=== Performance Characteristics ===")
    if improved_model:
        # Test real-time performance
        n_tests = 10000
        test_positions = positions[:n_tests]
        test_velocities = velocities[:n_tests]

        start_time = time.time()

        # Check if model uses enhanced features
        if hasattr(improved_model, 'train_enhanced') and improved_model.train_enhanced:
            # Use enhanced features
            enhanced_features = improved_model.enhance_features(test_positions, test_velocities)
            predictions = improved_model.predict_enhanced(enhanced_features)
        else:
            # Use standard features
            predictions = improved_model.predict(test_positions, test_velocities)

        total_time = time.time() - start_time

        avg_time = total_time / n_tests * 1000
        frequency = 1000 / avg_time if avg_time > 0 else float('inf')

        print(f"Improved model performance:")
        print(f"  Average prediction time: {avg_time:.6f} ms")
        print(f"  Prediction frequency: {frequency:.1f} Hz")
        print(f"  Real-time suitability: {'✅ Excellent' if frequency > 1000 else '✅ Good' if frequency > 500 else '✅ Suitable' if frequency > 300 else '❌ Insufficient'}")

    print(f"\n✅ Model comparison completed!")


if __name__ == "__main__":
    compare_models()