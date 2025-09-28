#!/usr/bin/env python3
"""
Test script for MLP gravity compensation model
"""

import numpy as np
import time
from mlp_gravity_compensation import LightweightMLPGravityCompensation


def main():
    """Main testing function"""
    print("=== Testing MLP Gravity Compensation Model ===")

    # Initialize model
    mlp_system = LightweightMLPGravityCompensation(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=42
    )

    # Try to load trained model
    model_path = "mlp_gravity_model.pkl"
    try:
        mlp_system.load_model(model_path)
        print("✅ Model loaded successfully")
    except FileNotFoundError:
        print("❌ Model file not found. Please train the model first.")
        return

    # Test positions
    test_positions = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, -0.3, 0.2, -0.1, 0.4, -0.2],
        [1.0, -0.5, 0.5, -0.3, 0.8, -0.4],
        [0.2, 0.3, -0.1, 0.6, -0.2, 0.4],
        [-0.3, 0.7, 0.1, -0.5, 0.3, -0.6]
    ])

    # Test predictions
    print("\n=== Test Predictions ===")
    for i, pos in enumerate(test_positions):
        pred = mlp_system.compute_gravity_compensation(pos)
        print(f"Position {i+1}: {pos}")
        print(f"Torques:    {pred}")
        print(f"Max torque: {np.max(np.abs(pred)):.3f} Nm")
        print()

    # Performance benchmark
    print("=== Performance Benchmark ===")
    n_tests = 10000

    # Generate random test positions
    np.random.seed(42)
    bench_positions = np.random.uniform(-1.5, 1.5, (n_tests, 6))

    # Single predictions
    start_time = time.time()
    for i in range(n_tests):
        pred = mlp_system.compute_gravity_compensation(bench_positions[i])
    single_time = (time.time() - start_time) / n_tests * 1000

    # Batch predictions
    start_time = time.time()
    pred_batch = mlp_system.predict(bench_positions)
    batch_time = (time.time() - start_time) / n_tests * 1000

    print(f"Single prediction time: {single_time:.3f} ms ({1000/single_time:.1f} Hz)")
    print(f"Batch prediction time:   {batch_time:.3f} ms ({1000/batch_time:.1f} Hz)")

    # Check suitability for real-time control
    target_freq = 300
    if 1000/single_time > target_freq:
        print(f"✅ Suitable for {target_freq}Hz control loop")
    else:
        print(f"⚠️ May not achieve {target_freq}Hz with single predictions")

    if 1000/batch_time > target_freq:
        print(f"✅ Suitable for {target_freq}Hz with batch processing")
    else:
        print(f"⚠️ May not achieve {target_freq}Hz with batch processing")

    # Verify safety limits
    print("\n=== Safety Check ===")
    max_torques = np.max(np.abs(pred_batch), axis=0)
    print(f"Maximum torques per joint: {max_torques}")
    print(f"Safety limit: {mlp_system.max_torque} Nm")

    if np.all(max_torques <= mlp_system.max_torque):
        print("✅ All torques within safety limits")
    else:
        print("⚠️ Some torques exceed safety limits")

    print("\n✅ MLP gravity compensation model testing completed!")


if __name__ == "__main__":
    main()