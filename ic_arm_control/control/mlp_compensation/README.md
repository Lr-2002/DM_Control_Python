# MLP-based Gravity Compensation

This module provides lightweight neural network-based gravity compensation for the IC ARM robot using scikit-learn MLPRegressor.

## Features

- **Lightweight**: Uses scikit-learn instead of heavy deep learning frameworks
- **Fast**: Optimized for real-time control (300+ Hz)
- **Safe**: Built-in torque limiting and validation
- **Accurate**: Separate MLP for each joint with cross-validation
- **Robust**: Robust scaling and outlier handling

## Files

- `mlp_gravity_compensation.py`: Main implementation
- `train_model.py`: Training script
- `test_model.py`: Testing and benchmarking script
- `__init__.py`: Module initialization

## Usage

### Training

```python
from mlp_compensation import LightweightMLPGravityCompensation

# Initialize model
model = LightweightMLPGravityCompensation(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    random_state=42
)

# Train with position and torque data
model.train(positions, torques, velocities)

# Save model
model.save_model('mlp_gravity_model.pkl')
```

### Inference

```python
# Load trained model
model = LightweightMLPGravityCompensation()
model.load_model('mlp_gravity_model.pkl')

# Compute gravity compensation for single position
gravity_torques = model.compute_gravity_compensation(joint_position)

# Batch predictions
predictions = model.predict(positions_batch, velocities_batch)
```

## Performance

- **Training time**: ~1-2 minutes for 10k samples
- **Inference time**: ~0.1-0.3 ms per prediction
- **Control frequency**: 300+ Hz achievable
- **Memory usage**: < 10MB for model and scalers

## Requirements

- numpy
- pandas
- scikit-learn
- matplotlib (for visualization)

## Model Architecture

- 6 separate MLP models (one per joint)
- Hidden layers: (100, 50) neurons
- Activation: ReLU
- Optimizer: Adam with adaptive learning rate
- Regularization: L2 with alpha=0.001
- Early stopping with 20 iteration patience

## Safety Features

- Torque limiting (±5.0 Nm)
- Input validation
- NaN and outlier filtering
- Robust scaling for input normalization