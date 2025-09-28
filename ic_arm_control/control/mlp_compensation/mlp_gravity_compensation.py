#!/usr/bin/env python3
"""
Lightweight MLP-based Gravity Compensation for IC ARM

This module uses scikit-learn's neural network for gravity compensation,
providing a lightweight alternative to deep learning frameworks.

Key features:
- Scikit-learn MLPRegressor
- Automatic cross-validation
- Early stopping
- Robust scaling
- Model persistence
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import time


class LightweightMLPGravityCompensation:
    """Lightweight MLP-based gravity compensation using scikit-learn"""

    def __init__(self, hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, max_torques=None):
        """
        Initialize lightweight MLP gravity compensation

        Args:
            hidden_layer_sizes: Tuple of hidden layer sizes
            max_iter: Maximum number of iterations
            random_state: Random seed for reproducibility
            max_torques: List of maximum torque limits for each joint (Nm)
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state

        # 设置力矩限制，默认使用指定的关节限制 [15, 12, 12, 4, 4, 3]
        if max_torques is None:
            self.max_torques = [15.0, 12.0, 12.0, 4.0, 4.0, 3.0]
        else:
            self.max_torques = max_torques[:6]  # 只取前6个关节

        # Initialize scalers
        self.input_scaler = RobustScaler()
        self.output_scaler = RobustScaler()

        # Initialize separate MLP for each joint (better performance)
        self.mlps = []
        for i in range(6):  # 6 joints
            mlp = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation='relu',
                solver='adam',
                alpha=0.001,  # L2 regularization
                batch_size=64,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=max_iter,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=20,
                random_state=random_state,
                tol=1e-6
            )
            self.mlps.append(mlp)

        self.is_trained = False
        self.train_scores = []
        self.val_scores = []

        # Enhanced training support
        self.train_enhanced = False
        self.enhanced_feature_dim = 12

    def train(self, positions, torques, velocities=None):
        """
        Train MLP models

        Args:
            positions: Joint positions (N, 6)
            torques: Target torques (N, 6)
            velocities: Joint velocities (N, 6) - optional
        """
        print("=== Training Lightweight MLP Models ===")

        # Prepare inputs
        if velocities is None:
            velocities = np.zeros_like(positions)

        inputs = np.concatenate([positions, velocities], axis=1)

        # Scale inputs and outputs
        inputs_scaled = self.input_scaler.fit_transform(inputs)
        torques_scaled = self.output_scaler.fit_transform(torques)

        # Train separate MLP for each joint
        train_r2_scores = []
        val_r2_scores = []

        for i, mlp in enumerate(self.mlps):
            print(f"Training MLP for Joint {i+1}...")

            # Split data for this joint
            X_train, X_val, y_train, y_val = train_test_split(
                inputs_scaled, torques_scaled[:, i], test_size=0.2, random_state=42
            )

            # Train MLP
            mlp.fit(X_train, y_train)

            # Evaluate
            train_pred = mlp.predict(X_train)
            val_pred = mlp.predict(X_val)

            train_r2 = r2_score(y_train, train_pred)
            val_r2 = r2_score(y_val, val_pred)

            train_r2_scores.append(train_r2)
            val_r2_scores.append(val_r2)

            print(f"  Joint {i+1}: Train R² = {train_r2:.4f}, Val R² = {val_r2:.4f}")

        self.train_scores = train_r2_scores
        self.val_scores = val_r2_scores
        self.is_trained = True

        print(f"✅ Training completed!")
        print(f"Average Train R²: {np.mean(train_r2_scores):.4f}")
        print(f"Average Val R²: {np.mean(val_r2_scores):.4f}")

    def predict(self, positions, velocities=None):
        """
        Predict gravity torques

        Args:
            positions: Joint positions (N, 6)
            velocities: Joint velocities (N, 6) - optional

        Returns:
            predicted_torques: Predicted gravity torques (N, 6)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        # Prepare inputs
        if velocities is None:
            velocities = np.zeros_like(positions)

        inputs = np.concatenate([positions, velocities], axis=1)
        inputs_scaled = self.input_scaler.transform(inputs)

        # Predict for each joint
        predictions_scaled = np.zeros((len(positions), 6))
        for i, mlp in enumerate(self.mlps):
            predictions_scaled[:, i] = mlp.predict(inputs_scaled)

        # Inverse transform
        predictions = self.output_scaler.inverse_transform(predictions_scaled)

        # Apply safety limits for each joint
        for i in range(predictions.shape[1]):
            if i < len(self.max_torques):
                max_torque = self.max_torques[i]
                predictions[:, i] = np.clip(predictions[:, i], -max_torque, max_torque)

        return predictions

    def compute_gravity_compensation(self, joint_positions):
        """
        Compute gravity compensation for single position

        Args:
            joint_positions: Single joint position vector (6,)

        Returns:
            gravity_torques: Gravity compensation torques (6,)
        """
        if joint_positions.ndim == 1:
            joint_positions = joint_positions.reshape(1, -1)

        torques = self.predict(joint_positions)
        return torques.flatten()

    def predict_enhanced(self, enhanced_features):
        """
        Predict using enhanced features for improved performance

        Args:
            enhanced_features: Enhanced feature matrix (N, enhanced_feature_dim)

        Returns:
            predictions: Predicted torques (N, 6)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        # Scale enhanced features
        enhanced_features_scaled = self.input_scaler.transform(enhanced_features)

        # Predict for each joint
        predictions_scaled = np.zeros((len(enhanced_features), 6))
        for i, mlp in enumerate(self.mlps):
            predictions_scaled[:, i] = mlp.predict(enhanced_features_scaled)

        # Inverse transform
        predictions = self.output_scaler.inverse_transform(predictions_scaled)

        # Apply safety limits for each joint
        for i in range(predictions.shape[1]):
            if i < len(self.max_torques):
                max_torque = self.max_torques[i]
                predictions[:, i] = np.clip(predictions[:, i], -max_torque, max_torque)

        return predictions

    def enhance_features(self, positions, velocities):
        """
        Enhance input features for better Joint 1 modeling

        Args:
            positions: Joint positions (N, 6)
            velocities: Joint velocities (N, 6)

        Returns:
            enhanced_features: Enhanced feature matrix (N, enhanced_feature_dim)
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

    def save_model(self, filepath):
        """Save trained model"""
        if not self.is_trained:
            print("⚠️ No model to save")
            return

        model_data = {
            'mlps': self.mlps,
            'input_scaler': self.input_scaler,
            'output_scaler': self.output_scaler,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'train_scores': self.train_scores,
            'val_scores': self.val_scores,
            'is_trained': self.is_trained
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.mlps = model_data['mlps']
        self.input_scaler = model_data['input_scaler']
        self.output_scaler = model_data['output_scaler']
        self.hidden_layer_sizes = model_data['hidden_layer_sizes']
        self.max_iter = model_data['max_iter']
        self.random_state = model_data['random_state']
        self.train_scores = model_data['train_scores']
        self.val_scores = model_data['val_scores']
        self.is_trained = model_data['is_trained']

        print(f"✅ Model loaded from {filepath}")

    def get_performance_summary(self):
        """Get model performance summary"""
        if not self.is_trained:
            return "Model not trained"

        summary = "=== Model Performance Summary ===\n"
        summary += f"Hidden layers: {self.hidden_layer_sizes}\n"
        summary += f"Max iterations: {self.max_iter}\n\n"

        summary += "Per-joint R² scores:\n"
        for i in range(6):
            summary += f"  Joint {i+1}: Train = {self.train_scores[i]:.4f}, Val = {self.val_scores[i]:.4f}\n"

        summary += f"\nAverage Train R²: {np.mean(self.train_scores):.4f}\n"
        summary += f"Average Val R²: {np.mean(self.val_scores):.4f}\n"

        return summary