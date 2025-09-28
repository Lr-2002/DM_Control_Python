"""
MLP-based Gravity Compensation Module

This module provides lightweight neural network-based gravity compensation
for the IC ARM robot using scikit-learn MLPRegressor.
"""

from .mlp_gravity_compensation import LightweightMLPGravityCompensation

__all__ = ['LightweightMLPGravityCompensation']