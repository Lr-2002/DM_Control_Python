#!/usr/bin/env python3
"""
计算最优振幅，在不超过速度限制的前提下最大化激励效果
"""

import numpy as np

def calculate_optimal_amplitude(max_velocity=1.57, safety_factor=1.2):
    """
    计算不同频率下的最优振幅

    Args:
        max_velocity: 最大允许速度 (rad/s)
        safety_factor: 安全余量

    Returns:
        不同频率对应的最大振幅
    """
    frequencies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]

    print("频率 (Hz) | 最大振幅 (rad) | 安全速度 (rad/s)")
    print("-" * 50)

    for freq in frequencies:
        omega = 2 * np.pi * freq
        # 对于正弦运动: v_max = A * omega
        # 所以: A_max = v_max / (omega * safety_factor)
        max_amplitude = max_velocity / (omega * safety_factor)
        safe_velocity = max_amplitude * omega

        print(f"{freq:8.1f} | {max_amplitude:13.3f} | {safe_velocity:13.3f}")

    # 计算在典型频率下的合理振幅
    typical_freq = 0.5  # Hz
    omega_typical = 2 * np.pi * typical_freq
    optimal_amplitude = max_velocity / (omega_typical * safety_factor)

    print(f"\n在 {typical_freq} Hz 频率下的最优振幅: {optimal_amplitude:.3f} rad")
    print(f"对应的安全速度: {optimal_amplitude * omega_typical:.3f} rad/s")
    print(f"速度利用率: {(optimal_amplitude * omega_typical / max_velocity) * 100:.1f}%")

    return optimal_amplitude

if __name__ == "__main__":
    calculate_optimal_amplitude()