#!/usr/bin/env python3
"""
详细解释18维特征的构成
"""

import numpy as np
import pandas as pd


def explain_feature_construction():
    """详细解释18维特征如何构成"""
    print("=== 18维特征构成详解 ===\n")

    print("📊 基础输入特征 (12维):")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│ 位置特征 (6维):                                            │")
    print("│   [p1, p2, p3, p4, p5, p6]                                │")
    print("│   各关节角度值 (弧度)                                      │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ 速度特征 (6维):                                            │")
    print("│   [v1, v2, v3, v4, v5, v6]                                │")
    print("│   各关节角速度值 (弧度/秒)                                 │")
    print("└─────────────────────────────────────────────────────────────┘\n")

    print("🔧 Joint 1 增强特征 (6维):")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│ 1. 位置平方:     p1²                                        │")
    print("│    作用: 捕获非线性关系，特别是重力补偿中的二次效应         │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ 2. 速度平方:     v1²                                        │")
    print("│    作用: 捕获速度相关的非线性效应，如离心力                 │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ 3. 位置×速度:   p1 × v1                                    │")
    print("│    作用: 捕获位置和速度的耦合效应                           │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ 4. sin(位置):    sin(p1)                                   │")
    print("│    作用: 捕获角度的周期性特征，对旋转关节特别重要           │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ 5. cos(位置):    cos(p1)                                   │")
    print("│    作用: 与sin互补，提供完整的角度信息                     │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ 6. 速度方向:    sign(v1)                                   │")
    print("│    作用: 捕获运动方向信息                                   │")
    print("└─────────────────────────────────────────────────────────────┘\n")

    print("📋 完整特征列表 (18维):")
    features = [
        # 基础位置特征 (1-6)
        "p1", "p2", "p3", "p4", "p5", "p6",
        # 基础速度特征 (7-12)
        "v1", "v2", "v3", "v4", "v5", "v6",
        # Joint 1 增强特征 (13-18)
        "p1²", "v1²", "p1×v1", "sin(p1)", "cos(p1)", "sign(v1)"
    ]

    print("┌─────┬─────────────────┬─────────────────────────────────────┐")
    print("│ 索引 │ 特征名称        │ 说明                                 │")
    print("├─────┼─────────────────┼─────────────────────────────────────┤")
    for i, feature in enumerate(features):
        if i < 6:
            category = "基础位置"
        elif i < 12:
            category = "基础速度"
        else:
            category = "Joint 1 增强"
        print(f"│ {i+1:2d}  │ {feature:<15} │ {category:<25} │")
    print("└─────┴─────────────────┴─────────────────────────────────────┘\n")

    print("🎯 为什么只对Joint 1做增强？")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│ 数据分析结果:                                              │")
    print("│ • Joint 1 扭矩方差: 30.985 (最大)                         │")
    print("│ • Joint 6 扭矩方差: 0.021 (最小)                          │")
    print("│ • 方差差异倍数: 1510.8x                                    │")
    print("│ • Joint 1 最难建模，需要特殊处理                           │")
    print("└─────────────────────────────────────────────────────────────┘\n")

    print("⚡ 特征工程的效果:")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│ 原始模型 (12维特征):                                       │")
    print("│ • Joint 1 R² = -0.5011 (完全失效)                         │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ 增强模型 (18维特征):                                       │")
    print("│ • Joint 1 R² = 0.5635 (显著提升)                          │")
    print("│ • 提升: +1.0646 R²                                       │")
    print("└─────────────────────────────────────────────────────────────┘")


def demonstrate_feature_construction():
    """演示特征构造过程"""
    print("=== 特征构造演示 ===\n")

    # 模拟输入数据
    np.random.seed(42)
    positions = np.random.uniform(-np.pi, np.pi, (3, 6))  # 3个样本，6个关节
    velocities = np.random.uniform(-2, 2, (3, 6))       # 3个样本，6个关节

    print("🔢 输入数据 (3个样本):")
    print("位置:")
    print(positions)
    print("\n速度:")
    print(velocities)

    # 基础特征
    base_features = np.concatenate([positions, velocities], axis=1)
    print(f"\n📊 基础特征 (12维): {base_features.shape}")
    print(base_features)

    # Joint 1 增强特征
    joint1_pos = positions[:, 0:1]
    joint1_vel = velocities[:, 0:1]

    enhanced = np.column_stack([
        joint1_pos ** 2,           # 位置平方
        joint1_vel ** 2,           # 速度平方
        joint1_pos * joint1_vel,   # 交叉项
        np.sin(joint1_pos),        # sin
        np.cos(joint1_pos),        # cos
        np.sign(joint1_vel)        # 速度方向
    ])

    print(f"\n🔧 Joint 1 增强特征 (6维): {enhanced.shape}")
    print(enhanced)

    # 完整特征
    full_features = np.concatenate([base_features, enhanced], axis=1)
    print(f"\n🎯 完整特征 (18维): {full_features.shape}")
    print(full_features)

    print(f"\n✅ 成功构造 {full_features.shape[1]} 维特征")


def main():
    """主函数"""
    explain_feature_construction()
    print("\n" + "="*60 + "\n")
    demonstrate_feature_construction()


if __name__ == "__main__":
    main()