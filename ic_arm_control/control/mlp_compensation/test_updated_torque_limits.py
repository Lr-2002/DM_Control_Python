#!/usr/bin/env python3
"""
测试更新后的力矩限制功能
复用MLP中的力矩限制，而不是在IC_ARM中重复实现
"""

import numpy as np
import sys
from pathlib import Path

# 添加路径
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(current_dir))

def test_mlp_torque_limits():
    """测试MLP中的力矩限制功能"""
    print("=== 测试MLP中的力矩限制功能 ===")

    try:
        # 测试1: 直接测试LightweightMLPGravityCompensation
        print("\n1. 测试LightweightMLPGravityCompensation...")
        from mlp_gravity_compensation import LightweightMLPGravityCompensation

        # 使用默认力矩限制
        mlp_default = LightweightMLPGravityCompensation()
        print(f"默认力矩限制: {mlp_default.max_torques}")

        # 使用自定义力矩限制
        custom_limits = [15.0, 12.0, 12.0, 4.0, 4.0, 3.0]
        mlp_custom = LightweightMLPGravityCompensation(max_torques=custom_limits)
        print(f"自定义力矩限制: {mlp_custom.max_torques}")

        if np.array_equal(mlp_custom.max_torques, custom_limits):
            print("✅ 自定义力矩限制设置成功")
        else:
            print("❌ 自定义力矩限制设置失败")
            return False

        # 测试2: 测试MLPGravityCompensation集成器
        print("\n2. 测试MLPGravityCompensation集成器...")
        from mlp_gravity_integrator import MLPGravityCompensation

        # 使用默认力矩限制
        integrator_default = MLPGravityCompensation()
        print(f"集成器默认力矩限制: {integrator_default.max_torques}")

        # 使用自定义力矩限制
        integrator_custom = MLPGravityCompensation(max_torques=custom_limits)
        print(f"集成器自定义力矩限制: {integrator_custom.max_torques}")

        if np.array_equal(integrator_custom.max_torques, custom_limits):
            print("✅ 集成器自定义力矩限制设置成功")
        else:
            print("❌ 集成器自定义力矩限制设置失败")
            return False

        # 测试3: 模拟重力补偿计算
        print("\n3. 测试重力补偿计算...")
        if integrator_custom.is_initialized:
            # 测试不同位置的力矩限制
            test_positions = [
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 零位置
                np.array([1.5, 1.0, -0.5, 0.5, -0.8, 0.3]),  # 极端位置
            ]

            for i, positions in enumerate(test_positions):
                print(f"\n测试位置 {i+1}: {positions}")

                # 计算重力补偿
                compensation = integrator_custom.get_gravity_compensation_torque(positions)
                print(f"补偿力矩: {compensation}")

                # 检查是否超限
                for j, torque in enumerate(compensation):
                    max_torque = custom_limits[j] if j < len(custom_limits) else 5.0
                    if abs(torque) > max_torque + 0.01:  # 允许小的数值误差
                        print(f"⚠️ 关节{j+1}超限: {torque:.2f}Nm > {max_torque}Nm")
                    else:
                        print(f"✅ 关节{j+1}安全: {torque:.2f}Nm ≤ {max_torque}Nm")

        print("\n✅ MLP力矩限制功能测试通过")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ic_arm_integration():
    """测试IC_ARM集成（不直接导入，检查代码）"""
    print("\n=== 测试IC_ARM集成 ===")

    try:
        ic_arm_path = parent_dir / "IC_ARM.py"

        with open(ic_arm_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查IC_ARM中是否移除了力矩限制代码
        removed_features = [
            "MAX_JOINT_TORQUES",
            "self.max_joint_torques",
            "_apply_torque_limits",
            "get_joint_torque_limits",
            "set_joint_torque_limits"
        ]

        print("检查IC_ARM中是否移除了重复的力矩限制代码:")
        for feature in removed_features:
            if feature in content:
                print(f"❌ 仍包含: {feature}")
                return False
            else:
                print(f"✅ 已移除: {feature}")

        # 检查IC_ARM中是否正确传递了力矩限制参数
        integration_features = [
            "max_torques=[15.0, 12.0, 12.0, 4.0, 4.0, 3.0]"
        ]

        print("\n检查IC_ARM中的力矩限制参数传递:")
        for feature in integration_features:
            if feature in content:
                print(f"✅ 包含: {feature}")
            else:
                print(f"❌ 缺少: {feature}")
                return False

        print("\n✅ IC_ARM集成检查通过")
        return True

    except Exception as e:
        print(f"❌ IC_ARM集成检查失败: {e}")
        return False

def test_torque_limiting_effectiveness():
    """测试力矩限制效果"""
    print("\n=== 测试力矩限制效果 ===")

    try:
        from mlp_gravity_integrator import MLPGravityCompensation

        # 创建MLP重力补偿实例
        mlp_gc = MLPGravityCompensation(
            max_torques=[15.0, 12.0, 12.0, 4.0, 4.0, 3.0]
        )

        if not mlp_gc.is_initialized:
            print("❌ MLP初始化失败")
            return False

        # 测试极端位置，应该触发力矩限制
        extreme_positions = np.array([2.0, 1.8, -1.5, 1.5, -1.8, 1.0])
        print(f"测试极端位置: {extreme_positions}")

        # 计算补偿力矩
        compensation = mlp_gc.get_gravity_compensation_torque(extreme_positions)
        print(f"限制后补偿力矩: {compensation}")

        # 检查力矩限制
        limits = [15.0, 12.0, 12.0, 4.0, 4.0, 3.0]
        all_within_limits = True

        for i, torque in enumerate(compensation):
            if i < len(limits):
                max_torque = limits[i]
                if abs(torque) > max_torque + 0.01:  # 允许小的数值误差
                    print(f"❌ 关节{i+1}超限: {torque:.2f}Nm > {max_torque}Nm")
                    all_within_limits = False
                else:
                    print(f"✅ 关节{i+1}安全: {torque:.2f}Nm ≤ {max_torque}Nm")

        if all_within_limits:
            print("✅ 所有关节力矩都在限制范围内")
            return True
        else:
            print("❌ 有关节力矩超限")
            return False

    except Exception as e:
        print(f"❌ 力矩限制效果测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=== 更新后的力矩限制功能测试 ===\n")
    print("现在复用MLP中的力矩限制功能，而不是在IC_ARM中重复实现\n")

    tests = [
        ("IC_ARM集成检查", test_ic_arm_integration),
        ("MLP力矩限制功能", test_mlp_torque_limits),
        ("力矩限制效果", test_torque_limiting_effectiveness)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        result = test_func()
        results.append((test_name, result))

    # 测试结果汇总
    print(f"\n{'='*60}")
    print("=== 测试结果汇总 ===")

    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\n总计: {passed}/{len(results)} 项测试通过")

    if passed == len(results):
        print("🎉 所有力矩限制测试通过！")
        print("\n📋 修改总结:")
        print("✅ 移除了IC_ARM中重复的力矩限制代码")
        print("✅ 复用了MLP中已有的力矩限制功能")
        print("✅ 正确设置了关节力矩限制: [15, 12, 12, 4, 4, 3] Nm")
        print("✅ 集成器正确传递了力矩限制参数")
        return True
    else:
        print("⚠️ 部分测试失败，请检查配置。")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)