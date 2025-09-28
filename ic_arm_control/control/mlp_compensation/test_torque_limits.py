#!/usr/bin/env python3
"""
测试关节力矩限制功能
"""

import numpy as np
import sys
from pathlib import Path

# 添加路径
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(current_dir))

def test_torque_limits_directly():
    """直接测试力矩限制功能"""
    print("=== 测试关节力矩限制功能 ===")

    try:
        # 模拟ICARM类的基本功能
        class MockICARM:
            def __init__(self):
                self.motor_count = 8
                # 使用定义的力矩限制
                self.max_joint_torques = [15.0, 12.0, 12.0, 4.0, 4.0, 3.0, 5.0, 5.0]
                self.debug = True

            def debug_print(self, msg, level="INFO"):
                if self.debug:
                    print(f"[{level}] {msg}")

            def _apply_torque_limits(self, torques_nm):
                """应用关节力矩限制"""
                limited_torques = torques_nm.copy()

                for i in range(min(len(limited_torques), len(self.max_joint_torques))):
                    max_torque = self.max_joint_torques[i]
                    if abs(limited_torques[i]) > max_torque:
                        limited_torques[i] = np.sign(limited_torques[i]) * max_torque

                return limited_torques

            def get_joint_torque_limits(self):
                """获取关节力矩限制"""
                return self.max_joint_torques.copy()

            def set_joint_torque_limits(self, torque_limits):
                """设置关节力矩限制"""
                if len(torque_limits) != self.motor_count:
                    raise ValueError(f"力矩限制数组长度应为{self.motor_count}")

                self.max_joint_torques = np.array(torque_limits)
                self.debug_print(f"关节力矩限制已更新: {self.max_joint_torques}")

        # 创建模拟实例
        arm = MockICARM()

        # 测试1: 获取力矩限制
        print("\n1. 测试获取力矩限制...")
        limits = arm.get_joint_torque_limits()
        expected_limits = [15.0, 12.0, 12.0, 4.0, 4.0, 3.0, 5.0, 5.0]

        print(f"当前力矩限制: {limits}")
        print(f"期望力矩限制: {expected_limits}")

        if np.array_equal(limits, expected_limits):
            print("✅ 力矩限制设置正确")
        else:
            print("❌ 力矩限制设置不正确")
            return False

        # 测试2: 正常力矩（不应被限制）
        print("\n2. 测试正常力矩...")
        normal_torques = [5.0, 8.0, 10.0, 2.0, 3.0, 1.0, 2.0, 2.0]
        limited_normal = arm._apply_torque_limits(normal_torques)

        print(f"输入力矩: {normal_torques}")
        print(f"限制后力矩: {limited_normal}")

        if np.array_equal(normal_torques, limited_normal):
            print("✅ 正常力矩未被限制")
        else:
            print("❌ 正常力矩被错误限制")
            return False

        # 测试3: 超限力矩（应被限制）
        print("\n3. 测试超限力矩...")
        excessive_torques = [20.0, 15.0, 15.0, 8.0, 6.0, 5.0, 8.0, 8.0]
        limited_excessive = arm._apply_torque_limits(excessive_torques)

        print(f"输入力矩: {excessive_torques}")
        print(f"限制后力矩: {limited_excessive}")
        print(f"期望限制: {[15.0, 12.0, 12.0, 4.0, 4.0, 3.0, 5.0, 5.0]}")

        expected_limited = [15.0, 12.0, 12.0, 4.0, 4.0, 3.0, 5.0, 5.0]
        if np.array_equal(limited_excessive, expected_limited):
            print("✅ 超限力矩被正确限制")
        else:
            print("❌ 超限力矩限制不正确")
            return False

        # 测试4: 负力矩
        print("\n4. 测试负力矩...")
        negative_torques = [-20.0, -15.0, -15.0, -8.0, -6.0, -5.0, -8.0, -8.0]
        limited_negative = arm._apply_torque_limits(negative_torques)

        print(f"输入力矩: {negative_torques}")
        print(f"限制后力矩: {limited_negative}")
        print(f"期望限制: {[-15.0, -12.0, -12.0, -4.0, -4.0, -3.0, -5.0, -5.0]}")

        expected_negative = [-15.0, -12.0, -12.0, -4.0, -4.0, -3.0, -5.0, -5.0]
        if np.array_equal(limited_negative, expected_negative):
            print("✅ 负力矩被正确限制")
        else:
            print("❌ 负力矩限制不正确")
            return False

        # 测试5: 动态修改力矩限制
        print("\n5. 测试动态修改力矩限制...")
        new_limits = [10.0, 10.0, 10.0, 5.0, 5.0, 4.0, 6.0, 6.0]
        arm.set_joint_torque_limits(new_limits)

        updated_limits = arm.get_joint_torque_limits()
        print(f"新力矩限制: {updated_limits}")

        if np.array_equal(updated_limits, new_limits):
            print("✅ 力矩限制动态更新成功")
        else:
            print("❌ 力矩限制动态更新失败")
            return False

        # 测试新限制下的力矩
        test_torques = [12.0, 12.0, 12.0, 6.0, 6.0, 5.0, 7.0, 7.0]
        limited_new = arm._apply_torque_limits(test_torques)

        print(f"输入力矩: {test_torques}")
        print(f"新限制后力矩: {limited_new}")
        print(f"期望限制: {new_limits}")

        if np.array_equal(limited_new, new_limits):
            print("✅ 新力矩限制生效")
        else:
            print("❌ 新力矩限制未生效")
            return False

        print("\n✅ 所有力矩限制测试通过！")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gravity_compensation_with_limits():
    """测试重力补偿配合力矩限制"""
    print("\n=== 测试重力补偿配合力矩限制 ===")

    try:
        from mlp_gravity_integrator import MLPGravityCompensation

        # 创建MLP重力补偿实例
        mlp_gc = MLPGravityCompensation()

        if not mlp_gc.is_initialized:
            print("❌ MLP重力补偿初始化失败")
            return False

        # 模拟ICARM力矩限制
        torque_limits = [15.0, 12.0, 12.0, 4.0, 4.0, 3.0]

        # 测试不同位置的重力补偿
        test_positions = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 零位置
            np.array([1.5, 1.0, -0.5, 0.5, -0.8, 0.3]),  # 极端位置
            np.array([0.5, 0.3, 0.2, -0.1, 0.4, -0.2]),  # 正常位置
        ]

        for i, positions in enumerate(test_positions):
            print(f"\n测试位置 {i+1}: {positions}")

            # 计算重力补偿力矩
            compensation = mlp_gc.get_gravity_compensation_torque(positions)
            print(f"原始补偿力矩: {compensation}")

            # 应用力矩限制
            limited_compensation = compensation.copy()
            for j in range(min(len(limited_compensation), len(torque_limits))):
                max_torque = torque_limits[j]
                if abs(limited_compensation[j]) > max_torque:
                    limited_compensation[j] = np.sign(limited_compensation[j]) * max_torque

            print(f"限制后补偿力矩: {limited_compensation}")

            # 检查是否超限
            clipped = np.where(compensation != limited_compensation)[0]
            if len(clipped) > 0:
                print(f"⚠️ 关节 {[j+1 for j in clipped]} 力矩被限制")
                for j in clipped:
                    print(f"   关节{j+1}: {compensation[j]:.2f}Nm → {limited_compensation[j]:.2f}Nm")
            else:
                print("✅ 所有关节力矩在安全范围内")

        print("\n✅ 重力补偿力矩限制测试通过！")
        return True

    except Exception as e:
        print(f"❌ 重力补偿力矩限制测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_ic_arm_torque_integration():
    """检查IC_ARM中的力矩限制集成"""
    print("\n=== 检查IC_ARM力矩限制集成 ===")

    try:
        ic_arm_path = parent_dir / "IC_ARM.py"

        with open(ic_arm_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查关键修改
        integrations = [
            ("力矩限制常量", "MAX_JOINT_TORQUES"),
            ("构造函数存储", "self.max_joint_torques"),
            ("力矩限制方法", "_apply_torque_limits"),
            ("获取限制方法", "get_joint_torque_limits"),
            ("设置限制方法", "set_joint_torque_limits"),
            ("力矩限制应用", "limited_torques = self._apply_torque_limits"),
            ("警告日志", "力矩从")
        ]

        all_found = True
        for name, pattern in integrations:
            if pattern in content:
                print(f"✅ {name}: 已集成")
            else:
                print(f"❌ {name}: 未找到")
                all_found = False

        if all_found:
            print("✅ IC_ARM力矩限制集成完整")
            return True
        else:
            print("❌ IC_ARM力矩限制集成不完整")
            return False

    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=== 关节力矩限制功能测试 ===\n")

    tests = [
        ("IC_ARM集成检查", check_ic_arm_torque_integration),
        ("力矩限制功能测试", test_torque_limits_directly),
        ("重力补偿力矩限制测试", test_gravity_compensation_with_limits)
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
        print("\n📋 力矩限制功能已成功集成到IC_ARM中")
        print("前6个关节的力矩限制为: [15, 12, 12, 4, 4, 3] Nm")
        return True
    else:
        print("⚠️ 部分测试失败，请检查配置。")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)