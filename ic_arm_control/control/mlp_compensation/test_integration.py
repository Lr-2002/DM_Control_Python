#!/usr/bin/env python3
"""
测试MLP重力补偿与IC_ARM的集成
"""

import numpy as np
import sys
from pathlib import Path

# 添加路径以导入模块
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def test_mlp_integrator():
    """测试MLP集成器"""
    print("=== 测试MLP重力补偿集成器 ===")

    try:
        from mlp_gravity_integrator import MLPGravityCompensation

        # 创建MLP重力补偿实例
        mlp_gc = MLPGravityCompensation(
            model_path="mlp_gravity_model_improved.pkl",
            enable_enhanced=True,
            debug=True
        )

        if not mlp_gc.is_initialized:
            print("❌ MLP重力补偿初始化失败")
            return False

        print("✅ MLP重力补偿集成器测试成功")

        # 测试基本功能
        test_positions = np.array([0.0, 0.5, 1.0, 0.2, -0.3, 0.8])
        torque = mlp_gc.get_gravity_compensation_torque(test_positions)

        print(f"测试位置: {test_positions}")
        print(f"计算力矩: {torque}")
        print(f"力矩范围: [{np.min(torque):.3f}, {np.max(torque):.3f}] Nm")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compatibility_with_static_gc():
    """测试与静态重力补偿的兼容性"""
    print("\n=== 测试兼容性 ===")

    try:
        from mlp_gravity_integrator import StaticGravityCompensation

        # 使用兼容性别名创建实例
        gc = StaticGravityCompensation()

        # 测试兼容的方法
        positions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        # 测试get_gravity_compensation_torque方法
        torque = gc.get_gravity_compensation_torque(positions)
        print(f"兼容性测试 - 重力补偿力矩: {torque}")

        # 测试calculate_torque方法
        full_torque = gc.calculate_torque(positions, np.zeros(6))
        print(f"兼容性测试 - 完整力矩: {full_torque}")

        # 测试calculate_coriolis_torque方法
        coriolis_torque = gc.calculate_coriolis_torque(positions, np.zeros(6))
        print(f"兼容性测试 - 科里奥利力矩: {coriolis_torque}")

        print("✅ 与静态重力补偿接口兼容")
        return True

    except Exception as e:
        print(f"❌ 兼容性测试失败: {e}")
        return False


def simulate_ic_arm_usage():
    """模拟IC_ARM使用场景"""
    print("\n=== 模拟IC_ARM使用场景 ===")

    try:
        from mlp_gravity_integrator import MLPGravityCompensation

        # 模拟IC_ARM的初始化过程
        class MockICARM:
            def __init__(self):
                self.gc_flag = True
                self.gc_type = "mlp"
                self.motor_count = 8
                self.q = np.zeros(8)  # 模拟关节状态

                # 初始化MLP重力补偿
                try:
                    self.gc = MLPGravityCompensation(
                        model_path="mlp_gravity_model_improved.pkl",
                        enable_enhanced=True,
                        debug=False
                    )
                    print("✅ IC_ARM模拟: MLP重力补偿初始化成功")
                except Exception as e:
                    print(f"❌ IC_ARM模拟: MLP重力补偿初始化失败: {e}")
                    self.gc = None

            def cal_gravity(self):
                """模拟IC_ARM的cal_gravity方法"""
                if not self.gc_flag or not self.gc:
                    return np.zeros(self.motor_count)

                if self.gc_type == "mlp":
                    # 使用MLP重力补偿
                    positions = self.q[:6]
                    compensation_torque = self.gc.get_gravity_compensation_torque(positions)

                    # 扩展到所有电机
                    full_compensation = np.zeros(self.motor_count)
                    full_compensation[:6] = compensation_torque

                    return full_compensation
                else:
                    return np.zeros(self.motor_count)

            def switch_to_mlp_gravity_compensation(self):
                """切换到MLP重力补偿"""
                try:
                    self.gc = MLPGravityCompensation(
                        model_path="mlp_gravity_model_improved.pkl",
                        enable_enhanced=True,
                        debug=False
                    )
                    self.gc_type = "mlp"
                    print("✅ 切换到MLP重力补偿成功")
                    return True
                except Exception as e:
                    print(f"❌ 切换失败: {e}")
                    return False

        # 创建模拟IC_ARM实例
        arm = MockICARM()

        # 测试重力补偿计算
        arm.q = np.array([0.1, 0.3, 0.5, -0.2, 0.8, -0.1, 0.0, 0.0])
        compensation = arm.cal_gravity()

        print(f"模拟关节状态: {arm.q[:6]}")
        print(f"计算的重力补偿: {compensation}")
        print(f"前6个关节补偿力矩: {compensation[:6]}")

        # 测试切换功能
        print("\n测试动态切换功能...")
        arm.switch_to_mlp_gravity_compensation()

        # 再次计算补偿
        new_compensation = arm.cal_gravity()
        print(f"切换后补偿力矩: {new_compensation[:6]}")

        print("✅ ICARM使用场景模拟成功")
        return True

    except Exception as e:
        print(f"❌ ICARM使用场景模拟失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def performance_test():
    """性能测试"""
    print("\n=== 性能测试 ===")

    try:
        from mlp_gravity_integrator import MLPGravityCompensation

        mlp_gc = MLPGravityCompensation()

        if not mlp_gc.is_initialized:
            print("❌ 性能测试: 模型未初始化")
            return False

        # 生成测试数据
        n_tests = 1000
        test_positions = np.random.uniform(-np.pi, np.pi, (n_tests, 6))

        # 性能测试
        print(f"进行 {n_tests} 次重力补偿计算...")

        import time
        start_time = time.time()

        for i in range(n_tests):
            torque = mlp_gc.get_gravity_compensation_torque(test_positions[i])

        total_time = time.time() - start_time
        avg_time = total_time / n_tests * 1000  # ms
        frequency = 1000 / avg_time

        print(f"总时间: {total_time:.3f} s")
        print(f"平均时间: {avg_time:.3f} ms")
        print(f"计算频率: {frequency:.1f} Hz")

        # 获取内部性能统计
        stats = mlp_gc.get_performance_stats()
        print(f"内部统计频率: {stats['frequency_hz']:.1f} Hz")

        # 评估是否满足实时要求
        if frequency > 1000:
            print("✅ 性能优秀: 满足1000Hz+实时控制要求")
        elif frequency > 500:
            print("✅ 性能良好: 满足500Hz控制要求")
        elif frequency > 300:
            print("✅ 性能合格: 满足300Hz控制要求")
        else:
            print("⚠️ 性能不足: 可能无法满足实时控制要求")

        return True

    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("=== MLP重力补偿集成完整测试 ===\n")

    tests = [
        ("MLP集成器测试", test_mlp_integrator),
        ("兼容性测试", test_compatibility_with_static_gc),
        ("IC_ARM使用场景模拟", simulate_ic_arm_usage),
        ("性能测试", performance_test)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        result = test_func()
        results.append((test_name, result))

    # 测试结果汇总
    print(f"\n{'='*50}")
    print("=== 测试结果汇总 ===")

    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\n总计: {passed}/{len(results)} 项测试通过")

    if passed == len(results):
        print("🎉 所有测试通过！MLP重力补偿集成就绪。")
        return True
    else:
        print("⚠️ 部分测试失败，请检查配置。")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)