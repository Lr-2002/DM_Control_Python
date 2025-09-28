#!/usr/bin/env python3
"""
快速测试MLP重力补偿集成
直接运行，验证基本功能
"""

import numpy as np
import sys
from pathlib import Path

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def quick_mlp_test():
    """快速MLP测试"""
    print("🚀 快速测试MLP重力补偿集成\n")

    try:
        # 导入MLP模块
        from mlp_gravity_integrator import MLPGravityCompensation

        print("✅ MLP重力补偿模块导入成功")

        # 创建实例
        mlp = MLPGravityCompensation()

        if mlp.is_initialized:
            print("✅ MLP模型初始化成功")
        else:
            print("❌ MLP模型初始化失败")
            return False

        # 测试计算
        test_pos = np.array([0.5, 1.0, 0.0, -0.5, 0.8, 0.2])
        torque = mlp.get_gravity_compensation_torque(test_pos)

        print(f"测试位置: {test_pos}")
        print(f"计算力矩: {torque}")
        print(f"力矩范围: [{np.min(torque):.2f}, {np.max(torque):.2f}] Nm")

        # 性能测试
        print("\n⏱️ 性能测试...")
        import time
        start = time.time()

        for _ in range(500):
            _ = mlp.get_gravity_compensation_torque(test_pos)

        elapsed = time.time() - start
        freq = 500 / elapsed

        print(f"500次计算时间: {elapsed:.3f}s")
        print(f"计算频率: {freq:.0f} Hz")

        if freq > 1000:
            print("🌟 性能优秀！")
        elif freq > 500:
            print("✅ 性能良好")
        else:
            print("⚠️ 性能需要优化")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def check_ic_arm_modifications():
    """检查IC_ARM修改"""
    print("\n🔍 检查IC_ARM.py修改...")

    try:
        ic_arm_path = current_dir.parent / "IC_ARM.py"

        with open(ic_arm_path, 'r') as f:
            content = f.read()

        # 关键修改检查
        checks = [
            ("MLP路径管理", "mlp_compensation_dir"),
            ("gc_type参数", 'gc_type="static"'),
            ("MLP初始化", 'if gc_type == "mlp":'),
            ("MLP方法", "def cal_gravity_mlp(self):"),
            ("切换方法", "def switch_to_mlp_gravity_compensation(self):")
        ]

        all_ok = True
        for name, pattern in checks:
            if pattern in content:
                print(f"✅ {name}")
            else:
                print(f"❌ {name} - 未找到")
                all_ok = False

        if all_ok:
            print("✅ IC_ARM.py修改完整")
        else:
            print("❌ IC_ARM.py修改不完整")

        return all_ok

    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

def main():
    """主函数"""
    print("=== MLP重力补偿集成快速测试 ===\n")

    # 检查代码修改
    code_ok = check_ic_arm_modifications()

    # 测试MLP功能
    mlp_ok = quick_mlp_test()

    # 结果
    print(f"\n{'='*50}")
    if code_ok and mlp_ok:
        print("🎉 集成成功！所有功能正常")
        print("\n📋 使用方法:")
        print("   arm = ICARM(gc=True, gc_type='mlp')")
        print("   compensation = arm.cal_gravity()")
        return True
    else:
        print("❌ 集成存在问题，请检查")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)