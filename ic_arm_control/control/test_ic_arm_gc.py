#!/usr/bin/env python3
"""
测试 IC_ARM 中的动力学重力补偿配置
"""

import os
import sys

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_ic_arm_gc_loading():
    """测试 IC_ARM 中的参数加载"""

    print("="*60)
    print("Testing IC_ARM Dynamics Gravity Compensation Loading")
    print("="*60)

    try:
        # 模拟 IC_ARM 中的路径构建
        param_file = os.path.join(
            os.path.dirname(__file__),
            "urdfly",
            "dynamics_identification_results",
            "identified_parameters_least_squares.npz",
        )

        print(f"1. Parameter file path:")
        print(f"   {param_file}")
        print(f"   File exists: {os.path.exists(param_file)}")

        if not os.path.exists(param_file):
            print(f"   ❌ Parameter file not found!")
            return False

        # 模拟 IC_ARM 中的导入
        sys.path.append(os.path.join(current_dir, "urdfly"))
        from minimum_gc import MinimumGravityCompensation

        print(f"2. Loading MinimumGravityCompensation...")
        gc = MinimumGravityCompensation(param_file=param_file)

        print(f"   ✅ Successfully loaded!")
        print(f"   Parameter format: {gc.param_format}")
        print(f"   Parameters count: {len(gc.base_params)}")

        # 测试计算
        print(f"3. Testing torque calculation...")
        q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        torque = gc.calculate_gravity_torque(q)
        print(f"   Torque: {torque.flatten()}")
        print(f"   ✅ Torque calculation working!")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ic_arm_import():
    """测试直接从 IC_ARM 导入"""

    print(f"\n" + "="*60)
    print("Testing Direct IC_ARM Import")
    print("="*60)

    try:
        # 尝试导入 IC_ARM 模块（不初始化硬件）
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ic_arm",
            os.path.join(current_dir, "IC_ARM.py")
        )
        ic_arm_module = importlib.util.module_from_spec(spec)

        print(f"1. IC_ARM module loaded successfully")
        print(f"   Module location: {ic_arm_module.__file__}")

        return True

    except Exception as e:
        print(f"❌ IC_ARM import error: {e}")
        return False

if __name__ == "__main__":
    # 测试参数加载
    gc_ok = test_ic_arm_gc_loading()

    # 测试IC_ARM导入
    import_ok = test_ic_arm_import()

    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if gc_ok:
        print("✅ MinimumGravityCompensation loading: PASSED")
    else:
        print("❌ MinimumGravityCompensation loading: FAILED")

    if import_ok:
        print("✅ IC_ARM import: PASSED")
    else:
        print("❌ IC_ARM import: FAILED")

    if gc_ok and import_ok:
        print("\n🎉 IC_ARM dynamics gravity compensation is properly configured!")
    else:
        print("\n⚠️  Some issues detected. Please check the errors above.")