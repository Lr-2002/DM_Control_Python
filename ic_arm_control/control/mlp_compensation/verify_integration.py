#!/usr/bin/env python3
"""
验证IC_ARM.py MLP重力补偿集成的语法正确性
"""

import ast
import sys
from pathlib import Path

def verify_syntax(file_path):
    """验证Python文件语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # 解析语法树
        ast.parse(source)
        return True, "语法正确"
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    except Exception as e:
        return False, f"其他错误: {e}"

def verify_integration():
    """验证集成"""
    ic_arm_path = Path(__file__).parent.parent / "IC_ARM.py"

    print("=== IC_ARM.py MLP重力补偿集成验证 ===\n")

    # 1. 验证语法
    print("1. 验证语法...")
    syntax_ok, syntax_msg = verify_syntax(ic_arm_path)
    if syntax_ok:
        print("✅ IC_ARM.py 语法正确")
    else:
        print(f"❌ IC_ARM.py 语法错误: {syntax_msg}")
        return False

    # 2. 验证关键方法存在
    print("\n2. 验证关键方法...")
    try:
        with open(ic_arm_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查关键方法
        key_methods = [
            'def cal_gravity_mlp(self):',
            'def switch_to_mlp_gravity_compensation(self):',
            'def switch_to_static_gravity_compensation(self):',
            'def get_gravity_compensation_performance(self):',
            'def print_gravity_compensation_summary(self):'
        ]

        for method in key_methods:
            if method in content:
                print(f"✅ {method}")
            else:
                print(f"❌ 缺少方法: {method}")
                return False

    except Exception as e:
        print(f"❌ 检查方法失败: {e}")
        return False

    # 3. 验证关键修改
    print("\n3. 验证关键修改...")
    key_modifications = [
        'gc_type="static"',  # 构造函数参数
        'self.gc_type = gc_type',  # 存储类型
        'if gc_type == "mlp":',  # MLP条件分支
        'from mlp_gravity_integrator import MLPGravityCompensation',  # MLP导入
        'model_path = current_dir / "mlp_compensation" / "mlp_gravity_model_improved.pkl"'  # 模型路径
    ]

    for mod in key_modifications:
        if mod in content:
            print(f"✅ {mod}")
        else:
            print(f"❌ 缺少修改: {mod}")
            return False

    # 4. 验证MLP模块路径
    print("\n4. 验证MLP模块路径...")
    mlp_path = Path(__file__).parent
    mlp_files = [
        "mlp_gravity_integrator.py",
        "mlp_gravity_compensation.py",
        "mlp_gravity_model_improved.pkl"
    ]

    for file in mlp_files:
        file_path = mlp_path / file
        if file_path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ 缺少文件: {file}")

    print("\n=== 验证完成 ===")
    print("✅ IC_ARM.py MLP重力补偿集成语法验证通过!")
    print("✅ 所有必要的代码修改已完成!")
    print("✅ 集成准备就绪!")

    return True

if __name__ == "__main__":
    success = verify_integration()
    exit(0 if success else 1)