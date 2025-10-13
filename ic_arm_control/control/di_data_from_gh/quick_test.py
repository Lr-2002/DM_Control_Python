#!/usr/bin/env python3
"""
快速测试脚本 - 验证辨识流程是否正常工作
"""

import os
import sys


def check_log_data():
    """检查日志数据是否存在"""
    print("检查日志数据...")
    
    log_dirs = [
        "/Users/lr-2002/project/instantcreation/IC_arm_control/logs/20251010_133145_ic_arm_control",
        "/Users/lr-2002/project/instantcreation/IC_arm_control/logs/20251010_132725_ic_arm_control",
    ]
    
    all_exist = True
    for log_dir in log_dirs:
        motor_states = os.path.join(log_dir, "motor_states.csv")
        joint_commands = os.path.join(log_dir, "joint_commands.csv")
        
        if os.path.exists(motor_states) and os.path.exists(joint_commands):
            print(f"  ✓ {os.path.basename(log_dir)}")
        else:
            print(f"  ✗ {os.path.basename(log_dir)} - 缺少数据文件")
            all_exist = False
    
    return all_exist


def check_dependencies():
    """检查依赖项"""
    print("\n检查依赖项...")
    
    required_packages = ['numpy', 'pandas', 'scipy', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n请安装缺失的包:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False
    
    return True


def run_quick_test():
    """运行快速测试"""
    print("\n" + "=" * 80)
    print("快速测试 - 动力学参数辨识流程")
    print("=" * 80)
    
    # 1. 检查依赖
    if not check_dependencies():
        return False
    
    # 2. 检查数据
    if not check_log_data():
        print("\n❌ 日志数据不完整")
        print("请确保以下目录存在并包含 motor_states.csv 和 joint_commands.csv:")
        print("  - logs/20251010_133145_ic_arm_control/")
        print("  - logs/20251010_132725_ic_arm_control/")
        return False
    
    print("\n✅ 所有检查通过!")
    print("\n可以运行完整的辨识流程:")
    print("  python run_identification.py")
    
    return True


if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)
