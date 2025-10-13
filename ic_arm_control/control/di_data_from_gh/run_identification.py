#!/usr/bin/env python3
"""
动力学参数辨识主控脚本
按照 shamilmamedov/dynamic_calibration 的完整流程

运行顺序:
1. Step 1: 数据预处理 (filterData.m)
2. Step 2: 参数估计 (ordinaryLeastSquareEstimation)
3. Step 3: 验证 (Validation)
"""

import os
import sys


def run_step(step_number, script_name, description):
    """运行单个步骤"""
    print("\n" + "=" * 80)
    print(f"Step {step_number}: {description}")
    print("=" * 80)
    
    # 运行脚本
    exit_code = os.system(f"python {script_name}")
    
    if exit_code != 0:
        print(f"\n❌ Step {step_number} 失败 (退出码: {exit_code})")
        return False
    
    print(f"\n✅ Step {step_number} 完成")
    return True


def main():
    """主函数 - 运行完整的辨识流程"""
    
    print("=" * 80)
    print("动力学参数辨识流程")
    print("基于 shamilmamedov/dynamic_calibration")
    print("=" * 80)
    
    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"\n工作目录: {script_dir}")
    
    # 询问用户要运行哪些步骤
    print("\n请选择要运行的步骤:")
    print("  1. 仅 Step 1 (数据预处理)")
    print("  2. 仅 Step 2 (参数估计)")
    print("  3. 仅 Step 3 (验证)")
    print("  4. 全部步骤 (推荐)")
    print("  5. Step 1 + Step 2")
    print("  6. Step 2 + Step 3")
    
    choice = input("\n请输入选择 (1-6, 默认4): ").strip() or "4"
    
    steps_to_run = {
        "1": [1],
        "2": [2],
        "3": [3],
        "4": [1, 2, 3],
        "5": [1, 2],
        "6": [2, 3]
    }
    
    if choice not in steps_to_run:
        print(f"❌ 无效选择: {choice}")
        return
    
    selected_steps = steps_to_run[choice]
    
    # 定义步骤
    steps = {
        1: ("step1_data_preprocessing.py", "数据预处理 (filterData.m)"),
        2: ("step2_parameter_estimation.py", "参数估计 (OLS)"),
        3: ("step3_validation.py", "验证")
    }
    
    # 运行选定的步骤
    for step_num in selected_steps:
        script_name, description = steps[step_num]
        
        success = run_step(step_num, script_name, description)
        
        if not success:
            print(f"\n流程在 Step {step_num} 中断")
            return
        
        # 询问是否继续
        if step_num < max(selected_steps):
            response = input(f"\n继续运行下一步? (y/n, 默认y): ").strip().lower()
            if response == 'n':
                print("流程已中断")
                return
    
    # 完成
    print("\n" + "=" * 80)
    print("✅ 动力学参数辨识流程完成!")
    print("=" * 80)
    
    print("\n生成的结果:")
    print("  1. 预处理数据: processed_data/")
    print("  2. 估计参数: estimation_results/estimated_parameters.npz")
    print("  3. 验证结果: validation_results/")
    
    print("\n下一步建议:")
    print("  - 查看 validation_results/validation_report.txt 了解验证性能")
    print("  - 如果性能不满意，调整滤波器参数或收集更多数据")
    print("  - 使用估计的参数进行重力补偿或前馈控制")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
