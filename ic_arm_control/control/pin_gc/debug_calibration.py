#!/usr/bin/env python3
"""
调试参数标定系统
"""

import numpy as np
import pinocchio as pin
from parameter_calibration import DynamicsParameterCalibrator

def debug_parameter_calibration():
    """调试参数标定系统"""
    print("=== 调试参数标定系统 ===")

    # 创建标定器
    urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/urdfs/ic_arm_8_dof/urdf/ic_arm_8_dof.urdf"
    calibrator = DynamicsParameterCalibrator(urdf_path)

    print(f"关节数量: {calibrator.original_model.njoints}")
    print(f"参数名称数量: {len(calibrator.parameter_names)}")
    print(f"参数名称: {calibrator.parameter_names}")

    # 测试参数提取
    initial_params = calibrator._extract_parameters_from_model(calibrator.original_model)
    print(f"提取的参数数量: {len(initial_params)}")

    # 测试边界设置
    bounds = calibrator._setup_parameter_bounds()
    print(f"边界数量: {len(bounds[0])} (下界), {len(bounds[1])} (上界)")

    # 检查初始参数是否在边界内
    print("\n=== 参数边界检查 ===")
    out_of_bounds = False
    for i, (param, lower, upper) in enumerate(zip(initial_params, bounds[0], bounds[1])):
        if param < lower or param > upper:
            print(f"参数 {i} ({calibrator.parameter_names[i]}): {param} 超出边界 [{lower}, {upper}]")
            out_of_bounds = True

    if not out_of_bounds:
        print("所有参数都在边界内")

    # 详细检查每个参数类型
    param_types = {}
    for param_name in calibrator.parameter_names:
        param_type, joint_idx, *extra = calibrator.param_indices[param_name]
        if param_type not in param_types:
            param_types[param_type] = 0
        param_types[param_type] += 1

    print(f"参数类型统计: {param_types}")

    # 检查每个关节数量
    joint_params = {}
    for param_name in calibrator.parameter_names:
        param_type, joint_idx, *extra = calibrator.param_indices[param_name]
        if joint_idx not in joint_params:
            joint_params[joint_idx] = []
        joint_params[joint_idx].append(param_name)

    print(f"每个关节的参数数量:")
    for joint_idx, params in joint_params.items():
        print(f"  关节 {joint_idx}: {len(params)} 个参数")
        print(f"    参数: {params}")

    # 检查边界和参数是否匹配
    if len(initial_params) == len(bounds[0]) and len(initial_params) == len(bounds[1]):
        print("✓ 参数维度匹配")

        # 测试添加一些标定数据
        print("\n=== 测试标定数据 ===")

        # 创建测试数据
        q_test = np.zeros(calibrator.original_model.nv)
        v_test = np.zeros(calibrator.original_model.nv)
        a_test = np.zeros(calibrator.original_model.nv)

        # 计算理想力矩
        tau_test = pin.rnea(calibrator.original_model, calibrator.original_model.createData(),
                          q_test, v_test, a_test)

        # 添加标定数据
        calibrator.add_calibration_data(q_test, v_test, a_test, tau_test)

        print(f"添加了 {len(calibrator.calibration_data)} 组标定数据")

        # 测试标定
        try:
            result = calibrator.calibrate_parameters()
            print("✓ 标定测试成功")
        except Exception as e:
            print(f"✗ 标定测试失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("✗ 参数维度不匹配")
        print(f"  初始参数: {len(initial_params)}")
        print(f"  下界: {len(bounds[0])}")
        print(f"  上界: {len(bounds[1])}")

if __name__ == "__main__":
    debug_parameter_calibration()