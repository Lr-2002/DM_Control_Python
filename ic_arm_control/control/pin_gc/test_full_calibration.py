#!/usr/bin/env python3
"""
完整的参数标定测试
"""

import numpy as np
import pinocchio as pin
from parameter_calibration import DynamicsParameterCalibrator

def test_full_calibration():
    """测试完整的参数标定系统"""
    print("=== 完整参数标定测试 ===")

    # 创建标定器
    urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/urdfs/ic_arm_8_dof/urdf/ic_arm_8_dof.urdf"
    calibrator = DynamicsParameterCalibrator(urdf_path)

    print(f"模型信息:")
    print(f"  关节数量: {calibrator.original_model.njoints}")
    print(f"  参数数量: {len(calibrator.parameter_names)}")

    # 生成丰富的测试数据
    print("\n=== 生成测试数据 ===")
    duration = 10.0
    sampling_rate = 20
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)

    # 生成激励轨迹
    q_data = np.zeros((n_samples, calibrator.original_model.nv))
    v_data = np.zeros((n_samples, calibrator.original_model.nv))
    a_data = np.zeros((n_samples, calibrator.original_model.nv))

    for joint_idx in range(calibrator.original_model.nv):
        # 为每个关节生成不同的激励
        lower_limit = calibrator.original_model.lowerPositionLimit[joint_idx]
        upper_limit = calibrator.original_model.upperPositionLimit[joint_idx]
        vel_limit = calibrator.original_model.velocityLimit[joint_idx]

        center = (lower_limit + upper_limit) / 2
        amplitude = (upper_limit - lower_limit) * 0.15

        # 多频率激励
        frequencies = [0.1, 0.3, 0.7]
        for freq in frequencies:
            phase = np.random.uniform(0, 2*np.pi)
            amp = amplitude / len(frequencies)

            q_data[:, joint_idx] += amp * np.sin(2 * np.pi * freq * t + phase)
            v_data[:, joint_idx] += amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t + phase)
            a_data[:, joint_idx] -= amp * (2 * np.pi * freq)**2 * np.sin(2 * np.pi * freq * t + phase)

    # 计算理想力矩并添加噪声
    print("计算理想力矩...")
    tau_ideal = np.zeros((n_samples, calibrator.original_model.nv))
    for i in range(n_samples):
        tau_ideal[i] = pin.rnea(calibrator.original_model, calibrator.original_model.createData(),
                              q_data[i], v_data[i], a_data[i])

    noise_level = 0.005  # 0.5%噪声
    tau_measured = tau_ideal + np.random.normal(0, noise_level, tau_ideal.shape)

    # 添加标定数据（每5个采样点添加一个，避免数据过多）
    print("添加标定数据...")
    for i in range(0, n_samples, 5):
        calibrator.add_calibration_data(q_data[i], v_data[i], a_data[i], tau_measured[i])

    print(f"添加了 {len(calibrator.calibration_data)} 组标定数据")

    # 执行标定
    print("\n=== 执行参数标定 ===")
    try:
        result = calibrator.calibrate_parameters()

        print(f"\n标定结果:")
        print(f"  成功: {result['success']}")
        print(f"  迭代次数: {result['iterations']}")
        print(f"  优化时间: {result['optimization_time']:.3f}s")
        print(f"  RMSE: {result['rmse']:.6f} N⋅m")
        print(f"  最大误差: {result['max_error']:.6f} N⋅m")

        # 比较参数变化
        print(f"\n=== 参数变化分析 ===")
        initial_params = result['initial_parameters']
        optimized_params = result['optimized_parameters']

        significant_changes = []
        for i, (init, opt, name) in enumerate(zip(initial_params, optimized_params, result['parameter_names'])):
            change_pct = abs(opt - init) / (abs(init) + 1e-10) * 100
            if change_pct > 1.0:  # 超过1%的变化
                significant_changes.append((name, init, opt, change_pct))

        print(f"显著变化的参数 ({len(significant_changes)} 个):")
        for name, init, opt, change_pct in significant_changes[:10]:  # 只显示前10个
            print(f"  {name}: {init:.6f} -> {opt:.6f} ({change_pct:.1f}%)")

        # 验证标定结果
        print(f"\n=== 验证标定结果 ===")
        validation = calibrator.validate_calibration()
        print(f"验证RMSE: {validation['calibrated_rmse']:.6f} N⋅m")
        print(f"原始RMSE: {validation['original_rmse']:.6f} N⋅m")
        print(f"改进: {validation['improvement_percentage']:.2f}%")

        # 保存结果
        print(f"\n=== 保存标定结果 ===")
        calibrator.save_calibration_results("full_calibration_results.json", result)
        print("标定结果已保存到 full_calibration_results.json")

        return True

    except Exception as e:
        print(f"标定失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_calibration()
    print(f"\n测试 {'成功' if success else '失败'}")