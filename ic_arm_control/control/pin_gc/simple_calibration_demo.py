#!/usr/bin/env python3
"""
简化的动力学参数标定演示
"""

import numpy as np
import pinocchio as pin
import json
from pinocchio_gravity_compensation import PinocchioGravityCompensation

def simple_calibration_demo():
    """简化的标定演示"""

    print("=== 简化的动力学参数标定演示 ===")

    # 1. 创建重力补偿器
    urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/urdfs/ic_arm_8_dof/urdf/ic_arm_8_dof.urdf"
    gc = PinocchioGravityCompensation(urdf_path)

    print(f"URDF模型加载成功:")
    print(f"  关节数量: {gc.nv}")
    print(f"  关节名称: {gc.joint_names}")

    # 2. 显示原始动力学参数
    print("\n=== 原始动力学参数 ===")
    for i in range(1, gc.model.njoints):
        joint_name = gc.model.names[i]
        inertia = gc.model.inertias[i]
        print(f"{joint_name}:")
        print(f"  质量: {inertia.mass:.6f} kg")
        print(f"  质心: {inertia.lever} m")
        print(f"  惯性矩阵: \n{inertia.inertia}")
        print()

    # 3. 创建测试数据
    print("=== 创建测试数据 ===")

    # 生成激励轨迹
    duration = 5.0  # 5秒
    sampling_rate = 50  # 50Hz
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)

    # 为每个关节生成不同的激励
    q_test = np.zeros((n_samples, gc.nv))
    v_test = np.zeros((n_samples, gc.nv))
    a_test = np.zeros((n_samples, gc.nv))

    for joint_idx in range(gc.nv):
        # 获取关节限制
        lower_limit = gc.model.lowerPositionLimit[joint_idx]
        upper_limit = gc.model.upperPositionLimit[joint_idx]
        vel_limit = gc.model.velocityLimit[joint_idx]

        # 生成多频率激励
        center = (lower_limit + upper_limit) / 2
        amplitude = (upper_limit - lower_limit) * 0.2  # 20%的范围

        frequencies = [0.2, 0.5, 1.0]  # Hz
        for freq in frequencies:
            phase = np.random.uniform(0, 2*np.pi)
            amp = amplitude / len(frequencies)

            q_test[:, joint_idx] += amp * np.sin(2 * np.pi * freq * t + phase)
            v_test[:, joint_idx] += amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t + phase)
            a_test[:, joint_idx] -= amp * (2 * np.pi * freq)**2 * np.sin(2 * np.pi * freq * t + phase)

    # 计算理想力矩
    tau_ideal = np.zeros((n_samples, gc.nv))
    for i in range(n_samples):
        tau_ideal[i] = pin.rnea(gc.model, gc.data, q_test[i], v_test[i], a_test[i])

    # 添加噪声
    noise_level = 0.01  # 1%噪声
    tau_measured = tau_ideal + np.random.normal(0, noise_level, tau_ideal.shape)

    print(f"生成 {n_samples} 组测试数据")
    print(f"噪声水平: {noise_level*100}%")

    # 4. 参数标定 (手动调整演示)
    print("\n=== 手动参数标定演示 ===")

    # 创建一个可调整的模型副本
    test_model = gc.model.copy()

    # 调整某个关节的参数作为演示
    joint_to_modify = 3  # joint3
    original_inertia = test_model.inertias[joint_to_modify].copy()

    print(f"修改 {gc.model.names[joint_to_modify]} 的参数:")
    print(f"原始质量: {original_inertia.mass}")
    print(f"原始质心: {original_inertia.lever}")

    # 人为修改参数 (模拟标定过程)
    modified_inertia = pin.Inertia(
        mass=original_inertia.mass * 1.1,  # 增加10%质量
        lever=original_inertia.lever + np.array([0.01, 0.0, 0.0]),  # 调整质心
        inertia=original_inertia.inertia * 1.05  # 增加5%惯性
    )

    test_model.inertias[joint_to_modify] = modified_inertia

    print(f"修改后质量: {modified_inertia.mass}")
    print(f"修改后质心: {modified_inertia.lever}")

    # 5. 验证标定效果
    print("\n=== 标定效果验证 ===")

    # 计算修改后的预测力矩
    tau_predicted = np.zeros((n_samples, gc.nv))
    for i in range(n_samples):
        tau_predicted[i] = pin.rnea(test_model, test_model.createData(), q_test[i], v_test[i], a_test[i])

    # 计算误差
    errors_original = np.linalg.norm(tau_measured - tau_ideal, axis=1)
    errors_modified = np.linalg.norm(tau_measured - tau_predicted, axis=1)

    rmse_original = np.sqrt(np.mean(errors_original**2))
    rmse_modified = np.sqrt(np.mean(errors_modified**2))

    print(f"原始模型RMSE: {rmse_original:.6f} N⋅m")
    print(f"修改后模型RMSE: {rmse_modified:.6f} N⋅m")
    print(f"改进: {((rmse_original - rmse_modified) / rmse_original * 100):+.2f}%")

    # 6. 保存标定结果
    print("\n=== 保存标定结果 ===")

    calibration_results = {
        "original_parameters": {},
        "calibrated_parameters": {},
        "performance": {
            "original_rmse": rmse_original,
            "calibrated_rmse": rmse_modified,
            "improvement_percent": (rmse_original - rmse_modified) / rmse_original * 100
        },
        "test_data_info": {
            "duration": duration,
            "sampling_rate": sampling_rate,
            "n_samples": n_samples,
            "noise_level": noise_level
        }
    }

    # 保存原始和标定后的参数
    for i in range(1, gc.model.njoints):
        joint_name = gc.model.names[i]
        orig_inertia = gc.model.inertias[i]
        calib_inertia = test_model.inertias[i]

        calibration_results["original_parameters"][joint_name] = {
            "mass": float(orig_inertia.mass),
            "com": orig_inertia.lever.tolist(),
            "inertia": orig_inertia.inertia.tolist()
        }

        calibration_results["calibrated_parameters"][joint_name] = {
            "mass": float(calib_inertia.mass),
            "com": calib_inertia.lever.tolist(),
            "inertia": calib_inertia.inertia.tolist()
        }

    # 保存到文件
    with open("simple_calibration_results.json", "w") as f:
        json.dump(calibration_results, f, indent=2)

    print("标定结果已保存到 simple_calibration_results.json")

    # 7. 创建标定数据文件
    print("\n=== 创建标定数据文件 ===")

    calibration_data = {
        "samples": [],
        "config": {
            "duration": duration,
            "sampling_rate": sampling_rate,
            "noise_level": noise_level
        }
    }

    # 只保存部分数据点
    save_every = 5  # 每5个点保存1个
    for i in range(0, n_samples, save_every):
        calibration_data["samples"].append({
            "q": q_test[i].tolist(),
            "v": v_test[i].tolist(),
            "a": a_test[i].tolist(),
            "tau": tau_measured[i].tolist(),
            "timestamp": float(i / sampling_rate)
        })

    with open("calibration_data_sample.json", "w") as f:
        json.dump(calibration_data, f, indent=2)

    print(f"标定数据已保存到 calibration_data_sample.json (包含 {len(calibration_data['samples'])} 个样本)")

    return calibration_results

if __name__ == "__main__":
    results = simple_calibration_demo()