#!/usr/bin/env python3
"""
IC ARM 动力学参数标定模块
基于Pinocchio的动力学参数识别和优化
"""

import numpy as np
import pinocchio as pin
from typing import List, Dict, Tuple, Optional, Any
from scipy.optimize import least_squares, minimize
from scipy.linalg import lstsq
import json
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pinocchio_gravity_compensation import PinocchioGravityCompensation

@dataclass
class CalibrationData:
    """标定数据结构"""
    q: np.ndarray      # 关节位置
    v: np.ndarray      # 关节速度
    a: np.ndarray      # 关节加速度
    tau: np.ndarray    # 测量力矩
    timestamp: float   # 时间戳

@dataclass
class InertialParameters:
    """惯性参数结构"""
    mass: float        # 质量
    com: np.ndarray    # 质心位置 [x, y, z]
    inertia: np.ndarray # 惯性张量 3x3

class DynamicsParameterCalibrator:
    """动力学参数标定器"""

    def __init__(self, urdf_path: str):
        """
        初始化标定器

        Args:
            urdf_path: URDF文件路径
        """
        self.gc = PinocchioGravityCompensation(urdf_path)
        self.original_model = self.gc.model.copy()

        # 标定数据
        self.calibration_data: List[CalibrationData] = []
        self.calibrated_model = None

        # 参数边界和约束
        self.param_bounds = {}
        self.parameter_names = []

        # 标定配置
        self.calibration_config = {
            'identify_mass': True,
            'identify_com': True,
            'identify_inertia': True,
            'mass_bounds': (0.01, 10.0),      # 质量范围 kg
            'com_bounds': (-0.5, 0.5),        # 质心范围 m
            'inertia_bounds': (0.00001, 0.01), # 惯性范围 kg⋅m² (只允许正值)
            'regularization': 1e-6,           # 正则化系数
            'max_iterations': 100,            # 最大迭代次数
            'tolerance': 1e-6                 # 收敛容差
        }

        self._setup_parameter_identification()

    def _setup_parameter_identification(self):
        """设置参数识别结构"""
        # 获取所有需要识别的参数
        self.parameter_names = []
        self.param_indices = {}

        for i in range(1, self.original_model.njoints):
            joint_name = self.original_model.names[i]
            inertia = self.original_model.inertias[i]

            # 质量参数
            if self.calibration_config['identify_mass']:
                mass_name = f"{joint_name}_mass"
                self.parameter_names.append(mass_name)
                self.param_indices[mass_name] = ('mass', i)

            # 质心参数
            if self.calibration_config['identify_com']:
                for axis in ['x', 'y', 'z']:
                    com_name = f"{joint_name}_com_{axis}"
                    self.parameter_names.append(com_name)
                    self.param_indices[com_name] = ('com', i, axis)

            # 惯性参数 (只识别对角元素以确保正定性)
            if self.calibration_config['identify_inertia']:
                inertia_names = [
                    f"{joint_name}_Ixx", f"{joint_name}_Iyy", f"{joint_name}_Izz"
                ]
                for inertia_name in inertia_names:
                    self.parameter_names.append(inertia_name)
                    component = inertia_name.split('_')[-1]
                    self.param_indices[inertia_name] = ('inertia', i, component)

        print(f"设置参数识别完成，共 {len(self.parameter_names)} 个参数")
        print(f"参数列表: {self.parameter_names}")

    def add_calibration_data(self, q: np.ndarray, v: np.ndarray, a: np.ndarray, tau: np.ndarray):
        """
        添加标定数据

        Args:
            q: 关节位置 (rad)
            v: 关节速度 (rad/s)
            a: 关节加速度 (rad/s²)
            tau: 测量力矩 (N⋅m)
        """
        data = CalibrationData(
            q=q.copy(),
            v=v.copy(),
            a=a.copy(),
            tau=tau.copy(),
            timestamp=time.time()
        )
        self.calibration_data.append(data)
        print(f"添加标定数据，当前共 {len(self.calibration_data)} 组")

    def load_calibration_data_from_file(self, filename: str):
        """
        从文件加载标定数据

        Args:
            filename: 数据文件名
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            self.calibration_data.clear()
            for sample in data['samples']:
                q = np.array(sample['q'])
                v = np.array(sample['v'])
                a = np.array(sample['a'])
                tau = np.array(sample['tau'])
                timestamp = sample['timestamp']

                calib_data = CalibrationData(q, v, a, tau, timestamp)
                self.calibration_data.append(calib_data)

            print(f"从 {filename} 加载了 {len(self.calibration_data)} 组标定数据")
        except Exception as e:
            print(f"加载标定数据失败: {e}")

    def save_calibration_data(self, filename: str):
        """
        保存标定数据到文件

        Args:
            filename: 文件名
        """
        data = {
            'samples': [],
            'config': self.calibration_config
        }

        for sample in self.calibration_data:
            data['samples'].append({
                'q': sample.q.tolist(),
                'v': sample.v.tolist(),
                'a': sample.a.tolist(),
                'tau': sample.tau.tolist(),
                'timestamp': sample.timestamp
            })

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"标定数据已保存到 {filename}")

    def _extract_parameters_from_model(self, model: pin.Model) -> np.ndarray:
        """从模型提取参数向量"""
        params = []

        for param_name in self.parameter_names:
            param_type, joint_idx, *extra = self.param_indices[param_name]
            inertia = model.inertias[joint_idx]

            if param_type == 'mass':
                params.append(inertia.mass)
            elif param_type == 'com':
                axis = extra[0]
                axis_idx = ['x', 'y', 'z'].index(axis)
                params.append(inertia.lever[axis_idx])
            elif param_type == 'inertia':
                component = extra[0]
                if component == 'Ixx':
                    params.append(inertia.inertia[0, 0])
                elif component == 'Iyy':
                    params.append(inertia.inertia[1, 1])
                elif component == 'Izz':
                    params.append(inertia.inertia[2, 2])

        return np.array(params)

    def _update_model_with_parameters(self, model: pin.Model, params: np.ndarray):
        """用参数向量更新模型"""
        param_idx = 0

        for joint_idx in range(1, model.njoints):
            inertia = model.inertias[joint_idx]
            new_mass = inertia.mass
            new_com = inertia.lever.copy()
            new_inertia = inertia.inertia.copy()

            # 更新质量
            if self.calibration_config['identify_mass']:
                new_mass = params[param_idx]
                param_idx += 1

            # 更新质心
            if self.calibration_config['identify_com']:
                for axis_idx in range(3):
                    new_com[axis_idx] = params[param_idx]
                    param_idx += 1

            # 更新惯性张量
            if self.calibration_config['identify_inertia']:
                # 读取3个对角参数
                Ixx, Iyy, Izz = params[param_idx:param_idx+3]
                param_idx += 3

                # 构建对角惯性张量（确保正定性）
                new_inertia = np.array([
                    [Ixx, 0.0, 0.0],
                    [0.0, Iyy, 0.0],
                    [0.0, 0.0, Izz]
                ])

            # 更新模型惯性参数
            model.inertias[joint_idx] = pin.Inertia(new_mass, new_com, new_inertia)

    def _compute_residuals(self, params: np.ndarray) -> np.ndarray:
        """计算预测误差（残差）"""
        # 创建临时模型
        temp_model = self.original_model.copy()
        self._update_model_with_parameters(temp_model, params)

        residuals = []

        for data in self.calibration_data:
            # 计算预测力矩
            tau_pred = pin.rnea(temp_model, temp_model.createData(), data.q, data.v, data.a)

            # 计算残差
            residual = tau_pred - data.tau
            residuals.extend(residual)

        return np.array(residuals)

    def _parameter_constraints(self, params: np.ndarray) -> np.ndarray:
        """参数约束函数"""
        constraints = []

        param_idx = 0
        for param_name in self.parameter_names:
            param_type, joint_idx, *extra = self.param_indices[param_name]

            if param_type == 'mass':
                # 质量必须为正
                constraints.append(params[param_idx])
                param_idx += 1
            elif param_type == 'com':
                # 质心位置约束
                param_idx += 3
            elif param_type == 'inertia':
                # 惯性张量正定性约束 (简化为对角元素为正)
                Ixx, Iyy, Izz = params[param_idx:param_idx+3]
                constraints.extend([Ixx, Iyy, Izz])
                param_idx += 6

        return np.array(constraints)

    def calibrate_parameters(self) -> Dict[str, Any]:
        """
        执行参数标定

        Returns:
            calibration_result: 标定结果
        """
        if len(self.calibration_data) == 0:
            raise ValueError("没有标定数据，请先添加数据")

        print(f"开始标定动力学参数...")
        print(f"使用 {len(self.calibration_data)} 组标定数据")
        print(f"识别 {len(self.parameter_names)} 个参数")

        # 初始参数
        initial_params = self._extract_parameters_from_model(self.original_model)
        print(f"初始参数: {initial_params}")

        # 设置参数边界
        bounds = self._setup_parameter_bounds()

        # 检查边界和初始参数的维度
        print(f"初始参数维度: {len(initial_params)}")
        print(f"下界维度: {len(bounds[0])}")
        print(f"上界维度: {len(bounds[1])}")

        # 执行优化
        start_time = time.time()

        result = least_squares(
            self._compute_residuals,
            initial_params,
            bounds=bounds,
            max_nfev=self.calibration_config['max_iterations'],
            ftol=self.calibration_config['tolerance'],
            xtol=self.calibration_config['tolerance'],
            gtol=self.calibration_config['tolerance']
        )

        optimization_time = time.time() - start_time

        # 更新模型
        self.calibrated_model = self.original_model.copy()
        self._update_model_with_parameters(self.calibrated_model, result.x)

        # 计算标定误差
        final_residuals = self._compute_residuals(result.x)
        rmse = np.sqrt(np.mean(final_residuals**2))
        max_error = np.max(np.abs(final_residuals))

        # 生成结果报告
        calibration_result = {
            'success': result.success,
            'optimized_parameters': result.x,
            'initial_parameters': initial_params,
            'parameter_names': self.parameter_names,
            'optimization_time': optimization_time,
            'iterations': result.nfev,
            'rmse': rmse,
            'max_error': max_error,
            'message': result.message,
            'num_samples': len(self.calibration_data)
        }

        print(f"标定完成!")
        print(f"优化时间: {optimization_time:.2f}s")
        print(f"迭代次数: {result.nfev}")
        print(f"RMSE: {rmse:.6f} N⋅m")
        print(f"最大误差: {max_error:.6f} N⋅m")
        print(f"状态: {'成功' if result.success else '失败'}")

        return calibration_result

    def _setup_parameter_bounds(self):
        """设置参数边界"""
        lower_bounds = []
        upper_bounds = []

        for param_name in self.parameter_names:
            param_type, joint_idx, *extra = self.param_indices[param_name]

            if param_type == 'mass':
                lower_bounds.append(self.calibration_config['mass_bounds'][0])
                upper_bounds.append(self.calibration_config['mass_bounds'][1])
            elif param_type == 'com':
                lower_bounds.append(self.calibration_config['com_bounds'][0])
                upper_bounds.append(self.calibration_config['com_bounds'][1])
            elif param_type == 'inertia':
                lower_bounds.append(self.calibration_config['inertia_bounds'][0])
                upper_bounds.append(self.calibration_config['inertia_bounds'][1])

        return (lower_bounds, upper_bounds)

    def validate_calibration(self, test_data: List[CalibrationData] = None) -> Dict[str, float]:
        """
        验证标定结果

        Args:
            test_data: 测试数据，如果为None则使用标定数据

        Returns:
            validation_metrics: 验证指标
        """
        if self.calibrated_model is None:
            raise ValueError("请先执行标定")

        if test_data is None:
            test_data = self.calibration_data

        print(f"验证标定结果，使用 {len(test_data)} 组数据...")

        errors = []
        for data in test_data:
            # 使用标定模型预测
            tau_pred = pin.rnea(self.calibrated_model, self.calibrated_model.createData(),
                              data.q, data.v, data.a)

            # 使用原始模型预测
            tau_orig = pin.rnea(self.original_model, self.original_model.createData(),
                               data.q, data.v, data.a)

            # 计算误差
            error_calibrated = np.linalg.norm(tau_pred - data.tau)
            error_original = np.linalg.norm(tau_orig - data.tau)

            errors.append({
                'calibrated': error_calibrated,
                'original': error_original,
                'improvement': error_original - error_calibrated
            })

        # 计算统计指标
        calibrated_errors = [e['calibrated'] for e in errors]
        original_errors = [e['original'] for e in errors]
        improvements = [e['improvement'] for e in errors]

        validation_metrics = {
            'calibrated_rmse': np.sqrt(np.mean(np.array(calibrated_errors)**2)),
            'original_rmse': np.sqrt(np.mean(np.array(original_errors)**2)),
            'mean_improvement': np.mean(improvements),
            'std_improvement': np.std(improvements),
            'improvement_percentage': (np.mean(original_errors) - np.mean(calibrated_errors)) / np.mean(original_errors) * 100,
            'success_rate': sum(1 for imp in improvements if imp > 0) / len(improvements) * 100
        }

        print(f"验证结果:")
        print(f"  标定前RMSE: {validation_metrics['original_rmse']:.6f} N⋅m")
        print(f"  标定后RMSE: {validation_metrics['calibrated_rmse']:.6f} N⋅m")
        print(f"  改进百分比: {validation_metrics['improvement_percentage']:.2f}%")
        print(f"  成功率: {validation_metrics['success_rate']:.2f}%")

        return validation_metrics

    def save_calibration_results(self, filename: str, calibration_result: Dict[str, Any]):
        """
        保存标定结果

        Args:
            filename: 文件名
            calibration_result: 标定结果
        """
        results = {
            'calibration_result': calibration_result,
            'parameter_mapping': self.param_indices,
            'config': self.calibration_config,
            'timestamp': time.time()
        }

        # 添加标定后的参数值
        parameter_values = {}
        for i, param_name in enumerate(self.parameter_names):
            parameter_values[param_name] = float(calibration_result['optimized_parameters'][i])
        results['parameter_values'] = parameter_values

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"标定结果已保存到 {filename}")

    def load_calibration_results(self, filename: str):
        """
        加载标定结果

        Args:
            filename: 文件名
        """
        try:
            with open(filename, 'r') as f:
                results = json.load(f)

            # 恢复参数
            optimized_params = np.array([results['parameter_values'][name] for name in self.parameter_names])

            # 更新模型
            self.calibrated_model = self.original_model.copy()
            self._update_model_with_parameters(self.calibrated_model, optimized_params)

            print(f"标定结果已从 {filename} 加载")
            print(f"模型已更新")
        except Exception as e:
            print(f"加载标定结果失败: {e}")

    def print_parameter_comparison(self):
        """打印参数对比"""
        if self.calibrated_model is None:
            print("没有标定结果可供对比")
            return

        print("=== 动力学参数对比 ===")
        print(f"{'参数':<15} {'原始值':<12} {'标定值':<12} {'变化':<12}")
        print("-" * 55)

        initial_params = self._extract_parameters_from_model(self.original_model)
        calibrated_params = self._extract_parameters_from_model(self.calibrated_model)

        for i, param_name in enumerate(self.parameter_names):
            initial_val = initial_params[i]
            calibrated_val = calibrated_params[i]
            change = calibrated_val - initial_val
            change_pct = (change / initial_val) * 100 if initial_val != 0 else 0

            print(f"{param_name:<15} {initial_val:<12.6f} {calibrated_val:<12.6f} {change_pct:+.2f}%")

        print("====================")

    def create_excitation_trajectories(self, duration: float = 30.0, sampling_rate: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        创建激励轨迹用于参数标定

        Args:
            duration: 持续时间 (s)
            sampling_rate: 采样率 (Hz)

        Returns:
            q_trajectory: 位置轨迹
            v_trajectory: 速度轨迹
            a_trajectory: 加速度轨迹
        """
        n_samples = int(duration * sampling_rate)
        t = np.linspace(0, duration, n_samples)

        # 为每个关节设计激励轨迹
        q_traj = np.zeros((n_samples, self.original_model.nv))
        v_traj = np.zeros((n_samples, self.original_model.nv))
        a_traj = np.zeros((n_samples, self.original_model.nv))

        for joint_idx in range(self.original_model.nv):
            joint_name = self.original_model.names[joint_idx + 1]

            # 获取关节限制
            lower_limit = self.original_model.lowerPositionLimit[joint_idx]
            upper_limit = self.original_model.upperPositionLimit[joint_idx]
            vel_limit = self.original_model.velocityLimit[joint_idx]

            # 设计多频率正弦激励
            center = (lower_limit + upper_limit) / 2
            amplitude = (upper_limit - lower_limit) * 0.3  # 使用30%的范围

            # 多频率组合
            frequencies = [0.1, 0.3, 0.5, 0.8, 1.2]  # Hz
            q_joint = center * np.ones(n_samples)
            v_joint = np.zeros(n_samples)
            a_joint = np.zeros(n_samples)

            for freq in frequencies:
                phase = np.random.uniform(0, 2*np.pi)
                amp = amplitude / len(frequencies)

                q_joint += amp * np.sin(2 * np.pi * freq * t + phase)
                v_joint += amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t + phase)
                a_joint -= amp * (2 * np.pi * freq)**2 * np.sin(2 * np.pi * freq * t + phase)

            # 确保在限制范围内
            q_joint = np.clip(q_joint, lower_limit, upper_limit)
            v_joint = np.clip(v_joint, -vel_limit, vel_limit)

            q_traj[:, joint_idx] = q_joint
            v_traj[:, joint_idx] = v_joint
            a_traj[:, joint_idx] = a_joint

        return q_traj, v_traj, a_traj


def create_dynamics_calibrator(urdf_path: str) -> DynamicsParameterCalibrator:
    """
    创建动力学参数标定器

    Args:
        urdf_path: URDF文件路径

    Returns:
        calibrator: 标定器实例
    """
    return DynamicsParameterCalibrator(urdf_path)


if __name__ == "__main__":
    # 测试代码
    urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/urdfs/ic_arm_8_dof/urdf/ic_arm_8_dof.urdf"

    try:
        # 创建标定器
        calibrator = create_dynamics_calibrator(urdf_path)

        # 创建激励轨迹
        print("创建激励轨迹...")
        q_traj, v_traj, a_traj = calibrator.create_excitation_trajectories(duration=10.0, sampling_rate=50)

        # 模拟测量数据（添加噪声）
        print("模拟测量数据...")
        for i in range(len(q_traj)):
            # 使用原始模型计算"真实"力矩
            tau_true = pin.rnea(calibrator.original_model, calibrator.original_model.createData(),
                               q_traj[i], v_traj[i], a_traj[i])

            # 添加测量噪声
            noise = np.random.normal(0, 0.01, len(tau_true))
            tau_measured = tau_true + noise

            # 添加到标定数据
            if i % 5 == 0:  # 每5个采样取1个，避免数据过多
                calibrator.add_calibration_data(q_traj[i], v_traj[i], a_traj[i], tau_measured)

        print(f"生成了 {len(calibrator.calibration_data)} 组标定数据")

        # 保存数据
        calibrator.save_calibration_data("calibration_data.json")

        # 执行标定
        print("开始参数标定...")
        calibration_result = calibrator.calibrate_parameters()

        # 验证结果
        validation_metrics = calibrator.validate_calibration()

        # 打印参数对比
        calibrator.print_parameter_comparison()

        # 保存结果
        calibrator.save_calibration_results("calibration_results.json", calibration_result)

        print("标定测试完成!")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()