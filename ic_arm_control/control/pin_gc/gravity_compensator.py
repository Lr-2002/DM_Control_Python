#!/usr/bin/env python3
"""
IC ARM 重力补偿器
基于Pinocchio实现的重力补偿功能
"""

import numpy as np
from typing import Optional, Dict, Any
from pinocchio_gravity_compensation import PinocchioGravityCompensation
import time

class ICArmGravityCompensator:
    """IC ARM 重力补偿器"""

    def __init__(self, urdf_path: str):
        """
        初始化重力补偿器

        Args:
            urdf_path: URDF文件路径
        """
        self.gc = PinocchioGravityCompensation(urdf_path)
        self.last_q = None
        self.last_v = None
        self.last_tau_gravity = None
        self.last_timestamp = None

        # 补偿参数
        self.enable_compensation = True
        self.compensation_gain = 1.0  # 补偿增益
        self.velocity_damping = 0.1  # 速度阻尼系数
        self.friction_compensation = 0.0  # 摩擦补偿系数

        # 性能统计
        self.compensation_count = 0
        self.total_computation_time = 0.0
        self.max_computation_time = 0.0

        # 初始化重力向量
        self.gc.set_gravity_vector(np.array([0.0, 0.0, -9.81]))

    def enable(self):
        """启用重力补偿"""
        self.enable_compensation = True
        print("重力补偿已启用")

    def disable(self):
        """禁用重力补偿"""
        self.enable_compensation = False
        print("重力补偿已禁用")

    def set_compensation_gain(self, gain: float):
        """
        设置补偿增益

        Args:
            gain: 补偿增益 (0.0-2.0)
        """
        self.compensation_gain = np.clip(gain, 0.0, 2.0)
        print(f"补偿增益已设置为: {self.compensation_gain}")

    def set_velocity_damping(self, damping: float):
        """
        设置速度阻尼系数

        Args:
            damping: 阻尼系数 (0.0-1.0)
        """
        self.velocity_damping = np.clip(damping, 0.0, 1.0)
        print(f"速度阻尼系数已设置为: {self.velocity_damping}")

    def set_friction_compensation(self, friction: float):
        """
        设置摩擦补偿系数

        Args:
            friction: 摩擦补偿系数 (0.0-1.0)
        """
        self.friction_compensation = np.clip(friction, 0.0, 1.0)
        print(f"摩擦补偿系数已设置为: {self.friction_compensation}")

    def compute_compensation(self, q: np.ndarray, v: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算重力补偿力矩

        Args:
            q: 关节位置向量 (rad)
            v: 关节速度向量 (rad/s)，可选

        Returns:
            tau_comp: 补偿力矩 (N⋅m)
        """
        if not self.enable_compensation:
            return np.zeros_like(q)

        start_time = time.time()

        try:
            # 计算重力力矩
            tau_gravity = self.gc.compute_gravity_torques(q)

            # 应用补偿增益
            tau_comp = tau_gravity * self.compensation_gain

            # 添加速度阻尼
            if v is not None:
                tau_comp -= self.velocity_damping * v

            # 添加摩擦补偿 (简化模型)
            if v is not None and self.friction_compensation > 0:
                friction_torque = self.friction_compensation * np.tanh(v * 10.0) * 0.1
                tau_comp -= friction_torque

            # 更新统计信息
            computation_time = time.time() - start_time
            self.compensation_count += 1
            self.total_computation_time += computation_time
            self.max_computation_time = max(self.max_computation_time, computation_time)

            # 保存最后的状态
            self.last_q = q.copy()
            if v is not None:
                self.last_v = v.copy()
            self.last_tau_gravity = tau_comp.copy()
            self.last_timestamp = time.time()

            return tau_comp

        except Exception as e:
            print(f"计算重力补偿时出错: {e}")
            return np.zeros_like(q)

    def get_full_dynamics(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> Dict[str, np.ndarray]:
        """
        获取完整的动力学信息

        Args:
            q: 关节位置向量 (rad)
            v: 关节速度向量 (rad/s)
            a: 关节加速度向量 (rad/s²)

        Returns:
            dynamics: 动力学信息字典
                - gravity: 重力力矩
                - coriolis: 科里奥利力和离心力力矩
                - total: 总力矩
                - mass_matrix: 质量矩阵
        """
        try:
            gravity = self.gc.compute_gravity_torques(q)
            coriolis = self.gc.compute_coriolis_centrifugal(q, v)
            total = self.gc.compute_rnea(q, v, a)
            mass_matrix = self.gc.get_mass_matrix(q)

            return {
                'gravity': gravity,
                'coriolis': coriolis,
                'total': total,
                'mass_matrix': mass_matrix
            }
        except Exception as e:
            print(f"计算完整动力学时出错: {e}")
            return {
                'gravity': np.zeros_like(q),
                'coriolis': np.zeros_like(q),
                'total': np.zeros_like(q),
                'mass_matrix': np.eye(len(q))
            }

    def get_joint_positions(self, q: np.ndarray) -> Dict[str, np.ndarray]:
        """
        获取关节位置信息

        Args:
            q: 关节位置向量 (rad)

        Returns:
            positions: 关节位置字典
        """
        try:
            positions = self.gc.get_joint_positions(q)
            return positions
        except Exception as e:
            print(f"获取关节位置时出错: {e}")
            return {}

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息

        Returns:
            stats: 统计信息字典
        """
        avg_computation_time = 0.0
        if self.compensation_count > 0:
            avg_computation_time = self.total_computation_time / self.compensation_count

        return {
            'compensation_count': self.compensation_count,
            'total_computation_time': self.total_computation_time,
            'avg_computation_time': avg_computation_time,
            'max_computation_time': self.max_computation_time,
            'enable_compensation': self.enable_compensation,
            'compensation_gain': self.compensation_gain,
            'velocity_damping': self.velocity_damping,
            'friction_compensation': self.friction_compensation,
            'last_timestamp': self.last_timestamp
        }

    def print_performance_stats(self):
        """打印性能统计信息"""
        stats = self.get_performance_stats()
        print("=== 重力补偿性能统计 ===")
        print(f"补偿次数: {stats['compensation_count']}")
        print(f"总计算时间: {stats['total_computation_time']:.4f}s")
        print(f"平均计算时间: {stats['avg_computation_time']:.4f}s")
        print(f"最大计算时间: {stats['max_computation_time']:.4f}s")
        print(f"补偿状态: {'启用' if stats['enable_compensation'] else '禁用'}")
        print(f"补偿增益: {stats['compensation_gain']:.2f}")
        print(f"速度阻尼: {stats['velocity_damping']:.2f}")
        print(f"摩擦补偿: {stats['friction_compensation']:.2f}")
        if stats['last_timestamp']:
            print(f"最后更新时间: {stats['last_timestamp']}")
        print("========================")

    def print_current_state(self):
        """打印当前状态"""
        print("=== 当前重力补偿状态 ===")
        if self.last_q is not None:
            print(f"关节位置: {self.last_q}")
        if self.last_v is not None:
            print(f"关节速度: {self.last_v}")
        if self.last_tau_gravity is not None:
            print(f"补偿力矩: {self.last_tau_gravity}")
        print(f"补偿状态: {'启用' if self.enable_compensation else '禁用'}")
        print("========================")

    def reset_stats(self):
        """重置统计信息"""
        self.compensation_count = 0
        self.total_computation_time = 0.0
        self.max_computation_time = 0.0
        print("统计信息已重置")

    def save_calibration_data(self, filename: str):
        """
        保存标定数据

        Args:
            filename: 文件名
        """
        import json

        data = {
            'compensation_gain': self.compensation_gain,
            'velocity_damping': self.velocity_damping,
            'friction_compensation': self.friction_compensation,
            'performance_stats': self.get_performance_stats()
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"标定数据已保存到: {filename}")

    def load_calibration_data(self, filename: str):
        """
        加载标定数据

        Args:
            filename: 文件名
        """
        import json

        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            self.compensation_gain = data.get('compensation_gain', 1.0)
            self.velocity_damping = data.get('velocity_damping', 0.1)
            self.friction_compensation = data.get('friction_compensation', 0.0)

            print(f"标定数据已从 {filename} 加载")
            print(f"补偿增益: {self.compensation_gain}")
            print(f"速度阻尼: {self.velocity_damping}")
            print(f"摩擦补偿: {self.friction_compensation}")

        except Exception as e:
            print(f"加载标定数据失败: {e}")


def create_ic_arm_gravity_compensator(urdf_path: str) -> ICArmGravityCompensator:
    """
    创建IC ARM重力补偿器

    Args:
        urdf_path: URDF文件路径

    Returns:
        compensator: 重力补偿器实例
    """
    return ICArmGravityCompensator(urdf_path)


if __name__ == "__main__":
    # 测试代码
    urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/urdfs/ic_arm_8_dof/urdf/ic_arm_8_dof.urdf"

    try:
        # 创建重力补偿器
        compensator = create_ic_arm_gravity_compensator(urdf_path)

        # 测试基本功能
        q_test = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        v_test = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        # 计算补偿力矩
        tau_comp = compensator.compute_compensation(q_test, v_test)
        print(f"补偿力矩: {tau_comp}")

        # 获取完整动力学信息
        dynamics = compensator.get_full_dynamics(q_test, v_test, np.zeros(6))
        print(f"重力力矩: {dynamics['gravity']}")
        print(f"科里奥利力矩: {dynamics['coriolis']}")
        print(f"质量矩阵形状: {dynamics['mass_matrix'].shape}")

        # 测试性能
        print("\n性能测试...")
        for i in range(100):
            q_random = np.random.uniform(-1, 1, 6)
            v_random = np.random.uniform(-0.5, 0.5, 6)
            compensator.compute_compensation(q_random, v_random)

        compensator.print_performance_stats()

        # 保存标定数据
        compensator.save_calibration_data("gravity_calibration.json")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()