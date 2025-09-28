#!/usr/bin/env python3
"""
基于Pinocchio的重力补偿模块
使用URDF文件计算重力向量
"""

import numpy as np
import pinocchio as pin
from typing import List, Tuple, Optional, Dict, Any
import os
import sys

class PinocchioGravityCompensation:
    """基于Pinocchio的重力补偿计算器"""

    def __init__(self, urdf_path: str):
        """
        初始化Pinocchio模型

        Args:
            urdf_path: URDF文件路径
        """
        self.urdf_path = urdf_path
        self.model = None
        self.data = None
        self.joint_names = []
        self.joint_limits = []
        self.nq = 0  # 位置维度
        self.nv = 0  # 速度维度

        # 重力向量 (默认向下)
        self.gravity_vector = np.array([0.0, 0.0, -9.81])

        self._load_model()

    def _load_model(self):
        """加载URDF模型"""
        try:
            # 加载URDF模型
            self.model = pin.buildModelFromUrdf(self.urdf_path)
            self.data = self.model.createData()

            # 获取关节信息
            self.joint_names = [self.model.names[i] for i in range(1, self.model.njoints)]
            self.nq = self.model.nq
            self.nv = self.model.nv

            # 获取关节限制
            self.joint_limits = []
            for i in range(1, self.model.njoints):
                joint_idx = self.model.joints[i].idx_q
                if hasattr(self.model, 'upperPositionLimit') and hasattr(self.model, 'lowerPositionLimit'):
                    lower_limit = self.model.lowerPositionLimit[joint_idx]
                    upper_limit = self.model.upperPositionLimit[joint_idx]
                    self.joint_limits.append((lower_limit, upper_limit))
                else:
                    self.joint_limits.append((-np.pi, np.pi))

            print(f"成功加载URDF模型: {self.urdf_path}")
            print(f"关节数量: {len(self.joint_names)}")
            print(f"关节名称: {self.joint_names}")

        except Exception as e:
            print(f"加载URDF模型失败: {e}")
            raise

    def set_gravity_vector(self, gravity: np.ndarray):
        """
        设置重力向量

        Args:
            gravity: 重力向量 [gx, gy, gz] (m/s²)
        """
        self.gravity_vector = gravity.copy()
        # 更新模型中的重力设置 (使用Motion对象)
        linear = np.array([gravity[0], gravity[1], gravity[2]])
        angular = np.array([0.0, 0.0, 0.0])
        self.model.gravity = pin.Motion(linear, angular)
        print(f"重力向量已设置为: {gravity}")

    def compute_gravity_torques(self, q: np.ndarray) -> np.ndarray:
        """
        计算重力补偿力矩

        Args:
            q: 关节位置向量 (rad)

        Returns:
            tau: 重力补偿力矩 (N⋅m)
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        if len(q) != self.nv:
            raise ValueError(f"期望输入长度为 {self.nv}，实际为 {len(q)}")

        # 计算重力力矩
        tau = pin.computeGeneralizedGravity(self.model, self.data, q)

        return tau

    def compute_coriolis_centrifugal(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        计算科里奥利力和离心力

        Args:
            q: 关节位置向量 (rad)
            v: 关节速度向量 (rad/s)

        Returns:
            tau_c: 科里奥利力和离心力力矩 (N⋅m)
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        if len(q) != self.nv or len(v) != self.nv:
            raise ValueError(f"期望输入长度为 {self.nv}，实际位置为 {len(q)}，速度为 {len(v)}")

        # 计算科里奥利力和离心力
        tau_c = pin.nonLinearEffects(self.model, self.data, q, v) - pin.computeGeneralizedGravity(self.model, self.data, q)

        return tau_c

    def compute_rnea(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        计算完整的动力学方程 (RNEA算法)

        Args:
            q: 关节位置向量 (rad)
            v: 关节速度向量 (rad/s)
            a: 关节加速度向量 (rad/s²)

        Returns:
            tau: 总力矩 (N⋅m)
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        if len(q) != self.nv or len(v) != self.nv or len(a) != self.nv:
            raise ValueError(f"期望输入长度为 {self.nv}")

        # 使用递归牛顿-欧拉算法计算总力矩
        tau = pin.rnea(self.model, self.data, q, v, a)

        return tau

    def get_mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        获取质量矩阵

        Args:
            q: 关节位置向量 (rad)

        Returns:
            M: 质量矩阵 (n×n)
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        if len(q) != self.nv:
            raise ValueError(f"期望输入长度为 {self.nv}，实际为 {len(q)}")

        # 计算质量矩阵
        M = pin.crba(self.model, self.data, q)

        return M

    def forward_kinematics(self, q: np.ndarray, joint_name: Optional[str] = None) -> pin.SE3:
        """
        正向运动学计算

        Args:
            q: 关节位置向量 (rad)
            joint_name: 指定关节名称，如果为None则返回末端执行器位置

        Returns:
            transform: 指定关节的位姿变换矩阵
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        if len(q) != self.nv:
            raise ValueError(f"期望输入长度为 {self.nv}，实际为 {len(q)}")

        # 计算正向运动学
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        if joint_name is None:
            # 返回末端执行器位置
            frame_id = self.model.getFrameId(self.model.frames[-1].name)
        else:
            # 返回指定关节位置
            frame_id = self.model.getFrameId(joint_name)

        return self.data.oMf[frame_id]

    def get_joint_positions(self, q: np.ndarray) -> Dict[str, pin.SE3]:
        """
        获取所有关节的位置

        Args:
            q: 关节位置向量 (rad)

        Returns:
            positions: 关节位置字典 {joint_name: transform}
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        if len(q) != self.nv:
            raise ValueError(f"期望输入长度为 {self.nv}，实际为 {len(q)}")

        # 计算正向运动学
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        positions = {}
        for i in range(1, self.model.njoints):
            joint_name = self.model.names[i]
            frame_id = self.model.getFrameId(joint_name)
            positions[joint_name] = self.data.oMf[frame_id]

        return positions

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            info: 模型信息字典
        """
        return {
            'urdf_path': self.urdf_path,
            'n_joints': len(self.joint_names),
            'joint_names': self.joint_names,
            'joint_limits': self.joint_limits,
            'nq': self.nq,
            'nv': self.nv,
            'gravity_vector': self.gravity_vector,
            'has_rotor_inertia': hasattr(self.model, 'rotorInertia'),
            'has_armature': hasattr(self.model, 'armature')
        }

    def print_model_info(self):
        """打印模型信息"""
        info = self.get_model_info()
        print("=== 模型信息 ===")
        print(f"URDF文件: {info['urdf_path']}")
        print(f"关节数量: {info['n_joints']}")
        print(f"关节名称: {info['joint_names']}")
        print(f"位置维度: {info['nq']}")
        print(f"速度维度: {info['nv']}")
        print(f"重力向量: {info['gravity_vector']}")
        print("关节限制:")
        for i, (name, (lower, upper)) in enumerate(zip(info['joint_names'], info['joint_limits'])):
            print(f"  {name}: [{lower:.3f}, {upper:.3f}] rad")
        print("===============")


def create_default_gravity_compensation(urdf_path: str) -> PinocchioGravityCompensation:
    """
    创建默认的重力补偿器

    Args:
        urdf_path: URDF文件路径

    Returns:
        gc: 重力补偿器实例
    """
    return PinocchioGravityCompensation(urdf_path)


if __name__ == "__main__":
    # 测试代码
    urdf_path = "/Users/lr-2002/project/instantcreation/IC_arm_control/urdfs/ic_arm_8_dof/urdf/ic_arm_8_dof.urdf"

    try:
        # 创建重力补偿器
        gc = create_default_gravity_compensation(urdf_path)
        gc.print_model_info()

        # 测试重力力矩计算
        q_test = np.zeros(gc.nv)  # 零位置
        tau_gravity = gc.compute_gravity_torques(q_test)
        print(f"零位置重力力矩: {tau_gravity}")

        # 测试不同位置的重力力矩
        q_test2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        tau_gravity2 = gc.compute_gravity_torques(q_test2)
        print(f"测试位置重力力矩: {tau_gravity2}")

        # 测试质量矩阵
        M = gc.get_mass_matrix(q_test)
        print(f"质量矩阵形状: {M.shape}")

        # 测试正向运动学
        end_effector_pos = gc.forward_kinematics(q_test)
        print(f"末端执行器位置: {end_effector_pos.translation}")
        print(f"末端执行器旋转: {end_effector_pos.rotation}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()