#!/usr/bin/env python3
"""
Unified Data Processing for Dynamic Identification
统一数据处理文件，整合日志转换、预处理和MAT格式转换功能

功能：
1. 日志数据转换为动力学格式
2. 数据预处理（滤波、加速度估计）
3. 转换为MAT格式供MATLAB使用

Author: Claude Code Assistant
"""

import numpy as np
import pandas as pd
import scipy.io
import os
from pathlib import Path
from scipy.signal import butter, filtfilt
from typing import List, Tuple, Optional, Dict, Any


class LogDataConverter:
    """日志数据转换器 - 对应 convert_log_data.py 的功能"""

    def __init__(self, dt: float = 0.0025):
        """
        初始化转换器

        Args:
            dt: 时间步长，默认 0.0025s (400Hz)
        """
        self.dt = dt

    def convert_log_dir_to_dynamics(self, log_dir: str, output_file: str) -> Optional[str]:
        """
        将单个日志目录转换为动力学格式

        Args:
            log_dir: 日志目录路径
            output_file: 输出CSV文件路径

        Returns:
            转换成功返回文件路径，失败返回None
        """
        print(f"转换日志目录: {log_dir}")

        # 检查必需文件
        motor_states_file = os.path.join(log_dir, "motor_states.csv")
        joint_commands_file = os.path.join(log_dir, "joint_commands.csv")

        if not os.path.exists(motor_states_file):
            print(f"  ❌ 缺失 motor_states.csv")
            return None

        if not os.path.exists(joint_commands_file):
            print(f"  ❌ 缺失 joint_commands.csv")
            return None

        try:
            # 读取数据
            motor_df = pd.read_csv(motor_states_file)
            command_df = pd.read_csv(joint_commands_file)

            print(f"  电机状态: {len(motor_df)} 数据点")
            print(f"  关节命令: {len(command_df)} 数据点")

            # 确保长度一致
            min_len = min(len(motor_df), len(command_df))
            motor_df = motor_df.iloc[:min_len]
            command_df = command_df.iloc[:min_len]

            # 创建动力学数据
            dynamics_data = []

            for i in range(len(motor_df)):
                row_data = {"time": i * self.dt}

                # 处理所有6个关节
                for joint_id in range(1, 7):
                    pos_col = f"position_motor_{joint_id}"
                    vel_col = f"velocity_motor_{joint_id}"
                    torque_col = f"torque_motor_{joint_id}"

                    if pos_col in motor_df.columns:
                        position = motor_df[pos_col].iloc[i]
                        velocity = motor_df[vel_col].iloc[i]
                        torque = motor_df[torque_col].iloc[i]

                        row_data[f"m{joint_id}_pos_actual"] = position
                        row_data[f"m{joint_id}_vel_actual"] = velocity
                        row_data[f"m{joint_id}_torque"] = torque

                dynamics_data.append(row_data)

            # 计算加速度
            for joint_id in range(1, 7):
                vel_col = f"m{joint_id}_vel_actual"
                if vel_col in dynamics_data[0]:
                    velocities = [row[vel_col] for row in dynamics_data]
                    accelerations = np.gradient(velocities, self.dt)

                    for j, acc in enumerate(accelerations):
                        dynamics_data[j][f"m{joint_id}_acc_actual"] = acc

            # 创建DataFrame并保存
            df = pd.DataFrame(dynamics_data)
            df = df.fillna(0)
            df.to_csv(output_file, index=False)

            print(f"  ✅ 保存: {output_file}")
            print(f"  形状: {df.shape}")

            return output_file

        except Exception as e:
            print(f"  ❌ 错误: {e}")
            return None


class DataPreprocessor:
    """数据预处理器 - 对应 step1_data_preprocessing.py 的功能"""

    def __init__(self, cutoff_freq: float = 10.0, fs: float = 400.0, filter_order: int = 4):
        """
        初始化滤波器参数

        Args:
            cutoff_freq: 截止频率 (Hz)
            fs: 采样频率 (Hz)
            filter_order: 滤波器阶数
        """
        self.cutoff_freq = cutoff_freq
        self.fs = fs
        self.filter_order = filter_order

        # 设计Butterworth低通滤波器
        nyquist = 0.5 * fs
        normal_cutoff = cutoff_freq / nyquist
        self.b, self.a = butter(filter_order, normal_cutoff, btype="low")

        print(f"滤波器参数:")
        print(f"  截止频率: {cutoff_freq} Hz")
        print(f"  采样频率: {fs} Hz")
        print(f"  滤波器阶数: {filter_order}")

    def zero_phase_filter(self, data: np.ndarray) -> np.ndarray:
        """
        零相位滤波 - 使用filtfilt (前向-后向滤波)

        Args:
            data: 原始数据 (1D array)

        Returns:
            滤波后的数据
        """
        if len(data) < 3 * self.filter_order:
            print(f"  警告: 数据长度 {len(data)} 太短，跳过滤波")
            return data

        # 零相位滤波 - 避免引入相位延迟
        filtered_data = filtfilt(self.b, self.a, data)
        return filtered_data

    def estimate_acceleration_central_diff(self, velocity: np.ndarray, time_array: np.ndarray) -> np.ndarray:
        """
        使用中心差分法估计加速度 (基于实际时间戳)

        Args:
            velocity: 速度数据 (1D array)
            time_array: 时间戳数组 (1D array)

        Returns:
            加速度估计值
        """
        n = len(velocity)
        acceleration = np.zeros(n)

        # 中心差分法 (更准确) - 使用实际时间间隔
        for i in range(1, n - 1):
            dt_forward = time_array[i + 1] - time_array[i]
            dt_backward = time_array[i] - time_array[i - 1]
            dt_total = time_array[i + 1] - time_array[i - 1]

            # 中心差分
            acceleration[i] = (velocity[i + 1] - velocity[i - 1]) / dt_total

        # 边界处理 - 使用前向/后向差分
        dt_0 = time_array[1] - time_array[0]
        acceleration[0] = (velocity[1] - velocity[0]) / dt_0

        dt_end = time_array[-1] - time_array[-2]
        acceleration[-1] = (velocity[-1] - velocity[-2]) / dt_end

        return acceleration

    def process_trajectory_data(
        self,
        q_raw: np.ndarray,
        dq_raw: np.ndarray,
        tau_raw: np.ndarray,
        time_array: np.ndarray,
        filter_position: bool = True,
        filter_velocity: bool = True,
        filter_torque: bool = True,
        filter_acceleration: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        处理轨迹数据 - 完整的滤波流程

        Args:
            q_raw: 原始位置数据 (n_samples, n_joints)
            dq_raw: 原始速度数据 (n_samples, n_joints)
            tau_raw: 原始力矩数据 (n_samples, n_joints)
            time_array: 时间戳数组 (n_samples,)
            filter_position: 是否滤波位置
            filter_velocity: 是否滤波速度
            filter_torque: 是否滤波力矩
            filter_acceleration: 是否滤波加速度

        Returns:
            q_filtered, dq_filtered, ddq_filtered, tau_filtered: 处理后的数据
        """
        n_samples, n_joints = q_raw.shape

        # 计算平均时间步长用于显示
        avg_dt = np.mean(np.diff(time_array))

        print(f"数据预处理:")
        print(f"  数据点数: {n_samples}")
        print(f"  关节数: {n_joints}")
        print(f"  平均时间步长: {avg_dt:.6f} s ({1 / avg_dt:.1f} Hz)")
        print(f"  时间范围: {time_array[0]:.3f} - {time_array[-1]:.3f} s")
        print(f"  总时长: {time_array[-1] - time_array[0]:.3f} s")

        # 1. 位置滤波
        if filter_position:
            print("  滤波位置数据...")
            q_filtered = np.zeros_like(q_raw)
            for j in range(n_joints):
                q_filtered[:, j] = self.zero_phase_filter(q_raw[:, j])
        else:
            print("  跳过位置滤波")
            q_filtered = q_raw.copy()

        # 2. 速度滤波
        if filter_velocity:
            print("  滤波速度数据...")
            dq_filtered = np.zeros_like(dq_raw)
            for j in range(n_joints):
                dq_filtered[:, j] = self.zero_phase_filter(dq_raw[:, j])
        else:
            print("  跳过速度滤波")
            dq_filtered = dq_raw.copy()

        # 3. 力矩滤波
        if filter_torque:
            print("  滤波力矩数据...")
            tau_filtered = np.zeros_like(tau_raw)
            for j in range(n_joints):
                tau_filtered[:, j] = self.zero_phase_filter(tau_raw[:, j])
        else:
            print("  跳过力矩滤波")
            tau_filtered = tau_raw.copy()

        # 4. 加速度估计
        print("  估计加速度 (中心差分法，基于时间戳)...")
        ddq_raw = np.zeros_like(dq_filtered)
        for j in range(n_joints):
            ddq_raw[:, j] = self.estimate_acceleration_central_diff(
                dq_filtered[:, j], time_array
            )

        # 5. 加速度滤波
        if filter_acceleration:
            print("  滤波加速度数据...")
            ddq_filtered = np.zeros_like(ddq_raw)
            for j in range(n_joints):
                ddq_filtered[:, j] = self.zero_phase_filter(ddq_raw[:, j])
        else:
            print("  跳过加速度滤波")
            ddq_filtered = ddq_raw.copy()

        print("  ✅ 数据预处理完成")

        return q_filtered, dq_filtered, ddq_filtered, tau_filtered


class MatConverter:
    """MAT格式转换器 - 对应 conver_data_to_mat.py 的功能"""

    @staticmethod
    def convert_processed_to_mat(csv_file_path: str, output_dir: Optional[str] = None) -> str:
        """
        将预处理后的CSV数据转换为MAT格式

        Args:
            csv_file_path: 预处理后的CSV文件路径
            output_dir: 输出目录，默认为CSV文件所在目录

        Returns:
            MAT文件路径
        """
        print(f"转换CSV到MAT: {Path(csv_file_path).name}")

        # 读取CSV数据
        df = pd.read_csv(csv_file_path)
        print(f"  数据形状: {df.shape}")

        # 提取时间数组
        t = df["time"].values

        # 提取关节数据 (6个关节)
        n_joints = 6
        n_samples = len(df)

        # 初始化矩阵
        q = np.zeros((n_samples, n_joints))
        dq = np.zeros((n_samples, n_joints))
        ddq = np.zeros((n_samples, n_joints))
        tau = np.zeros((n_samples, n_joints))

        # 填充数据
        for j in range(n_joints):
            joint_id = j + 1
            q[:, j] = df[f"q{joint_id}"].values
            dq[:, j] = df[f"dq{joint_id}"].values
            ddq[:, j] = df[f"ddq{joint_id}"].values
            tau[:, j] = df[f"tau{joint_id}"].values

        print(f"  时间范围: [{t[0]:.3f}, {t[-1]:.3f}] s")
        print(f"  关节数据形状: {q.shape}")

        # 生成输出文件名
        if output_dir is None:
            output_dir = Path(csv_file_path).parent

        csv_name = Path(csv_file_path).stem
        # 移除 "_filtered" 后缀
        if csv_name.endswith("_filtered"):
            csv_name = csv_name[:-9]

        mat_filename = f"{csv_name}.mat"
        mat_filepath = Path(output_dir) / mat_filename

        # 保存为MAT格式
        mat_data = {"t": t, "q": q, "dq": dq, "ddq": ddq, "tau": tau}

        scipy.io.savemat(mat_filepath, mat_data)

        print(f"  保存为: {mat_filepath}")
        print(f"  变量: t({t.shape}), q{q.shape}, dq{dq.shape}, ddq{ddq.shape}, tau{tau.shape}")

        return str(mat_filepath)


class UnifiedDataProcessor:
    """统一数据处理器 - 整合所有功能"""

    def __init__(self, base_dir: str = "/Users/lr-2002/project/instantcreation/IC_arm_control"):
        """
        初始化统一处理器

        Args:
            base_dir: 项目基础目录
        """
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "dyn_ana_for_mat"
        self.processed_dir = self.base_dir / "ic_arm_control/control/di_data_from_gh/processed_data_for_mat"
        self.mat_dir = self.base_dir / "ic_arm_control/control/di_data_from_gh/mat_data"

        # 创建组件
        self.log_converter = LogDataConverter(dt=0.0025)  # 400Hz
        self.preprocessor = DataPreprocessor(cutoff_freq=10.0, fs=400.0, filter_order=4)
        self.mat_converter = MatConverter()

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.mat_dir, exist_ok=True)

    def load_converted_data(self, csv_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        从转换后的CSV文件加载数据

        Args:
            csv_file: 转换后的CSV文件路径

        Returns:
            q, dq, tau, time_array: 位置、速度、力矩和时间数组
        """
        print(f"加载转换后的数据: {Path(csv_file).name}")

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"文件不存在: {csv_file}")

        # 读取数据
        df = pd.read_csv(csv_file)

        print(f"  数据点数: {len(df)}")
        print(f"  列数: {len(df.columns)}")

        # 提取数据 (前6个关节)
        n_joints = 6
        n_samples = len(df)

        q = np.zeros((n_samples, n_joints))
        dq = np.zeros((n_samples, n_joints))
        tau = np.zeros((n_samples, n_joints))

        for j in range(n_joints):
            motor_id = j + 1
            q[:, j] = df[f"m{motor_id}_pos_actual"].values
            dq[:, j] = df[f"m{motor_id}_vel_actual"].values
            tau[:, j] = df[f"m{motor_id}_torque"].values

        # 获取时间数组
        if "time" in df.columns:
            time_array = df["time"].values
        else:
            # 假设 400Hz
            time_array = np.arange(n_samples) * 0.0025

        avg_dt = np.mean(np.diff(time_array))
        print(f"  提取数据: {n_samples} 样本, {n_joints} 关节")
        print(f"  平均时间步长: {avg_dt:.6f} s ({1 / avg_dt:.1f} Hz)")

        return q, dq, tau, time_array

    def save_processed_data(
        self,
        output_file: str,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        tau: np.ndarray,
        time_array: np.ndarray
    ) -> None:
        """
        保存处理后的数据

        Args:
            output_file: 输出文件路径
            q, dq, ddq, tau: 处理后的数据
            time_array: 时间数组
        """
        n_samples, n_joints = q.shape

        # 构建DataFrame
        data_dict = {"time": time_array}

        for j in range(n_joints):
            joint_id = j + 1
            data_dict[f"q{joint_id}"] = q[:, j]
            data_dict[f"dq{joint_id}"] = dq[:, j]
            data_dict[f"ddq{joint_id}"] = ddq[:, j]
            data_dict[f"tau{joint_id}"] = tau[:, j]

        df = pd.DataFrame(data_dict)
        df.to_csv(output_file, index=False)

        print(f"保存处理后的数据: {output_file}")
        print(f"  形状: {df.shape}")
        print(f"  列: {list(df.columns)}")

    def process_log_directories(self, log_dirs: List[str]) -> List[str]:
        """
        处理多个日志目录

        Args:
            log_dirs: 日志目录列表

        Returns:
            转换后的CSV文件列表
        """
        print("=" * 80)
        print("步骤1: 转换日志目录为动力学格式")
        print("=" * 80)

        converted_files = []

        for log_dir in log_dirs:
            if os.path.exists(log_dir):
                output_file = self.output_dir / f"dynamics_{os.path.basename(log_dir)}.csv"
                result = self.log_converter.convert_log_dir_to_dynamics(log_dir, str(output_file))
                if result:
                    converted_files.append(result)
            else:
                print(f"❌ 目录不存在: {log_dir}")

        if converted_files:
            # 创建合并数据集
            print(f"\n合并 {len(converted_files)} 个文件...")
            all_data = []
            for file in converted_files:
                df = pd.read_csv(file)
                df["source"] = os.path.basename(file)
                all_data.append(df)

            merged_df = pd.concat(all_data, ignore_index=True)
            merged_df = merged_df.sort_values("time").reset_index(drop=True)

            merged_file = self.output_dir / "merged_log_data.csv"
            merged_df.to_csv(merged_file, index=False)

            print(f"合并文件: {merged_file}")
            print(f"总数据点: {len(merged_df)}")

            converted_files.append(str(merged_file))

        return converted_files

    def preprocess_data_files(self, csv_files: List[str]) -> List[str]:
        """
        预处理数据文件

        Args:
            csv_files: CSV文件列表

        Returns:
            处理后的CSV文件列表
        """
        print("\n" + "=" * 80)
        print("步骤2: 数据预处理 (滤波和加速度估计)")
        print("=" * 80)

        processed_files = []

        for i, csv_file in enumerate(csv_files):
            print(f"\n{'=' * 60}")
            print(f"处理文件 {i + 1}/{len(csv_files)}: {Path(csv_file).name}")
            print(f"{'=' * 60}")

            try:
                # 1. 加载转换后的数据
                q_raw, dq_raw, tau_raw, time_array = self.load_converted_data(csv_file)

                # 2. 数据预处理
                q_filtered, dq_filtered, ddq_filtered, tau_filtered = (
                    self.preprocessor.process_trajectory_data(
                        q_raw,
                        dq_raw,
                        tau_raw,
                        time_array,
                        filter_position=True,
                        filter_velocity=True,
                        filter_torque=True,
                        filter_acceleration=True,
                    )
                )

                # 3. 保存处理后的数据
                file_name = Path(csv_file).stem
                output_file = self.processed_dir / f"{file_name}_filtered.csv"
                self.save_processed_data(
                    str(output_file),
                    q_filtered,
                    dq_filtered,
                    ddq_filtered,
                    tau_filtered,
                    time_array,
                )

                processed_files.append(str(output_file))

                # 4. 数据质量检查
                print(f"\n数据质量检查:")
                print(f"  位置范围: [{np.min(q_filtered):.4f}, {np.max(q_filtered):.4f}] rad")
                print(f"  速度范围: [{np.min(dq_filtered):.4f}, {np.max(dq_filtered):.4f}] rad/s")
                print(f"  加速度范围: [{np.min(ddq_filtered):.4f}, {np.max(ddq_filtered):.4f}] rad/s²")
                print(f"  力矩范围: [{np.min(tau_filtered):.4f}, {np.max(tau_filtered):.4f}] Nm")

            except Exception as e:
                print(f"❌ 处理失败: {e}")
                import traceback
                traceback.print_exc()

        return processed_files

    def convert_to_mat_files(self, csv_files: List[str]) -> List[str]:
        """
        转换处理后的数据为MAT格式

        Args:
            csv_files: 处理后的CSV文件列表

        Returns:
            MAT文件列表
        """
        print("\n" + "=" * 80)
        print("步骤3: 转换为MAT格式")
        print("=" * 80)

        converted_files = []

        for csv_file in csv_files:
            try:
                mat_file = self.mat_converter.convert_processed_to_mat(csv_file, str(self.mat_dir))
                converted_files.append(mat_file)
            except Exception as e:
                print(f"❌ 转换失败 {Path(csv_file).name}: {e}")

        return converted_files

    def run_complete_pipeline(self, log_dirs: List[str]) -> Dict[str, List[str]]:
        """
        运行完整的数据处理管道

        Args:
            log_dirs: 日志目录列表

        Returns:
            包含各阶段输出文件的字典
        """
        print("=" * 100)
        print("统一动力学辨识数据处理管道")
        print("=" * 100)

        # 步骤1: 转换日志目录
        converted_files = self.process_log_directories(log_dirs)

        # 步骤2: 预处理数据
        processed_files = self.preprocess_data_files(converted_files)

        # 步骤3: 转换为MAT格式
        mat_files = self.convert_to_mat_files(processed_files)

        # 总结
        print("\n" + "=" * 100)
        print("管道执行完成!")
        print("=" * 100)
        print(f"处理的日志目录: {len(log_dirs)}")
        print(f"转换的CSV文件: {len(converted_files)}")
        print(f"预处理的文件: {len(processed_files)}")
        print(f"生成的MAT文件: {len(mat_files)}")

        print(f"\n输出目录:")
        print(f"  原始转换数据: {self.output_dir}")
        print(f"  预处理数据: {self.processed_dir}")
        print(f"  MAT格式数据: {self.mat_dir}")

        if mat_files:
            print(f"\n生成的MAT文件:")
            for mat_file in mat_files:
                print(f"  - {Path(mat_file).name}")

        return {
            "converted": converted_files,
            "processed": processed_files,
            "mat_files": mat_files
        }


def main():
    """主函数 - 演示完整流程"""

    # 日志目录列表 (根据实际情况修改)
    log_dirs = [
        "/Users/lr-2002/project/instantcreation/ic_can/tools/execitation_trajectories/ic_can_log_20251015_145900_414",
        "/Users/lr-2002/project/instantcreation/ic_can/tools/execitation_trajectories/ic_can_log_20251015_145946_130",
        "/Users/lr-2002/project/instantcreation/ic_can/tools/execitation_trajectories/ic_can_log_20251015_150027_011",
    ]

    # 创建统一处理器
    processor = UnifiedDataProcessor()

    # 运行完整管道
    results = processor.run_complete_pipeline(log_dirs)

    print(f"\n✅ 处理完成! 生成了 {len(results['mat_files'])} 个MAT文件可供MATLAB使用。")
    print(f"\n在MATLAB中使用:")
    print(f"  load('filename.mat')")
    print(f"  % 数据将加载为变量 t, q, dq, ddq, tau")


if __name__ == "__main__":
    main()
