#!/usr/bin/env python3
"""
将IC ARM数据集转换为动力学辨识所需的格式
从datasets目录的CSV文件转换为urdfly minimum_param.py所需的格式
"""

import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path
import sys

def convert_ic_arm_data_to_dynamics_format(dataset_dir="/Users/lr-2002/project/instantcreation/IC_arm_control/datasets",
                                         output_dir="/Users/lr-2002/project/instantcreation/IC_arm_control/dyn_ana"):
    """
    将IC ARM数据集转换为动力学辨识格式

    Args:
        dataset_dir: 输入数据集目录
        output_dir: 输出目录

    Returns:
        转换后的文件路径列表
    """
    print("=== IC ARM数据集格式转换 ===\n")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有数据集目录
    dataset_dirs = [d for d in Path(dataset_dir).iterdir() if d.is_dir()]
    print(f"找到 {len(dataset_dirs)} 个数据集目录")

    converted_files = []

    for dataset_path in dataset_dirs:
        print(f"\n处理数据集: {dataset_path.name}")

        # 检查必要的文件
        motor_states_file = dataset_path / "motor_states.csv"
        joint_commands_file = dataset_path / "joint_commands.csv"

        if not motor_states_file.exists():
            print(f"  ⚠️ 缺少motor_states.csv文件，跳过")
            continue

        if not joint_commands_file.exists():
            print(f"  ⚠️ 缺少joint_commands.csv文件，跳过")
            continue

        try:
            # 读取数据
            print(f"  读取motor_states.csv...")
            motor_df = pd.read_csv(motor_states_file)

            print(f"  读取joint_commands.csv...")
            command_df = pd.read_csv(joint_commands_file)

            # 数据统计
            print(f"  motor_states数据点: {len(motor_df)}")
            print(f"  joint_commands数据点: {len(command_df)}")

            # 检查时间戳对齐
            if len(motor_df) != len(command_df):
                print(f"  ⚠️ 数据点数不一致，将使用较少的数据点")
                min_len = min(len(motor_df), len(command_df))
                motor_df = motor_df.iloc[:min_len]
                command_df = command_df.iloc[:min_len]

            # 计算加速度（数值微分）
            print(f"  计算加速度...")
            dt = 0.002  # 假设采样频率为500Hz

            # 准备动力学辨识所需的数据结构
            dynamics_data = []

            # 处理每个时间点
            for j in range(len(motor_df)):
                time_point = j * dt
                row_data = {'time': time_point}

                # 提取前5个关节的数据（urdfly代码支持5个关节）
                for i in range(1, 6):  # 只处理前5个关节
                    pos_col = f'position_motor_{i}'
                    vel_col = f'velocity_motor_{i}'
                    torque_col = f'torque_motor_{i}'

                    if pos_col in motor_df.columns:
                        # 获取位置、速度、力矩数据
                        position = motor_df[pos_col].iloc[j]
                        velocity = motor_df[vel_col].iloc[j]
                        torque = motor_df[torque_col].iloc[j]

                        row_data[f'm{i}_pos_actual'] = position
                        row_data[f'm{i}_vel_actual'] = velocity
                        row_data[f'm{i}_torque'] = torque

                dynamics_data.append(row_data)

            # 计算所有关节的加速度（需要完整序列）
            for i in range(1, 6):
                vel_col = f'm{i}_vel_actual'
                if vel_col in dynamics_data[0]:
                    velocities = [row[vel_col] for row in dynamics_data]
                    accelerations = np.gradient(velocities, dt)

                    # 将加速度添加回数据
                    for j, acc in enumerate(accelerations):
                        dynamics_data[j][f'm{i}_acc_actual'] = acc

            # 转换为DataFrame
            if dynamics_data:
                final_df = pd.DataFrame(dynamics_data)

                # 填充可能的NaN值为0
                final_df = final_df.fillna(0)

                # 保存转换后的数据
                output_file = os.path.join(output_dir, f"dynamics_data_{dataset_path.name}.csv")
                final_df.to_csv(output_file, index=False)

                print(f"  ✓ 转换完成: {output_file}")
                print(f"    数据点数: {len(final_df)}")
                print(f"    列数: {len(final_df.columns)}")

                # 数据统计
                print(f"    位置范围:")
                for i in range(1, 6):
                    pos_col = f'm{i}_pos_actual'
                    if pos_col in final_df.columns:
                        pos_data = final_df[pos_col]
                        pos_range_deg = np.degrees([pos_data.min(), pos_data.max()])
                        print(f"      关节{i}: {pos_range_deg[0]:.1f}° ~ {pos_range_deg[1]:.1f}°")

                converted_files.append(output_file)
            else:
                print(f"  ⚠️ 没有找到有效数据")

        except Exception as e:
            print(f"  ❌ 转换失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n=== 转换完成 ===")
    print(f"成功转换 {len(converted_files)} 个数据集")

    if converted_files:
        print(f"\n转换后的文件:")
        for file in converted_files:
            file_size = os.path.getsize(file) / (1024*1024)  # MB
            print(f"  {file} ({file_size:.2f} MB)")

    return converted_files

def create_merged_dataset(converted_files, output_file="merged_dynamics_data.csv"):
    """
    合并所有转换后的数据集

    Args:
        converted_files: 转换后的文件列表
        output_file: 输出文件路径

    Returns:
        合并后的文件路径
    """
    print(f"\n=== 合并数据集 ===")

    if not converted_files:
        print("没有可合并的文件")
        return None

    # 读取所有文件
    all_data = []
    for file in converted_files:
        try:
            df = pd.read_csv(file)
            df['data_source'] = os.path.basename(file)
            all_data.append(df)
            print(f"  读取: {os.path.basename(file)} ({len(df)} 数据点)")
        except Exception as e:
            print(f"  ⚠️ 读取失败 {file}: {e}")

    if not all_data:
        print("所有文件读取失败")
        return None

    # 合并数据
    merged_df = pd.concat(all_data, ignore_index=True)

    # 按时间排序
    merged_df = merged_df.sort_values('time').reset_index(drop=True)

    # 保存合并后的数据
    merged_df.to_csv(output_file, index=False)

    print(f"✓ 合并完成: {output_file}")
    print(f"  总数据点数: {len(merged_df)}")
    print(f"  数据源数量: {merged_df['data_source'].nunique()}")
    print(f"  时间范围: {merged_df['time'].min():.3f}s ~ {merged_df['time'].max():.3f}s")

    return output_file

def main():
    """主函数"""
    print("IC ARM数据集转换为动力学辨识格式\n")

    # 转换数据集
    converted_files = convert_ic_arm_data_to_dynamics_format()

    if converted_files:
        # 创建合并数据集
        merged_file = create_merged_dataset(converted_files)

        print(f"\n=== 转换总结 ===")
        print(f"✓ 成功转换 {len(converted_files)} 个数据集")
        print(f"✓ 合并数据集: {merged_file}")
        print(f"✓ 数据格式已适配urdfly minimum_param.py")

        return True
    else:
        print(f"\n❌ 没有成功转换任何数据集")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)