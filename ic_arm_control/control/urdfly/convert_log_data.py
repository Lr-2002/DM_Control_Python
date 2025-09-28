#!/usr/bin/env python3
"""
Convert log directories directly to dynamics format
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

def convert_log_dir_to_dynamics(log_dir, output_file):
    """
    Convert a single log directory to dynamics format

    Args:
        log_dir: Log directory path
        output_file: Output CSV file path
    """
    print(f"Converting {log_dir}...")

    # Check for required files
    motor_states_file = os.path.join(log_dir, "motor_states.csv")
    joint_commands_file = os.path.join(log_dir, "joint_commands.csv")

    if not os.path.exists(motor_states_file):
        print(f"  Missing motor_states.csv")
        return None

    if not os.path.exists(joint_commands_file):
        print(f"  Missing joint_commands.csv")
        return None

    try:
        # Read data
        motor_df = pd.read_csv(motor_states_file)
        command_df = pd.read_csv(joint_commands_file)

        print(f"  Motor states: {len(motor_df)} points")
        print(f"  Joint commands: {len(command_df)} points")

        # Ensure same length
        min_len = min(len(motor_df), len(command_df))
        motor_df = motor_df.iloc[:min_len]
        command_df = command_df.iloc[:min_len]

        # Create dynamics data
        dt = 0.002  # 500Hz
        dynamics_data = []

        for i in range(len(motor_df)):
            row_data = {'time': i * dt}

            # Process all 5 joints
            for joint_id in range(1, 7):
                pos_col = f'position_motor_{joint_id}'
                vel_col = f'velocity_motor_{joint_id}'
                torque_col = f'torque_motor_{joint_id}'

                if pos_col in motor_df.columns:
                    position = motor_df[pos_col].iloc[i]
                    velocity = motor_df[vel_col].iloc[i]
                    torque = motor_df[torque_col].iloc[i]

                    row_data[f'm{joint_id}_pos_actual'] = position
                    row_data[f'm{joint_id}_vel_actual'] = velocity
                    row_data[f'm{joint_id}_torque'] = torque

            dynamics_data.append(row_data)

        # Calculate accelerations
        for joint_id in range(1, 7):
            vel_col = f'm{joint_id}_vel_actual'
            if vel_col in dynamics_data[0]:
                velocities = [row[vel_col] for row in dynamics_data]
                accelerations = np.gradient(velocities, dt)

                for j, acc in enumerate(accelerations):
                    dynamics_data[j][f'm{joint_id}_acc_actual'] = acc

        # Create DataFrame and save
        df = pd.DataFrame(dynamics_data)
        df = df.fillna(0)
        df.to_csv(output_file, index=False)

        print(f"  Saved: {output_file}")
        print(f"  Shape: {df.shape}")

        return output_file

    except Exception as e:
        print(f"  Error: {e}")
        return None

def main():
    """Main conversion function"""
    log_dirs = [
        # "/Users/lr-2002/project/instantcreation/IC_arm_control/logs/20250928_171020",
        # "/Users/lr-2002/project/instantcreation/IC_arm_control/logs/20250928_170759",
        # "/Users/lr-2002/project/instantcreation/IC_arm_control/logs/20250928_170840",
        # "/Users/lr-2002/project/instantcreation/IC_arm_control/logs/20250928_170933",
        "/Users/lr-2002/project/instantcreation/IC_arm_control/logs/20250928_210525",
        "/Users/lr-2002/project/instantcreation/IC_arm_control/logs/20250928_210408",
        "/Users/lr-2002/project/instantcreation/IC_arm_control/logs/20250928_214524",
        "/Users/lr-2002/project/instantcreation/IC_arm_control/logs/20250928_214427",
        "/Users/lr-2002/project/instantcreation/IC_arm_control/logs/20250928_220012",
        "/Users/lr-2002/project/instantcreation/IC_arm_control/logs/20250928_222206"
    ]

    output_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control/dyn_ana"
    os.makedirs(output_dir, exist_ok=True)

    converted_files = []

    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            output_file = os.path.join(output_dir, f"dynamics_{os.path.basename(log_dir)}.csv")
            result = convert_log_dir_to_dynamics(log_dir, output_file)
            if result:
                converted_files.append(result)
        else:
            print(f"Directory not found: {log_dir}")

    if converted_files:
        # Create merged dataset
        print(f"\nMerging {len(converted_files)} files...")
        all_data = []
        for file in converted_files:
            df = pd.read_csv(file)
            df['source'] = os.path.basename(file)
            all_data.append(df)

        merged_df = pd.concat(all_data, ignore_index=True)
        merged_df = merged_df.sort_values('time').reset_index(drop=True)

        merged_file = os.path.join(output_dir, "merged_log_data.csv")
        merged_df.to_csv(merged_file, index=False)

        print(f"Merged file: {merged_file}")
        print(f"Total points: {len(merged_df)}")

        return merged_file
    else:
        print("No files converted")
        return None

if __name__ == "__main__":
    main()