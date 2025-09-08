#!/usr/bin/env python3
"""
统一电机控制系统测试示例
"""

import sys
import os
import time
import numpy as np

# 添加Python目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from unified_motor_control import *
from src import usb_class, can_value_type
from damiao import Motor_Control, DmActData, DM_Motor_Type, Control_Mode
from damiao import limit_param as dm_limit
from ht_motor import HTMotorManager


# 初始化示例
def initialize_unified_motor_system():
    """
    初始化统一电机控制系统
    """
    try:
        # 创建USB硬件接口
        print("正在初始化USB硬件接口...")
        usb_hw = usb_class(1000000, 5000000, "F561E08C892274DB09496BCC1102DBC5")

        # 创建管理器
        print("创建电机管理器...")
        manager = MotorManager(usb_hw)

        # 配置达妙电机数据
        damiao_data = [
            DmActData(DM_Motor_Type.DM10010L, Control_Mode.MIT_MODE, 0x01, 0x11, 0, 0),
            DmActData(DM_Motor_Type.DM4340, Control_Mode.MIT_MODE, 0x02, 0x12, 0, 0),
            DmActData(DM_Motor_Type.DM6248, Control_Mode.MIT_MODE, 0x03, 0x13, 0, 0),
            DmActData(DM_Motor_Type.DM4340, Control_Mode.MIT_MODE, 0x04, 0x14, 0, 0),
            DmActData(DM_Motor_Type.DM4340, Control_Mode.MIT_MODE, 0x05, 0x15, 0, 0),
            DmActData(DM_Motor_Type.DM4340, Control_Mode.MIT_MODE, 0x06, 0x16, 0, 0),
        ]

        # 添加达妙电机协议
        print("初始化达妙电机协议...")
        motor_control = Motor_Control(
            usb_hw=usb_hw,  data_ptr=damiao_data
        )
        manager.add_damiao_protocol(motor_control)

        # 添加HT电机协议
        print("初始化HT电机协议...")
        ht_manager = HTMotorManager(usb_hw)
        manager.add_ht_protocol(ht_manager)

        # 添加具体电机
        print("添加达妙电机...")
        # 达妙电机 (1-6)
        for i in range(6):
            motor_info = MotorInfo(
                motor_id=i + 1,
                motor_type=MotorType.DAMIAO,
                can_id=damiao_data[i].can_id,
                name=f"damiao_{i+1}",
                kp=damiao_data[i].kp,
                kd=damiao_data[i].kd,
                limits=dm_limit[
                    damiao_data[i].motorType.value
                ],  # [pos_limit, vel_limit, torque_limit]
            )
            success = manager.add_motor(
                i + 1, "damiao", motor_info, can_id=damiao_data[i].can_id
            )
            if success:
                print(f"  ✓ 达妙电机 {i+1} 添加成功")
            else:
                print(f"  ✗ 达妙电机 {i+1} 添加失败")

        print("添加HT电机...")
        # HT电机 (7-8)
        for i in range(2):
            motor_info = MotorInfo(
                motor_id=i + 7,
                motor_type=MotorType.HIGH_TORQUE,
                can_id=0x8094,  # HT电机使用固定的发送ID
                name=f"ht_{i+7}",
                kp=0,
                kd=0,
                limits=[12.5, 50.0, 20.0],  # HT电机扭矩限制更高
            )
            success = manager.add_motor(i + 7, "ht", motor_info, ht_motor_id=i + 7)
            if success:
                print(f"  ✓ HT电机 {i+7} 添加成功")
            else:
                print(f"  ✗ HT电机 {i+7} 添加失败")

        print("电机系统初始化完成!")
        return manager

    except Exception as e:
        print(f"初始化失败: {e}")
        return None


def simple_test(manager):
    """
    简单的电机测试函数
    """
    print("\n开始简单测试...")

    try:
        # 使能所有电机
        print("使能所有电机...")
        if manager.enable_all():
            print("✓ 所有电机使能成功")
        else:
            print("✗ 部分电机使能失败")

        # 设置零位
        print("设置零位...")
        if manager.set_all_zero():
            print("✓ 零位设置成功")
        else:
            print("✗ 零位设置失败")

        # 测试读取状态
        print("\n读取电机状态:")
        manager.update_all_states()

        for i in range(1, 9):
            motor = manager.get_motor(i)
            if motor:
                state = motor.get_state()
                print(
                    f"电机 {i}: pos={state['position']:.3f}, vel={state['velocity']:.3f}, tau={state['torque']:.3f}"
                )

        # 简单的位置控制测试
        print("\n执行简单位置控制测试...")
        test_positions = [0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.0, 0.0]  # 小幅度运动
        target_velocities = [0.0] * 8
        kps = [30.0] * 8  # 适中的刚度
        kds = [1.0] * 8  # 适中的阻尼
        target_torques = [0.0] * 8

        success = manager.control_mit_batch(
            list(range(1, 9)),
            test_positions,
            target_velocities,
            kps,
            kds,
            target_torques,
        )

        if success:
            print("✓ 控制命令发送成功")
        else:
            print("✗ 控制命令发送失败")

        # 等待一段时间观察运动
        print("等待2秒观察运动...")
        time.sleep(2.0)

        # 读取最终状态
        print("\n最终状态:")
        manager.update_all_states()
        for i in range(1, 9):
            motor = manager.get_motor(i)
            if motor:
                state = motor.get_state()
                print(
                    f"电机 {i}: pos={state['position']:.3f}, vel={state['velocity']:.3f}, tau={state['torque']:.3f}"
                )

        # 回到零位
        print("\n回到零位...")
        zero_positions = [0.0] * 8
        manager.control_mit_batch(
            list(range(1, 9)),
            zero_positions,
            target_velocities,
            kps,
            kds,
            target_torques,
        )

        time.sleep(1.0)
        print("测试完成!")

    except Exception as e:
        print(f"测试过程中出错: {e}")

    finally:
        # 失能所有电机
        print("失能所有电机...")
        manager.disable_all()


def continuous_control_loop(manager, duration=10.0):
    """
    连续控制循环示例
    """
    print(f"\n开始连续控制循环 (持续 {duration} 秒)...")

    start_time = time.time()
    loop_count = 0

    try:
        # 使能所有电机
        manager.enable_all()

        while (time.time() - start_time) < duration:
            # 更新所有电机状态
            manager.update_all_states()

            # 生成正弦波轨迹
            t = time.time() - start_time
            frequency = 0.5  # Hz
            amplitude = 0.2  # rad

            target_positions = []
            for i in range(8):
                # 每个电机不同的相位
                phase = i * np.pi / 4
                pos = amplitude * np.sin(2 * np.pi * frequency * t + phase)
                target_positions.append(pos)

            target_velocities = [0.0] * 8
            kps = [20.0] * 8
            kds = [1.0] * 8
            target_torques = [0.0] * 8

            # 批量控制
            manager.control_mit_batch(
                list(range(1, 9)),
                target_positions,
                target_velocities,
                kps,
                kds,
                target_torques,
            )

            loop_count += 1

            # 每秒打印一次状态
            if loop_count % 1000 == 0:
                print(f"循环 {loop_count}, 时间: {t:.1f}s")
                for i in range(1, 3):  # 只打印前两个电机的状态
                    motor = manager.get_motor(i)
                    if motor:
                        print(
                            f"  电机{i}: pos={motor.get_position():.3f}, vel={motor.get_velocity():.3f}"
                        )

            time.sleep(0.001)  # 1000Hz控制频率

    except KeyboardInterrupt:
        print("\n用户中断控制循环")
    except Exception as e:
        print(f"\n控制循环出错: {e}")
    finally:
        # 回到零位并失能
        print("回到零位...")
        zero_positions = [0.0] * 8
        target_velocities = [0.0] * 8
        kps = [30.0] * 8
        kds = [2.0] * 8
        target_torques = [0.0] * 8

        manager.control_mit_batch(
            list(range(1, 9)),
            zero_positions,
            target_velocities,
            kps,
            kds,
            target_torques,
        )

        time.sleep(1.0)
        manager.disable_all()
        print(f"控制循环结束，总共执行了 {loop_count} 次循环")


def read_only_mode(manager):
    """
    只读模式 - 读取所有关节角度
    """
    print("\n=== 只读模式 - 关节角度读取 ===")
    print("按 Ctrl+C 退出只读模式")
    print("按 Enter 键刷新数据\n")

    try:
        manager.enable_all()
        while True:
            # 更新所有电机状态
            print("正在读取电机状态...")

            manager.update_all_states()

            # 清屏显示当前状态
            print("\033[2J\033[H")  # 清屏并移动光标到左上角
            print("=== 关节角度实时读取 ===")
            print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 60)

            # 读取并显示所有电机的状态
            motor_data = []
            for i in range(1, 9):  # 8个电机
                motor = manager.get_motor(i)
                if motor:
                    state = motor.get_state()
                    motor_type = "Damiao" if i <= 6 else "HT"
                    motor_data.append(
                        {
                            "id": i,
                            "type": motor_type,
                            "position": state["position"],
                            "velocity": state["velocity"],
                            "torque": state["torque"],
                            "error": state["error_code"],
                        }
                    )
                else:
                    motor_data.append(
                        {
                            "id": i,
                            "type": "N/A",
                            "position": 0.0,
                            "velocity": 0.0,
                            "torque": 0.0,
                            "error": -1,
                        }
                    )

            # 格式化显示
            print(
                f"{'ID':<3} {'Type':<7} {'Position(rad)':<13} {'Velocity(rad/s)':<15} {'Torque(Nm)':<12} {'Error':<5}"
            )
            print("-" * 60)

            for data in motor_data:
                status_icon = "✓" if data["error"] == 0 else "✗"
                print(
                    f"{data['id']:<3} {data['type']:<7} {data['position']:<13.4f} {data['velocity']:<15.4f} {data['torque']:<12.4f} {status_icon:<5}"
                )

            print("-" * 60)
            print("统计信息:")

            # 计算统计信息
            positions = [d["position"] for d in motor_data if d["type"] != "N/A"]
            velocities = [d["velocity"] for d in motor_data if d["type"] != "N/A"]
            torques = [d["torque"] for d in motor_data if d["type"] != "N/A"]

            if positions:
                print(f"位置范围: {min(positions):.4f} ~ {max(positions):.4f} rad")
                print(f"速度范围: {min(velocities):.4f} ~ {max(velocities):.4f} rad/s")
                print(f"扭矩范围: {min(torques):.4f} ~ {max(torques):.4f} Nm")

                # 计算总能耗（简化估算）
                # total_power = sum(abs(t * v) for t, v in zip(torques, velocities))
                # print(f"估算总功率: {total_power:.2f} W")
            #
            # print("\n按 Enter 刷新，按 Ctrl+C 退出")
            #
            # # 等待用户输入或自动刷新
            # try:
            #     import select
            #     import sys
            #
            #     # 非阻塞式输入检测（仅限Unix/Linux/macOS）
            #     if select.select([sys.stdin], [], [], 2.0)[0]:  # 2秒超时
            #         input()  # 清空输入缓冲
            #     else:
            #         time.sleep(0.5)  # 自动刷新间隔
            # except:
            #     # Windows系统或其他情况下的备用方案
            #     time.sleep(1.0)
            #
    except KeyboardInterrupt:
        print("\n\n退出只读模式")
    except Exception as e:
        print(f"\n只读模式出错: {e}")


def main():
    """
    主函数
    """
    print("=== 统一电机控制系统测试 ===")

    # 初始化系统
    manager = initialize_unified_motor_system()
    if manager is None:
        print("系统初始化失败，退出")
        return

    try:

        read_only_mode(manager)
        # while True:
        #     print("\n请选择测试模式:")
        #     print("1. 简单测试")
        #     print("2. 连续控制循环 (10秒)")
        #     print("3. 连续控制循环 (自定义时间)")
        #     print("4. 只读模式 - 读取所有关节角度")
        #     print("5. 退出")

        #     read_only_mode(manager)
            # choice = input("请输入选择 (1-5): ").strip()

            # if choice == "1":
            #     simple_test(manager)
            # elif choice == "2":
            #     continuous_control_loop(manager, 10.0)
            # elif choice == "3":
            #     try:
            #         duration = float(input("请输入持续时间(秒): "))
            #         continuous_control_loop(manager, duration)
            #     except ValueError:
            #         print("无效的时间输入")
            # # elif choice == "4":
            #     read_only_mode(manager)
            # elif choice == "5":
            #     break
            # else:
            #     print("无效选择，请重新输入")

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        print("程序结束")


if __name__ == "__main__":
    main()

