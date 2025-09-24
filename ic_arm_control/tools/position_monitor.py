#!/usr/bin/env python3
"""
ICARM Position Monitor
持续监控电机位置并使用 rerun 实时显示，支持CSV数据保存
"""

from ic_arm_control.control.IC_ARM import ICARM
import argparse
from datetime import datetime


def main():
    """Main function to run position monitoring"""
    parser = argparse.ArgumentParser(
        description="ICARM Position Monitor with CSV support"
    )
    parser.add_argument("--csv", action="store_true", help="保存数据到CSV文件")
    parser.add_argument("--filename", type=str, help="指定CSV文件名（可选）")
    parser.add_argument("--rate", type=float, default=500, help="更新频率 (Hz，默认10)")
    parser.add_argument("--duration", type=float, help="监控时长（秒，默认无限制）")

    args = parser.parse_args()

    print("ICARM Position Monitor")
    print("=" * 50)

    # 显示配置信息
    print(f"更新频率: {args.rate} Hz")
    if args.duration:
        print(f"监控时长: {args.duration} 秒")
    else:
        print("监控时长: 无限制")

    if args.csv:
        if args.filename:
            print(f"CSV保存: 启用 -> {args.filename}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"icarm_positions_{timestamp}.csv"
            print(f"CSV保存: 启用 -> {filename}")
    else:
        print("CSV保存: 禁用")

    print("\n按 Ctrl+C 停止监控\n")

    try:
        # Initialize ICARM
        arm = ICARM(debug=False, gc=False)
        arm.enable()
        # Start continuous position monitoring with CSV support
        arm.monitor_positions_continuous(
            update_rate=args.rate,
            duration=args.duration,
            save_csv=args.csv,
            csv_filename=args.filename,
        )

    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            arm.close()
        except:
            pass


def interactive_main():
    """Interactive version for easy use"""
    print("=== ICARM Position Monitor (交互模式) ===")
    print()

    print("\n=== 开始监控 ===")

    arm = ICARM(debug=False, gc=False)
    arm.monitor_positions_continuous(
        update_rate=500,
        duration=None,
        save_csv=False,
        csv_filename=None,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 命令行模式
        main()
    else:
        # 交互模式
        interactive_main()
