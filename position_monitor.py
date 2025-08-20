#!/usr/bin/env python3
"""
ICARM Position Monitor
持续监控电机位置并使用 rerun 实时显示，支持CSV数据保存
"""

from IC_ARM import ICARM
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

    # 询问是否保存CSV
    save_csv = input("是否保存数据到CSV文件? (y/N): ").lower() == "y"

    csv_filename = None
    if save_csv:
        filename_input = input("请输入CSV文件名 (留空自动生成): ").strip()
        if filename_input:
            csv_filename = filename_input

    # 询问更新频率
    rate_input = input("请输入更新频率 (Hz，默认10): ").strip()
    try:
        update_rate = float(rate_input) if rate_input else 10.0
    except ValueError:
        update_rate = 10.0

    # 询问监控时长
    duration_input = input("请输入监控时长 (秒，留空为无限制): ").strip()
    try:
        duration = float(duration_input) if duration_input else None
    except ValueError:
        duration = None

    print("\n=== 开始监控 ===")
    print(f"更新频率: {update_rate} Hz")
    print(f"监控时长: {duration if duration else '无限制'} 秒")
    print(f"CSV保存: {'启用' if save_csv else '禁用'}")
    if save_csv and csv_filename:
        print(f"CSV文件: {csv_filename}")
    print("\n按 Ctrl+C 停止监控\n")

    try:
        arm = ICARM(debug=False, gc=True)
        arm.monitor_positions_continuous(
            update_rate=update_rate,
            duration=duration,
            save_csv=save_csv,
            csv_filename=csv_filename,
        )
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            arm.close()
        except:
            pass


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 命令行模式
        main()
    else:
        # 交互模式
        interactive_main()
