#!/usr/bin/env python3
"""
ICARM Position Monitor - 高频性能测试版
持续监控电机位置并测试最大读取频率，支持FPS性能分析
"""

from ic_arm_control.control.IC_ARM import ICARM
import argparse
from datetime import datetime
import time
import threading
import numpy as np


def main():
    """Main function to run position monitoring"""
    parser = argparse.ArgumentParser(
        description="ICARM Position Monitor - 高频性能测试版"
    )
    parser.add_argument("--csv", action="store_true", help="保存数据到CSV文件")
    parser.add_argument("--filename", type=str, help="指定CSV文件名（可选）")
    parser.add_argument("--rate", type=float, default=500, help="目标更新频率 (Hz，默认500)")
    parser.add_argument("--duration", type=float, help="监控时长（秒，默认无限制）")
    parser.add_argument("--method", type=str, default="ultra_fast",
                       choices=["normal", "fast", "ultra_fast", "cached"],
                       help="状态读取方法 (默认ultra_fast)")
    parser.add_argument("--fps-test", action="store_true", help="启用FPS性能测试模式")
    parser.add_argument("--max-rate", action="store_true", help="测试最大可达频率")

    args = parser.parse_args()

    print("ICARM Position Monitor - 高频性能测试版")
    print("=" * 60)

    # 显示配置信息
    print(f"目标频率: {args.rate} Hz")
    print(f"读取方法: {args.method}")
    if args.duration:
        print(f"监控时长: {args.duration} 秒")
    else:
        print("监控时长: 无限制")

    if args.fps_test:
        print("模式: FPS性能测试")
    if args.max_rate:
        print("模式: 最大频率测试")

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
        arm = ICARM(debug=False, gc=False, enable_buffered_control=False)
        arm.enable()

        if args.max_rate:
            # 测试最大可达频率
            test_max_frequency(arm, args.method, args.duration)
        elif args.fps_test:
            # FPS性能测试模式
            run_fps_test(arm, args.method, args.rate, args.duration, args.csv, args.filename)
        else:
            # 标准监控模式
            run_high_frequency_monitor(arm, args.method, args.rate, args.duration, args.csv, args.filename)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            arm.close()
        except:
            pass


def run_high_frequency_monitor(arm, method, target_rate, duration, save_csv, filename):
    """高频监控模式 - 使用优化的状态读取方法"""
    print(f"=== 开始高频监控 ({method}方法) ===")
    print(f"目标频率: {target_rate} Hz")
    print("正在读取数据，按 Ctrl+C 停止...")
    print(f"{'时间(s)':<8} {'计数':<6} {'位置数据(度) M1-M9':<80} {'间隔(ms)':<8}")
    print("-" * 105)

    # 获取对应的状态刷新方法
    method_map = {
        "normal": arm._refresh_all_states,
        "fast": arm._refresh_all_states_fast,
        "ultra_fast": arm._refresh_all_states_ultra_fast,
        "cached": arm._refresh_all_states_cached
    }
    refresh_func = method_map[method]

    # 控制参数
    dt = 1.0 / target_rate
    start_time = time.time()
    count = 0
    last_time = start_time
    last_fps_time = start_time
    fps_count = 0

    try:
        while True:
            loop_start = time.time()

            # 检查时长
            if duration and (loop_start - start_time) >= duration:
                break

            # 使用优化的状态刷新方法
            refresh_func()

            # 获取位置数据
            positions = arm.get_positions_degrees(refresh=False)
            velocities = arm.get_velocities_degrees(refresh=False)

            # 计算时间间隔
            current_time = time.time()
            interval_ms = (current_time - last_time) * 1000
            last_time = current_time

            # FPS计算
            count += 1
            fps_count += 1

            # 格式化位置数据 - 显示所有电机
            pos_str = ", ".join([f"{p:6.1f}" for p in positions])  # 显示所有电机

            elapsed = current_time - start_time

            # 实时打印数据
            print(f"\r{elapsed:6.2f}  {count:<6} {pos_str:<60} {interval_ms:6.2f}", end="", flush=True)

            # 每秒显示一次FPS统计
            if current_time - last_fps_time >= 1.0:
                actual_fps = fps_count / (current_time - last_fps_time)
                print(f"\n[FPS: {actual_fps:1f} Hz | 目标: {target_rate:1f} Hz]", end="", flush=True)
                fps_count = 0
                last_fps_time = current_time

            # 控制循环频率
            loop_time = current_time - loop_start
            sleep_time = dt - loop_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n\n监控被用户中断")

    # 最终统计
    total_time = time.time() - start_time
    avg_fps = count / total_time if total_time > 0 else 0
    print(f"\n=== 监控统计 ===")
    print(f"总时长: {total_time:.2f} 秒")
    print(f"总读取次数: {count}")
    print(f"平均FPS: {avg_fps:.1f} Hz")
    print(f"目标FPS: {target_rate:.1f} Hz")
    print(f"达成率: {(avg_fps/target_rate)*100:.1f}%")
    print(f"使用方法: {method}")
    print(f"平均间隔: {1000.0/avg_fps:.2f} ms")


def run_fps_test(arm, method, target_rate, duration, save_csv, filename):
    """FPS性能测试模式"""
    print(f"=== FPS性能测试 ({method}方法) ===")
    print(f"测试目标频率: {target_rate} Hz")

    # 获取对应的状态刷新方法
    method_map = {
        "normal": arm._refresh_all_states,
        "fast": arm._refresh_all_states_fast,
        "ultra_fast": arm._refresh_all_states_ultra_fast,
        "cached": arm._refresh_all_states_cached
    }
    refresh_func = method_map[method]

    # 测试参数
    test_duration = duration if duration else 10.0  # 默认10秒
    start_time = time.time()
    count = 0
    timestamps = []

    try:
        print(f"开始{test_duration}秒的FPS测试...")
        while time.time() - start_time < test_duration:
            loop_start = time.time()

            # 执行状态刷新
            refresh_func()

            # 记录时间戳
            timestamps.append(time.time())
            count += 1

            # 控制到目标频率
            dt = 1.0 / target_rate
            loop_time = time.time() - loop_start
            if loop_time < dt:
                time.sleep(dt - loop_time)

    except KeyboardInterrupt:
        print("测试被用户中断")

    # 计算性能统计
    total_time = time.time() - start_time
    avg_fps = count / total_time if total_time > 0 else 0

    # 计算FPS稳定性
    if len(timestamps) > 1:
        intervals = np.diff(timestamps)
        fps_values = 1.0 / intervals
        fps_std = np.std(fps_values)
        fps_min = np.min(fps_values)
        fps_max = np.max(fps_values)
    else:
        fps_std = fps_min = fps_max = 0

    print(f"\n=== FPS测试结果 ===")
    print(f"测试方法: {method}")
    print(f"目标频率: {target_rate:.1f} Hz")
    print(f"实际平均FPS: {avg_fps:.1f} Hz")
    print(f"FPS标准差: {fps_std:.1f} Hz")
    print(f"FPS范围: {fps_min:.1f} - {fps_max:.1f} Hz")
    print(f"稳定性: {((1 - fps_std/avg_fps) * 100):.1f}%")
    print(f"测试时长: {total_time:.2f} 秒")
    print(f"总读取次数: {count}")


def test_max_frequency(arm, method, duration):
    """测试最大可达频率"""
    print(f"=== 最大频率测试 ({method}方法) ===")

    # 获取对应的状态刷新方法
    method_map = {
        "normal": arm._refresh_all_states,
        "fast": arm._refresh_all_states_fast,
        "ultra_fast": arm._refresh_all_states_ultra_fast,
        "cached": arm._refresh_all_states_cached
    }
    refresh_func = method_map[method]

    test_duration = duration if duration else 5.0  # 默认5秒

    # 预热
    print("预热中...")
    for _ in range(10):
        refresh_func()
    time.sleep(0.1)

    # 开始最大频率测试
    print(f"开始{test_duration}秒的最大频率测试...")
    print("正在全速读取数据，按 Ctrl+C 停止...")
    print(f"{'时间(s)':<8} {'计数':<8} {'当前FPS':<10} {'间隔(ms)':<8} {'位置数据(前3个电机)'}")
    print("-" * 80)

    start_time = time.time()
    count = 0
    last_time = start_time
    last_fps_time = start_time
    fps_count = 0
    last_positions = None

    try:
        while time.time() - start_time < test_duration:
            loop_start = time.time()

            # 执行状态刷新
            refresh_func()

            # 获取位置数据
            positions = arm.get_positions_degrees(refresh=False)

            # 计算时间间隔和FPS
            current_time = time.time()
            interval_ms = (current_time - last_time) * 1000
            last_time = current_time

            count += 1
            fps_count += 1

            # 计算当前FPS
            current_fps = 1000.0 / interval_ms if interval_ms > 0 else 0

            # 格式化位置数据（只显示前3个电机）
            if positions is not None and len(positions) >= 3:
                pos_str = f"{positions[0]:6.1f}, {positions[1]:6.1f}, {positions[2]:6.1f}"
            else:
                pos_str = "N/A"

            elapsed = current_time - start_time

            # 实时显示数据
            print(f"\r{elapsed:6.2f}  {count:<8} {current_fps:<9.1f} {interval_ms:<8.2f} {pos_str}", end="", flush=True)

            # 检查位置数据是否在变化
            if last_positions is not None and positions is not None:
                pos_changed = any(abs(p - lp) > 0.01 for p, lp in zip(positions[:3], last_positions[:3]))
                if pos_changed:
                    print(f"\n[检测到位置变化!]", end="", flush=True)
            last_positions = positions

            # 每秒显示统计
            if current_time - last_fps_time >= 1.0:
                actual_fps = fps_count / (current_time - last_fps_time)
                print(f"\n[平均FPS: {actual_fps:.1f}]", end="", flush=True)
                fps_count = 0
                last_fps_time = current_time

    except KeyboardInterrupt:
        print("\n测试被用户中断")

    # 计算结果
    total_time = time.time() - start_time
    max_fps = count / total_time if total_time > 0 else 0

    print(f"\n\n=== 最大频率测试结果 ===")
    print(f"测试方法: {method}")
    print(f"测试时长: {total_time:.2f} 秒")
    print(f"总读取次数: {count}")
    print(f"最大可达FPS: {max_fps:.1f} Hz")
    print(f"平均间隔: {1000.0/max_fps:.2f} ms")

    # 性能评估
    if max_fps >= 1000:
        print("性能等级: 🚀 超高频 (>=1kHz)")
    elif max_fps >= 500:
        print("性能等级: ⚡ 高频 (500Hz-1kHz)")
    elif max_fps >= 100:
        print("性能等级: 📊 中频 (100Hz-500Hz)")
    else:
        print("性能等级: 🐌 低频 (<100Hz)")

    print(f"\n💡 说明: 如果位置数据一直为0.00，可能是因为:")
    print("   1. 电机未启用或未连接")
    print("   2. 电机在零位且没有移动")
    print("   3. 读取的是缓存数据")
    print("   但FPS测试仍然有效，反映了读取速度")


def interactive_main():
    """Interactive version for easy use"""
    print("=== ICARM Position Monitor (交互模式) ===")
    print("选择测试模式:")
    print("1. 高频监控 (默认)")
    print("2. FPS性能测试")
    print("3. 最大频率测试")
    print("4. 方法对比测试")

    choice = input("\n请输入选择 (1-4, 默认1): ").strip() or "1"

    print("\n=== 开始测试 ===")

    arm = ICARM(debug=False, gc=False, enable_buffered_control=False)
    arm.enable()

    if choice == "1":
        # 高频监控
        rate = float(input("输入目标频率 (Hz, 默认500): ") or "500")
        method = input("输入读取方法 (normal/fast/ultra_fast/cached, 默认ultra_fast): ").strip() or "ultra_fast"
        run_high_frequency_monitor(arm, method, rate, None, False, None)
    elif choice == "2":
        # FPS性能测试
        rate = float(input("输入目标频率 (Hz, 默认500): ") or "500")
        method = input("输入读取方法 (normal/fast/ultra_fast/cached, 默认ultra_fast): ").strip() or "ultra_fast"
        duration = float(input("输入测试时长 (秒, 默认10): ") or "10")
        run_fps_test(arm, method, rate, duration, False, None)
    elif choice == "3":
        # 最大频率测试
        method = input("输入读取方法 (normal/fast/ultra_fast/cached, 默认ultra_fast): ").strip() or "ultra_fast"
        duration = float(input("输入测试时长 (秒, 默认5): ") or "5")
        test_max_frequency(arm, method, duration)
    elif choice == "4":
        # 方法对比测试
        compare_all_methods(arm)
    else:
        print("无效选择，使用默认高频监控")
        run_high_frequency_monitor(arm, "ultra_fast", 500, None, False, None)


def compare_all_methods(arm):
    """对比所有方法的性能"""
    print("=== 所有方法性能对比 ===")
    methods = ["normal", "fast", "ultra_fast", "cached"]
    results = {}

    test_duration = 3.0  # 每个方法测试3秒

    for method in methods:
        print(f"\n测试方法: {method}")
        start_time = time.time()
        count = 0

        method_map = {
            "normal": arm._refresh_all_states,
            "fast": arm._refresh_all_states_fast,
            "ultra_fast": arm._refresh_all_states_ultra_fast,
            "cached": arm._refresh_all_states_cached
        }
        refresh_func = method_map[method]

        # 预热
        for _ in range(5):
            refresh_func()
        time.sleep(0.05)

        # 测试
        test_start = time.time()
        while time.time() - test_start < test_duration:
            refresh_func()
            count += 1

        actual_duration = time.time() - test_start
        fps = count / actual_duration
        results[method] = fps

        print(f"FPS: {fps:.1f} Hz")

    # 显示结果
    print(f"\n=== 对比结果 ===")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print(f"{'方法':<12s} {'FPS':<8s} {'性能比':<8s}")
    print("-" * 35)
    best_fps = sorted_results[0][1]

    for method, fps in sorted_results:
        ratio = fps / best_fps * 100
        print(f"{method:<12s} {fps:<8.1f} {ratio:<7.1f}%")

    print(f"\n🏆 最佳方法: {sorted_results[0][0]} ({sorted_results[0][1]:.1f} Hz)")
    print(f"📈 性能提升: {best_fps/sorted_results[-1][1]:.1f}x")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 命令行模式
        main()
    else:
        # 交互模式
        interactive_main()
