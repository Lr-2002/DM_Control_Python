#!/usr/bin/env python3
"""
简化的500Hz频率校准验证脚本
检查主要配置文件中的频率设置
"""

import sys
import os
import re

def check_config_file_frequencies():
    """检查配置文件中的频率设置"""
    print("=== IC ARM 500Hz频率校准验证 ===\n")

    results = {}

    # 1. 检查BufferControlThread
    print("1. 检查BufferControlThread...")
    try:
        with open('control/buffer_control_thread.py', 'r') as f:
            content = f.read()
            # 查找默认频率
            match = re.search(r'control_freq.*?(\d+)', content)
            if match:
                freq = int(match.group(1))
                results['BufferControlThread'] = freq
                print(f"   默认控制频率: {freq}Hz")

                # 检查注释
                if '500Hz' in content:
                    print(f"   注释: 包含500Hz标识")
    except Exception as e:
        print(f"   ❌ 无法读取: {e}")
        results['BufferControlThread'] = None

    # 2. 检查IC_ARM
    print("\n2. 检查IC_ARM...")
    try:
        with open('control/IC_ARM.py', 'r') as f:
            content = f.read()
            # 查找默认频率
            match = re.search(r'control_freq.*?(\d+)', content)
            if match:
                freq = int(match.group(1))
                results['IC_ARM'] = freq
                print(f"   默认控制频率: {freq}Hz")
    except Exception as e:
        print(f"   ❌ 无法读取: {e}")
        results['IC_ARM'] = None

    # 3. 检查轨迹生成器
    print("\n3. 检查轨迹生成器...")
    try:
        with open('tools/trajectory_generator.py', 'r') as f:
            content = f.read()
            # 查找dt设置
            dt_matches = re.findall(r'dt.*?=.*?([\d.]+)', content)
            dt_values = [float(dt) for dt in dt_matches]
            if dt_values:
                avg_dt = sum(dt_values) / len(dt_values)
                freq = 1.0 / avg_dt if avg_dt > 0 else 0
                results['TrajectoryGenerator'] = freq
                print(f"   平均采样间隔: {avg_dt*1000:.2f}ms")
                print(f"   对应频率: {freq:.0f}Hz")
    except Exception as e:
        print(f"   ❌ 无法读取: {e}")
        results['TrajectoryGenerator'] = None

    # 4. 检查轨迹执行器
    print("\n4. 检查轨迹执行器...")
    try:
        with open('tools/trajectory_executor.py', 'r') as f:
            content = f.read()
            # 查找sleep设置
            match = re.search(r'time\.sleep\(([\d.]+)\)', content)
            if match:
                sleep_time = float(match.group(1))
                freq = 1.0 / sleep_time if sleep_time > 0 else 0
                results['TrajectoryExecutor'] = freq
                print(f"   睡眠时间: {sleep_time*1000:.2f}ms")
                print(f"   对应频率: {freq:.0f}Hz")
    except Exception as e:
        print(f"   ❌ 无法读取: {e}")
        results['TrajectoryExecutor'] = None

    # 5. 检查配置加载器
    print("\n5. 检查配置加载器...")
    try:
        with open('utils/config_loader.py', 'r') as f:
            content = f.read()
            # 查找默认频率
            match = re.search(r'default_frequency.*?(\d+)', content)
            if match:
                freq = int(match.group(1))
                results['ConfigLoader'] = freq
                print(f"   默认控制频率: {freq}Hz")
    except Exception as e:
        print(f"   ❌ 无法读取: {e}")
        results['ConfigLoader'] = None

    # 6. 检查LeRobot配置
    print("\n6. 检查LeRobot配置...")
    try:
        with open('lerobot_integration/config.py', 'r') as f:
            content = f.read()
            # 查找频率设置
            sample_rate_matches = re.findall(r'sample_rate.*?([\d.]+)', content)
            control_freq_matches = re.findall(r'control_frequency.*?([\d.]+)', content)

            if sample_rate_matches:
                sample_rates = [float(rate) for rate in sample_rate_matches]
                avg_sample_rate = sum(sample_rates) / len(sample_rates)
                results['LeRobotSampleRate'] = avg_sample_rate
                print(f"   平均采样率: {avg_sample_rate:.0f}Hz")

            if control_freq_matches:
                control_freqs = [float(freq) for freq in control_freq_matches]
                avg_control_freq = sum(control_freqs) / len(control_freqs)
                results['LeRobotControlFreq'] = avg_control_freq
                print(f"   平均控制频率: {avg_control_freq:.0f}Hz")
    except Exception as e:
        print(f"   ❌ 无法读取: {e}")
        results['LeRobotSampleRate'] = None
        results['LeRobotControlFreq'] = None

    return results

def verify_consistency(results):
    """验证频率一致性"""
    print("\n=== 频率一致性验证 ===")

    target_freq = 500.0
    consistent = True

    for component, freq in results.items():
        if freq is None:
            print(f"   ❌ {component}: 无法读取")
            consistent = False
        elif abs(freq - target_freq) < 50:  # 允许50Hz误差
            print(f"   ✅ {component}: {freq:.0f}Hz")
        else:
            print(f"   ❌ {component}: {freq:.0f}Hz (期望{target_freq}Hz)")
            consistent = False

    if consistent:
        print(f"\n🎉 所有组件频率校准完成: ~{target_freq}Hz")
    else:
        print(f"\n⚠️ 发现频率不一致，请检查配置")

    return consistent

def print_summary():
    """打印总结信息"""
    print("\n=== 500Hz校准总结 ===")
    print("已更新的组件:")
    print("• BufferControlThread: 300Hz → 500Hz")
    print("• IC_ARM: 300Hz → 500Hz")
    print("• 轨迹生成器: 100Hz/1000Hz → 500Hz")
    print("• 轨迹执行器: 1000Hz → 500Hz")
    print("• 配置加载器: 新增500Hz支持")
    print("• LeRobot配置: 新增control_500hz配置")

    print("\n性能要求:")
    print("• 控制周期: 2ms")
    print("• 精度要求: <0.5ms")
    print("• 建议系统: macOS/Linux")

if __name__ == "__main__":
    # 检查配置文件频率
    results = check_config_file_frequencies()

    # 验证一致性
    is_consistent = verify_consistency(results)

    # 打印总结
    print_summary()

    if is_consistent:
        print("\n✅ IC ARM系统500Hz频率校准完成")
        sys.exit(0)
    else:
        print("\n❌ 频率校准存在问题")
        sys.exit(1)