#!/usr/bin/env python3
"""
IC ARM 伪重力补偿演示程序

提供多种测试模式来验证重力补偿效果：
1. 基础补偿模式 - 标准的位置保持
2. 刚度测试模式 - 不同Kp值的对比
3. 阻尼测试模式 - 不同Kd值的对比
4. 频率测试模式 - 不同控制频率的对比
5. 交互测试模式 - 用户可以手动调整参数

使用方法:
    python gravity_compensation_demo.py [模式]

模式:
    basic       基础补偿测试 (默认)
    stiffness   刚度对比测试
    damping     阻尼对比测试
    frequency   频率对比测试
    interactive 交互式测试
"""

import sys
import argparse
import time
import numpy as np
from IC_ARM import ICARM

def basic_compensation_test(arm):
    """基础重力补偿测试"""
    print("=== 基础重力补偿测试 ===")
    print("这个测试会运行标准的重力补偿，保持当前姿态")
    print("你可以轻推机械臂，观察其回到原位置的能力")
    print()
    
    input("按Enter开始基础补偿测试...")
    
    return arm.pseudo_gravity_compensation(
        update_rate=200,
        duration=None,
        kp_scale=1.0,
        kd_scale=1.0,
        enable_logging=True
    )

def stiffness_comparison_test(arm):
    """刚度对比测试"""
    print("=== 刚度对比测试 ===")
    print("这个测试会依次使用不同的Kp值，让你感受刚度差异")
    
    kp_values = [0.5, 1.0, 1.5, 2.0]
    test_duration = 15.0
    
    for i, kp_scale in enumerate(kp_values):
        print(f"\n--- 测试 {i+1}/{len(kp_values)}: Kp缩放 = {kp_scale} ---")
        print(f"当前刚度: {'较软' if kp_scale < 1.0 else '标准' if kp_scale == 1.0 else '较硬'}")
        print("轻推机械臂感受刚度差异...")
        
        input("按Enter开始这个刚度测试...")
        
        success = arm.pseudo_gravity_compensation(
            update_rate=50.0,
            duration=test_duration,
            kp_scale=kp_scale,
            kd_scale=1.0,
            enable_logging=True
        )
        
        if not success:
            return False
        
        if i < len(kp_values) - 1:
            print(f"刚度测试 {i+1} 完成，准备下一个...")
            time.sleep(2)
    
    print("\n✓ 所有刚度测试完成")
    return True

def damping_comparison_test(arm):
    """阻尼对比测试"""
    print("=== 阻尼对比测试 ===")
    print("这个测试会依次使用不同的Kd值，让你感受阻尼差异")
    
    kd_values = [0.5, 1.0, 1.5, 2.0]
    test_duration = 15.0
    
    for i, kd_scale in enumerate(kd_values):
        print(f"\n--- 测试 {i+1}/{len(kd_values)}: Kd缩放 = {kd_scale} ---")
        print(f"当前阻尼: {'较小' if kd_scale < 1.0 else '标准' if kd_scale == 1.0 else '较大'}")
        print("快速推拉机械臂感受阻尼差异...")
        
        input("按Enter开始这个阻尼测试...")
        
        success = arm.pseudo_gravity_compensation(
            update_rate=50.0,
            duration=test_duration,
            kp_scale=1.0,
            kd_scale=kd_scale,
            enable_logging=True
        )
        
        if not success:
            return False
        
        if i < len(kd_values) - 1:
            print(f"阻尼测试 {i+1} 完成，准备下一个...")
            time.sleep(2)
    
    print("\n✓ 所有阻尼测试完成")
    return True

def frequency_comparison_test(arm):
    """频率对比测试"""
    print("=== 控制频率对比测试 ===")
    print("这个测试会使用不同的控制频率，观察响应性差异")
    
    frequencies = [20, 50, 100]
    test_duration = 20.0
    
    for i, freq in enumerate(frequencies):
        print(f"\n--- 测试 {i+1}/{len(frequencies)}: 控制频率 = {freq} Hz ---")
        print(f"当前频率: {'较低' if freq < 50 else '标准' if freq == 50 else '较高'}")
        print("推拉机械臂观察响应速度...")
        
        input("按Enter开始这个频率测试...")
        
        success = arm.pseudo_gravity_compensation(
            update_rate=freq,
            duration=test_duration,
            kp_scale=1.0,
            kd_scale=1.0,
            enable_logging=True
        )
        
        if not success:
            return False
        
        if i < len(frequencies) - 1:
            print(f"频率测试 {i+1} 完成，准备下一个...")
            time.sleep(2)
    
    print("\n✓ 所有频率测试完成")
    return True

def interactive_test(arm):
    """交互式测试"""
    print("=== 交互式重力补偿测试 ===")
    print("你可以实时调整参数并观察效果")
    print()
    
    # 默认参数
    frequency = 50.0
    kp_scale = 1.0
    kd_scale = 1.0
    duration = 30.0
    
    while True:
        print(f"当前参数:")
        print(f"  控制频率: {frequency} Hz")
        print(f"  Kp缩放: {kp_scale}")
        print(f"  Kd缩放: {kd_scale}")
        print(f"  测试时长: {duration}s")
        print()
        print("选项:")
        print("  1. 开始测试")
        print("  2. 调整控制频率")
        print("  3. 调整Kp缩放")
        print("  4. 调整Kd缩放")
        print("  5. 调整测试时长")
        print("  0. 退出")
        
        try:
            choice = input("\n请选择 (0-5): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                print(f"\n开始重力补偿测试 (频率:{frequency}Hz, Kp:{kp_scale}, Kd:{kd_scale}, 时长:{duration}s)")
                success = arm.pseudo_gravity_compensation(
                    update_rate=frequency,
                    duration=duration,
                    kp_scale=kp_scale,
                    kd_scale=kd_scale,
                    enable_logging=True
                )
                if not success:
                    print("测试异常结束")
                else:
                    print("测试完成")
                print()
            elif choice == '2':
                new_freq = float(input("输入新的控制频率 (Hz, 10-200): "))
                if 10 <= new_freq <= 200:
                    frequency = new_freq
                    print(f"控制频率已设置为 {frequency} Hz")
                else:
                    print("频率应在10-200Hz范围内")
            elif choice == '3':
                new_kp = float(input("输入新的Kp缩放 (0.1-5.0): "))
                if 0.1 <= new_kp <= 5.0:
                    kp_scale = new_kp
                    print(f"Kp缩放已设置为 {kp_scale}")
                else:
                    print("Kp缩放应在0.1-5.0范围内")
            elif choice == '4':
                new_kd = float(input("输入新的Kd缩放 (0.1-5.0): "))
                if 0.1 <= new_kd <= 5.0:
                    kd_scale = new_kd
                    print(f"Kd缩放已设置为 {kd_scale}")
                else:
                    print("Kd缩放应在0.1-5.0范围内")
            elif choice == '5':
                new_duration = float(input("输入新的测试时长 (秒, 5-300): "))
                if 5 <= new_duration <= 300:
                    duration = new_duration
                    print(f"测试时长已设置为 {duration}s")
                else:
                    print("测试时长应在5-300秒范围内")
            else:
                print("无效选择")
                
        except ValueError:
            print("输入格式错误，请输入数字")
        except KeyboardInterrupt:
            print("\n用户中断")
            break
        
        print()
    
    return True

def main():
    parser = argparse.ArgumentParser(description='IC ARM 伪重力补偿演示')
    parser.add_argument('mode', nargs='?', default='basic',
                        choices=['basic', 'stiffness', 'damping', 'frequency', 'interactive'],
                        help='测试模式 (默认: basic)')
    parser.add_argument('--port', type=str, default='/dev/cu.usbmodem00000000050C1',
                        help='串口端口')
    
    args = parser.parse_args()
    
    print("=== IC ARM 伪重力补偿演示程序 ===")
    print(f"测试模式: {args.mode}")
    print(f"串口: {args.port}")
    print()
    
    try:
        # 初始化IC ARM
        print("初始化IC ARM...")
        arm = ICARM(port=args.port, debug=True)
        
        # 显示当前状态
        print("当前机械臂状态:")
        positions = arm.get_positions_degrees()
        for i, pos in enumerate(positions):
            print(f"  关节{i+1} (m{i+1}): {pos:6.1f}°")
        print()
        
        # 根据模式执行测试
        if args.mode == 'basic':
            success = basic_compensation_test(arm)
        elif args.mode == 'stiffness':
            success = stiffness_comparison_test(arm)
        elif args.mode == 'damping':
            success = damping_comparison_test(arm)
        elif args.mode == 'frequency':
            success = frequency_comparison_test(arm)
        elif args.mode == 'interactive':
            success = interactive_test(arm)
        else:
            print(f"未知模式: {args.mode}")
            return 1
        
        if success:
            print("✓ 演示完成")
        else:
            print("✗ 演示异常结束")
            return 1
            
    except KeyboardInterrupt:
        print("\n用户中断演示")
        return 0
    except Exception as e:
        print(f"✗ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        try:
            if 'arm' in locals():
                print("关闭连接...")
                arm.close()
        except:
            pass
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
