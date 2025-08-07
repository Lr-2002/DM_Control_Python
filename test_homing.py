#!/usr/bin/env python3
"""
测试回零功能
"""

from IC_ARM import ICARM
import time

def main():
    """测试回零功能"""
    print("=== 测试IC ARM回零功能 ===")
    
    arm = ICARM()
    
    try:
        # 显示当前位置
        print("\n当前位置:")
        current_pos = arm.get_current_positions_deg()
        print(f"位置: {[f'{p:.2f}°' for p in current_pos]}")
        
        # 选择插值类型
        print("\n选择插值类型:")
        print("1. 线性插值 (linear)")
        print("2. 平滑插值 (smooth) - 推荐")
        
        choice = input("请选择 (1/2，默认2): ")
        interpolation_type = 'linear' if choice == '1' else 'smooth'
        
        # 选择运动时间
        duration_input = input("请输入运动时间 (秒，默认3.0): ")
        try:
            duration = float(duration_input) if duration_input else 5.0
        except ValueError:
            duration = 3.0
        
        print(f"\n开始回零运动...")
        print(f"插值类型: {interpolation_type}")
        print(f"运动时间: {duration}秒")
        print("注意：运动过程中可以按Ctrl+C安全停止")
        
        # 倒计时
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
        
        # 执行回零
        success = arm.home_to_zero(duration=duration, interpolation_type=interpolation_type)
        
        if success:
            print("\n🎉 回零测试成功！")
        else:
            print("\n⚠️ 回零精度不够，但运动已完成")
        
        # 显示最终位置
        final_pos = arm.get_current_positions_deg()
        print(f"\n最终位置: {[f'{p:.2f}°' for p in final_pos]}")
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
    finally:
        # 确保安全关闭
        try:
            arm.disable_all_motors()
        except:
            pass
        arm.close()
        print("测试结束")

if __name__ == "__main__":
    main()
