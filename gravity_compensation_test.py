#!/usr/bin/env python3
"""
测试新的重力补偿功能
使用辨识出的动力学模型进行重力+科里奥利补偿
"""

import time
from IC_ARM import ICARM

def test_gravity_compensation():
    """测试重力补偿模式"""
    print("=== 测试重力补偿功能 ===")
    
    try:
        # 初始化机械臂，启用重力补偿
        print("初始化机械臂...")
        arm = ICARM(gc=True)
        
        print("\n1. 显示当前位置:")
        current_pos = arm.get_positions_degrees()
        print(f"当前位置: {[f'{p:.1f}°' for p in current_pos]}")
        
        print("\n2. 启动重力补偿模式...")
        print("机械臂将使用重力+科里奥利补偿保持位置")
        print("特点:")
        print("  - 使用辨识的动力学模型")
        print("  - kp=0, kd=0 (纯力矩控制)")
        print("  - 实时计算补偿力矩")
        print("可以手动轻推机械臂测试补偿效果")
        print("按 Ctrl+C 停止")
        
        # 运行重力补偿模式，无限时长
        arm.start_gravity_compensation_mode(duration=None, update_rate=100)
        
        print("\n3. 测试完成")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            arm.close()
        except:
            pass

def test_gravity_compensation_short():
    """短时间测试重力补偿"""
    print("=== 短时间重力补偿测试 ===")
    
    try:
        arm = ICARM(gc=True)
        
        print("运行10秒重力补偿测试...")
        arm.start_gravity_compensation_mode(duration=10, update_rate=50)
        
    except Exception as e:
        print(f"测试失败: {e}")
    finally:
        try:
            arm.close()
        except:
            pass

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "short":
        test_gravity_compensation_short()
    else:
        test_gravity_compensation()
