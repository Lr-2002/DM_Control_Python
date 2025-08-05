#!/usr/bin/env python3
"""
零点设置工具
用于设置IC ARM的零点位置
"""

from IC_ARM import ICARM
import time

def main():
    """主函数：零点设置工具"""
    print("=== IC ARM 零点设置工具 ===")
    print("请先手动移动机械臂到期望的零点位置")
    print()
    
    # 初始化机械臂
    arm = ICARM()
    
    try:
        print("连接到机械臂...")
        
        # 实时显示当前位置
        print("\n实时关节位置显示 (按Ctrl+C停止并进入菜单):")
        print("移动机械臂到期望的零点位置...")
        print()
        
        # 实时位置显示循环
        try:
            while True:
                # 获取并显示当前位置
                current_positions = arm.get_positions_only()
                position_str = " | ".join([f"{name}: {pos['deg']:.2f}°" if pos['deg'] is not None else f"{name}: --" 
                                         for name, pos in current_positions.items()])
                print(f"\r{position_str}", end="", flush=True)
                
                time.sleep(0.02)  # 1秒刷新一次
                
        except KeyboardInterrupt:
            print("\n")
        
        print("\n\n选择操作:")
        print("1. 设置所有关节的零点")
        print("2. 设置单个关节的零点")
        print("3. 退出")
        
        choice = input("\n请选择 (1/2/3): ")
        
        if choice == '1':
            # 设置所有关节零点
            success = arm.set_all_zero_positions()
            if success:
                print("\n✓ 所有关节零点设置成功！")
            else:
                print("\n✗ 部分关节零点设置失败，请检查连接")
                
        elif choice == '2':
            # 设置单个关节零点
            print("\n可用关节: m1, m2, m3, m4, m5")
            motor_name = input("请输入要设置零点的关节名称: ")
            
            if motor_name in ['m1', 'm2', 'm3', 'm4', 'm5']:
                success = arm.set_single_zero_position(motor_name)
                if success:
                    print(f"\n✓ {motor_name} 零点设置成功！")
                else:
                    print(f"\n✗ {motor_name} 零点设置失败")
            else:
                print("无效的关节名称")
                
        elif choice == '3':
            print("退出程序")
            
        else:
            print("无效选择")
    
    except Exception as e:
        print(f"错误: {e}")
    
    finally:
        # 确保安全关闭
        try:
            arm.disable_all_motors()
        except:
            pass
        arm.close()

if __name__ == "__main__":
    main()
