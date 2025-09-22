#!/usr/bin/env python3
"""
零点设置工具
用于设置IC ARM的零点位置（支持前8个电机：m1-m6为Damiao电机，m7-m8为HT电机，排除servo电机）
"""

from ic_arm_control.control.IC_ARM import ICARM
import time
import sys
MOTOR_LIST = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8']

def display_current_positions(arm, duration=None):
    """
    实时显示当前关节位置
    
    Args:
        arm: ICARM实例
        duration: 显示时长(秒)，None为无限制
    """
    import sys
    import select
    import termios
    import tty
    
    print("\n实时关节位置显示 (按'q'键停止并进入菜单):")
    print("移动机械臂到期望的零点位置...")
    print("格式: 关节名: 角度° | 关节名: 角度° ...")
    print("-" * 60)
    
    # 保存终端设置
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        # 设置终端为原始模式，不需要回车就能读取输入
        tty.setraw(sys.stdin.fileno())
        
        start_time = time.time()
        
        while True:
            # 非阻塞检查键盘输入
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == 'q':
                    print("\n检测到'q'键，退出显示...")
                    break
                
            # 检查时长限制
            if duration and (time.time() - start_time) > duration:
                break
                
            # 获取并显示当前位置
            try:
                current_positions = arm.get_positions_degrees()
                if current_positions is not None and len(current_positions) > 0:
                    position_str = " | ".join([
                        f"{name}: {current_positions[i]:.2f}°" 
                        for i, name in enumerate(MOTOR_LIST) if i < len(current_positions)
                    ])
                    print(f"\r{position_str}", end="", flush=True)
                else:
                    print(f"\r无法获取位置数据...", end="", flush=True)
            except Exception as e:
                print(f"\r读取位置出错: {e}", end="", flush=True)
                
            time.sleep(0.05)  # 20Hz刷新率
            
    except Exception as e:
        print(f"\n获取位置时出错: {e}")
        return False
    finally:
        # 恢复终端设置
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    return True

def confirm_action(message):
    """确认操作"""
    while True:
        response = input(f"{message} (y/n): ").lower().strip()
        if response in ['y', 'yes', '是']:
            return True
        elif response in ['n', 'no', '否']:
            return False
        else:
            print("请输入 y/yes/是 或 n/no/否")

def show_menu():
    """显示主菜单"""
    print("\n" + "="*50)
    print("选择操作:")
    print("1. 设置所有关节的零点")
    print("2. 设置单个关节的零点")
    print("3. 重新显示当前位置")
    print("4. 显示当前位置(一次)")
    print("5. 退出")
    print("="*50)

def main():
    """主函数：零点设置工具"""
    print("=" * 60)
    print("           IC ARM 零点设置工具")
    print("=" * 60)
    print("支持电机:")
    print("• m1-m6: Damiao电机")
    print("• m7-m8: HT电机")
    print("• 注意：servo电机(m9)不支持零点设置")
    print()
    print("注意事项:")
    print("• 请先手动移动机械臂到期望的零点位置")
    print("• 零点设置为软件零位，重启后需要重新设置")
    print("• 确保机械臂处于安全位置再进行零点设置")
    print("=" * 60)
    
    # 初始化机械臂
    try:
        print("\n正在连接机械臂...")
        arm = ICARM()
        print("✓ 机械臂连接成功")
    except Exception as e:
        print(f"✗ 机械臂连接失败: {e}")
        return
    
    try:
        # 首次显示当前位置
        if not display_current_positions(arm):
            print("无法获取位置信息，请检查连接")
            return
        
        while True:
            show_menu()
            choice = input("\n请选择 (1-5): ").strip()
            
            if choice == '1':
                # 设置所有关节零点
                print("\n准备设置所有关节零点...")
                
                # 显示当前位置
                try:
                    current_positions = arm.get_positions_degrees()
                    if current_positions is not None and len(current_positions) > 0:
                        print("当前关节位置:")
                        motor_names =MOTOR_LIST 
                        for i, name in enumerate(motor_names):
                            if i < len(current_positions):
                                print(f"  {name}: {current_positions[i]:.2f}°")
                            else:
                                print(f"  {name}: 无法读取")
                    else:
                        print("无法获取当前位置")
                except Exception as e:
                    print(f"获取位置时出错: {e}")
                
                if confirm_action("\n确认将当前位置设为所有关节的零点?"):
                    success = arm.set_all_zero_positions()
                    if success:
                        print("\n✓ 所有关节零点设置成功！")
                    else:
                        print("\n✗ 部分关节零点设置失败，请检查连接")
                else:
                    print("操作已取消")
                    
            elif choice == '2':
                # 设置单个关节零点
                print(f"\n可用关节: {MOTOR_LIST}")
                motor_name = input("请输入要设置零点的关节名称: ").strip()
                
                if motor_name in MOTOR_LIST :
                    # 显示该关节当前位置
                    try:
                        current_positions = arm.get_positions_degrees()
                        motor_names = MOTOR_LIST
                        motor_index = motor_names.index(motor_name)
                        if current_positions is not None and motor_index < len(current_positions):
                            print(f"{motor_name} 当前位置: {current_positions[motor_index]:.2f}°")
                        else:
                            print(f"{motor_name} 无法读取当前位置")
                    except Exception as e:
                        print(f"获取 {motor_name} 位置时出错: {e}")
                    
                    if confirm_action(f"\n确认将当前位置设为 {motor_name} 的零点?"):
                        success = arm.set_single_zero_position(motor_name)
                        if success:
                            print(f"\n✓ {motor_name} 零点设置成功！")
                        else:
                            print(f"\n✗ {motor_name} 零点设置失败")
                    else:
                        print("操作已取消")
                else:
                    print("✗ 无效的关节名称，请输入 m1, m2, m3, m4, m5 中的一个")
                    
            elif choice == '3':
                # 重新显示当前位置
                display_current_positions(arm)
                
            elif choice == '4':
                # 显示当前位置(一次)
                print("\n当前关节位置:")
                try:
                    current_positions = arm.get_positions_degrees()
                    if current_positions is not None and len(current_positions) > 0:
                        motor_names =MOTOR_LIST 
                        for i, name in enumerate(motor_names):
                            if i < len(current_positions):
                                print(f"  {name}: {current_positions[i]:.2f}°")
                            else:
                                print(f"  {name}: 无法读取")
                    else:
                        print("  无法获取位置数据")
                except Exception as e:
                    print(f"  获取位置时出错: {e}")
                    
            elif choice == '5':
                print("\n退出程序...")
                break
                
            else:
                print("✗ 无效选择，请输入 1-5")
    
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行错误: {e}")
    
    finally:
        # 确保安全关闭
        print("\n正在安全关闭机械臂...")
        try:
            arm.disable_all_motors()
            print("✓ 电机已禁用")
        except Exception as e:
            print(f"⚠ 禁用电机时出错: {e}")
        
        try:
            arm.close()
            print("✓ 连接已关闭")
        except Exception as e:
            print(f"⚠ 关闭连接时出错: {e}")
        
        print("程序已退出")

if __name__ == "__main__":
    main()
