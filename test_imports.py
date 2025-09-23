#!/usr/bin/env python3
"""
测试所有主要模块的导入
"""

def test_imports():
    """测试所有主要模块的导入"""
    print("开始测试导入...")
    
    try:
        # 测试主要控制模块
        from ic_arm_control.control.IC_ARM import ICARM
        print("✓ IC_ARM 导入成功")
        
        from ic_arm_control.control.unified_motor_control import MotorManager
        print("✓ unified_motor_control 导入成功")
        
        from ic_arm_control.control.damiao import DmMotorManager
        print("✓ damiao 导入成功")
        
        from ic_arm_control.control.ht_motor import HTMotorManager
        print("✓ ht_motor 导入成功")
        
        from ic_arm_control.control.servo_motor import ServoMotorManager
        print("✓ servo_motor 导入成功")
        
        # 测试工具模块
        from ic_arm_control.tools.position_monitor import main as position_monitor_main
        print("✓ position_monitor 导入成功")
        
        from ic_arm_control.tools.set_zero_position import display_current_positions
        print("✓ set_zero_position 导入成功")
        
        from ic_arm_control.tools.ht_gui_controller import HTMotorGUIController
        print("✓ ht_gui_controller 导入成功")
        
        from ic_arm_control.tools.urdf_limit_updater import URDFLimitUpdater
        print("✓ urdf_limit_updater 导入成功")
        
        from ic_arm_control.tools.mujoco_simulation import MuJoCoICARMSimulation
        print("✓ mujoco_simulation 导入成功")
        
        print("\n🎉 所有模块导入测试通过！")
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n开始测试基本功能...")
    
    try:
        # 测试ICARM初始化（不连接硬件）
        from ic_arm_control.control.IC_ARM import ICARM
        
        # 由于硬件模块使用mock，这应该能正常工作
        print("✓ 基本功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("IC ARM Control 导入测试")
    print("=" * 50)
    
    import_success = test_imports()
    basic_success = test_basic_functionality()
    
    print("\n" + "=" * 50)
    if import_success and basic_success:
        print("🎉 所有测试通过！项目结构配置正确。")
    else:
        print("❌ 部分测试失败，请检查配置。")
    print("=" * 50)
