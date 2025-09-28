
#!/usr/bin/env python3
"""
IC_ARM MLP重力补偿使用示例
"""

from ic_arm_control.control.IC_ARM import ICARM

def demo_mlp_gravity_compensation():
    """演示MLP重力补偿功能"""

    print("=== IC_ARM MLP重力补偿演示 ===")

    # 1. 使用MLP重力补偿初始化
    arm = ICARM(
        device_sn="F561E08C892274DB09496BCC1102DBC5",
        debug=True,
        gc=True,                    # 启用重力补偿
        gc_type="mlp",               # 使用MLP重力补偿
        enable_buffered_control=True,
        control_freq=300
    )

    # 2. 连接设备
    if not arm.connect():
        print("❌ 设备连接失败")
        return

    try:
        # 3. 启动设备
        if not arm.start_device():
            print("❌ 设备启动失败")
            return

        # 4. 测试MLP重力补偿
        print("\n1. 测试MLP重力补偿计算...")
        arm.refresh_all_states()
        compensation_torque = arm.cal_gravity()
        print(f"重力补偿力矩: {compensation_torque}")

        # 5. 启动重力补偿模式
        print("\n2. 启动重力补偿模式...")
        arm.start_gravity_compensation_mode(duration=10, update_rate=100)

        # 6. 查看性能统计
        print("\n3. 重力补偿性能统计:")
        arm.print_gravity_compensation_summary()

        # 7. 动态切换补偿模式
        print("\n4. 切换到静态重力补偿...")
        if arm.switch_to_static_gravity_compensation():
            print("✅ 成功切换到静态重力补偿")

            # 测试静态补偿
            static_compensation = arm.cal_gravity()
            print(f"静态补偿力矩: {static_compensation}")

        print("\n5. 切换回MLP重力补偿...")
        if arm.switch_to_mlp_gravity_compensation():
            print("✅ 成功切换到MLP重力补偿")

            # 再次测试MLP补偿
            mlp_compensation = arm.cal_gravity()
            print(f"MLP补偿力矩: {mlp_compensation}")

    except KeyboardInterrupt:
        print("\n⏹️  用户中断")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
    finally:
        # 关闭设备
        arm.close()
        print("\n🔌 设备已关闭")


def compare_gravity_compensation_methods():
    """比较不同重力补偿方法"""

    print("=== 重力补偿方法比较 ===")

    # 创建IC_ARM实例
    arm = ICARM(debug=True)

    # 不连接硬件，只测试算法
    test_positions = [0.0, 0.5, 1.0, 0.2, -0.3, 0.8]

    print(f"测试位置: {test_positions}")

    # 1. 无重力补偿
    arm.gc_flag = False
    no_gc_torque = arm.cal_gravity()
    print(f"\n1. 无重力补偿: {no_gc_torque}")

    # 2. 静态重力补偿
    arm.gc_flag = True
    arm.gc_type = "static"
    arm.switch_to_static_gravity_compensation()
    static_torque = arm.cal_gravity()
    print(f"2. 静态重力补偿: {static_torque}")

    # 3. MLP重力补偿
    arm.gc_type = "mlp"
    arm.switch_to_mlp_gravity_compensation()
    mlp_torque = arm.cal_gravity()
    print(f"3. MLP重力补偿: {mlp_torque}")

    # 4. 比较结果
    print(f"\n=== 结果比较 ===")
    print(f"静态补偿范围: [{np.min(static_torque):.3f}, {np.max(static_torque):.3f}] Nm")
    print(f"MLP补偿范围:   [{np.min(mlp_torque):.3f}, {np.max(mlp_torque):.3f}] Nm")
    print(f"差异: {np.linalg.norm(static_torque - mlp_torque):.3f} Nm")


if __name__ == "__main__":
    # 取消注释以运行演示
    # demo_mlp_gravity_compensation()
    compare_gravity_compensation_methods()
