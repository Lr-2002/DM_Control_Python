#!/usr/bin/env python3
"""
HT电机GUI控制器
基于统一电机控制接口，控制两个HT电机的角度
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import math
from typing import Optional

# 导入必要的模块
HARDWARE_AVAILABLE = False
try:
    # 尝试导入硬件相关模块
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'Python'))
    sys.path.append(os.path.dirname(__file__))
    
    from Python.ht_motor import HTMotorManager
    from Python.unified_motor_control import MotorManager, MotorInfo, MotorType
    
    # 尝试导入USB硬件接口
    try:
        # from Python.src.usb_class import USB_HW
        from Python.src.usb_class import usb_class as USB_HW
        HARDWARE_AVAILABLE = True
        print("硬件接口可用")
    except ImportError as e:
        print(f"硬件接口不可用: {e}")
        # 创建模拟的USB_HW类
       
except ImportError as e:
    print(f"导入模块失败: {e}")
    # 创建模拟类用于GUI测试
    class MotorType:
        HIGH_TORQUE = 2
    
    class MotorInfo:
        def __init__(self, motor_id, motor_type, can_id, name=""):
            self.motor_id = motor_id
            self.motor_type = motor_type
            self.can_id = can_id
            self.name = name
    
    class USB_HW:
        def init(self): return True
        def close(self): pass
        def setFrameCallback(self, callback): pass
        def fdcanFrameSend(self, data, can_id): pass
    
    class MockMotor:
        def __init__(self):
            self.position = 0.0
            self.velocity = 0.0
            self.torque = 0.0
        
        def enable(self): return True
        def disable(self): return True
        def set_zero(self): return True
        def set_command(self, pos, vel, kp, kd, tau): return True
        def update_state(self): return True
        def get_position(self): return self.position
        def get_velocity(self): return self.velocity
        def get_torque(self): return self.torque
    
    class MotorManager:
        def __init__(self, usb_hw):
            self.motors = {}
        
        def add_ht_protocol(self, ht_manager): pass
        def add_motor(self, motor_id, motor_type, motor_info, **kwargs):
            self.motors[motor_id] = MockMotor()
            return True
        
        def get_motor(self, motor_id):
            return self.motors.get(motor_id)
        
        def update_all_states(self): return True
        def send_all_commands(self): return True
    
    class HTMotorManager:
        def __init__(self, usb_hw, as_sub_module=False): pass


class HTMotorGUIController:
    """HT电机GUI控制器"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HT电机双轴控制器")
        self.root.geometry("800x600")
        
        # 电机系统初始化
        self.usb_hw = None
        self.motor_manager = None
        self.ht_manager = None
        self.motor1 = None
        self.motor2 = None
        
        # 控制参数
        self.motor1_target_angle = tk.DoubleVar(value=0.0)
        self.motor2_target_angle = tk.DoubleVar(value=0.0)
        self.motor1_current_angle = tk.DoubleVar(value=0.0)
        self.motor2_current_angle = tk.DoubleVar(value=0.0)
        
        # PID参数
        self.kp = tk.DoubleVar(value=50.0)
        self.kd = tk.DoubleVar(value=5.0)
        
        # 控制状态
        self.is_connected = False
        self.is_enabled = False
        self.monitoring_thread = None
        self.stop_monitoring = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 连接控制区域
        connection_frame = ttk.LabelFrame(main_frame, text="连接控制", padding="10")
        connection_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.connect_btn = ttk.Button(connection_frame, text="连接电机", command=self.connect_motors)
        self.connect_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.disconnect_btn = ttk.Button(connection_frame, text="断开连接", command=self.disconnect_motors, state="disabled")
        self.disconnect_btn.grid(row=0, column=1, padx=(0, 10))
        
        self.enable_btn = ttk.Button(connection_frame, text="使能电机", command=self.enable_motors, state="disabled")
        self.enable_btn.grid(row=0, column=2, padx=(0, 10))
        
        self.disable_btn = ttk.Button(connection_frame, text="失能电机", command=self.disable_motors, state="disabled")
        self.disable_btn.grid(row=0, column=3, padx=(0, 10))
        
        self.zero_btn = ttk.Button(connection_frame, text="设置零位", command=self.set_zero_position, state="disabled")
        self.zero_btn.grid(row=0, column=4)
        
        # 状态显示
        self.status_label = ttk.Label(connection_frame, text="状态: 未连接", foreground="red")
        self.status_label.grid(row=1, column=0, columnspan=5, pady=(10, 0))
        
        # PID参数设置
        pid_frame = ttk.LabelFrame(main_frame, text="PID参数", padding="10")
        pid_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(pid_frame, text="Kp:").grid(row=0, column=0, sticky=tk.W)
        kp_entry = ttk.Entry(pid_frame, textvariable=self.kp, width=10)
        kp_entry.grid(row=0, column=1, padx=(5, 20))
        
        ttk.Label(pid_frame, text="Kd:").grid(row=0, column=2, sticky=tk.W)
        kd_entry = ttk.Entry(pid_frame, textvariable=self.kd, width=10)
        kd_entry.grid(row=0, column=3, padx=(5, 0))
        
        # 电机1控制区域
        motor1_frame = ttk.LabelFrame(main_frame, text="电机1控制", padding="10")
        motor1_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # 电机1角度控制
        ttk.Label(motor1_frame, text="目标角度 (度):").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        motor1_angle_frame = ttk.Frame(motor1_frame)
        motor1_angle_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        motor1_scale = ttk.Scale(motor1_angle_frame, from_=-180, to=180, 
                               variable=self.motor1_target_angle, orient=tk.HORIZONTAL, length=300)
        motor1_scale.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        motor1_entry = ttk.Entry(motor1_angle_frame, textvariable=self.motor1_target_angle, width=10)
        motor1_entry.grid(row=0, column=1, padx=(10, 0))
        
        # 电机1状态显示
        ttk.Label(motor1_frame, text="当前角度:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.motor1_pos_label = ttk.Label(motor1_frame, text="0.00°", font=("Arial", 12, "bold"))
        self.motor1_pos_label.grid(row=3, column=0, sticky=tk.W)
        
        ttk.Label(motor1_frame, text="当前速度:").grid(row=4, column=0, sticky=tk.W, pady=(5, 0))
        self.motor1_vel_label = ttk.Label(motor1_frame, text="0.00 rad/s")
        self.motor1_vel_label.grid(row=5, column=0, sticky=tk.W)
        
        ttk.Label(motor1_frame, text="当前力矩:").grid(row=6, column=0, sticky=tk.W, pady=(5, 0))
        self.motor1_torque_label = ttk.Label(motor1_frame, text="0.00 Nm")
        self.motor1_torque_label.grid(row=7, column=0, sticky=tk.W)
        
        # 电机1控制按钮
        motor1_btn_frame = ttk.Frame(motor1_frame)
        motor1_btn_frame.grid(row=8, column=0, pady=(20, 0))
        
        ttk.Button(motor1_btn_frame, text="移动到目标", 
                  command=lambda: self.move_motor_to_angle(1)).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(motor1_btn_frame, text="停止", 
                  command=lambda: self.stop_motor(1)).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(motor1_btn_frame, text="读取角度", 
                  command=lambda: self.read_motor_angle(1)).grid(row=0, column=2)
        
        # 电机2控制区域
        motor2_frame = ttk.LabelFrame(main_frame, text="电机2控制", padding="10")
        motor2_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        # 电机2角度控制
        ttk.Label(motor2_frame, text="目标角度 (度):").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        motor2_angle_frame = ttk.Frame(motor2_frame)
        motor2_angle_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        motor2_scale = ttk.Scale(motor2_angle_frame, from_=-180, to=180, 
                               variable=self.motor2_target_angle, orient=tk.HORIZONTAL, length=300)
        motor2_scale.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        motor2_entry = ttk.Entry(motor2_angle_frame, textvariable=self.motor2_target_angle, width=10)
        motor2_entry.grid(row=0, column=1, padx=(10, 0))
        
        # 电机2状态显示
        ttk.Label(motor2_frame, text="当前角度:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.motor2_pos_label = ttk.Label(motor2_frame, text="0.00°", font=("Arial", 12, "bold"))
        self.motor2_pos_label.grid(row=3, column=0, sticky=tk.W)
        
        ttk.Label(motor2_frame, text="当前速度:").grid(row=4, column=0, sticky=tk.W, pady=(5, 0))
        self.motor2_vel_label = ttk.Label(motor2_frame, text="0.00 rad/s")
        self.motor2_vel_label.grid(row=5, column=0, sticky=tk.W)
        
        ttk.Label(motor2_frame, text="当前力矩:").grid(row=6, column=0, sticky=tk.W, pady=(5, 0))
        self.motor2_torque_label = ttk.Label(motor2_frame, text="0.00 Nm")
        self.motor2_torque_label.grid(row=7, column=0, sticky=tk.W)
        
        # 电机2控制按钮
        motor2_btn_frame = ttk.Frame(motor2_frame)
        motor2_btn_frame.grid(row=8, column=0, pady=(20, 0))
        
        ttk.Button(motor2_btn_frame, text="移动到目标", 
                  command=lambda: self.move_motor_to_angle(2)).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(motor2_btn_frame, text="停止", 
                  command=lambda: self.stop_motor(2)).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(motor2_btn_frame, text="读取角度", 
                  command=lambda: self.read_motor_angle(2)).grid(row=0, column=2)
        
        # 全局控制按钮
        global_frame = ttk.Frame(main_frame)
        global_frame.grid(row=3, column=0, columnspan=2, pady=(20, 0))
        
        ttk.Button(global_frame, text="同时移动两个电机", 
                  command=self.move_both_motors).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(global_frame, text="停止所有电机", 
                  command=self.stop_all_motors).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(global_frame, text="回到零位", 
                  command=self.move_to_zero).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(global_frame, text="读取所有角度", 
                  command=self.read_all_angles).grid(row=0, column=3)
        
        # 配置网格权重
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        motor1_frame.columnconfigure(0, weight=1)
        motor2_frame.columnconfigure(0, weight=1)
        motor1_angle_frame.columnconfigure(0, weight=1)
        motor2_angle_frame.columnconfigure(0, weight=1)
        
    def connect_motors(self):
        """连接电机"""
        try:
            # 初始化USB硬件 - 按照IC_ARM.py的方式
            from usb_hw_wrapper import USBHardwareWrapper
            usb_hw = USB_HW(1000000, 5000000, "F561E08C892274DB09496BCC1102DBC5")
            self.usb_hw = USBHardwareWrapper(usb_hw)

            # 创建电机管理器
            self.motor_manager = MotorManager(self.usb_hw)
            
            # 创建HT电机管理器 - 按照IC_ARM.py的方式
            self.ht_manager = HTMotorManager(self.usb_hw)
            self.motor_manager.add_ht_protocol(self.ht_manager)
            
            # 添加两个HT电机 - 使用正确的配置
            motor1_info = MotorInfo(
                motor_id=1,
                motor_type=MotorType.HIGH_TORQUE,
                can_id=0x8094,  # HT电机使用固定的发送ID 0x8094
                master_id=None,
                name="HT_Motor_1",
                kp=50.0,
                kd=5.0,
                torque_offset=0.0,
                limits=[12.5, 50.0, 20.0]  # HT电机扭矩限制更高
            )
            
            motor2_info = MotorInfo(
                motor_id=2,
                motor_type=MotorType.HIGH_TORQUE,
                can_id=0x8094,  # HT电机使用固定的发送ID 0x8094
                master_id=None,
                name="HT_Motor_2",
                kp=50.0,
                kd=5.0,
                torque_offset=0.0,
                limits=[12.5, 50.0, 20.0]  # HT电机扭矩限制更高
            )
            
            # 添加电机到管理器 - HT电机ID从1开始
            if not self.motor_manager.add_motor(1, 'ht', motor1_info, ht_motor_id=1):
                raise Exception("电机1添加失败")
                
            if not self.motor_manager.add_motor(2, 'ht', motor2_info, ht_motor_id=2):
                raise Exception("电机2添加失败")
            
            # 获取电机实例
            self.motor1 = self.motor_manager.get_motor(1)
            self.motor2 = self.motor_manager.get_motor(2)
            
            if not self.motor1 or not self.motor2:
                raise Exception("获取电机实例失败")
            
            self.is_connected = True
            self.update_ui_state()
            self.status_label.config(text="状态: 已连接", foreground="green")
            
            # 启动监控线程
            self.start_monitoring()
            
            messagebox.showinfo("成功", "电机连接成功！")
            
        except Exception as e:
            messagebox.showerror("错误", f"连接失败: {str(e)}")
            self.disconnect_motors()
    
    def disconnect_motors(self):
        """断开电机连接"""
        try:
            self.stop_monitoring()
            
            if self.is_enabled:
                self.disable_motors()
            
            if self.usb_hw:
                self.usb_hw.close()
            
            self.usb_hw = None
            self.motor_manager = None
            self.ht_manager = None
            self.motor1 = None
            self.motor2 = None
            
            self.is_connected = False
            self.update_ui_state()
            self.status_label.config(text="状态: 未连接", foreground="red")
            
        except Exception as e:
            messagebox.showerror("错误", f"断开连接时出错: {str(e)}")
    
    def enable_motors(self):
        """使能电机"""
        try:
            if not self.is_connected:
                messagebox.showerror("错误", "请先连接电机")
                return
            
            # 使能两个电机
            if self.motor1.enable() and self.motor2.enable():
                self.is_enabled = True
                self.update_ui_state()
                messagebox.showinfo("成功", "电机使能成功！")
            else:
                messagebox.showerror("错误", "电机使能失败")
                
        except Exception as e:
            messagebox.showerror("错误", f"使能电机时出错: {str(e)}")
    
    def disable_motors(self):
        """失能电机"""
        try:
            if self.motor1:
                self.motor1.disable()
            if self.motor2:
                self.motor2.disable()
            
            self.is_enabled = False
            self.update_ui_state()
            
        except Exception as e:
            messagebox.showerror("错误", f"失能电机时出错: {str(e)}")
    
    def set_zero_position(self):
        """设置零位"""
        try:
            if not self.is_connected:
                messagebox.showerror("错误", "请先连接电机")
                return
            
            result = messagebox.askyesno("确认", "确定要设置当前位置为零位吗？\n设置后需要重启电机。")
            if result:
                if self.motor1.set_zero() and self.motor2.set_zero():
                    messagebox.showinfo("成功", "零位设置成功！请重启电机。")
                else:
                    messagebox.showerror("错误", "零位设置失败")
                    
        except Exception as e:
            messagebox.showerror("错误", f"设置零位时出错: {str(e)}")
    
    def move_motor_to_angle(self, motor_num):
        """移动指定电机到目标角度"""
        try:
            if not self.is_enabled:
                messagebox.showerror("错误", "请先使能电机")
                return
            
            if motor_num == 1:
                target_rad = math.radians(self.motor1_target_angle.get())
                motor = self.motor1
            else:
                target_rad = math.radians(self.motor2_target_angle.get())
                motor = self.motor2
            
            # 使用MIT控制模式 - HT电机的正确方式
            kp = self.kp.get()
            kd = self.kd.get()
            
            # 设置命令到缓存
            motor.set_command(target_rad, 0.0, kp, kd, 0.0)
            
            # 批量发送所有HT电机命令（一拖多方式）
            self.motor_manager.send_all_commands()
            
        except Exception as e:
            messagebox.showerror("错误", f"移动电机时出错: {str(e)}")
    
    def stop_motor(self, motor_num):
        """停止指定电机 - HT电机需要批量停止"""
        try:
            if motor_num == 1:
                motor = self.motor1
            else:
                motor = self.motor2
            
            if motor:
                # HT电机停止：获取当前位置并保持
                motor.update_state()
                current_pos = motor.get_position()
                # 设置停止命令到缓存（保持当前位置，速度为0）
                motor.set_command(current_pos, 0.0, self.kp.get(), self.kd.get(), 0.0)
                # 批量发送命令（HT电机一拖多控制）
                self.motor_manager.send_all_commands()
                
        except Exception as e:
            messagebox.showerror("错误", f"停止电机时出错: {str(e)}")
    
    def move_both_motors(self):
        """同时移动两个电机 - HT电机一拖多控制的正确实现"""
        try:
            if not self.is_enabled:
                messagebox.showerror("错误", "请先使能电机")
                return
            
            target1_rad = math.radians(self.motor1_target_angle.get())
            target2_rad = math.radians(self.motor2_target_angle.get())
            
            kp = self.kp.get()
            kd = self.kd.get()
            
            # HT电机一拖多控制：先缓存所有电机命令
            self.motor1.set_command(target1_rad, 0.0, kp, kd, 0.0)
            self.motor2.set_command(target2_rad, 0.0, kp, kd, 0.0)
            
            # 一次性批量发送所有命令（通过0x8094发送）
            self.motor_manager.send_all_commands()
            
        except Exception as e:
            messagebox.showerror("错误", f"移动电机时出错: {str(e)}")
    
    def stop_all_motors(self):
        """停止所有电机 - HT电机批量停止的正确实现"""
        try:
            if not self.motor1 or not self.motor2:
                messagebox.showerror("错误", "电机未初始化")
                return
            
            # 更新所有电机状态
            self.motor_manager.update_all_states()
            
            # 获取当前位置并设置停止命令
            current_pos1 = self.motor1.get_position()
            current_pos2 = self.motor2.get_position()
            
            kp = self.kp.get()
            kd = self.kd.get()
            
            # 缓存所有电机的停止命令
            self.motor1.set_command(current_pos1, 0.0, kp, kd, 0.0)
            self.motor2.set_command(current_pos2, 0.0, kp, kd, 0.0)
            
            # 一次性批量发送停止命令
            self.motor_manager.send_all_commands()
            
        except Exception as e:
            messagebox.showerror("错误", f"停止电机时出错: {str(e)}")
    
    def move_to_zero(self):
        """回到零位 - HT电机批量回零"""
        try:
            # 设置目标角度为0
            self.motor1_target_angle.set(0.0)
            self.motor2_target_angle.set(0.0)
            
            # 使用批量控制回零
            self.move_both_motors()
        except Exception as e:
            messagebox.showerror("错误", f"回零时出错: {str(e)}")
    
    def read_motor_angle(self, motor_num):
        """读取指定电机的当前角度"""
        if not self.is_connected:
            messagebox.showerror("错误", "请先连接电机")
            return
        
        try:
            if motor_num == 1:
                motor = self.motor1
                motor_name = "电机1"
            else:
                motor = self.motor2
                motor_name = "电机2"
            
            if not motor:
                messagebox.showerror("错误", f"{motor_name}未初始化")
                return
            
            # 强制更新电机状态
            motor.update_state()
            
            # 获取当前角度
            current_rad = motor.get_position()
            current_deg = math.degrees(current_rad)
            
            # 显示角度信息
            messagebox.showinfo(
                f"{motor_name}角度读取", 
                f"{motor_name}当前角度:\n"
                f"角度: {current_deg:.2f}°\n"
                f"弧度: {current_rad:.4f} rad\n"
                f"速度: {motor.get_velocity():.2f} rad/s\n"
                f"力矩: {motor.get_torque():.2f} Nm"
            )
            
            # 更新目标角度为当前角度（可选）
            if motor_num == 1:
                self.motor1_target_angle.set(current_deg)
            else:
                self.motor2_target_angle.set(current_deg)
                
        except Exception as e:
            messagebox.showerror("错误", f"读取{motor_name}角度失败: {str(e)}")
    
    def read_all_angles(self):
        """读取所有电机的当前角度"""
        if not self.is_connected:
            messagebox.showerror("错误", "请先连接电机")
            return
        
        try:
            if not self.motor1 or not self.motor2:
                messagebox.showerror("错误", "电机未初始化")
                return
            
            # 强制更新所有电机状态
            self.motor_manager.update_all_states()
            
            # 获取角度信息
            pos1_rad = self.motor1.get_position()
            pos1_deg = math.degrees(pos1_rad)
            vel1 = self.motor1.get_velocity()
            torque1 = self.motor1.get_torque()
            
            pos2_rad = self.motor2.get_position()
            pos2_deg = math.degrees(pos2_rad)
            vel2 = self.motor2.get_velocity()
            torque2 = self.motor2.get_torque()
            
            # 显示所有角度信息
            info_text = (
                "所有电机角度读取结果:\n\n"
                f"电机1:\n"
                f"  角度: {pos1_deg:.2f}° ({pos1_rad:.4f} rad)\n"
                f"  速度: {vel1:.2f} rad/s\n"
                f"  力矩: {torque1:.2f} Nm\n\n"
                f"电机2:\n"
                f"  角度: {pos2_deg:.2f}° ({pos2_rad:.4f} rad)\n"
                f"  速度: {vel2:.2f} rad/s\n"
                f"  力矩: {torque2:.2f} Nm"
            )
            
            messagebox.showinfo("所有电机角度", info_text)
            
            # 更新目标角度为当前角度
            self.motor1_target_angle.set(pos1_deg)
            self.motor2_target_angle.set(pos2_deg)
            
        except Exception as e:
            messagebox.showerror("错误", f"读取电机角度失败: {str(e)}")
    
    def update_ui_state(self):
        """更新UI状态"""
        if self.is_connected:
            self.connect_btn.config(state="disabled")
            self.disconnect_btn.config(state="normal")
            self.enable_btn.config(state="normal" if not self.is_enabled else "disabled")
            self.disable_btn.config(state="normal" if self.is_enabled else "disabled")
            self.zero_btn.config(state="normal")
        else:
            self.connect_btn.config(state="normal")
            self.disconnect_btn.config(state="disabled")
            self.enable_btn.config(state="disabled")
            self.disable_btn.config(state="disabled")
            self.zero_btn.config(state="disabled")
    
    def start_monitoring(self):
        """启动监控线程"""
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(target=self.monitor_motors, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """停止监控线程"""
        self.stop_monitoring = True
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
    
    def monitor_motors(self):
        """监控电机状态"""
        while not self.stop_monitoring and self.is_connected:
            try:
                if self.motor1 and self.motor2:
                    # 更新电机状态
                    self.motor_manager.update_all_states()
                    
                    # 更新UI显示
                    self.root.after(0, self.update_motor_display)
                
                time.sleep(0.1)  # 100ms更新间隔
                
            except Exception as e:
                print(f"监控线程错误: {e}")
                time.sleep(0.5)
    
    def update_motor_display(self):
        """更新电机显示信息"""
        try:
            if self.motor1:
                pos1_deg = math.degrees(self.motor1.get_position())
                vel1 = self.motor1.get_velocity()
                torque1 = self.motor1.get_torque()
                
                self.motor1_pos_label.config(text=f"{pos1_deg:.2f}°")
                self.motor1_vel_label.config(text=f"{vel1:.2f} rad/s")
                self.motor1_torque_label.config(text=f"{torque1:.2f} Nm")
                
                self.motor1_current_angle.set(pos1_deg)
            
            if self.motor2:
                pos2_deg = math.degrees(self.motor2.get_position())
                vel2 = self.motor2.get_velocity()
                torque2 = self.motor2.get_torque()
                
                self.motor2_pos_label.config(text=f"{pos2_deg:.2f}°")
                self.motor2_vel_label.config(text=f"{vel2:.2f} rad/s")
                self.motor2_torque_label.config(text=f"{torque2:.2f} Nm")
                
                self.motor2_current_angle.set(pos2_deg)
                
        except Exception as e:
            print(f"更新显示时出错: {e}")
    
    def on_closing(self):
        """窗口关闭事件"""
        try:
            self.disconnect_motors()
        except:
            pass
        finally:
            self.root.destroy()
    
    def run(self):
        """运行GUI"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


def main():
    """主函数"""
    app = HTMotorGUIController()
    app.run()


if __name__ == "__main__":
    main()