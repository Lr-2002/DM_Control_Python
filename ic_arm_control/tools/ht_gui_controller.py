#!/usr/bin/env python3
"""
HT电机GUI控制器
基于统一电机控制接口，控制两个HT电机的角度
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import math
import json
import os
from datetime import datetime
from typing import Optional, List, Dict

# 导入必要的模块
HARDWARE_AVAILABLE = False
try:
    # 尝试导入硬件相关模块
    import sys
    import os
    from ic_arm_control.control.ht_motor import HTMotorManager
    from ic_arm_control.control.unified_motor_control import MotorManager, MotorInfo, MotorType
    
    # 尝试导入USB硬件接口
    try:
        # from Python.src.usb_class import USB_HW
        from ic_arm_control.control.src import usb_class as USB_HW
        HARDWARE_AVAILABLE = True
        print("硬件接口可用")
    except ImportError as e:
        print(f"硬件接口不可用: {e}")
        # 创建模拟的USB_HW类
       
except ImportError as e:
    print(f"导入模块失败: {e}")
    pass
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
        
        # 录制回放功能
        self.is_recording = False
        self.is_playing = False
        self.recorded_trajectory = []
        self.recording_start_time = None
        self.playback_thread = None
        self.stop_playback = False
        self.recording_interval = 0.005  # 5ms录制间隔 (200Hz)
        
        # 连续控制功能
        self.is_continuous_control = False
        self.continuous_control_thread = None
        self.stop_continuous_control = False
        self.last_target1 = 0.0
        self.last_target2 = 0.0
        
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
        
        ttk.Button(motor1_btn_frame, text="停止", 
                  command=lambda: self.stop_motor(1)).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(motor1_btn_frame, text="读取角度", 
                  command=lambda: self.read_motor_angle(1)).grid(row=0, column=1)
        
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
        
        ttk.Button(motor2_btn_frame, text="停止", 
                  command=lambda: self.stop_motor(2)).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(motor2_btn_frame, text="读取角度", 
                  command=lambda: self.read_motor_angle(2)).grid(row=0, column=1)
        
        # 全局控制按钮
        global_frame = ttk.Frame(main_frame)
        global_frame.grid(row=3, column=0, columnspan=2, pady=(20, 0))
        
        self.move_btn = ttk.Button(global_frame, text="开始连续控制", 
                                 command=self.toggle_continuous_control, 
                                 style="Accent.TButton")
        self.move_btn.grid(row=0, column=0, padx=(0, 10))
        ttk.Button(global_frame, text="停止所有电机", 
                  command=self.stop_all_motors).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(global_frame, text="回到零位", 
                  command=self.move_to_zero).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(global_frame, text="读取所有角度", 
                  command=self.read_all_angles).grid(row=0, column=3)
        
        # 录制回放控制区域
        record_frame = ttk.LabelFrame(main_frame, text="录制回放控制", padding="10")
        record_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 录制控制按钮
        record_btn_frame = ttk.Frame(record_frame)
        record_btn_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.record_btn = ttk.Button(record_btn_frame, text="开始录制", 
                                   command=self.start_recording, state="disabled")
        self.record_btn.grid(row=0, column=0, padx=(0, 5))
        
        self.stop_record_btn = ttk.Button(record_btn_frame, text="停止录制", 
                                        command=self.stop_recording, state="disabled")
        self.stop_record_btn.grid(row=0, column=1, padx=(0, 5))
        
        self.play_btn = ttk.Button(record_btn_frame, text="开始回放", 
                                 command=self.start_playback, state="disabled")
        self.play_btn.grid(row=0, column=2, padx=(0, 5))
        
        self.stop_play_btn = ttk.Button(record_btn_frame, text="停止回放", 
                                      command=self.stop_playback, state="disabled")
        self.stop_play_btn.grid(row=0, column=3, padx=(0, 5))
        
        # 文件操作按钮
        file_btn_frame = ttk.Frame(record_frame)
        file_btn_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(file_btn_frame, text="保存轨迹", 
                  command=self.save_trajectory).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(file_btn_frame, text="加载轨迹", 
                  command=self.load_trajectory).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(file_btn_frame, text="清除轨迹", 
                  command=self.clear_trajectory).grid(row=0, column=2, padx=(0, 5))
        
        # 录制状态显示
        self.record_status_label = ttk.Label(record_frame, text="状态: 未录制", foreground="blue")
        self.record_status_label.grid(row=2, column=0, pady=(10, 0))
        
        # 轨迹信息显示
        self.trajectory_info_label = ttk.Label(record_frame, text="轨迹点数: 0")
        self.trajectory_info_label.grid(row=3, column=0, pady=(5, 0))
        
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
                motor_id=7,
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
                motor_id=8,
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
            if not self.motor_manager.add_motor(7, 'ht', motor1_info, ht_motor_id=7):
                raise Exception("电机1添加失败")
                
            if not self.motor_manager.add_motor(8, 'ht', motor2_info, ht_motor_id=8):
                raise Exception("电机2添加失败")
            
            # 获取电机实例
            self.motor1 = self.motor_manager.get_motor(7)
            self.motor2 = self.motor_manager.get_motor(8)
            
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
    
    def toggle_continuous_control(self):
        """切换连续控制模式"""
        try:
            if not self.is_enabled:
                messagebox.showerror("错误", "请先使能电机")
                return
            
            if self.is_continuous_control:
                # 停止连续控制
                self.stop_continuous_control_mode()
            else:
                # 开始连续控制
                self.start_continuous_control_mode()
                
        except Exception as e:
            messagebox.showerror("错误", f"切换连续控制模式时出错: {str(e)}")
    
    def start_continuous_control_mode(self):
        """开始连续控制模式"""
        try:
            self.is_continuous_control = True
            self.stop_continuous_control = False
            
            # 记录初始目标角度
            self.last_target1 = self.motor1_target_angle.get()
            self.last_target2 = self.motor2_target_angle.get()
            
            # 更新UI状态
            self.update_ui_state()
            
            # 启动连续控制线程
            self.continuous_control_thread = threading.Thread(target=self.continuous_control_loop, daemon=True)
            self.continuous_control_thread.start()
            
            print("连续控制模式已启动")
            
        except Exception as e:
            messagebox.showerror("错误", f"启动连续控制失败: {str(e)}")
    
    def stop_continuous_control_mode(self):
        """停止连续控制模式"""
        try:
            self.stop_continuous_control = True
            self.is_continuous_control = False
            
            # 更新UI状态
            self.update_ui_state()
            
            print("连续控制模式已停止")
            
        except Exception as e:
            messagebox.showerror("错误", f"停止连续控制失败: {str(e)}")
    
    def continuous_control_loop(self):
        """连续控制循环线程"""
        try:
            # 确保电机已初始化
            if not self.motor1 or not self.motor2:
                print("错误: 电机未初始化")
                return
                
            print(f"连续控制开始: 初始目标角度 电机1={self.last_target1}°, 电机2={self.last_target2}°")
            
            while not self.stop_continuous_control and self.is_continuous_control:
                # 获取当前目标角度
                current_target1 = self.motor1_target_angle.get()
                current_target2 = self.motor2_target_angle.get()
                
                # # 检查目标角度是否改变
                # target_changed = (abs(current_target1 - self.last_target1) > 0.1 or 
                #                 abs(current_target2 - self.last_target2) > 0.1)
                
                # if target_changed:
                #     print(f"目标角度改变: 电机1: {self.last_target1:.1f}° -> {current_target1:.1f}°, "
                #           f"电机2: {self.last_target2:.1f}° -> {current_target2:.1f}°")
                #     self.last_target1 = current_target1
                #     self.last_target2 = current_target2
                
                # 转换为弧度
                target1_rad = math.radians(current_target1)
                target2_rad = math.radians(current_target2)
                
                # 获取PID参数
                kp = self.kp.get()
                kd = self.kd.get()
                
                # # 调试输出
                # print(f"发送命令: 电机1={target1_rad:.4f}rad({current_target1:.1f}°), "
                #       f"电机2={target2_rad:.4f}rad({current_target2:.1f}°), "
                #       f"Kp={kp}, Kd={kd}")
                
                # 发送控制命令
                self.motor1.set_command(target1_rad, 0.0, kp, kd, 0.0)
                self.motor2.set_command(target2_rad, 0.0, kp, kd, 0.0)
                self.motor_manager.send_all_commands()
                
                # 控制频率 - 200Hz
                time.sleep(0.005)
                
        except Exception as e:
            print(f"连续控制循环错误: {e}")
            self.root.after(0, lambda: messagebox.showerror("错误", f"连续控制过程中出错: {str(e)}"))
        finally:
            self.is_continuous_control = False
            self.root.after(0, self.update_ui_state)
    
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
            # 录制功能需要电机使能且不在连续控制中
            if self.is_enabled and not self.is_recording and not self.is_playing and not self.is_continuous_control:
                self.record_btn.config(state="normal")
            else:
                self.record_btn.config(state="disabled")
            
            # 连续控制按钮状态
            if self.is_enabled and not self.is_recording and not self.is_playing:
                self.move_btn.config(state="normal")
                if self.is_continuous_control:
                    self.move_btn.config(text="停止连续控制")
                else:
                    self.move_btn.config(text="开始连续控制")
            else:
                self.move_btn.config(state="disabled")
        else:
            self.connect_btn.config(state="normal")
            self.disconnect_btn.config(state="disabled")
            self.enable_btn.config(state="disabled")
            self.disable_btn.config(state="disabled")
            self.zero_btn.config(state="disabled")
            # 断开连接时禁用所有录制功能和连续控制
            self.record_btn.config(state="disabled")
            self.stop_record_btn.config(state="disabled")
            self.play_btn.config(state="disabled")
            self.stop_play_btn.config(state="disabled")
            self.move_btn.config(state="disabled")
            self.move_btn.config(text="开始连续控制")
    
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
        last_record_time = 0
        
        while not self.stop_monitoring and self.is_connected:
            try:
                if self.motor1 and self.motor2:
                    # 更新电机状态
                    self.motor_manager.update_all_states()
                    
                    # 录制功能：记录当前位置
                    current_time = time.time()
                    if (self.is_recording and 
                        current_time - last_record_time >= self.recording_interval):
                        
                        motor1_pos = self.motor1.get_position()
                        motor2_pos = self.motor2.get_position()
                        
                        # 记录相对时间戳
                        relative_time = current_time - self.recording_start_time
                        self.recorded_trajectory.append([relative_time, motor1_pos, motor2_pos])
                        
                        last_record_time = current_time
                        
                        # 更新轨迹信息显示
                        self.root.after(0, lambda: 
                            self.trajectory_info_label.config(text=f"轨迹点数: {len(self.recorded_trajectory)}"))
                    
                    # 更新UI显示
                    self.root.after(0, self.update_motor_display)
                
                time.sleep(0.005)  # 5ms更新间隔 (200Hz)，提高录制精度
                
            except Exception as e:
                print(f"监控线程错误: {e}")
                time.sleep(0.1)
    
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
    
    def start_recording(self):
        """开始录制电机轨迹"""
        try:
            if not self.is_enabled:
                messagebox.showerror("错误", "请先使能电机")
                return
            
            self.is_recording = True
            self.recorded_trajectory = []
            self.recording_start_time = time.time()
            
            # 更新UI状态
            self.record_btn.config(state="disabled")
            self.stop_record_btn.config(state="normal")
            self.play_btn.config(state="disabled")
            self.record_status_label.config(text="状态: 正在录制...", foreground="red")
            
            messagebox.showinfo("录制", "开始录制电机轨迹！\n请手动移动电机到各个位置。")
            
        except Exception as e:
            messagebox.showerror("错误", f"开始录制失败: {str(e)}")
    
    def stop_recording(self):
        """停止录制"""
        try:
            self.is_recording = False
            
            # 更新UI状态
            self.record_btn.config(state="normal")
            self.stop_record_btn.config(state="disabled")
            if len(self.recorded_trajectory) > 0:
                self.play_btn.config(state="normal")
            
            duration = time.time() - self.recording_start_time if self.recording_start_time else 0
            self.record_status_label.config(text=f"状态: 录制完成 ({duration:.1f}秒)", foreground="green")
            self.trajectory_info_label.config(text=f"轨迹点数: {len(self.recorded_trajectory)}")
            
            messagebox.showinfo("录制", f"录制完成！\n录制时长: {duration:.1f}秒\n轨迹点数: {len(self.recorded_trajectory)}")
            
        except Exception as e:
            messagebox.showerror("错误", f"停止录制失败: {str(e)}")
    
    def start_playback(self):
        """开始回放轨迹"""
        try:
            if not self.is_enabled:
                messagebox.showerror("错误", "请先使能电机")
                return
            
            if len(self.recorded_trajectory) == 0:
                messagebox.showerror("错误", "没有可回放的轨迹")
                return
            
            result = messagebox.askyesno("确认", f"确定要回放轨迹吗？\n轨迹点数: {len(self.recorded_trajectory)}")
            if not result:
                return
            
            self.is_playing = True
            self.stop_playback = False
            
            # 更新UI状态
            self.record_btn.config(state="disabled")
            self.play_btn.config(state="disabled")
            self.stop_play_btn.config(state="normal")
            self.record_status_label.config(text="状态: 正在回放...", foreground="orange")
            
            # 启动回放线程
            self.playback_thread = threading.Thread(target=self.playback_trajectory, daemon=True)
            self.playback_thread.start()
            
        except Exception as e:
            messagebox.showerror("错误", f"开始回放失败: {str(e)}")
    
    def stop_playback(self):
        """停止回放"""
        try:
            self.stop_playback = True
            self.is_playing = False
            
            # 更新UI状态
            self.record_btn.config(state="normal")
            self.play_btn.config(state="normal")
            self.stop_play_btn.config(state="disabled")
            self.record_status_label.config(text="状态: 回放已停止", foreground="blue")
            
        except Exception as e:
            messagebox.showerror("错误", f"停止回放失败: {str(e)}")
    
    def playback_trajectory(self):
        """回放轨迹线程函数"""
        try:
            start_time = time.time()
            
            for i, point in enumerate(self.recorded_trajectory):
                if self.stop_playback:
                    break
                
                timestamp, motor1_pos, motor2_pos = point
                
                # 计算目标时间
                target_time = start_time + timestamp
                current_time = time.time()
                
                # 等待到目标时间，但不超过最大等待时间
                if current_time < target_time:
                    wait_time = min(target_time - current_time, 0.1)  # 最大等待100ms
                    if wait_time > 0:
                        time.sleep(wait_time)
                
                # 设置电机位置
                kp = self.kp.get()
                kd = self.kd.get()
                
                self.motor1.set_command(motor1_pos, 0.0, kp, kd, 0.0)
                self.motor2.set_command(motor2_pos, 0.0, kp, kd, 0.0)
                self.motor_manager.send_all_commands()
                
                # 更新进度
                progress = (i + 1) / len(self.recorded_trajectory) * 100
                self.root.after(0, lambda p=progress: 
                    self.record_status_label.config(text=f"状态: 回放中... ({p:.1f}%)", foreground="orange"))
            
            # 回放完成
            if not self.stop_playback:
                self.root.after(0, lambda: 
                    self.record_status_label.config(text="状态: 回放完成", foreground="green"))
            
            # 重置状态
            self.is_playing = False
            self.root.after(0, lambda: [
                self.record_btn.config(state="normal"),
                self.play_btn.config(state="normal"),
                self.stop_play_btn.config(state="disabled")
            ])
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"回放过程中出错: {str(e)}"))
            self.is_playing = False
    
    def save_trajectory(self):
        """保存轨迹到文件"""
        try:
            if len(self.recorded_trajectory) == 0:
                messagebox.showerror("错误", "没有可保存的轨迹")
                return
            
            # 选择保存文件
            filename = filedialog.asksaveasfilename(
                title="保存轨迹文件",
                defaultextension=".json",
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],
                initialname=f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            if filename:
                trajectory_data = {
                    "timestamp": datetime.now().isoformat(),
                    "motor_count": 2,
                    "recording_interval": self.recording_interval,
                    "trajectory": self.recorded_trajectory,
                    "metadata": {
                        "total_points": len(self.recorded_trajectory),
                        "duration": self.recorded_trajectory[-1][0] if self.recorded_trajectory else 0
                    }
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("保存成功", f"轨迹已保存到:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存轨迹失败: {str(e)}")
    
    def load_trajectory(self):
        """从文件加载轨迹"""
        try:
            # 选择加载文件
            filename = filedialog.askopenfilename(
                title="加载轨迹文件",
                filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
            )
            
            if filename:
                with open(filename, 'r', encoding='utf-8') as f:
                    trajectory_data = json.load(f)
                
                self.recorded_trajectory = trajectory_data.get("trajectory", [])
                
                # 更新UI显示
                self.trajectory_info_label.config(text=f"轨迹点数: {len(self.recorded_trajectory)}")
                if len(self.recorded_trajectory) > 0:
                    self.play_btn.config(state="normal")
                    duration = trajectory_data.get("metadata", {}).get("duration", 0)
                    self.record_status_label.config(text=f"状态: 已加载轨迹 ({duration:.1f}秒)", foreground="blue")
                
                messagebox.showinfo("加载成功", 
                    f"轨迹加载成功！\n文件: {os.path.basename(filename)}\n轨迹点数: {len(self.recorded_trajectory)}")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载轨迹失败: {str(e)}")
    
    def clear_trajectory(self):
        """清除当前轨迹"""
        try:
            if len(self.recorded_trajectory) == 0:
                messagebox.showinfo("提示", "当前没有轨迹数据")
                return
            
            result = messagebox.askyesno("确认", "确定要清除当前轨迹吗？")
            if result:
                self.recorded_trajectory = []
                self.trajectory_info_label.config(text="轨迹点数: 0")
                self.play_btn.config(state="disabled")
                self.record_status_label.config(text="状态: 轨迹已清除", foreground="blue")
                messagebox.showinfo("清除", "轨迹已清除")
            
        except Exception as e:
            messagebox.showerror("错误", f"清除轨迹失败: {str(e)}")
    
    def on_closing(self):
        """窗口关闭事件"""
        try:
            # 停止录制、回放和连续控制
            if self.is_recording:
                self.stop_recording()
            if self.is_playing:
                self.stop_playback()
            if self.is_continuous_control:
                self.stop_continuous_control_mode()
            
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