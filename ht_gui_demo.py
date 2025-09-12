#!/usr/bin/env python3
"""
HT电机GUI控制器演示版本
包含模拟数据用于演示界面功能
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import math
import random
from typing import Optional


class MockMotor:
    """模拟电机类用于演示"""
    
    def __init__(self, motor_id):
        self.motor_id = motor_id
        self.position = 0.0  # 当前位置 (弧度)
        self.velocity = 0.0  # 当前速度 (rad/s)
        self.torque = 0.0    # 当前力矩 (Nm)
        self.target_position = 0.0
        self.enabled = False
        
    def enable(self):
        self.enabled = True
        return True
    
    def disable(self):
        self.enabled = False
        self.velocity = 0.0
        return True
    
    def set_zero(self):
        self.position = 0.0
        return True
    
    def set_command(self, pos, vel, kp, kd, tau):
        if self.enabled:
            self.target_position = pos
        return True
    
    def update_state(self):
        if self.enabled:
            # 模拟电机运动到目标位置
            error = self.target_position - self.position
            if abs(error) > 0.01:  # 如果误差大于0.01弧度
                # 简单的位置控制模拟
                self.velocity = error * 2.0  # 简单比例控制
                self.position += self.velocity * 0.1  # 积分
                self.torque = error * 10.0 + random.uniform(-0.5, 0.5)  # 模拟力矩
            else:
                self.velocity *= 0.9  # 阻尼
                if abs(self.velocity) < 0.01:
                    self.velocity = 0.0
        return True
    
    def get_position(self):
        return self.position
    
    def get_velocity(self):
        return self.velocity
    
    def get_torque(self):
        return self.torque


class HTMotorGUIDemo:
    """HT电机GUI控制器演示版"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HT电机双轴控制器 - 演示版")
        self.root.geometry("800x600")
        
        # 创建模拟电机
        self.motor1 = MockMotor(1)
        self.motor2 = MockMotor(2)
        
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
        
        # 演示说明
        demo_label = ttk.Label(main_frame, text="🔧 演示模式 - 使用模拟数据", 
                              font=("Arial", 12, "bold"), foreground="blue")
        demo_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # 连接控制区域
        connection_frame = ttk.LabelFrame(main_frame, text="连接控制", padding="10")
        connection_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
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
        pid_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(pid_frame, text="Kp:").grid(row=0, column=0, sticky=tk.W)
        kp_entry = ttk.Entry(pid_frame, textvariable=self.kp, width=10)
        kp_entry.grid(row=0, column=1, padx=(5, 20))
        
        ttk.Label(pid_frame, text="Kd:").grid(row=0, column=2, sticky=tk.W)
        kd_entry = ttk.Entry(pid_frame, textvariable=self.kd, width=10)
        kd_entry.grid(row=0, column=3, padx=(5, 0))
        
        # 电机1控制区域
        motor1_frame = ttk.LabelFrame(main_frame, text="电机1控制", padding="10")
        motor1_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
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
        motor2_frame.grid(row=3, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
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
        global_frame.grid(row=4, column=0, columnspan=2, pady=(20, 0))
        
        ttk.Button(global_frame, text="同时移动两个电机", 
                  command=self.move_both_motors).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(global_frame, text="停止所有电机", 
                  command=self.stop_all_motors).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(global_frame, text="回到零位", 
                  command=self.move_to_zero).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(global_frame, text="读取所有角度", 
                  command=self.read_all_angles).grid(row=0, column=3)
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        motor1_frame.columnconfigure(0, weight=1)
        motor2_frame.columnconfigure(0, weight=1)
        motor1_angle_frame.columnconfigure(0, weight=1)
        motor2_angle_frame.columnconfigure(0, weight=1)
        
    def connect_motors(self):
        """连接电机（演示版）"""
        self.is_connected = True
        self.update_ui_state()
        self.status_label.config(text="状态: 已连接 (演示模式)", foreground="green")
        self.start_monitoring()
        messagebox.showinfo("成功", "电机连接成功！(演示模式)")
    
    def disconnect_motors(self):
        """断开电机连接"""
        self.stop_monitoring()
        if self.is_enabled:
            self.disable_motors()
        
        self.is_connected = False
        self.update_ui_state()
        self.status_label.config(text="状态: 未连接", foreground="red")
    
    def enable_motors(self):
        """使能电机"""
        if not self.is_connected:
            messagebox.showerror("错误", "请先连接电机")
            return
        
        self.motor1.enable()
        self.motor2.enable()
        self.is_enabled = True
        self.update_ui_state()
        messagebox.showinfo("成功", "电机使能成功！")
    
    def disable_motors(self):
        """失能电机"""
        self.motor1.disable()
        self.motor2.disable()
        self.is_enabled = False
        self.update_ui_state()
    
    def set_zero_position(self):
        """设置零位"""
        if not self.is_connected:
            messagebox.showerror("错误", "请先连接电机")
            return
        
        result = messagebox.askyesno("确认", "确定要设置当前位置为零位吗？")
        if result:
            self.motor1.set_zero()
            self.motor2.set_zero()
            messagebox.showinfo("成功", "零位设置成功！")
    
    def move_motor_to_angle(self, motor_num):
        """移动指定电机到目标角度"""
        if not self.is_enabled:
            messagebox.showerror("错误", "请先使能电机")
            return
        
        if motor_num == 1:
            target_rad = math.radians(self.motor1_target_angle.get())
            self.motor1.set_command(target_rad, 0.0, self.kp.get(), self.kd.get(), 0.0)
        else:
            target_rad = math.radians(self.motor2_target_angle.get())
            self.motor2.set_command(target_rad, 0.0, self.kp.get(), self.kd.get(), 0.0)
    
    def stop_motor(self, motor_num):
        """停止指定电机"""
        if motor_num == 1:
            current_pos = self.motor1.get_position()
            self.motor1.set_command(current_pos, 0.0, self.kp.get(), self.kd.get(), 0.0)
        else:
            current_pos = self.motor2.get_position()
            self.motor2.set_command(current_pos, 0.0, self.kp.get(), self.kd.get(), 0.0)
    
    def move_both_motors(self):
        """同时移动两个电机"""
        if not self.is_enabled:
            messagebox.showerror("错误", "请先使能电机")
            return
        
        target1_rad = math.radians(self.motor1_target_angle.get())
        target2_rad = math.radians(self.motor2_target_angle.get())
        
        kp = self.kp.get()
        kd = self.kd.get()
        
        self.motor1.set_command(target1_rad, 0.0, kp, kd, 0.0)
        self.motor2.set_command(target2_rad, 0.0, kp, kd, 0.0)
    
    def stop_all_motors(self):
        """停止所有电机"""
        self.stop_motor(1)
        self.stop_motor(2)
    
    def move_to_zero(self):
        """回到零位"""
        self.motor1_target_angle.set(0.0)
        self.motor2_target_angle.set(0.0)
        self.move_both_motors()
    
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
            # 强制更新所有电机状态
            self.motor1.update_state()
            self.motor2.update_state()
            
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
                # 更新电机状态
                self.motor1.update_state()
                self.motor2.update_state()
                
                # 更新UI显示
                self.root.after(0, self.update_motor_display)
                
                time.sleep(0.1)  # 100ms更新间隔
                
            except Exception as e:
                print(f"监控线程错误: {e}")
                time.sleep(0.5)
    
    def update_motor_display(self):
        """更新电机显示信息"""
        try:
            # 更新电机1显示
            pos1_deg = math.degrees(self.motor1.get_position())
            vel1 = self.motor1.get_velocity()
            torque1 = self.motor1.get_torque()
            
            self.motor1_pos_label.config(text=f"{pos1_deg:.2f}°")
            self.motor1_vel_label.config(text=f"{vel1:.2f} rad/s")
            self.motor1_torque_label.config(text=f"{torque1:.2f} Nm")
            
            self.motor1_current_angle.set(pos1_deg)
            
            # 更新电机2显示
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
    app = HTMotorGUIDemo()
    app.run()


if __name__ == "__main__":
    main()
