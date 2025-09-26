#!/usr/bin/env python3
"""
URDF Limit Updater
持续监控关节角度，记录每个关节的最大和最小值，然后生成更新的URDF文件
"""

import numpy as np
import time
import xml.etree.ElementTree as ET
import argparse
from datetime import datetime
from ic_arm_control.control.IC_ARM import ICARM
import os
import shutil

class URDFLimitUpdater:
    """URDF限制更新器"""
    
    def __init__(self, original_urdf_path, output_urdf_path=None):
        """
        初始化URDF限制更新器
        
        Args:
            original_urdf_path: 原始URDF文件路径
            output_urdf_path: 输出URDF文件路径（可选）
        """
        self.original_urdf_path = original_urdf_path
        self.output_urdf_path = output_urdf_path or self._generate_output_path()
        
        # 关节限制记录
        self.joint_limits = {}
        self.joint_names = ['joint1', 'joint2','joint3','joint4','joint5','joint6',]  # 对应URDF中的关节名
        
        # 初始化每个关节的限制
        for joint_name in self.joint_names:
            self.joint_limits[joint_name] = {
                'min': float('inf'),
                'max': float('-inf'),
                'samples': 0
            }
        
        print(f"原始URDF: {self.original_urdf_path}")
        print(f"输出URDF: {self.output_urdf_path}")
    
    def _generate_output_path(self):
        """生成输出文件路径"""
        base_dir = os.path.dirname(self.original_urdf_path)
        base_name = os.path.splitext(os.path.basename(self.original_urdf_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(base_dir, f"{base_name}_updated_{timestamp}.urdf")
    
    def update_limits(self, joint_positions_rad):
        """
        更新关节限制
        
        Args:
            joint_positions_rad: 关节位置数组（弧度）
        """
        for i, joint_name in enumerate(self.joint_names):
            if i < len(joint_positions_rad):
                pos = joint_positions_rad[i]
                
                # 更新最小值和最大值
                if pos < self.joint_limits[joint_name]['min']:
                    self.joint_limits[joint_name]['min'] = pos
                if pos > self.joint_limits[joint_name]['max']:
                    self.joint_limits[joint_name]['max'] = pos
                
                self.joint_limits[joint_name]['samples'] += 1
    
    def print_current_limits(self):
        """打印当前限制"""
        print("\n=== 当前关节限制 ===")
        print(f"{'关节':<10} {'最小值(度)':<12} {'最大值(度)':<12} {'范围(度)':<12} {'样本数':<8}")
        print("-" * 60)
        
        for joint_name in self.joint_names:
            limits = self.joint_limits[joint_name]
            if limits['samples'] > 0:
                min_deg = np.degrees(limits['min'])
                max_deg = np.degrees(limits['max'])
                range_deg = max_deg - min_deg
                print(f"{joint_name:<10} {min_deg:<12.2f} {max_deg:<12.2f} {range_deg:<12.2f} {limits['samples']:<8}")
            else:
                print(f"{joint_name:<10} {'无数据':<12} {'无数据':<12} {'无数据':<12} {limits['samples']:<8}")
    
    def generate_updated_urdf(self, safety_margin_deg=5.0):
        """
        生成更新的URDF文件
        
        Args:
            safety_margin_deg: 安全边距（度）
        """
        try:
            # 解析原始URDF
            tree = ET.parse(self.original_urdf_path)
            root = tree.getroot()
            
            # 查找并更新关节限制
            joints_updated = 0
            for joint in root.findall('.//joint'):
                joint_name = joint.get('name')
                joint_type = joint.get('type')
                
                if joint_name in self.joint_limits and joint_type in ['continuous', 'revolute']:
                    limits = self.joint_limits[joint_name]
                    
                    if limits['samples'] > 0:
                        # 计算带安全边距的限制
                        safety_margin_rad = np.radians(safety_margin_deg)
                        lower_limit = limits['min'] - safety_margin_rad
                        upper_limit = limits['max'] + safety_margin_rad
                        
                        # 将关节类型从continuous改为revolute
                        joint.set('type', 'revolute')
                        
                        # 添加或更新limit元素
                        limit_elem = joint.find('limit')
                        if limit_elem is None:
                            limit_elem = ET.SubElement(joint, 'limit')
                        
                        limit_elem.set('lower', f"{lower_limit:.6f}")
                        limit_elem.set('upper', f"{upper_limit:.6f}")
                        limit_elem.set('effort', "100.0")  # 默认力矩限制
                        limit_elem.set('velocity', "3.14")  # 默认速度限制
                        
                        joints_updated += 1
                        print(f"更新关节 {joint_name}: [{np.degrees(lower_limit):.2f}°, {np.degrees(upper_limit):.2f}°]")
            
            # 更新mesh路径为相对路径
            for mesh in root.findall('.//mesh'):
                filename = mesh.get('filename')
                if filename and filename.startswith('/Users/'):
                    # 转换为相对路径
                    relative_path = os.path.relpath(filename, os.path.dirname(self.output_urdf_path))
                    mesh.set('filename', relative_path)
            
            # 保存更新的URDF
            tree.write(self.output_urdf_path, encoding='utf-8', xml_declaration=True)
            
            print(f"\n✓ URDF文件已更新: {self.output_urdf_path}")
            print(f"✓ 更新了 {joints_updated} 个关节的限制")
            print(f"✓ 安全边距: ±{safety_margin_deg}°")
            
            return True
            
        except Exception as e:
            print(f"✗ 生成URDF文件失败: {e}")
            return False
    
    def monitor_and_update(self, arm, update_rate=10.0, duration=None, auto_save_interval=30.0):
        """
        监控关节角度并更新限制
        
        Args:
            arm: ICARM实例
            update_rate: 更新频率 (Hz)
            duration: 监控时长（秒，None为无限制）
            auto_save_interval: 自动保存间隔（秒）
        """
        print(f"\n=== 开始监控关节限制 ===")
        print(f"更新频率: {update_rate} Hz")
        print(f"监控时长: {duration if duration else '无限制'} 秒")
        print(f"自动保存间隔: {auto_save_interval} 秒")
        print("\n按 Ctrl+C 停止监控并生成URDF\n")
        
        start_time = time.time()
        last_save_time = start_time
        last_print_time = start_time
        update_interval = 1.0 / update_rate
        
        try:
            while True:
                loop_start = time.time()
                
                # 检查运行时间
                if duration and (loop_start - start_time) >= duration:
                    print(f"\n达到预设监控时长 {duration} 秒")
                    break
                
                # 读取关节位置
                try:
                    positions = arm.get_joint_positions(refresh=True)
                    self.update_limits(positions)
                    
                    # 每5秒打印一次当前限制
                    if loop_start - last_print_time >= 5.0:
                        self.print_current_limits()
                        last_print_time = loop_start
                    
                    # 自动保存
                    if loop_start - last_save_time >= auto_save_interval:
                        print(f"\n--- 自动保存 URDF (已运行 {loop_start - start_time:.1f}s) ---")
                        self.generate_updated_urdf()
                        last_save_time = loop_start
                    
                except Exception as e:
                    print(f"读取关节位置失败: {e}")
                
                # 控制更新频率
                elapsed = time.time() - loop_start
                if elapsed < update_interval:
                    time.sleep(update_interval - elapsed)
                    
        except KeyboardInterrupt:
            print("\n\n收到中断信号，正在生成最终URDF...")
        
        # 生成最终URDF
        self.print_current_limits()
        print(f"\n=== 生成最终URDF ===")
        success = self.generate_updated_urdf()
        
        if success:
            print(f"\n✓ 监控完成，总时长: {time.time() - start_time:.1f} 秒")
            print(f"✓ 更新的URDF文件: {self.output_urdf_path}")
        else:
            print(f"\n✗ URDF生成失败")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="URDF关节限制更新器")
    parser.add_argument("--urdf", type=str, 
                       default="/Users/lr-2002/project/instantcreation/IC_arm_control/urdfs/ic_arm_8_dof/urdf/ic_arm_8_dof.urdf",
                       help="原始URDF文件路径")
    parser.add_argument("--output", type=str, help="输出URDF文件路径（可选）")
    parser.add_argument("--rate", type=float, default=10.0, help="更新频率 (Hz)")
    parser.add_argument("--duration", type=float, help="监控时长（秒）")
    parser.add_argument("--auto-save", type=float, default=30.0, help="自动保存间隔（秒）")
    
    args = parser.parse_args()
    
    print("URDF关节限制更新器")
    print("=" * 50)
    
    # 检查原始URDF文件是否存在
    if not os.path.exists(args.urdf):
        print(f"✗ 原始URDF文件不存在: {args.urdf}")
        return
    
    try:
        # 初始化URDF更新器
        updater = URDFLimitUpdater(args.urdf, args.output)
        
        # 初始化机械臂
        print("初始化机械臂连接...")
        arm = ICARM(debug=False, gc=False)
        arm.enable()
        
        # 开始监控
        updater.monitor_and_update(
            arm=arm,
            update_rate=args.rate,
            duration=args.duration,
            auto_save_interval=args.auto_save
        )
        
    except Exception as e:
        print(f"错误: {e}")
    finally:
        try:
            arm.close()
        except:
            pass

if __name__ == "__main__":
    main()
