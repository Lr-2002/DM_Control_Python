#!/usr/bin/env python3
"""
电机参数读取工具
基于DM_Control_Python库读取达妙电机的内部参数
特别针对01号电机的参数读取和分析
"""

import time
import json
from typing import Dict, Any, Optional
from DM_CAN import *

class MotorParameterReader:
    """电机参数读取器"""
    
    def __init__(self, port: str = '/dev/cu.usbmodem00000000050C1'):
        """
        初始化电机参数读取器
        
        Args:
            port: CAN通信端口
        """
        self.port = port
        self.mc = None
        self.motor = None
        self.is_connected = False
        
    def connect(self):
        """连接到电机控制器"""
        try:
            print(f"连接到端口: {self.port}")
            self.mc = MotorControl(self.port)
            
            # 添加01号电机
            self.motor = self.mc.addMotor(DM_Motor_Type.DM4340, 0x01, 0x00)
            print("✓ 01号电机已添加")
            
            # 启用电机
            self.mc.enable(self.motor)
            print("✓ 01号电机已启用")
            
            self.is_connected = True
            return True
            
        except Exception as e:
            print(f"连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.is_connected and self.mc and self.motor:
            try:
                self.mc.disable(self.motor)
                print("✓ 01号电机已禁用")
                self.is_connected = False
            except Exception as e:
                print(f"断开连接时出错: {e}")
    
    def read_all_parameters(self) -> Dict[str, Any]:
        """
        读取所有可用的电机参数
        
        Returns:
            包含所有参数的字典
        """
        if not self.is_connected:
            print("错误: 未连接到电机")
            return {}
        
        print("开始读取01号电机参数...")
        params = {}
        
        # 定义要读取的参数列表
        param_list = [
            # 基本参数
            ('PMAX', DM_variable.PMAX, '最大位置'),
            ('VMAX', DM_variable.VMAX, '最大速度'),
            ('TMAX', DM_variable.TMAX, '最大力矩'),
            ('MST_ID', DM_variable.MST_ID, '主机ID'),
            
            # 控制参数
            ('KP_APR', DM_variable.KP_APR, '位置比例增益'),
            ('KD_APR', DM_variable.KD_APR, '位置微分增益'),
            
            # 电机信息
            ('MOTOR_TYPE', DM_variable.MOTOR_TYPE, '电机类型'),
            ('FW_VER', DM_variable.FW_VER, '固件版本'),
            ('HW_VER', DM_variable.HW_VER, '硬件版本'),
            
            # 温度和电压
            ('TEMP_MOS', DM_variable.TEMP_MOS, 'MOS温度'),
            ('TEMP_MOTOR', DM_variable.TEMP_MOTOR, '电机温度'),
            ('V_BUS', DM_variable.V_BUS, '总线电压'),
            
            # 错误状态
            ('ERROR', DM_variable.ERROR, '错误状态'),
        ]
        
        # 读取参数
        for param_name, param_var, description in param_list:
            try:
                # 刷新电机状态
                self.mc.refresh_motor_status(self.motor)
                time.sleep(0.01)  # 短暂延迟确保数据更新
                
                # 读取参数
                value = self.mc.read_motor_param(self.motor, param_var)
                params[param_name] = {
                    'value': value,
                    'description': description,
                    'unit': self._get_parameter_unit(param_name)
                }
                print(f"  {param_name} ({description}): {value} {self._get_parameter_unit(param_name)}")
                
            except Exception as e:
                print(f"  读取 {param_name} 失败: {e}")
                params[param_name] = {
                    'value': None,
                    'description': description,
                    'error': str(e)
                }
        
        return params
    
    def read_current_state(self) -> Dict[str, Any]:
        """
        读取电机当前状态
        
        Returns:
            包含当前状态的字典
        """
        if not self.is_connected:
            print("错误: 未连接到电机")
            return {}
        
        try:
            # 刷新电机状态
            self.mc.refresh_motor_status(self.motor)
            
            state = {
                'position': self.motor.getPosition(),
                'velocity': self.motor.getVelocity(),
                'torque': self.motor.getTorque(),
                'timestamp': time.time()
            }
            
            print("当前电机状态:")
            print(f"  位置: {state['position']:.4f} rad ({np.degrees(state['position']):.2f}°)")
            print(f"  速度: {state['velocity']:.4f} rad/s ({np.degrees(state['velocity']):.2f}°/s)")
            print(f"  力矩: {state['torque']:.4f} N·m")
            
            return state
            
        except Exception as e:
            print(f"读取电机状态失败: {e}")
            return {}
    
    def _get_parameter_unit(self, param_name: str) -> str:
        """获取参数单位"""
        units = {
            'PMAX': 'rad',
            'VMAX': 'rad/s',
            'TMAX': 'N·m',
            'MST_ID': '',
            'KP_APR': '',
            'KD_APR': '',
            'MOTOR_TYPE': '',
            'FW_VER': '',
            'HW_VER': '',
            'TEMP_MOS': '°C',
            'TEMP_MOTOR': '°C',
            'V_BUS': 'V',
            'ERROR': '',
        }
        return units.get(param_name, '')
    
    def save_parameters_to_file(self, params: Dict[str, Any], filename: str = None):
        """
        保存参数到JSON文件
        
        Args:
            params: 参数字典
            filename: 文件名，如果为None则自动生成
        """
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f'motor_01_params_{timestamp}.json'
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
            print(f"✓ 参数已保存到: {filename}")
        except Exception as e:
            print(f"保存参数失败: {e}")
    
    def generate_parameter_report(self, params: Dict[str, Any]) -> str:
        """
        生成参数报告
        
        Args:
            params: 参数字典
            
        Returns:
            格式化的报告字符串
        """
        report = []
        report.append("=" * 50)
        report.append("01号电机参数报告")
        report.append("=" * 50)
        report.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 按类别组织参数
        categories = {
            '基本参数': ['PMAX', 'VMAX', 'TMAX', 'MST_ID'],
            '控制参数': ['KP_APR', 'KD_APR'],
            '设备信息': ['MOTOR_TYPE', 'FW_VER', 'HW_VER'],
            '状态监控': ['TEMP_MOS', 'TEMP_MOTOR', 'V_BUS', 'ERROR']
        }
        
        for category, param_names in categories.items():
            report.append(f"【{category}】")
            for param_name in param_names:
                if param_name in params:
                    param_info = params[param_name]
                    value = param_info.get('value', 'N/A')
                    description = param_info.get('description', '')
                    unit = param_info.get('unit', '')
                    
                    if 'error' in param_info:
                        report.append(f"  {description} ({param_name}): 读取失败 - {param_info['error']}")
                    else:
                        report.append(f"  {description} ({param_name}): {value} {unit}")
            report.append("")
        
        return '\n'.join(report)


def main():
    """主函数"""
    print("=== 01号电机参数读取工具 ===\n")
    
    # 创建参数读取器
    reader = MotorParameterReader()
    
    try:
        # 连接到电机
        if not reader.connect():
            print("连接失败，程序退出")
            return
        
        print("\n等待电机稳定...")
        time.sleep(2)
        
        # 读取当前状态
        print("\n1. 读取当前电机状态:")
        current_state = reader.read_current_state()
        
        # 读取所有参数
        print("\n2. 读取所有电机参数:")
        all_params = reader.read_all_parameters()
        
        # 生成报告
        print("\n3. 生成参数报告:")
        report = reader.generate_parameter_report(all_params)
        print(report)
        
        # # 保存参数到文件
        # print("\n4. 保存参数到文件:")
        # combined_data = {
        #     'current_state': current_state,
        #     'parameters': all_params,
        #     'report': report
        # }
        # reader.save_parameters_to_file(combined_data)
        
        print("\n✓ 参数读取完成!")
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
    finally:
        # 断开连接
        reader.disconnect()
        print("程序结束")


if __name__ == "__main__":
    main()
