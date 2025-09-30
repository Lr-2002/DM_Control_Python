#!/usr/bin/env python3
"""
配置文件加载器
Configuration file loader for IC ARM Control
"""

import os
import yaml
from typing import Dict, Any, Optional, List

class ConfigLoader:
    """配置文件加载器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径，如果为None则自动查找
        """
        self.config_path = config_path or self._find_config_file()
        self.config = self._load_config()
    
    def _find_config_file(self) -> str:
        """自动查找配置文件"""
        # 获取当前文件的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 向上查找项目根目录中的config.yaml
        search_paths = [
            # 从utils目录向上两级到项目根目录
            os.path.join(current_dir, '..', '..', 'config.yaml'),
            # 从当前目录查找
            os.path.join(current_dir, 'config.yaml'),
            # 从工具目录查找
            os.path.join(current_dir, '..', 'tools', 'config.yaml'),
        ]
        
        for path in search_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
        
        # 如果找不到，返回默认路径
        default_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'config.yaml'))
        print(f"Warning: Config file not found, using default path: {default_path}")
        return default_path
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                print(f"Config loaded from: {self.config_path}")
                return config
        except FileNotFoundError:
            print(f"Warning: Config file not found: {self.config_path}")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'urdf': {
                'default_path': "/Users/lr-2002/project/instantcreation/IC_arm_control/urdfs/ic_arm_8_dof/urdf/ic_arm_8_dof.urdf",
                'alternative_paths': [
                    "/home/lr-2002/project/InstantCreation/DM_Control_Python/ic_arm_8dof/urdf/ic_arm_8dof.urdf"
                ]
            },
            'simulation': {
                'default_update_rate': 30.0,
                'default_print_interval': 2.0
            },
            'urdf_updater': {
                'default_update_rate': 10.0,
                'default_auto_save_interval': 30.0,
                'default_safety_margin': 5.0
            },
            'joint_mapping': {
                'motor_to_joint': {
                    'm1': 'joint1', 'm2': 'joint2', 'm3': 'joint3',
                    'm4': 'joint4', 'm5': 'joint5', 'm6': 'joint6'
                },
                'joint_names': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            },
            'control': {
                'default_frequency': 500.0,  # Hz - 500Hz控制频率
                'buffer_size': 500,
                'max_frequency': 1000.0,
                'min_frequency': 100.0
            }
        }
    
    def get_urdf_path(self) -> str:
        """获取URDF文件路径，自动检查文件是否存在"""
        urdf_config = self.config.get('urdf', {})
        
        # 首先尝试默认路径
        default_path = urdf_config.get('default_path')
        if default_path and os.path.exists(default_path):
            return default_path
        
        # 尝试备用路径
        alternative_paths = urdf_config.get('alternative_paths', [])
        for path in alternative_paths:
            if os.path.exists(path):
                print(f"Using alternative URDF path: {path}")
                return path
        
        # 如果都不存在，返回默认路径（让调用者处理错误）
        print(f"Warning: URDF file not found, returning default path: {default_path}")
        return default_path or "/Users/lr-2002/project/instantcreation/IC_arm_control/urdfs/ic_arm_8_dof/urdf/ic_arm_8_dof.urdf"
    
    def get_simulation_config(self) -> Dict[str, Any]:
        """获取仿真配置"""
        return self.config.get('simulation', {
            'default_update_rate': 30.0,
            'default_print_interval': 2.0
        })
    
    def get_urdf_updater_config(self) -> Dict[str, Any]:
        """获取URDF更新器配置"""
        return self.config.get('urdf_updater', {
            'default_update_rate': 10.0,
            'default_auto_save_interval': 30.0,
            'default_safety_margin': 5.0
        })
    
    def get_joint_mapping(self) -> Dict[str, Any]:
        """获取关节映射配置"""
        return self.config.get('joint_mapping', {
            'motor_to_joint': {
                'm1': 'joint1', 'm2': 'joint2', 'm3': 'joint3',
                'm4': 'joint4', 'm5': 'joint5', 'm6': 'joint6'
            },
            'joint_names': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        })
    
    def get_joint_names(self) -> List[str]:
        """获取关节名称列表"""
        joint_mapping = self.get_joint_mapping()
        return joint_mapping.get('joint_names', ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
    
    def get_motor_to_joint_mapping(self) -> Dict[str, str]:
        """获取电机到关节的映射"""
        joint_mapping = self.get_joint_mapping()
        return joint_mapping.get('motor_to_joint', {
            'm1': 'joint1', 'm2': 'joint2', 'm3': 'joint3',
            'm4': 'joint4', 'm5': 'joint5', 'm6': 'joint6'
        })

    def get_control_config(self) -> Dict[str, Any]:
        """获取控制配置"""
        return self.config.get('control', {
            'default_frequency': 500.0,
            'buffer_size': 500,
            'max_frequency': 1000.0,
            'min_frequency': 100.0
        })

    def get_default_control_frequency(self) -> float:
        """获取默认控制频率"""
        control_config = self.get_control_config()
        return control_config.get('default_frequency', 500.0)

# 全局配置实例
_global_config = None

def get_config() -> ConfigLoader:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = ConfigLoader()
    return _global_config

def get_urdf_path() -> str:
    """快捷方法：获取URDF路径"""
    return get_config().get_urdf_path()

def get_joint_names() -> List[str]:
    """快捷方法：获取关节名称列表"""
    return get_config().get_joint_names()

def get_motor_to_joint_mapping() -> Dict[str, str]:
    """快捷方法：获取电机到关节的映射"""
    return get_config().get_motor_to_joint_mapping()

def get_control_config() -> Dict[str, Any]:
    """快捷方法：获取控制配置"""
    return get_config().get_control_config()

def get_default_control_frequency() -> float:
    """快捷方法：获取默认控制频率"""
    return get_config().get_default_control_frequency()
