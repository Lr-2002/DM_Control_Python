#!/usr/bin/env python3
"""
最小惯性参数重力补偿计算器
使用辨识出的基参数和回归器计算动力学力矩
"""

import numpy as np
import sys
import os

# 添加urdfly目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'urdfly'))

from regressor import CalcDynamics

class MinimumGravityCompensation:
    """
    基于最小惯性参数的重力补偿类
    """
    
    def __init__(self, regressor_lib_path=None, param_file=None):
        """
        初始化重力补偿器
        
        Args:
            regressor_lib_path: 回归器动态库路径
            param_file: 辨识参数文件路径
        """
        # 默认路径
        if regressor_lib_path is None:
            regressor_lib_path = os.path.join(
                os.path.dirname(__file__), 
                'urdfly', 
                'dyn_regress.dylib'
            )
        
        if param_file is None:
            # 使用最新的参数文件
            param_file = self._find_latest_param_file()
        
        # 初始化回归器
        self.regressor = CalcDynamics(regressor_lib_path)
        
        # 加载辨识参数和基参数索引
        self.base_params, self.base_columns = self._load_parameters(param_file)
        
        print(f"重力补偿器初始化完成:")
        print(f"  回归器库: {regressor_lib_path}")
        print(f"  参数文件: {param_file}")
        print(f"  基参数数量: {len(self.base_params)}")
    def _find_latest_param_file(self):
        """查找最新的参数文件"""
        results_dir = os.path.join(os.path.dirname(__file__), 'identification_results')
        if not os.path.exists(results_dir):
            raise FileNotFoundError(f"参数结果目录不存在: {results_dir}")
        
        # 查找所有参数文件
        param_files = []
        for filename in os.listdir(results_dir):
            if filename.startswith('comprehensive_params_') and filename.endswith('.npz'):
                param_files.append(os.path.join(results_dir, filename))
        
        if not param_files:
            raise FileNotFoundError("未找到参数文件")
        
        # 返回最新的文件
        latest_file = max(param_files, key=os.path.getmtime)
        return latest_file
    
    def _load_parameters(self, param_file):
        """加载辨识参数"""
        try:
            data = np.load(param_file)
            
            # 检查文件中的键名，适配不同的参数文件格式
            if 'theta_base' in data:
                # 旧格式
                base_params = data['theta_base']
                base_columns = data['base_columns']
            elif 'identified_params' in data:
                # 新格式 (comprehensive_params_*.npz)
                base_params = data['identified_params']
                base_columns = data['base_columns'] if 'base_columns' in data else None
            else:
                # 检查所有可用的键
                available_keys = list(data.keys())
                raise ValueError(f"未找到参数数据，可用键: {available_keys}")
            
            print(f"成功加载参数:")
            print(f'  参数位置: {param_file}')
            print(f"  参数数量: {len(base_params)}")
            if base_columns is not None:
                print(f"  基参数索引数量: {len(base_columns)}")
            print(f"  参数范围: [{base_params.min():.6f}, {base_params.max():.6f}]")
            print(f"  参数标准差: {base_params.std():.6f}")
            
            return base_params, base_columns
            
        except Exception as e:
            raise RuntimeError(f"加载参数文件失败: {e}")
    
    def calculate_torque(self, q, dq, ddq):
        """
        计算动力学力矩
        
        Args:
            q: 关节位置 (5,) 或 (N, 5)
            dq: 关节速度 (5,) 或 (N, 5)  
            ddq: 关节加速度 (5,) 或 (N, 5)
            
        Returns:
            tau: 关节力矩 (5,) 或 (N, 5)
        """
        # 转换为numpy数组
        q = np.asarray(q, dtype=np.float64)
        dq = np.asarray(dq, dtype=np.float64)
        ddq = np.asarray(ddq, dtype=np.float64)
        
        # 处理单点和多点情况
        single_point = False
        if q.ndim == 1:
            single_point = True
            q = q.reshape(1, -1)
            dq = dq.reshape(1, -1)
            ddq = ddq.reshape(1, -1)
        
        # 检查输入维度
        if q.shape[1] != 5 or dq.shape[1] != 5 or ddq.shape[1] != 5:
            raise ValueError("输入必须是5个关节的数据")
        
        if q.shape[0] != dq.shape[0] or q.shape[0] != ddq.shape[0]:
            raise ValueError("q, dq, ddq的数据点数必须相同")
        
        # 计算力矩
        tau_list = []
        for i in range(q.shape[0]):
            # 计算回归器矩阵 (单行，48维)
            regressor_row = self.regressor.calc(q[i], dq[i], ddq[i])
            
            # 提取基参数对应的列 (40维)
            regressor_base = regressor_row[:, self.base_columns]
            
            # 计算力矩: tau = regressor_base @ base_params
            tau_i = regressor_base @ self.base_params
            tau_list.append(tau_i)
        
        tau = np.array(tau_list)
        
        # 如果输入是单点，返回一维数组
        if single_point:
            tau = tau.flatten()
        
        return tau
    
    def calculate_gravity_torque(self, q):
        """
        计算重力力矩 (加速度为0)
        
        Args:
            q: 关节位置 (5,) 或 (N, 5)
            
        Returns:
            tau_g: 重力力矩 (5,) 或 (N, 5)
        """
        # 转换为numpy数组
        q = np.asarray(q, dtype=np.float64)
        
        # 处理维度
        single_point = False
        if q.ndim == 1:
            single_point = True
            q = q.reshape(1, -1)
        
        # 速度和加速度都为0
        dq = np.zeros_like(q)
        ddq = np.zeros_like(q)
        
        # 计算重力力矩
        tau_g = self.calculate_torque(q, dq, ddq)
        
        return tau_g
    
    def calculate_coriolis_torque(self, q, dq):
        """
        计算科里奥利力矩 (加速度为0)
        
        Args:
            q: 关节位置 (5,) 或 (N, 5)
            dq: 关节速度 (5,) 或 (N, 5)
            
        Returns:
            tau_c: 科里奥利力矩 (5,) 或 (N, 5)
        """
        # 转换为numpy数组
        q = np.asarray(q, dtype=np.float64)
        dq = np.asarray(dq, dtype=np.float64)
        
        # 处理维度
        single_point = False
        if q.ndim == 1:
            single_point = True
            q = q.reshape(1, -1)
            dq = dq.reshape(1, -1)
        
        # 加速度为0
        ddq = np.zeros_like(q)
        
        # 计算科里奥利力矩
        tau_c = self.calculate_torque(q, dq, ddq)
        
        return tau_c
    
    def get_parameter_info(self):
        """获取参数信息"""
        return {
            'num_base_params': len(self.base_params),
            'param_range': [self.base_params.min(), self.base_params.max()],
            'param_std': self.base_params.std(),
            'base_params': self.base_params.copy()
        }
    
    def validate_calculation(self, q, dq, ddq, tau_measured=None):
        """
        验证计算结果
        
        Args:
            q, dq, ddq: 关节状态
            tau_measured: 实测力矩 (可选)
            
        Returns:
            validation_results: 验证结果字典
        """
        # 计算预测力矩
        tau_predicted = self.calculate_torque(q, dq, ddq)
        
        results = {
            'tau_predicted': tau_predicted,
            'q_range': [np.degrees(np.min(q)), np.degrees(np.max(q))],
            'dq_range': [np.degrees(np.min(dq)), np.degrees(np.max(dq))],
            'ddq_range': [np.degrees(np.min(ddq)), np.degrees(np.max(ddq))],
            'tau_range': [np.min(tau_predicted), np.max(tau_predicted)]
        }
        
        if tau_measured is not None:
            tau_measured = np.asarray(tau_measured)
            residual = tau_measured - tau_predicted
            rms_error = np.sqrt(np.mean(residual**2))
            relative_error = rms_error / np.std(tau_measured) * 100
            
            results.update({
                'tau_measured': tau_measured,
                'residual': residual,
                'rms_error': rms_error,
                'relative_error': relative_error,
                'max_error': np.max(np.abs(residual))
            })
        
        return results

def test_gravity_compensation():
    """测试重力补偿功能"""
    print("=== 最小惯性参数重力补偿测试 ===")
    
    try:
        # 初始化重力补偿器
        gc = MinimumGravityCompensation()
        
        # 测试单点计算
        print("\n1. 单点测试:")
        q_test = np.array([0.1, -0.5, 0.3, -0.2, 0.4])  # 弧度
        dq_test = np.array([0.1, 0.2, -0.1, 0.3, -0.2])  # 弧度/秒
        ddq_test = np.array([0.5, -0.3, 0.2, -0.4, 0.1])  # 弧度/秒²
        
        tau = gc.calculate_torque(q_test, dq_test, ddq_test)
        print(f"输入位置: {np.degrees(q_test)} 度")
        print(f"输入速度: {np.degrees(dq_test)} 度/秒")
        print(f"输入加速度: {np.degrees(ddq_test)} 度/秒²")
        print(f"计算力矩: {tau} Nm")
        
        # 测试重力力矩
        print("\n2. 重力力矩测试:")
        tau_g = gc.calculate_gravity_torque(q_test)
        print(f"重力力矩: {tau_g} Nm")
        
        # 测试科里奥利力矩
        print("\n3. 科里奥利力矩测试:")
        tau_c = gc.calculate_coriolis_torque(q_test, dq_test)
        print(f"科里奥利力矩: {tau_c} Nm")
        
        # 测试多点计算
        print("\n4. 多点测试:")
        n_points = 5
        q_multi = np.random.uniform(-1, 1, (n_points, 5))
        dq_multi = np.random.uniform(-2, 2, (n_points, 5))
        ddq_multi = np.random.uniform(-5, 5, (n_points, 5))
        
        tau_multi = gc.calculate_torque(q_multi, dq_multi, ddq_multi)
        print(f"多点计算形状: {tau_multi.shape}")
        print(f"力矩范围: [{tau_multi.min():.3f}, {tau_multi.max():.3f}] Nm")
        
        # 参数信息
        print("\n5. 参数信息:")
        param_info = gc.get_parameter_info()
        print(f"基参数数量: {param_info['num_base_params']}")
        print(f"参数范围: [{param_info['param_range'][0]:.6f}, {param_info['param_range'][1]:.6f}]")
        print(f"参数标准差: {param_info['param_std']:.6f}")
        
        print("\n✅ 重力补偿器测试完成!")
        return gc
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_gravity_compensation()