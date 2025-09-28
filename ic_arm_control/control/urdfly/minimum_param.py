#!/usr/bin/env python3
"""
最小惯性参数辨识
基于论文公式(20)实现: Θ = A_N^†(q_N, q̇_N, q̈_N) × τ_N
使用QR分解进行列选择，找到可辨识的基参数集合
"""

import numpy as np
import pandas as pd
from numpy.linalg import lstsq, pinv
from scipy.linalg import qr
import matplotlib.pyplot as plt
from regressor import CalcDynamics
import os
import glob
import time
from datetime import datetime

class MinimumParameterIdentification:
    def __init__(self, regressor_lib_path=None):
        """
        初始化最小参数辨识器
        
        Args:
            regressor_lib_path: 动力学回归器库文件路径
        """
        if regressor_lib_path is None:
            regressor_lib_path = '/Users/lr-2002/project/instantcreation/IC_arm_control/ic_arm_control/control/urdfly/dyn_regress.dylib'
        
        self.calc_dynamics = CalcDynamics(regressor_lib_path)
        self.n_joints = 5
        self.identified_params = None
        self.base_columns = None
        self.P_matrix = None
        self.identification_results = {}
        
    def load_motion_data(self, data_source):
        """
        加载运动数据
        
        Args:
            data_source: CSV文件路径或npz文件路径或(q, dq, ddq, tau)元组
            
        Returns:
            q, dq, ddq, tau: 位置、速度、加速度、力矩数组
        """
        if isinstance(data_source, str):
            if data_source.endswith('.csv'):
                return self._load_from_csv(data_source)
            elif data_source.endswith('.npz'):
                return self._load_from_npz(data_source)
            else:
                raise ValueError("不支持的文件格式，请使用CSV或NPZ文件")
        elif isinstance(data_source, tuple) and len(data_source) == 4:
            return data_source
        else:
            raise ValueError("数据源格式错误")
    
    def _load_from_csv(self, csv_file):
        """从CSV文件加载数据"""
        print(f"从CSV文件加载数据: {csv_file}")
        data = pd.read_csv(csv_file)
        
        print(f"CSV文件列数: {data.shape[1]}")
        print(f"列名: {list(data.columns)}")
        
        # 检测CSV格式
        if 'm1_pos_actual' in data.columns:
            # IC ARM格式：每个电机有多列数据
            print("检测到IC ARM格式数据")
            
            # 提取实际位置、速度、加速度、力矩
            pos_cols = [f'm{i}_pos_actual' for i in range(1, 6)]
            vel_cols = [f'm{i}_vel_actual' for i in range(1, 6)]
            acc_cols = [f'm{i}_acc_actual' for i in range(1, 6)]
            torque_cols = [f'm{i}_torque' for i in range(1, 6)]
            
            # 检查所有需要的列是否存在
            missing_cols = []
            for cols, name in [(pos_cols, '位置'), (vel_cols, '速度'), (acc_cols, '加速度'), (torque_cols, '力矩')]:
                for col in cols:
                    if col not in data.columns:
                        missing_cols.append(f"{name}列{col}")
            
            if missing_cols:
                raise ValueError(f"缺少必要的列: {missing_cols}")
            
            # 提取数据
            q = data[pos_cols].values.astype(np.float64)
            dq = data[vel_cols].values.astype(np.float64)
            ddq = data[acc_cols].values.astype(np.float64)
            tau = data[torque_cols].values.astype(np.float64)
            
            print(f"成功提取数据:")
            print(f"  位置列: {pos_cols}")
            print(f"  速度列: {vel_cols}")
            print(f"  加速度列: {acc_cols}")
            print(f"  力矩列: {torque_cols}")
            
        else:
            # 传统格式：假设数据结构为时间戳 + 5个关节位置 + 5个关节速度 + 5个关节加速度 + 5个关节力矩
            print("使用传统格式解析")
            if data.shape[1] >= 21:  # 至少需要21列
                q = data.iloc[:, 1:6].values.astype(np.float64)
                dq = data.iloc[:, 6:11].values.astype(np.float64) 
                ddq = data.iloc[:, 11:16].values.astype(np.float64)
                tau = data.iloc[:, 16:21].values.astype(np.float64)
            else:
                raise ValueError(f"CSV文件列数不足，期望至少21列，实际{data.shape[1]}列")
        
        # 数据范围检查和单位转换提示
        print(f"\n数据范围检查:")
        for i in range(5):
            pos_range = q[:, i].max() - q[:, i].min()
            vel_range = dq[:, i].max() - dq[:, i].min()
            
            print(f"关节{i+1}:")
            print(f"  位置范围: [{q[:, i].min():.6f}, {q[:, i].max():.6f}] (变化: {pos_range:.6f})")
            print(f"  速度范围: [{dq[:, i].min():.6f}, {dq[:, i].max():.6f}] (变化: {vel_range:.6f})")
            print(f"  力矩范围: [{tau[:, i].min():.3f}, {tau[:, i].max():.3f}]")
            
            # 检查数据合理性
            if pos_range > 10:  # 位置变化超过10弧度不合理
                print(f"  ⚠️  位置数据可能不是弧度单位")
            if vel_range < 0.01:  # 速度变化太小
                print(f"  ⚠️  速度激励不足")
            
        print(f"加载数据点数: {len(q)}")
        return q, dq, ddq, tau
    
    def _load_from_npz(self, npz_file):
        """从NPZ文件加载数据"""
        print(f"从NPZ文件加载数据: {npz_file}")
        data = np.load(npz_file)
        q = data['q']
        dq = data['dq'] 
        ddq = data['ddq']
        tau = data['tau'] if 'tau' in data else None
        
        if tau is None:
            raise ValueError("NPZ文件中缺少力矩数据(tau)")
            
        print(f"加载数据点数: {len(q)}")
        return q, dq, ddq, tau
    
    def build_regressor_matrix(self, q, dq, ddq, max_points=None):
        """
        构建回归器矩阵 A_N
        
        Args:
            q, dq, ddq: 关节位置、速度、加速度
            max_points: 最大使用数据点数（用于限制计算量）
            
        Returns:
            A_N: 回归器矩阵 (N*n_joints, n_params)
        """
        N = len(q)
        if max_points is not None and N > max_points:
            # 均匀采样
            indices = np.linspace(0, N-1, max_points, dtype=int)
            q = q[indices]
            dq = dq[indices]
            ddq = ddq[indices]
            N = max_points
            print(f"数据点数限制为: {N}")
        
        print(f"构建回归器矩阵，数据点数: {N}")
        
        # 逐个时间点计算回归器
        regressor_list = []
        for i in range(N):
            if i % 1000 == 0:
                print(f"处理进度: {i}/{N} ({i/N*100:.1f}%)")
            
            # 计算单个时间点的回归器矩阵 (n_joints, n_params)
            regressor = self.calc_dynamics.calc(q[i], dq[i], ddq[i])
            regressor_list.append(regressor)
        
        # 堆叠成 (N*n_joints, n_params)
        A_N = np.vstack(regressor_list)
        
        print(f"回归器矩阵形状: {A_N.shape}")
        return A_N
    
    def identify_base_parameters(self, A_N, tau_N, tolerance=1e-6, regularization=1e-3, method='fast_svd'):
        """
        使用多种方法进行基参数辨识
        
        Args:
            A_N: 回归器矩阵 (N*n_joints, n_params)
            tau_N: 力矩向量 (N*n_joints,)
            tolerance: 分解的容差
            regularization: 正则化参数
            method: 'qr', 'svd', 'fast_svd', 'correlation'
            
        Returns:
            theta_b: 基参数向量
        """
        print(f"\n=== 基参数辨识 (方法: {method}) ===")
        print(f"回归器矩阵形状: {A_N.shape}")
        print(f"力矩向量长度: {len(tau_N)}")
        
        if method == 'qr':
            # 原始QR分解方法
            print("执行QR分解...")
            Q, R, P = qr(A_N, pivoting=True)
            rank = np.sum(np.abs(np.diag(R)) > tolerance)
            self.base_columns = P[:rank]
            
        elif method == 'svd':
            # SVD分解方法
            print("执行SVD分解...")
            U, s, Vt = np.linalg.svd(A_N, full_matrices=False)
            rank = np.sum(s > tolerance)
            # 选择奇异值最大的列
            self.base_columns = np.argsort(s)[::-1][:rank]
            
        elif method == 'fast_svd':
            # 快速SVD方法 - 只计算前k个奇异值
            print("执行快速SVD分解...")
            from scipy.sparse.linalg import svds
            k = min(A_N.shape[1] - 1, 40)  # 估计需要的基参数数量
            try:
                U, s, Vt = svds(A_N, k=k)
                # 按奇异值大小排序
                idx = np.argsort(s)[::-1]
                s = s[idx]
                rank = np.sum(s > tolerance)
                self.base_columns = np.arange(k)[idx][:rank]
            except:
                print("快速SVD失败，回退到标准SVD...")
                U, s, Vt = np.linalg.svd(A_N, full_matrices=False)
                rank = np.sum(s > tolerance)
                self.base_columns = np.argsort(s)[::-1][:rank]
                
        elif method == 'correlation':
            # 改进的相关性方法 - 更严格的条件数控制
            print("执行相关性分析...")
            # 计算每列与力矩向量的相关性
            correlations = np.abs([np.corrcoef(A_N[:, i], tau_N)[0, 1] for i in range(A_N.shape[1])])
            correlations = np.nan_to_num(correlations)  # 处理NaN值
            
            # 选择相关性最高的列
            sorted_indices = np.argsort(correlations)[::-1]
            
            # 逐步添加列，使用更严格的条件数控制
            selected_cols = []
            max_condition = 1e6  # 降低条件数阈值
            
            for idx in sorted_indices:
                test_cols = selected_cols + [idx]
                A_test = A_N[:, test_cols]
                
                if len(test_cols) == 1:
                    selected_cols.append(idx)
                else:
                    cond_num = np.linalg.cond(A_test.T @ A_test)
                    if cond_num < max_condition:
                        selected_cols.append(idx)
                    else:
                        print(f"跳过列{idx}，条件数过大: {cond_num:.2e}")
                
                if len(selected_cols) >= min(30, A_N.shape[1]):  # 减少最大列数
                    break
            
            self.base_columns = np.array(selected_cols)
            rank = len(self.base_columns)
        
        print(f"矩阵秩: {rank}/{A_N.shape[1]}")
        
        # 选择基列
        A_b = A_N[:, self.base_columns]  # 基回归器矩阵
        
        print(f"基参数数量: {len(self.base_columns)}")
        print(f"基回归器矩阵形状: {A_b.shape}")
        
        # 构造从基参数到完整参数的映射矩阵
        n_params = A_N.shape[1]
        self.P_matrix = np.zeros((n_params, rank))
        self.P_matrix[self.base_columns, :] = np.eye(rank)
        
        # 使用改进的正则化最小二乘求解基参数
        # 根据公式(20): Θ = A_N^†(q_N, q̇_N, q̈_N) × τ_N
        print("求解基参数...")
        
        # 数据预处理：标准化回归器矩阵
        A_b_mean = np.mean(A_b, axis=0)
        A_b_std = np.std(A_b, axis=0) + 1e-8  # 避免除零
        A_b_normalized = (A_b - A_b_mean) / A_b_std
        
        # 方法1: 改进的正则化最小二乘
        ATA = A_b_normalized.T @ A_b_normalized
        ATb = A_b_normalized.T @ tau_N
        
        # 自适应正则化：基于条件数调整
        cond_num = np.linalg.cond(ATA)
        if cond_num > 1e8:
            adaptive_reg = regularization * cond_num / 1e8
            print(f"使用自适应正则化: {adaptive_reg:.2e} (条件数: {cond_num:.2e})")
        else:
            adaptive_reg = regularization
        
        theta_b_norm = np.linalg.solve(ATA + adaptive_reg * np.eye(rank), ATb)
        
        # 反标准化参数
        theta_b = theta_b_norm / A_b_std
        
        # 方法2: 伪逆（作为对比）
        theta_b_pinv = pinv(A_b) @ tau_N
        
        print(f"基参数求解完成")
        print(f"正则化解与伪逆解的差异: {np.linalg.norm(theta_b - theta_b_pinv):.6e}")
        
        # 计算拟合误差
        tau_pred = A_b @ theta_b
        residual = tau_N - tau_pred
        rms_error = np.sqrt(np.mean(residual**2))
        
        print(f"RMS拟合误差: {rms_error:.6f}")
        print(f"相对误差: {rms_error/np.std(tau_N)*100:.2f}%")
        
        # 保存结果
        self.identified_params = theta_b
        self.identification_results = {
            'theta_base': theta_b,
            'base_columns': self.base_columns,
            'rank': rank,
            'rms_error': rms_error,
            'relative_error': rms_error/np.std(tau_N)*100,
            'condition_number': np.linalg.cond(A_b),
            'regularization': regularization
        }
        
        return theta_b
    
    def validate_identification(self, q_test, dq_test, ddq_test, tau_test):
        """
        验证辨识结果
        
        Args:
            q_test, dq_test, ddq_test, tau_test: 测试数据
            
        Returns:
            validation_results: 验证结果字典
        """
        if self.identified_params is None:
            raise ValueError("请先进行参数辨识")
        
        print(f"\n=== 验证辨识结果 ===")
        
        # 构建测试数据的回归器矩阵
        A_test = self.build_regressor_matrix(q_test, dq_test, ddq_test)
        A_test_base = A_test[:, self.base_columns]
        
        # 预测力矩
        tau_test_flat = tau_test.flatten()
        tau_pred = A_test_base @ self.identified_params
        
        # 计算验证误差
        residual = tau_test_flat - tau_pred
        rms_error = np.sqrt(np.mean(residual**2))
        relative_error = rms_error / np.std(tau_test_flat) * 100
        
        # 计算每个关节的误差
        joint_errors = []
        for j in range(self.n_joints):
            joint_indices = slice(j, len(residual), self.n_joints)
            joint_residual = residual[joint_indices]
            joint_rms = np.sqrt(np.mean(joint_residual**2))
            joint_errors.append(joint_rms)
        
        validation_results = {
            'rms_error': rms_error,
            'relative_error': relative_error,
            'joint_errors': joint_errors,
            'correlation': np.corrcoef(tau_test_flat, tau_pred)[0, 1]
        }
        
        print(f"验证RMS误差: {rms_error:.6f}")
        print(f"验证相对误差: {relative_error:.2f}%")
        print(f"预测相关系数: {validation_results['correlation']:.4f}")
        
        for j, error in enumerate(joint_errors):
            print(f"关节{j+1}误差: {error:.6f}")
        
        return validation_results
    
    def save_results(self, output_dir="identification_results"):
        """
        保存辨识结果
        
        Args:
            output_dir: 输出目录
        """
        if self.identified_params is None:
            raise ValueError("请先进行参数辨识")
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存参数
        param_file = os.path.join(output_dir, f"identified_params_{timestamp}.npz")
        
        # 创建保存数据字典，避免键名冲突
        save_data = {
            'theta_base': self.identified_params,
            'base_columns': self.base_columns,
            'P_matrix': self.P_matrix,
            'rank': self.identification_results['rank'],
            'rms_error': self.identification_results['rms_error'],
            'relative_error': self.identification_results['relative_error'],
            'condition_number': self.identification_results['condition_number'],
            'regularization': self.identification_results['regularization']
        }
        
        np.savez(param_file, **save_data)
        
        # 保存报告
        report_file = os.path.join(output_dir, f"identification_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== 最小惯性参数辨识报告 ===\n\n")
            f.write(f"辨识时间: {datetime.now()}\n")
            f.write(f"基参数数量: {len(self.base_columns)}\n")
            f.write(f"矩阵秩: {self.identification_results['rank']}\n")
            f.write(f"条件数: {self.identification_results['condition_number']:.2e}\n")
            f.write(f"RMS误差: {self.identification_results['rms_error']:.6f}\n")
            f.write(f"相对误差: {self.identification_results['relative_error']:.2f}%\n")
            f.write(f"正则化参数: {self.identification_results['regularization']:.2e}\n\n")
            
            f.write("基参数索引:\n")
            for i, col in enumerate(self.base_columns):
                f.write(f"  {i+1}: 原参数{col+1}\n")
            
            f.write("\n基参数值:\n")
            for i, param in enumerate(self.identified_params):
                f.write(f"  θ_{i+1}: {param:.6f}\n")
        
        print(f"结果已保存到: {output_dir}")
        return param_file, report_file

def run_identification_example(data_file, max_points=10000, method='fast_svd'):
    """
    运行参数辨识示例
    
    Args:
        data_file: 数据文件路径
        max_points: 最大使用数据点数
        method: 辨识方法 ('qr', 'svd', 'fast_svd', 'correlation')
    """
    print("=== 最小惯性参数辨识示例 ===")
    
    # 初始化辨识器
    identifier = MinimumParameterIdentification()
    
    try:
        # 加载数据
        q, dq, ddq, tau = identifier.load_motion_data(data_file)
        
        # 限制数据点数
        if len(q) > max_points:
            print(f"数据点数({len(q)})超过限制，将使用前{max_points}个点")
            q = q[:max_points]
            dq = dq[:max_points]
            ddq = ddq[:max_points]
            tau = tau[:max_points]
        
        print(f"使用数据点数: {len(q)}")
        
        # 构建回归器矩阵
        A_N = identifier.build_regressor_matrix(q, dq, ddq)
        tau_N = tau.flatten()  # 展平为一维向量
        
        # 进行参数辨识
        theta_b = identifier.identify_base_parameters(A_N, tau_N, method=method)
        
        # 保存结果
        param_file, report_file = identifier.save_results()
        
        print(f"\n=== 辨识完成 ===")
        print(f"参数文件: {param_file}")
        print(f"报告文件: {report_file}")
        
        return identifier
        
    except Exception as e:
        print(f"辨识过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_methods(data_file, max_points=1000):
    """
    比较不同方法的性能
    
    Args:
        data_file: 数据文件路径
        max_points: 测试数据点数
    """
    import time
    
    print("=== 方法性能比较 ===")
    
    # 初始化辨识器
    identifier = MinimumParameterIdentification()
    
    # 加载数据
    q, dq, ddq, tau = identifier.load_motion_data(data_file)
    if len(q) > max_points:
        q = q[:max_points]
        dq = dq[:max_points]
        ddq = ddq[:max_points]
        tau = tau[:max_points]
    
    # 构建回归器矩阵
    A_N = identifier.build_regressor_matrix(q, dq, ddq)
    tau_N = tau.flatten()
    
    methods = ['correlation', 'fast_svd', 'svd', 'qr']
    results = {}
    
    for method in methods:
        print(f"\n--- 测试方法: {method} ---")
        start_time = time.time()
        
        try:
            theta_b = identifier.identify_base_parameters(A_N, tau_N, method=method)
            elapsed_time = time.time() - start_time
            
            # 计算拟合误差
            A_b = A_N[:, identifier.base_columns]
            tau_pred = A_b @ theta_b
            rms_error = np.sqrt(np.mean((tau_N - tau_pred)**2))
            
            results[method] = {
                'time': elapsed_time,
                'rank': len(identifier.base_columns),
                'rms_error': rms_error,
                'condition_number': np.linalg.cond(A_b)
            }
            
            print(f"耗时: {elapsed_time:.3f}秒")
            print(f"基参数数量: {len(identifier.base_columns)}")
            print(f"RMS误差: {rms_error:.6f}")
            
        except Exception as e:
            print(f"方法 {method} 失败: {e}")
            results[method] = {'error': str(e)}
    
    # 输出比较结果
    print(f"\n=== 性能比较总结 ===")
    print(f"{'方法':<12} {'耗时(秒)':<10} {'基参数':<8} {'RMS误差':<12} {'条件数':<12}")
    print("-" * 60)
    
    for method, result in results.items():
        if 'error' not in result:
            print(f"{method:<12} {result['time']:<10.3f} {result['rank']:<8} {result['rms_error']:<12.6f} {result['condition_number']:<12.2e}")
        else:
            print(f"{method:<12} 失败")
    
    return results

def load_all_dynamics_data(data_dir="/Users/lr-2002/project/instantcreation/IC_arm_control/dyn_ana"):
    """
    加载指定目录下的所有动力学数据CSV文件并合并
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        合并后的DataFrame，如果没有找到文件则返回None
    """
    import os
    import glob
    
    # 查找所有CSV文件
    csv_pattern = os.path.join(data_dir, "dynamics_data_*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"在目录 {data_dir} 中未找到动力学数据文件")
        return None
    
    print(f"找到 {len(csv_files)} 个动力学数据文件:")
    combined_data = []
    
    for i, csv_file in enumerate(sorted(csv_files)):
        filename = os.path.basename(csv_file)
        print(f"  {i+1}. {filename}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 添加文件标识列
            df['data_source'] = filename
            df['file_index'] = i
            
            combined_data.append(df)
            print(f"     ✓ 加载成功: {len(df)} 个数据点")
            
        except Exception as e:
            print(f"     ✗ 加载失败: {e}")
            continue
    
    if not combined_data:
        print("所有文件加载失败")
        return None
    
    # 合并所有数据
    print(f"\n合并 {len(combined_data)} 个数据文件...")
    merged_df = pd.concat(combined_data, ignore_index=True)
    
    print(f"✓ 合并完成: 总共 {len(merged_df)} 个数据点")
    print(f"  数据时间跨度: {merged_df['time'].min():.2f}s - {merged_df['time'].max():.2f}s")
    print(f"  数据来源文件数: {merged_df['file_index'].nunique()}")
    
    return merged_df

def run_comprehensive_identification(data_dir="/Users/lr-2002/project/instantcreation/IC_arm_control/dyn_ana", 
                                   max_points=None, method='correlation'):
    """
    使用所有可用数据进行综合参数辨识
    
    Args:
        data_dir: 数据目录路径
        max_points: 最大使用数据点数（None表示使用所有数据）
        method: 辨识方法
        
    Returns:
        辨识器实例
    """
    print("=== IC ARM 综合动力学参数辨识 ===\n")
    
    # 加载所有数据
    combined_data = load_all_dynamics_data(data_dir)
    if combined_data is None:
        return None
    
    # 限制数据点数（如果指定）
    if max_points and len(combined_data) > max_points:
        print(f"\n数据点过多，随机采样 {max_points} 个点进行辨识...")
        combined_data = combined_data.sample(n=max_points, random_state=42).reset_index(drop=True)
        print(f"✓ 采样完成: {len(combined_data)} 个数据点")
    
    # 数据统计信息
    print(f"\n=== 数据统计信息 ===")
    print(f"总数据点数: {len(combined_data)}")
    print(f"数据来源文件: {combined_data['file_index'].nunique()} 个")
    print(f"时间范围: {combined_data['time'].min():.2f}s - {combined_data['time'].max():.2f}s")
    
    # 显示各电机的数据范围
    for motor_id in range(1, 6):
        pos_col = f'm{motor_id}_pos_actual'
        if pos_col in combined_data.columns:
            pos_data = combined_data[pos_col]
            pos_range = np.degrees([pos_data.min(), pos_data.max()])
            print(f"电机 {motor_id} 位置范围: {pos_range[0]:.1f}° ~ {pos_range[1]:.1f}°")
    
    # 执行参数辨识
    print(f"\n=== 开始参数辨识 (方法: {method}) ===")
    
    try:
        # 创建辨识器
        identifier = MinimumParameterIdentification()
        
        # 保存合并数据到临时CSV文件
        temp_csv_file = "temp_combined_data.csv"
        combined_data.to_csv(temp_csv_file, index=False)
        print(f"✓ 临时数据文件已创建: {temp_csv_file}")
        
        # 加载数据
        q, dq, ddq, tau = identifier.load_motion_data(temp_csv_file)
        print(f"✓ 数据加载完成")
        
        # 构建回归器矩阵
        print(f"构建回归器矩阵...")
        A_N = identifier.build_regressor_matrix(q, dq, ddq)
        tau_N = tau.flatten()
        
        # 执行辨识
        start_time = time.time()
        params = identifier.identify_base_parameters(A_N, tau_N, method=method)
        identification_time = time.time() - start_time
        
        if params is not None:
            print(f"✓ 参数辨识成功!")
            print(f"  辨识时间: {identification_time:.2f}s")
            print(f"  辨识参数数量: {len(params)}")
            print(f"  参数范围: [{params.min():.6f}, {params.max():.6f}]")
            
            # 保存结果
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            result_file = f"identification_results/comprehensive_params_{timestamp}.npz"
            
            # 手动保存结果
            os.makedirs("identification_results", exist_ok=True)
            np.savez(result_file, 
                    identified_params=params,
                    base_columns=identifier.base_columns if hasattr(identifier, 'base_columns') else None,
                    identification_time=identification_time,
                    method=method,
                    data_points=len(combined_data))
            print(f"✓ 结果已保存到: {result_file}")
            
            # 生成综合报告
            report_file = f"identification_results/comprehensive_report_{timestamp}.txt"
            generate_comprehensive_report(identifier, combined_data, report_file, params, identification_time)
            print(f"✓ 综合报告已保存到: {report_file}")
            
        else:
            print("✗ 参数辨识失败")
            return None
            
    except Exception as e:
        print(f"✗ 辨识过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return identifier

def generate_comprehensive_report(identifier, data_df, report_file, params, identification_time):
    """生成综合辨识报告"""
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== IC ARM 综合动力学参数辨识报告 ===\n\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 数据信息
        f.write("【数据信息】\n")
        f.write(f"总数据点数: {len(data_df)}\n")
        f.write(f"数据来源文件数: {data_df['file_index'].nunique()}\n")
        f.write(f"时间跨度: {data_df['time'].min():.2f}s - {data_df['time'].max():.2f}s\n")
        
        # 各文件贡献
        f.write(f"\n【各文件数据贡献】\n")
        file_stats = data_df.groupby('data_source').size().sort_values(ascending=False)
        for filename, count in file_stats.items():
            percentage = (count / len(data_df)) * 100
            f.write(f"{filename}: {count} 点 ({percentage:.1f}%)\n")
        
        # 电机运动范围
        f.write(f"\n【电机运动范围】\n")
        for motor_id in range(1, 6):
            pos_col = f'm{motor_id}_pos_actual'
            if pos_col in data_df.columns:
                pos_data = data_df[pos_col]
                pos_range = np.degrees([pos_data.min(), pos_data.max()])
                pos_std = np.degrees(pos_data.std())
                f.write(f"电机 {motor_id}: {pos_range[0]:.1f}° ~ {pos_range[1]:.1f}° (标准差: {pos_std:.1f}°)\n")
        
        # 辨识结果
        if params is not None:
            f.write(f"\n【辨识结果】\n")
            f.write(f"辨识时间: {identification_time:.2f}s\n")
            f.write(f"参数数量: {len(params)}\n")
            f.write(f"参数范围: [{params.min():.6f}, {params.max():.6f}]\n")
            f.write(f"参数标准差: {params.std():.6f}\n")
            f.write(f"参数均值: {params.mean():.6f}\n")
            
            # 参数分布统计
            f.write(f"\n【参数分布统计】\n")
            f.write(f"正参数数量: {np.sum(params > 0)}\n")
            f.write(f"负参数数量: {np.sum(params < 0)}\n")
            f.write(f"零参数数量: {np.sum(np.abs(params) < 1e-10)}\n")
        
        f.write(f"\n报告生成完成。\n")

if __name__ == "__main__":
    # 综合参数辨识：使用所有可用数据
    data_dir = "/Users/lr-2002/project/instantcreation/IC_arm_control/dyn_ana"
    
    print("=== IC ARM 综合动力学参数辨识 ===\n")
    
    # 首先进行方法性能比较（使用部分数据）
    print("1. 方法性能比较测试...")
    sample_files = glob.glob(os.path.join(data_dir, "dynamics_data_*.csv"))
    if sample_files:
        sample_file = sample_files[0]  # 使用第一个文件进行测试
        print(f"使用样本文件: {os.path.basename(sample_file)}")
        compare_methods(sample_file, max_points=3000)
    
    print("\n" + "="*80)
    
    # 综合辨识：使用所有数据
    print("\n2. 综合参数辨识...")
    identifier = run_comprehensive_identification(
        data_dir=data_dir, 
        max_points=15000,  # 限制最大数据点数以控制计算时间
        method='fast_svd'  # 使用改进的相关性方法
    )