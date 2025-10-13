# 动力学参数辨识流程

基于 [shamilmamedov/dynamic_calibration](https://github.com/shamilmamedov/dynamic_calibration) 的完整实现

## 📋 概述

本项目实现了机器人动力学参数辨识的完整流程，严格遵循 `shamilmamedov/dynamic_calibration` 的方法论。

### 核心流程

```
原始日志数据 (q, dq, tau)
    ↓
Step 1: 数据预处理 (filterData.m)
    - 零相位滤波
    - 中心差分法估计加速度
    ↓
Step 2: 参数估计 (ordinaryLeastSquareEstimation)
    - 构建回归矩阵
    - 最小二乘求解
    ↓
Step 3: 验证 (Validation)
    - 在新轨迹上测试
    - 生成验证报告
```

## 🚀 快速开始

### 1. 准备数据

确保你有以下日志数据：
```
/Users/lr-2002/project/instantcreation/IC_arm_control/logs/20251010_133145_ic_arm_control/
├── motor_states.csv
└── joint_commands.csv

/Users/lr-2002/project/instantcreation/IC_arm_control/logs/20251010_132725_ic_arm_control/
├── motor_states.csv
└── joint_commands.csv
```

### 2. 运行完整流程

```bash
cd /Users/lr-2002/project/instantcreation/IC_arm_control/ic_arm_control/control/di_data_from_gh
python run_identification.py
```

### 3. 或者分步运行

```bash
# Step 1: 数据预处理
python step1_data_preprocessing.py

# Step 2: 参数估计
python step2_parameter_estimation.py

# Step 3: 验证
python step3_validation.py
```

## 📂 文件结构

```
di_data_from_gh/
├── step1_data_preprocessing.py      # 数据预处理 (filterData.m)
├── step2_parameter_estimation.py    # 参数估计 (OLS)
├── step3_validation.py              # 验证
├── run_identification.py            # 主控脚本
├── README_IDENTIFICATION.md         # 本文档
│
├── processed_data/                  # Step 1 输出
│   ├── 20251010_133145_ic_arm_control_filtered.csv
│   └── 20251010_132725_ic_arm_control_filtered.csv
│
├── estimation_results/              # Step 2 输出
│   ├── estimated_parameters.npz
│   └── prediction_results.png
│
└── validation_results/              # Step 3 输出
    ├── validation_report.txt
    ├── 20251010_133145_ic_arm_control_validation.png
    └── 20251010_132725_ic_arm_control_validation.png
```

## 🔬 技术细节

### Step 1: 数据预处理

**实现的功能 (对应 `filterData.m`):**

1. **零相位滤波** - 使用 `scipy.signal.filtfilt`
   ```python
   # 避免引入相位延迟
   filtered_data = filtfilt(b, a, data)
   ```

2. **中心差分法估计加速度**
   ```python
   # 更准确的加速度估计
   acc[i] = (vel[i+1] - vel[i-1]) / (2*dt)
   ```

3. **滤波器参数**
   - 截止频率: 10 Hz (可调整)
   - 采样频率: 250 Hz
   - 滤波器阶数: 4

**关键参数调整:**

在 `step1_data_preprocessing.py` 中修改:
```python
preprocessor = DataPreprocessor(
    cutoff_freq=10.0,  # 截止频率 (Hz) - 根据数据调整
    fs=250.0,          # 采样频率 (Hz)
    filter_order=4     # 滤波器阶数
)
```

### Step 2: 参数估计

**实现的功能 (对应 `ordinaryLeastSquareEstimation`):**

1. **构建回归矩阵**
   ```
   τ = Y(q, dq, ddq) * θ_base
   
   其中:
   - τ: 力矩向量
   - Y: 回归矩阵
   - θ_base: 基参数向量
   ```

2. **最小二乘求解**
   ```python
   θ_base = (Y^T * Y)^(-1) * Y^T * τ
   ```

3. **性能评估**
   - RMSE (均方根误差)
   - MAE (平均绝对误差)
   - R² score
   - 最大误差

**注意事项:**

- 当前使用简化的回归矩阵
- 实际应用中需要根据 URDF 模型生成完整的回归矩阵
- 条件数过大 (>1e10) 表示数据质量不佳

### Step 3: 验证

**实现的功能 (对应 Validation):**

1. 在新轨迹上测试参数
2. 预测力矩 vs 测量力矩对比
3. 生成详细的验证报告
4. 可视化预测结果

## 📊 输出结果

### 1. 预处理数据 (`processed_data/`)

CSV 文件包含:
- `time`: 时间序列
- `q1-q6`: 滤波后的位置
- `dq1-dq6`: 滤波后的速度
- `ddq1-ddq6`: 估计并滤波后的加速度
- `tau1-tau6`: 滤波后的力矩

### 2. 估计参数 (`estimation_results/`)

**estimated_parameters.npz** 包含:
- `theta_base`: 估计的基参数
- `rmse`, `mae`, `max_error`: 各关节误差
- `r2_score`: R² 分数
- `mean_rmse`, `mean_mae`, `mean_r2`: 平均性能

**加载方式:**
```python
import numpy as np
data = np.load('estimated_parameters.npz')
theta_base = data['theta_base']
```

### 3. 验证结果 (`validation_results/`)

**validation_report.txt** 包含:
- 每个轨迹的详细性能指标
- 每个关节的 RMSE, MAE, R²
- 总体性能评估

## 🎯 与原项目的对应关系

| 原项目 (MATLAB) | 本实现 (Python) | 功能 |
|----------------|----------------|------|
| `filterData.m` | `step1_data_preprocessing.py` | 数据滤波和加速度估计 |
| `ordinaryLeastSquareEstimation` | `step2_parameter_estimation.py` | 最小二乘参数估计 |
| `ur_idntfcn_real.m` | `step2_parameter_estimation.py` | 参数辨识主脚本 |
| Validation | `step3_validation.py` | 验证 |

## 🔧 常见问题

### Q1: 预测精度不满意怎么办?

**A:** 按以下顺序检查:

1. **调整滤波器参数**
   ```python
   # 在 step1_data_preprocessing.py 中
   cutoff_freq=10.0  # 降低截止频率可以更平滑，但可能丢失高频信息
   filter_order=4    # 增加阶数可以更陡峭的滤波
   ```

2. **检查数据质量**
   - 轨迹是否充分激励所有关节?
   - 是否包含足够的运动范围?
   - 传感器噪声是否过大?

3. **收集更多数据**
   - 使用轨迹优化生成激励轨迹
   - 确保持续激励条件 (Persistent Excitation)

### Q2: 条件数过大怎么办?

**A:** 条件数过大 (>1e10) 表示回归矩阵接近奇异:

1. 收集更多样化的轨迹数据
2. 使用轨迹优化 (最小化条件数)
3. 考虑使用正则化方法

### Q3: 如何使用估计的参数?

**A:** 估计的参数可用于:

1. **重力补偿**
   ```python
   tau_gravity = Y_gravity @ theta_base
   ```

2. **前馈控制**
   ```python
   tau_feedforward = Y(q_desired, dq_desired, ddq_desired) @ theta_base
   ```

3. **模型预测控制 (MPC)**

## 📚 参考文献

1. [shamilmamedov/dynamic_calibration](https://github.com/shamilmamedov/dynamic_calibration)
2. Swevers et al. "Optimal robot excitation and identification"
3. Khalil & Dombre "Modeling, Identification and Control of Robots"

## 🛠️ 依赖项

```bash
pip install numpy pandas scipy matplotlib
```

## 📝 许可

本实现遵循原项目的方法论，用于学术研究和教育目的。

---

**作者:** IC ARM Control Team  
**日期:** 2025-10-10  
**版本:** 1.0.0
