# 静态数据动力学辨识解决方案

## 问题描述
当使用静态数据（速度接近零）进行动力学辨识时，传统的辨识方法效果很差，因为：
1. 缺乏动态激励信号
2. 惯性和摩擦参数难以辨识
3. 特征矩阵缺乏信息量

## 解决方案

### 1. 新增静态数据专用模式
```python
# 自动检测数据类型
identifier = MultiJointIdentification(n_joints=6)
results = identifier.identify_all_joints(data, data_mode="auto")

# 强制使用静态数据模式
results = identifier.identify_all_joints(data, data_mode="static")
```

### 2. 智能模型选择
系统会根据数据质量自动选择最适合的模型：

#### 常规静态数据（力矩变化 > 0.05Nm）
- 静态数据专用特征：sin(q), cos(q), sin(2q), cos(2q), sin(3q), cos(3q)
- 静摩擦特征：极敏感的方向检测
- 位置非线性项：q², q³
- RidgeCV交叉验证优化

#### 低激励静态数据（力矩变化 < 0.05Nm）
- 简化模型：仅使用 constant, sin(q), cos(q)
- 强正则化（alpha=0.1）
- 专注于重力补偿的基本参数

### 3. 物理约束优化
- 自动检测数据质量
- 根据力矩变化范围选择模型复杂度
- 物理合理性验证
- 提供数据质量警告和建议

### 4. 使用方法

#### 命令行使用
```bash
# 自动检测数据类型
python multi_joint_identification.py auto

# 强制静态数据模式
python multi_joint_identification.py static

# 强制动态数据模式
python multi_joint_identification.py dynamic
```

#### Python API使用
```python
from multi_joint_identification import MultiJointIdentification, load_data_from_csv

# 加载静态数据
data = load_data_from_csv("your_static_data.csv")

# 创建辨识器
identifier = MultiJointIdentification(n_joints=6)

# 使用静态数据专用方法
results = identifier.identify_all_joints(
    data,
    regularization=0.01,
    data_mode="static"
)

# 查看结果
for result in results:
    if result:
        print(f"Joint {result['joint_id']}: R² = {result['r2']:.4f}")
```

## 期望效果

1. **更好的重力参数辨识**：静态数据主要用于重力补偿参数辨识
2. **避免过拟合**：通过强正则化防止噪声数据的影响
3. **物理验证**：检查位置和力矩变化范围是否合理
4. **专门的特征工程**：针对静态数据优化的特征矩阵

## 注意事项

1. **位置变化范围**：静态数据需要有足够的位置变化（建议>3度）才能有效辨识重力参数
2. **力矩变化范围**：力矩变化应该足够明显（建议>0.1Nm）
3. **数据质量**：静态数据通常比动态数据更干净，但仍需去除异常值
4. **适用场景**：此方法主要适用于重力参数辨识，惯性参数仍需动态数据