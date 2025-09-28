# 激励轨迹设计指导文档（Excitation Trajectory Design Guide）

> 基于论文 *“Excitation Trajectory Optimization for Dynamic Parameter Identification Using Virtual Constraints in Hands-on Robotic System”* 的复现与扩展实现  
> 可用于任意含 URDF 模型的机械臂（KUKA, Franka, UR5, etc.）

---

## 🧩 一、设计目标

通过生成一条 **激励轨迹 (excitation trajectory)**，让机器人执行时能充分激发系统的动力学模式，从而使参数辨识更准确、稳定、快速收敛。

要求：

- ✅ 无自碰撞、末端不与工具相撞  
- ✅ 满足关节上下限与速度约束  
- ✅ 轨迹周期性、连续且可平滑执行  
- ✅ 激励矩阵条件数低（表示轨迹信息丰富）

---

## 📚 二、理论基础

### 1. 动力学模型

机器人动力学方程：
$$
\tau = Y(q, \dot{q}, \ddot{q}) \, \theta
$$
其中：

- \(\tau \in \mathbb{R}^n\)：关节力矩向量  
- \(Y(q, \dot{q}, \ddot{q})\)：回归矩阵（Regression matrix）  
- \(\theta\)：动力学参数（质量、惯量、摩擦等）

---

### 2. 激励质量指标

轨迹的“激励程度”通过 \(Y_b^T Y_b\) 的**条件数**衡量。  
理想情况下：
$$
\text{cond}(Y_b^T Y_b) \text{ 越小越好}
$$

为便于优化，论文采用了替代指标：

$$
r_c = \|Y_b^T Y_b\|_F + \|(Y_b^T Y_b)^{-1}\|_F
$$

其中：

- \(\|\cdot\|_F\)：Frobenius 范数  
- \(r_c\)：比直接优化条件数更稳定、易求导的目标函数  

目标函数定义为：
$$
J = \frac{1}{r_c}
$$
即希望 **最小化 \(r_c\)** 或 **最大化 \(1/r_c\)**。

---

### 3. 轨迹参数化（Fourier series）

每个关节角 \(q_i(t)\) 被参数化为有限项傅里叶级数：

$$
q_i(t) = q_{i0} + \sum_{k=1}^{N_h} a_{ik} \sin(k\omega t) + b_{ik} \cos(k\omega t)
$$

其中：

- \(N_h\)：谐波数（一般取 3～5）  
- \(\omega = \frac{2\pi}{T}\)：轨迹基本频率  
- 参数向量：\(\{a_{ik}, b_{ik}$\) 为优化变量  

优点：

- 自动保证轨迹平滑、周期性  
- 可精确控制幅值与速度范围

---

### 4. 约束条件

1. **起终点一致（周期性）**  
   $$
   q_i(0) = q_i(T)
   $$

2. **关节范围限制**  
   $$
   q_{\min} \le q_i(t) \le q_{\max}
   $$

3. **速度限制**  
   $$
   |\dot{q}_i(t)| \le \dot{q}_{\max}
   $$

4. **自碰撞约束**  
   通过 URDF 模型生成每个 link 的椭球或 capsule 边界体，确保：
   $$
   \text{dist}(L_j, L_k) > d_{\text{safe}}
   $$
   即所有 link 间距离大于安全距离。

---

## ⚙️ 三、实现流程

### Step 1. 准备 URDF 模型

- 获取机器人的 `xxx.urdf` 文件  
- 使用 `pinocchio`, `pybullet`, 或 `urdf_parser_py` 解析  
- 从模型中提取：
  - 关节数量 n  
  - 每个关节的上下限与类型（revolute / prismatic）  
  - 链接的几何（用于碰撞检测）

```python
import pinocchio as pin
model = pin.buildModelFromUrdf("franka_panda.urdf")
data = model.createData()
```

---

### Step 2. 初始化轨迹参数

设：

```python
num_harmonics = 4
T = 6.0   # 轨迹周期(s)
omega = 2 * np.pi / T
```

为每个关节随机初始化：

```python
a = np.random.uniform(-0.3, 0.3, (n, num_harmonics))
b = np.random.uniform(-0.3, 0.3, (n, num_harmonics))
```

---

### Step 3. 构建优化问题

**目标函数：**

```python
def objective(params):
    Yb = compute_regression_matrix(q, dq, ddq)
    rc = np.linalg.norm(Yb.T @ Yb, 'fro') + np.linalg.norm(np.linalg.inv(Yb.T @ Yb), 'fro')
    return rc
```

**约束：**

- Joint limits (from URDF)
- Collision avoidance (via capsule distance)
- Periodicity

**优化器：**
推荐使用 [IPOPT](https://coin-or.github.io/Ipopt/) 或 [CasADi](https://web.casadi.org/)

---

### Step 4. 求解优化问题

```python
import casadi as ca

opti = ca.Opti()
x = opti.variable(param_dim)
opti.minimize(objective(x))
# 添加约束 opti.subject_to(...)
opti.solver('ipopt')
sol = opti.solve()
```

---

### Step 5. 验证生成轨迹

- 在仿真环境中播放轨迹（如 PyBullet / IsaacGym）  
- 检查是否：
  - 无自碰撞  
  - 轨迹平滑  
  - 速度/加速度均在限制内  

---

### Step 6. 执行与数据采集

采集：

- 关节角 q(t)  
- 关节速度 \(\dot{q}(t)\)  
- 力矩 \(\tau(t)\)

可选地通过 tracking differentiator / 滤波器提取平滑加速度。

---

### Step 7. 参数辨识

用最小二乘求解：
$$
\hat{\theta} = (Y^T Y)^{-1} Y^T \tau
$$

或带约束的：
$$
\hat{\theta} = \arg\min_\theta \|Y\theta - \tau\|^2
$$

---

## 💡 四、实践建议

| 项目 | 建议 |
|------|------|
| 谐波数量 \(N_h\) | 3～5 即可，越多越复杂 |
| 周期时间 \(T\) | 4～8 秒较合适 |
| 优化初始化 | 多起点（multi-start）可提升收敛性 |
| 碰撞检测 | 建议用 capsule/ellipsoid 简化 link 几何 |
| 轨迹采样率 | ≥ 200 Hz（采样越密，辨识越稳定） |
| 噪声处理 | 一阶 tracking differentiator 或 Savitzky–Golay 滤波器 |

---

## ⚠️ 五、注意事项

- IPOPT 求解可能陷入局部最优，可多次运行随机初值  
- 若 URDF 较复杂（含末端工具），碰撞检测可能成为计算瓶颈  
- 在实际机器人执行时需加速度/力矩限幅器，防止超限  
- 对冗余机械臂，可选激励部分关节或添加正则项抑制无意义摆动  

---

## ✅ 六、参考文献

1. **Zhang, et al. (2024)**. *Excitation Trajectory Optimization for Dynamic Parameter Identification Using Virtual Constraints in Hands-on Robotic System*. arXiv:2401.16566.  
2. **Gautier, M. (1997)**. *Dynamic Identification of Robots with Geometric and Inertial Parameters*. IEEE Trans. Robotics and Automation.  
3. **Atkeson, et al. (1986)**. *Estimation of Inertial Parameters of Manipulator Loads and Links*. Int. J. Robotics Research.

---

## 🧭 七、总结

> 通过基于 URDF 的激励轨迹优化方法，可为任意机器人自动生成**免碰撞、信息丰富、辨识友好**的激励轨迹。  
>
> 优化目标是平衡“激励充分性”和“物理可行性”，最终使动力学辨识更精准，从而提升控制性能。

---
