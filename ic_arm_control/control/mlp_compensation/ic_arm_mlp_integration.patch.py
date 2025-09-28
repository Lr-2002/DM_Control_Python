
# ===== IC_ARM.py MLP重力补偿集成补丁 =====

# 1. 在文件顶部的import部分添加:
import sys
from pathlib import Path

# 添加mlp_compensation模块路径
current_dir = Path(__file__).parent
mlp_compensation_dir = current_dir / "mlp_compensation"
if mlp_compensation_dir.exists() and str(mlp_compensation_dir) not in sys.path:
    sys.path.append(str(mlp_compensation_dir))

# 2. 修改__init__方法参数 (大约在第92行):
def __init__(
    self, device_sn="F561E08C892274DB09496BCC1102DBC5", debug=False, gc=False,
    gc_type="static",  # 新增参数：重力补偿类型
    enable_buffered_control=True, control_freq=300
):
    """Initialize IC ARM with unified motor control system

    Args:
        device_sn: 设备序列号
        debug: 调试模式
        gc: 是否启用重力补偿
        gc_type: 重力补偿类型 ("static" 或 "mlp")
        enable_buffered_control: 启用缓冲控制
        control_freq: 控制频率
    """
    self.debug = debug
    self.use_ht = True
    self.enable_buffered_control = enable_buffered_control
    self.control_freq = control_freq
    self.gc_type = gc_type  # 存储重力补偿类型
    debug_print("=== 初始化IC_ARM_Unified ===")

    # ... (现有的初始化代码保持不变)

    # 3. 修改重力补偿初始化部分 (大约在第163行):
    self.gc_flag = gc
    if self.gc_flag:
        debug_print(f"初始化重力补偿系统，类型: {gc_type}")

        if gc_type == "mlp":
            # 使用MLP重力补偿
            try:
                from mlp_gravity_integrator import MLPGravityCompensation
                # 模型路径相对于IC_ARM.py的位置
                model_path = current_dir / "mlp_compensation" / "mlp_gravity_model_improved.pkl"
                self.gc = MLPGravityCompensation(
                    model_path=str(model_path),
                    enable_enhanced=True,
                    debug=debug
                )
                debug_print("✅ MLP重力补偿初始化成功")
            except Exception as e:
                debug_print(f"❌ MLP重力补偿初始化失败: {e}，回退到静态补偿")
                from utils.static_gc import StaticGravityCompensation
                self.gc = StaticGravityCompensation()
                self.gc_type = "static"
        else:
            # 使用原有的静态重力补偿
            from utils.static_gc import StaticGravityCompensation
            self.gc = StaticGravityCompensation()
            debug_print("✅ 静态重力补偿初始化成功")
    else:
        self.gc = None
        debug_print("重力补偿未启用")

# 4. 添加MLP重力补偿专用方法 (在cal_gravity方法附近):
def cal_gravity_mlp(self):
    """使用MLP计算重力补偿力矩"""
    if not self.gc_flag or self.gc_type != "mlp":
        return np.zeros(self.motor_count)

    try:
        self._refresh_all_states_ultra_fast()
        # MLP重力补偿只需要位置信息
        positions = self.q[:6]  # 前6个关节
        compensation_torque = self.gc.get_gravity_compensation_torque(positions)

        # 扩展到所有电机（保持与原有接口兼容）
        full_compensation = np.zeros(self.motor_count)
        full_compensation[:6] = compensation_torque

        return full_compensation
    except Exception as e:
        debug_print(f"MLP重力补偿计算失败: {e}", "ERROR")
        return np.zeros(self.motor_count)

def switch_to_mlp_gravity_compensation(self):
    """切换到MLP重力补偿模式"""
    if not self.gc_flag:
        debug_print("重力补偿未启用", "ERROR")
        return False

    try:
        from mlp_gravity_integrator import MLPGravityCompensation
        model_path = Path(__file__).parent / "mlp_compensation" / "mlp_gravity_model_improved.pkl"

        self.gc = MLPGravityCompensation(
            model_path=str(model_path),
            enable_enhanced=True,
            debug=self.debug
        )
        self.gc_type = "mlp"
        debug_print("✅ 已切换到MLP重力补偿模式")
        return True
    except Exception as e:
        debug_print(f"切换到MLP重力补偿失败: {e}", "ERROR")
        return False

def switch_to_static_gravity_compensation(self):
    """切换到静态重力补偿模式"""
    if not self.gc_flag:
        debug_print("重力补偿未启用", "ERROR")
        return False

    try:
        from utils.static_gc import StaticGravityCompensation
        self.gc = StaticGravityCompensation()
        self.gc_type = "static"
        debug_print("✅ 已切换到静态重力补偿模式")
        return True
    except Exception as e:
        debug_print(f"切换到静态重力补偿失败: {e}", "ERROR")
        return False

def get_gravity_compensation_performance(self):
    """获取重力补偿性能统计"""
    if not self.gc_flag or self.gc_type != "mlp":
        return None

    try:
        return self.gc.get_performance_stats()
    except Exception as e:
        debug_print(f"获取性能统计失败: {e}", "ERROR")
        return None

def print_gravity_compensation_summary(self):
    """打印重力补偿性能摘要"""
    if not self.gc_flag:
        print("重力补偿未启用")
        return

    print(f"=== 重力补偿系统状态 ===")
    print(f"类型: {self.gc_type}")
    print(f"状态: {'启用' if self.gc_flag else '禁用'}")

    if self.gc_type == "mlp":
        try:
            self.gc.print_performance_summary()
        except Exception as e:
            print(f"MLP性能统计获取失败: {e}")

# 5. 修改cal_gravity方法以支持MLP (大约在第1148行):
def cal_gravity(self):
    """计算重力补偿力矩"""
    if not self.gc_flag:
        return np.zeros(self.motor_count)

    if self.gc_type == "mlp":
        return self.cal_gravity_mlp()
    else:
        # 原有的静态重力补偿逻辑
        self._refresh_all_states_ultra_fast()
        return self.gc.get_gravity_compensation_torque(self.q)

# ===== 集成补丁结束 =====
