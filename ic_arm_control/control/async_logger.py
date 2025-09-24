"""
异步日志管理器 - 不干扰主线程的日志记录系统
专为IC ARM控制系统设计的高性能日志记录模块
"""

import time
import threading
import queue
import json
import csv
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List


class AsyncLogManager:
    """异步日志管理器 - 不干扰主线程的日志记录系统"""
    
    def __init__(self, log_dir: str = "logs", log_name: str = "ic_arm_log", 
                 save_json: bool = False, save_csv: bool = True, max_queue_size: int = 1000):
        """
        初始化异步日志管理器
        
        Args:
            log_dir: 日志目录路径
            log_name: 日志文件名称（不包含扩展名）
            save_json: 是否保存JSON格式日志
            save_csv: 是否保存CSV格式日志
            max_queue_size: 日志队列最大大小，防止内存溢出
        """
        self.log_dir = Path(log_dir)
        self.log_name = log_name
        self.save_json = save_json
        self.save_csv = save_csv
        
        # 生成时间戳文件名
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.json_file_path = self.log_dir / f"{log_name}_{self.timestamp}.jsonl" if save_json else None
        self.csv_file_path = self.log_dir / f"{log_name}_{self.timestamp}.csv" if save_csv else None
        
        self.log_queue = queue.Queue(maxsize=max_queue_size)
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.is_running = False
        
        # 确保日志目录存在
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 性能统计
        self.total_logs = 0
        self.dropped_logs = 0
        
        # CSV文件头部标志
        self.csv_headers_written = {"motor_states": False, "joint_command": False}
        
    def start(self):
        """启动异步日志线程"""
        if not self.is_running:
            self.stop_event.clear()
            self.worker_thread = threading.Thread(target=self._log_worker, daemon=True)
            self.worker_thread.start()
            self.is_running = True
            files_info = []
            if self.json_file_path:
                files_info.append(f"JSON: {self.json_file_path}")
            if self.csv_file_path:
                files_info.append(f"CSV: {self.csv_file_path}")
            print(f"[AsyncLogManager] 日志系统已启动，{', '.join(files_info)}")
            
    def stop(self):
        """停止异步日志线程"""
        if self.is_running:
            self.stop_event.set()
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=2.0)
            self.is_running = False
            print(f"[AsyncLogManager] 日志系统已停止，总日志数: {self.total_logs}, 丢弃: {self.dropped_logs}")
            
    def log_motor_states(self, positions: np.ndarray, velocities: np.ndarray, torques: np.ndarray):
        """
        记录电机状态 - 非阻塞
        
        Args:
            positions: 电机位置数组
            velocities: 电机速度数组  
            torques: 电机力矩数组
        """
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "motor_states",
                "data": {
                    "positions": positions.tolist(),
                    "velocities": velocities.tolist(),
                    "torques": torques.tolist()
                }
            }
            self.log_queue.put_nowait(log_entry)
            self.total_logs += 1
        except queue.Full:
            # 队列满时静默丢弃，避免阻塞主线程
            self.dropped_logs += 1
            
    def log_joint_command(self, positions_rad: np.ndarray, velocities_rad_s: np.ndarray, torques_nm: np.ndarray):
        """
        记录关节命令 - 非阻塞
        
        Args:
            positions_rad: 目标位置（弧度）
            velocities_rad_s: 目标速度（弧度/秒）
            torques_nm: 目标力矩（牛米）
        """
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "joint_command",
                "data": {
                    "target_positions": positions_rad.tolist(),
                    "target_velocities": velocities_rad_s.tolist(),
                    "target_torques": torques_nm.tolist()
                }
            }
            self.log_queue.put_nowait(log_entry)
            self.total_logs += 1
        except queue.Full:
            # 队列满时静默丢弃，避免阻塞主线程
            self.dropped_logs += 1
            
    def log_custom_event(self, event_type: str, data: dict):
        """
        记录自定义事件 - 非阻塞
        
        Args:
            event_type: 事件类型
            data: 事件数据
        """
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": event_type,
                "data": data
            }
            self.log_queue.put_nowait(log_entry)
            self.total_logs += 1
        except queue.Full:
            self.dropped_logs += 1
            
    def get_stats(self) -> dict:
        """获取日志统计信息"""
        return {
            "is_running": self.is_running,
            "total_logs": self.total_logs,
            "dropped_logs": self.dropped_logs,
            "queue_size": self.log_queue.qsize(),
            "json_file": str(self.json_file_path) if self.json_file_path else None,
            "csv_file": str(self.csv_file_path) if self.csv_file_path else None
        }
            
    def _log_worker(self):
        """后台日志写入线程"""
        json_file = None
        csv_files = {}
        
        try:
            # 打开JSON文件
            if self.save_json and self.json_file_path:
                json_file = open(self.json_file_path, 'a', encoding='utf-8')
                
            while not self.stop_event.is_set():
                try:
                    # 等待日志条目，超时检查停止信号
                    log_entry = self.log_queue.get(timeout=0.1)
                    
                    # 写入JSON文件
                    if json_file:
                        json_line = json.dumps(log_entry, ensure_ascii=False)
                        json_file.write(json_line + '\n')
                        json_file.flush()
                    
                    # 写入CSV文件
                    if self.save_csv:
                        self._write_csv_entry(log_entry, csv_files)
                    
                    self.log_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    # 记录日志系统本身的错误，但不影响主程序
                    print(f"[AsyncLogManager] 写入日志时出错: {e}")
                    
            # 处理剩余的日志条目
            while not self.log_queue.empty():
                try:
                    log_entry = self.log_queue.get_nowait()
                    
                    if json_file:
                        json_line = json.dumps(log_entry, ensure_ascii=False)
                        json_file.write(json_line + '\n')
                    
                    if self.save_csv:
                        self._write_csv_entry(log_entry, csv_files)
                    
                    self.log_queue.task_done()
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"[AsyncLogManager] 清理日志时出错: {e}")
                    
        except Exception as e:
            print(f"[AsyncLogManager] 日志文件操作失败: {e}")
        finally:
            # 关闭所有文件
            if json_file:
                json_file.close()
            for csv_file in csv_files.values():
                if csv_file:
                    csv_file.close()
    
    def _write_csv_entry(self, log_entry: dict, csv_files: dict):
        """写入CSV条目"""
        log_type = log_entry.get('type')
        if not log_type or not self.csv_file_path:
            return
            
        try:
            # 为不同类型的日志创建不同的CSV文件
            csv_file_path = self.log_dir / f"{self.log_name}_{log_type}_{self.timestamp}.csv"
            
            # 如果该类型的CSV文件还没有打开，则打开它
            if log_type not in csv_files:
                csv_files[log_type] = open(csv_file_path, 'w', newline='', encoding='utf-8')
                
            csv_file = csv_files[log_type]
            
            if log_type == 'motor_states':
                self._write_motor_states_csv(log_entry, csv_file)
            elif log_type == 'joint_command':
                self._write_joint_command_csv(log_entry, csv_file)
            else:
                self._write_generic_csv(log_entry, csv_file)
                
        except Exception as e:
            print(f"[AsyncLogManager] CSV写入错误: {e}")
    
    def _write_motor_states_csv(self, log_entry: dict, csv_file):
        """写入电机状态CSV"""
        writer = csv.writer(csv_file)
        
        # 写入头部（仅第一次）
        if not self.csv_headers_written.get('motor_states', False):
            headers = ['timestamp']
            data = log_entry['data']
            motor_count = len(data['positions'])
            
            # 添加位置列
            for i in range(motor_count):
                headers.append(f'position_motor_{i+1}')
            # 添加速度列
            for i in range(motor_count):
                headers.append(f'velocity_motor_{i+1}')
            # 添加力矩列
            for i in range(motor_count):
                headers.append(f'torque_motor_{i+1}')
                
            writer.writerow(headers)
            self.csv_headers_written['motor_states'] = True
        
        # 写入数据行
        row = [log_entry['timestamp']]
        data = log_entry['data']
        row.extend(data['positions'])
        row.extend(data['velocities'])
        row.extend(data['torques'])
        
        writer.writerow(row)
        csv_file.flush()
    
    def _write_joint_command_csv(self, log_entry: dict, csv_file):
        """写入关节命令CSV"""
        writer = csv.writer(csv_file)
        
        # 写入头部（仅第一次）
        if not self.csv_headers_written.get('joint_command', False):
            headers = ['timestamp']
            data = log_entry['data']
            motor_count = len(data['target_positions'])
            
            # 添加目标位置列
            for i in range(motor_count):
                headers.append(f'target_position_motor_{i+1}')
            # 添加目标速度列
            for i in range(motor_count):
                headers.append(f'target_velocity_motor_{i+1}')
            # 添加目标力矩列
            for i in range(motor_count):
                headers.append(f'target_torque_motor_{i+1}')
                
            writer.writerow(headers)
            self.csv_headers_written['joint_command'] = True
        
        # 写入数据行
        row = [log_entry['timestamp']]
        data = log_entry['data']
        row.extend(data['target_positions'])
        row.extend(data['target_velocities'])
        row.extend(data['target_torques'])
        
        writer.writerow(row)
        csv_file.flush()
    
    def _write_generic_csv(self, log_entry: dict, csv_file):
        """写入通用CSV格式"""
        writer = csv.writer(csv_file)
        
        # 简单的键值对格式
        if not hasattr(self, f'_generic_header_written_{log_entry["type"]}'):
            writer.writerow(['timestamp', 'type', 'data'])
            setattr(self, f'_generic_header_written_{log_entry["type"]}', True)
        
        writer.writerow([
            log_entry['timestamp'],
            log_entry['type'],
            json.dumps(log_entry['data'], ensure_ascii=False)
        ])
        csv_file.flush()
                    
    def __del__(self):
        """析构函数，确保线程正确关闭"""
        self.stop()


class LogAnalyzer:
    """日志分析器 - 用于分析生成的日志文件"""
    
    def __init__(self, log_file_path: str):
        self.log_file_path = Path(log_file_path)
        
    def load_logs(self, log_type: Optional[str] = None) -> list:
        """
        加载日志数据
        
        Args:
            log_type: 过滤特定类型的日志，None表示加载所有
            
        Returns:
            日志条目列表
        """
        logs = []
        if not self.log_file_path.exists():
            return logs
            
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        if log_type is None or log_entry.get('type') == log_type:
                            logs.append(log_entry)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[LogAnalyzer] 读取日志文件失败: {e}")
            
        return logs
        
    def get_motor_trajectory(self) -> dict:
        """获取电机轨迹数据"""
        motor_logs = self.load_logs('motor_states')
        
        if not motor_logs:
            return {}
            
        timestamps = []
        positions = []
        velocities = []
        torques = []
        
        for log in motor_logs:
            timestamps.append(log['timestamp'])
            positions.append(log['data']['positions'])
            velocities.append(log['data']['velocities'])
            torques.append(log['data']['torques'])
            
        return {
            'timestamps': timestamps,
            'positions': np.array(positions),
            'velocities': np.array(velocities),
            'torques': np.array(torques)
        }
        
    def get_command_trajectory(self) -> dict:
        """获取命令轨迹数据"""
        command_logs = self.load_logs('joint_command')
        
        if not command_logs:
            return {}
            
        timestamps = []
        target_positions = []
        target_velocities = []
        target_torques = []
        
        for log in command_logs:
            timestamps.append(log['timestamp'])
            target_positions.append(log['data']['target_positions'])
            target_velocities.append(log['data']['target_velocities'])
            target_torques.append(log['data']['target_torques'])
            
        return {
            'timestamps': timestamps,
            'target_positions': np.array(target_positions),
            'target_velocities': np.array(target_velocities),
            'target_torques': np.array(target_torques)
        }


# 全局日志管理器实例（可选）
_global_logger: Optional[AsyncLogManager] = None


def get_global_logger(log_file_path: str = "ic_arm_log.jsonl") -> AsyncLogManager:
    """获取全局日志管理器实例"""
    global _global_logger
    if _global_logger is None:
        _global_logger = AsyncLogManager(log_file_path)
        _global_logger.start()
    return _global_logger


def cleanup_global_logger():
    """清理全局日志管理器"""
    global _global_logger
    if _global_logger is not None:
        _global_logger.stop()
        _global_logger = None
