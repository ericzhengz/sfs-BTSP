"""
Unified Evaluation Protocol & Metrics
统一评测协议：Coverage, RMSE, C*, APB, 等预算合同
"""

import torch
import numpy as np
import math
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
import csv
import json

@dataclass
class ResourceBudget:
    """资源预算合同"""
    max_flops: float = 1e12      # 最大FLOPs
    max_memory_mb: float = 8192   # 峰值显存(MB) 
    max_bytes: float = 1e9       # 外部存储字节
    max_train_time: float = 3600  # 训练时延(秒)
    max_infer_time: float = 100   # 推理时延(毫秒)

@dataclass 
class PerformanceSnapshot:
    """性能快照"""
    task_id: int
    step: int
    accuracy_total: float
    accuracy_old: float  
    accuracy_new: float
    bwt: float          # Backward Transfer
    lambda_load: float
    bytes_used: float
    flops_used: float
    memory_peak_mb: float
    latency_ms: float

class CoverageRMSECalculator:
    """Coverage和RMSE计算器"""
    
    def __init__(self, tolerance: float = 0.1):
        self.tolerance = tolerance  # Coverage的容差
        
    def compute_coverage_rmse(self, 
                            actual_trajectory: List[float],
                            target_trajectory: List[float]) -> Tuple[float, float]:
        """
        计算Coverage和RMSE
        
        Coverage: 实际轨迹被目标轨迹±容差覆盖的比例
        RMSE: 均方根误差
        """
        if len(actual_trajectory) == 0 or len(target_trajectory) == 0:
            return 0.0, float('inf')
            
        # 对齐长度
        min_len = min(len(actual_trajectory), len(target_trajectory))
        actual = actual_trajectory[-min_len:]
        target = target_trajectory[-min_len:]
        
        # Coverage计算
        covered_count = 0
        for a, t in zip(actual, target):
            if abs(a - t) <= self.tolerance * max(abs(t), 1e-6):
                covered_count += 1
        coverage = covered_count / len(actual) if len(actual) > 0 else 0.0
        
        # RMSE计算  
        mse = sum((a - t) ** 2 for a, t in zip(actual, target)) / len(actual)
        rmse = math.sqrt(mse)
        
        return coverage, rmse

class KneePointDetector:
    """拐点C*检测器 - 实现Kneedle算法的简化版本"""
    
    def __init__(self, sensitivity: float = 1.0):
        self.sensitivity = sensitivity
        
    def detect_knee(self, 
                   x_values: List[float], 
                   y_values: List[float],
                   n_bootstrap: int = 100) -> Tuple[float, float, float]:
        """
        检测膝点并给出置信区间
        
        返回: (knee_position, ci_low, ci_high)
        """
        if len(x_values) < 5:
            return 0.0, 0.0, 0.0
            
        x_arr = np.array(x_values)
        y_arr = np.array(y_values)
        
        # 数据预处理：归一化
        x_norm = (x_arr - x_arr.min()) / (x_arr.max() - x_arr.min() + 1e-8)
        y_norm = (y_arr - y_arr.min()) / (y_arr.max() - y_arr.min() + 1e-8)
        
        # 计算"差值曲线"
        diff_curve = y_norm - x_norm
        
        # 找到差值曲线的最大值点作为膝点
        knee_idx = np.argmax(diff_curve)
        knee_x = x_arr[knee_idx]
        
        # Bootstrap置信区间
        if n_bootstrap > 0:
            knee_positions = []
            for _ in range(n_bootstrap):
                # 带噪声重采样
                noise_scale = 0.05 * (y_arr.max() - y_arr.min())
                y_noisy = y_arr + np.random.normal(0, noise_scale, len(y_arr))
                y_noisy_norm = (y_noisy - y_noisy.min()) / (y_noisy.max() - y_noisy.min() + 1e-8)
                diff_noisy = y_noisy_norm - x_norm
                knee_idx_noisy = np.argmax(diff_noisy)
                knee_positions.append(x_arr[knee_idx_noisy])
            
            knee_positions = np.array(knee_positions)
            ci_low = np.percentile(knee_positions, 5)   # 90% CI
            ci_high = np.percentile(knee_positions, 95)
        else:
            # 简单估计
            ci_range = 0.1 * (x_arr.max() - x_arr.min())
            ci_low = knee_x - ci_range
            ci_high = knee_x + ci_range
            
        return float(knee_x), float(ci_low), float(ci_high)

class APBCalculator:
    """Accuracy-Per-Byte (APB) 计算器"""
    
    def compute_apb(self, accuracy: float, total_bytes: float) -> float:
        """计算APB = accuracy / (bytes / 1MB)"""
        mb_used = total_bytes / (1024 * 1024)  # 转换为MB
        return accuracy / max(mb_used, 1e-6)
    
    def compute_efficiency_score(self, 
                               accuracy: float,
                               bytes_used: float, 
                               flops_used: float,
                               latency_ms: float) -> Dict[str, float]:
        """计算多维度效率分数"""
        return {
            "apb": self.compute_apb(accuracy, bytes_used),
            "acc_per_gflop": accuracy / max(flops_used / 1e9, 1e-6),
            "acc_per_ms": accuracy / max(latency_ms, 1e-6),
            "combined_efficiency": accuracy / (
                max(bytes_used / 1e6, 1) * 
                max(flops_used / 1e9, 1) * 
                max(latency_ms / 100, 1)
            )
        }

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.start_time = None
        self.peak_memory = 0.0
        self.total_flops = 0.0
        self.total_bytes = 0.0
        
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
    def record_operation(self, flops: float, bytes_io: float = 0.0):
        """记录一次操作的资源消耗"""
        self.total_flops += flops
        self.total_bytes += bytes_io
        
    def get_current_stats(self) -> Dict[str, float]:
        """获取当前资源统计"""
        elapsed_time = time.time() - (self.start_time or time.time())
        
        if torch.cuda.is_available():
            peak_memory_bytes = torch.cuda.max_memory_allocated()
            peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        else:
            peak_memory_mb = 0.0
            
        return {
            "elapsed_time": elapsed_time,
            "peak_memory_mb": peak_memory_mb,
            "total_flops": self.total_flops,
            "total_bytes": self.total_bytes
        }

class UnifiedEvaluationProtocol:
    """统一评测协议主类"""
    
    def __init__(self, 
                 budget: ResourceBudget,
                 target_trajectories: Dict[str, List[float]] = None):
        self.budget = budget
        self.target_trajectories = target_trajectories or {
            "conservative": [0.05, 0.08, 0.12, 0.15, 0.18],
            "linear": [0.10, 0.15, 0.20, 0.25, 0.30],
            "aggressive": [0.15, 0.25, 0.35, 0.45, 0.55]
        }
        
        # 子模块
        self.coverage_calculator = CoverageRMSECalculator()
        self.knee_detector = KneePointDetector()
        self.apb_calculator = APBCalculator()
        self.resource_monitor = ResourceMonitor()
        
        # 历史记录
        self.performance_history: List[PerformanceSnapshot] = []
        self.lambda_values: List[float] = []
        self.accuracy_values: List[float] = []
        
        # 统一日志
        self.unified_logs: List[Dict] = []
        
    def start_evaluation(self):
        """开始评测"""
        self.resource_monitor.start_monitoring()
        
    def record_performance(self, snapshot: PerformanceSnapshot):
        """记录性能快照"""
        self.performance_history.append(snapshot)
        self.lambda_values.append(snapshot.lambda_load)
        self.accuracy_values.append(snapshot.accuracy_total)
        
    def record_unified_log(self, log_entry: Dict):
        """记录统一日志条目"""
        # 补充资源信息
        resource_stats = self.resource_monitor.get_current_stats()
        log_entry.update({
            "bytes_total": resource_stats["total_bytes"],
            "flops_total": resource_stats["total_flops"], 
            "peak_mem": resource_stats["peak_memory_mb"],
            "latency_ms": resource_stats["elapsed_time"] * 1000
        })
        
        self.unified_logs.append(log_entry)
        
    def compute_final_metrics(self) -> Dict[str, any]:
        """计算最终评估指标"""
        if len(self.performance_history) == 0:
            return {"error": "No performance data recorded"}
            
        metrics = {}
        
        # 1. Coverage & RMSE for each trajectory
        error_trajectory = [1.0 - p.accuracy_old for p in self.performance_history]
        
        for traj_name, target_traj in self.target_trajectories.items():
            coverage, rmse = self.coverage_calculator.compute_coverage_rmse(
                error_trajectory, target_traj
            )
            metrics[f"coverage_{traj_name}"] = coverage
            metrics[f"rmse_{traj_name}"] = rmse
            
        # 2. Knee point detection
        if len(self.lambda_values) >= 5:
            knee_pos, ci_low, ci_high = self.knee_detector.detect_knee(
                self.lambda_values, self.accuracy_values
            )
            metrics.update({
                "knee_pos": knee_pos,
                "knee_ci_low": ci_low,
                "knee_ci_high": ci_high,
                "knee_error_pct": 0.0  # 需要与理论值比较
            })
        
        # 3. APB and efficiency
        final_snapshot = self.performance_history[-1]
        efficiency = self.apb_calculator.compute_efficiency_score(
            final_snapshot.accuracy_total,
            final_snapshot.bytes_used,
            final_snapshot.flops_used,
            final_snapshot.latency_ms
        )
        metrics.update(efficiency)
        
        # 4. Budget compliance
        resource_stats = self.resource_monitor.get_current_stats()
        metrics.update({
            "budget_flops_ratio": resource_stats["total_flops"] / self.budget.max_flops,
            "budget_memory_ratio": resource_stats["peak_memory_mb"] / self.budget.max_memory_mb,
            "budget_bytes_ratio": resource_stats["total_bytes"] / self.budget.max_bytes,
            "budget_time_ratio": resource_stats["elapsed_time"] / self.budget.max_train_time
        })
        
        # 5. Robustness metrics
        accuracies = [p.accuracy_total for p in self.performance_history]
        if len(accuracies) > 1:
            metrics.update({
                "accuracy_mean": np.mean(accuracies),
                "accuracy_std": np.std(accuracies),
                "accuracy_min": np.min(accuracies),
                "final_bwt": final_snapshot.bwt
            })
            
        return metrics
    
    def export_logs(self, filepath: str):
        """导出统一日志为CSV"""
        if not self.unified_logs:
            return
            
        # 确保所有日志条目有相同的键
        all_keys = set()
        for log in self.unified_logs:
            all_keys.update(log.keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            
            for log in self.unified_logs:
                # 填补缺失键
                complete_log = {key: log.get(key, '') for key in all_keys}
                writer.writerow(complete_log)
                
    def generate_report(self) -> Dict[str, any]:
        """生成完整评估报告"""
        metrics = self.compute_final_metrics()
        
        report = {
            "protocol_version": "1.0",
            "budget": {
                "max_flops": self.budget.max_flops,
                "max_memory_mb": self.budget.max_memory_mb, 
                "max_bytes": self.budget.max_bytes,
                "max_train_time": self.budget.max_train_time
            },
            "metrics": metrics,
            "performance_trajectory": [
                {
                    "task": p.task_id,
                    "step": p.step,
                    "acc_total": p.accuracy_total,
                    "acc_old": p.accuracy_old,
                    "acc_new": p.accuracy_new,
                    "bwt": p.bwt,
                    "lambda_load": p.lambda_load
                } 
                for p in self.performance_history
            ],
            "resource_usage": self.resource_monitor.get_current_stats(),
            "target_trajectories": self.target_trajectories
        }
        
        return report
    
    def check_coverage_threshold(self, threshold: float = 0.9) -> bool:
        """检查Coverage是否达到阈值"""
        metrics = self.compute_final_metrics()
        for traj_name in self.target_trajectories:
            coverage_key = f"coverage_{traj_name}"
            if coverage_key in metrics and metrics[coverage_key] >= threshold:
                return True
        return False
    
    def check_budget_compliance(self) -> Dict[str, bool]:
        """检查预算合规性"""
        resource_stats = self.resource_monitor.get_current_stats()
        
        return {
            "flops_compliant": resource_stats["total_flops"] <= self.budget.max_flops,
            "memory_compliant": resource_stats["peak_memory_mb"] <= self.budget.max_memory_mb,
            "bytes_compliant": resource_stats["total_bytes"] <= self.budget.max_bytes,
            "time_compliant": resource_stats["elapsed_time"] <= self.budget.max_train_time
        } 