"""
Event-Driven Control System for Incremental Learning
基于 "资格迹 × 稀疏门控 × 有限状态转移" 的可计算控制律
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import math
from collections import deque
from .gate_policy import p_gate_from_epsilon, epsilon_from_p_gate, T_eff as T_eff_formula

class ControlMode(Enum):
    """控制模式：A=Analysis-only, B=Intervention"""
    SIDECAR_ANALYSIS = "A"  # 不改损失，仅控制/评估
    INTERVENTION = "B"      # 可插拔进入loss的轻量辅助

@dataclass 
class ControlState:
    """有限状态机的状态"""
    task_id: int
    step: int
    p_gate_current: float
    T_eff_current: float
    epsilon_target: float
    lambda_load: float
    knee_detected: bool = False
    coverage_ratio: float = 0.0
    
class EligibilityTrace:
    """资格迹机制：指数衰减 + 激活重置"""
    def __init__(self, size: int, beta: float = 0.95, theta: float = 0.01):
        self.beta = beta  # 衰减因子
        self.theta = theta  # 有效阈值
        self.traces = torch.zeros(size, dtype=torch.float32)
        self.T_eff = -math.log(theta) / math.log(beta)  # 有效时间窗
        
    def update(self, active_indices: torch.Tensor):
        """更新资格迹：衰减 + 激活重置"""
        # 指数衰减
        self.traces *= self.beta
        # 激活位置重置为1
        if len(active_indices) > 0:
            self.traces[active_indices] = 1.0
            
    def get_effective_window(self) -> float:
        """返回有效时间窗 T_eff"""
        return self.T_eff
        
    def get_active_mass(self) -> float:
        """返回当前活跃质量"""
        return float(self.traces.sum().item())

class BLineController:
    """B线：信息/可分性控制律 - 容量拐点检测与负载坐标"""
    
    def __init__(self, N_bits: int, num_classes: int):
        self.N_bits = N_bits
        self.num_classes = num_classes
        self.history = deque(maxlen=1000)  # 性能历史
        self.lambda_history = deque(maxlen=1000)  # 负载历史
        
    def compute_lambda_load(self, N_eff: float, p_pre: float, rho: float, T_eff: float) -> float:
        """
        计算归一化负载坐标 λ
        λ = (N_eff × p_pre × ρ) / (T_eff × log(num_classes))
        """
        denominator = T_eff * math.log(max(2, self.num_classes))
        lambda_load = (N_eff * p_pre * rho) / max(1e-8, denominator)
        return float(lambda_load)
        
    def detect_knee_point(self, lambda_values: List[float], acc_values: List[float]) -> Tuple[float, float, float]:
        """
        使用Kneedle算法检测拐点C*
        返回: (knee_pos, confidence_low, confidence_high)
        """
        if len(lambda_values) < 5:
            return 0.0, 0.0, 0.0
            
        # 简化的膝点检测：二阶导数最大值
        lambda_arr = np.array(lambda_values)
        acc_arr = np.array(acc_values)
        
        if len(lambda_arr) < 3:
            return 0.0, 0.0, 0.0
            
        # 计算二阶差分
        d2_acc = np.gradient(np.gradient(acc_arr))
        knee_idx = np.argmax(np.abs(d2_acc))
        
        knee_pos = lambda_arr[knee_idx]
        # 简单置信区间：±10% 范围
        ci_range = 0.1 * knee_pos
        
        return knee_pos, knee_pos - ci_range, knee_pos + ci_range
        
    def update_and_check(self, lambda_load: float, accuracy: float) -> Dict[str, float]:
        """更新历史并检查拐点"""
        self.lambda_history.append(lambda_load)
        self.history.append(accuracy)
        
        if len(self.history) >= 5:
            knee_pos, ci_low, ci_high = self.detect_knee_point(
                list(self.lambda_history), list(self.history)
            )
            return {
                "lambda_load": lambda_load,
                "knee_pos": knee_pos,
                "knee_ci_low": ci_low, 
                "knee_ci_high": ci_high,
                "knee_detected": knee_pos > 0
            }
        return {"lambda_load": lambda_load, "knee_detected": False}

class DLineController:
    """D线：动态/控制律 - ε-control反演律与时间常数"""
    
    def __init__(self, epsilon_0: float = 0.1, tau_e: float = 4.0):
        self.epsilon_0 = epsilon_0  # 目标遗忘率
        self.tau_e = tau_e  # 时间常数
        self.error_history = deque(maxlen=100)
        self.p_gate_history = deque(maxlen=100)
        
    def epsilon_control_inversion(self, current_error: float, M: int, p_pre: float, T_eff: float) -> float:
        """
        ε-control 反演律（统一到 gate_policy 的精确链路）
        使用 p_gate_from_epsilon(eps0, M, p_pre, T_eff)
        """
        try:
            eps0 = float(self.epsilon_0)
            T_val = max(float(T_eff), 1e-8)
            p_val, _achievable = p_gate_from_epsilon(eps0, max(int(M), 1), max(float(p_pre), 1e-8), T_val)
            # 与 gate_policy 一致的上限裁剪由其内部处理
            return float(p_val)
        except Exception:
            return 0.5
        
    def update_trajectory(self, current_error: float, p_gate: float):
        """更新误差轨迹"""
        self.error_history.append(current_error)
        self.p_gate_history.append(p_gate)
        
    def compute_coverage_rmse(self, target_trajectory: List[float]) -> Tuple[float, float]:
        """
        计算Coverage和RMSE指标
        Coverage: 实际轨迹被目标轨迹±容差覆盖的比例
        RMSE: 均方根误差
        """
        if len(self.error_history) == 0 or len(target_trajectory) == 0:
            return 0.0, float('inf')
            
        actual = list(self.error_history)
        min_len = min(len(actual), len(target_trajectory))
        
        if min_len == 0:
            return 0.0, float('inf')
            
        actual = actual[-min_len:]
        target = target_trajectory[-min_len:]
        
        # Coverage: ±10%容差
        tolerance = 0.1
        covered = 0
        for a, t in zip(actual, target):
            if abs(a - t) <= tolerance * abs(t):
                covered += 1
        coverage = covered / len(actual)
        
        # RMSE
        mse = sum((a - t) ** 2 for a, t in zip(actual, target)) / len(actual)
        rmse = math.sqrt(mse)
        
        return coverage, rmse

class EventDrivenControlSystem:
    """事件驱动控制系统主类"""
    
    def __init__(self, 
                 N_bits: int,
                 num_classes: int,
                 feat_dim: int,
                 mode: ControlMode = ControlMode.SIDECAR_ANALYSIS,
                 epsilon_0: float = 0.1,
                 tau_e: float = 4.0,
                 alpha: float = 0.8):
        
        self.mode = mode
        self.N_bits = N_bits
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        
        # 控制状态
        self.state = ControlState(
            task_id=0, step=0, p_gate_current=0.5, 
            T_eff_current=tau_e, epsilon_target=epsilon_0, lambda_load=0.0
        )
        
        # 资格迹
        self.eligibility_trace = EligibilityTrace(N_bits, beta=0.95)
        
        # 双线控制器
        self.b_line = BLineController(N_bits, num_classes)
        self.d_line = DLineController(epsilon_0, tau_e)
        
        # 融合参数
        self.alpha = alpha
        self.alpha_trainable = (mode == ControlMode.INTERVENTION)
        
        # 日志字段
        self.unified_log = {}
        
    def event_step(self, 
                   active_bits: torch.Tensor,
                   targets: torch.Tensor,
                   base_logits: torch.Tensor,
                   btsp_logits: torch.Tensor,
                   N_eff: float,
                   p_pre: float,
                   rho: float) -> Dict[str, any]:
        """
        事件驱动步骤：处理一个batch的控制更新
        
        返回: {
            "fused_logits": 融合后的logits,
            "control_signals": 控制信号字典,
            "metrics": 评估指标,
            "should_update_memory": 是否更新记忆
        }
        """
        
        # 1. 更新资格迹
        if len(active_bits) > 0:
            active_indices = torch.nonzero(active_bits.any(dim=1)).squeeze(-1)
            self.eligibility_trace.update(active_indices)
        
        # 2. 计算当前T_eff
        T_eff_current = self.eligibility_trace.get_effective_window()
        self.state.T_eff_current = T_eff_current
        
        # 3. B线：负载坐标与拐点检测
        lambda_load = self.b_line.compute_lambda_load(N_eff, p_pre, rho, T_eff_current)
        self.state.lambda_load = lambda_load
        
        current_acc = self._compute_accuracy(base_logits, targets)
        b_metrics = self.b_line.update_and_check(lambda_load, current_acc)
        
        # 4. D线：ε-control反演
        current_error = 1.0 - current_acc  # 简化的误差定义
        p_gate_target = self.d_line.epsilon_control_inversion(
            current_error, self.state.task_id + 1, p_pre, T_eff_current
        )
        
        self.state.p_gate_current = p_gate_target
        self.d_line.update_trajectory(current_error, p_gate_target)
        
        # 5. 融合决策
        if self.mode == ControlMode.SIDECAR_ANALYSIS:
            # A模式：训练时不融合，仅记录
            fused_logits = base_logits.detach()  # 训练时使用base
            fusion_for_metrics = (1.0 - self.alpha) * base_logits + self.alpha * btsp_logits
        else:
            # B模式：可融合
            fused_logits = (1.0 - self.alpha) * base_logits + self.alpha * btsp_logits
            fusion_for_metrics = fused_logits
            
        # 6. 控制信号生成
        control_signals = {
            "p_gate_target": p_gate_target,
            "T_eff": T_eff_current,
            "lambda_load": lambda_load,
            "should_write": self._should_write_memory(p_gate_target),
            "homeostasis_trigger": self._check_homeostasis_trigger()
        }
        
        # 7. 更新统一日志
        self._update_unified_log(b_metrics, control_signals, current_acc)
        
        # 8. 状态转移
        self.state.step += 1
        
        return {
            "fused_logits": fused_logits,
            "fusion_for_metrics": fusion_for_metrics,
            "control_signals": control_signals,
            "metrics": {**b_metrics, "current_accuracy": current_acc},
            "should_update_memory": control_signals["should_write"]
        }
    
    def _compute_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """计算当前准确率"""
        with torch.no_grad():
            if logits.size(0) == 0 or targets.size(0) == 0:
                return 0.0
            preds = logits.argmax(dim=1)
            correct = (preds == targets)
            if correct.numel() == 0:
                return 0.0
            acc = correct.float().mean().item()
        return acc
    
    def _should_write_memory(self, p_gate: float) -> bool:
        """基于门控概率决定是否写入记忆"""
        return torch.rand(1).item() < p_gate
    
    def _check_homeostasis_trigger(self) -> bool:
        """检查是否触发稳态调节"""
        return self.state.step % 50 == 0
    
    def _update_unified_log(self, b_metrics: Dict, control_signals: Dict, accuracy: float):
        """更新统一日志字段"""
        self.unified_log.update({
            "task": self.state.task_id,
            "step": self.state.step,
            "mode": self.mode.value,
            "target_eps0": self.d_line.epsilon_0,
            "p_gate_plan": control_signals["p_gate_target"],
            "p_gate_current": self.state.p_gate_current,  # 修正字段名
            "p_gate_mean": self.state.p_gate_current,
            "T_eff_current": control_signals["T_eff"],  # 添加缺失字段
            "T_eff": control_signals["T_eff"],
            "acc_total": accuracy,
            "lambda_load": control_signals["lambda_load"],
            "knee_detected": b_metrics.get("knee_detected", False),
            "knee_pos": b_metrics.get("knee_pos", 0.0),
            "alpha_value": self.alpha,
            "use_fusion_train": (self.mode == ControlMode.INTERVENTION),
            "use_fusion_eval": True,
            "target_trajectory": list(self.d_line.error_history) if self.d_line.error_history else []  # 添加轨迹字段
        })
    
    def get_unified_log(self) -> Dict[str, any]:
        """获取统一格式的日志"""
        return self.unified_log.copy()
    
    def set_task(self, task_id: int):
        """切换任务"""
        self.state.task_id = task_id
        logging.info(f"[Control System] 切换到任务 {task_id}")
    
    def get_control_state(self) -> ControlState:
        """获取当前控制状态"""
        return self.state
    
    def compute_coverage_rmse(self, target_trajectory: List[float]) -> Tuple[float, float]:
        """计算覆盖率和RMSE"""
        return self.d_line.compute_coverage_rmse(target_trajectory) 