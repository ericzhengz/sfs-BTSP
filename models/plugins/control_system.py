"""
任务级 SLA 控制系统（方案 v3.1 核心）

职责：
- 任务选择器：选择最需要修复的旧任务
- 反演控制：根据目标错误率计算门控率
- 窗口判据：检查 p_gate 是否达到窗口上限
- 资源护栏：检查占用率、翻转次数等约束
- 计划持久化：任务间传递控制计划
"""

from __future__ import annotations
import math
import logging
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from .gate_policy import p_gate_from_epsilon, epsilon_from_p_gate, p_gate_window_cap


@dataclass 
class ControlPlan:
    """控制计划：跨任务传递的门控参数"""
    task: int
    mode: str  # 'task_sla'
    selected_task: Optional[int]
    selected_ranges: List[Tuple[int, int]]
    p_gate: float
    target_eps: float
    achievable: bool
    T_eff: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task': self.task,
            'mode': self.mode,
            'selected_task': self.selected_task,
            'selected_ranges': self.selected_ranges,
            'p_gate': self.p_gate,
            'target_eps': self.target_eps,
            'achievable': self.achievable,
            'T_eff': self.T_eff,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ControlPlan':
        return cls(
            task=data['task'],
            mode=data.get('mode', 'task_sla'),
            selected_task=data.get('selected_task'),
            selected_ranges=data.get('selected_ranges', []),
            p_gate=data['p_gate'],
            target_eps=data.get('target_eps', 0.05),
            achievable=data.get('achievable', True),
            T_eff=data.get('T_eff', 5.0),
        )


class TaskSLAController:
    """
    任务级 SLA 控制器（方案 v3.1 §5）
    
    核心功能：
    1. 选择器：选择最需修复的任务（worst_violation）
    2. 反演控制：计算 p_gate 以满足 SLA 目标
    3. 窗口判据：检查是否需要增加 T_eff
    4. 资源护栏：防止占用率过高
    5. 计划持久化：跨任务传递控制参数
    """
    
    def __init__(self, 
                 delta_pp: float = 5.0,
                 selector_mode: str = "worst_violation",
                 calibration_ema: float = 0.7):
        """
        Args:
            delta_pp: SLA 允许的准确率下降（百分点）
            selector_mode: 选择器模式（'worst_violation' | 'round_robin'）
            calibration_ema: 预测器校准的 EMA 动量
        """
        self.delta_pp = delta_pp
        self.selector_mode = selector_mode
        self.calibration_ema = calibration_ema
        
        # 状态：任务基准和SLA目标
        self._task_baselines: Dict[int, float] = {}  # {task_id: A_base}
        self._task_eps_targets: Dict[int, float] = {}  # {task_id: eps_target}
        self._task_ranges: Dict[int, Tuple[int, int]] = {}  # {task_id: (start, end)}
        
        # 状态：校准因子
        self._calibration_c: float = 1.0
        
        # 状态：round-robin 计数器
        self._round_robin_idx: int = 0
    
    def record_task_baseline(self, task_id: int, acc_fused: float, class_range: Tuple[int, int]):
        """
        记录任务基准（首次转旧时调用）
        
        Args:
            task_id: 任务编号
            acc_fused: 融合输出的准确率（%）
            class_range: 类别范围 (start, end)
        """
        self._task_baselines[task_id] = acc_fused
        self._task_ranges[task_id] = class_range
        logging.info(f"[TaskSLA] Recorded baseline for task {task_id}: "
                    f"acc={acc_fused:.2f}%, classes={class_range}")
    
    def compute_sla_targets(self):
        """
        计算所有旧任务的 SLA 目标误差率
        
        公式：eps_target = 1 - (A_base - Δpp) / 100
        """
        for task_id, acc_base in self._task_baselines.items():
            acc_sla = acc_base - self.delta_pp
            eps_target = 1.0 - acc_sla / 100.0
            self._task_eps_targets[task_id] = eps_target
    
    def select_task_worst_violation(self, task_eps_actual: Dict[int, float]
                                   ) -> Tuple[Optional[int], float, float]:
        """
        选择器：worst_violation（推荐）
        
        选择违约最严重的任务：
        k* = argmax_k (eps_actual[k] - eps_target[k])_+
        
        Args:
            task_eps_actual: {task_id: eps_actual}
            
        Returns:
            (selected_task, violation, eps_target)
            若所有任务都满足 SLA，返回 (None, 0.0, 0.0)
        """
        worst_task = None
        worst_violation = 0.0
        worst_eps_target = 0.0
        
        for task_id in self._task_eps_targets:
            if task_id not in task_eps_actual:
                continue
            
            eps_target = self._task_eps_targets[task_id]
            eps_actual = task_eps_actual[task_id]
            violation = max(0.0, eps_actual - eps_target)
            
            if violation > worst_violation:
                worst_violation = violation
                worst_task = task_id
                worst_eps_target = eps_target
        
        return worst_task, worst_violation, worst_eps_target
    
    def select_task_round_robin(self, task_eps_actual: Dict[int, float]
                               ) -> Tuple[Optional[int], float, float]:
        """
        选择器：round_robin
        
        轮询选择旧任务，适合方差较小场景
        
        Args:
            task_eps_actual: {task_id: eps_actual}
            
        Returns:
            (selected_task, violation, eps_target)
        """
        old_tasks = sorted(self._task_eps_targets.keys())
        if not old_tasks:
            return None, 0.0, 0.0
        
        # 循环选择
        selected_task = old_tasks[self._round_robin_idx % len(old_tasks)]
        self._round_robin_idx += 1
        
        eps_target = self._task_eps_targets[selected_task]
        eps_actual = task_eps_actual.get(selected_task, eps_target)
        violation = max(0.0, eps_actual - eps_target)
        
        return selected_task, violation, eps_target
    
    def select_task(self, task_eps_actual: Dict[int, float]
                   ) -> Tuple[Optional[int], float, float]:
        """
        统一选择器接口
        
        Args:
            task_eps_actual: {task_id: eps_actual}
            
        Returns:
            (selected_task, violation, eps_target)
        """
        if self.selector_mode == "round_robin":
            return self.select_task_round_robin(task_eps_actual)
        else:
            return self.select_task_worst_violation(task_eps_actual)
    
    def inverse_control(self, eps_target: float, M: int, p_pre: float, T_eff: float,
                       calibration_c: Optional[float] = None
                       ) -> Tuple[float, bool]:
        """
        反演控制：目标误差率 → 门控率
        
        应用校准因子：eps_target_calibrated = eps_target / c
        
        Args:
            eps_target: 目标错误率
            M: 任务数量
            p_pre: 预激活概率
            T_eff: 有效时间窗口
            calibration_c: 校准因子（None 则使用内部状态）
            
        Returns:
            (p_gate, achievable)
        """
        c = calibration_c if calibration_c is not None else self._calibration_c
        eps_target_calibrated = eps_target / max(c, 0.1)
        
        p_gate, achievable = p_gate_from_epsilon(eps_target_calibrated, M, p_pre, T_eff)
        return p_gate, achievable
    
    def check_window_cap(self, p_gate_target: float, T_eff: float, tau: float = 0.99
                        ) -> Tuple[bool, float]:
        """
        窗口饱和上限检查
        
        判断 p_gate_target 是否超过窗口饱和上限：
        p_gate_max = -ln(1 - tau) / T_eff
        
        Args:
            p_gate_target: 目标门控率
            T_eff: 当前有效时间窗口
            tau: 窗口饱和阈值（默认 0.99）
            
        Returns:
            (exceeded, cap_value)
        """
        cap = p_gate_window_cap(T_eff, tau)
        exceeded = (p_gate_target > cap)
        return exceeded, cap
    
    def calibrate_predictor(self, eps_actual: float, eps_predicted: float):
        """
        预测器在线标定（方案 v3.1 §4.3）
        
        公式：c ← η·c + (1-η)·(eps_actual / eps_predicted)
        
        Args:
            eps_actual: 实际测得的错误率
            eps_predicted: 模型预测的错误率
        """
        if eps_predicted > 1e-8:
            ratio = eps_actual / eps_predicted
            self._calibration_c = (self.calibration_ema * self._calibration_c + 
                                  (1 - self.calibration_ema) * ratio)
            
            logging.debug(f"[TaskSLA] Calibration update: "
                         f"c={self._calibration_c:.4f}, "
                         f"eps_actual={eps_actual:.4f}, "
                         f"eps_pred={eps_predicted:.4f}")
    
    def get_task_ranges(self, task_id: int) -> List[Tuple[int, int]]:
        """
        获取任务对应的类别范围（用于门控作用域）
        
        Args:
            task_id: 任务编号
            
        Returns:
            [(start, end), ...] 类别范围列表
        """
        if task_id in self._task_ranges:
            return [self._task_ranges[task_id]]
        return []
    
    def get_calibration_c(self) -> float:
        """获取当前校准因子"""
        return self._calibration_c
    
    def get_num_old_tasks(self) -> int:
        """获取已记录的旧任务数量"""
        return len(self._task_baselines)


class ResourceBudget:
    """
    资源护栏：统一口径约束（方案 v3.1 §3.3）
    
    检查项：
    - 占用率（全局 / 任务级）
    - 翻转次数
    - APB（Accuracy Per Byte）
    - 推理延迟
    """
    
    def __init__(self,
                 occ_global_high: float = 0.13,
                 occ_task_high: float = 0.18,
                 flips_max: Optional[int] = None,
                 apb_min: Optional[float] = None,
                 latency_max: Optional[float] = None):
        """
        Args:
            occ_global_high: 全局占用率上限
            occ_task_high: 任务级占用率上限
            flips_max: 最大翻转次数
            apb_min: 最小 APB（准确率 / 字节）
            latency_max: 最大推理延迟（ms）
        """
        self.occ_global_high = occ_global_high
        self.occ_task_high = occ_task_high
        self.flips_max = flips_max
        self.apb_min = apb_min
        self.latency_max = latency_max
        
        self._violation_flags: List[str] = []
    
    def check_occupancy(self, global_occ: float, task_occ: Optional[float] = None) -> bool:
        """
        检查占用率
        
        Args:
            global_occ: 全局占用率
            task_occ: 任务级占用率（可选）
            
        Returns:
            True 如果违反约束
        """
        violated = False
        
        if global_occ > self.occ_global_high:
            self._violation_flags.append(f"global_occ={global_occ:.4f}>{self.occ_global_high:.4f}")
            violated = True
        
        if task_occ is not None and task_occ > self.occ_task_high:
            self._violation_flags.append(f"task_occ={task_occ:.4f}>{self.occ_task_high:.4f}")
            violated = True
        
        return violated
    
    def check_flips(self, flips_total: int) -> bool:
        """
        检查翻转次数
        
        Args:
            flips_total: 累计翻转次数
            
        Returns:
            True 如果违反约束
        """
        if self.flips_max is not None and flips_total > self.flips_max:
            self._violation_flags.append(f"flips={flips_total}>{self.flips_max}")
            return True
        return False
    
    def check_apb(self, accuracy: float, bytes_used: int) -> bool:
        """
        检查 APB（Accuracy Per Byte）
        
        Args:
            accuracy: 准确率（%）
            bytes_used: 使用的字节数
            
        Returns:
            True 如果违反约束
        """
        if self.apb_min is not None and bytes_used > 0:
            apb = accuracy / bytes_used
            if apb < self.apb_min:
                self._violation_flags.append(f"apb={apb:.6f}<{self.apb_min:.6f}")
                return True
        return False
    
    def check_latency(self, latency_ms: float) -> bool:
        """
        检查推理延迟
        
        Args:
            latency_ms: 推理延迟（毫秒）
            
        Returns:
            True 如果违反约束
        """
        if self.latency_max is not None and latency_ms > self.latency_max:
            self._violation_flags.append(f"latency={latency_ms:.2f}>{self.latency_max:.2f}")
            return True
        return False
    
    def get_violation_flags(self) -> List[str]:
        """获取违反约束的标记列表"""
        return self._violation_flags.copy()
    
    def clear_flags(self):
        """清除违反约束标记"""
        self._violation_flags.clear()
    
    def any_violated(self) -> bool:
        """是否有任何约束被违反"""
        return len(self._violation_flags) > 0


def compute_coverage_and_rmse(task_eps_actual: Dict[int, float],
                              task_eps_targets: Dict[int, float]
                              ) -> Tuple[float, float]:
    """
    计算任务级覆盖率和违约 RMSE（方案 v3.1 §7.1）
    
    覆盖率：满足 SLA 的任务比例
    RMSE：违约量的均方根误差
    
    Args:
        task_eps_actual: {task_id: eps_actual}
        task_eps_targets: {task_id: eps_target}
        
    Returns:
        (coverage, rmse)
    """
    total_tasks = len(task_eps_targets)
    if total_tasks == 0:
        return 1.0, 0.0
    
    # 覆盖率
    coverage_count = sum(
        1 for task_id in task_eps_targets
        if task_id in task_eps_actual and task_eps_actual[task_id] <= task_eps_targets[task_id]
    )
    coverage = coverage_count / total_tasks
    
    # RMSE
    rmse_values = [
        max(0.0, task_eps_actual[tid] - task_eps_targets[tid])
        for tid in task_eps_targets if tid in task_eps_actual
    ]
    rmse = math.sqrt(sum(v**2 for v in rmse_values) / max(len(rmse_values), 1)) if rmse_values else 0.0
    
    return coverage, rmse 