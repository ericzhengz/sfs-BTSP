# models/plugins/btsp_plugin.py
from __future__ import annotations
import torch, torch.nn as nn
from .projection import OrthProjection
from .memory import BTSPMemory
from .fuse import fuse_logits
from .gate_policy import T_eff_from_params, apply_gate_schedule, epsilon_from_p_gate, schedule_p_gate, T_eff, p_gate_from_epsilon, p_gate_window_cap
import math, logging, numpy as np, traceback

# --- Experiment Logging Imports ---
from .experiment_logger import ExperimentLogger
from .metrics import estimate_cov, n_eff_from_rho, n_eff_robust_from_cov
from collections import deque
from typing import Any

# --- 新增：事件驱动控制系统 ---
try:
    from .control_system import EventDrivenControlSystem, ControlMode
    # PerformanceSnapshot
except ImportError:
    EventDrivenControlSystem = None
    ControlMode = None


def _arg_get(args: Any, key: str, default: Any = None):
    """Safe accessor supporting dict-like and namespace arguments."""
    if args is None:
        return default
    if isinstance(args, dict):
        return args.get(key, default)
    return getattr(args, key, default)

class BTSPPlugin(nn.Module):
    """
    BTSP Plugin with proper epsilon control implementation
    """
    def __init__(self, inner: nn.Module, 
                 # === 🔥 智能推断参数 ===
                 num_classes: int = None, feat_dim: int = None, num_tasks: int = None,
                 data_manager=None, args=None,
                 # === 核心BTSP参数 (从args中读取，提供合理默认值) ===
                 N_bits=None, topk=None, theta=None, tau_e_steps=None,
                 branches=None, alpha=None, homeo_interval=None, zstats_interval=None,
                 log_interval: int = None,
                 # --- Experiment Logging Params ---
                 experiment_logging: bool = None,
                 log_file: str | None = None,
                 stc_probe_deltas: list[int] | None = None,
                 capacity_probe: bool = None,
                 epsilon_control_probe: bool = None,
                 stc_probe: bool = None,
                 x_bits_buffer_size: int = None,
                 # --- 新增：事件驱动控制参数 ---
                 control_mode: str = None,  # "A"=Analysis-only, "B"=Intervention
                 target_epsilon: float = None,  # 目标遗忘率
                 enable_unified_protocol: bool = None,  # 启用统一评测协议
                 resource_budget=None,  # ResourceBudget = None
                 # --- 新增：epsilon控制参数 ---
                 epsilon_hysteresis: float = None,  # 滞回带，防止震荡
                 epsilon_tolerance: float = None,   # 容差范围
                 ):
        
        super().__init__()
        
        # === 🔥 参数智能推断：优先使用传入参数，其次从args读取，最后使用默认值 ===
        if args is None:
            args = {}
        
        # === SSOT config resolution ===
        cfg_norm = None
        decision_rows = []
        try:
            if isinstance(args, dict):
                from btsp.config_resolver import resolve_btsp_config
                cfg_norm, decision_rows = resolve_btsp_config(args, strict=False)
                try:
                    ok_cnt = sum(1 for _r in decision_rows if str(_r.get('status')) == 'OK')
                    warn_cnt = sum(1 for _r in decision_rows if 'WARNING' in str(_r.get('status')))
                    err_cnt = sum(1 for _r in decision_rows if 'ERROR' in str(_r.get('status')))
                    logging.info(f"[BTSP Config] resolved: ok={ok_cnt}, warn={warn_cnt}, err={err_cnt}")
                except Exception:
                    pass
                # Extract config_hash if present for a stable fingerprint
                try:
                    for row in decision_rows:
                        if row.get('key') == 'config_hash':
                            logging.info(f"[BTSP Config] config_hash={row.get('value')}")
                            break
                except Exception:
                    pass
        except Exception as e:
            logging.error(f"[BTSP Config] resolution failed: {e}")
            raise
            
        # 核心BTSP参数统一从规范键读取（SSOT）
        N_bits = N_bits if N_bits is not None else _arg_get(args, "btsp_N_bits", 8192)
        topk = topk if topk is not None else _arg_get(args, "btsp_topk", 256)
        theta = float(cfg_norm["btsp.memory.init.theta"])  # 仅使用规范键
        tau_e_steps = float(cfg_norm["btsp.memory.init.tau_e_steps"])  # 仅使用规范键
        branches = int(cfg_norm.get("btsp.memory.init.branches", _arg_get(args, "btsp_branches", 8))) if branches is None else branches
        alpha = float(cfg_norm["btsp.alpha.train_value"]) if alpha is None else float(alpha)
        homeo_interval = homeo_interval if homeo_interval is not None else _arg_get(args, "btsp_homeo_interval", 100)
        zstats_interval = zstats_interval if zstats_interval is not None else _arg_get(args, "btsp_zstats_interval", 30)
        log_interval = int(cfg_norm.get("btsp.logging.interval", _arg_get(args, "btsp_log_interval", 50))) if log_interval is None else int(log_interval)
        
        # 控制系统参数（规范键）
        control_mode = control_mode if control_mode is not None else cfg_norm.get("btsp.plugin.mode", "A")
        if target_epsilon is None:
            _te = cfg_norm.get("btsp.target_epsilon", _arg_get(args, "btsp.target_epsilon", _arg_get(args, "btsp_target_epsilon", _arg_get(args, "target_epsilon", 0.05))))
            target_epsilon = float(_te)
        else:
            target_epsilon = float(target_epsilon)
        enable_unified_protocol = bool(cfg_norm.get("btsp.enabled", True)) if enable_unified_protocol is None else bool(enable_unified_protocol)
        gate_mode_default = str(_arg_get(args, "btsp_gate_mode", "rate")).lower()
        delta_q_default = float(_arg_get(args, "btsp_delta_q_target", 5e-4))
        
        # 实验参数（保留非规范键，对外部非BTSP字段宽容）
        experiment_logging = experiment_logging if experiment_logging is not None else _arg_get(args, "btsp_experiment_logging", _arg_get(args, "btsp_enable_unified_logging", False))
        log_file = log_file if log_file is not None else _arg_get(args, "btsp_log_file", None)
        capacity_probe = capacity_probe if capacity_probe is not None else _arg_get(args, "btsp_capacity_probe", False)
        epsilon_control_probe = epsilon_control_probe if epsilon_control_probe is not None else _arg_get(args, "btsp_epsilon_control_probe", False)
        stc_probe = stc_probe if stc_probe is not None else _arg_get(args, "btsp_stc_probe", False)
        x_bits_buffer_size = x_bits_buffer_size if x_bits_buffer_size is not None else _arg_get(args, "btsp_x_bits_buffer_size", 256)
        
        # epsilon控制参数（规范键）
        epsilon_hysteresis = float(cfg_norm.get("btsp.p_gate.tolerance", 0.01)) if epsilon_hysteresis is None else float(epsilon_hysteresis)
        epsilon_tolerance = float(cfg_norm.get("btsp.p_gate.tolerance", 0.005)) if epsilon_tolerance is None else float(epsilon_tolerance)
        
        # 资源与护栏（规范键）
        self._p_gate_hard_cap = float(cfg_norm.get("btsp.p_gate.max", 0.05))
        self._p_gate_min = float(cfg_norm.get("btsp.p_gate.min", 0.006))
        self._p_gate_delta_abs = float(cfg_norm.get("btsp.p_gate.delta_abs", 0.002))
        self._p_gate_delta_rel = float(cfg_norm.get("btsp.p_gate.delta_rel", 0.5))
        self._occ_upper_band = float(cfg_norm.get("btsp.occupancy.global_high", 0.13))
        self._occ_lower_band = float(cfg_norm.get("btsp.occupancy.task_high", 0.11))
        # 记录 task/全局阈值以便其他分支使用
        self._occ_global_high = self._occ_upper_band
        self._occ_task_high = self._occ_lower_band
        
        # teff 相关（规范键）
        self._teff_step_on_cap = float(cfg_norm.get("btsp.teff.step_on_cap", 0.5))
        self._teff_min = float(cfg_norm.get("btsp.teff.min", 1.0))
        self._teff_max = float(cfg_norm.get("btsp.teff.max", 8.0))
        
        # control protocol（规范键）
        self._control_mode = str(cfg_norm.get("btsp.control.mode", "task_sla")).lower()
        self._control_selector = str(cfg_norm.get("btsp.control.selector", "worst_violation"))
        self._write_mask_enabled = bool(cfg_norm.get("btsp.control.write_mask_enabled", True))
        
        # 资源预算处理
        if resource_budget is None and enable_unified_protocol:
            resource_budget_config = _arg_get(args, "btsp_resource_budget", {})
            if resource_budget_config:
                from models.plugins.unified_metrics import ResourceBudget
                resource_budget = ResourceBudget(
                    max_flops=resource_budget_config.get("max_flops", 1e12),
                    max_memory_mb=resource_budget_config.get("max_memory_mb", 8192),
                    max_bytes=resource_budget_config.get("max_bytes", resource_budget_config.get("max_storage_bytes", 1e9)),
                    max_train_time=resource_budget_config.get("max_time_seconds", 3600),
                    max_infer_time=resource_budget_config.get("max_infer_time", 100)
                )
        
        # === 🔥 智能推断：从data_manager和args中获取关键参数 ===
        num_classes, feat_dim, num_tasks = self._infer_parameters(
            num_classes, feat_dim, num_tasks, data_manager, args, inner
        )
        
        # 存储内层learner
        self._modules['inner'] = inner
        
        # 核心组件初始化
        self.alpha = alpha
        # Fusion toggles (A-shape defaults: train=False, eval=True)
        if cfg_norm:
            self.use_fusion_train = bool(cfg_norm.get("btsp.eval.use_fusion_train", False))
            self.use_fusion_eval = bool(cfg_norm.get("btsp.eval.use_fusion_eval", True))
        else:
            self.use_fusion_train = bool(_arg_get(args, "btsp_use_fusion_train", False))
            self.use_fusion_eval = bool(_arg_get(args, "btsp_use_fusion_eval", True))
        self.homeo_interval = homeo_interval
        self.zstats_interval = zstats_interval
        self.topk = topk
        self.p_pre = max(1e-8, min(1.0, float(topk) / float(N_bits)))
        self._gstep = 0
        self.num_tasks = num_tasks
        
        # BTSP核心组件
        self._modules['proj'] = OrthProjection(feat_dim, N_bits)
        self._modules['btsp'] = BTSPMemory(num_classes, N_bits, branches, theta, tau_e_steps)
        
        # --- 新增：epsilon控制状态 ---
        self.target_epsilon = target_epsilon
        self.epsilon_hysteresis = epsilon_hysteresis
        self.epsilon_tolerance = epsilon_tolerance
        self.current_task = -1
        self._btsp_task_range = (0, 0)  # (known_classes, total_classes)
        gate_mode_default = gate_mode_default if gate_mode_default in {"achievable", "rate"} else "rate"
        self._default_gate_mode = gate_mode_default
        self._delta_q_default = delta_q_default
        # 🔥 修复：achievable默认启用，让控制分支能生效
        self._achievable_enabled = bool(_arg_get(args, "btsp_enable_achievable", True))  # 默认True
        # p_gate limits
        # 护栏参数由规范键提供
        # 兼容旧配置仅保留safe_rate（若外部逻辑仍使用）
        self._p_gate_safe_rate = float(_arg_get(args, "btsp_safe_rate", 0.005))
        # occupancy bands (canonical override)
        self._occ_upper_band = float(cfg_norm.get("btsp.occupancy.global_high", _arg_get(args, "btsp_occ_upper", 0.13)) if cfg_norm else _arg_get(args, "btsp_occ_upper", 0.13))
        self._occ_lower_band = float(cfg_norm.get("btsp.occupancy.task_high", _arg_get(args, "btsp_occ_lower", 0.11)) if cfg_norm else _arg_get(args, "btsp_occ_lower", 0.11))
        self._last_p_gate_value = float(self._modules['btsp'].p_gate.mean().item())
        self._last_mode = self._default_gate_mode
        self._last_delta_q_target = self._delta_q_default
        self._last_achievable = False
        self._last_pred_eps = None
        self._next_gate_plan = None
        self._epsilon_history = []  # epsilon历史记录
        self._control_adjustments = []  # 控制调整历史
        self.fixed_val_loader = None  # 固定验证加载器
        self._alpha_dampen_threshold = float(_arg_get(args, "btsp_alpha_dampen_threshold", 6.0))
        self._alpha_dampen_strength = float(_arg_get(args, "btsp_alpha_dampen_strength", 0.5))
        # 新增：α软钳制边界
        self._alpha_min = float(_arg_get(args, "btsp_alpha_min", 0.7))
        self._alpha_max = float(_arg_get(args, "btsp_alpha_max", 0.9))
        # 新增：EMA低通滤波和死区
        self._eps_ema_gamma = float(_arg_get(args, "btsp_eps_ema_gamma", 0.7))
        self._eps_deadzone = float(_arg_get(args, "btsp_eps_deadzone", 0.005))
        self._eps_actual_ema = None
        
        # 新增：epsilon校准功能（可选）
        self._eps_calib_enabled = bool(_arg_get(args, "btsp_eps_calib_enabled", False))
        self._eps_calib_gain = 1.0
        self._eps_calib_history = []
        
        # 实验日志组件
        self.experiment_logging = experiment_logging
        self.epsilon_control_probe = epsilon_control_probe
        self.capacity_probe = capacity_probe
        self.stc_probe = stc_probe
        
        if self.experiment_logging:
            self.logger = ExperimentLogger(log_file)
        else:
            self.logger = None
        
        # 控制系统和评测协议
        if enable_unified_protocol and EventDrivenControlSystem is not None:
            try:
                # 简化的评测协议：使用基本的日志记录功能
                self.evaluation_protocol = self
                
                if control_mode.upper() in ["A", "B"]:
                    mode = ControlMode.SIDECAR_ANALYSIS if control_mode.upper() == "A" else ControlMode.INTERVENTION
                else:
                    mode = ControlMode.SIDECAR_ANALYSIS
                
                self.control_system = EventDrivenControlSystem(
                    N_bits, num_classes, feat_dim, mode, target_epsilon, tau_e_steps, alpha
                )
                logging.info(f"BTSP control system initialized: mode={control_mode}, target_eps={target_epsilon}")
            except ImportError as e:
                logging.warning(f"Unified protocol disabled due to import error: {e}")

    def _clamp_p_gate(self, p_gate: float, prev_gate: float | None = None, 
                      mode: str = 'normal', global_occ: float = 0.0) -> tuple[float, list[str]]:
        """
        改进的p_gate钳制函数，包含全局下限和比例+绝对双钳制Δ带宽
        
        Args:
            p_gate: 候选门控率
            prev_gate: 上一次门控率
            mode: 模式('normal', 'rate', 'achievable')
            global_occ: 全局占用率
            
        Returns:
            tuple: (clamped_p_gate, flags)
        """
        flags: list[str] = []
        p_gate = float(p_gate)
        
        # 1. 硬上限约束
        if p_gate > self._p_gate_hard_cap:
            flags.append('hard_cap_hit')
            p_gate = self._p_gate_hard_cap
        
        # 2. 全局下限约束 
        if p_gate < self._p_gate_min:
            flags.append('min_cap_hit')
            p_gate = self._p_gate_min
        
        # 3. 改进的Δ带宽约束：比例+绝对双钳制
        if prev_gate is not None:
            # 计算动态Δ上限：max(相对变化, 绝对变化)
            delta_cap = max(prev_gate * self._p_gate_delta_rel, self._p_gate_delta_abs)
            upper = prev_gate + delta_cap
            lower = max(self._p_gate_min, prev_gate - delta_cap)  # 下界受全局下限约束
            
            if p_gate > upper:
                flags.append('delta_cap_hit')
                p_gate = upper
            if p_gate < lower:
                flags.append('delta_cap_hit')
                p_gate = lower
        
        # 4. 占用带约束
        if global_occ >= self._occ_upper_band:
            flags.append('occ_cap_hit')
            if prev_gate is not None:
                p_gate = min(p_gate, prev_gate)
        elif global_occ <= self._occ_lower_band and mode == 'rate':
            flags.append('occ_cap_hit')
            p_gate = max(p_gate, self._p_gate_min)  # 使用全局下限而非safe_rate
            
        # 5. 最终兜底约束
        p_gate = max(self._p_gate_min, min(p_gate, self._p_gate_hard_cap))
        
        return p_gate, flags

    def _achievable_check(self, p_gate_candidate: float, T_eff: float, 
                         stats: dict | None = None, budgets: dict | None = None) -> tuple[bool, str]:
        """
        可达性判据升级：检查资源合同约束
        🔥 修复：放宽判据标准，让achievable能返回true
        
        Args:
            p_gate_candidate: 候选门控率
            T_eff: 有效时间窗口
            stats: 当前统计信息 {'global_occ', 'flips_per_task', 'apb', 'latency_ms'}
            budgets: 资源预算 {'occ_high', 'F_max', 'APB_min', 'L_max'}
            
        Returns:
            tuple: (ok, reason)
        """
        if stats is None:
            stats = {}
        if budgets is None:
            budgets = {}
            
        # 🔥 放宽预算参数，让可达性更容易满足
        tau_occ_high = budgets.get('occ_high', 0.20)  # 占用率上限从0.13放宽到0.20
        occ_hysteresis_low = budgets.get('occ_low', 0.08)  # 占用率滞回下限从0.11放宽到0.08
        F_max = budgets.get('F_max', 10000.0)  # 每任务最大翻转数从1000放宽到10000
        APB_min = budgets.get('APB_min', 0.0001)  # 最小APB从0.001放宽到0.0001
        L_max = budgets.get('L_max', 1000.0)  # 最大延时从100ms放宽到1000ms
        
        # 1. 预测全局占用率检查（带滞回）
        global_occ = stats.get('global_occ', 0.0)
        # 简单预测：当前占用 + p_gate影响估算
        p_eff_est = 1.0 - math.exp(-p_gate_candidate * T_eff) if T_eff > 0 else 0.1
        occ_increase_est = p_eff_est * 0.005  # 减小影响估算（从0.01降为0.005）
        global_occ_next = global_occ + occ_increase_est
        
        # 🔥 放宽滞回检查
        if global_occ > occ_hysteresis_low and global_occ_next > tau_occ_high:
            return False, 'occ_cap_hit'
        elif global_occ <= occ_hysteresis_low and global_occ_next > tau_occ_high * 1.2:  # 滞回高于上限20%（从5%放宽）
            return False, 'occ_cap_hit'
            
        # 2. 翻转数预算检查（更宽松）
        flips_per_task = stats.get('flips_per_task', 0.0)
        if flips_per_task > F_max:
            return False, 'flips_cap_hit'
            
        # 3. APB检查（如果可得，更宽松）
        apb = stats.get('apb', None)
        if apb is not None and apb < APB_min and apb > 0:  # 只有正值且小于最小值才算失败
            return False, 'apb_cap_hit'
            
        # 4. 延迟检查（如果可得，更宽松）
        latency_ms = stats.get('latency_ms', None)
        if latency_ms is not None and latency_ms > L_max:
            return False, 'latency_cap_hit'
            
        # 追加：task级占用上限
        task_occ = stats.get('task_occ', None) if stats is not None else None
        task_high = budgets.get('task_high', self._occ_task_high if hasattr(self, '_occ_task_high') else 0.18)
        if task_occ is not None and task_occ > task_high:
            return False, 'task_occ_cap_hit'
        return True, 'ok'

    def _update_eps_calibration(self, p_gate: float, T_eff: float, global_occ: float, 
                               M: int, eps_actual: float):
        """更新epsilon校准参数"""
        if not self._eps_calib_enabled:
            return
            
        # 记录校准数据点
        calib_data = {
            'p_gate': p_gate,
            'T_eff': T_eff,
            'occ': global_occ,
            'M': M,
            'eps_actual': eps_actual
        }
        
        self._eps_calib_history.append(calib_data)
        
        # 保持最近4个任务的数据
        if len(self._eps_calib_history) > 4:
            self._eps_calib_history = self._eps_calib_history[-4:]
            
        # 如果有足够数据，计算线性校准增益
        if len(self._eps_calib_history) >= 3:
            try:
                # 简化线性校准：计算预测/实际比值的平均
                ratios = []
                for data in self._eps_calib_history:
                    eps_pred = epsilon_from_p_gate(data['p_gate'], data['M'], self.p_pre, data['T_eff'])
                    if eps_pred > 0:
                        ratio = data['eps_actual'] / eps_pred
                        if 0.2 <= ratio <= 5.0:  # 合理范围内的比值
                            ratios.append(ratio)
                
                if len(ratios) >= 2:
                    # 温和校准：原增益70% + 新比值30%
                    new_gain = np.mean(ratios)
                    self._eps_calib_gain = 0.7 * self._eps_calib_gain + 0.3 * new_gain
                    self._eps_calib_gain = max(0.5, min(2.0, self._eps_calib_gain))  # 限制范围
            except Exception as e:
                logging.warning(f"[BTSP Calibration] Update failed: {e}")

    def _calibrated_eps_pred(self, p_gate: float, M: int, p_pre: float, T_eff: float) -> float:
        """返回校准后的epsilon预测值"""
        raw_pred = epsilon_from_p_gate(p_gate, M, p_pre, T_eff)
        if self._eps_calib_enabled and hasattr(self, '_eps_calib_gain'):
            return raw_pred * self._eps_calib_gain
        return raw_pred

    @torch.no_grad()
    def before_task(self, task_id: int, known_classes: int, new_classes: int, eps0: float = 0.05,
                    delta_q_target: float = 0.0005, val_loader=None):
        """Configure BTSP gating schedule before each incremental task."""
        btsp = self._modules['btsp']
        btsp.expand_classes(known_classes + new_classes)

        if val_loader is not None:
            self.fixed_val_loader = val_loader

        if hasattr(btsp, 'recompute_teff'):
            btsp.recompute_teff()
        T_eff_val = float(btsp.T_eff.item())
        M = task_id + 1
        p_pre = self.p_pre

        class_range = (known_classes, known_classes + new_classes)
        self.current_task = task_id
        self._btsp_task_range = class_range

        global_occ = 0.0
        if hasattr(btsp, 'S'):
            global_occ = float(btsp.S.float().mean().item())

        q_c = 0.0
        if new_classes > 0 and hasattr(btsp, 'S'):
            occ = btsp.S.float().mean(dim=1)
            end_idx = min(class_range[1], occ.numel())
            if class_range[0] < end_idx:
                q_c = occ[class_range[0]:end_idx].mean().item()

        prev_gate = getattr(self, '_last_p_gate_value', float(btsp.p_gate.mean().item()))

        # 🔥 持久化计划读取：优先使用checkpoint中恢复的计划
        plan = None
        plan_source = 'fallback'
        if self._next_gate_plan and self._next_gate_plan.get('task') == task_id:
            plan = self._next_gate_plan
            # 区分计划来源
            if hasattr(self, '_plan_restored_from_checkpoint') and self._plan_restored_from_checkpoint:
                plan_source = 'checkpoint'
                self._plan_restored_from_checkpoint = False  # 重置标记
            else:
                plan_source = 'fresh'
            # 消费一次
            self._next_gate_plan = None
        
        # 记录提议的 p_gate（若有）
        proposed_p_gate = float(plan.get('p_gate', prev_gate)) if plan else prev_gate
        mode = str(plan.get('mode', self._default_gate_mode)).lower() if plan else self._default_gate_mode
        if mode not in {'achievable', 'rate'}:
            mode = 'rate'
        planned_eps = float(plan.get('target_eps', self.target_epsilon if self.target_epsilon is not None else eps0)) if plan else float(self.target_epsilon if self.target_epsilon is not None else eps0)
        delta_q_plan = float(plan.get('delta_q_target', self._delta_q_default)) if plan else float(self._delta_q_default)
        if eps0 is not None:
            planned_eps = float(eps0)
        planned_T_eff = float(plan.get('T_eff', T_eff_val)) if plan else T_eff_val
        planned_achievable = bool(plan.get('achievable', True)) if plan else False

        fallback_flags: list[str] = []
        p_gate_executed = prev_gate

        # 读取→可达性复核→钳制→执行
        if plan is not None:
            # 默认采用计划值，之后做复核与钳制
            candidate = float(proposed_p_gate)
            achievable_ok = planned_achievable
            # 仅对achievable计划复核资源护栏
            if mode == 'achievable' and achievable_ok:
                try:
                    flips_total = int(btsp.flip_counter.sum().item()) if hasattr(btsp, 'flip_counter') else 0
                    flips_per_task = flips_total / max(1, task_id) if task_id > 0 else flips_total
                    stats = {
                        'global_occ': global_occ,
                        'flips_per_task': flips_per_task
                    }
                    budgets = {
                        'occ_high': self._occ_upper_band,
                        'occ_low': self._occ_lower_band
                    }
                    resource_ok, resource_reason = self._achievable_check(candidate, planned_T_eff, stats, budgets)
                    if not resource_ok:
                        fallback_flags.append(f'achievable_{resource_reason}')
                        achievable_ok = False
                except Exception as exc:
                    fallback_flags.append('achievable_check_error')
                    achievable_ok = False
            
            if achievable_ok:
                # 钳制并执行
                p_gate_final, clamp_flags = self._clamp_p_gate(candidate, prev_gate, mode, global_occ)
                fallback_flags.extend(clamp_flags)
                p_gate_executed = p_gate_final
            else:
                # 不可达 → 回退到上一轮配置
                plan_source = 'fallback'
                p_gate_executed = prev_gate
        else:
            # 无计划 → 回退到上一轮配置
            plan_source = 'fallback'
            p_gate_executed = prev_gate

        btsp.p_gate.fill_(float(p_gate_executed))
        pred_eps = epsilon_from_p_gate(float(p_gate_executed), M=M, p_pre=p_pre, T_eff=T_eff_val)

        self._last_mode = mode
        self._last_delta_q_target = float(delta_q_plan)
        self._last_achievable = bool(planned_achievable)
        self._last_pred_eps = pred_eps
        self._last_p_gate_value = float(p_gate_executed)
        self._last_plan_source = plan_source
        self._next_gate_plan = None

        btsp.record_task_baseline(class_range=class_range)

        fallback_reason = 'none' if not fallback_flags else '|'.join(fallback_flags)
        
        # 使用统一日志格式
        unified_data = {
            'task': task_id,
            'classes_from': class_range[0],
            'classes_to': class_range[1],
            'mode': mode,
            'p_gate_proposed': float(proposed_p_gate),
            'p_gate_executed': float(p_gate_executed),
            'p_gate': float(p_gate_executed),
            'achievable': bool(planned_achievable),
            'eps_predicted': pred_eps,
            'eps_target': planned_eps,
            'plan_source': plan_source,
            'T_eff': T_eff_val,
            'p_pre': p_pre,
            'q_c': q_c,
            'global_occ': global_occ,
            'p_gate_prev': prev_gate,
            'fallback': fallback_reason,
            # 预算护栏参数（用于等预算复现）
            'budget_occ_high': self._occ_upper_band,
            'budget_occ_low': self._occ_lower_band,
            'budget_p_gate_min': self._p_gate_min,
            'budget_p_gate_cap': self._p_gate_hard_cap,
            # alpha 记录
            'alpha_is_btsp_weight': True,
            'alpha_eval': float(self.alpha if self.training else self._adaptive_alpha()),
            'alpha_mode': 'static' if self.training else 'adaptive',
        }
        self._log_unified_control("BTSP before_task", unified_data)

        if plan_source == 'scheduled':
            # 删除重复日志，统一格式已包含该信息
            pass

        if self.epsilon_control_probe:
            log_payload = {
                'probe': 'epsilon_control_pure_gating',
                'task_id': task_id,
                'M': M,
                'mode': mode,
                'eps0': planned_eps,
                'delta_q_plan': delta_q_plan,
                'delta_q_target_param': delta_q_target,
                'plan_source': plan_source,
                'fallback': fallback_reason,
                'p_pre': p_pre,
                'T_eff': T_eff_val,
                'q_c': q_c,
                'global_occ': global_occ,
                'p_gate_final': p_gate_executed,
                'p_gate_prev': prev_gate,
                'achievable': bool(planned_achievable),
                'pred_eps': pred_eps
            }
            self.log_data(log_payload)

        # 读取计划后，设置写掩码目标任务（若启用）
        task_target = None
        if plan is not None:
            task_target = plan.get('task_target', None)
        self._write_mask_target_task = int(task_target) if (task_target is not None and self._write_mask_enabled) else None

    @property
    def _network(self):
        """透传内层learner的_network属性供trainer使用"""
        inner = self._get_inner()
        return inner._network if hasattr(inner, '_network') else None
    
    def __getattr__(self, name):
        """Delegate missing attributes to the wrapped learner while keeping plugin state safe."""
        modules = object.__getattribute__(self, '_modules')
        if name in {'inner', 'proj', 'btsp'}:
            if name in modules:
                return modules[name]
            raise AttributeError(f"BTSPPlugin ?? {name} ??????")

        attrs = object.__getattribute__(self, '__dict__')
        if name in attrs:
            return attrs[name]

        internal_defaults = {
            '_default_gate_mode': attrs.get('_default_gate_mode', 'rate'),
            '_delta_q_default': attrs.get('_delta_q_default', 0.002),
            '_achievable_enabled': attrs.get('_achievable_enabled', False),
            '_p_gate_hard_cap': attrs.get('_p_gate_hard_cap', 0.05),
            '_p_gate_min': attrs.get('_p_gate_min', 0.006),
            '_p_gate_delta_limit': attrs.get('_p_gate_delta_limit', 0.002),
            '_p_gate_delta_abs': attrs.get('_p_gate_delta_abs', 0.002),
            '_p_gate_delta_rel': attrs.get('_p_gate_delta_rel', 0.5),
            '_p_gate_safe_rate': attrs.get('_p_gate_safe_rate', 0.005),
            '_occ_upper_band': attrs.get('_occ_upper_band', 0.13),
            '_occ_lower_band': attrs.get('_occ_lower_band', 0.11),
            '_eps_ema_gamma': attrs.get('_eps_ema_gamma', 0.7),
            '_eps_deadzone': attrs.get('_eps_deadzone', 0.005),
            '_eps_actual_ema': None,
            '_eps_calib_enabled': attrs.get('_eps_calib_enabled', False),
            '_eps_calib_gain': attrs.get('_eps_calib_gain', 1.0),
            '_eps_calib_history': attrs.get('_eps_calib_history', []),
            '_next_gate_plan': None,
            '_last_mode': attrs.get('_default_gate_mode', 'rate'),
            '_last_delta_q_target': attrs.get('_delta_q_default', 0.002),
            '_last_achievable': True,
            '_last_pred_eps': None,
            '_last_p_gate_value': attrs.get('_last_p_gate_value', 0.0),
            'current_task': -1,
            '_btsp_task_range': (0, 0),
            'fixed_val_loader': None,
            # 新增：窗口上限口径常量（统一为99%）
            '_window_cap_tau': attrs.get('_window_cap_tau', 0.99),
            # 新增：ACC匹配连续不一致计数器
            '_acc_mismatch_count': attrs.get('_acc_mismatch_count', 0),
        }
        if name in internal_defaults:
            return internal_defaults[name]
        
        inner = modules.get('inner')
        if inner is not None and hasattr(inner, name):
            return getattr(inner, name)
        raise AttributeError(name)

    def _get_inner(self):
        if 'inner' in self._modules:
            return self._modules['inner']
        return getattr(self, 'inner', None) or getattr(self, 'inner_ref', None)

    def forward(self, images: torch.Tensor, targets: torch.Tensor | None = None):
        inner = self._get_inner()
        if inner is None:
            raise RuntimeError("BTSPPlugin could not find inner instance, cannot forward")

        # ========== 调用底层模型，兼容不同签名 ==========
        try:
            # 尝试不同的调用方式，从复杂到简单
            out = None
            
            # 特殊处理：TUNA模型
            if hasattr(inner, '_network') and hasattr(inner._network, 'forward_orig'):
                try:
                    out = inner._network.forward_orig(images)
                    logging.debug("[BTSP] 使用TUNA的forward_orig方法")
                except Exception as e:
                    logging.warning(f"[BTSP] TUNA forward_orig失败: {e}")
            
            # 1. 尝试带targets的调用 (某些learner需要)
            if out is None:
                try:
                    out = inner(images, targets)
                except TypeError:
                    pass
                
            # 2. 尝试基础调用
            if out is None:
                try:
                    out = inner(images)
                except TypeError:
                    pass
            
            # 3. 对于有特殊参数的网络，尝试常见组合
            if out is None:
                import inspect
                sig = inspect.signature(inner.forward) if hasattr(inner, 'forward') else None
                if sig is not None:
                    params = sig.parameters
                    kwargs = {}
                    
                    # 检测常见的特殊参数并设置默认值
                    if 'bcb_no_grad' in params:
                        kwargs['bcb_no_grad'] = False
                    if 'fc_only' in params:
                        kwargs['fc_only'] = False
                    if 'train' in params:
                        kwargs['train'] = self.training
                    if 'task_id' in params and hasattr(self, 'current_task'):
                        kwargs['task_id'] = getattr(self, 'current_task', 0)
                    if 'adapter_id' in params and hasattr(self, 'current_task'):
                        kwargs['adapter_id'] = getattr(self, 'current_task', 0)
                        
                    try:
                        out = inner(images, **kwargs)
                    except Exception:
                        pass
            
            if out is None:
                out = self._fallback_inner_forward(inner, images, targets)
            if out is None:
                raise RuntimeError("Cannot call inner model, all signature attempts failed")
                
        except Exception as e:
            raise RuntimeError(f"Inner forward call failed: {e}")

        base_logits, feat = None, None
        # 统一解析输出
        if isinstance(out, dict):
            base_logits = out.get("logits")
            if base_logits is None:
                base_logits = out.get("output")
            feat = out.get("feat")
            if feat is None:
                feat = out.get("features")
            if feat is None:
                feat = out.get("pre_logits")  # DualPrompt 使用 pre_logits
        elif isinstance(out, tuple):
            if len(out) == 2:
                # 检查是否是 (logits, loss) 这种模式 (如 coda_prompt)
                # 如果第二个元素是标量loss，就不能当作特征
                first, second = out
                if torch.is_tensor(first) and torch.is_tensor(second):
                    if second.dim() == 0 or (second.dim() == 1 and second.numel() == 1):
                        # second 是标量损失，first 应该是 logits
                        base_logits = first
                        feat = None  # 需要后续处理
                    else:
                        # 正常的 (logits, features) 模式
                        base_logits, feat = first, second
                else:
                    base_logits, feat = first, second
            elif len(out) > 2:
                base_logits, feat = out[0], out[-1]
        elif torch.is_tensor(out):
            base_logits = out

        # 进一步回退：尝试 extract_vector 或 _network.extract_vector
        if feat is None:
            extract_holder = None
            if hasattr(inner, 'extract_vector'):
                extract_holder = inner
            elif hasattr(inner, '_network') and hasattr(inner._network, 'extract_vector'):
                extract_holder = inner._network
            if extract_holder is not None:
                try:
                    feat = extract_holder.extract_vector(images.to(self.proj.W.device))
                except Exception:
                    feat = None
                    
        # 如果底层 dict 里已有 features 但未放入 feat
        if feat is None and isinstance(out, dict):
            for k in ('feature','rep','emb','embedding','pre_logits'):  # 添加 pre_logits
                if k in out and torch.is_tensor(out[k]):
                    feat = out[k]; break
                    
        # 最后的回退：尝试调用backbone直接提取特征 (如 coda_prompt 等)
        if feat is None:
            backbone_holder = None
            if hasattr(inner, 'backbone'):
                backbone_holder = inner.backbone
            elif hasattr(inner, '_network') and hasattr(inner._network, 'backbone'):
                backbone_holder = inner._network.backbone
            
            if backbone_holder is not None:
                try:
                    with torch.no_grad():
                        backbone_out = backbone_holder(images.to(self.proj.W.device))
                        if isinstance(backbone_out, dict):
                            feat = backbone_out.get('features') or backbone_out.get('pre_logits')
                        elif torch.is_tensor(backbone_out):
                            feat = backbone_out
                        # 对于ViT类backbone，可能需要[:,0,:]选择CLS token
                        if feat is not None and feat.dim() == 3:
                            feat = feat[:, 0, :]  # CLS token
                except Exception as e:
                    logging.warning(f"[BTSP] backbone特征提取失败: {e}")
                    feat = None

        if feat is None or base_logits is None:
            raise RuntimeError("Cannot parse logits and features from inner output, need to adapt model output format.")

        if feat.dim() > 2:
            feat = feat.view(feat.size(0), -1)

        # 旁路（GPU→CPU）：feat→二值→BTSP 检索
        x_bits_gpu = self.proj(feat, topk=self.topk)
        x_bits_cpu = x_bits_gpu.to(torch.bool).cpu()
        
        # --- Store x_bits for capacity probe ---
        if self.training and (self.capacity_probe or self.stc_probe) and self.x_bits_buffer is not None:
            self.x_bits_buffer.append(x_bits_cpu.numpy())
        
        # 🔥 稳健的z-score更新策略：训练时间隔更新，评测时冻结
        if self.training:
            # 获取当前scores用于可能的统计量更新
            raw_scores = self.btsp.raw_scores(x_bits_cpu)
            # 每隔zstats_interval步更新一次z-score统计量（类似BatchNorm的moving stats）
            if self._gstep % self.zstats_interval == 0:
                self.btsp.update_zstats(raw_scores)
            # 总是使用最新的统计量进行检索
            btsp_logits_cpu = self.btsp.retrieve(x_bits_cpu, update_z=False)
        else:
            # 评测时完全冻结，保障可复现性
            btsp_logits_cpu = self.btsp.retrieve(x_bits_cpu, update_z=False)
            
        # === 🔥 修复：设备对齐后再融合 ===
        # 使用固定fusion_alpha融合策略，epsilon控制移到任务级别（after_task中）
        btsp_logits = btsp_logits_cpu.to(base_logits.device, non_blocking=True)
        # 统一语义：alpha 表示 BTSP 权重（fused = (1-α)*base + α*btsp）。
        # 训练默认不融合(eval=True)符合形态A；评测使用融合输出作为"系统最终输出"。
        btsp_weight_alpha = float(self.alpha)
        if not self.training:
            btsp_weight_alpha = float(self._adaptive_alpha())
        fusion_for_metrics = fuse_logits(base_logits, btsp_logits, fusion_alpha=btsp_weight_alpha)
        
        # 训练期不融合（默认），评估期融合（默认）；可由JSON开关覆盖
        if self.training:
            logits = fusion_for_metrics if self.use_fusion_train else base_logits
        else:
            logits = fusion_for_metrics if self.use_fusion_eval else base_logits

        # === Class-range alignment for evaluation ===
        # Ensure predictions only consider currently known classes, matching trainer protocol
        try:
            if not self.training and isinstance(logits, torch.Tensor):
                inner = self._get_inner()
                total_classes = getattr(inner, '_total_classes', None)
                if isinstance(total_classes, int) and total_classes > 0 and logits.shape[1] >= total_classes:
                    logits = logits[:, :total_classes]
        except Exception:
            pass

        # 🔥 简化诊断日志：融合能量探针（降低频率）
        if self._gstep % max(100, self.log_interval * 10) == 0:  # 从500改为基于log_interval
            base_norm = float(base_logits.norm().item())
            btsp_norm = float(btsp_logits.norm().item()) 
            diff_norm = float((fusion_for_metrics - base_logits).norm().item())
            
            # 检查z-score和温度
            z_mu_mean = float(self.btsp.z_mu.mean().item()) if hasattr(self.btsp, 'z_mu') else 0.0
            z_std_mean = float(self.btsp.z_std.mean().item()) if hasattr(self.btsp, 'z_std') else 1.0
            temperature = float(self.btsp.temperature) if hasattr(self.btsp, 'temperature') else 1.0
            
            # 占用率概览
            if hasattr(self.btsp, 'S'):
                occ_mean = float(self.btsp.S.float().mean().item())
                occ_max = float(self.btsp.S.float().mean(dim=1).max().item())
            else:
                occ_mean = occ_max = 0.0
            
            # 门控状态
            p_gate_mean = float(self.btsp.p_gate.mean().item()) if hasattr(self.btsp, 'p_gate') else 0.0
            T_eff_val = float(self.btsp.T_eff.item()) if hasattr(self.btsp, 'T_eff') else 1.0
            
            logging.debug(f"[BTSP Fusion] step={self._gstep}: "
                        f"||base||={base_norm:.3f}, ||btsp||={btsp_norm:.3f}, "
                        f"||fused-base||={diff_norm:.3f}, α={self.alpha:.2f}, "
                        f"p_gate={p_gate_mean:.4f}, T_eff={T_eff_val:.2f}, "
                        f"z_std={z_std_mean:.3f}, T={temperature:.2f}, occ_mean={occ_mean:.4f}")
            
            # 融合效果检查
            if diff_norm < 1e-3:
                logging.warning(f"[BTSP Control] Warning: weak fusion effect ||fused-base||={diff_norm:.6f} < 0.001")
        
        # 评测前保真验证（每次eval开始时记录一次）
        if not self.training and self._gstep % max(10, self.log_interval) == 0:
            flips_total = int(self.btsp.flip_counter.sum().item()) if hasattr(self.btsp, 'flip_counter') else 0
            occ_mean = float(self.btsp.S.sum() / self.btsp.S.numel()) if hasattr(self.btsp, 'S') else 0.0
            if flips_total == 0:
                logging.warning("[BTSP] Eval found zero flips, BTSP memory may be empty!")
            elif occ_mean < 0.001:
                logging.warning(f"[BTSP] Eval occupancy too low: {occ_mean:.6f}")
            else:
                logging.debug(f"[BTSP] Eval state normal: flips={flips_total}, occ={occ_mean:.6f}")

        if self.training and (targets is not None):
            # 简化写入逻辑：训练时总是写入
            should_write = True
            
            if should_write:
                # 写掩码：仅针对 task_sla 模式且启用时，筛选属于目标队列的标签
                if self._write_mask_enabled and self._control_mode == 'task_sla' and isinstance(self._write_mask_target_task, int):
                    inner = self._get_inner()
                    args_inner = getattr(inner, 'args', None)
                    init_cls = getattr(args_inner, 'init_cls', getattr(inner, 'init_cls', 10))
                    increment = getattr(args_inner, 'increment', getattr(inner, 'increment', 10))
                    a = int(self._write_mask_target_task * increment)
                    b = int(a + increment)
                    mask = (targets >= a) & (targets < b)
                    with torch.no_grad():
                        x_bits_cpu = x_bits_gpu.to(torch.bool).cpu()
                        if mask.any():
                            self.btsp.write(x_bits_cpu[mask.cpu().numpy()], targets[mask].cpu().long(), tau_e=None)
                    self._gstep += 1
                    if (self._gstep % self.zstats_interval) == 0:
                        scores = (x_bits_cpu.float() @ self.btsp.S.transpose(0,1).float())
                        self.btsp.update_zstats(scores)
                    if (self._gstep % self.homeo_interval) == 0:
                        try:
                            homeo_stats = self.btsp.homeostasis_step()
                            if homeo_stats.get('adjusted', False):
                                logging.info(f"[BTSP Homeostasis] step={self._gstep}: p_gate={homeo_stats.get('p_gate_new', 0.0):.6f}")
                        except Exception as e:
                            logging.warning(f"[BTSP Homeostasis] failed: {e}")
                else:
                    with torch.no_grad():
                        x_bits_cpu = x_bits_gpu.to(torch.bool).cpu()
                        self.btsp.write(x_bits_cpu, targets.cpu().long(), tau_e=None)
                    self._gstep += 1
                    if (self._gstep % self.zstats_interval) == 0:
                        scores = (x_bits_cpu.float() @ self.btsp.S.transpose(0,1).float())
                        self.btsp.update_zstats(scores)
                    if (self._gstep % self.homeo_interval) == 0:
                        try:
                            homeo_stats = self.btsp.homeostasis_step()
                            if homeo_stats.get('adjusted', False):
                                logging.info(f"[BTSP Homeostasis] step={self._gstep}: p_gate={homeo_stats.get('p_gate_new', 0.0):.6f}")
                        except Exception as e:
                            logging.warning(f"[BTSP Homeostasis] failed: {e}")
        
        # === 🧪 简化评测记录 ===
        if self.training and self.evaluation_protocol and targets is not None and self._gstep % 100 == 0:  # 降低频率到每100步
            try:
                # 计算当前精度指标
                with torch.no_grad():
                    preds = logits.argmax(dim=1)
                    acc_total = (preds == targets).float().mean().item()
                    
                # 简化记录
                unified_log = {
                    "step": self._gstep,
                    "task": self.current_task,
                    "acc_total": acc_total,
                    "p_gate_mean": float(self.btsp.p_gate.mean().item()) if hasattr(self.btsp, 'p_gate') else 0.0,
                    "occ_mean": float(self.btsp.S.float().mean().item()) if hasattr(self.btsp, 'S') else 0.0
                }
                self.evaluation_protocol.record_unified_log(unified_log)
                
            except Exception as e:
                logging.warning(f"[Unified Protocol] Recording failed: {e}")

        return {"logits": logits, "base_logits": base_logits, "feat": feat, "fusion_for_metrics": fusion_for_metrics}

    # =======================================================
    # ========= Experiment Logging & Probes Methods =========
    # =======================================================
    def _setup_logging(self):
        if self.experiment_logging and self.log_file:
            self.logger = ExperimentLogger(self.log_file)
            print(f"Experiment logging enabled. Writing to {self.log_file}")

    def log_data(self, data_dict: dict):
        """Logs a dictionary of data to the experiment log file."""
        if self.logger:
            payload = {
                "gstep": self._gstep,
                "task": self.current_task,
                **data_dict
            }
            self.logger.log_step(payload)
            
    def log_eval_metrics(self, metrics_dict: dict):
        """Public method to log metrics computed externally (e.g., in the trainer)."""
        payload = {"probe": "eval_metrics", **metrics_dict}
        self.log_data(payload)

    def record_unified_log(self, log_dict: dict):
        """Record unified evaluation log - compatibility method for evaluation_protocol."""
        if self.logger:
            # 添加unified前缀来区分这类日志
            payload = {"probe": "unified_eval", **log_dict}
            self.log_data(payload)
        else:
            # 如果没有logger，至少记录到标准日志
            logging.info(f"[Unified Eval] step={log_dict.get('step', 'N/A')} "
                        f"task={log_dict.get('task', 'N/A')} "
                        f"acc={log_dict.get('acc_total', 'N/A'):.3f}")
    
    def _log_unified_control(self, prefix: str, data: dict, level: str = "info"):
        """统一日志输出函数：同时记录结构化数据和格式化控制台输出"""
        try:
            # 1. 结构化日志记录（实验复现）
            if self.experiment_logging and self.logger:
                log_payload = {"probe": prefix.lower().replace(" ", "_"), **data}
                self.log_data(log_payload)
            
            # 2. 控制台格式化输出（实时监控）
            # 针对已知前缀做专门格式化，保证字段顺序与验收工具一致
            # 2.1 after_task：BTSP Epsilon Control
            if prefix == "BTSP Epsilon Control":
                frags = []
                # 必备字段（顺序固定）
                if 'task' in data:
                    frags.append(f"task={data['task']}")
                if 'eps_target' in data and data['eps_target'] is not None:
                    frags.append(f"eps_target={data['eps_target']:.4f}")
                if 'eps_actual' in data and data['eps_actual'] is not None:
                    frags.append(f"eps_actual={data['eps_actual']:.4f}")
                # 统一 eps_pred 名称
                eps_pred_val = data.get('eps_predicted', data.get('eps_pred', None))
                if eps_pred_val is not None:
                    frags.append(f"eps_pred={float(eps_pred_val):.4f}")
                # 统一 control_err 名称
                ctrl_err = data.get('control_error', data.get('control_err', None))
                if ctrl_err is not None:
                    frags.append(f"control_err={float(ctrl_err):.4f}")
                if 'coverage' in data and data['coverage'] is not None:
                    frags.append(f"coverage={float(data['coverage']):.1f}")
                if 'rmse' in data and data['rmse'] is not None:
                    frags.append(f"rmse={float(data['rmse']):.4f}")
                if 'coverage_rate' in data and data['coverage_rate'] is not None:
                    frags.append(f"coverage_rate={float(data['coverage_rate']):.3f}")
                if 'acc_old' in data and data['acc_old'] is not None:
                    frags.append(f"acc_old={float(data['acc_old']):.2f}")
                if 'acc_total' in data and data['acc_total'] is not None:
                    frags.append(f"acc_total={float(data['acc_total']):.2f}")
                if 'acc_new' in data and data['acc_new'] is not None:
                    frags.append(f"acc_new={float(data['acc_new']):.2f}")
                # 追加：task-slice字段
                if 'slice_mode' in data:
                    frags.append(f"slice_mode={data['slice_mode']}")
                if 'task_target' in data and data['task_target'] is not None:
                    frags.append(f"task_target={int(data['task_target'])}")
                if 'classes' in data and data['classes'] is not None:
                    cls_val = data['classes']
                    if isinstance(cls_val, (list, tuple)) and len(cls_val) == 2:
                        frags.append(f"classes=({int(cls_val[0])},{int(cls_val[1])})")
                if 'A_base_task' in data and data['A_base_task'] is not None:
                    frags.append(f"A_base_task={float(data['A_base_task']):.2f}")
                if 'A_old_task' in data and data['A_old_task'] is not None:
                    frags.append(f"A_old_task={float(data['A_old_task']):.2f}")
                if 'eps_target_task' in data and data['eps_target_task'] is not None:
                    frags.append(f"eps_target_task={float(data['eps_target_task']):.4f}")
                if 'eps_actual_task' in data and data['eps_actual_task'] is not None:
                    frags.append(f"eps_actual_task={float(data['eps_actual_task']):.4f}")
                if 'violation_task' in data and data['violation_task'] is not None:
                    frags.append(f"violation_task={float(data['violation_task']):.4f}")
                # p_gate 当前
                p_cur = data.get('p_gate_current', data.get('p_gate', None))
                if p_cur is not None:
                    frags.append(f"p_gate={float(p_cur):.6f}")
                # T_eff 当前
                t_cur = data.get('T_eff_current', data.get('T_eff', None))
                if t_cur is not None:
                    frags.append(f"T_eff={float(t_cur):.2f}")
                # 占用
                if 'occ_task' in data and data['occ_task'] is not None:
                    frags.append(f"occ_task={float(data['occ_task']):.4f}")
                if 'global_occ' in data and data['global_occ'] is not None:
                    frags.append(f"global_occ={float(data['global_occ']):.4f}")
                # 资源
                if 'bytes_btsp' in data and data['bytes_btsp'] is not None:
                    frags.append(f"bytes_btsp={float(data['bytes_btsp']):.0f}")
                if 'apb' in data and data['apb'] is not None:
                    frags.append(f"apb={float(data['apb']):.6f}")
                # 控制/下一步
                if 'need_adjustment' in data:
                    frags.append(f"need_adj={data['need_adjustment']}")
                if 'p_gate_next' in data and data['p_gate_next'] is not None:
                    frags.append(f"p_gate_next={float(data['p_gate_next']):.6f}")
                # flags 最后
                if 'flags' in data:
                    flags_val = data['flags']
                    if isinstance(flags_val, list):
                        flags_str = '|'.join(flags_val)
                    else:
                        flags_str = str(flags_val)
                    frags.append(f"flags={flags_str}")
                metrics_str = " ".join(frags)
                log_message = f"[{prefix}] {metrics_str}"
                if level.lower() == "debug":
                    logging.debug(log_message)
                elif level.lower() == "warning":
                    logging.warning(log_message)
                elif level.lower() == "error":
                    logging.error(log_message)
                else:
                    logging.info(log_message)
                return
            
            # 2.2 before_task：BTSP before_task
            if prefix == "BTSP before_task":
                frags = []
                if 'task' in data:
                    frags.append(f"task={data['task']}")
                # classes 范围
                c_from = data.get('classes_from')
                c_to = data.get('classes_to')
                if c_from is not None and c_to is not None:
                    frags.append(f"classes={int(c_from)}->{int(c_to)}")
                if 'mode' in data:
                    frags.append(f"mode={data['mode']}")
                if 'p_gate_proposed' in data:
                    frags.append(f"p_gate_proposed={float(data['p_gate_proposed']):.6f}")
                if 'p_gate_executed' in data:
                    frags.append(f"p_gate_executed={float(data['p_gate_executed']):.6f}")
                if 'p_gate' in data:
                    frags.append(f"p_gate={float(data['p_gate']):.6f}")
                if 'achievable' in data:
                    frags.append(f"achievable={data['achievable']}")
                # 统一 pred_eps 与 target_eps 命名
                eps_pred_val = data.get('eps_predicted', data.get('eps_pred', None))
                if eps_pred_val is not None:
                    frags.append(f"pred_eps={float(eps_pred_val):.4f}")
                if 'eps_target' in data:
                    frags.append(f"target_eps={float(data['eps_target']):.4f}")
                # 计划来源
                if 'plan_source' in data:
                    frags.append(f"plan={data['plan_source']}")
                # 其他生命体征
                if 'T_eff' in data:
                    frags.append(f"T_eff={float(data['T_eff']):.2f}")
                if 'p_pre' in data:
                    frags.append(f"p_pre={float(data['p_pre']):.6f}")
                if 'q_c' in data:
                    frags.append(f"q_c={float(data['q_c']):.4f}")
                if 'global_occ' in data:
                    frags.append(f"global_occ={float(data['global_occ']):.4f}")
                if 'p_gate_prev' in data:
                    frags.append(f"prev_gate={float(data['p_gate_prev']):.6f}")
                if 'fallback' in data:
                    frags.append(f"fallback={data['fallback']}")
                if 'alpha_eval' in data:
                    frags.append(f"alpha_eval={float(data['alpha_eval']):.2f}")
                if 'alpha_mode' in data:
                    frags.append(f"alpha_mode={data['alpha_mode']}")
                metrics_str = " ".join(frags)
                log_message = f"[{prefix}] {metrics_str}"
                if level.lower() == "debug":
                    logging.debug(log_message)
                elif level.lower() == "warning":
                    logging.warning(log_message)
                elif level.lower() == "error":
                    logging.error(log_message)
                else:
                    logging.info(log_message)
                return
            
            # 2.3 通用格式（向后兼容）
            # 核心指标键值对
            key_metrics = []
            if 'task' in data or 'task_id' in data:
                task_val = data.get('task', data.get('task_id', 'N/A'))
                key_metrics.append(f"task={task_val}")
            if 'eps_target' in data:
                key_metrics.append(f"eps_target={data['eps_target']:.4f}")
            if 'eps_actual' in data:
                key_metrics.append(f"eps_actual={data['eps_actual']:.4f}")
            if 'eps_predicted' in data or 'eps_pred' in data:
                eps_pred = data.get('eps_predicted', data.get('eps_pred', None))
                if eps_pred is not None:
                    key_metrics.append(f"eps_pred={eps_pred:.4f}")
            if 'control_error' in data or 'control_err' in data:
                ctrl_err = data.get('control_error', data.get('control_err', None))
                if ctrl_err is not None:
                    key_metrics.append(f"control_err={ctrl_err:.4f}")
            if 'p_gate' in data:
                key_metrics.append(f"p_gate={data['p_gate']:.6f}")
            if 'p_gate_current' in data:
                key_metrics.append(f"p_gate={data['p_gate_current']:.6f}")
            if 'p_gate_next' in data:
                key_metrics.append(f"p_gate_next={data['p_gate_next']:.6f}")
            if 'global_occ' in data or 'occupancy' in data:
                occ_val = data.get('global_occ', data.get('occupancy', None))
                if occ_val is not None:
                    key_metrics.append(f"occ={occ_val:.4f}")
            # T_eff current/next
            t_eff_val = data.get('T_eff', data.get('T_eff_current', None))
            if t_eff_val is not None:
                key_metrics.append(f"T_eff={t_eff_val:.2f}")
            if 'T_eff_next' in data and data['T_eff_next'] is not None:
                key_metrics.append(f"T_eff_next={data['T_eff_next']:.2f}")
            if 'mode' in data:
                key_metrics.append(f"mode={data['mode']}")
            if 'achievable' in data:
                key_metrics.append(f"achievable={data['achievable']}")
            if 'flags' in data or 'notes' in data:
                flag_list = data.get('flags', data.get('notes', []))
                if isinstance(flag_list, list) and flag_list:
                    key_metrics.append(f"flags={'|'.join(flag_list)}")
                elif isinstance(flag_list, str) and flag_list:
                    key_metrics.append(f"flags={flag_list}")
            
            # 准确率字段
            if 'acc_old' in data and 'acc_total' in data and 'acc_new' in data:
                key_metrics.append(f"acc_old={data['acc_old']:.2f}")
                key_metrics.append(f"acc_total={data['acc_total']:.2f}")
                key_metrics.append(f"acc_new={data['acc_new']:.2f}")
            if 'ACC_MATCH' in data and data['ACC_MATCH'] is not None:
                key_metrics.append(f"ACC_MATCH={float(data['ACC_MATCH']):.2f}%")
            
            # 资源字段
            if 'bytes_btsp' in data:
                bytes_val = float(data['bytes_btsp'])
                key_metrics.append(f"bytes_btsp={bytes_val:.0f}")
                if bytes_val > 1e6:
                    key_metrics.append(f"mem={bytes_val/1e6:.3f}MB")
                else:
                    key_metrics.append(f"mem={bytes_val/1e3:.1f}KB")
            if 'apb' in data and data['apb'] is not None:
                key_metrics.append(f"apb={data['apb']:.6f}")
            
            # 控制状态字段
            if 'coverage' in data:
                key_metrics.append(f"coverage={data['coverage']:.1f}")
            if 'rmse' in data:
                key_metrics.append(f"rmse={data['rmse']:.4f}")
            if 'need_adjustment' in data:
                key_metrics.append(f"need_adj={data['need_adjustment']}")
            
            # 生成统一格式的日志
            metrics_str = " ".join(key_metrics)
            log_message = f"[{prefix}] {metrics_str}"
            
            # 根据级别输出
            if level.lower() == "debug":
                logging.debug(log_message)
            elif level.lower() == "warning":
                logging.warning(log_message)
            elif level.lower() == "error":
                logging.error(log_message)
            else:
                logging.info(log_message)
                
        except Exception as e:
            logging.warning(f"[Unified Log] {prefix} failed: {e}")

    def run_probes(self, task_id: int):
        """Public method to be called after a task's evaluation."""
        if not self.experiment_logging:
            return
            
        logging.info(f"Running post-task probes for task {task_id}...")
        if self.capacity_probe:
            self._run_capacity_probe(task_id)
        if self.stc_probe:
            self._run_stc_probe(task_id)

    def _run_capacity_probe(self, task_id: int):
        """Calculates and logs capacity-related metrics."""
        if not self.x_bits_buffer:
            logging.warning("[Capacity Probe] No recent x_bits found to analyze.")
            return

        all_x_bits = np.concatenate(list(self.x_bits_buffer), axis=0)
        self.x_bits_buffer.clear() # Clear buffer after use

        cov_matrix = estimate_cov(all_x_bits)
        if cov_matrix is None:
            return
        
        N = cov_matrix.shape[0]
        # (sum of off-diagonal elements) / (number of off-diagonal elements)
        rho_bar = (cov_matrix.sum() - np.trace(cov_matrix)) / (N * (N - 1))
        
        n_eff_simple = n_eff_from_rho(N, rho_bar)
        n_eff_robust = n_eff_robust_from_cov(cov_matrix)

        log_payload = {
            "probe": "capacity",
            "task_id": task_id,
            "N_eff_simple": n_eff_simple,
            "N_eff_robust": n_eff_robust,
            "rho_bar": float(rho_bar),
            "p_pre": self.p_pre
        }
        self.log_data(log_payload)
        logging.info(f"[Capacity Probe] N_eff_robust={n_eff_robust:.2f}, rho_bar={rho_bar:.4f}")

    def _run_stc_probe(self, task_id: int):
        """Runs the STC probe by calling write() in dry-run mode."""
        if not self.stc_probe_deltas:
            logging.warning("[STC Probe] No `stc_probe_deltas` defined.")
            return
        
        # We need a sample of data to run the probe
        # This part requires access to a dataloader, which the plugin doesn't have.
        # This is a placeholder for the logic that should be triggered from the main script.
        # The main script should provide a sample batch (images, labels) to this function.
        # For now, we will just log a warning.
        logging.warning("[STC Probe] STC probe needs a data sample and is not fully implemented inside the plugin."
                        " Please call it from your training loop with a data batch.")
        # Example of what would be logged:
        # for delta_k in self.stc_probe_deltas:
        #     # Assume self.btsp.write can return probe results in dry_run
        #     probe_results = self.btsp.write(x_bits_sample, labels_sample, dry_run=True, delta_k=delta_k)
        #     log_payload = {
        #         "probe": "stc",
        #         "task_id": task_id,
        #         "delta_k": delta_k,
        #         "flips_per_sample": probe_results.get("flips_per_sample"),
        #         "delta_score": probe_results.get("delta_score")
        #     }
        #     self.log_data(log_payload)

    def close_logger(self):
        """Closes the experiment logger file."""
        if self.logger:
            self.logger.close()
            logging.info(f"Experiment log file {self.log_file} closed.")

    # ========= 显式转发训练流程相关接口，避免属性透传阶段的潜在问题 ========= #
    def incremental_train(self, data_manager, *args, **kwargs):
        """
        调用内层模型的 incremental_train，然后添加 BTSP write-pass
        处理顺序: SGD → write_pass(到 q_target) → update_zstats(完整) → eval
        """
        # 调用内层的 incremental_train
        result = self.inner.incremental_train(data_manager, *args, **kwargs)

        # BTSP write-pass 阶段：从训练数据中提取特征并写入
        if hasattr(self, 'current_task'):
            task_id = self.current_task
        else:
            task_id = 0
        
        try:
            # 🔥 Fix: Use correct class range recorded in before_task
            if hasattr(self, '_btsp_task_range'):
                k0, k1 = self._btsp_task_range
                logging.info(f"[BTSP write-pass] Using recorded class range: task={task_id} class_range=[{k0}, {k1})")
            elif hasattr(data_manager, 'get_task_size'):
                # Backup: recalculate current task's class range
                known_classes = sum(data_manager.get_task_size(i) for i in range(task_id))
                new_classes = data_manager.get_task_size(task_id)
                k0, k1 = known_classes, known_classes + new_classes
                logging.warning(f"[BTSP write-pass] Recalculating class range: task={task_id} class_range=[{k0}, {k1})")
            else:
                # Final fallback
                k0, k1 = 0, 100
                logging.error(f"[BTSP write-pass] Using fallback class range: task={task_id} class_range=[{k0}, {k1})")
            
            # 获取训练数据加载器用于 write-pass
            try:
                import numpy as np
                from torch.utils.data import DataLoader
                
                train_dataset = data_manager.get_dataset(
                    np.arange(k0, k1), source='train', mode='train', 
                    appendent=None
                )
                loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
                
                logging.info(f"[BTSP write-pass] Starting task {task_id} class range: [{k0}, {k1})")
                
                inner = self.inner
                steps = 0
                proj_device = self.proj.W.device  # W 是 buffer，不是 parameter
                for batch in loader:
                    if isinstance(batch, (list, tuple)) and len(batch) == 3:
                        _, images, labels = batch
                    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                        images, labels = batch
                    else:
                        continue
                    images = images.to(proj_device, non_blocking=True)
                    labels = labels.to('cpu').long()  # 确保 dtype=torch.long
                    with torch.no_grad():
                        feat = self._try_extract_any_feature(inner, images)
                        if feat is None:
                            continue  # 无法取到特征则跳过
                        if feat.dim() > 2:
                            feat = feat.view(feat.size(0), -1)
                        if feat.dim() != 2:
                            # 展平除 batch 外其余维度
                            feat = feat.view(feat.size(0), -1)
                        x_bits_gpu = self.proj(feat, topk=self.topk)
                        self.btsp.write(x_bits_gpu.to(torch.bool).cpu(), labels, tau_e=None)
                        steps += 1
                        if steps % max(1, self.zstats_interval//4) == 0:
                            scores = (x_bits_gpu.to(torch.float32).cpu() @ self.btsp.S.transpose(0,1).float())
                            self.btsp.update_zstats(scores)
                        if steps % max(1, self.homeo_interval//4) == 0:
                            self.btsp.homeostasis_step()
                        # 移除20步限制，让write-pass完整执行
                        # if steps >= 20:
                        #     break
                
                logging.info(f"[BTSP write-pass] task_range=({k0},{k1}) flips_total={int(self.btsp.flip_counter.sum().item())}")
                
                # 🔥 Critical fix: Complete z-score statistics update before evaluation
                logging.info("[BTSP] Complete z-score statistics update before evaluation...")
                
                # 重新遍历数据来获取完整的分数分布
                all_scores = []
                for batch in loader:
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        if len(batch) == 3:
                            _, images, _ = batch
                        else:
                            images, _ = batch
                    else:
                        continue
                    images = images.to(proj_device, non_blocking=True)
                    with torch.no_grad():
                        feat = self._try_extract_any_feature(inner, images)
                        if feat is None:
                            continue
                        if feat.dim() > 2:
                            feat = feat.view(feat.size(0), -1)
                        if feat.dim() != 2:
                            feat = feat.view(feat.size(0), -1)
                        x_bits_gpu = self.proj(feat, topk=self.topk)
                        x_bits_cpu = x_bits_gpu.to(torch.bool).cpu()
                        scores = (x_bits_cpu.to(torch.float32) @ self.btsp.S.transpose(0,1).float())
                        all_scores.append(scores)
                
                if all_scores:
                    final_scores = torch.cat(all_scores, dim=0)
                    self.btsp.update_zstats(final_scores)
                    samples_used = final_scores.shape[0]
                    z_mu_mean = self.btsp.z_mu.mean().item()
                    z_std_mean = self.btsp.z_std.mean().item()
                    logging.info(f"[BTSP] z-score statistics complete: used {samples_used} samples, z_mu={z_mu_mean:.4f}, z_std={z_std_mean:.4f}")
                
                # 🔥 新增: 任务完成后的保真日志
                self._log_task_completion_stats(task_id, k0, k1)
            except Exception as e:
                logging.error(f"[BTSP write-pass] Failed: {e}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            logging.error(f"[BTSP incremental_train] Outer exception: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    @torch.no_grad()
    def _resolve_device(self) -> torch.device:
        """Find a reasonable device for moving tensors during evaluation."""
        inner = self._get_inner()
        # Try inner network parameters
        if inner is not None:
            net = getattr(inner, '_network', None)
            if net is not None and hasattr(net, 'parameters'):
                try:
                    param = next(net.parameters())
                    return param.device
                except (StopIteration, TypeError, AttributeError):
                    pass
            if hasattr(inner, 'parameters'):
                try:
                    param = next(inner.parameters())
                    return param.device
                except (StopIteration, TypeError, AttributeError):
                    pass
            dev_attr = getattr(inner, 'device', None)
            if isinstance(dev_attr, torch.device):
                return dev_attr
            if isinstance(dev_attr, str):
                return torch.device(dev_attr)

        proj = getattr(self, 'proj', None)
        if proj is not None and hasattr(proj, 'W'):
            return proj.W.device

        return torch.device('cpu')

    def _adaptive_alpha(self) -> float:
        """Dampen BTSP fusion weight when retrieval statistics become unstable."""
        alpha = self.alpha
        btsp = self._modules.get('btsp') if hasattr(self, '_modules') else None
        if btsp is None:
            btsp = getattr(self, 'btsp', None)
        if btsp is None or not hasattr(btsp, 'z_mu'):
            return alpha
        try:
            z_mu = btsp.z_mu
            if z_mu is None:
                return alpha
            z_mu_mean = float(z_mu.abs().mean().item())
        except Exception:
            return alpha
            
        # 🔥 修复：提高评估期α到0.7-0.8，避免BTSP过强盖住基线
        # 软钳制：α ∈ [α_min, α_max]，新默认 [0.7, 0.9] 
        alpha_min = getattr(self, '_alpha_min', 0.7)  # 从0.5提高到0.7
        alpha_max = getattr(self, '_alpha_max', 0.9)
        
        # 记录阈值与z_mu状态到日志（仅在eval时）
        if not self.training:
            # 使用标准logging而不是self._logger
            logging.debug(
                f"adaptive_alpha: z_mu_mean={z_mu_mean:.4f}, "
                f"threshold={self._alpha_dampen_threshold:.4f}, "
                f"bounds=[{alpha_min:.2f}, {alpha_max:.2f}]"
            )
        
        if z_mu_mean <= self._alpha_dampen_threshold:
            # 无需调节，但仍需应用软钳制
            return max(alpha_min, min(alpha_max, alpha))
            
        ratio = min(1.0, (z_mu_mean - self._alpha_dampen_threshold) / max(1.0, self._alpha_dampen_threshold))
        damp_scale = max(0.1, 1.0 - self._alpha_dampen_strength * ratio)
        btsp_weight = 1.0 - alpha
        adjusted_weight = btsp_weight * damp_scale
        adjusted_alpha = max(0.0, min(1.0, 1.0 - adjusted_weight))
        
        # 应用软钳制边界
        final_alpha = max(alpha_min, min(alpha_max, adjusted_alpha))
        
        if not self.training:
            # 使用标准logging而不是self._logger
            logging.debug(
                f"adaptive_alpha: raw={adjusted_alpha:.4f} -> clamped={final_alpha:.4f}"
            )
            
        return final_alpha

    def _fallback_inner_forward(self, inner, images, targets=None):
        """Fallback path when direct learner call fails; attempts network/backbone routes."""
        candidates = []
        net = getattr(inner, "_network", None)
        if net is not None:
            candidates.append(net)
            module = getattr(net, "module", None)
            if module is not None and module is not net:
                candidates.append(module)
        for module in candidates:
            call_fn = getattr(module, "forward", None)
            try_fns = []
            if callable(module):
                try_fns.append(module)
            if callable(call_fn) and call_fn is not module:
                try_fns.append(call_fn)
            for fn in try_fns:
                try:
                    # Prefer passing task_id/train if supported by the callee
                    import inspect
                    kwargs = {}
                    sig = inspect.signature(fn)
                    if 'task_id' in sig.parameters:
                        cur_task = getattr(inner, '_cur_task', getattr(self, 'current_task', -1))
                        kwargs['task_id'] = cur_task
                    if 'train' in sig.parameters:
                        kwargs['train'] = self.training
                    out = fn(images, **kwargs) if kwargs else fn(images)
                    if out is not None:
                        logging.debug("[BTSP] Fallback inner forward succeeded via network module")
                        return out
                except Exception:
                    continue
        if hasattr(inner, "classifier") and callable(getattr(inner, "classifier")):
            try:
                out = inner.classifier(images)
                logging.debug("[BTSP] Fallback inner forward used inner.classifier")
                return out
            except Exception:
                pass
        if hasattr(inner, "backbone") and callable(getattr(inner, "backbone")):
            try:
                features = inner.backbone(images)
                if isinstance(features, torch.Tensor):
                    head = getattr(inner, "classifier", None)
                    if callable(head):
                        out = head(features)
                        logging.debug("[BTSP] Fallback inner forward used backbone->classifier")
                        return out
            except Exception:
                pass
        logging.error("[BTSP] Fallback inner forward failed; returning None")
        return None

    def _log_task_completion_stats(self, task_id: int, k0: int, k1: int):
        """Record key statistics after task completion (integrity log)"""
        try:
            btsp = self.btsp
            
            # 1. Class range confirmation
            logging.info(f"[BTSP Integrity Log] task={task_id} class_range=[{k0}, {k1})")
            
            # 2. Per-class occupancy distribution (current task classes)
            occ_per_class = btsp.S.float().mean(dim=1)
            task_occ = occ_per_class[k0:k1] if k1 <= len(occ_per_class) else occ_per_class[k0:]
            q_mean = task_occ.mean().item()
            q_std = task_occ.std().item() if len(task_occ) > 1 else 0.0
            q_min, q_max = task_occ.min().item(), task_occ.max().item()
            
            logging.info(f"[BTSP Integrity Log] Current task class occupancy: mean={q_mean:.4f}, std={q_std:.4f}, range=[{q_min:.4f}, {q_max:.4f}]")
            
            # 3. Flip statistics
            flips_total = int(btsp.flip_counter.sum().item())
            logging.info(f"[BTSP Integrity Log] Total flips: {flips_total}")
            
            # 4. Gate status
            p_gate_mean = btsp.p_gate.mean().item()
            p_gate_std = btsp.p_gate.std().item()
            logging.info(f"[BTSP Integrity Log] Gate status: mean={p_gate_mean:.6f}, std={p_gate_std:.6f}")
            
        except Exception as e:
            logging.warning(f"[BTSP Integrity Log] Recording failed: {e}")

    # --------------------------------------------------
    # 辅助：尽最大努力抽取特征，适配各种 inc_net / backbone 结构
    # --------------------------------------------------
    @torch.no_grad()
    def _try_extract_any_feature(self, inner, images: torch.Tensor):
        """多策略特征抽取：
        顺序：
          1. inner._network.backbone(x)
          2. inner._network(x)[feat|features]
          3. inner.extract_vector(x) / inner._network.extract_vector(x)
          4. inner(x) / inner._network(x) 输出 tuple/dict 解析
        失败返回 None
        """
        net = getattr(inner, '_network', None)
        candidates = []
        if net is not None:
            net_mod = getattr(net, 'module', net)
            candidates.append(('backbone_call', getattr(net_mod, 'backbone', None)))
        # 1. backbone 直接调用
        for tag, fn in candidates:
            if callable(fn):
                try:
                    feat = fn(images)
                    if torch.is_tensor(feat):
                        return feat
                    if isinstance(feat, dict):
                        for k in ('features','feat','embedding','rep'):
                            if k in feat and torch.is_tensor(feat[k]):
                                return feat[k]
                except Exception:
                    pass
        # 2. net(images) -> dict/tuple
        if net is not None:
            try:
                out_tmp = net(images)
                if isinstance(out_tmp, dict):
                    for k in ('feat','features','embedding','rep','pre_logits'):  # 添加 pre_logits
                        if k in out_tmp and torch.is_tensor(out_tmp[k]):
                            return out_tmp[k]
                elif isinstance(out_tmp, tuple) and len(out_tmp) >= 2 and torch.is_tensor(out_tmp[1]):
                    return out_tmp[1]
            except Exception:
                pass
        # 3. extract_vector 接口
        for holder in (inner, net):
            if holder is None: continue
            if hasattr(holder, 'extract_vector'):
                try:
                    feat = holder.extract_vector(images)
                    if torch.is_tensor(feat):
                        return feat
                except Exception:
                    pass
        # 4. 直接 inner(images)
        try:
            out2 = inner(images)
            if isinstance(out2, dict):
                for k in ('feat','features','embedding','rep','pre_logits'):  # 添加 pre_logits
                    if k in out2 and torch.is_tensor(out2[k]):
                        return out2[k]
            elif isinstance(out2, tuple) and len(out2) >= 2 and torch.is_tensor(out2[1]):
                return out2[1]
        except Exception:
            pass
        return None

    def after_task(self):
        """Closed-loop epsilon control after each task."""
        if self.current_task <= 0:
            inner = self._get_inner()
            if hasattr(inner, 'after_task'):
                inner.after_task()
            return

        try:
            inner = self._get_inner()
            test_loader = self.fixed_val_loader or getattr(inner, 'test_loader', None)

            # --- Unified defaults at entry ---
            btsp = self._modules['btsp']
            T_eff_val = float(btsp.T_eff.item())
            # Assert consistency with analytical T_eff
            teff_mismatch = False
            try:
                _calc_teff = T_eff(float(btsp.tau_e_steps.item()), float(btsp.theta.item())) if hasattr(btsp, 'tau_e_steps') else T_eff_val
                if abs(_calc_teff - T_eff_val) > 1e-3:
                    teff_mismatch = True
                    logging.warning(f"[BTSP] T_eff mismatch: memory={T_eff_val:.6f} calc={_calc_teff:.6f}")
            except Exception:
                pass
            T_eff_next = T_eff_val
            p_gate_current = float(btsp.p_gate.mean().item())
            p_gate_next = p_gate_current
            achievable_next = True
            mode_next = self._last_mode if self._last_mode else 'achievable'

            # If no loader, generate fallback plan and exit
            if test_loader is None:
                logging.warning("[BTSP Epsilon Control] No test_loader available, generating fallback plan")
                p_gate_next, _ = self._clamp_p_gate(p_gate_current, p_gate_current, 'rate', 0.0)
                next_task_id = self.current_task + 1
                if next_task_id < self.num_tasks:
                    self._next_gate_plan = {
                        'task': next_task_id,
                        'mode': 'rate',
                        'p_gate': p_gate_next,
                        'target_eps': self.target_epsilon if self.target_epsilon is not None else 0.05,
                        'delta_q_target': self._delta_q_default,
                        'notes': ['fallback_noeval'],
                        'T_eff': float(T_eff_next),
                        'achievable': False,
                    }
                    logging.info(f"[BTSP Fallback] Generated fallback plan for task {next_task_id}: p_gate={p_gate_next:.6f}")
                if hasattr(inner, 'after_task'):
                    inner.after_task()
                return

            # --- Unified evaluation intake ---
            fed = getattr(self, '_fed_acc', None)
            if fed is not None and int(fed.get('task', -1)) == int(self.current_task):
                acc_old = float(fed.get('acc_old', 0.0))
                acc_total = float(fed.get('acc_total', 0.0))
                acc_new = float(fed.get('acc_new', 0.0))
                # consume once per task
                self._fed_acc = None
            else:
                acc_old, acc_total, acc_new = self._evaluate_old_class_accuracy(test_loader)

            eps_old_actual = 1.0 - (acc_old / 100.0)

            # EMA smoothing
            if self._eps_actual_ema is None:
                self._eps_actual_ema = eps_old_actual
            else:
                self._eps_actual_ema = (self._eps_ema_gamma * self._eps_actual_ema + 
                                       (1 - self._eps_ema_gamma) * eps_old_actual)

            # Diagnostics and predictions
            global_occ = float(btsp.S.float().mean().item()) if hasattr(btsp, 'S') else 0.0
            bytes_btsp = float(btsp.S.numel()) / 8.0 if hasattr(btsp, 'S') else 0.0
            M = self.current_task + 1
            pred_eps = epsilon_from_p_gate(p_gate_current, M, self.p_pre, T_eff_val)
            self._update_eps_calibration(p_gate_current, T_eff_val, global_occ, M, eps_old_actual)

            control_error = self._eps_actual_ema - self.target_epsilon
            coverage = 1.0 if abs(control_error) <= self.epsilon_tolerance else 0.0
            need_adjustment = abs(control_error) > self._eps_deadzone

            apb = 0.0
            if bytes_btsp > 0.0:
                apb = (acc_total / 100.0) / (bytes_btsp / 1e6)

            next_flags: list[str] = []
            allow_achievable = self._achievable_enabled and global_occ < self._occ_upper_band

            # ACC_MATCH check when fed metrics available
            acc_old_trainer = None
            if fed is not None and int(fed.get('task', -1)) == int(self.current_task):
                acc_old_trainer = float(fed.get('acc_old', 0.0))
                acc_match = abs(acc_old - acc_old_trainer)
                if acc_match > 0.5:
                    self._acc_mismatch_count = getattr(self, '_acc_mismatch_count', 0) + 1
                    if self._acc_mismatch_count >= 2:
                        logging.warning(f"[BTSP Epsilon Control] ACC_MATCH={acc_match:.2f}% > 0.5% x2, freeze adjustment this round")
                        need_adjustment = False
                        next_flags.append('acc_mismatch_freeze')
                    else:
                        logging.warning(f"[BTSP Epsilon Control] ACC_MATCH={acc_match:.2f}% > 0.5%, skip adjustment this round")
                        need_adjustment = False
                        next_flags.append('acc_mismatch')
                else:
                    self._acc_mismatch_count = 0
                    next_flags.append(f"ACC_MATCH={acc_match:.2f}%")
            else:
                next_flags.append("ACC_MATCH=NA")

            # If T_eff mismatch, freeze this round
            if teff_mismatch:
                need_adjustment = False
                next_flags.append('teff_mismatch')

            # Adjustment policy — propose only, do not execute here
            if need_adjustment and (self.current_task + 1) < self.num_tasks:
                # Only adjust T_eff if p_gate target is unreachable
                if allow_achievable:
                    eps_target_next = max(0.001, min(0.5, self.target_epsilon))
                    gate_target = p_gate_from_epsilon(eps_target_next, M + 1, self.p_pre, T_eff_val)
                    gate_cap_window = p_gate_window_cap(T_eff_val, tau=getattr(self, '_window_cap_tau', 0.99))
                    if gate_target > gate_cap_window:
                        # Increase T_eff by at most +0.5 per task within [T_min, T_max]
                        T_min, T_max = 1.0, 64.0
                        T_eff_next = max(T_min, min(T_eff_val + 0.5, T_max))
                        next_flags.append('T_eff_increased')
                        try:
                            theta_val = float(btsp.theta.item())
                            tau_e_new = T_eff_next / math.log(1.0 / max(theta_val, 1e-6))
                            btsp.tau_e_steps.fill_(tau_e_new)
                            btsp.recompute_teff()
                            logging.info(f"[BTSP T_eff Adjust] Increased T_eff: {T_eff_val:.2f} -> {T_eff_next:.2f}")
                        except Exception as e:
                            logging.warning(f"[BTSP T_eff Adjust] Failed: {e}")
                            T_eff_next = T_eff_val
                    else:
                        # p_gate reachable: use proposed next
                        p_gate_next = gate_target
                        achievable_next = True
                        mode_next = 'achievable'
                        next_flags.append('achievable_plan')
                else:
                    next_flags.append('achievable_disabled')
                    # hold p_gate
            else:
                # deadzone or coverage ok: hold
                p_gate_next = p_gate_current
                T_eff_next = T_eff_val

            # Engineering clamp and hysteresis for p_gate
            # Step clamp: 0.67x .. 1.5x of current
            lower = p_gate_current * 0.67
            upper = p_gate_current * 1.5
            # Early-stage gain: allow stronger approach within step bounds
            if achievable_next and (len(self._epsilon_history) <= 2):
                approach_gain = 2.0
            elif achievable_next:
                approach_gain = 1.25
            else:
                approach_gain = 1.0
            p_gate_next = p_gate_current + approach_gain * (p_gate_next - p_gate_current)
            p_gate_next = max(lower, min(p_gate_next, upper))
            # Window cap after potential T_eff change (use unified 99% cap)
            cap_after = p_gate_window_cap(T_eff_next, tau=getattr(self, '_window_cap_tau', 0.99))
            p_gate_next = min(p_gate_next, 0.95 * cap_after)
            # Final clamp with occupancy-aware rules
            p_gate_next, clamp_flags = self._clamp_p_gate(p_gate_next, p_gate_current, mode_next, global_occ)
            next_flags.extend(clamp_flags)

            # Record history (task-level window)
            self._epsilon_history.append({
                'task': self.current_task,
                'eps_target': self.target_epsilon,
                'eps_actual': eps_old_actual,
                'eps_actual_ema': self._eps_actual_ema,
                'eps_predicted': pred_eps,
                'control_error': control_error,
                'coverage': coverage,
                'p_gate': p_gate_current,
                'acc_old': acc_old,
                'acc_total': acc_total,
                'acc_new': acc_new,
                'global_occ': global_occ,
            })

            # Coverage/RMSE window metrics
            if len(self._epsilon_history) >= 1:
                window_size = min(5, len(self._epsilon_history))
                recent_history = self._epsilon_history[-window_size:]
                errors = [h['control_error'] for h in recent_history]
                coverages = [h['coverage'] for h in recent_history]
                rmse = np.sqrt(np.mean([e**2 for e in errors]))
                coverage_rate = np.mean(coverages)
                if len(self._epsilon_history) > 10:
                    self._epsilon_history = self._epsilon_history[-10:]
            else:
                rmse = abs(control_error)
                coverage_rate = coverage

            # Unified log — proposed only (no execution here)
            unified_data = {
                'task': self.current_task,
                'eps_target': self.target_epsilon,
                'eps_actual': eps_old_actual,
                'eps_predicted': pred_eps,
                'control_error': control_error,
                'coverage': coverage,
                'coverage_rate': coverage_rate,
                'rmse': rmse,
                'acc_old': acc_old,
                'acc_total': acc_total,
                'acc_new': acc_new,
                'p_gate_current': p_gate_current,
                'p_gate_next': p_gate_next,
                'T_eff_current': T_eff_val,
                'T_eff_next': T_eff_next,
                'global_occ': global_occ,
                'bytes_btsp': bytes_btsp,
                'apb': apb,
                'need_adjustment': bool(need_adjustment),
                'mode_next': mode_next,
                'achievable_next': bool(achievable_next),
                'flags': next_flags,
                # alpha 记录（对齐语义：fused = (1-α)*base + α*btsp）
                'alpha_is_btsp_weight': True,
                'alpha_eval': float(self.alpha if self.training else self._adaptive_alpha()),
                'alpha_mode': 'static' if self.training else 'adaptive',
            }
            if acc_old_trainer is not None:
                unified_data['ACC_MATCH'] = abs(acc_old - acc_old_trainer)
            self._log_unified_control("BTSP Epsilon Control", unified_data)

            # Persist plan for next task (proposed), do not execute now
            next_task_id = self.current_task + 1
            if next_task_id < self.num_tasks:
                self._next_gate_plan = {
                    'task': next_task_id,
                    'mode': mode_next,
                    'p_gate': float(p_gate_next),
                    'target_eps': float(self.target_epsilon),
                    'delta_q_target': float(self._last_delta_q_target if hasattr(self, '_last_delta_q_target') else self._delta_q_default),
                    'notes': list(next_flags),
                    'T_eff': float(T_eff_next),
                    'achievable': bool(achievable_next),
                }

            # Call inner after_task regardless
            if hasattr(inner, 'after_task'):
                inner.after_task()

            # 统一评估入口：获取分组准确率，含每task段
            # 使用 utils.toolkit.accuracy 提供的分段键，如 "00-09","10-19" 等
            # 我们需要：
            #  - A_old_task[k]: 每个队列k（0..t-1）的段精度
            #  - A_base_task[k]: 首次学完该段时的精度（持久化）
            #  - eps_actual_task/eps_target_task/violation_task
            from utils.toolkit import accuracy
            # 先拿 fused 的预测与真值
            device = self._resolve_device()
            y_pred_list, y_true_list = [], []
            with torch.no_grad():
                for batch_data in test_loader:
                    if len(batch_data) == 3:
                        _, inputs, targets = batch_data
                    else:
                        inputs, targets = batch_data
                    inputs = inputs.to(device, non_blocking=True)
                    outputs = self.forward(inputs, targets)
                    logits = outputs['logits'] if isinstance(outputs, dict) and 'logits' in outputs else outputs
                    preds = torch.argmax(logits, dim=1)
                    y_pred_list.append(preds.cpu().numpy())
                    y_true_list.append(targets.cpu().numpy())
            import numpy as _np
            y_pred = _np.concatenate(y_pred_list) if y_pred_list else _np.zeros((0,), dtype=_np.int64)
            y_true = _np.concatenate(y_true_list) if y_true_list else _np.zeros((0,), dtype=_np.int64)
            args_inner = getattr(inner, 'args', None)
            init_cls = getattr(args_inner, 'init_cls', getattr(inner, 'init_cls', 10))
            increment = getattr(args_inner, 'increment', getattr(inner, 'increment', 10))
            known_classes = getattr(inner, '_known_classes', init_cls)
            all_acc = accuracy(y_pred, y_true, known_classes, init_cls, increment)
            # 解析 task 段键
            def _label_for_task(k: int) -> str:
                a = k * increment
                b = a + increment - 1
                return f"{str(a).rjust(2,'0')}-{str(b).rjust(2,'0')}"
            t_cur = int(self.current_task)
            # 更新首训完成时的基线（本任务段）
            if t_cur not in self._task_baseline_acc:
                lbl_cur = _label_for_task(t_cur)
                base_acc_cur = float(all_acc.get(lbl_cur, 0.0))
                self._task_baseline_acc[t_cur] = base_acc_cur
            # 计算各旧段指标（k <= t-1）
            eps_actual_task = {}
            eps_target_task = {}
            violation_task = {}
            acc_old_task = {}
            A_base_task = {}
            for k in range(0, max(0, t_cur)):
                lbl = _label_for_task(k)
                A_k = float(all_acc.get(lbl, 0.0))
                acc_old_task[k] = A_k
                A_base_k = self._task_baseline_acc.get(k, A_k)
                A_base_task[k] = A_base_k
                eps_act = 1.0 - A_k / 100.0
                eps_tar = max(0.0, 1.0 - (A_base_k - self._control_delta_pp) / 100.0)
                eps_actual_task[k] = eps_act
                eps_target_task[k] = eps_tar
                violation_task[k] = max(0.0, eps_act - eps_tar)
            # 选择目标队列 k*
            k_star = None
            if violation_task:
                if self._control_selector == 'round_robin':
                    # 轮转：从上次的k开始
                    last_k = getattr(self, '_last_task_target', -1)
                    for offset in range(1, len(violation_task) + 1):
                        cand = (last_k + offset) % len(violation_task)
                        if violation_task.get(cand, 0.0) > 0.0:
                            k_star = cand
                            break
                else:
                    # 默认：最大违约
                    k_star = max(violation_task.keys(), key=lambda k: violation_task[k])
                if k_star is None and len(violation_task) > 0:
                    k_star = max(violation_task.keys())
            self._last_task_target = k_star if k_star is not None else getattr(self, '_last_task_target', None)
            # 计算控制误差与提议的 p_gate
            eps_old_actual = 1.0 - (float(all_acc.get('old', 0.0)) / 100.0)
            T_eff_val = float(self.btsp.T_eff.item())
            M = self.current_task + 1
            p_gate_current = float(self.btsp.p_gate.mean().item())
            pred_eps = epsilon_from_p_gate(p_gate_current, M, self.p_pre, T_eff_val)
            # 默认 next 保持原策略
            p_gate_next = p_gate_current
            mode_next = self._last_mode if self._last_mode else 'achievable'
            achievable_next = True
            flags = []
            T_eff_next = T_eff_val
            if self._control_mode == 'task_sla' and k_star is not None and violation_task.get(k_star, 0.0) > 0.0:
                eps_act_k = eps_actual_task[k_star]
                eps_tar_k = eps_target_task[k_star]
                # 以该队列的目标误差构造反演
                gate_target = p_gate_from_epsilon(eps_tar_k, M + 1, self.p_pre, T_eff_val)
                gate_cap_window = p_gate_window_cap(T_eff_val, tau=getattr(self, '_window_cap_tau', 0.99))
                # 可达性：若超cap，则先动T_eff
                if gate_target > gate_cap_window:
                    T_eff_next = max(self._teff_min, min(T_eff_val + self._teff_step_on_cap, self._teff_max))
                    flags.append('T_eff_increased')
                else:
                    p_gate_next = gate_target
                    mode_next = 'achievable'
                    achievable_next = True
                    flags.append('achievable_plan')
            else:
                # 退化为 global 行为
                control_error = (1.0 - (all_acc.get('old', 0.0) / 100.0)) - (self.target_epsilon if self.target_epsilon is not None else 0.05)
                need_adjustment = abs(control_error) > self._eps_deadzone
                if need_adjustment:
                    eps_target_next = max(0.001, min(0.5, self.target_epsilon))
                    gate_target = p_gate_from_epsilon(eps_target_next, M + 1, self.p_pre, T_eff_val)
                    gate_cap_window = p_gate_window_cap(T_eff_val, tau=getattr(self, '_window_cap_tau', 0.99))
                    if gate_target > gate_cap_window:
                        T_eff_next = max(self._teff_min, min(T_eff_val + self._teff_step_on_cap, self._teff_max))
                        flags.append('T_eff_increased')
                    else:
                        p_gate_next = gate_target
                        achievable_next = True
                        mode_next = 'achievable'
                        flags.append('achievable_plan')
            # 任务占用统计（目标队列）
            occ_per_class = self.btsp.S.float().mean(dim=1) if hasattr(self.btsp, 'S') else None
            occ_task = None
            classes_target = None
            if isinstance(occ_per_class, torch.Tensor) and k_star is not None:
                a = int(k_star * increment)
                b = int(a + increment)
                classes_target = (a, b)
                occ_task = float(occ_per_class[a:b].mean().item()) if b <= occ_per_class.numel() else float(occ_per_class[a:].mean().item())
                self._task_occ[k_star] = occ_task
            # clamp & achievable check（加入 task_occ）
            global_occ = float(self.btsp.S.float().mean().item()) if hasattr(self.btsp, 'S') else 0.0
            p_gate_next, clamp_flags = self._clamp_p_gate(p_gate_next, p_gate_current, mode_next, global_occ)
            flags.extend(clamp_flags)
            try:
                flips_total = int(self.btsp.flip_counter.sum().item()) if hasattr(self.btsp, 'flip_counter') else 0
                flips_per_task = flips_total / max(1, self.current_task)
                stats = {'global_occ': global_occ, 'flips_per_task': flips_per_task}
                if occ_task is not None:
                    stats['task_occ'] = occ_task
                budgets = {'occ_high': self._occ_global_high, 'occ_low': self._occ_lower_band, 'task_high': self._occ_task_high}
                ok, reason = self._achievable_check(p_gate_next, T_eff_next, stats, budgets)
                if not ok:
                    flags.append(reason)
                    p_gate_next = p_gate_current
                    mode_next = self._last_mode
                    achievable_next = False
            except Exception:
                flags.append('achievable_check_error')
            # 记录并持久化计划（带 task_target, classes）
            bytes_btsp = float(self.btsp.S.numel()) / 8.0 if hasattr(self.btsp, 'S') else 0.0
            apb = (float(all_acc.get('total', 0.0)) / 100.0) / (bytes_btsp / 1e6) if bytes_btsp > 0 else 0.0
            unified_data = {
                'task': self.current_task,
                'slice_mode': 'task' if self._control_mode == 'task_sla' else 'global',
                'task_target': k_star,
                'classes': classes_target if classes_target is not None else None,
                'A_base_task': A_base_task.get(k_star, None) if k_star is not None else None,
                'A_old_task': acc_old_task.get(k_star, None) if k_star is not None else None,
                'eps_target_task': eps_target_task.get(k_star, None) if k_star is not None else None,
                'eps_actual_task': eps_actual_task.get(k_star, None) if k_star is not None else None,
                'violation_task': violation_task.get(k_star, None) if k_star is not None else None,
                'eps_target': self.target_epsilon if self._control_mode != 'task_sla' else (eps_target_task.get(k_star, None) if k_star is not None else None),
                'eps_actual': 1.0 - (float(all_acc.get('old', 0.0)) / 100.0),
                'eps_predicted': pred_eps,
                'control_error': None,
                'coverage': 1.0 if (k_star is not None and violation_task.get(k_star, 0.0) <= 0.0) else 0.0,
                'coverage_rate': 0.0,
                'rmse': 0.0,
                'acc_old': float(all_acc.get('old', 0.0)),
                'acc_total': float(all_acc.get('total', 0.0)),
                'acc_new': float(all_acc.get('new', 0.0)),
                'p_gate_current': p_gate_current,
                'p_gate_next': p_gate_next,
                'T_eff_current': T_eff_val,
                'T_eff_next': T_eff_next,
                'occ_task': occ_task if occ_task is not None else None,
                'global_occ': global_occ,
                'bytes_btsp': bytes_btsp,
                'apb': apb,
                'need_adjustment': True if (self._control_mode == 'task_sla' and k_star is not None and violation_task.get(k_star, 0.0) > 0.0) else False,
                'mode_next': mode_next,
                'achievable_next': bool(achievable_next),
                'flags': flags,
                'alpha_is_btsp_weight': True,
                'alpha_eval': float(self.alpha if self.training else self._adaptive_alpha()),
                'alpha_mode': 'static' if self.training else 'adaptive',
            }
            self._log_unified_control("BTSP Epsilon Control", unified_data)
            next_task_id = self.current_task + 1
            if next_task_id < self.num_tasks:
                self._next_gate_plan = {
                    'task': next_task_id,
                    'mode': mode_next,
                    'p_gate': float(p_gate_next),
                    'target_eps': float(self.target_epsilon if self._control_mode != 'task_sla' else eps_target_task.get(k_star, self.target_epsilon)),
                    'delta_q_target': float(self._last_delta_q_target if hasattr(self, '_last_delta_q_target') else self._delta_q_default),
                    'notes': list(flags),
                    'T_eff': float(T_eff_next),
                    'achievable': bool(achievable_next),
                    'task_target': int(k_star) if k_star is not None else None,
                    'classes': [int(classes_target[0]), int(classes_target[1])] if classes_target is not None else None,
                }
            # 调用内层 after_task
            if hasattr(inner, 'after_task'):
                inner.after_task()
            return
        except Exception as e:
            # Robust fallback: disable adjustment, still record epsilon history and call inner.after_task
            try:
                # ensure history is updated with minimal info
                btsp = self._modules['btsp']
                p_gate_current = float(btsp.p_gate.mean().item())
                global_occ = float(btsp.S.float().mean().item()) if hasattr(btsp, 'S') else 0.0
                self._epsilon_history.append({
                    'task': self.current_task,
                    'eps_target': self.target_epsilon,
                    'eps_actual': None,
                    'eps_actual_ema': None,
                    'eps_predicted': None,
                    'control_error': None,
                    'coverage': None,
                    'p_gate': p_gate_current,
                    'acc_old': None,
                    'acc_total': None,
                    'acc_new': None,
                    'global_occ': global_occ,
                })
            except Exception:
                pass
            logging.warning(f"[BTSP Epsilon Control] after_task failed: {e}")
            if hasattr(inner, 'after_task'):
                try:
                    inner.after_task()
                except Exception:
                    pass
            return

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """扩展state_dict以包含计划持久化"""
        state = super().state_dict(destination, prefix, keep_vars)
        
        # 添加持久化的计划状态
        if self._next_gate_plan is not None:
            state[prefix + 'btsp_next_plan'] = self._next_gate_plan
        
        # 添加其他重要状态
        if self._eps_actual_ema is not None:
            state[prefix + 'btsp_eps_actual_ema'] = self._eps_actual_ema
            
        if hasattr(self, '_eps_calib_gain'):
            state[prefix + 'btsp_eps_calib_gain'] = self._eps_calib_gain
            
        return state
        
    def load_state_dict(self, state_dict, strict=True):
        """扩展load_state_dict以恢复计划持久化"""
        # 提取BTSP特有状态
        btsp_keys = ['btsp_next_plan', 'btsp_eps_actual_ema', 'btsp_eps_calib_gain']
        btsp_state = {}
        
        for key in list(state_dict.keys()):
            for btsp_key in btsp_keys:
                if key.endswith(btsp_key):
                    btsp_state[btsp_key] = state_dict.pop(key)
                    break
        
        # 调用父类的load_state_dict
        result = super().load_state_dict(state_dict, strict)
        
        # 恢复BTSP状态
        if 'btsp_next_plan' in btsp_state:
            self._next_gate_plan = btsp_state['btsp_next_plan']
            self._plan_restored_from_checkpoint = True  # 设置恢复标记
            logging.info(f"[BTSP] Restored gate plan from checkpoint: {self._next_gate_plan}")
            
        if 'btsp_eps_actual_ema' in btsp_state:
            self._eps_actual_ema = btsp_state['btsp_eps_actual_ema']
            
        if 'btsp_eps_calib_gain' in btsp_state:
            self._eps_calib_gain = btsp_state['btsp_eps_calib_gain']
            
        return result

    def _evaluate_old_class_accuracy(self, test_loader):
        """
        在固定验证集上评估旧类准确率（形态A标准协议）
        
        Returns:
            tuple: (acc_old, acc_total, acc_new) 百分比格式
        """
        inner = self._get_inner()
        if not hasattr(inner, '_known_classes') or not hasattr(inner, '_total_classes'):
            # Fallback: 使用简化评估
            return 50.0, 50.0, 50.0
        
        known_classes = inner._known_classes  # 旧类数量
        total_classes = inner._total_classes   # 总类数
        
        # 如果没有旧类，返回0
        if known_classes <= 0:
            return 0.0, 0.0, 0.0
            
        # 使用fused输出进行评估（系统最终输出）
        # 🔥 修复：只设置BTSP组件为eval模式，避免调用inner.train()
        was_training = self.training
        self.training = False  # 手动设置为eval模式
        if hasattr(self.proj, 'eval'):
            self.proj.eval()
        # btsp在CPU上，不需要设置eval模式
        
        y_pred_list = []
        y_true_list = []
        
        try:
            device = self._resolve_device()
            with torch.no_grad():
                for batch_data in test_loader:
                    if len(batch_data) == 3:
                        _, inputs, targets = batch_data
                    else:
                        inputs, targets = batch_data
                    inputs = inputs.to(device, non_blocking=True)
                    
                    # 获取系统最终输出（fused logits）
                    outputs = self.forward(inputs, targets)
                    if isinstance(outputs, dict) and 'logits' in outputs:
                        logits = outputs['logits']
                    else:
                        logits = outputs
                        
                    preds = torch.argmax(logits, dim=1)
                    y_pred_list.append(preds.cpu().numpy())
                    y_true_list.append(targets.cpu().numpy())
        finally:
            # 恢复原始训练状态
            self.training = was_training
            if hasattr(self.proj, 'train'):
                self.proj.train(was_training)
        
        y_pred = np.concatenate(y_pred_list)
        y_true = np.concatenate(y_true_list)
        
        # 使用toolkit.py的accuracy函数计算分组准确率
        from utils.toolkit import accuracy
        args = getattr(inner, 'args', None)
        init_cls = getattr(args, 'init_cls', None) if args else None
        if init_cls is None:
            init_cls = getattr(inner, 'init_cls', 10)
        increment = getattr(args, 'increment', None) if args else None
        if increment is None:
            increment = getattr(inner, 'increment', 10)
        all_acc = accuracy(y_pred, y_true, known_classes, init_cls, increment)
        
        acc_old = all_acc.get('old', 0.0)
        acc_total = all_acc.get('total', 0.0) 
        acc_new = all_acc.get('new', 0.0)
        
        return acc_old, acc_total, acc_new

    # --------------------------------------------------
    # 评测口径对齐：接受外部评测指标（trainer提供）
    # --------------------------------------------------
    def feed_eval_metrics(self, task_id: int, acc_old: float, acc_new: float, acc_total: float):
        try:
            self._fed_acc = {
                'task': int(task_id),
                'acc_old': float(acc_old),
                'acc_new': float(acc_new),
                'acc_total': float(acc_total),
            }
        except Exception:
            self._fed_acc = None
    
    @property
    def btsp_control(self):
        return self

    # --------------------------------------------------
    # 辅助：BTSP统计快照，为trainer.py提供关键指标
    # --------------------------------------------------
    @torch.no_grad()
    def stats_snapshot(self):
        """
        为trainer.py提供BTSP统计快照
        
        Returns:
            dict: 包含关键统计指标
        """
        try:
            btsp = self._modules['btsp']
            
            # 基础统计
            p_gate_mean = float(btsp.p_gate.mean().item()) if hasattr(btsp, 'p_gate') else 0.0
            T_eff_val = float(btsp.T_eff.item()) if hasattr(btsp, 'T_eff') else 1.0
            occ_mean = float(btsp.S.float().mean().item()) if hasattr(btsp, 'S') else 0.0
            flips_total = int(btsp.flip_counter.sum().item()) if hasattr(btsp, 'flip_counter') else 0
            
            return {
                'p_gate_mean': p_gate_mean,
                'T_eff': T_eff_val,
                'occ_mean': occ_mean,
                'flips_total': flips_total,
            }
        except Exception as e:
            logging.warning(f"[BTSP] stats_snapshot failed: {e}")
            return {}

    def _train(self, *args, **kwargs):
        inner = self._get_inner()
        if inner is not None and hasattr(inner, '_train'):
            return inner._train(*args, **kwargs)
        raise AttributeError("Inner learner does not have _train method")

    def _infer_parameters(self, num_classes, feat_dim, num_tasks, data_manager, args, inner):
        """
        🔥 智能推断：从data_manager、args、inner中自动获取关键参数
        
        优先级：
        1. 显式传入的参数（最高优先级）
        2. data_manager提供的信息
        3. args配置文件中的信息
        4. inner模型的属性推断
        5. 合理的默认值（最低优先级）
        
        Returns:
            tuple: (num_classes, feat_dim, num_tasks)
        """
        logging.info("[BTSP] Starting intelligent parameter inference...")
        arg_map: dict[str, object] = {}
        if args is not None:
            if isinstance(args, dict):
                arg_map = args
            else:
                try:
                    arg_map = vars(args)
                except TypeError:
                    arg_map = {}

        
        # === 1. 推断总类别数 num_classes ===
        if num_classes is None:
            if data_manager is not None:
                # 从data_manager获取总类别数
                if hasattr(data_manager, 'nb_classes'):
                    num_classes = data_manager.nb_classes
                    logging.info(f"[BTSP] Inferred num_classes={num_classes} from data_manager.nb_classes")
                elif hasattr(data_manager, '_class_order'):
                    num_classes = len(data_manager._class_order)
                    logging.info(f"[BTSP] Inferred num_classes={num_classes} from data_manager._class_order")
            
            if num_classes is None and args is not None:
                # 从args推断：init_cls + increment * (nb_tasks - 1)
                init_cls = _arg_get(args, 'init_cls', 10)
                increment = _arg_get(args, 'increment', 10) 
                if 'nb_tasks' in arg_map:
                    nb_tasks = arg_map['nb_tasks']
                    num_classes = init_cls + increment * (nb_tasks - 1)
                    logging.info(f"[BTSP] Inferred num_classes={num_classes} from args (init_cls={init_cls}, increment={increment}, nb_tasks={nb_tasks})")
                elif data_manager is not None and hasattr(data_manager, 'nb_tasks'):
                    nb_tasks = data_manager.nb_tasks
                    num_classes = init_cls + increment * (nb_tasks - 1) 
                    logging.info(f"[BTSP] Inferred num_classes={num_classes} from args+data_manager (init_cls={init_cls}, increment={increment}, nb_tasks={nb_tasks})")
            
            if num_classes is None:
                # 最后的fallback：常见数据集默认值
                dataset_name = arg_map.get('dataset', 'unknown') if arg_map else 'unknown'
                dataset_defaults = {
                    'cifar100': 100, 'imagenet100': 100, 'imagenet1000': 1000,
                    'cifar10': 10, 'cub': 200, 'objectnet': 113
                }
                num_classes = dataset_defaults.get(dataset_name.lower(), 100)
                logging.warning(f"[BTSP] Using fallback num_classes={num_classes} for dataset '{dataset_name}'")
        
        # === 2. 推断特征维度 feat_dim ===
        if feat_dim is None:
            # 尝试从inner模型推断
            if hasattr(inner, 'feature_dim'):
                feat_dim = inner.feature_dim
                logging.info(f"[BTSP] Inferred feat_dim={feat_dim} from inner.feature_dim")
            elif hasattr(inner, '_network') and hasattr(inner._network, 'feature_dim'):
                feat_dim = inner._network.feature_dim
                logging.info(f"[BTSP] Inferred feat_dim={feat_dim} from inner._network.feature_dim")
            elif hasattr(inner, '_network') and hasattr(inner._network, 'out_dim'):
                feat_dim = inner._network.out_dim
                logging.info(f"[BTSP] Inferred feat_dim={feat_dim} from inner._network.out_dim")
            
            if feat_dim is None and args is not None:
                # 从args推断特征维度
                if 'feat_dim' in arg_map:
                    feat_dim = arg_map['feat_dim']
                    logging.info(f"[BTSP] Inferred feat_dim={feat_dim} from arg_map['feat_dim']")
                elif 'convnet_type' in arg_map:
                    # 根据backbone类型推断
                    backbone_defaults = {
                        'resnet18': 512, 'resnet32': 64, 'resnet50': 2048,
                        'vit_base_patch16_224': 768, 'vit_large_patch16_224': 1024,
                        'clip_vit_b16': 512, 'deit_base_patch16_224': 768
                    }
                    convnet_type = arg_map['convnet_type'].lower()
                    feat_dim = backbone_defaults.get(convnet_type, 512)
                    logging.info(f"[BTSP] Inferred feat_dim={feat_dim} from backbone type '{convnet_type}'")
            
            if feat_dim is None:
                # 最后的fallback
                feat_dim = 512  # 常见的特征维度
                logging.warning(f"[BTSP] Using fallback feat_dim={feat_dim}")
        
        # === 3. 推断任务数 num_tasks ===
        if num_tasks is None:
            if data_manager is not None and hasattr(data_manager, 'nb_tasks'):
                num_tasks = data_manager.nb_tasks
                logging.info(f"[BTSP] Inferred num_tasks={num_tasks} from data_manager.nb_tasks")
            elif args is not None and 'nb_tasks' in arg_map:
                num_tasks = arg_map['nb_tasks']
                logging.info(f"[BTSP] Inferred num_tasks={num_tasks} from arg_map['nb_tasks']")
            elif args is not None:
                # 从init_cls和increment推算
                init_cls = _arg_get(args, 'init_cls', 10)
                increment = _arg_get(args, 'increment', 10)
                if num_classes is not None:
                    num_tasks = 1 + max(0, (num_classes - init_cls) // increment)
                    logging.info(f"[BTSP] Calculated num_tasks={num_tasks} from num_classes={num_classes}, init_cls={init_cls}, increment={increment}")
            
            if num_tasks is None:
                num_tasks = 10  # 常见的任务数
                logging.warning(f"[BTSP] Using fallback num_tasks={num_tasks}")
        
        # === 4. 最终验证和总结 ===
        assert num_classes > 0, f"Invalid num_classes: {num_classes}"
        assert feat_dim > 0, f"Invalid feat_dim: {feat_dim}"
        assert num_tasks > 0, f"Invalid num_tasks: {num_tasks}"
        
        logging.info(f"[BTSP] Parameter inference complete: num_classes={num_classes}, feat_dim={feat_dim}, num_tasks={num_tasks}")
        
        return num_classes, feat_dim, num_tasks
