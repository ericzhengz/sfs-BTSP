# btsp/gate_policy.py
from __future__ import annotations
import math, torch
from typing import Literal

def T_eff(tau_e_steps: float, theta: float) -> float:
    """
    统一的有效时间窗口计算：T_eff = τ_e * ln(1/θ)
    
    Args:
        theta: 遗忘因子 ∈ (0, 1)
        tau_e_steps: 资格迹时间常数（步数）
        
    Returns:
        T_eff: 有效时间窗口（步数）
    """
    theta = max(float(theta), 1e-6)  # 避免log(0)
    tau_e_steps = max(float(tau_e_steps), 1e-6)
    return tau_e_steps * math.log(1.0 / theta)

# For backward compatibility
def T_eff_from_params(theta: float, tau_e_steps: float) -> float:
    """Backward compatibility wrapper for T_eff"""
    return T_eff(tau_e_steps, theta)

def p_gate_epsilon(eps0: float, M: int, p_pre: float, T: float) -> float:
    """
    ε-control反演：目标错误率 → 门控率
    
    Args:
        eps0: 目标错误率 ∈ [0, 0.5]
        M: 任务数量
        p_pre: 预激活概率
        T: 有效时间窗口 T_eff
        
    Returns:
        p_gate: 门控率
    """
    # 数值安全化
    eps0 = min(max(float(eps0), 0.0), 0.5)
    M = max(int(M), 1)
    p_pre = max(float(p_pre), 1e-8)
    T = max(float(T), 1e-8)
    
    # 错误率→翻转率的反演公式
    p_flip_star = 0.5 * (1.0 - (1.0 - 2.0*eps0)**(1.0/M))
    
    # 翻转率 → 有效激活率
    p_eff_star = 2.0 * p_flip_star / p_pre
    
    # 有效激活率 → 门控率
    p_gate = -math.log(max(1.0 - p_eff_star, 1e-6)) / T
    
    return float(p_gate)

def p_gate_rate(delta_q_target: float, p_pre: float, q_c: float, T: float) -> float:
    """
    资源速率控制：目标增量占用率 → 门控率
    
    Args:
        delta_q_target: 目标增量占用率
        p_pre: 预激活概率  
        q_c: 当前占用率
        T: 有效时间窗口 T_eff
        
    Returns:
        p_gate: 门控率
    """
    delta_q_target = max(float(delta_q_target), 1e-8)
    p_pre = max(float(p_pre), 1e-8)
    q_c = max(min(float(q_c), 0.999), 0.0)  # 限制在[0, 0.999]
    T = max(float(T), 1e-8)
    
    # 资源速率控制公式
    numerator = 2.0 * delta_q_target
    denominator = p_pre * (1.0 - q_c)
    
    if numerator >= denominator:
        # 当目标增量过大时，返回较大的门控率（但有上限）
        return min(0.5, math.log(100.0) / T)
    
    p_gate = -math.log(1.0 - numerator / denominator) / T
    
    return float(p_gate)

def p_gate_window_cap(T: float, tau: float = 0.95) -> float:
    """
    窗口饱和上限：生物可达上限
    
    Args:
        T: 有效时间窗口 T_eff
        tau: 窗口内命中概率上限，默认0.95
        
    Returns:
        p_gate: 门控率上限
    """
    T = max(float(T), 1e-8)
    tau = max(min(float(tau), 0.999), 0.001)
    
    p_gate_cap = -math.log(1.0 - tau) / T
    
    return float(p_gate_cap)

def schedule_p_gate(mode: Literal["achievable", "rate"], *,
                    eps0: float | None = None,
                    M: int | None = None,
                    p_pre: float,
                    T: float,
                    q_c: float,
                    delta_q_target: float | None = None,
                    tau_cap: float = 0.95) -> float:
    """
    统一的门控调度器：根据模式计算最优门控率
    
    Args:
        mode: "achievable" 使用ε-control, "rate" 使用资源速率控制
        eps0: 目标错误率（achievable模式必需）
        M: 任务数量（achievable模式必需）
        p_pre: 预激活概率
        T: 有效时间窗口 T_eff
        q_c: 当前占用率
        delta_q_target: 目标增量占用率（rate模式必需）
        tau_cap: 窗口饱和上限参数
        
    Returns:
        p_gate: 最终门控率
    """
    p_gate_cap = p_gate_window_cap(T, tau_cap)
    
    if mode == "achievable":
        if eps0 is None or M is None:
            raise ValueError("achievable模式需要eps0和M参数")
        p_gate_target = p_gate_epsilon(eps0, M, p_pre, T)
    elif mode == "rate":
        if delta_q_target is None:
            raise ValueError("rate模式需要delta_q_target参数")
        p_gate_target = p_gate_rate(delta_q_target, p_pre, q_c, T)
    else:
        raise ValueError(f"不支持的模式: {mode}")
    
    # 应用窗口上限
    p_gate_final = min(p_gate_target, p_gate_cap)
    
    return float(p_gate_final)

def p_gate_from_epsilon(eps0: float, M: int, p_pre: float, T_eff: float) -> tuple[float, bool]:
    """
    反向工程：目标错误率 → 门控率（ε-control 核心算法）
    
    Args:
        eps0: 目标错误率 ∈ [0, 0.5]，典型值 0.05
        M: 任务数量，随任务增加而增加
        p_pre: 预激活概率，由topk/N决定
        T_eff: 有效时间窗口，T_eff = τ_e * ln(1/θ)
        
    Returns:
        p_gate: 门控率
        achievable: 是否可达（若否，则p_gate被设为保守值）
    """
    # 1. 目标错误率 → 累积翻转率
    eps0 = min(max(float(eps0), 0.0), 0.5)  # 约束到[0, 0.5]
    M = max(int(M), 1)                      # 至少1个任务
    p_pre = max(float(p_pre), 1e-8)         # 防止0除
    T_eff = max(float(T_eff), 1e-8)         # 防止0除
    
    # 错误率→翻转率的反演公式
    # eps(M) = 1/2 * [1 - (1 - 2*p_flip)^M]
    # 解出 p_flip_star
    p_flip_star = 0.5 * (1.0 - (1.0 - 2.0*eps0)**(1.0/M))
    
    # 判断是否可达（核心判定：p_flip_star <= p_pre/2）
    achievable = (2.0 * p_flip_star) <= p_pre + 1e-12
    
    if achievable:
        # 2. 翻转率 → 有效激活率
        # p_flip = (p_pre/2) * p_eff
        p_eff_star = 2.0 * p_flip_star / p_pre
        
        # 3. 有效激活率 → 门控率
        # p_eff = 1 - exp(-p_gate * T_eff)
        # 解出 p_gate
        p_gate = -math.log(max(1.0 - p_eff_star, 1e-6)) / T_eff

        # 上限裁剪
        p_gate_cap = min(0.5, math.log(100.0) / T_eff)  # 99%饱和上限
        p_gate = min(p_gate, p_gate_cap)
    else:
        # 不可达时，使用两级有理上限中的较小者
        # 1. 窗口"接近饱和"的速率上限
        tau = 0.95  # 窗口内95%命中概率
        p_gate_sat = math.log(1.0/(1.0 - tau)) / T_eff
        
        # 2. 占用增长预算的上限（无量纲化）
        c_occ = 0.375  # 占用预算系数（从0.25提到0.375，与memory.py的1.5倍保持一致）
        alpha_budget = c_occ * (p_pre / 2.0)  # 无量纲占用预算
        r = max(1e-6, min(1.0 - 2.0*alpha_budget/p_pre, 1.0 - 1e-6))
        p_gate_occ = -math.log(r) / T_eff
        
        # 取两者中的较小值
        p_gate = min(p_gate_sat, p_gate_occ)
    
    return float(p_gate), achievable

@torch.no_grad()
def apply_gate_schedule(btsp_mem, eps0: float, M: int, p_pre: float, T_eff: float
                       ) -> tuple[float, bool, int | None]:
    """
    应用ε-control门控策略到BTSP记忆：计算并直接设置最优门控率
    
    核心功能:
    1. 基于目标错误率eps0计算最优门控率p_gate
    2. 检查参数可达性：是否能在给定条件下实现目标错误率
    3. 计算最小任务数建议：当不可达时提供达标所需的最小M值
    4. 直接更新记忆对象：将计算结果fill_到btsp_mem.p_gate
    
    Args:
        btsp_mem: BTSPMemory对象
            目标记忆对象，其p_gate buffer将被就地修改
        eps0: float
            目标旧类错误率 ∈ (0, 0.5)，推荐范围[0.01, 0.1]
        M: int  
            任务总数 ≥ 1，影响累积错误的时间尺度
        p_pre: float
            预激活稀疏率 ∈ (0, 1)，通常由Top-k设置决定
        T_eff: float
            有效时间窗口（步数），通常来自T_eff_from_params()
            
    Returns:
        tuple[float, bool, int | None]:
            - p_gate: 计算得到的最优门控率，已应用99%饱和上限
            - achievable: 目标是否可达，False表示参数组合无法实现eps0
            - M_min: 最小建议任务数，仅在achievable=False时有值
            
    Side Effects:
        关键副作用：直接修改btsp_mem.p_gate buffer
        btsp_mem.p_gate.fill_(p_gate) 会将所有分支的门控率设为相同值
        
    Monitoring:
        建议在调用后记录返回值用于ε-control诊断：
        ```python
        p_gate, achievable, M_min = apply_gate_schedule(btsp, eps0, M, p_pre, T_eff)
        
        if not achievable:
            logger.warning(f"ε-control不达标: eps0={eps0:.4f}, M={M}, 建议M_min≥{M_min}")
        else:
            logger.info(f"ε-control已配置: p_gate={p_gate:.6f}, eps0={eps0:.4f}")
        ```
        
    Algorithm:
        基于反向工程的ε-control策略：
        1. 目标错误率 eps0 → 累积翻转率 p_flip_cumulative  
        2. 考虑M任务效应 → 单任务翻转率 p_flip
        3. 预激活约束 → 有效门控率 p_eff
        4. 时间窗口映射 → 最终门控率 p_gate
        
    Example:
        >>> from btsp import BTSPMemory
        >>> from btsp.gate_policy import apply_gate_schedule, T_eff_from_params
        >>> 
        >>> btsp = BTSPMemory(num_classes=10, num_bits=1024)
        >>> T_eff = T_eff_from_params(theta=0.3, tau_e_steps=6.0)
        >>> 
        >>> p_gate, achievable, M_min = apply_gate_schedule(
        ...     btsp, eps0=0.05, M=5, p_pre=0.125, T_eff=T_eff)
        >>> 
        >>> if achievable:
        ...     print(f"ε-control配置成功: p_gate={p_gate:.6f}")  
        ... else:
        ...     print(f"参数不可达，建议M≥{M_min}")
        ...     
        >>> # 验证副作用：btsp.p_gate已被修改
        >>> print(f"记忆门控率: {btsp.p_gate[0].item():.6f}")  # 应该等于p_gate
        
    Note:
        - 函数具有就地修改的副作用，调用前请确保btsp_mem状态正确
        - 返回的p_gate已应用数值安全保护和99%饱和上限
        - achievable=False时，考虑增加M或降低eps0以达成目标
        - 所有分支共享相同的门控率，后续可通过homeostasis_step()调整
    """
    p_gate, achievable = p_gate_from_epsilon(eps0, M, p_pre, T_eff)
    
    # 计算最小可达任务数（如果不可达）
    M_min = None
    if not achievable and eps0 > 0 and p_pre > 0:
        num = math.log(max(1e-12, 1.0 - 2.0*eps0))
        den = math.log(max(1e-12, 1.0 - p_pre))
        M_min = max(1, int(math.ceil(num / den))) if den != 0 else None
    
    # 关键副作用：就地修改记忆对象的门控率
    btsp_mem.p_gate.fill_(p_gate)
    return float(p_gate), achievable, M_min

def epsilon_from_p_gate(p_gate: float, M: int, p_pre: float, T_eff: float) -> float:
    """
    正向链：门控率 → 预期旧类错误率（验证ε-control策略效果）
    
    公式链（与反演保持一致）：
      p_eff = 1 - exp(-p_gate * T_eff)                  # 窗口内命中概率
      p_flip = (p_pre/2) * p_eff                        # 实际翻转率  
      eps(M) = 1/2 * [1 - (1 - 2*p_flip)^M]            # M任务累积错误率
    
    Args:
        p_gate: 每步每分支门控概率 (自动应用99%饱和上限)
        M: 任务数量
        p_pre: 预激活稀疏率
        T_eff: 有效时间窗口（步数），与反演律保持一致：T_eff = τ_e * ln(1/θ)
        
    Returns:
        eps: 预期旧类错误率 ∈ [0, 0.5]
    """
    M = max(int(M), 1)
    p_pre = max(float(p_pre), 1e-8)
    T_eff = max(float(T_eff), 1e-8)
    p_gate = max(float(p_gate), 0.0)
    
    # 统一上限裁剪：99%饱和上限 (与 p_gate_from_epsilon 保持一致)
    p_gate_cap = min(0.5, math.log(100.0) / T_eff)  # 99%饱和
    p_gate = min(p_gate, p_gate_cap)
    
    # 1. 门控率 -> 窗口内有效激活概率 (与 memory.py 写入逻辑一致)
    # 数值安全：限制指数参数范围
    exp_arg = min(p_gate * T_eff, 50.0)  # 防止exp溢出
    p_eff = 1.0 - math.exp(-exp_arg)
    
    # 2. 有效激活 -> 实际翻转概率
    p_flip = (p_pre / 2.0) * p_eff
    
    # 3. 翻转概率 -> M任务累积错误率
    # 原理：M次独立伯努利过程的累积失败概率
    # 数值安全：防止0的负幂运算
    base = max(1.0 - 2.0 * p_flip, 1e-15)
    if M > 100:  # 大M时使用对数域计算避免下溢
        log_survival = M * math.log(base)
        survival_rate = math.exp(max(log_survival, -50.0))  # 防止下溢
    else:
        survival_rate = base ** M
    
    eps_M = 0.5 * (1.0 - survival_rate)
    
    # 最终NaN防护
    if math.isnan(eps_M) or math.isinf(eps_M):
        eps_M = 0.5 if p_flip > 0.25 else 0.0
    
    return float(eps_M)
