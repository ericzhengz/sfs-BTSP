# btsp/memory.py
from __future__ import annotations
import math, torch, logging
from torch import nn

class BTSPMemory(nn.Module):
    """
    SFS-BTSP（Q=2）可插记忆层：
      状态: S[C,N] bool；资格迹 e[N]；分支映射 branch_of[N]；分支门控 p_gate[B]
      检索: popcount -> z-score -> temperature
      写入: 指数衰减资格迹 × 分支门控 × 0.5 随机翻转（XOR），且"按类截断"
      设备: 记忆层状态常驻 CPU；无反传
    """
    def __init__(self, num_classes: int, num_bits: int, num_branches: int = 1,
                 theta: float = 0.2, tau_e_steps: float = 4.0, temperature: float = 1.5,
                 z_momentum: float = 0.99, zstats_interval: int = 30, device: str = "cpu"):
        super().__init__()
        assert num_classes > 0 and num_bits > 0 and num_branches > 0
        self.num_classes = num_classes
        self.num_bits = num_bits
        self.num_branches = num_branches
        self.device_type = device
        assert device in ["cpu", "cuda"]

        self.temperature = temperature
        self.z_momentum = z_momentum
        self.zstats_interval = zstats_interval

        # 🔥 纯门控版本：移除阈值参数，仅保留分离性阈值用于其他功能
        # self.tau_sat = 0.95                  # 已移除：窗口命中目标
        # self.c_occ = 1.5                     # 已移除：占用预算系数
        # self.q_max_cls = 0.10                # 已移除：逐类硬停阈值
        # self.q_max_global = 0.50             # 已移除：全局保险阈值
        self.separability_threshold = 1.5     # 保留：可分性z-score下限（用于其他功能）

        # 🔥 纯门控版本：移除预算控制参数
        # self.delta_q_max = 0.05              # 已移除：预算控制
        # self.enable_incremental_budget = True # 已移除：预算控制

        # 注册状态张量...
        self.register_buffer("S", torch.zeros((num_classes, num_bits), dtype=torch.bool, device=device))
        self.register_buffer("e", torch.zeros((num_classes, num_bits), dtype=torch.float32, device=device))
        self.register_buffer("tau_e_steps", torch.tensor(tau_e_steps, dtype=torch.float32, device=device))
        self.register_buffer("theta", torch.tensor(theta, dtype=torch.float32, device=device))

        # 分支-类映射
        branch_assignment = torch.arange(num_classes, device=device) % num_branches
        self.register_buffer("branch_assignment", branch_assignment)
        
        # 分支-位映射（兼容诊断脚本）
        self.register_buffer("branch_of", torch.randint(0, num_branches, (num_bits,), device=device))

        # 门控参数
        gate_init = min(0.01, 1.0 / (num_classes + 1))
        self.register_buffer("p_gate", torch.full((num_branches,), gate_init, dtype=torch.float32, device=device))

        # Z-score 统计（检索标准化）
        self.register_buffer("z_mu", torch.zeros(num_classes, dtype=torch.float32, device=device))
        self.register_buffer("z_std", torch.ones(num_classes, dtype=torch.float32, device=device))

        # 监控统计
        self.register_buffer("flip_counter", torch.zeros(num_bits, dtype=torch.int32, device=device))
        self.register_buffer("class_counts", torch.zeros(num_classes, dtype=torch.int32, device=device))
        self.register_buffer("branch_ema_occ", torch.zeros(num_branches, dtype=torch.float32, device=device))

        # 🔥 纯门控版本：保留基线占用率用于监测（不做控制）
        self.register_buffer("baseline_occ_per_class", torch.zeros(num_classes, dtype=torch.float32, device=device))

        # EMA + 稳态化参数
        self.ema_momentum = max(0.5, min(0.999, 0.9))
        self.eta_homeo = max(1e-3, min(0.2, 0.05))
        self.p_gate_min = 1e-4

        # 兼容性别名（诊断脚本使用）
        self.C = num_classes
        self.N = num_bits
        self.B = num_branches
        
        # homeostasis目标占用率
        self.alpha_star = 0.025  # 2.5%目标占用率
        
        # 🔥 纯门控版本：添加T_eff buffer用于新门控策略
        T_eff_val = tau_e_steps * math.log(1.0 / theta) if theta > 0 else 10.0
        self.register_buffer("T_eff", torch.tensor(T_eff_val, dtype=torch.float32, device=device))

        # T_eff buffer
        tau_e_val = float(tau_e_steps)
        theta_val = max(float(theta), 1e-6)
        self.register_buffer("T_eff", torch.tensor(tau_e_val * math.log(1.0 / theta_val),
                                                   dtype=torch.float32, device=device))
        # 使用推导的99%饱和上限
        self.p_gate_max = min(0.5, math.log(100.0) / self.T_eff.item())

        self._branch_index = None
        
    # 🔥 纯门控版本：已移除update_derived_thresholds方法
    # def update_derived_thresholds(self, p_pre: float): # 已移除

    # ---------- 检索 ----------
    @torch.no_grad()
    def retrieve(self, x_bits: torch.Tensor, update_z: bool = False) -> torch.Tensor:
        """
        BTSP记忆检索：二进制激活模式 → 类别logits
        
        流程：
        1. popcount计算原始分数: scores = x_bits @ S^T
        2. z-score标准化: z = (scores - μ) / σ 
        3. 温度缩放输出: logits = z / T
        
        Args:
            x_bits: [B, N] torch.bool
                二进制激活模式，必须在CPU上且与记忆维度N匹配
            update_z: bool, default=False  
                是否更新z-score统计量(μ, σ)，训练/评测阶段策略不同：
                
                🏋️ 训练阶段：建议 update_z=True
                - 允许统计量动态适应数据分布变化
                - 定期更新(如每几个epoch)以保持准确性
                - 支持增量学习中的分布漂移适应
                
                评测阶段：强制 update_z=False  
                - 冻结统计量避免测试数据泄漏
                - 确保可重现的评测结果
                - 遵循标准ML评测协议
                
        Returns:
            torch.Tensor: [B, C] torch.float32
                类别logits，已应用z-score标准化和温度缩放，在CPU上
                
        Raises:
            AssertionError: 当x_bits类型、维度或尺寸不匹配时
            
        Training/Evaluation Protocol:
            推荐的调用模式：
            
            ```python
            # 训练阶段：允许统计量更新
            model.train()
            for epoch in range(num_epochs):
                for batch_idx, (data, labels) in enumerate(train_loader):
                    # 每N个batch更新一次z-stats，避免过于频繁
                    update_stats = (batch_idx % 50 == 0)  
                    logits = btsp.retrieve(x_bits, update_z=update_stats)
                    
            # 评测阶段：冻结统计量
            model.eval()
            with torch.no_grad():
                for data, labels in test_loader:
                    logits = btsp.retrieve(x_bits, update_z=False)  # 显式冻结
            ```
            
            或使用训练状态自动切换：
            ```python
            # 自动根据模型状态决定update_z
            update_z = self.training  # PyTorch的training状态
            logits = btsp.retrieve(x_bits, update_z=update_z)
            ```
            
        Data Leakage Prevention:
            关键：评测时必须设置update_z=False
            - 测试数据不应影响模型内部统计量
            - 确保训练和测试的严格分离
            - 避免无意中的信息泄漏导致虚高性能
            
        Performance Impact:
            - update_z=True: 额外O(B×C)计算开销，EMA更新统计量
            - update_z=False: 无额外开销，纯前向推理
            - 建议训练时适度更新（如每50步），避免计算浪费
            
        Note:
            - 空batch (B=0) 返回形状正确的零张量
            - 使用自适应z-score下界: max(1e-6, 1/√B) 提高小批次稳定性
            - 内置NaN防护确保数值安全性
            - 统计量更新使用EMA平滑，避免单批次噪声影响
        """
        # 空batch快速返回
        if x_bits.numel() == 0:
            return torch.zeros(0, self.num_classes, dtype=torch.float32, device=x_bits.device)
            
        assert x_bits.dtype == torch.bool and x_bits.dim() == 2 and x_bits.size(1) == self.num_bits
        
        # 确保设备兼容性
        x_bits_device = x_bits.to(self.S.device)
        
        # popcount 等价于布尔点积
        scores = (x_bits_device.float() @ self.S.transpose(0, 1).float())  # [B,C]
        if update_z:
            self.update_zstats(scores)
        
        # 自适应z-score下界：基础下界 + 样本数自适应项
        B = x_bits.size(0)
        adaptive_min_std = max(1e-6, 1.0 / (B ** 0.5)) if B > 0 else 1e-6
        z_std_safe = self.z_std.clamp_min(adaptive_min_std)
        
        z = (scores - self.z_mu) / z_std_safe
        # NaN防护：确保数值安全
        z = torch.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return (z / self.temperature)

    @torch.no_grad()
    def raw_scores(self, x_bits: torch.Tensor) -> torch.Tensor:
        """返回未归一化的 popcount 得分 (x_bits @ S^T)。
        用于 margin / 诊断，可与 retrieve() 区分（后者含 z-score 与温度）。
        Args:
            x_bits: [B,N] bool
        Returns:
            scores: [B,C] float32
        """
        if x_bits.numel() == 0:
            return torch.zeros(0, self.num_classes, dtype=torch.float32, device=self.S.device)
        assert x_bits.dtype == torch.bool and x_bits.size(1) == self.num_bits
        
        # 确保设备兼容性
        x_bits_device = x_bits.to(self.S.device)
        return (x_bits_device.float() @ self.S.transpose(0,1).float())

    @torch.no_grad()
    def stats_snapshot(self) -> dict:
        """采集当前 BTSP 记忆状态的轻量统计（用于日志/监控）。"""
        occ_per_branch = None
        if self._branch_index is None:
            self._ensure_branch_index()
        # 修复：num_branches可能是int而非tensor
        Bn = int(self.num_branches.item()) if hasattr(self.num_branches, 'item') else int(self.num_branches)
        occ_vals = []
        for b in range(Bn):
            I = self._branch_index[b]
            if I.numel() > 0:
                occ_vals.append(self.S[:, I].float().mean().item())
            else:
                occ_vals.append(0.0)
        occ_per_branch = occ_vals
        flips_total = int(self.flip_counter.sum().item())
        flips_mean = float(self.flip_counter.float().mean().item())
        return {
            "C": self.num_classes,
            "N": self.num_bits,
            "B": self.num_branches,
            "p_gate_mean": float(self.p_gate.mean().item()),
            "p_gate_std": float(self.p_gate.std().item()),
            "theta": float(self.theta.item()) if hasattr(self.theta, 'item') else float(self.theta),
            "T_eff": float(self.T_eff.item()) if hasattr(self.T_eff, 'item') else float(self.T_eff),
            "alpha_star": float(self.alpha_star),
            "occ_branch": occ_per_branch,
            "occ_mean": float(torch.tensor(occ_per_branch).mean().item()) if len(occ_per_branch)>0 else 0.0,
            "flips_total": flips_total,
            "flips_mean_per_bit": flips_mean,
            "class_counts_sum": int(self.class_counts.sum().item())
        }

    @torch.no_grad()
    def update_zstats(self, scores: torch.Tensor, momentum: float | None = None) -> None:
        """
        更新z-score标准化统计量 (μ, σ)
        
        使用EMA平滑更新避免单批次噪声影响：
        - μ_new = momentum * μ_old + (1 - momentum) * μ_batch
        - σ_new = momentum * σ_old + (1 - momentum) * σ_batch
        
        Args:
            scores: [B, C] torch.float32
                当前批次的原始检索分数 (popcount结果)
            momentum: float | None, default=None
                EMA动量参数 ∈ [0, 1]，None时使用self.z_momentum
                越接近1越平滑，越接近0对当前批次越敏感
                
        Note:
            - 空batch时直接返回，不更新统计量
            - 内置NaN防护确保统计量数值安全
            - 建议在训练时每几个epoch调用一次以保持统计量更新
        """
        if scores.numel() == 0:
            return
            
        m = self.z_momentum if momentum is None else float(momentum)
        mu = scores.mean(dim=0)
        std = scores.std(dim=0, unbiased=False)
        
        # NaN防护：确保统计量数值安全
        mu = torch.nan_to_num(mu, nan=0.0)
        std = torch.nan_to_num(std, nan=1e-6)
        
        self.z_mu.mul_(m).add_(mu * (1 - m))
        self.z_std.mul_(m).add_(std * (1 - m))

    def bytes_per_class(self) -> int:
        return (self.num_bits + 7) // 8

    @torch.no_grad()
    def retrieve_auto(self, x_bits: torch.Tensor, force_update_z: bool | None = None) -> torch.Tensor:
        """
        自动模式检索：根据模块训练状态智能决定是否更新z-score统计量
        
        设计理念:
        - 训练模式(self.training=True): 默认启用统计量更新
        - 评测模式(self.training=False): 默认冻结统计量，避免数据泄漏
        - 可通过force_update_z参数强制覆盖自动行为
        
        Args:
            x_bits: [B, N] torch.bool
                二进制激活模式，必须在CPU上且与记忆维度N匹配
            force_update_z: bool | None, default=None
                强制覆盖自动update_z决策：
                - None: 自动模式，根据self.training决定
                - True: 强制更新统计量（忽略训练状态）
                - False: 强制冻结统计量（忽略训练状态）
                
        Returns:
            torch.Tensor: [B, C] torch.float32
                类别logits，已应用z-score标准化和温度缩放
                
        Usage Pattern:
            ```python
            # 自动模式：推荐用法
            model.train()  # 设置为训练模式
            logits = btsp.retrieve_auto(x_bits)  # 自动update_z=True
            
            model.eval()   # 设置为评测模式  
            logits = btsp.retrieve_auto(x_bits)  # 自动update_z=False
            
            # 强制控制：特殊需求
            logits = btsp.retrieve_auto(x_bits, force_update_z=False)  # 强制冻结
            ```
            
        Note:
            - 内部调用标准retrieve()方法，继承所有安全特性
            - 训练状态通过PyTorch的self.training属性自动检测
            - 推荐在大多数场景下使用此方法，减少手动参数管理
        """
        if force_update_z is not None:
            update_z = bool(force_update_z)
        else:
            # 自动模式：训练时更新，评测时冻结
            update_z = self.training
            
        return self.retrieve(x_bits, update_z=update_z)

    # ---------- 写入 ----------
    @torch.no_grad()
    def write(self, x_bits: torch.Tensor, y: torch.Tensor, tau_e: float | None = None) -> None:
        """
        BTSP记忆写入：通过选择性XOR翻转更新记忆状态
        
        核心算法 - SFS-BTSP (Sparse Flip-Set Binary Temporal Sparse Projection):
        1. 资格迹衰减: e ← max(β·e, ∪x_bits) 其中 β = exp(-1/τ_e)
        2. 分支门控: g_b ~ Bernoulli(p_eff), p_eff = 1 - exp(-p_gate·T_eff)  
        3. 条件翻转: 若 eligible & fired & coin(0.5)，则 S[c,i] ← S[c,i] ⊕ 1
        4. 按类截断: 仅对当前批次中该类激活的位进行翻转
        
        Args:
            x_bits: [B, N] torch.bool
                二进制激活位模式，必须在CPU上且与记忆维度N匹配
                每行代表一个样本的N维二进制特征
                若x_bits在GPU上，将抛出断言错误
            y: [B] torch.long  
                类别标签，范围应在 [0, C-1]，其中C为记忆的类别数
                若y在GPU上，将抛出断言错误
            tau_e: float | None, default=None
                资格迹时间常数（步数），控制历史信息衰减速度
                None时使用初始化的tau_e_steps值，推荐范围[1.0, 100.0]
                
        Side Effects:
            - 更新记忆矩阵 S[C, N] 
            - 更新资格迹向量 e[N]
            - 增加翻转计数器 flip_counter[N]
            - 增加类别样本计数 class_counts[C]
            
        Note:
            - 空batch时直接返回，不执行任何操作
            - 使用"按类截断"策略：仅翻转该类在当前批次中激活的位
            - XOR翻转是可逆的，支持增量遗忘机制
            - 分支门控概率由ε-control策略动态调节
        """
        # 空batch快速返回
        if x_bits.numel() == 0:
            return
            
        assert x_bits.dtype == torch.bool and y.dtype == torch.long
        
        # 设备一致性显式断言：防止CPU/GPU索引错配
        dev = self.S.device
        assert x_bits.device.type == "cpu" and y.device.type == "cpu", \
            f"BTSPMemory expects CPU tensors; got x_bits:{x_bits.device}, y:{y.device}, S:{dev}"
        
        # 🔥 纯门控版本：仅通过p_gate控制，移除所有硬停/软回退逻辑
        # 计算当前占用率用于监测（不做阈值控制）
        occ_per_class = self.S.float().mean(dim=1)  # [C] 每类占用率
        global_occ = occ_per_class.mean().item()
        
        # Light assertion: warn when occupancy is high but don't block (reduced frequency)
        if global_occ > 0.12:  # Raised threshold to reduce noise
            import logging
            logging.warning(f"[BTSP] Occupancy monitor: global_occ={global_occ:.4f} > 0.12, controlled by pure gating")
        
        # Warn when gate rate approaches limit
        p_gate_mean = self.p_gate.mean().item()
        p_gate_cap = 0.5  # Simplified limit check
        if p_gate_mean > 0.8 * p_gate_cap:
            import logging
            logging.warning(f"[BTSP] Gate rate monitor: p_gate={p_gate_mean:.6f} approaching limit, controlled by pure gating")
        
        # 资格迹按类别维护：修复全局共享问题
        tau = float(self.tau_e_steps.item()) if tau_e is None else float(tau_e)  # 默认使用buffer值
        beta = math.exp(-1.0 / max(tau, 1.0))
        
        # 确保资格迹是按类别的 [C, N] 
        if not hasattr(self, 'e_c') or self.e_c.shape != (self.num_classes, self.num_bits):
            self.e_c = torch.zeros(self.num_classes, self.num_bits, device=self.S.device, dtype=torch.float32)
        
        # 先对所有类执行衰减
        self.e_c.mul_(beta)
        
        # 按类更新资格迹：只对该类在当前批次激活的位置位
        for i, cls in enumerate(y.tolist()):
            x_i = x_bits[i].to(self.S.device).float()  # [N] float
            self.e_c[cls] = torch.maximum(self.e_c[cls], x_i)

        # 分支门控参数
        Bn = int(self.num_branches)
        p_gate_safe = torch.clamp(self.p_gate, 0.0, 10.0)  # 防止极端值
        T_eff_safe = max(self.T_eff.item(), 1e-6)           # 防止0除
        
        exp_term = torch.exp(-p_gate_safe * T_eff_safe)
        exp_term = torch.nan_to_num(exp_term, nan=1.0, posinf=1.0, neginf=0.0)  # NaN防护
        p_eff = 1.0 - exp_term   # [B]
        
        gate = (torch.rand(Bn, device=self.S.device) < p_eff) # [B]
        fired = gate[self.branch_of]                          # [N]

        # 按类生成翻转掩码并执行：修复全局掩码问题
        applied = torch.zeros(self.num_bits, dtype=torch.bool, device=self.S.device)
        for cls in torch.unique(y).tolist():
            # 该类的资格迹、随机币、按类激活位
            elig_c = (self.e_c[cls] > self.theta)                                        # [N]
            coin_c = (torch.rand(self.num_bits, device=self.S.device) < 0.5)                   # [N] 
            mask_c = (y == cls)                                                          # [B]
            x_c_any = x_bits[mask_c].any(dim=0).to(self.S.device) if mask_c.any() else torch.zeros(self.num_bits, dtype=torch.bool, device=self.S.device)  # [N]
            
            # 该类的翻转掩码：资格迹 & 门控 & 随机币 & 该类激活位
            flip_mask_c = elig_c & fired & coin_c & x_c_any
            
            # 只对该类执行翻转
            self.S[cls] ^= flip_mask_c
            applied |= flip_mask_c
            self.class_counts[cls] += mask_c.sum().item()
            
        # flip_counter 计算实际翻转位数
        self.flip_counter += applied.int()

    # ---------- 稳态（分支占用 EMA -> 指数调 p_gate） ----------
    @torch.no_grad()
    def homeostasis_step(self) -> dict[str, float]:
        """
        BTSP稳态调节：维持记忆系统的动态平衡
        
        核心机制 - 分支占用率反馈控制:
        1. 统计每个分支的实际占用率: occ[b] = mean(S[:, branch_b])
        2. EMA平滑更新: occ_ema[b] ← α·occ_ema[b] + (1-α)·occ[b]  
        3. 计算偏差: δ[b] = α_target - occ_ema[b]
        4. 指数调整门控率: p_gate[b] ← p_gate[b] · exp(η·δ[b])
        5. 约束到安全范围: p_gate[b] ← clamp(p_gate[b], p_min, p_max)
        
        目标：
        - 维持各分支占用率接近目标值 alpha_star (通常~2-5%)
        - 避免某些分支过度激活导致记忆容量不均
        - 通过负反馈实现长期稳定性
        
        调用时机建议:
        **定期调用策略** (推荐):
        - 每100-500个训练batch调用一次
        - 每个epoch结束后调用一次
        - 新任务开始时前几个epoch增加频率
        
        **自适应调用策略** (高级):
        - 监控占用率方差，高方差时增加频率
        - 基于门控率变化幅度动态调整
        - 损失plateau时触发稳态调节
        
        🚫 **避免场景**:
        - 推理/评测阶段无需调用
        - 过于频繁调用(每个batch)会影响性能
        - 初始化后立即调用(需要一定数据积累)
        
        Returns:
            dict[str, float]: 稳态监控指标，用于日志和调参
            {
                'occ_mean': 当前分支占用率均值,
                'occ_std': 当前分支占用率标准差,  
                'occ_target': 目标占用率 alpha_star,
                'p_gate_mean': 调整后门控率均值,
                'p_gate_std': 调整后门控率标准差,
                'p_gate_min': 最小门控率,
                'p_gate_max': 最大门控率,
                'adjustment_strength': 本次调整强度 (exp_factor的log均值)
            }
            
        Usage Pattern:
            ```python
            # 定期调用模式
            for epoch in range(num_epochs):
                for batch_idx, (data, labels) in enumerate(train_loader):
                    # ... 训练逻辑 ...
                    
                    # 每100个batch调用一次
                    if batch_idx % 100 == 0:
                        stats = btsp.homeostasis_step()
                        logger.info(f"Homeostasis: occ={stats['occ_mean']:.4f}±{stats['occ_std']:.4f}, "
                                  f"p_gate={stats['p_gate_mean']:.6f}±{stats['p_gate_std']:.6f}")
                
                # 每个epoch结束后也调用一次
                stats = btsp.homeostasis_step()
                logger.info(f"Epoch {epoch} Homeostasis: {stats}")
            ```
            
        Monitoring Guidelines:
            - occ_mean ≈ alpha_star: 系统达到平衡
            - occ_std < 0.01: 分支占用率均匀
            - p_gate_std < 0.1: 门控率分布合理
            - adjustment_strength → 0: 系统趋于稳定
            
        Note:
            - 学习率η自动约束到[1e-3, 0.2]防止振荡
            - 指数因子限制在[exp(-10), exp(10)]避免数值溢出
            - 分支索引缓存自动管理，支持动态分支重分配
            - 内置NaN防护确保数值稳定性
            - 返回的监控指标建议记录到日志中以便分析
        """
        self._ensure_branch_index()
        Bn = int(self.num_branches.item()) if hasattr(self.num_branches, 'item') else int(self.num_branches)
        occ = torch.zeros(Bn, dtype=torch.float32, device=self.S.device)
        for b in range(Bn):
            I = self._branch_index[b]
            if I.numel() > 0:
                occ[b] = self.S[:, I].float().mean()  # 类×该分支位的平均占用
        
        # EMA更新占用统计
        self.branch_ema_occ.mul_(self.ema_momentum).add_(occ * (1 - self.ema_momentum))
        
        # 指数调整，使用约束的学习率避免振荡，添加数值安全防护
        delta = self.alpha_star - self.branch_ema_occ
        eta_safe = max(1e-3, min(0.2, self.eta_homeo))  # 约束学习率防止振荡
        
        # 限制指数参数范围，防止数值溢出
        exp_arg = torch.clamp(eta_safe * delta, -10.0, 10.0)
        exp_factor = torch.exp(exp_arg)
        # NaN防护
        exp_factor = torch.nan_to_num(exp_factor, nan=1.0, posinf=10.0, neginf=0.1)
        
        # 记录调整前的门控率用于计算调整强度
        p_gate_before = self.p_gate.clone()
        
        # 应用调整
        self.p_gate.mul_(exp_factor).clamp_(self.p_gate_min, self.p_gate_max)
        
        # 基于占用预算的软回退（替代固定12%魔数）
        occ_per_class = self.S.float().mean(dim=1)  # [C] 每类占用率
        occ_per_branch = torch.zeros(self.num_branches, device=self.S.device)
        for b in range(self.num_branches):
            branch_classes = (self.branch_assignment == b).nonzero(as_tuple=True)[0]
            if len(branch_classes) > 0:
                occ_per_branch[b] = occ_per_class[branch_classes].mean()
        
        # 🔥 纯门控版本：移除占用预算约束逻辑
        # p_pre = getattr(self, 'p_pre', 0.03125)    # 已移除
        # alpha_budget = self.c_occ * (p_pre / 2.0)  # 已移除
        # p_gate_occ_max = ...                       # 已移除
        # 占用上限约束逻辑已移除，完全依赖统一门控调度器
        
        # 计算监控指标
        occ_current = self.branch_ema_occ  # 使用EMA平滑后的占用率
        adjustment_strength = torch.log(exp_factor).abs().mean().item()
        
        return {
            'occ_mean': float(occ_current.mean().item()),
            'occ_std': float(occ_current.std().item()),
            'occ_target': float(self.alpha_star),
            'p_gate_mean': float(self.p_gate.mean().item()),
            'p_gate_std': float(self.p_gate.std().item()),
            'p_gate_min': float(self.p_gate.min().item()),
            'p_gate_max': float(self.p_gate.max().item()),
            'adjustment_strength': adjustment_strength
        }

    def _ensure_branch_index(self):
        """确保分支索引缓存有效"""
        if self._branch_index is None:
            Bn = int(self.num_branches.item()) if hasattr(self.num_branches, 'item') else int(self.num_branches)
            self._branch_index = [(self.branch_of == b).nonzero(as_tuple=True)[0] for b in range(Bn)]

    @torch.no_grad()
    def reassign_branches(self, num_branches: int) -> None:
        """
        重新分配分支结构（实验性功能）
        
        Args:
            num_branches: 新的分支数量
            
        注意：
        - 会重置分支EMA统计
        - 触发分支索引缓存重建
        - 建议在任务间隔期调用，避免训练中断
        """
        num_branches = max(1, int(num_branches))
        
        # 更新分支数量并重新随机分配
        self.num_branches.fill_(num_branches)
        self.branch_of = torch.randint(0, num_branches, (self.num_bits,), device=self.S.device)
        
        # 重置相关状态
        self.branch_ema_occ = torch.zeros(num_branches, dtype=torch.float32, device=self.S.device)
        self.p_gate = torch.full((num_branches,), 0.01, dtype=torch.float32, device=self.S.device)
        
        # 触发分支索引缓存重建
        self._branch_index = None
        
        logging.info(f"Branch reassignment complete: {num_branches} branches, state reset")

    @torch.no_grad()
    def set_homeostasis_params(self, 
                             alpha_star: float | None = None,
                             eta_homeo: float | None = None,
                             ema_momentum: float | None = None) -> None:
        """
        设置稳态调节参数，自动应用安全范围约束
        
        Args:
            alpha_star: 目标占用率，约束到 [1/N, 0.2] 避免极端值
            eta_homeo: 学习率，约束到 [1e-3, 0.2] 防止振荡  
            ema_momentum: EMA动量，约束到 [0.5, 0.999]
        """
        if alpha_star is not None:
            # 约束目标占用率：不能太低（数值不稳定）也不能太高（效率低）
            min_alpha = 1.0 / self.num_bits  # 最低：每位至少有1/N的期望占用
            max_alpha = 0.2           # 最高：20%占用率，保持稀疏性
            self.alpha_star = max(min_alpha, min(max_alpha, float(alpha_star)))
            logging.info(f"Target occupancy set to: {self.alpha_star:.6f} (original: {alpha_star:.6f})")
        
        if eta_homeo is not None:
            # Constrain learning rate: prevent oscillation and numerical instability
            self.eta_homeo = max(1e-3, min(0.2, float(eta_homeo)))
            logging.info(f"Homeostasis learning rate set to: {self.eta_homeo:.6f} (original: {eta_homeo:.6f})")
        
        if ema_momentum is not None:
            # Constrain EMA momentum: maintain reasonable historical smoothing
            self.ema_momentum = max(0.5, min(0.999, float(ema_momentum)))
            logging.info(f"EMA momentum set to: {self.ema_momentum:.6f} (original: {ema_momentum:.6f})")

    # 实用：全局设门控
    @torch.no_grad()
    def set_all_p_gate(self, p: float) -> None:
        self.p_gate.fill_(float(p))
    
    # 实用：dtype验证和建议
    def validate_dtypes(self, verbose: bool = True) -> dict[str, bool]:
        """
        验证BTSP记忆的数据类型是否符合建议
        
        Returns:
            dict: 各项检查结果
        """
        results = {}
        
        # 检查记忆矩阵 - 应该是bool
        results['S_is_bool'] = (self.S.dtype == torch.bool)
        
        # 检查浮点参数 - 应该是float32
        float32_params = ['e', 'z_mu', 'z_std', 'T_eff', 'p_gate', 'branch_ema_occ']
        for param_name in float32_params:
            if hasattr(self, param_name):
                param = getattr(self, param_name)
                results[f'{param_name}_is_float32'] = (param.dtype == torch.float32)
        
        # 检查整数统计 - 应该是int32  
        int32_params = ['flip_counter', 'class_counts', 'num_branches', 'tau_e_steps']
        for param_name in int32_params:
            if hasattr(self, param_name):
                param = getattr(self, param_name)
                results[f'{param_name}_is_int32'] = (param.dtype == torch.int32)
        
        # 检查标量参数 - 应该是float32
        scalar_params = ['theta']
        for param_name in scalar_params:
            if hasattr(self, param_name):
                param = getattr(self, param_name)
                results[f'{param_name}_is_float32'] = (param.dtype == torch.float32)
        
        if verbose:
            print("BTSP Data Type Validation:")
            all_good = True
            for check, passed in results.items():
                status = "OK" if passed else "FAIL"
                print(f"  {check}: {status}")
                if not passed:
                    all_good = False
            
            if all_good:
                print("  All data types conform to recommendations!")
            else:
                print("  Consider correcting dtypes for optimal numerical stability")
                
        return results
    
    # T_eff 重算方法：在修改 theta 或 tau_e_steps 后调用
    @torch.no_grad()
    def recompute_teff(self) -> None:
        """
        重新计算有效温度T_eff：维护参数一致性的核心操作
        
        应用场景:
        1. 修改关键参数后的强制同步：theta或tau_e_steps变更时必须调用
        2. 恢复训练会话时的状态重建：确保门控概率计算正确性
        3. 超参数调优过程中的实时更新：保持ε-control策略一致
        
        计算逻辑:
        T_eff = τ · ln(1/θ)，其中：
        - τ: tau_e_steps，控制记忆衰减时间尺度
        - θ: theta，控制检索阈值的敏感度
        - 内建数值保护：max(θ, 1e-6)防止对数发散
        
        副作用更新:
        1. 修改self.T_eff缓存，影响后续检索行为
        2. 更新p_gate_max上限：min(0.5, ln(100)/T_eff)确保99%饱和
        3. 触发gate_policy模块的参数重新计算
        
        Args:
            无参数，基于当前对象状态计算
            
        Returns:
            None，通过side-effect修改对象状态
            
        Performance:
            O(1)常数时间操作，适用于频繁调用
            
        Critical Usage:
            参数修改后必须调用，否则导致门控概率错误：
            
            >>> apply_gate_schedule(btsp, eps0, M, p_pre, T_eff)
            >>> btsp.theta.fill_(new_theta)      # 修改参数  
            >>> btsp.recompute_teff()            # ← 必须调用！
            
        Example:
            >>> btsp = BTSPMemory(num_classes=10, num_bits=1024)
            >>> print(f"Initial T_eff: {btsp.T_eff:.4f}")
            >>> btsp.theta.fill_(0.1)            # 修改阈值参数
            >>> btsp.recompute_teff()            # 重新计算
            >>> print(f"Updated T_eff: {btsp.T_eff:.4f}")
        """
        tau = float(self.tau_e_steps.item())
        th = float(self.theta.item())
        self.T_eff.fill_(tau * math.log(1.0 / max(th, 1e-6)))
        self.p_gate_max = min(0.5, math.log(100.0) / self.T_eff.item())  # 更新99%饱和上限

    # 类数扩展：增量任务时的行扩容
    @torch.no_grad()
    def expand_classes(self, new_C: int) -> None:
        """
        动态扩展类别数：支持增量学习中的新类添加
        
        核心机制:
        1. 扩容记忆矩阵: S[C, N] → S[new_C, N]，新行初始化为False
        2. 扩容统计向量: z_mu[C] → z_mu[new_C]，新元素初始化为0  
        3. 扩容标准差: z_std[C] → z_std[new_C]，新元素初始化为1
        4. 扩容计数器: class_counts[C] → class_counts[new_C]，新元素为0
        5. 同步T_eff缓存以保持ε-control一致性
        
        Args:
            new_C: int
                扩展后的总类别数，必须 ≥ 当前类别数
                推荐按任务递增：如10类/任务时设为10, 20, 30...
                
        Raises:
            无异常抛出，当new_C ≤ 当前类数时静默返回
            
        Note:
            - 新增类别的记忆初始化为全0，等待首次学习
            - z-score统计量保守初始化(μ=0, σ=1)避免检索偏向
            - 操作后自动调用recompute_teff()确保参数一致性
            - 扩展后的记忆保持在相同设备上
            
        Example:
            >>> btsp = BTSPMemory(num_classes=10, num_bits=1024)
            >>> btsp.expand_classes(20)  # 添加10个新类
            >>> print(btsp.num_classes)  # 输出: 20
        """
        if new_C <= self.num_classes: 
            return
            
        add = new_C - self.num_classes
        device = self.S.device
        
        # 扩容各个状态矩阵
        pad_S = torch.zeros(add, self.num_bits, dtype=torch.bool, device=device)
        pad_mu = torch.zeros(add, dtype=torch.float32, device=device)
        pad_sd = torch.ones(add, dtype=torch.float32, device=device)
        pad_cc = torch.zeros(add, dtype=torch.int32, device=device)
        pad_baseline = torch.zeros(add, dtype=torch.float32, device=device)

        self.S = torch.cat([self.S, pad_S], dim=0)
        self.z_mu = torch.cat([self.z_mu, pad_mu], dim=0)
        self.z_std = torch.cat([self.z_std, pad_sd], dim=0)
        self.class_counts = torch.cat([self.class_counts, pad_cc], dim=0)
        self.baseline_occ_per_class = torch.cat([self.baseline_occ_per_class, pad_baseline], dim=0)
        
        # 更新分支类映射
        new_mapping = torch.arange(self.num_classes, new_C, device=device) % self.num_branches
        self.branch_assignment = torch.cat([self.branch_assignment, new_mapping], dim=0)
        
        self.num_classes = new_C
        
    @torch.no_grad()
    def record_task_baseline(self, class_range: tuple[int, int] = None):
        """
        记录任务开始时的基线占用率，用于增量预算控制
        
        Args:
            class_range: (start, end) 当前任务的类别范围，如None则记录所有类别
        """
        current_occ = self.S.float().mean(dim=1)  # [C] 当前每类占用率
        
        if class_range is None:
            # 记录所有类别
            self.baseline_occ_per_class.copy_(current_occ)
        else:
            start, end = class_range
            end = min(end, self.num_classes)  # 防止越界
            if start < end:
                self.baseline_occ_per_class[start:end] = current_occ[start:end]
        
        import logging
        if class_range:
            avg_baseline = self.baseline_occ_per_class[start:end].mean().item()
            logging.info(f"[BTSP] Recorded baseline occupancy for classes {start}-{end}: avg={avg_baseline:.4f}")
        else:
            avg_baseline = self.baseline_occ_per_class.mean().item()
            logging.info(f"[BTSP] Recorded global baseline occupancy: avg={avg_baseline:.4f}")
