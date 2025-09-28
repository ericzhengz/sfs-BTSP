# btsp/metrics.py
from __future__ import annotations
import torch
from typing import Iterator
import numpy as np

@torch.no_grad()
def n_eff_from_batch_bits(x_bits: torch.Tensor) -> float:
    """
    x_bits: [B,N] bool -> N_eff ≈ (tr Σ)^2 / ||Σ||_F^2, Σ 为协方差
    用批内估计；大 N 时可滑窗累计一阶/二阶矩避免 O(N^2) 储存
    """
    x = x_bits.float()
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)   # [N,N]（注意内存！小N可用）
    tr = torch.trace(cov)
    fro2 = torch.linalg.matrix_norm(cov, ord='fro') ** 2
    return float((tr * tr / (fro2 + 1e-9)).clamp(min=1.0))

@torch.no_grad()
def n_eff_streaming(x_bits_iter: Iterator[torch.Tensor], eps: float = 1e-9, momentum: float = 0.9) -> float:
    """
    流式N_eff计算：避免O(N²)内存复杂度，适用于大N场景
    
    原理：
    - 仅跟踪一阶矩(均值)和二阶矩(方差)，忽略协方差项
    - 近似：tr(Σ) = Σᵢ var[i]，||Σ||²_F ≈ Σᵢ var[i]² （保守估计）
    - 使用EMA平滑更新避免对单个batch的过敏感
    
    Args:
        x_bits_iter: 迭代器，产出[B,N] bool张量 
        eps: 数值稳定项
        momentum: EMA动量，越大越平滑
        
    Returns:
        N_eff: 有效维度的保守估计（偏小的上界）
        
    注意：
        - 这是保守估计，实际N_eff可能更大
        - 忽略协方差项会低估相关性带来的维度折叠效应
        - 适用于大N时的归一化和趋势监控
    """
    n = None
    m1 = None  # 一阶矩 (均值)
    q2 = None  # 二阶原始矩 E[X²]
    batch_count = 0
    
    for xb in x_bits_iter:
        if xb.numel() == 0:  # 空batch跳过
            continue
            
        x = xb.float()  # [B,N]
        B, N = x.shape
        
        # 初始化
        if n is None:
            n = N
            m1 = torch.zeros(N, dtype=torch.float32, device=x.device)
            q2 = torch.zeros(N, dtype=torch.float32, device=x.device)
        elif N != n:
            raise ValueError(f"维度不一致: 期望{n}, 得到{N}")
        
        # 批内统计
        batch_mean = x.mean(dim=0)  # [N]
        batch_q2 = (x**2).mean(dim=0)  # [N] 
        
        # EMA更新
        alpha = 1.0 - momentum  # 学习率
        m1 = momentum * m1 + alpha * batch_mean
        q2 = momentum * q2 + alpha * batch_q2
        
        batch_count += 1
    
    if batch_count == 0:
        return 1.0  # 无数据时返回最小有效维度
    
    # 计算方差: Var[X] = E[X²] - E[X]²
    var = (q2 - m1**2).clamp_min(0.0)  # [N]
    
    # 近似N_eff计算（保守估计）
    # tr(Σ) ≈ Σᵢ var[i]
    tr_approx = var.sum()
    
    # ||Σ||²_F ≈ Σᵢ var[i]² （忽略协方差的交叉项）
    # 注意：真实值 ||Σ||²_F = Σᵢⱼ Σᵢⱼ² 包含协方差，这里只计算对角项
    fro2_approx = (var**2).sum() + eps
    
    # N_eff = (tr Σ)² / ||Σ||²_F
    n_eff = (tr_approx**2 / fro2_approx).clamp(min=1.0)
    
    return float(n_eff)


class StreamingNEffCalculator:
    """
    流式N_eff计算器：支持增量更新和大N场景
    
    使用方法:
        calculator = StreamingNEffCalculator(momentum=0.9)
        for batch in data_loader:
            calculator.update(batch)
        n_eff = calculator.compute()
    """
    
    def __init__(self, momentum: float = 0.9, eps: float = 1e-9):
        self.momentum = momentum
        self.eps = eps
        self.reset()
    
    def reset(self):
        """重置计算器状态"""
        self.n = None
        self.m1 = None
        self.q2 = None
        self.batch_count = 0
        
    @torch.no_grad()
    def update(self, x_bits: torch.Tensor):
        """
        更新统计量
        
        Args:
            x_bits: [B,N] bool 激活位模式
        """
        if x_bits.numel() == 0:
            return
            
        x = x_bits.float()
        B, N = x.shape
        
        # 初始化
        if self.n is None:
            self.n = N
            self.m1 = torch.zeros(N, dtype=torch.float32, device=x.device)
            self.q2 = torch.zeros(N, dtype=torch.float32, device=x.device)
        elif N != self.n:
            raise ValueError(f"维度不一致: 期望{self.n}, 得到{N}")
        
        # 批内统计
        batch_mean = x.mean(dim=0)
        batch_q2 = (x**2).mean(dim=0)
        
        # EMA更新
        alpha = 1.0 - self.momentum
        self.m1 = self.momentum * self.m1 + alpha * batch_mean
        self.q2 = self.momentum * self.q2 + alpha * batch_q2
        
        self.batch_count += 1
    
    @torch.no_grad() 
    def compute(self) -> float:
        """
        计算当前的N_eff估计
        
        Returns:
            N_eff: 有效维度的保守估计
        """
        if self.batch_count == 0:
            return 1.0
        
        # 计算方差
        var = (self.q2 - self.m1**2).clamp_min(0.0)
        
        # 近似N_eff (保守估计)
        tr_approx = var.sum()
        fro2_approx = (var**2).sum() + self.eps
        
        n_eff = (tr_approx**2 / fro2_approx).clamp(min=1.0)
        return float(n_eff)
    
    def get_statistics(self) -> dict:
        """
        获取详细统计信息
        
        Returns:
            dict: 包含均值、方差、维度等信息
        """
        if self.batch_count == 0:
            return {"batch_count": 0, "n_eff": 1.0}
        
        var = (self.q2 - self.m1**2).clamp_min(0.0)
        
        return {
            "batch_count": self.batch_count,
            "n_eff": self.compute(),
            "mean_activation_rate": float(self.m1.mean()),
            "std_activation_rate": float(self.m1.std()),
            "mean_variance": float(var.mean()),
            "total_variance": float(var.sum()),
            "dimension": int(self.n) if self.n is not None else 0
        }

def estimate_cov(x_bits):
    """
    Estimates the covariance matrix from binary activations.
    :param x_bits: A numpy array of shape (batch_size, num_bits)
    :return: A numpy array of shape (num_bits, num_bits) representing the covariance matrix.
    """
    if x_bits is None or len(x_bits) == 0:
        return None
    # Ensure x_bits is a numpy array
    x_bits = np.asarray(x_bits, dtype=np.float32)
    return np.cov(x_bits, rowvar=False)

def n_eff_from_rho(N, rho_bar):
    """
    Calculates N_eff based on the equal correlation assumption.
    :param N: Total number of bits.
    :param rho_bar: Average correlation.
    :return: Effective number of bits.
    """
    if 1 + (N - 1) * rho_bar <= 0:
        return N # Avoid division by zero or negative values
    return N / (1 + (N - 1) * rho_bar)

def n_eff_robust_from_cov(cov_matrix):
    """
    Calculates a robust N_eff from the covariance matrix.
    N_eff = (sum(Var[Xi]))^2 / sum(Cov[Xi,Xj]^2)
    :param cov_matrix: A numpy array representing the covariance matrix.
    :return: Robust effective number of bits.
    """
    if cov_matrix is None:
        return 0
    # Sum of variances is the trace of the covariance matrix
    var_sum = np.trace(cov_matrix)
    if var_sum == 0:
        return 0

    # Sum of squared covariances is the sum of all elements squared
    cov_squared_sum = np.sum(cov_matrix ** 2)
    if cov_squared_sum == 0:
        return 0

    n_eff = (var_sum ** 2) / cov_squared_sum
    return n_eff
