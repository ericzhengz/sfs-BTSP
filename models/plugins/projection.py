# btsp/projection.py
"""
持久化投影矩阵管理器 - 适用于长期训练和生产部署

设计理念：
- 面向对象设计，状态自动管理
- 投影矩阵作为buffer自动保存/恢复
- 设备迁移和状态同步自动处理

适用场景：
长期训练和生产部署
需要保存/恢复投影状态的场景
多设备间的自动状态同步
规模化部署和模型检查点

vs proj_bin函数：
- OrthProjection: 类，持久，自动状态管理与保存/恢复
- proj_bin: 函数，轻量，手动管理W矩阵

推荐使用：
```python
# 长期训练
proj = OrthProjection(D=768, N=1024, device="cuda")
x_bits = proj(features, topk=128)

# 保存模型时投影矩阵自动包含在state_dict中
torch.save(model.state_dict(), "checkpoint.pth")
```

设计优势：
- 自动设备迁移：feat.to(self.W.device)确保计算一致性
- 持久化支持：register_buffer使W矩阵随模型保存/加载
- 实验重现：reinitialize()支持可控的随机种子重置
"""
from __future__ import annotations
import torch
from torch import nn
import math


class OrthProjection(nn.Module):
    """
    正交投影矩阵管理器：生产级的持久化投影解决方案
    
    核心功能：
    - 投影矩阵W持久化：随模型自动保存/加载，确保跨训练会话一致性
    - 自动设备管理：输入特征自动迁移到投影矩阵设备，避免设备不匹配
    - 严格Top-k二值化：索引掩码确保精确k位激活，避免并列值干扰
    - 实验重现支持：可控随机种子重置，保证结果可复现
    
    vs proj_bin函数的区别：
    - 状态管理：类自动管理投影矩阵生命周期 vs 函数需手动传入W
    - 持久化：buffer自动保存/恢复 vs 需手动保存W矩阵
    - 设备处理：自动迁移 vs 需手动确保设备一致性
    - 接口复杂度：稍重 vs 轻量级
    
    适合长期训练、生产部署和需要模型检查点的场景。
    """
    
    def __init__(self, 
                 D: int, 
                 N: int,
                 init: str = "gauss",
                 device: str = "cuda") -> None:
        super().__init__()
        
        self.D = D  # 输入特征维度
        self.N = N  # 输出比特维度
        
        # 创建投影矩阵W: [D, N] - 注册为buffer，不参与梯度但会保存/加载
        W = torch.empty(D, N, device=device, dtype=torch.float32)
        if init == "gauss":
            W.normal_(0, 1.0 / (D ** 0.5))  # 高斯初始化
        elif init == "orthogonal":
            nn.init.orthogonal_(W)  # 正交初始化
        else:
            raise ValueError(f"Unknown initialization: {init}")
            
        self.register_buffer("W", W)
        
    def forward(self, feat: torch.Tensor, topk: int) -> torch.Tensor:
        """
        前向传播：特征投影 + 严格Top-k二值化
        
        Args:
            feat: [B,D] 输入特征（自动迁移到W的设备）
            topk: int Top-k稀疏激活数量
            
        Returns:
            x_bits: [B,N] bool 二进制激活矩阵
        """
        assert feat.dim() == 2 and feat.size(1) == self.D
        
        # 投影: [B,D] @ [D,N] -> [B,N]（自动在W的设备上计算）
        x = feat.to(self.W.device) @ self.W
        
        if topk and topk > 0:
            k = min(topk, x.size(1))
            # 严格 k 位：用索引掩码避免并列值导致的激活数 ≠ k
            vals, idx = torch.topk(x, k, dim=1)
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask.scatter_(1, idx, True)   # 严格 k 个 True
            return mask
        
        # 符号门限：p_pre ≈ 0.5
        return (x > 0)
        
    def get_sparsity_rate(self, topk: int) -> float:
        """计算稀疏率 p_pre = topk / N"""
        return topk / self.N if topk > 0 else 0.5
    
    def get_projection_matrix(self) -> torch.Tensor:
        """获取投影矩阵W"""
        return self.W
    
    def reinitialize(self, init: str | None = None, seed: int | None = None) -> None:
        """
        重新初始化投影矩阵（用于实验重现）
        
        Args:
            init: 初始化方法，None时使用原方法
            seed: 随机种子
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        if init == "gauss" or init is None:
            self.W.normal_(0, 1.0 / (self.D ** 0.5))
        elif init == "orthogonal":
            nn.init.orthogonal_(self.W)
        else:
            raise ValueError(f"Unknown initialization: {init}")
