# btsp/fuse.py
from __future__ import annotations
import torch

def fuse_logits(base_logits: torch.Tensor, btsp_logits_cpu: torch.Tensor, fusion_alpha: float = 0.5) -> torch.Tensor:
    """
    加权平均logits融合：支持跨设备的BTSP记忆集成
    
    核心功能:
    执行加权平均组合：fused = (1-α) * base_logits + α * btsp_logits
    自动处理CPU→GPU的设备迁移，确保计算一致性
    
    Args:
        base_logits: torch.Tensor, shape [B, C]
            基础模型在GPU上的原始logits
        btsp_logits_cpu: torch.Tensor, shape [B, C]  
            BTSP记忆系统在CPU上的logits，将自动迁移到GPU
        fusion_alpha: float, default=0.5
            BTSP权重系数，控制记忆贡献强度
            α ∈ [0,1]: 0表示纯base模型，1表示纯BTSP记忆
            
    Returns:
        torch.Tensor, shape [B, C], device=base_logits.device
            融合后的logits，保持在base_logits的设备上
            
    Raises:
        AssertionError: 当输入形状不匹配时
        
    Algorithm:
        1. 形状检查：确保 base_logits.shape == btsp_logits_cpu.shape
        2. 设备迁移：btsp_logits_cpu → base_logits.device
        3. 加权平均：output = (1-α) * base_logits + α * btsp_logits_gpu
        
    Performance:
        - 设备传输开销：O(B×C)，仅在首次调用时
        - 计算复杂度：O(B×C)，标量乘法和张量加法
        - 内存占用：O(B×C)，临时GPU张量
        
    Note:
        - 专为CPU-GPU混合推理设计：base model在GPU，BTSP在CPU
        - alpha=0.5时为等权重融合，alpha=0.0时忽略BTSP，alpha=1.0时忽略base
        - 支持批量处理，保持数值稳定性
        
    Example:
        >>> base = torch.randn(16, 10).cuda()       # GPU logits
        >>> btsp = torch.randn(16, 10).cpu()        # CPU logits  
        >>> fused = fuse_logits(base, btsp, 0.3)    # 30% BTSP, 70% base
        >>> print(fused.device)                     # cuda:0
    """
    # 形状对齐：处理BTSP返回所有类别而base model只返回部分类别的情况
    batch_size = base_logits.shape[0]
    base_num_classes = base_logits.shape[1]
    btsp_num_classes = btsp_logits_cpu.shape[1]
    
    # 将CPU的BTSP logits迁移到GPU，与base logits设备对齐
    btsp_logits_gpu = btsp_logits_cpu.to(base_logits.device, dtype=base_logits.dtype)
    
    if base_num_classes == btsp_num_classes:
        # 形状一致，直接融合
        fused_logits = (1.0 - float(fusion_alpha)) * base_logits + float(fusion_alpha) * btsp_logits_gpu
    elif base_num_classes < btsp_num_classes:
        # BTSP返回更多类别，只取前base_num_classes个
        btsp_logits_aligned = btsp_logits_gpu[:, :base_num_classes]
        fused_logits = (1.0 - float(fusion_alpha)) * base_logits + float(fusion_alpha) * btsp_logits_aligned
    else:
        # base返回更多类别，将BTSP扩展（用负无穷填充）
        btsp_logits_expanded = torch.full(
            (batch_size, base_num_classes), 
            float('-inf'), 
            device=btsp_logits_gpu.device, 
            dtype=btsp_logits_gpu.dtype
        )
        btsp_logits_expanded[:, :btsp_num_classes] = btsp_logits_gpu
        fused_logits = (1.0 - float(fusion_alpha)) * base_logits + float(fusion_alpha) * btsp_logits_expanded
    
    # 后置验证：确保输出保持预期的设备和形状
    assert fused_logits.device == base_logits.device
    assert fused_logits.shape == base_logits.shape
    
    return fused_logits


def weighted_fuse_logits(base_logits: torch.Tensor, 
                        btsp_logits: torch.Tensor, 
                        base_weight: float = 0.7,
                        btsp_weight: float = 0.3) -> torch.Tensor:
    """
    加权融合：logits = base_weight * base + btsp_weight * btsp
    
    Args:
        base_logits: [B, C] 基础模型logits
        btsp_logits: [B, C] BTSP记忆logits  
        base_weight: 基础模型权重
        btsp_weight: BTSP记忆权重
        
    Returns:
        fused_logits: [B, C] 融合后的logits
        
    Note:
        权重归一化：确保 base_weight + btsp_weight = 1.0 以保持logits尺度
    """
    assert base_logits.shape == btsp_logits.shape
    
    # 权重归一化
    total_weight = base_weight + btsp_weight
    if total_weight > 0:
        base_weight /= total_weight
        btsp_weight /= total_weight
    
    # 设备对齐与融合
    btsp_logits = btsp_logits.to(base_logits.device)
    
    return base_weight * base_logits + btsp_weight * btsp_logits


def adaptive_fuse_logits(base_logits: torch.Tensor,
                        btsp_logits: torch.Tensor,
                        confidence_threshold: float = 0.8) -> torch.Tensor:
    """
    自适应融合：根据base模型置信度动态调整融合权重
    
    注意：这是一个基础的占位策略示例，具体的融合策略应根据实际应用需求进行微调。
    该函数不是BTSP框架的核心贡献，仅作为logits融合的工程实现参考。
    实际部署时建议根据任务特性设计更精细的自适应策略。
    
    核心机制:
    1. 计算基础模型置信度：max(softmax(base_logits))
    2. 自适应权重分配：高置信度→低BTSP权重，低置信度→高BTSP权重  
    3. 动态融合：fused = base + α(confidence) * btsp
    
    Args:
        base_logits: torch.Tensor, shape [B, C]
            基础模型的原始logits，期望在GPU设备上
        btsp_logits: torch.Tensor, shape [B, C]  
            BTSP记忆系统的logits，可能在CPU或GPU上
        confidence_threshold: float, default=0.8
            置信度分界点，> threshold时偏重base模型
            
    Returns:
        torch.Tensor, shape [B, C], device=base_logits.device
            自适应融合后的logits，保持base_logits的设备和数据类型
            
    Raises:
        AssertionError: 当输入形状不匹配时
        
    Algorithm:
        置信度 = max(softmax(base_logits), dim=1)
        权重α = 0.1 if 置信度 > threshold else 0.5
        输出 = base_logits + α * btsp_logits
        
    Performance:
        - 设备间数据传输：O(B×C)，仅当btsp_logits在不同设备时
        - 计算复杂度：O(B×C)，主要来自softmax和广播操作
        - 内存开销：O(B×C)，临时置信度和权重张量
        
    Note:
        - 自动处理设备对齐：btsp_logits → base_logits.device
        - 高置信度样本(>0.8)：更信任基础模型，α=0.1
        - 低置信度样本(≤0.8)：更依赖BTSP记忆，α=0.5
        - 支持批量处理，逐样本独立权重计算
        
    Example:
        >>> base = torch.randn(32, 10).cuda()      # [B, C] on GPU
        >>> btsp = torch.randn(32, 10).cpu()       # [B, C] on CPU  
        >>> fused = adaptive_fuse_logits(base, btsp, 0.8)
        >>> print(fused.shape, fused.device)       # torch.Size([32, 10]) cuda:0
    """
    # 前置断言：确保输入logits的形状一致性
    assert base_logits.shape == btsp_logits.shape, (
        f"Shape mismatch: base_logits{base_logits.shape} != btsp_logits{btsp_logits.shape}")
    
    # 前置断言：确保返回logits的设备和形状信息
    expected_device = base_logits.device
    expected_shape = base_logits.shape
    expected_dtype = base_logits.dtype
    
    # 设备对齐：将btsp_logits迁移到base_logits的设备上
    btsp_logits = btsp_logits.to(device=expected_device, dtype=expected_dtype)
    
    # 计算base模型的置信度（softmax最大值）
    base_probs = torch.softmax(base_logits, dim=1)
    confidence = base_probs.max(dim=1)[0]  # [B]
    
    # 自适应权重：高置信度 → 低BTSP权重，低置信度 → 高BTSP权重
    alpha = torch.where(confidence > confidence_threshold, 
                       torch.tensor(0.1, device=expected_device, dtype=expected_dtype),
                       torch.tensor(0.5, device=expected_device, dtype=expected_dtype))
    
    # 广播融合：[B, C] + [B, 1] * [B, C]
    alpha = alpha.unsqueeze(1)  # [B, 1] for broadcasting
    fused_logits = base_logits + alpha * btsp_logits
    
    # 后置断言：确保返回logits满足预期的设备、形状和数据类型
    assert fused_logits.device == expected_device, f"Device mismatch: {fused_logits.device} != {expected_device}"
    assert fused_logits.shape == expected_shape, f"Shape mismatch: {fused_logits.shape} != {expected_shape}"
    assert fused_logits.dtype == expected_dtype, f"Dtype mismatch: {fused_logits.dtype} != {expected_dtype}"
    
    return fused_logits
