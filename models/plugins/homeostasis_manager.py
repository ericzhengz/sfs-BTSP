# btsp/homeostasis_manager.py
"""
BTSP Homeostasis调用管理器

规范化homeostasis_step的调用时机，提供监控和日志功能
"""

from __future__ import annotations
import logging
import time
from typing import Dict, Any, Optional
from collections import defaultdict
import torch

class HomeostasisManager:
    """
    BTSP稳态调节管理器：规范化调用时机与监控
    
    功能：
    - 智能调用时机控制：基于batch数、epoch数或时间间隔
    - 详细监控日志：占用率、门控率统计和趋势分析
    - 自适应调整：根据系统稳定性动态调整调用频率
    - 性能优化：避免过于频繁的调用影响训练性能
    """
    
    def __init__(self,
                 btsp_memory,
                 call_interval_batches: int = 100,
                 call_every_epoch: bool = True,
                 adaptive_mode: bool = False,
                 logger: Optional[logging.Logger] = None):
        """
        初始化Homeostasis管理器
        
        Args:
            btsp_memory: BTSPMemory对象
            call_interval_batches: 每隔多少个batch调用一次
            call_every_epoch: 是否在每个epoch结束时调用
            adaptive_mode: 是否启用自适应调用频率
            logger: 日志器，None时创建默认日志器
        """
        self.btsp = btsp_memory
        self.call_interval_batches = call_interval_batches
        self.call_every_epoch = call_every_epoch
        self.adaptive_mode = adaptive_mode
        
        # 状态跟踪
        self.batch_count = 0
        self.epoch_count = 0
        self.last_call_time = time.time()
        self.call_history = []  # 存储历史调用结果
        
        # 日志设置
        if logger is None:
            self.logger = logging.getLogger('BTSP.Homeostasis')
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        else:
            self.logger = logger
    
    def on_batch_end(self, batch_idx: int, epoch: int, force: bool = False) -> Optional[Dict[str, float]]:
        """
        在batch结束时调用，自动判断是否需要执行homeostasis
        
        Args:
            batch_idx: 当前batch索引（从0开始）
            epoch: 当前epoch（从0开始）
            force: 强制执行homeostasis，忽略调用间隔
            
        Returns:
            稳态监控指标字典，如果未调用则返回None
        """
        self.batch_count += 1
        self.epoch_count = epoch
        
        should_call = force or (self.batch_count % self.call_interval_batches == 0)
        
        # 自适应模式：根据系统稳定性调整频率
        if self.adaptive_mode and len(self.call_history) >= 3:
            should_call = should_call or self._need_adaptive_call()
        
        if should_call:
            return self._execute_homeostasis(context=f"batch_{batch_idx}_epoch_{epoch}")
        
        return None
    
    def on_epoch_end(self, epoch: int) -> Optional[Dict[str, float]]:
        """
        在epoch结束时调用homeostasis
        
        Args:
            epoch: 当前epoch（从0开始）
            
        Returns:
            稳态监控指标字典，如果配置不调用则返回None
        """
        if self.call_every_epoch:
            return self._execute_homeostasis(context=f"epoch_end_{epoch}")
        
        return None
    
    def manual_call(self, context: str = "manual") -> Dict[str, float]:
        """
        手动调用homeostasis
        
        Args:
            context: 调用上下文标识，用于日志
            
        Returns:
            稳态监控指标字典
        """
        return self._execute_homeostasis(context=context)
    
    def _execute_homeostasis(self, context: str) -> Dict[str, float]:
        """执行homeostasis并记录监控信息"""
        start_time = time.time()
        
        # 执行稳态调节
        stats = self.btsp.homeostasis_step()
        
        # 记录执行时间
        execution_time = (time.time() - start_time) * 1000  # ms
        stats['execution_time_ms'] = execution_time
        stats['context'] = context
        stats['batch_count'] = self.batch_count
        stats['epoch'] = self.epoch_count
        
        # 保存历史记录（保留最近100次）
        self.call_history.append(stats)
        if len(self.call_history) > 100:
            self.call_history.pop(0)
        
        # 记录日志
        self._log_homeostasis_stats(stats, context)
        
        # 更新调用时间
        self.last_call_time = time.time()
        
        return stats
    
    def _log_homeostasis_stats(self, stats: Dict[str, float], context: str):
        """记录详细的homeostasis统计信息"""
        
        # 基础信息
        self.logger.info(f"[{context}] Homeostasis执行完成 (耗时: {stats['execution_time_ms']:.2f}ms)")
        
        # 占用率分析
        occ_status = "平衡" if abs(stats['occ_mean'] - stats['occ_target']) < 0.01 else "不平衡"
        self.logger.info(f"  占用率: {stats['occ_mean']:.4f}±{stats['occ_std']:.4f} "
                        f"(目标: {stats['occ_target']:.4f}) {occ_status}")
        
        # 门控率分析
        p_gate_range = stats['p_gate_max'] - stats['p_gate_min']
        p_gate_status = "均匀" if stats['p_gate_std'] < 0.1 else "分散"
        self.logger.info(f"  门控率: {stats['p_gate_mean']:.6f}±{stats['p_gate_std']:.6f} "
                        f"[{stats['p_gate_min']:.6f}, {stats['p_gate_max']:.6f}] {p_gate_status}")
        
        # 调整强度
        adjustment_status = ("大调整" if stats['adjustment_strength'] > 1.0 
                           else "小调整" if stats['adjustment_strength'] > 0.1 
                           else "稳定")
        self.logger.info(f"  调整强度: {stats['adjustment_strength']:.4f} {adjustment_status}")
        
        # 趋势分析（如果有历史数据）
        if len(self.call_history) >= 3:
            self._log_trend_analysis()
    
    def _log_trend_analysis(self):
        """分析最近几次调用的趋势"""
        if len(self.call_history) < 3:
            return
            
        recent = self.call_history[-3:]
        
        # 占用率趋势
        occ_means = [s['occ_mean'] for s in recent]
        occ_trend = "上升" if occ_means[-1] > occ_means[0] else "下降" if occ_means[-1] < occ_means[0] else "稳定"
        
        # 门控率稳定性
        p_gate_stds = [s['p_gate_std'] for s in recent]
        stability_trend = "分散" if p_gate_stds[-1] > p_gate_stds[0] else "收敛" if p_gate_stds[-1] < p_gate_stds[0] else "稳定"
        
        # 调整强度趋势
        adj_strengths = [s['adjustment_strength'] for s in recent]
        adj_trend = "增强" if adj_strengths[-1] > adj_strengths[0] else "减弱" if adj_strengths[-1] < adj_strengths[0] else "稳定"
        
        self.logger.info(f"  趋势: 占用率{occ_trend}, 门控稳定性{stability_trend}, 调整强度{adj_trend}")
    
    def _need_adaptive_call(self) -> bool:
        """自适应模式：判断是否需要额外调用"""
        if len(self.call_history) < 3:
            return False
            
        recent = self.call_history[-3:]
        
        # 如果占用率标准差持续上升，增加调用频率
        occ_stds = [s['occ_std'] for s in recent]
        if occ_stds[-1] > occ_stds[0] * 1.5:  # 标准差增加50%
            self.logger.debug("自适应触发: 占用率标准差上升")
            return True
        
        # 如果调整强度持续很大，增加调用频率
        adj_strengths = [s['adjustment_strength'] for s in recent]
        if all(s > 0.5 for s in adj_strengths):  # 连续大调整
            self.logger.debug("自适应触发: 持续大调整")
            return True
        
        return False
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """获取累计统计信息"""
        if not self.call_history:
            return {"message": "暂无homeostasis调用记录"}
        
        # 计算平均值和趋势
        occ_means = [s['occ_mean'] for s in self.call_history]
        occ_stds = [s['occ_std'] for s in self.call_history]
        p_gate_means = [s['p_gate_mean'] for s in self.call_history]
        p_gate_stds = [s['p_gate_std'] for s in self.call_history]
        adj_strengths = [s['adjustment_strength'] for s in self.call_history]
        exec_times = [s['execution_time_ms'] for s in self.call_history]
        
        return {
            "total_calls": len(self.call_history),
            "batch_count": self.batch_count,
            "avg_occ_mean": sum(occ_means) / len(occ_means),
            "avg_occ_std": sum(occ_stds) / len(occ_stds),
            "avg_p_gate_mean": sum(p_gate_means) / len(p_gate_means),
            "avg_p_gate_std": sum(p_gate_stds) / len(p_gate_stds),
            "avg_adjustment_strength": sum(adj_strengths) / len(adj_strengths),
            "avg_execution_time_ms": sum(exec_times) / len(exec_times),
            "latest_stats": self.call_history[-1] if self.call_history else None
        }
    
    def set_call_interval(self, interval_batches: int):
        """动态调整调用间隔"""
        self.call_interval_batches = max(1, interval_batches)
        self.logger.info(f"调用间隔已调整为: 每{self.call_interval_batches}个batch")
    
    def enable_adaptive_mode(self, enable: bool = True):
        """启用或禁用自适应模式"""
        self.adaptive_mode = enable
        status = "启用" if enable else "禁用"
        self.logger.info(f"自适应调用模式已{status}")
    
    def reset_statistics(self):
        """重置统计信息"""
        self.call_history.clear()
        self.batch_count = 0
        self.epoch_count = 0
        self.logger.info("Homeostasis统计信息已重置")


# 使用示例和最佳实践
def create_recommended_manager(btsp_memory, training_phase: str = "normal") -> HomeostasisManager:
    """
    创建推荐配置的Homeostasis管理器
    
    Args:
        btsp_memory: BTSPMemory对象
        training_phase: 训练阶段 ("initial", "normal", "fine_tune")
    
    Returns:
        配置好的HomeostasisManager
    """
    
    if training_phase == "initial":
        # 初始训练：较高频率，快速适应
        return HomeostasisManager(
            btsp_memory,
            call_interval_batches=50,
            call_every_epoch=True,
            adaptive_mode=True
        )
    elif training_phase == "normal":
        # 正常训练：标准频率
        return HomeostasisManager(
            btsp_memory,
            call_interval_batches=100,
            call_every_epoch=True,
            adaptive_mode=True
        )
    elif training_phase == "fine_tune":
        # 微调阶段：较低频率，保持稳定
        return HomeostasisManager(
            btsp_memory,
            call_interval_batches=200,
            call_every_epoch=False,
            adaptive_mode=False
        )
    else:
        raise ValueError(f"Unknown training phase: {training_phase}")


if __name__ == "__main__":
    # 使用示例
    from btsp import BTSPMemory
    
    # 创建BTSP记忆和管理器
    btsp = BTSPMemory(num_classes=10, num_bits=1024)
    manager = create_recommended_manager(btsp, "normal")
    
    # 模拟训练循环
    print("模拟训练循环...")
    for epoch in range(3):
        print(f"\nEpoch {epoch}")
        
        for batch_idx in range(150):
            # 模拟batch处理...
            # （这里应该是实际的训练逻辑）
            
            # batch结束时检查homeostasis
            stats = manager.on_batch_end(batch_idx, epoch)
            
            if stats:
                print(f"  Batch {batch_idx}: Homeostasis已执行")
        
        # epoch结束时的homeostasis
        epoch_stats = manager.on_epoch_end(epoch)
        if epoch_stats:
            print(f"  Epoch {epoch}: 结束时Homeostasis已执行")
    
    # 输出汇总统计
    summary = manager.get_summary_stats()
    print(f"\n训练汇总:")
    print(f"  总调用次数: {summary['total_calls']}")
    print(f"  平均占用率: {summary['avg_occ_mean']:.4f}±{summary['avg_occ_std']:.4f}")
    print(f"  平均门控率: {summary['avg_p_gate_mean']:.6f}±{summary['avg_p_gate_std']:.6f}")
    print(f"  平均执行时间: {summary['avg_execution_time_ms']:.2f}ms")
