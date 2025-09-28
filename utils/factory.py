def _arg_get(args, key, default=None):
    if isinstance(args, dict):
        return args.get(key, default)
    return getattr(args, key, default)

def get_model(model_name, args):
    name = model_name.lower()
    if name == "simplecil":
        from models.simplecil import Learner
    elif name == "aper_finetune":
        from models.aper_finetune import Learner
    elif name == "aper_ssf":
        from models.aper_ssf import Learner
    elif name == "aper_vpt":
        from models.aper_vpt import Learner 
    elif name == "aper_adapter":
        from models.aper_adapter import Learner
    elif name == "l2p":
        from models.l2p import Learner
    elif name == "dualprompt":
        from models.dualprompt import Learner
    elif name == "coda_prompt":
        from models.coda_prompt import Learner
    elif name == "finetune":
        from models.finetune import Learner
    elif name == "icarl":
        from models.icarl import Learner
    elif name == "der":
        from models.der import Learner
    elif name == "coil":
        from models.coil import Learner
    elif name == "foster":
        from models.foster import Learner
    elif name == "memo":
        from models.memo import Learner
    elif name == 'ranpac':
        from models.ranpac import Learner
    elif name == "ease":
        from models.ease import Learner
    elif name == 'slca':
        from models.slca import Learner
    elif name == 'lae':
        from models.lae import Learner
    elif name == 'fecam':
        from models.fecam import Learner
    elif name == 'dgr':
        from models.dgr import Learner
    elif name == 'mos':
        from models.mos import Learner
    elif name == 'cofima':
        from models.cofima import Learner
    elif name == 'duct':
        from models.duct import Learner
    elif name == 'tuna':
        from models.tuna import Learner
    else:
        assert 0
    
    # Build baseline model
    inner = Learner(args)
    
    # Wrap with BTSP plugin if enabled
    if _arg_get(args, "use_btsp", False):
        from models.plugins.btsp_plugins import BTSPPlugin
        from models.plugins.unified_metrics import ResourceBudget
        
        # 1) 移除强制nb_classes要求，让BTSPPlugin的智能推断逻辑处理
        num_classes = _arg_get(args, "nb_classes", _arg_get(args, "init_cls", None))
        # 注意：不再强制要求num_classes，BTSPPlugin会通过data_manager推断

        # 2) 统一 tau_e_steps 参数命名（兼容旧键 btsp_tau_e）
        tau_e_steps = _arg_get(args, "btsp_tau_e_steps", _arg_get(args, "btsp_tau_e", 4.0))

        # 3) 鲁棒获取特征维度：尝试 feature_dim / out_dim / 配置 / 默认 768
        net = getattr(inner, "_network", None)
        feat_dim = None
        if net is not None:
            feat_dim = getattr(net, "feature_dim", None)
            if feat_dim is None:
                feat_dim = getattr(net, "out_dim", None)
        if feat_dim is None:
            feat_dim = _arg_get(args, "feat_dim", 768)
            
        # 4) 新增：事件驱动控制系统参数
        control_mode = _arg_get(args, "btsp_control_mode", "A")  # A=Analysis-only, B=Intervention
        target_epsilon = _arg_get(args, "btsp_target_epsilon", 0.1)
        enable_unified_protocol = _arg_get(args, "btsp_enable_unified_protocol", True)
        
        # 5) 资源预算配置
        resource_budget = None
        if enable_unified_protocol:
            resource_budget = ResourceBudget(
                max_flops=_arg_get(args, "budget_max_flops", 1e12),
                max_memory_mb=_arg_get(args, "budget_max_memory_mb", 8192),
                max_bytes=_arg_get(args, "budget_max_bytes", 1e9),
                max_train_time=_arg_get(args, "budget_max_train_time", 3600),
                max_infer_time=_arg_get(args, "budget_max_infer_time", 100)
            )

        return BTSPPlugin(
            inner,
            num_classes=num_classes,
            feat_dim=feat_dim,
            args=args,  # 🔥 传递完整的args，让BTSPPlugin自己解析参数
            # 保留关键参数的显式传递以保证兼容性
            N_bits=_arg_get(args, "btsp_N_bits", 8192),
            topk=_arg_get(args, "btsp_topk", 256),
            theta=_arg_get(args, "btsp_theta", 0.2),
            tau_e_steps=tau_e_steps,
            branches=_arg_get(args, "btsp_branches", 8),
            alpha=_arg_get(args, "btsp_alpha", 0.5),
            homeo_interval=_arg_get(args, "btsp_homeo_interval", 100),
            zstats_interval=_arg_get(args, "btsp_zstats_interval", 30),
            # 新增事件驱动控制参数
            control_mode=control_mode,
            target_epsilon=target_epsilon,
            enable_unified_protocol=enable_unified_protocol,
            resource_budget=resource_budget,
            # 实验参数
            experiment_logging=_arg_get(args, "btsp_enable_unified_logging", False),
            log_file=_arg_get(args, "btsp_log_file", None)
        )
    return inner