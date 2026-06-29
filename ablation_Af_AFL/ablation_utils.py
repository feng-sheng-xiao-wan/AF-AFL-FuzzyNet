# -*- coding: utf-8 -*-
"""
通用的 AF/AFL 消融实验工具：
- 提供 run_ablation()，可按需关闭噪声分支 / 关闭模糊规则 / 关闭边界感知
- 内部通过 monkey patch 的方式替换 ECGDataset.__getitem__，以零向量替代噪声特征或模糊 logits
"""

import torch
import noise_aware_ecg_af_afl_simple as base
from train_af_vs_afl_afdb_ltafdb import run_training_binary


def run_ablation(
    afdb_csv: str = None,
    ltafdb_csv: str = None,
    combined_csv: str = None,
    output_dir: str = "outputs/ablation",
    *,
    disable_noise: bool = False,
    disable_fuzzy: bool = False,
    disable_morph: bool = False,
    disable_rr: bool = False,
    use_boundary_aware: bool = True,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "auto",
    use_oversampling: bool = True,
    oversample_target_ratio: float = 1.0,
    oversample_strategy: str = "random",
    use_precomputed: bool = True,
    recompute_fuzzy_only: bool = False,
    **kwargs,
):
    """
    运行一次消融实验。
    Args:
        afdb_csv: AFDB 分段 CSV（向后兼容，如果 combined_csv 提供则忽略）
        ltafdb_csv: LTAFDB 分段 CSV（向后兼容，如果 combined_csv 提供则忽略）
        combined_csv: 合并的 CSV 文件路径（推荐，包含预计算特征）
        output_dir: 输出目录
        disable_noise: True 时噪声特征置零（等价于关闭噪声分支）
        disable_fuzzy: True 时模糊 logits 置零（等价于关闭模糊规则融合）
        disable_morph: True 时将 ECG 波形置零（等价于关闭形态分支）
        disable_rr: True 时将 RR 序列置零（等价于关闭节律分支）
        use_boundary_aware: 是否启用边界感知
        epochs, batch_size, lr: 训练超参
        device: 训练设备（auto/cpu/cuda/cuda:0）
        use_oversampling: 是否使用过采样
        oversample_target_ratio: 过采样目标比例（1.0=完全平衡）
        oversample_strategy: 过采样策略（"random" 或 "patient_level"）
        use_precomputed: 是否使用预计算特征
        recompute_fuzzy_only: 是否仅重新计算模糊规则特征
        **kwargs: 其他传递给 run_training_binary 的参数
    """
    # 备份原始数据集类
    OriginalDataset = base.ECGDataset

    class PatchedDataset(base.ECGDataset):
        def __getitem__(self, idx):
            batch = super().__getitem__(idx)
            if disable_morph:
                batch["ecg"] = torch.zeros_like(batch["ecg"])
            if disable_rr:
                batch["rr_seq"] = torch.zeros_like(batch["rr_seq"])
            if disable_noise:
                batch["noise_feat"] = torch.zeros_like(batch["noise_feat"])
            if disable_fuzzy:
                batch["fuzzy_logits"] = torch.zeros_like(batch["fuzzy_logits"])
            return batch

    # monkey patch
    base.ECGDataset = PatchedDataset

    try:
        # 确定使用哪个CSV
        if combined_csv:
            # 使用合并的CSV（推荐）
            csv_arg = combined_csv
            ltafdb_arg = ""
        else:
            # 使用分离的CSV（向后兼容）
            csv_arg = afdb_csv
            ltafdb_arg = ltafdb_csv or ""
        
        run_training_binary(
            csv_arg,
            ltafdb_arg,
            output_dir=output_dir,
            batch_size=batch_size,
            lr=lr,
            epochs=epochs,
            device=device,
            use_boundary_aware=use_boundary_aware,
            validate_every_batch=False,  # ablation 加速
            use_oversampling=use_oversampling,
            oversample_target_ratio=oversample_target_ratio,
            oversample_strategy=oversample_strategy,
            use_precomputed=use_precomputed,
            recompute_fuzzy_only=recompute_fuzzy_only,
            **kwargs,
        )
    finally:
        # 恢复原始 Dataset
        base.ECGDataset = OriginalDataset

