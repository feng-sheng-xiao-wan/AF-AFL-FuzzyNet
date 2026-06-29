#!/usr/bin/env python3
"""
测试保存的模型
用法: python test_saved_model.py --checkpoint <模型路径> [--test_csv <测试集CSV>] [--device <设备>]
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 导入训练脚本中的必要组件
from noise_aware_ecg_af_afl_simple import (
    ECGNet,
    ECGDataset,
    evaluate,
    compute_confusion_matrix,
    get_device,
    split_dataset_csv,
    TrainConfig,
)


def load_checkpoint(checkpoint_path: str, device: str):
    """加载checkpoint并返回模型、配置和元数据"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # PyTorch 2.6+ requires weights_only=False for checkpoints containing numpy arrays
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 恢复配置
    cfg_dict = checkpoint.get("cfg", {})
    cfg = TrainConfig(**cfg_dict)
    
    # 获取元数据
    val_f1 = checkpoint.get("val_macroF1", -1.0)
    epoch = checkpoint.get("epoch", "unknown")
    adaptive_params = checkpoint.get("adaptive_params", {})
    
    print(f"  ✓ Checkpoint loaded")
    print(f"    - Saved at epoch: {epoch}")
    print(f"    - Val F1: {val_f1:.4f}")
    
    if adaptive_params:
        print(f"    - Adaptive parameters:")
        print(f"      * alpha_afl_max: {adaptive_params.get('adaptive_alpha_afl_max', 'N/A'):.3f}")
        print(f"      * af_protect_coef: {adaptive_params.get('adaptive_af_protect_coef', 'N/A'):.3f}")
        print(f"      * af_protect_threshold: {adaptive_params.get('adaptive_af_protect_threshold', 'N/A'):.3f}")
        print(f"      * alpha_afl_boost_coef: {adaptive_params.get('adaptive_alpha_afl_boost_coef', 'N/A'):.3f}")
    
    return checkpoint, cfg, val_f1, epoch, adaptive_params


def build_model(cfg: TrainConfig, device: str, noise_dim: int, rr_len: int):
    """根据配置构建模型"""
    print(f"\nBuilding model...")
    print(f"  - num_leads: {cfg.num_leads}")
    print(f"  - num_arrhythmia_classes: {cfg.num_arrhythmia_classes}")
    print(f"  - noise_feat_dim: {noise_dim}")
    print(f"  - rr_seq_len: {rr_len}")
    
    model = ECGNet(
        num_leads=cfg.num_leads,
        num_arrhythmia_classes=cfg.num_arrhythmia_classes,
        noise_feat_dim=noise_dim,
        rr_seq_len=rr_len,
        d_model=128,
        rhythm_emb_dim=64,
        noise_emb_dim=32,
        fuzzy_weight_init=cfg.fuzzy_weight_init,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total params: {total_params:,}")
    
    return model


def load_model_weights(model, checkpoint: dict, device: str):
    """加载模型权重"""
    print(f"\nLoading model weights...")
    state_dict = checkpoint["model"]
    
    # 过滤掉debug相关的buffer（这些在测试时不需要）
    filtered_state_dict = {}
    debug_keys = ["_debug_alpha", "_debug_fuzzy_af", "_debug_fuzzy_afl", "_debug_net_af", 
                  "_debug_net_afl", "_debug_fused_af", "_debug_fused_afl", "_debug_alpha_afl",
                  "_debug_alpha_raw"]
    
    for key, value in state_dict.items():
        # 跳过debug相关的键
        if any(debug_key in key for debug_key in debug_keys):
            continue
        filtered_state_dict[key] = value
    
    # 使用strict=False允许部分匹配（忽略debug keys）
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    if missing_keys:
        print(f"  Warning: Missing keys (will use default values): {len(missing_keys)} keys")
        if len(missing_keys) <= 5:
            for key in missing_keys:
                print(f"    - {key}")
        else:
            for key in missing_keys[:5]:
                print(f"    - {key}")
            print(f"    ... and {len(missing_keys) - 5} more")
    
    if unexpected_keys:
        # 只显示非debug的unexpected keys
        non_debug_unexpected = [k for k in unexpected_keys if not any(dk in k for dk in debug_keys)]
        if non_debug_unexpected:
            print(f"  Warning: Unexpected keys (ignored): {len(non_debug_unexpected)} keys")
            if len(non_debug_unexpected) <= 5:
                for key in non_debug_unexpected:
                    print(f"    - {key}")
            else:
                for key in non_debug_unexpected[:5]:
                    print(f"    - {key}")
                print(f"    ... and {len(non_debug_unexpected) - 5} more")
    
    model.eval()
    print(f"  ✓ Model weights loaded successfully")
    return model


def print_detailed_metrics(metrics: dict, num_classes: int):
    """打印详细的评估指标"""
    print("\n" + "=" * 80)
    print("Detailed Evaluation Results")
    print("=" * 80)
    
    print(f"\nOverall Metrics:")
    print(f"  - Loss: {metrics['loss']:.4f}")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  - Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  - Macro Recall: {metrics['macro_recall']:.4f}")
    
    if num_classes == 2:
        # 二分类：AF vs AFL
        cm = metrics.get('confusion_matrix')
        if cm is not None:
            print(f"\nConfusion Matrix:")
            print(f"            Pred: AF  AFL")
            print(f"  True: AF  [{cm[0,0]:4d} {cm[0,1]:4d}]")
            print(f"       AFL  [{cm[1,0]:4d} {cm[1,1]:4d}]")
            
            # 计算每个类别的详细指标
            af_tp = cm[0, 0]
            af_fn = cm[0, 1]
            af_fp = cm[1, 0]
            afl_tp = cm[1, 1]
            afl_fn = cm[1, 0]
            afl_fp = cm[0, 1]
            
            af_recall = af_tp / max(af_tp + af_fn, 1)
            af_precision = af_tp / max(af_tp + af_fp, 1)
            af_f1 = 2 * af_recall * af_precision / max(af_recall + af_precision, 1e-8)
            
            afl_recall = afl_tp / max(afl_tp + afl_fn, 1)
            afl_precision = afl_tp / max(afl_tp + afl_fp, 1)
            afl_f1 = 2 * afl_recall * afl_precision / max(afl_recall + afl_precision, 1e-8)
            
            print(f"\nPer-Class Metrics:")
            print(f"  AF:")
            print(f"    - Recall: {af_recall:.4f} ({af_tp}/{af_tp + af_fn})")
            print(f"    - Precision: {af_precision:.4f} ({af_tp}/{af_tp + af_fp})")
            print(f"    - F1: {af_f1:.4f}")
            print(f"  AFL:")
            print(f"    - Recall: {afl_recall:.4f} ({afl_tp}/{afl_tp + afl_fn})")
            print(f"    - Precision: {afl_precision:.4f} ({afl_tp}/{afl_tp + afl_fp})")
            print(f"    - F1: {afl_f1:.4f}")
            
            print(f"\nError Analysis:")
            print(f"  - AF误分为AFL: {af_fn} ({af_fn/(af_tp+af_fn)*100:.1f}%)")
            print(f"  - AFL误分为AF: {afl_fp} ({afl_fp/(afl_tp+afl_fp)*100:.1f}%)")
    
    elif num_classes == 4:
        # 四分类：AF/AFL/PSVT/Normal
        cm = metrics.get('confusion_matrix')
        if cm is not None:
            print(f"\nConfusion Matrix:")
            print(f"            Pred:   AF  AFL PSVT  NOR")
            print(f"  True: AF   [{cm[0,0]:4d} {cm[0,1]:4d} {cm[0,2]:4d} {cm[0,3]:4d}]")
            print(f"       AFL   [{cm[1,0]:4d} {cm[1,1]:4d} {cm[1,2]:4d} {cm[1,3]:4d}]")
            print(f"       PSVT  [{cm[2,0]:4d} {cm[2,1]:4d} {cm[2,2]:4d} {cm[2,3]:4d}]")
            print(f"       NOR   [{cm[3,0]:4d} {cm[3,1]:4d} {cm[3,2]:4d} {cm[3,3]:4d}]")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Test a saved ECG arrhythmia classification model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="",
        help="Path to test CSV file (if not provided, will use data_csv and split)",
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        default="",
        help="Path to full data CSV (will be split to get test set if test_csv not provided)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, or cuda:0)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for testing",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--no_use_precomputed",
        action="store_true",
        help="Disable precomputed features (will compute all features on-the-fly)",
    )
    parser.add_argument(
        "--recompute_fuzzy_only",
        action="store_true",
        help="Recompute only fuzzy_logits (other features still use precomputed files)",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ECG Arrhythmia Model Testing")
    print("=" * 80)
    
    # 获取设备
    device = get_device(args.device)
    print(f"\n[1/5] Device: {device}")
    
    # 加载checkpoint
    print(f"\n[2/5] Loading checkpoint...")
    checkpoint, cfg, val_f1, epoch, adaptive_params = load_checkpoint(args.checkpoint, device)
    
    # 准备测试数据
    print(f"\n[3/5] Preparing test data...")
    if args.test_csv:
        # 直接使用提供的测试CSV
        test_csv = args.test_csv
        print(f"  - Using provided test CSV: {test_csv}")
    elif args.data_csv:
        # 从完整数据集中划分测试集（使用相同的随机种子）
        print(f"  - Splitting dataset from: {args.data_csv}")
        _, _, test_df = split_dataset_csv(args.data_csv, 0.8, 0.1, 0.1, seed=42)
        test_csv = os.path.join("outputs", "temp_test.csv")
        os.makedirs("outputs", exist_ok=True)
        test_df.to_csv(test_csv, index=False)
        print(f"  - Test set saved to: {test_csv}")
    elif cfg.data_csv:
        # 使用配置中的data_csv
        print(f"  - Splitting dataset from config: {cfg.data_csv}")
        _, _, test_df = split_dataset_csv(cfg.data_csv, 0.8, 0.1, 0.1, seed=42)
        test_csv = os.path.join("outputs", "temp_test.csv")
        os.makedirs("outputs", exist_ok=True)
        test_df.to_csv(test_csv, index=False)
        print(f"  - Test set saved to: {test_csv}")
    else:
        raise ValueError("No test data provided. Use --test_csv, --data_csv, or ensure checkpoint has data_csv in config.")
    
    # 创建测试数据集
    # 如果指定了 --no_use_precomputed，覆盖配置中的设置
    use_precomputed = cfg.use_precomputed if not args.no_use_precomputed else False
    if args.no_use_precomputed:
        print(f"  - Precomputed features disabled (will compute all features on-the-fly)")
    elif args.recompute_fuzzy_only:
        print(f"  - Recomputing fuzzy_logits only (other features use precomputed files)")
    
    test_ds = ECGDataset(
        test_csv,
        max_len=cfg.max_len,
        num_leads=cfg.num_leads,
        max_rr_intervals=cfg.max_rr_intervals,
        use_stft_dbscan=cfg.use_stft_dbscan,
        multi_lead_method=cfg.multi_lead_method,
        flutter_method=cfg.flutter_method,
        lead_names=cfg.lead_names,
        use_precomputed=use_precomputed,
        recompute_fuzzy_only=args.recompute_fuzzy_only,
    )
    
    # 获取数据维度
    sample = test_ds[0]
    noise_dim = sample["noise_feat"].shape[0]
    rr_len = sample["rr_seq"].shape[0]
    
    print(f"  - Test samples: {len(test_ds)}")
    print(f"  - Noise feat dim: {noise_dim}")
    print(f"  - RR seq len: {rr_len}")
    
    # 创建测试数据加载器
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    
    # 构建模型
    print(f"\n[4/5] Building model...")
    model = build_model(cfg, device, noise_dim, rr_len)
    
    # 加载模型权重
    model = load_model_weights(model, checkpoint, device)
    
    # 评估模型
    print(f"\n[5/5] Evaluating model on test set...")
    test_metrics = evaluate(model, test_loader, device, verbose=True)
    
    # 打印详细结果
    print_detailed_metrics(test_metrics, cfg.num_arrhythmia_classes)
    
    # 对比验证集和测试集性能
    print(f"\nPerformance Comparison:")
    print(f"  - Validation F1 (from checkpoint): {val_f1:.4f}")
    print(f"  - Test F1: {test_metrics['macro_f1']:.4f}")
    print(f"  - Difference: {test_metrics['macro_f1'] - val_f1:+.4f}")
    
    print("\n" + "=" * 80)
    print("Testing completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

