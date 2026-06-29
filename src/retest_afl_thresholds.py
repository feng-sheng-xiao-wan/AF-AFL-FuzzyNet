import os
import argparse
import numpy as np
import torch

import noise_aware_ecg_af_afl_simple as base


def main():
    parser = argparse.ArgumentParser(description="Retest a saved checkpoint under different AFL probability thresholds.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.10, 0.15, 0.20, 0.30, 0.50],
        help="AFL probability thresholds to test (pred=AFL if P(AFL)>=t).",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda:0 ...")
    parser.add_argument("--no_record_agg", action="store_true", default=True, help="Force segment-level eval (default).")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    device = base.get_device(args.device)
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg_dict = checkpoint.get("cfg", {})

    # Build TrainConfig from checkpoint cfg (fallbacks for safety)
    cfg = base.TrainConfig(**cfg_dict) if isinstance(cfg_dict, dict) else base.TrainConfig()
    if not cfg.data_csv:
        raise ValueError("Checkpoint cfg does not contain data_csv; cannot rebuild test split.")

    print("=" * 80)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Data CSV: {cfg.data_csv}")
    print(f"Device: {device}")
    print("=" * 80)

    # Rebuild split (same default ratios as training)
    train_df, val_df, test_df = base.split_dataset_csv(cfg.data_csv, 0.8, 0.1, 0.1)
    if len(test_df) == 0:
        raise ValueError("Test split is empty.")

    output_dir = os.path.dirname(args.ckpt) or "outputs"
    temp_test_csv = os.path.join(output_dir, "temp_test_retest.csv")
    test_df.to_csv(temp_test_csv, index=False)

    test_ds = base.ECGDataset(
        temp_test_csv,
        max_len=cfg.max_len,
        num_leads=cfg.num_leads,
        max_rr_intervals=cfg.max_rr_intervals,
        use_stft_dbscan=cfg.use_stft_dbscan,
        multi_lead_method=cfg.multi_lead_method,
        flutter_method=cfg.flutter_method,
        lead_names=cfg.lead_names,
        use_precomputed=cfg.use_precomputed,
        recompute_fuzzy_only=getattr(cfg, "recompute_fuzzy_only", False),
        augment=False,
    )

    # Infer rr_len/noise_dim from dataset
    sample = test_ds[0]
    rr_len = sample["rr_seq"].shape[0]
    noise_dim = sample["noise_feat"].shape[0]

    model = base.ECGNet(
        num_leads=cfg.num_leads,
        num_arrhythmia_classes=2,
        noise_feat_dim=noise_dim,
        rr_seq_len=rr_len,
        d_model=128,
        rhythm_emb_dim=64,
        noise_emb_dim=32,
        fuzzy_weight_init=cfg.fuzzy_weight_init,
    ).to(device)
    # 兼容：某些 checkpoint 里包含训练期调试用的 buffer（_debug_*），这里允许忽略
    state = checkpoint["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"[Warn] Ignored unexpected keys in checkpoint: {unexpected[:8]}" + (" ..." if len(unexpected) > 8 else ""))
    if missing:
        print(f"[Warn] Missing keys when loading checkpoint: {missing[:8]}" + (" ..." if len(missing) > 8 else ""))
    model.eval()

    pin_memory = device.type == "cuda"
    test_loader = base.DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    # Force segment-level eval by default
    enable_record_agg = False if args.no_record_agg else bool(getattr(cfg, "enable_record_agg", False))

    # Run evaluation for each threshold
    for t in args.thresholds:
        metrics = base.evaluate(
            model,
            test_loader,
            device,
            epoch=0,
            total_epochs=0,
            verbose=True,
            afl_logit_bias_inference=float(getattr(cfg, "afl_logit_bias_inference", 0.0)),
            afl_prob_threshold=float(t),
            tune_afl_threshold=False,
            afl_threshold_search_steps=int(getattr(cfg, "afl_threshold_search_steps", 101)),
            afl_threshold_search_min=float(getattr(cfg, "afl_threshold_search_min", 0.01)),
            afl_threshold_search_max=float(getattr(cfg, "afl_threshold_search_max", 0.99)),
            enable_record_agg=enable_record_agg,
            record_agg_method=str(getattr(cfg, "record_agg_method", "mean")),
            record_agg_topk=int(getattr(cfg, "record_agg_topk", 5)),
        )

        cm = metrics.get("confusion_matrix", np.zeros((2, 2), dtype=np.int64))
        print("-" * 80)
        print(f"[TEST @ threshold={t:.3f}] loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f} "
              f"macroF1={metrics['macro_f1']:.4f} AFL-F1={metrics.get('afl_f1', 0.0):.4f} "
              f"AFL-P={metrics.get('afl_precision', 0.0):.4f} AFL-R={metrics.get('afl_recall', 0.0):.4f}")
        print("Confusion matrix (rows=true, cols=pred):")
        print("          Pred: AF  AFL")
        print(f"True: AF  [{int(cm[0,0]):5d} {int(cm[0,1]):5d}]")
        print(f"      AFL  [{int(cm[1,0]):5d} {int(cm[1,1]):5d}]")

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()

