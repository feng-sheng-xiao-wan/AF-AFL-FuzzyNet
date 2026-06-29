import argparse
import csv
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import noise_aware_ecg_af_afl_simple as base


def parse_args():
    p = argparse.ArgumentParser(description="Scan AFL thresholds and plot AFL-F1 / Macro-F1 curve.")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint .pt path")
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
        help="AFL prob thresholds (predict AFL if P(AFL) >= t).",
    )
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42, help="Split seed for test_df rebuild.")
    p.add_argument("--out_csv", type=str, default="", help="Output CSV path (optional).")
    p.add_argument("--out_png", type=str, default="", help="Output PNG path (optional).")
    p.add_argument("--bias_override", type=float, default=None, help="Override afl_logit_bias_inference at inference time (optional).")
    p.add_argument("--silent", action="store_true", default=True, help="Suppress per-threshold console spam.")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    device = base.get_device(args.device)
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg_dict = checkpoint.get("cfg", {})
    cfg = base.TrainConfig(**cfg_dict) if isinstance(cfg_dict, dict) else base.TrainConfig()

    if not cfg.data_csv:
        raise ValueError("Checkpoint cfg does not contain data_csv; cannot rebuild test split.")

    # Rebuild the same test split used for evaluation
    _, _, test_df = base.split_dataset_csv(cfg.data_csv, 0.8, 0.1, 0.1, seed=args.seed)
    if len(test_df) == 0:
        raise ValueError("Test split is empty.")

    output_dir = os.path.dirname(args.ckpt) or "."
    temp_test_csv = os.path.join(output_dir, "temp_test_retest_afl_thresh_scan.csv")
    test_df.to_csv(temp_test_csv, index=False)

    test_ds = base.ECGDataset(
        temp_test_csv,
        max_len=cfg.max_len,
        num_leads=cfg.num_leads,
        max_rr_intervals=cfg.max_rr_intervals,
        use_stft_dbscan=getattr(cfg, "use_stft_dbscan", False),
        multi_lead_method=getattr(cfg, "multi_lead_method", "mean"),
        flutter_method=getattr(cfg, "flutter_method", "default"),
        lead_names=getattr(cfg, "lead_names", None),
        use_precomputed=cfg.use_precomputed,
        recompute_fuzzy_only=getattr(cfg, "recompute_fuzzy_only", False),
        augment=False,
    )

    sample = test_ds[0]
    rr_len = int(sample["rr_seq"].shape[0])
    noise_dim = int(sample["noise_feat"].shape[0])

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

    # load checkpoint weights
    state = checkpoint["model"]
    model.load_state_dict(state, strict=False)
    model.eval()

    pin_memory = device.type == "cuda"
    test_loader = base.DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    default_bias = float(getattr(cfg, "afl_logit_bias_inference", 0.0) or 0.0)
    bias = float(args.bias_override) if args.bias_override is not None else default_bias

    rows = []
    for t in args.thresholds:
        if args.silent:
            with redirect_stdout(io.StringIO()):
                metrics = base.evaluate(
                    model,
                    test_loader,
                    str(device),
                    epoch=0,
                    total_epochs=0,
                    verbose=False,
                    afl_logit_bias_inference=bias,
                    afl_prob_threshold=float(t),
                    tune_afl_threshold=False,
                    afl_threshold_search_steps=int(getattr(cfg, "afl_threshold_search_steps", 101)),
                    afl_threshold_search_min=float(getattr(cfg, "afl_threshold_search_min", 0.01)),
                    afl_threshold_search_max=float(getattr(cfg, "afl_threshold_search_max", 0.99)),
                    enable_record_agg=False,
                )
        else:
            metrics = base.evaluate(
                model,
                test_loader,
                str(device),
                epoch=0,
                total_epochs=0,
                verbose=True,
                afl_logit_bias_inference=bias,
                afl_prob_threshold=float(t),
                tune_afl_threshold=False,
                afl_threshold_search_steps=int(getattr(cfg, "afl_threshold_search_steps", 101)),
                afl_threshold_search_min=float(getattr(cfg, "afl_threshold_search_min", 0.01)),
                afl_threshold_search_max=float(getattr(cfg, "afl_threshold_search_max", 0.99)),
                enable_record_agg=False,
            )

        rows.append(
            {
                "threshold": float(t),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "macro_precision": float(metrics.get("macro_precision", 0.0)),
                "macro_recall": float(metrics.get("macro_recall", 0.0)),
                "macro_f1": float(metrics.get("macro_f1", 0.0)),
                "afl_precision": float(metrics.get("afl_precision", 0.0)),
                "afl_recall": float(metrics.get("afl_recall", 0.0)),
                "afl_f1": float(metrics.get("afl_f1", 0.0)),
            }
        )

    # CSV output
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "threshold",
                    "accuracy",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                    "afl_precision",
                    "afl_recall",
                    "afl_f1",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    # Plot output
    if args.out_png:
        import matplotlib.pyplot as plt

        xs = [r["threshold"] for r in rows]
        macro_f1 = [r["macro_f1"] for r in rows]
        afl_f1 = [r["afl_f1"] for r in rows]

        plt.figure(figsize=(7.5, 4.8))
        plt.plot(xs, macro_f1, marker="o", label="Macro-F1")
        plt.plot(xs, afl_f1, marker="s", label="AFL-F1")
        plt.xlabel("AFL probability threshold")
        plt.ylabel("F1")
        plt.title(f"AFL threshold sensitivity\nckpt={os.path.basename(args.ckpt)} bias={bias}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(args.out_png, dpi=200)

    # Print best summary
    best_afl = max(rows, key=lambda r: r["afl_f1"])
    best_macro = max(rows, key=lambda r: r["macro_f1"])
    print(f"[Done] bias={bias} (default_bias={default_bias})")
    print(f"Best AFL-F1: t={best_afl['threshold']:.4f} afl_f1={best_afl['afl_f1']:.6f} macro_f1={best_afl['macro_f1']:.6f}")
    print(f"Best Macro-F1: t={best_macro['threshold']:.4f} afl_f1={best_macro['afl_f1']:.6f} macro_f1={best_macro['macro_f1']:.6f}")


if __name__ == "__main__":
    main()

