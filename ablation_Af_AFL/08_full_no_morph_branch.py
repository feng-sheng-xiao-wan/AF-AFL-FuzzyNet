# -*- coding: utf-8 -*-
"""
消融实验 08：完整模型，但关闭形态分支（仅 RR + 噪声 + 模糊规则 + 边界感知）
"""

import argparse
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ablation_Af_AFL.ablation_utils import run_ablation


def main():
    parser = argparse.ArgumentParser(description="Ablation 08: Full model without morphology branch")
    parser.add_argument("--combined_csv", type=str, help="Combined CSV path (with precomputed features, recommended)")
    parser.add_argument("--afdb_csv", type=str, help="AFDB segments CSV (legacy, use --combined_csv if available)")
    parser.add_argument("--ltafdb_csv", type=str, help="LTAFDB segments CSV (legacy, use --combined_csv if available)")
    parser.add_argument("--output_dir", type=str, default="outputs/ablation/08_full_no_morph")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--use_oversampling", action="store_true", default=True)
    parser.add_argument("--no_use_oversampling", dest="use_oversampling", action="store_false")
    parser.add_argument("--oversample_target_ratio", type=float, default=1.0)
    parser.add_argument("--oversample_strategy", type=str, default="random", choices=["random", "patient_level"])
    parser.add_argument("--use_precomputed", action="store_true", default=True)
    parser.add_argument("--no_use_precomputed", dest="use_precomputed", action="store_false")
    parser.add_argument("--recompute_fuzzy_only", action="store_true", default=False)
    args = parser.parse_args()

    if not args.combined_csv and (not args.afdb_csv or not args.ltafdb_csv):
        parser.error("Either --combined_csv or both --afdb_csv and --ltafdb_csv must be provided")

    run_ablation(
        combined_csv=args.combined_csv,
        afdb_csv=args.afdb_csv,
        ltafdb_csv=args.ltafdb_csv,
        output_dir=args.output_dir,
        disable_noise=False,
        disable_fuzzy=False,
        disable_morph=True,  # 关键：关闭形态分支
        disable_rr=False,
        use_boundary_aware=True,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        use_oversampling=args.use_oversampling,
        oversample_target_ratio=args.oversample_target_ratio,
        oversample_strategy=args.oversample_strategy,
        use_precomputed=args.use_precomputed,
        recompute_fuzzy_only=args.recompute_fuzzy_only,
    )


if __name__ == "__main__":
    main()

