# -*- coding: utf-8 -*-
"""
消融实验 01：仅形态 + RR（关闭噪声分支、关闭模糊规则、关闭边界感知）
"""

import argparse
import sys
import os

# 添加项目根目录到路径，以便导入 ablation_Af_AFL 模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ablation_Af_AFL.ablation_utils import run_ablation


def main():
    parser = argparse.ArgumentParser(description="Ablation 01: Morph + RR only")
    # CSV 参数（支持合并CSV或分离CSV）
    parser.add_argument("--combined_csv", type=str, help="Combined CSV path (with precomputed features, recommended)")
    parser.add_argument("--afdb_csv", type=str, help="AFDB segments CSV (legacy, use --combined_csv if available)")
    parser.add_argument("--ltafdb_csv", type=str, help="LTAFDB segments CSV (legacy, use --combined_csv if available)")
    parser.add_argument("--output_dir", type=str, default="outputs/ablation/01_morph_rr")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    # 过采样参数
    parser.add_argument("--use_oversampling", action="store_true", default=True, help="Use oversampling for training data")
    parser.add_argument("--no_use_oversampling", dest="use_oversampling", action="store_false", help="Disable oversampling")
    parser.add_argument("--oversample_target_ratio", type=float, default=1.0, help="Oversampling target ratio (1.0=fully balanced)")
    parser.add_argument("--oversample_strategy", type=str, default="random", choices=["random", "patient_level"], help="Oversampling strategy")
    # 预计算特征参数
    parser.add_argument("--use_precomputed", action="store_true", default=True, help="Use precomputed features if available")
    parser.add_argument("--no_use_precomputed", dest="use_precomputed", action="store_false", help="Disable precomputed features")
    parser.add_argument("--recompute_fuzzy_only", action="store_true", default=False, help="Recompute only fuzzy features even if precomputed features exist")
    args = parser.parse_args()

    # 验证参数
    if not args.combined_csv and (not args.afdb_csv or not args.ltafdb_csv):
        parser.error("Either --combined_csv or both --afdb_csv and --ltafdb_csv must be provided")

    run_ablation(
        combined_csv=args.combined_csv,
        afdb_csv=args.afdb_csv,
        ltafdb_csv=args.ltafdb_csv,
        output_dir=args.output_dir,
        disable_noise=True,
        disable_fuzzy=True,
        use_boundary_aware=False,
        # 评估口径与主实验一致：使用 segment-level AFL-F1（关闭 record 聚合）
        enable_record_agg=False,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_oversampling=args.use_oversampling,
        oversample_target_ratio=args.oversample_target_ratio,
        oversample_strategy=args.oversample_strategy,
        use_precomputed=args.use_precomputed,
        recompute_fuzzy_only=args.recompute_fuzzy_only,
    )


if __name__ == "__main__":
    main()

