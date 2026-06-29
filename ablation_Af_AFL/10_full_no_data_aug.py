# -*- coding: utf-8 -*-
"""
消融实验 10：完整模型，但取消“数据增强”

定义（按你的要求）：
- 不使用重采样（oversampling）
- 不使用噪声扩增数据（因此需要传入“不含噪声扩增”的 combined_csv）

注意：
- “噪声扩增”是离线生成的新样本，运行时无法通过开关关闭；
  需要通过选择未扩增的 CSV 来实现（例如 multi_dataset_combined_with_features.csv）。
"""

import argparse
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ablation_Af_AFL.ablation_utils import run_ablation


def main():
    parser = argparse.ArgumentParser(description="Ablation 10: Full model without data augmentation (no oversampling + no noise-aug CSV)")
    parser.add_argument(
        "--combined_csv",
        type=str,
        required=True,
        help="Combined CSV WITHOUT noise augmentation (e.g. data/precompute_features/multi_dataset_combined_with_features.csv)",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/ablation/10_full_no_data_aug")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/cuda:0")
    # 预计算特征参数
    parser.add_argument("--use_precomputed", action="store_true", default=True, help="Use precomputed features if available")
    parser.add_argument("--no_use_precomputed", dest="use_precomputed", action="store_false", help="Disable precomputed features")
    parser.add_argument("--recompute_fuzzy_only", action="store_true", default=False, help="Recompute only fuzzy features even if precomputed features exist")
    args = parser.parse_args()

    run_ablation(
        combined_csv=args.combined_csv,
        output_dir=args.output_dir,
        disable_noise=False,
        disable_fuzzy=False,
        disable_morph=False,
        disable_rr=False,
        use_boundary_aware=True,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        use_oversampling=False,  # 关键：关闭重采样
        use_precomputed=args.use_precomputed,
        recompute_fuzzy_only=args.recompute_fuzzy_only,
    )


if __name__ == "__main__":
    main()

