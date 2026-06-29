# -*- coding: utf-8 -*-
"""
消融实验 06：只用模糊规则，不用深度模型（Fuzzy-only baseline，评估 AF vs AFL）

说明：
- 不进行神经网络训练，直接对每个样本应用 FuzzyRuleSystem / apply_fuzzy_rules，
  取前两个 logits 作为 AF / AFL 的得分，argmax 得到预测标签。
- 评估指标：Accuracy / Precision / Recall / F1（macro，与主实验保持一致）。
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import noise_aware_ecg_af_afl_simple as base


def evaluate_fuzzy_only(csv_path: str):
    ds = base.ECGDataset(
        csv_path,
        use_precomputed=True,
        recompute_fuzzy_only=False,
    )
    y_true, y_pred = [], []
    for i in range(len(ds)):
        sample = ds[i]
        label = int(sample["label"])
        # 若已有预计算 fuzzy_logits，则直接使用；否则在线计算
        if sample["fuzzy_logits"] is not None:
            logits = sample["fuzzy_logits"].numpy()
        else:
            # 在线计算：从 rr_seq / ecg / noise 特征中推理
            # 这里只依赖 RR 统计特征和模糊规则系统
            ecg = sample["ecg"].numpy()
            rr_seq = sample["rr_seq"].numpy()
            # 重新计算模糊规则 logits
            logits_full = base.apply_fuzzy_rules(
                ecg=ecg,
                rr_seq=rr_seq,
                noise_feats=None,
            )
            logits = logits_full[:2]
        if logits.shape[0] > 2:
            logits = logits[:2]
        pred = int(np.argmax(logits))
        y_true.append(label)
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Ablation 06: Fuzzy-only baseline (no deep model)")
    parser.add_argument(
        "--combined_csv",
        type=str,
        required=True,
        help="Combined CSV path (with precomputed features, recommended)",
    )
    args = parser.parse_args()

    csv_path = Path(args.combined_csv)
    if not csv_path.is_file():
        parser.error(f"CSV not found: {csv_path}")

    print("Fuzzy-only baseline on CSV:", csv_path)
    acc, precision, recall, f1 = evaluate_fuzzy_only(str(csv_path))
    print(f"Accuracy: {acc:.6f}")
    print(f"Precision (macro): {precision:.6f}")
    print(f"Recall (macro): {recall:.6f}")
    print(f"F1 (macro): {f1:.6f}")


if __name__ == "__main__":
    main()

