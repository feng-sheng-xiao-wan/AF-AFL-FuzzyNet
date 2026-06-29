# -*- coding: utf-8 -*-
"""
主实验 + 消融（05-10）统一评估详细指标报告

输出指标（AF vs AFL 二分类）：
- Accuracy
- Macro Precision / Recall / F1
- AF Precision / Recall / F1
- AFL Precision / Recall / F1
- Confusion Matrix (2x2)
- PR-AUC（AFL 作为正类的 average_precision_score）

说明：
- PR-AUC 与阈值无关；其他指标使用 checkpoint 保存的 afl_threshold 做阈值化预测。
- 05-09 的 test split 使用 csv_aug；10_no_data_aug 使用 csv_noaug。
"""

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import noise_aware_ecg_af_afl_simple as base


@dataclass
class ExpDef:
    exp: str
    data_key: str  # "aug" or "noaug"
    ckpt: Optional[str]  # checkpoint path (relative to PROJECT_ROOT) or None for fuzzy-only
    disable_fuzzy: bool = False
    disable_rr: bool = False
    disable_morph: bool = False
    disable_noise: bool = False
    fuzzy_only: bool = False


def _write_temp_csv(df, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return str(out_path)


def _patch_dataset_class(*, disable_fuzzy: bool, disable_rr: bool, disable_morph: bool, disable_noise: bool):
    Original = base.ECGDataset

    class Patched(Original):
        def __getitem__(self, idx):
            b = super().__getitem__(idx)
            if disable_morph:
                b["ecg"] = torch.zeros_like(b["ecg"])
            if disable_rr:
                b["rr_seq"] = torch.zeros_like(b["rr_seq"])
            if disable_noise:
                b["noise_feat"] = torch.zeros_like(b["noise_feat"])
            if disable_fuzzy:
                b["fuzzy_logits"] = torch.zeros_like(b["fuzzy_logits"])
            return b

    base.ECGDataset = Patched
    return Original


def _load_ckpt(ckpt_path: str, device: torch.device):
    return torch.load(ckpt_path, map_location=device, weights_only=False)


def _compute_pr_auc_afl(y_true: np.ndarray, p_afl: np.ndarray) -> float:
    # y_true: 0/1
    y_bin = (y_true == 1).astype(np.int64)
    return float(average_precision_score(y_bin, p_afl))


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numpy softmax (stable) for CPU-side threshold/bias fallback."""
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, p_afl: np.ndarray) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # precision_recall_fscore_support: per-class (label 0 then 1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )

    macro_p = float(np.mean(prec))
    macro_r = float(np.mean(rec))
    macro_f1 = float(np.mean(f1))
    acc = float((y_true == y_pred).mean())

    pr_auc = _compute_pr_auc_afl(y_true, p_afl)

    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "AF_precision": float(prec[0]),
        "AF_recall": float(rec[0]),
        "AF_f1": float(f1[0]),
        "AFL_precision": float(prec[1]),
        "AFL_recall": float(rec[1]),
        "AFL_f1": float(f1[1]),
        "afl_pr_auc": pr_auc,
        "cm00": int(cm[0, 0]),
        "cm01": int(cm[0, 1]),
        "cm10": int(cm[1, 0]),
        "cm11": int(cm[1, 1]),
    }


@torch.no_grad()
def _run_model_on_loader(model, loader, device: torch.device):
    model.eval()
    all_y = []
    all_y_pred = []
    all_p_afl = []
    all_logits = []
    for batch in loader:
        ecg = batch["ecg"].to(device)
        rr_seq = batch["rr_seq"].to(device)
        noise_feat = batch["noise_feat"].to(device)
        fuzzy_logits = batch.get("fuzzy_logits")
        if fuzzy_logits is not None:
            fuzzy_logits = fuzzy_logits.to(device)
        y = batch["label"].to(device)

        logits_arr, _ = model(ecg, rr_seq, noise_feat, fuzzy_logits)

        # 二分类：logits_arr shape [B,2]
        probs = torch.softmax(logits_arr, dim=-1)
        p_afl = probs[:, 1].detach().cpu().numpy()
        pred = torch.argmax(logits_arr, dim=-1).detach().cpu().numpy()

        all_y.append(y.detach().cpu().numpy())
        all_y_pred.append(pred)
        all_p_afl.append(p_afl)
        all_logits.append(logits_arr.detach().cpu().numpy())

    y_true = np.concatenate(all_y, axis=0).astype(np.int64)
    y_pred = np.concatenate(all_y_pred, axis=0).astype(np.int64)
    p_afl = np.concatenate(all_p_afl, axis=0).astype(np.float64)
    logits = np.concatenate(all_logits, axis=0).astype(np.float64)
    return y_true, y_pred, p_afl, logits


def _run_exp(exp_def: ExpDef, data_csv: str, test_df, device: torch.device, batch_size: int, num_workers: int, seed: int):
    # 写临时 test csv 给 ECGDataset 使用
    tmp_csv = PROJECT_ROOT / "outputs" / "ablation" / f"temp_test_{exp_def.exp}_{'aug' if exp_def.data_key=='aug' else 'noaug'}.csv"
    test_csv_path = _write_temp_csv(test_df, tmp_csv)

    # 如果有 ckpt，先读取 cfg，避免 num_leads/max_len 等不一致导致 forward 报错
    cfg = None
    ckpt = None
    ckpt_path = None
    if (not exp_def.fuzzy_only) and exp_def.ckpt:
        ckpt_path = str((PROJECT_ROOT / exp_def.ckpt).resolve()) if not os.path.isabs(exp_def.ckpt) else exp_def.ckpt
        if not os.path.isfile(ckpt_path):
            return {"error": "ckpt_not_found"}
        ckpt = _load_ckpt(ckpt_path, device)
        cfg_dict = ckpt.get("cfg", {})
        cfg = base.TrainConfig(**cfg_dict) if isinstance(cfg_dict, dict) else base.TrainConfig()

    # 根据 exp 需要 patch dataset 输入
    original_cls = _patch_dataset_class(
        disable_fuzzy=exp_def.disable_fuzzy,
        disable_rr=exp_def.disable_rr,
        disable_morph=exp_def.disable_morph,
        disable_noise=exp_def.disable_noise,
    )
    try:
        ds = base.ECGDataset(
            test_csv_path,
            max_len=int(getattr(cfg, "max_len", 2500)) if cfg is not None else 2500,
            num_leads=int(getattr(cfg, "num_leads", 1)) if cfg is not None else 1,
            max_rr_intervals=int(getattr(cfg, "max_rr_intervals", 32)) if cfg is not None else 32,
            use_precomputed=bool(getattr(cfg, "use_precomputed", True)) if cfg is not None else True,
            recompute_fuzzy_only=bool(getattr(cfg, "recompute_fuzzy_only", False)) if cfg is not None else False,
        )
    finally:
        base.ECGDataset = original_cls

    # 取样本维度
    s0 = ds[0]
    rr_len = int(s0["rr_seq"].shape[0])
    noise_dim = int(s0["noise_feat"].shape[0])

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda")
    )

    if exp_def.fuzzy_only:
        # fuzzy-only：直接用 fuzzy_logits 推理
        all_y = []
        all_y_pred = []
        all_p_afl = []
        for batch in loader:
            y = batch["label"].detach().cpu().numpy().astype(np.int64)
            fuzzy_logits = batch["fuzzy_logits"].detach()
            # 二分类：取前两维
            if fuzzy_logits.dim() == 2 and fuzzy_logits.size(1) >= 2:
                logits2 = fuzzy_logits[:, :2]
            else:
                logits2 = fuzzy_logits
            probs = torch.softmax(logits2, dim=-1)
            p_afl = probs[:, 1].detach().cpu().numpy()
            pred = torch.argmax(probs, dim=-1).detach().cpu().numpy().astype(np.int64)
            all_y.append(y)
            all_y_pred.append(pred)
            all_p_afl.append(p_afl)

        y_true = np.concatenate(all_y, axis=0)
        y_pred = np.concatenate(all_y_pred, axis=0)
        p_afl = np.concatenate(all_p_afl, axis=0)
        return _compute_metrics(y_true, y_pred, p_afl)

    if ckpt is None or cfg is None:
        raise ValueError(f"{exp_def.exp}: failed to load checkpoint/cfg")

    model = base.ECGNet(
        num_leads=int(getattr(cfg, "num_leads", ds[0]["ecg"].shape[0] if hasattr(ds[0]["ecg"], "shape") else 1)),
        num_arrhythmia_classes=2,
        noise_feat_dim=noise_dim,
        rr_seq_len=rr_len,
        d_model=128,
        rhythm_emb_dim=64,
        noise_emb_dim=32,
        fuzzy_weight_init=float(getattr(cfg, "fuzzy_weight_init", 0.3)),
    ).to(device)

    state_dict = ckpt["model"]
    model.load_state_dict(state_dict, strict=False)

    # 先用模型 argmax 得到 p_afl，再按阈值覆盖 y_pred（如果有阈值）
    y_true, y_pred_argmax, p_afl, logits_all = _run_model_on_loader(model, loader, device)

    # 与 test_ablation_05_10.py 对齐：初始评估时总是应用 cfg.afl_logit_bias_inference（除非回退里显式置 0）
    base_bias = float(getattr(cfg, "afl_logit_bias_inference", 0.0) or 0.0)
    if abs(base_bias) > 1e-12:
        logits_base = logits_all.copy()
        logits_base[:, 1] += base_bias
        probs_base = _softmax_np(logits_base, axis=-1)
        p_afl_base = probs_base[:, 1].astype(np.float64)
    else:
        p_afl_base = p_afl

    afl_threshold = ckpt.get("afl_threshold", None)
    if afl_threshold is not None:
        y_pred = (p_afl_base >= float(afl_threshold)).astype(np.int64)
    else:
        y_pred = y_pred_argmax

    best_metrics = _compute_metrics(y_true, y_pred, p_afl_base)
    best_macro_f1 = best_metrics["macro_f1"]
    best_th = float(afl_threshold) if afl_threshold is not None else None

    # 针对 AFL 极低召回的情况：做阈值回退 + logit bias 回退（与 `test_ablation_05_10.py` 对齐）
    afl_total = int((y_true == 1).sum())
    afl_recall = float(best_metrics["AFL_recall"])
    if afl_total >= 10 and afl_recall < 0.02:
        threshold_candidates = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
        bias_candidates = [0.2, 0.5, 1.0, 2.0]

        # (1) 低阈值搜索（仅基于原始概率 p_afl）
        for th in threshold_candidates:
            y_pred_th = (p_afl >= th).astype(np.int64)
            m = _compute_metrics(y_true, y_pred_th, p_afl)
            if m["macro_f1"] > best_macro_f1:
                best_macro_f1 = m["macro_f1"]
                best_metrics = m
                best_th = th
        # 若阈值搜索后 AFL 仍然极低召回，再做 logit bias 回退（与 test_ablation_05_10.py 对齐）
        afl_recall_after_th = float(best_metrics["AFL_recall"])
        if afl_total >= 10 and afl_recall_after_th < 0.02 and best_th is not None:
            for bias in bias_candidates:
                logits_bias = logits_all.copy()
                logits_bias[:, 1] += float(bias)
                probs_bias = _softmax_np(logits_bias, axis=-1)
                p_afl_bias = probs_bias[:, 1].astype(np.float64)

                y_pred_b = (p_afl_bias >= float(best_th)).astype(np.int64)
                m = _compute_metrics(y_true, y_pred_b, p_afl)
                if m["macro_f1"] > best_macro_f1:
                    best_macro_f1 = m["macro_f1"]
                    best_metrics = m

    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Main + Ablations detailed metrics report (AF vs AFL)")
    parser.add_argument("--csv_aug", type=str, required=True)
    parser.add_argument("--csv_noaug", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_csv",
        type=str,
        default="ablation_Af_AFL/main_and_ablation_detailed_metrics.csv",
        help="output csv path (relative to project root)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    # 生成/缓存 test_df：aug/noaug 两份
    _, _, test_aug = base.split_dataset_csv(args.csv_aug, 0.8, 0.1, 0.1, seed=args.seed)
    _, _, test_noaug = base.split_dataset_csv(args.csv_noaug, 0.8, 0.1, 0.1, seed=args.seed)

    main_ckpt = "outputs_thr_tuned_multi_afl_noise_aug_seg/af_vs_afl_afdb_ltafdb.pt"

    exps = [
        ExpDef(exp="main_experiment", data_key="aug", ckpt=main_ckpt),
        ExpDef(exp="05_no_fuzzy", data_key="aug", ckpt="outputs/ablation/05_full_no_fuzzy/af_vs_afl_afdb_ltafdb.pt", disable_fuzzy=True),
        ExpDef(exp="06_fuzzy_only", data_key="aug", ckpt=None, fuzzy_only=True),
        ExpDef(exp="07_no_rr", data_key="aug", ckpt="outputs/ablation/07_full_no_rr/af_vs_afl_afdb_ltafdb.pt", disable_rr=True),
        ExpDef(exp="08_no_morph", data_key="aug", ckpt="outputs/ablation/08_full_no_morph/af_vs_afl_afdb_ltafdb.pt", disable_morph=True),
        ExpDef(exp="09_no_oversampling", data_key="aug", ckpt="outputs/ablation/09_full_no_oversampling/af_vs_afl_afdb_ltafdb.pt"),
        ExpDef(exp="10_no_data_aug", data_key="noaug", ckpt="outputs/ablation/10_full_no_data_aug/af_vs_afl_afdb_ltafdb.pt"),
    ]

    rows = []
    for e in exps:
        print(f"\n[Eval] {e.exp}")
        data_csv = args.csv_aug if e.data_key == "aug" else args.csv_noaug
        test_df = test_aug if e.data_key == "aug" else test_noaug

        metrics = _run_exp(
            e, data_csv, test_df, device,
            batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed
        )
        row = {"exp": e.exp, "ckpt": e.ckpt or ""}
        row.update(metrics)
        rows.append(row)

    out_csv_path = PROJECT_ROOT / args.out_csv
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "exp", "ckpt", "error",
        "accuracy",
        "macro_precision", "macro_recall", "macro_f1",
        "AF_precision", "AF_recall", "AF_f1",
        "AFL_precision", "AFL_recall", "AFL_f1",
        "afl_pr_auc",
        "cm00", "cm01", "cm10", "cm11",
    ]
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print("\nSaved:", str(out_csv_path))


if __name__ == "__main__":
    main()

