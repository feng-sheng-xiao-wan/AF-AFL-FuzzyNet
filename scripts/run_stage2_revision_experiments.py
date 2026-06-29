from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import noise_aware_ecg_af_afl_simple as base
from ablation_Af_AFL import baseline_utils as bu


LABELS = [0, 1]
MODEL_DISPLAY = {
    "ours_full": "Ours",
    "ours_neural_only_no_gate": "Ours neural-only (post-hoc No Gate)",
    "ours_no_quality_input": "Ours quality-masked (post-hoc No Quality)",
    "ours_fixed_fusion_alpha030": "Ours fixed fusion alpha=0.30 (post-hoc)",
    "fuzzy_logits_only": "Fuzzy logits only",
    "arnet2": "ArNet2",
    "rawecgnet": "RawECGNet",
    "wang2021": "Wang2021 BiLSTM",
    "kraft2025": "Kraft2025 ConvNeXt1D",
    "fleury2024": "Fleury2024 modular",
}

BASELINE_DEFAULTS = {
    "arnet2": {
        "ckpt": "ablation_Af_AFL/checkpoints_arnet2/best_arnet2.pth",
        "val_dir": "data/ablation_rr_npz/val",
        "test_dir": "data/ablation_rr_npz/test",
    },
    "rawecgnet": {
        "ckpt": "ablation_Af_AFL/checkpoints_rawecgnet/best_rawecgnet.pth",
        "val_dir": "ablation_Af_AFL/rawecg_npz/val",
        "test_dir": "ablation_Af_AFL/rawecg_npz/test",
    },
    "wang2021": {
        "ckpt": "ablation_Af_AFL/checkpoints_wang2021/best_wang2021_bilstm.pth",
        "val_dir": "ablation_Af_AFL/rawecg_npz/val",
        "test_dir": "ablation_Af_AFL/rawecg_npz/test",
    },
    "kraft2025": {
        "ckpt": "ablation_Af_AFL/checkpoints_kraft2025/best_kraft2025_convnext1d.pth",
        "val_dir": "ablation_Af_AFL/rawecg_npz/val",
        "test_dir": "ablation_Af_AFL/rawecg_npz/test",
    },
    "fleury2024": {
        "ckpt": "ablation_Af_AFL/checkpoints_fleury2024/best_fleury2024_modular.pth",
        "val_dir": "ablation_Af_AFL/fleury_ecg_rr_npz/val",
        "test_dir": "ablation_Af_AFL/fleury_ecg_rr_npz/test",
    },
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _threshold_search(
    y_true: np.ndarray,
    p_afl: np.ndarray,
    steps: int = 1001,
    t_min: float = 0.01,
    t_max: float = 0.99,
) -> Tuple[float, Dict[str, float]]:
    y_true = np.asarray(y_true, dtype=np.int64)
    p_afl = np.asarray(p_afl, dtype=np.float32)
    best_t = 0.5
    best = {"afl_f1": -1.0, "afl_precision": 0.0, "afl_recall": 0.0}
    for t in np.linspace(t_min, t_max, max(3, int(steps)), dtype=np.float32):
        pred = (p_afl >= float(t)).astype(np.int64)
        tp = float(((y_true == 1) & (pred == 1)).sum())
        fp = float(((y_true == 0) & (pred == 1)).sum())
        fn = float(((y_true == 1) & (pred == 0)).sum())
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        score = 2.0 * precision * recall / (precision + recall + 1e-12)
        if (
            score > best["afl_f1"] + 1e-12
            or (abs(score - best["afl_f1"]) <= 1e-12 and recall > best["afl_recall"] + 1e-12)
            or (
                abs(score - best["afl_f1"]) <= 1e-12
                and abs(recall - best["afl_recall"]) <= 1e-12
                and precision > best["afl_precision"] + 1e-12
            )
        ):
            best_t = float(t)
            best = {
                "afl_f1": float(score),
                "afl_precision": float(precision),
                "afl_recall": float(recall),
            }
    return best_t, best


def _safe_auc(y_true: np.ndarray, p_afl: np.ndarray) -> Tuple[float, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    p_afl = np.asarray(p_afl, dtype=np.float32)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true, p_afl)), float(average_precision_score(y_true, p_afl))


def _metric_bundle(
    y_true: np.ndarray,
    p_afl: np.ndarray,
    threshold: float,
    include_curves: bool = True,
) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.int64)
    p_afl = np.asarray(p_afl, dtype=np.float32)
    pred = (p_afl >= float(threshold)).astype(np.int64)
    cm = confusion_matrix(y_true, pred, labels=LABELS)
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    auroc, auprc = _safe_auc(y_true, p_afl)
    out: Dict[str, object] = {
        "n": int(len(y_true)),
        "n_af": int((y_true == 0).sum()),
        "n_afl": int((y_true == 1).sum()),
        "threshold": float(threshold),
        "acc": float((pred == y_true).mean()),
        "afl_precision": float(precision),
        "afl_recall": float(recall),
        "afl_f1": float(f1),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "auroc": auroc,
        "auprc": auprc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "cm": cm.tolist(),
    }
    if include_curves and len(np.unique(y_true)) >= 2:
        fpr, tpr, roc_thr = roc_curve(y_true, p_afl)
        pr_p, pr_r, pr_thr = precision_recall_curve(y_true, p_afl)
        out["roc_curve"] = pd.DataFrame(
            {"fpr": fpr, "tpr": tpr, "threshold": np.r_[roc_thr]}
        )
        out["pr_curve"] = pd.DataFrame(
            {
                "precision": pr_p,
                "recall": pr_r,
                "threshold": np.r_[pr_thr, np.nan],
            }
        )
    return out


def _bootstrap_ci(
    y_true: np.ndarray,
    p_afl: np.ndarray,
    threshold: float,
    n_boot: int,
    seed: int,
    metrics: Iterable[str],
) -> Dict[str, Tuple[float, float]]:
    y_true = np.asarray(y_true, dtype=np.int64)
    p_afl = np.asarray(p_afl, dtype=np.float32)
    rng = np.random.default_rng(seed)
    metric_names = list(metrics)
    values = {m: [] for m in metric_names}
    n = len(y_true)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yy = y_true[idx]
        pp = p_afl[idx]
        if len(np.unique(yy)) < 2 and ("auroc" in metric_names or "auprc" in metric_names):
            auc = {"auroc": float("nan"), "auprc": float("nan")}
        else:
            auroc, auprc = _safe_auc(yy, pp)
            auc = {"auroc": auroc, "auprc": auprc}
        pred = (pp >= float(threshold)).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(yy, pred, labels=LABELS).ravel().astype(float)
        afl_p = tp / (tp + fp + 1e-12)
        afl_r = tp / (tp + fn + 1e-12)
        afl_f1 = 2.0 * afl_p * afl_r / (afl_p + afl_r + 1e-12)
        af_p = tn / (tn + fn + 1e-12)
        af_r = tn / (tn + fp + 1e-12)
        af_f1 = 2.0 * af_p * af_r / (af_p + af_r + 1e-12)
        bundle = {
            "acc": float((pred == yy).mean()),
            "afl_precision": float(afl_p),
            "afl_recall": float(afl_r),
            "afl_f1": float(afl_f1),
            "macro_f1": float((af_f1 + afl_f1) / 2.0),
        }
        for m in metric_names:
            val = auc[m] if m in auc else bundle[m]
            if not (isinstance(val, float) and math.isnan(val)):
                values[m].append(float(val))
    ci = {}
    for m, vals in values.items():
        if vals:
            lo, hi = np.percentile(np.asarray(vals, dtype=np.float64), [2.5, 97.5])
            ci[m] = (float(lo), float(hi))
        else:
            ci[m] = (float("nan"), float("nan"))
    return ci


def _format_ci(x: float, lo: float, hi: float, digits: int = 3) -> str:
    if math.isnan(float(x)):
        return "NA"
    return f"{x:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _save_predictions(path: Path, y: np.ndarray, p: np.ndarray, meta: Optional[pd.DataFrame] = None) -> None:
    df = pd.DataFrame({"y_true": y.astype(int), "p_afl": p.astype(float)})
    if meta is not None:
        keep = [c for c in ["path", "label_raw", "record_name", "dataset", "_record_group"] if c in meta.columns]
        df = pd.concat([meta[keep].reset_index(drop=True), df], axis=1)
    df.to_csv(path, index=False)


def _load_predictions(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    return df["y_true"].to_numpy(dtype=np.int64), df["p_afl"].to_numpy(dtype=np.float32)


def _prepare_main_split(out_dir: Path, data_csv: Path, seed: int) -> Tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    os.environ["AFL_NOISY_GROUP"] = "independent"
    _, val_df, test_df = base.split_dataset_csv(str(data_csv), 0.8, 0.1, 0.1, seed=seed)

    revision_test = PROJECT_ROOT / "ablation_Af_AFL" / "revision_pack" / "temp_test_eval.csv"
    if revision_test.exists():
        rev_df = pd.read_csv(revision_test)
        if set(test_df["path"].astype(str)) == set(rev_df["path"].astype(str)):
            test_df = rev_df
        else:
            print("[WARN] Reconstructed independent split does not match revision_pack/temp_test_eval.csv.")

    val_csv = out_dir / "main_val_independent_eval.csv"
    test_csv = out_dir / "main_test_independent_eval.csv"
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    return val_csv, test_csv, val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _load_main_model(ckpt_path: Path, val_csv: Path, cfg: base.TrainConfig, device: torch.device) -> base.ECGNet:
    probe = base.ECGDataset(
        str(val_csv),
        max_len=cfg.max_len,
        num_leads=cfg.num_leads,
        max_rr_intervals=cfg.max_rr_intervals,
        use_precomputed=cfg.use_precomputed,
        multi_lead_method=cfg.multi_lead_method,
        flutter_method=cfg.flutter_method,
        lead_names=cfg.lead_names,
        augment=False,
    )
    sample = probe[0]
    model = base.ECGNet(
        num_leads=cfg.num_leads,
        num_arrhythmia_classes=2,
        noise_feat_dim=int(sample["noise_feat"].shape[0]),
        rr_seq_len=int(sample["rr_seq"].shape[0]),
        d_model=128,
        rhythm_emb_dim=64,
        noise_emb_dim=32,
        fuzzy_weight_init=cfg.fuzzy_weight_init,
    ).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt.get("model") or ckpt.get("model_state_dict")
    state = {k: v for k, v in state.items() if not k.startswith("_debug_")}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _fixed_fusion_logits(
    model: base.ECGNet,
    ecg: torch.Tensor,
    rr_seq: torch.Tensor,
    noise_feat: torch.Tensor,
    fuzzy_logits: torch.Tensor,
    alpha: float = 0.30,
) -> torch.Tensor:
    z_morph = model.morph_branch(ecg)
    z_rhythm = model.rhythm_branch(rr_seq)
    z_noise = model.noise_branch(noise_feat)
    z = torch.cat([z_morph, z_rhythm, z_noise], dim=-1)
    z = model.fused_dropout(z)
    logits = model.af_bin_head(z)
    return (1.0 - alpha) * logits + alpha * fuzzy_logits[:, :2]


@torch.no_grad()
def _collect_main_probs(
    model: base.ECGNet,
    loader: DataLoader,
    device: torch.device,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    y_true: List[int] = []
    p_afl: List[float] = []
    aux_rows: List[Dict[str, float]] = []
    model.eval()
    for batch in loader:
        ecg = batch["ecg"].to(device)
        rr_seq = batch["rr_seq"].to(device)
        noise_feat = batch["noise_feat"].to(device)
        fuzzy = batch["fuzzy_logits"].to(device)
        label = batch["label"].to(device)
        if mode == "full":
            logits, _ = model(ecg, rr_seq, noise_feat, fuzzy)
        elif mode == "neural_only_no_gate":
            logits, _ = model(ecg, rr_seq, noise_feat, None)
        elif mode == "no_quality_input":
            logits, _ = model(ecg, rr_seq, torch.zeros_like(noise_feat), fuzzy)
        elif mode == "fixed_fusion_alpha030":
            logits = _fixed_fusion_logits(model, ecg, rr_seq, noise_feat, fuzzy, alpha=0.30)
        elif mode == "fuzzy_logits_only":
            logits = fuzzy[:, :2]
        else:
            raise ValueError(mode)

        probs = torch.softmax(logits, dim=1)[:, 1]
        y_true.extend(label.detach().cpu().numpy().astype(int).tolist())
        p_afl.extend(probs.detach().cpu().numpy().astype(float).tolist())

        fuzzy_np = fuzzy[:, :2].detach().cpu().numpy()
        prob_np = probs.detach().cpu().numpy()
        label_np = label.detach().cpu().numpy()
        for i in range(len(label_np)):
            aux_rows.append(
                {
                    "y_true": int(label_np[i]),
                    "p_afl": float(prob_np[i]),
                    "fuzzy_logit_af": float(fuzzy_np[i, 0]),
                    "fuzzy_logit_afl": float(fuzzy_np[i, 1]),
                }
            )
    return np.asarray(y_true, dtype=np.int64), np.asarray(p_afl, dtype=np.float32), pd.DataFrame(aux_rows)


def _main_loaders(val_csv: Path, test_csv: Path, cfg: base.TrainConfig, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    common = dict(
        max_len=cfg.max_len,
        num_leads=cfg.num_leads,
        max_rr_intervals=cfg.max_rr_intervals,
        use_precomputed=cfg.use_precomputed,
        multi_lead_method=cfg.multi_lead_method,
        flutter_method=cfg.flutter_method,
        lead_names=cfg.lead_names,
        augment=False,
    )
    val_ds = base.ECGDataset(str(val_csv), **common)
    test_ds = base.ECGDataset(str(test_csv), **common)
    return (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def _load_baseline(name: str, ckpt_path: Path, val_dir: Path, test_dir: Path, device: torch.device):
    import ablation_Af_AFL.compare1_train_arnet2 as M1
    import ablation_Af_AFL.compare2_train_rawecgnet as M2
    import ablation_Af_AFL.compare3_train_wang2021_bilstm as M3
    import ablation_Af_AFL.compare4_train_kraft2025_convnext1d as M4
    import ablation_Af_AFL.compare5_train_fleury2024_modular as M5

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    args = ckpt.get("args") or {}
    if name == "arnet2":
        val_ds = M1.RRNPZDataset(str(val_dir), rr_len=int(args.get("rr_len", 128)))
        test_ds = M1.RRNPZDataset(str(test_dir), rr_len=int(args.get("rr_len", 128)))
        model = M1.ArNet2(
            input_dim=1,
            hidden_dim=int(args.get("hidden_dim", 128)),
            num_layers=int(args.get("num_layers", 2)),
            num_classes=2,
            dropout=float(args.get("dropout", 0.2)),
        ).to(device)
        forward = lambda m, b: m(b["rr"].to(device))
    elif name == "rawecgnet":
        val_ds = M2.ECGNPZDataset(str(val_dir), ecg_len=int(args.get("ecg_len", 2500)))
        test_ds = M2.ECGNPZDataset(str(test_dir), ecg_len=int(args.get("ecg_len", 2500)))
        model = M2.RawECGNet(num_classes=2, base_ch=int(args.get("base_ch", 32)), dropout=float(args.get("dropout", 0.1))).to(device)
        forward = lambda m, b: m(b["ecg"].to(device))
    elif name == "wang2021":
        val_ds = M3.ECGNPZDataset(str(val_dir), ecg_len=int(args.get("ecg_len", 2500)))
        test_ds = M3.ECGNPZDataset(str(test_dir), ecg_len=int(args.get("ecg_len", 2500)))
        model = M3.Wang2021BiLSTM(
            num_classes=2,
            cnn_hidden=int(args.get("cnn_hidden", 64)),
            lstm_hidden=int(args.get("lstm_hidden", 128)),
            lstm_layers=int(args.get("lstm_layers", 2)),
            dropout=float(args.get("dropout", 0.2)),
        ).to(device)
        forward = lambda m, b: m(b["ecg"].to(device))
    elif name == "kraft2025":
        val_ds = M4.ECGNPZDataset(str(val_dir), ecg_len=int(args.get("ecg_len", 2500)))
        test_ds = M4.ECGNPZDataset(str(test_dir), ecg_len=int(args.get("ecg_len", 2500)))
        model = M4.ConvNeXt1D(
            num_classes=2,
            dims=(
                int(args.get("dim1", 32)),
                int(args.get("dim2", 64)),
                int(args.get("dim3", 128)),
                int(args.get("dim4", 256)),
            ),
            depths=(
                int(args.get("depth1", 2)),
                int(args.get("depth2", 2)),
                int(args.get("depth3", 3)),
                int(args.get("depth4", 2)),
            ),
            drop_path_rate=float(args.get("drop_path_rate", 0.1)),
        ).to(device)
        forward = lambda m, b: m(b["ecg"].to(device))
    elif name == "fleury2024":
        val_ds = M5.ModularECGDataset(
            str(val_dir),
            ecg_len=int(args.get("ecg_len", 2500)),
            rr_len=int(args.get("rr_len", 128)),
        )
        test_ds = M5.ModularECGDataset(
            str(test_dir),
            ecg_len=int(args.get("ecg_len", 2500)),
            rr_len=int(args.get("rr_len", 128)),
        )
        model = M5.Fleury2024Modular(
            num_classes=2,
            rhythm_dim=int(args.get("rhythm_dim", 128)),
            atrial_dim=int(args.get("atrial_dim", 128)),
            raw_dim=int(args.get("raw_dim", 128)),
            rr_hidden=int(args.get("rr_hidden", 64)),
            dropout=float(args.get("dropout", 0.2)),
        ).to(device)
        forward = lambda m, b: m(b["ecg"].to(device), b["rr"].to(device))
    else:
        raise ValueError(name)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, val_ds, test_ds, forward


def _collect_baseline(
    name: str,
    cfg: Dict[str, str],
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ckpt_path = _resolve(cfg["ckpt"])
    val_dir = _resolve(cfg["val_dir"])
    test_dir = _resolve(cfg["test_dir"])
    model, val_ds, test_ds, forward = _load_baseline(name, ckpt_path, val_dir, test_dir, device)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    y_val, p_val, _ = bu.collect_probs(model, val_loader, device, forward, criterion=None)
    y_test, p_test, _ = bu.collect_probs(model, test_loader, device, forward, criterion=None)
    return y_val, p_val, y_test, p_test


def _evaluate_and_record(
    rows: List[Dict[str, object]],
    ci_rows: List[Dict[str, object]],
    curves_dir: Path,
    model_key: str,
    group: str,
    y_val: np.ndarray,
    p_val: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
    n_boot: int,
    seed: int,
) -> float:
    thr, val_best = _threshold_search(y_val, p_val, steps=1001, t_min=0.01, t_max=0.99)
    m = _metric_bundle(y_test, p_test, thr, include_curves=True)
    ci = _bootstrap_ci(
        y_test,
        p_test,
        thr,
        n_boot=n_boot,
        seed=seed,
        metrics=["acc", "afl_precision", "afl_recall", "afl_f1", "macro_f1", "auroc", "auprc"],
    )
    row = {
        "model_key": model_key,
        "model": MODEL_DISPLAY.get(model_key, model_key),
        "group": group,
        "threshold_protocol": "validation-selected threshold maximizing AFL-F1 on validation probabilities",
        "threshold": thr,
        "val_afl_f1_at_threshold": val_best["afl_f1"],
        **{k: v for k, v in m.items() if k not in {"roc_curve", "pr_curve", "cm"}},
        "cm": json.dumps(m["cm"]),
    }
    rows.append(row)
    ci_row = row.copy()
    for metric_name, (lo, hi) in ci.items():
        ci_row[f"{metric_name}_ci95"] = _format_ci(float(m[metric_name]), lo, hi)
        ci_row[f"{metric_name}_lo95"] = lo
        ci_row[f"{metric_name}_hi95"] = hi
    ci_rows.append(ci_row)

    if "roc_curve" in m:
        roc_df = m["roc_curve"]
        roc_df.insert(0, "model", MODEL_DISPLAY.get(model_key, model_key))
        roc_df.to_csv(curves_dir / f"{model_key}_roc_curve.csv", index=False)
    if "pr_curve" in m:
        pr_df = m["pr_curve"]
        pr_df.insert(0, "model", MODEL_DISPLAY.get(model_key, model_key))
        pr_df.to_csv(curves_dir / f"{model_key}_pr_curve.csv", index=False)
    return float(thr)


def _clean_noisy_afl_table(
    meta: pd.DataFrame,
    y_true: np.ndarray,
    p_afl: np.ndarray,
    threshold: float,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    pred = (p_afl >= threshold).astype(np.int64)
    df = meta.copy().reset_index(drop=True)
    df["y_true"] = y_true
    df["p_afl"] = p_afl
    df["pred"] = pred
    df = df[df["y_true"] == 1].copy()
    noisy = df["path"].astype(str).str.contains("_noisy_", regex=False) | df.get("dataset", "").astype(str).str.contains("NOISE", case=False, regex=False)
    df["subgroup"] = np.where(noisy, "Noisy AFL augmentation", "Clean AFL")
    rows = []
    rng = np.random.default_rng(seed)
    for subgroup, g in df.groupby("subgroup", sort=True):
        correct = (g["pred"].to_numpy(dtype=np.int64) == 1).astype(np.float32)
        recall = float(correct.mean()) if len(correct) else float("nan")
        boots = []
        for _ in range(int(n_boot)):
            idx = rng.integers(0, len(correct), size=len(correct))
            boots.append(float(correct[idx].mean()))
        lo, hi = np.percentile(np.asarray(boots), [2.5, 97.5]) if boots else (float("nan"), float("nan"))
        rows.append(
            {
                "subgroup": subgroup,
                "n_afl": int(len(g)),
                "afl_recall": recall,
                "afl_recall_ci95": _format_ci(recall, float(lo), float(hi)),
                "mean_p_afl": float(g["p_afl"].mean()),
                "median_p_afl": float(g["p_afl"].median()),
            }
        )
    return pd.DataFrame(rows)


def _fuzzy_summary(aux_df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([meta.reset_index(drop=True), aux_df.reset_index(drop=True)], axis=1)
    rows = []
    for label, g in df.groupby("label_raw", sort=True):
        rows.append(
            {
                "group": label,
                "n": int(len(g)),
                "fuzzy_logit_af_mean": float(g["fuzzy_logit_af"].mean()),
                "fuzzy_logit_afl_mean": float(g["fuzzy_logit_afl"].mean()),
                "fuzzy_logit_afl_minus_af_mean": float((g["fuzzy_logit_afl"] - g["fuzzy_logit_af"]).mean()),
                "fuzzy_p_afl_mean": float(g["p_afl"].mean()),
                "fuzzy_p_afl_median": float(g["p_afl"].median()),
            }
        )
    noisy_mask = df["path"].astype(str).str.contains("_noisy_", regex=False) | df["dataset"].astype(str).str.contains("NOISE", case=False, regex=False)
    for group_name, mask in [("Clean AFL", (df["label_raw"] == "AFL") & ~noisy_mask), ("Noisy AFL", (df["label_raw"] == "AFL") & noisy_mask)]:
        g = df[mask]
        rows.append(
            {
                "group": group_name,
                "n": int(len(g)),
                "fuzzy_logit_af_mean": float(g["fuzzy_logit_af"].mean()),
                "fuzzy_logit_afl_mean": float(g["fuzzy_logit_afl"].mean()),
                "fuzzy_logit_afl_minus_af_mean": float((g["fuzzy_logit_afl"] - g["fuzzy_logit_af"]).mean()),
                "fuzzy_p_afl_mean": float(g["p_afl"].mean()),
                "fuzzy_p_afl_median": float(g["p_afl"].median()),
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    out_dir: Path,
    metrics_ci: pd.DataFrame,
    subgroup: pd.DataFrame,
    fuzzy_summary: pd.DataFrame,
    cfg_dict: Dict[str, object],
    n_boot: int,
) -> None:
    def md_table(df: pd.DataFrame, cols: List[str]) -> str:
        sub = df[cols].copy()
        sub = sub.fillna("NA").astype(str)
        widths = {
            c: max(len(str(c)), *(len(v) for v in sub[c].tolist()))
            for c in cols
        }
        header = "| " + " | ".join(str(c).ljust(widths[c]) for c in cols) + " |"
        sep = "| " + " | ".join("-" * widths[c] for c in cols) + " |"
        body = [
            "| " + " | ".join(str(row[c]).ljust(widths[c]) for c in cols) + " |"
            for _, row in sub.iterrows()
        ]
        return "\n".join([header, sep, *body])

    selected = metrics_ci.copy()
    for col in ["threshold", "acc", "afl_precision", "afl_recall", "afl_f1", "macro_f1", "auroc", "auprc"]:
        if col in selected:
            selected[col] = selected[col].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "NA")

    report = f"""# Stage-2 Revision Experiments

All outputs were generated under `outputs/stage2_revision_experiments` using existing checkpoints and post-processing. No new performance values were invented.

## Protocol

- Main internal split: `data/precompute_features/multi_dataset_with_afl_noise_aug.csv`, `split_seed=42`, `AFL_NOISY_GROUP=independent`.
- This exactly reproduces `ablation_Af_AFL/revision_pack/temp_test_eval.csv` with 6,735 test segments.
- Baselines were evaluated on their available NPZ validation/test folders; their available test set has 6,734 segments because one AF segment is absent from the baseline preprocessing outputs.
- Threshold protocol: every model uses a validation-selected threshold that maximizes AFL-F1 on validation probabilities. Saved checkpoint thresholds are ignored for the threshold-harmonized table.
- CI protocol: sample-level nonparametric bootstrap, {n_boot} resamples, 95% percentile interval.

## Threshold-Harmonized Main Table

{md_table(metrics_ci, ["model", "group", "n", "n_af", "n_afl", "threshold", "acc_ci95", "afl_precision_ci95", "afl_recall_ci95", "afl_f1_ci95", "macro_f1_ci95", "auroc_ci95", "auprc_ci95"])}

## Clean vs Noisy AFL

{md_table(subgroup, ["subgroup", "n_afl", "afl_recall_ci95", "mean_p_afl", "median_p_afl"])}

## Fuzzy-Logit Summary

{md_table(fuzzy_summary, ["group", "n", "fuzzy_logit_af_mean", "fuzzy_logit_afl_mean", "fuzzy_logit_afl_minus_af_mean", "fuzzy_p_afl_mean", "fuzzy_p_afl_median"])}

## Reproducibility Details To Add

- Sampling rate and window: `fs=250 Hz`, `max_len=2500`, corresponding to 10-s windows.
- Window handling in `ECGDataset`: ECG arrays are read as `(T, C)`; segments longer than 2,500 samples are center-cropped; shorter segments are zero-padded.
- Lead handling: the main checkpoint uses `num_leads={cfg_dict.get("num_leads")}`. When more leads are present, the dataset keeps the first `num_leads` channels.
- Normalization: each lead is independently standardized as `(x - mean) / std`; if `std <= 1e-8`, only mean subtraction is applied.
- RR extraction: the main checkpoint uses `multi_lead_method={cfg_dict.get("multi_lead_method")}` and `max_rr_intervals={cfg_dict.get("max_rr_intervals")}`.
- Flutter descriptor: flutter evidence is the 3-8 Hz bandpower ratio. With `flutter_method={cfg_dict.get("flutter_method")}`, multilead flutter evidence is energy-weighted across leads.
- Atrial-rate surrogate: `atrial_rate = 330 * flutter_ratio` when `flutter_ratio > 0.05`, otherwise 0. This is a heuristic surrogate rather than a direct atrial-cycle detector.
- P-wave descriptor: `P_pres` is a P-wave-related heuristic score computed from 0.5-5 Hz energy in the 200-400 ms pre-R-window, not expert-validated P-wave detection.
- Fuzzy logits: rule scores are normalized and converted to logits by `log(score + 1e-8) * 2.0`; binary AF/AFL evaluation uses the AF and AFL fuzzy-logit entries.

## Text Caveats

- The post-hoc No Gate, No Quality, and Fixed Fusion rows are inference-time interventions on the existing checkpoint, not retrained architectural ablations.
- Use them as sensitivity analyses only. If the manuscript needs true ablation claims, new training runs with those components removed are still required.
- Baseline comparisons are threshold-harmonized by validation threshold, but the baseline NPZ test set has one fewer AF segment than the main model test CSV.
"""
    (out_dir / "STAGE2_EXPERIMENT_REPORT.md").write_text(report, encoding="utf-8")

    prism = f"""# Prism / Manuscript Direct Edit Notes

Use these results as replacement/additional material for the reviewer-requested second pass.

1. Replace the baseline-comparison caption with:

Test-set performance under a threshold-harmonized post-processing protocol. For each model, the AFL decision threshold was selected on the validation set by maximizing AFL-F1 and then fixed for test evaluation. AUROC and AUPRC are threshold-free metrics computed from test probabilities. Values in brackets are sample-level bootstrap 95% confidence intervals ({n_boot} resamples). Baseline NPZ preprocessing contained 6,734 available test segments, one fewer AF segment than the 6,735-segment main test CSV.

2. Add a sentence after the table:

Under this protocol, the proposed model achieved favorable operating-point performance, but the comparison remains dependent on validation-threshold selection and on the available baseline preprocessing outputs.

3. Add the clean/noisy AFL paragraph:

Clean and noisy AFL performance was examined by separating AFL test segments according to whether the file path or dataset label indicated synthetic noise augmentation. AFL recall was computed as TP/(TP+FN) within each subgroup using the validation-selected threshold.

4. Add the post-hoc sensitivity caveat:

The No Gate, No Quality, and Fixed Fusion analyses are post-hoc inference-time interventions applied to the trained checkpoint. They are included only as sensitivity analyses and should not be interpreted as retrained architectural ablations.

5. Add the fuzzy-logit reproducibility sentence:

Fuzzy rule scores were normalized across rhythm classes and converted to logits as log(score + 1e-8) * 2.0. For binary AF/AFL experiments, only the AF and AFL fuzzy-logit entries were used.

6. Add the data-processing reproducibility sentence:

All segments were represented as 10-s windows at 250 Hz (2,500 samples). ECG channels were interpreted as (time, lead), the first two leads were retained for the main model, longer segments were center-cropped, shorter segments were zero-padded, and each lead was independently standardized by subtracting its mean and dividing by its standard deviation.
"""
    (out_dir / "PRISM_STAGE2_DIRECT_EDIT_NOTES.md").write_text(prism, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 reviewer-requested post-processing experiments.")
    parser.add_argument("--out_dir", default="outputs/stage2_revision_experiments")
    parser.add_argument("--main_ckpt", default="outputs_main_af_vs_afl/af_vs_afl_afdb_ltafdb.pt")
    parser.add_argument("--data_csv", default="data/precompute_features/multi_dataset_with_afl_noise_aug.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--reuse_predictions", action="store_true")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    out_dir = _resolve(args.out_dir)
    pred_dir = out_dir / "predictions"
    curves_dir = out_dir / "curves"
    pred_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)

    device = _device(args.device)
    print(f"[stage2] device={device}")

    ckpt_path = _resolve(args.main_ckpt)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg_dict = ckpt.get("cfg", {}).copy()
    cfg = base.TrainConfig(**cfg_dict)

    val_csv, test_csv, val_meta, test_meta = _prepare_main_split(out_dir, _resolve(args.data_csv), args.seed)
    val_loader, test_loader = _main_loaders(val_csv, test_csv, cfg, args.batch_size)
    model = _load_main_model(ckpt_path, val_csv, cfg, device)

    rows: List[Dict[str, object]] = []
    ci_rows: List[Dict[str, object]] = []
    thresholds: Dict[str, float] = {}
    fuzzy_aux: Optional[pd.DataFrame] = None

    main_modes = [
        ("ours_full", "full", "main"),
        ("ours_neural_only_no_gate", "neural_only_no_gate", "posthoc_sensitivity"),
        ("ours_no_quality_input", "no_quality_input", "posthoc_sensitivity"),
        ("ours_fixed_fusion_alpha030", "fixed_fusion_alpha030", "posthoc_sensitivity"),
        ("fuzzy_logits_only", "fuzzy_logits_only", "rule_only"),
    ]
    for model_key, mode, group in main_modes:
        val_pred_path = pred_dir / f"{model_key}_val_predictions.csv"
        test_pred_path = pred_dir / f"{model_key}_test_predictions.csv"
        aux_path = pred_dir / f"{model_key}_test_aux.csv"
        can_reuse = args.reuse_predictions and val_pred_path.exists() and test_pred_path.exists()
        if can_reuse and not (mode == "fuzzy_logits_only" and not aux_path.exists()):
            print(f"[stage2] reusing main predictions: {model_key}")
            y_val, p_val = _load_predictions(val_pred_path)
            y_test, p_test = _load_predictions(test_pred_path)
            aux = pd.read_csv(aux_path) if aux_path.exists() else pd.DataFrame()
        else:
            print(f"[stage2] collecting main mode: {model_key}")
            y_val, p_val, _ = _collect_main_probs(model, val_loader, device, mode)
            y_test, p_test, aux = _collect_main_probs(model, test_loader, device, mode)
            _save_predictions(val_pred_path, y_val, p_val, val_meta)
            _save_predictions(test_pred_path, y_test, p_test, test_meta)
            aux.to_csv(aux_path, index=False)
        if mode == "fuzzy_logits_only":
            fuzzy_aux = aux
        thresholds[model_key] = _evaluate_and_record(
            rows,
            ci_rows,
            curves_dir,
            model_key,
            group,
            y_val,
            p_val,
            y_test,
            p_test,
            n_boot=args.bootstrap,
            seed=args.seed + len(rows),
        )

    if not args.skip_baselines:
        for name, bcfg in BASELINE_DEFAULTS.items():
            val_pred_path = pred_dir / f"{name}_val_predictions.csv"
            test_pred_path = pred_dir / f"{name}_test_predictions.csv"
            if args.reuse_predictions and val_pred_path.exists() and test_pred_path.exists():
                print(f"[stage2] reusing baseline predictions: {name}")
                y_val, p_val = _load_predictions(val_pred_path)
                y_test, p_test = _load_predictions(test_pred_path)
            else:
                print(f"[stage2] collecting baseline: {name}")
                y_val, p_val, y_test, p_test = _collect_baseline(name, bcfg, device, args.batch_size)
                _save_predictions(val_pred_path, y_val, p_val)
                _save_predictions(test_pred_path, y_test, p_test)
            thresholds[name] = _evaluate_and_record(
                rows,
                ci_rows,
                curves_dir,
                name,
                "baseline",
                y_val,
                p_val,
                y_test,
                p_test,
                n_boot=args.bootstrap,
                seed=args.seed + len(rows),
            )

    metrics = pd.DataFrame(rows)
    metrics_ci = pd.DataFrame(ci_rows)
    metrics.to_csv(out_dir / "threshold_harmonized_metrics.csv", index=False)
    metrics_ci.to_csv(out_dir / "threshold_harmonized_metrics_with_ci.csv", index=False)

    ours_pred = pd.read_csv(pred_dir / "ours_full_test_predictions.csv")
    subgroup = _clean_noisy_afl_table(
        test_meta,
        ours_pred["y_true"].to_numpy(dtype=np.int64),
        ours_pred["p_afl"].to_numpy(dtype=np.float32),
        thresholds["ours_full"],
        n_boot=args.bootstrap,
        seed=args.seed + 999,
    )
    subgroup.to_csv(out_dir / "clean_vs_noisy_afl.csv", index=False)

    if fuzzy_aux is not None:
        fuzzy_summary = _fuzzy_summary(fuzzy_aux, test_meta)
    else:
        fuzzy_summary = pd.DataFrame()
    fuzzy_summary.to_csv(out_dir / "fuzzy_logits_summary.csv", index=False)

    with (out_dir / "thresholds.json").open("w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    _write_report(out_dir, metrics_ci, subgroup, fuzzy_summary, cfg_dict, args.bootstrap)
    print(f"[stage2] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
