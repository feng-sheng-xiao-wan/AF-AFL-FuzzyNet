import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import noise_aware_ecg_af_afl_simple as base


@dataclass
class ModelDef:
    name: str
    ckpt: str
    disable_fuzzy: bool = False


def _patch_dataset_class(*, disable_fuzzy: bool):
    Original = base.ECGDataset

    class Patched(Original):
        def __getitem__(self, idx):
            b = super().__getitem__(idx)
            if disable_fuzzy:
                b["fuzzy_logits"] = torch.zeros_like(b["fuzzy_logits"])
            return b

    base.ECGDataset = Patched
    return Original


def _load_ckpt(path: str, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


@torch.no_grad()
def _infer_probs_and_labels(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_y = []
    all_p = []
    for batch in loader:
        ecg = batch["ecg"].to(device)
        rr_seq = batch["rr_seq"].to(device)
        noise_feat = batch["noise_feat"].to(device)
        fuzzy_logits = batch.get("fuzzy_logits")
        if fuzzy_logits is not None:
            fuzzy_logits = fuzzy_logits.to(device)
        y = batch["label"].to(device)

        logits, _ = model(ecg, rr_seq, noise_feat, fuzzy_logits)
        probs = torch.softmax(logits, dim=-1)
        all_y.append(y.detach().cpu().numpy().astype(np.int64))
        all_p.append(probs.detach().cpu().numpy().astype(np.float64))
    y_true = np.concatenate(all_y, axis=0)
    probs = np.concatenate(all_p, axis=0)
    return y_true, probs


def _metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    return {
        "n": int(y_true.shape[0]),
        "accuracy": float((y_true == y_pred).mean()) if y_true.size else 0.0,
        "macro_precision": float(np.mean(prec)) if y_true.size else 0.0,
        "macro_recall": float(np.mean(rec)) if y_true.size else 0.0,
        "macro_f1": float(np.mean(f1)) if y_true.size else 0.0,
        "AF_precision": float(prec[0]) if y_true.size else 0.0,
        "AF_recall": float(rec[0]) if y_true.size else 0.0,
        "AF_f1": float(f1[0]) if y_true.size else 0.0,
        "AFL_precision": float(prec[1]) if y_true.size else 0.0,
        "AFL_recall": float(rec[1]) if y_true.size else 0.0,
        "AFL_f1": float(f1[1]) if y_true.size else 0.0,
        "cm00": int(cm[0, 0]),
        "cm01": int(cm[0, 1]),
        "cm10": int(cm[1, 0]),
        "cm11": int(cm[1, 1]),
    }


def _make_bins_from_margin(margin: np.ndarray, q1: float, q2: float) -> np.ndarray:
    # 0=low, 1=mid, 2=high
    t1 = float(np.quantile(margin, q1))
    t2 = float(np.quantile(margin, q2))
    bins = np.zeros_like(margin, dtype=np.int64)
    bins[margin >= t1] = 1
    bins[margin >= t2] = 2
    return bins


def main():
    p = argparse.ArgumentParser(description="Compare confidence bins (pmax-psecond) for Proposed vs NoFuzzy on test split.")
    p.add_argument("--csv_aug", type=str, required=True, help="Augmented CSV used to rebuild test split.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--q1", type=float, default=1 / 3, help="Lower quantile for mid/high split (default 1/3).")
    p.add_argument("--q2", type=float, default=2 / 3, help="Upper quantile for mid/high split (default 2/3).")
    p.add_argument("--out_csv", type=str, default="ablation_Af_AFL/confidence_bins_proposed_vs_nofuzzy.csv")
    p.add_argument("--main_ckpt", type=str, default="outputs_thr_tuned_multi_afl_noise_aug_seg/af_vs_afl_afdb_ltafdb.pt")
    p.add_argument("--nofuzzy_ckpt", type=str, default="outputs/ablation/05_full_no_fuzzy/af_vs_afl_afdb_ltafdb.pt")
    args = p.parse_args()

    device = torch.device(args.device)
    _, _, test_df = base.split_dataset_csv(args.csv_aug, 0.8, 0.1, 0.1, seed=args.seed)

    # write a temp test csv once
    tmp_csv = PROJECT_ROOT / "outputs" / "ablation" / f"temp_test_conf_bins_aug_seed{args.seed}.csv"
    tmp_csv.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(tmp_csv, index=False)

    models = [
        ModelDef(name="proposed", ckpt=args.main_ckpt, disable_fuzzy=False),
        ModelDef(name="no_fuzzy", ckpt=args.nofuzzy_ckpt, disable_fuzzy=True),
    ]

    out_rows = []
    # Use Proposed margins to define bins (same bins for both models)
    proposed_margin = None
    proposed_bins = None
    y_true_ref = None

    cached = {}
    for m in models:
        ckpt_path = str((PROJECT_ROOT / m.ckpt).resolve()) if not os.path.isabs(m.ckpt) else m.ckpt
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
        ckpt = _load_ckpt(ckpt_path, device)
        cfg_dict = ckpt.get("cfg", {})
        cfg = base.TrainConfig(**cfg_dict) if isinstance(cfg_dict, dict) else base.TrainConfig()

        original_cls = _patch_dataset_class(disable_fuzzy=m.disable_fuzzy)
        try:
            ds = base.ECGDataset(
                str(tmp_csv),
                max_len=int(getattr(cfg, "max_len", 2500)),
                num_leads=int(getattr(cfg, "num_leads", 1)),
                max_rr_intervals=int(getattr(cfg, "max_rr_intervals", 32)),
                use_precomputed=bool(getattr(cfg, "use_precomputed", True)),
                recompute_fuzzy_only=bool(getattr(cfg, "recompute_fuzzy_only", False)),
            )
        finally:
            base.ECGDataset = original_cls

        s0 = ds[0]
        rr_len = int(s0["rr_seq"].shape[0])
        noise_dim = int(s0["noise_feat"].shape[0])

        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

        model = base.ECGNet(
            num_leads=int(getattr(cfg, "num_leads", 1)),
            num_arrhythmia_classes=2,
            noise_feat_dim=noise_dim,
            rr_seq_len=rr_len,
            d_model=128,
            rhythm_emb_dim=64,
            noise_emb_dim=32,
            fuzzy_weight_init=float(getattr(cfg, "fuzzy_weight_init", 0.3)),
        ).to(device)
        model.load_state_dict(ckpt["model"], strict=False)

        y_true, probs = _infer_probs_and_labels(model, loader, device)
        cached[m.name] = (y_true, probs)

        if y_true_ref is None:
            y_true_ref = y_true
        else:
            if not np.array_equal(y_true_ref, y_true):
                raise RuntimeError("y_true mismatch across models (dataset order should be identical).")

        # margin = pmax - psecond
        p_sorted = np.sort(probs, axis=1)
        margin = p_sorted[:, -1] - p_sorted[:, -2]
        if m.name == "proposed":
            proposed_margin = margin

    assert proposed_margin is not None and y_true_ref is not None
    proposed_bins = _make_bins_from_margin(proposed_margin, args.q1, args.q2)

    bin_names = {0: "low", 1: "mid", 2: "high"}
    for m in models:
        y_true, probs = cached[m.name]
        pred = np.argmax(probs, axis=1).astype(np.int64)
        # per-bin
        for b in [0, 1, 2]:
            idx = (proposed_bins == b)
            met = _metrics_from_preds(y_true[idx], pred[idx])
            out_rows.append(
                {
                    "model": m.name,
                    "bin": bin_names[b],
                    "q1": float(args.q1),
                    "q2": float(args.q2),
                    **met,
                }
            )

    out_path = PROJECT_ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "bin", "q1", "q2",
        "n",
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "AF_precision", "AF_recall", "AF_f1",
        "AFL_precision", "AFL_recall", "AFL_f1",
        "cm00", "cm01", "cm10", "cm11",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print("Saved:", str(out_path))


if __name__ == "__main__":
    main()

