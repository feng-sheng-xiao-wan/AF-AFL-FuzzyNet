from __future__ import annotations

import csv
import json
import os
import sys
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import noise_aware_ecg_af_afl_simple as base
from ablation_Af_AFL import baseline_utils as bu


OUT_DIR = ROOT / "outputs" / "strict_main_revision" / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def collect_ours(device: torch.device, batch_size: int = 128) -> dict:
    ckpt_path = ROOT / "outputs_main_af_vs_afl" / "af_vs_afl_afdb_ltafdb.pt"
    test_csv = ROOT / "ablation_Af_AFL" / "revision_pack" / "temp_test_eval.csv"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = base.TrainConfig(**ckpt.get("cfg", {}))
    ds = base.ECGDataset(
        str(test_csv),
        max_len=cfg.max_len,
        num_leads=cfg.num_leads,
        max_rr_intervals=cfg.max_rr_intervals,
        use_precomputed=cfg.use_precomputed,
        augment=False,
    )
    sample = ds[0]
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
    state = {k: v for k, v in ckpt["model"].items() if not k.startswith("_debug_")}
    model.load_state_dict(state, strict=False)
    model.eval()

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    y_true, p_afl = [], []
    for batch in loader:
        y = batch["label"].to(device)
        logits = model(
            batch["ecg"].to(device),
            batch["rr_seq"].to(device),
            batch["noise_feat"].to(device),
            batch["fuzzy_logits"].to(device),
        )
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1)[:, 1]
        y_true.extend(y.cpu().numpy().tolist())
        p_afl.extend(probs.cpu().numpy().tolist())

    return {
        "model": "Ours",
        "y_true": np.asarray(y_true, dtype=np.int64),
        "p_afl": np.asarray(p_afl, dtype=np.float32),
        "threshold": float(ckpt.get("afl_threshold", 0.5)),
        "threshold_source": "checkpoint",
    }


def collect_baseline(name: str, device: torch.device, batch_size: int = 64) -> dict:
    import ablation_Af_AFL.compare1_train_arnet2 as M1
    import ablation_Af_AFL.compare2_train_rawecgnet as M2
    import ablation_Af_AFL.compare3_train_wang2021_bilstm as M3
    import ablation_Af_AFL.compare4_train_kraft2025_convnext1d as M4
    import ablation_Af_AFL.compare5_train_fleury2024_modular as M5

    defaults = {
        "ArNet2": {
            "ckpt": ROOT / "ablation_Af_AFL" / "checkpoints_arnet2" / "best_arnet2.pth",
            "val_dir": ROOT / "data" / "ablation_rr_npz" / "val",
            "test_dir": ROOT / "data" / "ablation_rr_npz" / "test",
        },
        "RawECGNet": {
            "ckpt": ROOT / "ablation_Af_AFL" / "checkpoints_rawecgnet" / "best_rawecgnet.pth",
            "val_dir": ROOT / "ablation_Af_AFL" / "rawecg_npz" / "val",
            "test_dir": ROOT / "ablation_Af_AFL" / "rawecg_npz" / "test",
        },
        "Wang2021": {
            "ckpt": ROOT / "ablation_Af_AFL" / "checkpoints_wang2021" / "best_wang2021_bilstm.pth",
            "val_dir": ROOT / "ablation_Af_AFL" / "rawecg_npz" / "val",
            "test_dir": ROOT / "ablation_Af_AFL" / "rawecg_npz" / "test",
        },
        "ConvNeXt-ECG": {
            "ckpt": ROOT / "ablation_Af_AFL" / "checkpoints_kraft2025" / "best_kraft2025_convnext1d.pth",
            "val_dir": ROOT / "ablation_Af_AFL" / "rawecg_npz" / "val",
            "test_dir": ROOT / "ablation_Af_AFL" / "rawecg_npz" / "test",
        },
        "Fleury2024": {
            "ckpt": ROOT / "ablation_Af_AFL" / "checkpoints_fleury2024" / "best_fleury2024_modular.pth",
            "val_dir": ROOT / "ablation_Af_AFL" / "fleury_ecg_rr_npz" / "val",
            "test_dir": ROOT / "ablation_Af_AFL" / "fleury_ecg_rr_npz" / "test",
        },
    }
    cfg_paths = defaults[name]
    ckpt = torch.load(str(cfg_paths["ckpt"]), map_location=device, weights_only=False)
    args = ckpt.get("args") or {}

    if name == "ArNet2":
        rr_len = int(args.get("rr_len", 128))
        val_ds = M1.RRNPZDataset(str(cfg_paths["val_dir"]), rr_len=rr_len)
        test_ds = M1.RRNPZDataset(str(cfg_paths["test_dir"]), rr_len=rr_len)
        model = M1.ArNet2(
            input_dim=1,
            hidden_dim=int(args.get("hidden_dim", 128)),
            num_layers=int(args.get("num_layers", 2)),
            num_classes=2,
            dropout=float(args.get("dropout", 0.2)),
        ).to(device)
        forward = lambda m, b: m(b["rr"].to(device))
    elif name == "RawECGNet":
        ecg_len = int(args.get("ecg_len", 2500))
        val_ds = M2.ECGNPZDataset(str(cfg_paths["val_dir"]), ecg_len=ecg_len)
        test_ds = M2.ECGNPZDataset(str(cfg_paths["test_dir"]), ecg_len=ecg_len)
        model = M2.RawECGNet(
            num_classes=2,
            base_ch=int(args.get("base_ch", 32)),
            dropout=float(args.get("dropout", 0.1)),
        ).to(device)
        forward = lambda m, b: m(b["ecg"].to(device))
    elif name == "Wang2021":
        ecg_len = int(args.get("ecg_len", 2500))
        val_ds = M3.ECGNPZDataset(str(cfg_paths["val_dir"]), ecg_len=ecg_len)
        test_ds = M3.ECGNPZDataset(str(cfg_paths["test_dir"]), ecg_len=ecg_len)
        model = M3.Wang2021BiLSTM(
            num_classes=2,
            cnn_hidden=int(args.get("cnn_hidden", 64)),
            lstm_hidden=int(args.get("lstm_hidden", 128)),
            lstm_layers=int(args.get("lstm_layers", 2)),
            dropout=float(args.get("dropout", 0.2)),
        ).to(device)
        forward = lambda m, b: m(b["ecg"].to(device))
    elif name == "ConvNeXt-ECG":
        ecg_len = int(args.get("ecg_len", 2500))
        val_ds = M4.ECGNPZDataset(str(cfg_paths["val_dir"]), ecg_len=ecg_len)
        test_ds = M4.ECGNPZDataset(str(cfg_paths["test_dir"]), ecg_len=ecg_len)
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
    elif name == "Fleury2024":
        ecg_len = int(args.get("ecg_len", 2500))
        rr_len = int(args.get("rr_len", 128))
        val_ds = M5.ModularECGDataset(str(cfg_paths["val_dir"]), ecg_len=ecg_len, rr_len=rr_len)
        test_ds = M5.ModularECGDataset(str(cfg_paths["test_dir"]), ecg_len=ecg_len, rr_len=rr_len)
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
    criterion = torch.nn.CrossEntropyLoss()
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    min_r = float(args.get("min_afl_recall", 0.2))
    if ckpt.get("afl_threshold") is None:
        val_m = bu.evaluate_split(
            model, val_loader, criterion, device, forward, search_threshold=True, min_afl_recall=min_r
        )
        thr = float(val_m["threshold"])
        source = "val_search"
    else:
        thr = float(ckpt["afl_threshold"])
        source = "checkpoint"
    y_true, p_afl, _ = bu.collect_probs(model, test_loader, device, forward, criterion)
    return {
        "model": name,
        "y_true": y_true,
        "p_afl": p_afl,
        "threshold": thr,
        "threshold_source": source,
    }


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        y = row["y_true"]
        p = row["p_afl"]
        pred = (p >= row["threshold"]).astype(np.int64)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        try:
            ap = average_precision_score(y, p)
        except Exception:
            ap = float("nan")
        try:
            auc = roc_auc_score(y, p)
        except Exception:
            auc = float("nan")
        out.append(
            {
                "model": row["model"],
                "n": int(len(y)),
                "n_af": int((y == 0).sum()),
                "n_afl": int((y == 1).sum()),
                "threshold": row["threshold"],
                "threshold_source": row["threshold_source"],
                "afl_precision": prec,
                "afl_recall": rec,
                "afl_f1": f1,
                "average_precision": ap,
                "roc_auc": auc,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
    return out


def save_predictions(rows: list[dict]) -> None:
    pred_path = OUT_DIR.parent / "calibration_predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "sample_idx", "label", "prob_afl", "threshold", "threshold_source"])
        for row in rows:
            for i, (label, prob) in enumerate(zip(row["y_true"], row["p_afl"])):
                writer.writerow([row["model"], i, int(label), float(prob), row["threshold"], row["threshold_source"]])


def plot_probability_distribution(rows: list[dict]) -> None:
    n = len(rows)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True, sharey=True)
    axes = axes.ravel()
    bins = np.linspace(0, 1, 31)
    for ax, row in zip(axes, rows):
        y, p = row["y_true"], row["p_afl"]
        ax.hist(p[y == 0], bins=bins, alpha=0.65, density=True, label="AF", color="#4c78a8")
        ax.hist(p[y == 1], bins=bins, alpha=0.65, density=True, label="AFL", color="#f58518")
        ax.axvline(row["threshold"], color="#222222", linestyle="--", linewidth=1.2)
        ax.set_title(row["model"])
        ax.grid(alpha=0.25)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(frameon=False)
    fig.supxlabel("Predicted AFL probability")
    fig.supylabel("Density")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "prediction_probability_distributions.png", dpi=300)
    plt.close(fig)


def plot_pr_roc(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for row in rows:
        y, p = row["y_true"], row["p_afl"]
        precision, recall, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        ap = average_precision_score(y, p)
        auc = roc_auc_score(y, p)
        axes[0].plot(recall, precision, linewidth=1.8, label=f"{row['model']} (AP={ap:.3f})")
        axes[1].plot(fpr, tpr, linewidth=1.8, label=f"{row['model']} (AUC={auc:.3f})")
    pos_rate = np.mean(rows[0]["y_true"] == 1)
    axes[0].axhline(pos_rate, color="#777777", linestyle=":", linewidth=1.0, label="Prevalence")
    axes[1].plot([0, 1], [0, 1], color="#777777", linestyle=":", linewidth=1.0, label="Chance")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-recall curves")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("ROC curves")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pr_roc_curves_all_models.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["Ours", "ArNet2", "RawECGNet", "Wang2021", "ConvNeXt-ECG", "Fleury2024"],
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    device = torch.device(args.device)
    print(f"device={device}", flush=True)
    rows = []
    for name in args.models:
        print(f"collecting {name}", flush=True)
        if name == "Ours":
            rows.append(collect_ours(device))
        else:
            rows.append(collect_baseline(name, device))
        save_predictions(rows)
        with (OUT_DIR.parent / "calibration_summary_partial.json").open("w", encoding="utf-8") as f:
            json.dump(summarize(rows), f, indent=2)
        print(f"finished {name}", flush=True)
    save_predictions(rows)
    plot_probability_distribution(rows)
    plot_pr_roc(rows)
    summary = summarize(rows)
    with (OUT_DIR.parent / "calibration_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
