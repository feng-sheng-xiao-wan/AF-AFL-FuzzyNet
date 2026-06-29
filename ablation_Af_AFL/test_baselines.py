"""
Evaluate baseline checkpoints with fair AFL-threshold protocol (same as Ours).
Usage:
  python ablation_Af_AFL/test_baselines.py --baseline all --out_csv ablation_Af_AFL/baseline_test_metrics.csv
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ablation_Af_AFL import baseline_utils as bu


def _load_ckpt(ckpt_path: str, device: torch.device):
    return torch.load(ckpt_path, map_location=device, weights_only=False)


def _eval_baseline(name: str, ckpt_path: str, val_dir: str, test_dir: str, device, batch_size: int = 32):
    import ablation_Af_AFL.compare1_train_arnet2 as M1
    import ablation_Af_AFL.compare2_train_rawecgnet as M2
    import ablation_Af_AFL.compare3_train_wang2021_bilstm as M3
    import ablation_Af_AFL.compare4_train_kraft2025_convnext1d as M4
    import ablation_Af_AFL.compare5_train_fleury2024_modular as M5

    ckpt = _load_ckpt(ckpt_path, device)
    args = ckpt.get("args") or {}

    if name == "arnet2":
        rr_len = int(args.get("rr_len", 128))
        hidden_dim = int(args.get("hidden_dim", 128))
        num_layers = int(args.get("num_layers", 2))
        dropout = float(args.get("dropout", 0.2))
        val_ds = M1.RRNPZDataset(val_dir, rr_len=rr_len)
        test_ds = M1.RRNPZDataset(test_dir, rr_len=rr_len)
        model = M1.ArNet2(
            input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=2, dropout=dropout
        ).to(device)
        forward = lambda m, b: m(b["rr"].to(device))
    elif name == "rawecgnet":
        ecg_len = int(args.get("ecg_len", 2500))
        base_ch = int(args.get("base_ch", 32))
        dropout = float(args.get("dropout", 0.1))
        val_ds = M2.ECGNPZDataset(val_dir, ecg_len=ecg_len)
        test_ds = M2.ECGNPZDataset(test_dir, ecg_len=ecg_len)
        model = M2.RawECGNet(num_classes=2, base_ch=base_ch, dropout=dropout).to(device)
        forward = lambda m, b: m(b["ecg"].to(device))
    elif name == "wang2021":
        ecg_len = int(args.get("ecg_len", 2500))
        val_ds = M3.ECGNPZDataset(val_dir, ecg_len=ecg_len)
        test_ds = M3.ECGNPZDataset(test_dir, ecg_len=ecg_len)
        model = M3.Wang2021BiLSTM(
            num_classes=2,
            cnn_hidden=int(args.get("cnn_hidden", 64)),
            lstm_hidden=int(args.get("lstm_hidden", 128)),
            lstm_layers=int(args.get("lstm_layers", 2)),
            dropout=float(args.get("dropout", 0.2)),
        ).to(device)
        forward = lambda m, b: m(b["ecg"].to(device))
    elif name == "kraft2025":
        ecg_len = int(args.get("ecg_len", 2500))
        val_ds = M4.ECGNPZDataset(val_dir, ecg_len=ecg_len)
        test_ds = M4.ECGNPZDataset(test_dir, ecg_len=ecg_len)
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
        ecg_len = int(args.get("ecg_len", 2500))
        rr_len = int(args.get("rr_len", 128))
        val_ds = M5.ModularECGDataset(val_dir, ecg_len=ecg_len, rr_len=rr_len)
        test_ds = M5.ModularECGDataset(test_dir, ecg_len=ecg_len, rr_len=rr_len)
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
    criterion = torch.nn.CrossEntropyLoss()
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    min_r = float(args.get("min_afl_recall", 0.2))
    thr_saved = ckpt.get("afl_threshold")
    if thr_saved is None:
        val_m = bu.evaluate_split(
            model, val_loader, criterion, device, forward,
            search_threshold=True, min_afl_recall=min_r,
        )
        thr = float(val_m["threshold"])
        thr_source = "val_search"
    else:
        thr = float(thr_saved)
        thr_source = "checkpoint"

    test_m = bu.evaluate_split(model, test_loader, criterion, device, forward, threshold=thr)
    y_test = test_m["y_true"]
    return {
        "baseline": name,
        "error": "",
        "protocol": "AFL-threshold (val-tuned, min_afl_recall=0.2)",
        "threshold": thr,
        "threshold_source": thr_source,
        "test_n": int(len(y_test)),
        "test_n_af": int((y_test == 0).sum()),
        "test_n_afl": int((y_test == 1).sum()),
        "test_acc": f"{test_m['thr_acc']:.6f}",
        "test_afl_precision": f"{test_m['thr_afl_precision']:.6f}",
        "test_afl_recall": f"{test_m['thr_afl_recall']:.6f}",
        "test_afl_f1": f"{test_m['thr_afl_f1']:.6f}",
        "test_macro_f1": f"{test_m['thr_macro_f1']:.6f}",
        "test_argmax_afl_f1": f"{test_m['argmax_afl_f1']:.6f}",
        "test_argmax_macro_f1": f"{test_m['argmax_macro_f1']:.6f}",
        "cm": test_m["thr_cm"].tolist(),
        "report": test_m["thr_report"],
    }


DEFAULTS = {
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


def main():
    parser = argparse.ArgumentParser(description="Fair baseline eval with AFL-threshold")
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        choices=["arnet2", "rawecgnet", "wang2021", "kraft2025", "fleury2024", "all"],
    )
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_csv", type=str, default=None)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    device = torch.device(args.device)
    names = list(DEFAULTS.keys()) if args.baseline == "all" else [args.baseline]
    rows = []

    for name in names:
        cfg = DEFAULTS[name]
        ckpt_path = args.ckpt or str(PROJECT_ROOT / cfg["ckpt"])
        val_dir = args.val_dir or str(PROJECT_ROOT / cfg["val_dir"])
        test_dir = args.test_dir or str(PROJECT_ROOT / cfg["test_dir"])
        if not os.path.isfile(ckpt_path):
            print(f"[{name}] skip: ckpt missing {ckpt_path}")
            rows.append({"baseline": name, "error": "ckpt_not_found"})
            continue
        if not os.path.isdir(test_dir):
            print(f"[{name}] skip: test_dir missing {test_dir}")
            rows.append({"baseline": name, "error": "test_dir_not_found"})
            continue
        print(f"\n{'='*60}\n[{name}] val={val_dir}\n  test={test_dir}\n  ckpt={ckpt_path}\n{'='*60}")
        try:
            row = _eval_baseline(name, ckpt_path, val_dir, test_dir, device, args.batch_size)
        except Exception as e:
            print(f"[{name}] error: {e}")
            import traceback
            traceback.print_exc()
            rows.append({"baseline": name, "error": str(e)[:120]})
            continue
        print(
            f"thr={row['threshold']:.4f} ({row['threshold_source']}) | "
            f"Acc={row['test_acc']} AFL-F1={row['test_afl_f1']} "
            f"(argmax AFL-F1={row['test_argmax_afl_f1']})"
        )
        print(row["report"])
        print("CM (AFL-threshold):")
        print(np.array(row["cm"]))
        rows.append(row)

    if args.out_csv:
        out_path = PROJECT_ROOT / args.out_csv
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "baseline", "error", "protocol", "threshold", "threshold_source",
            "test_n", "test_n_af", "test_n_afl",
            "test_acc", "test_afl_precision", "test_afl_recall", "test_afl_f1", "test_macro_f1",
            "test_argmax_afl_f1", "test_argmax_macro_f1", "cm",
        ]
        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
