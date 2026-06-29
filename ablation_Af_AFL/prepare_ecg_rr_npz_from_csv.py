"""
从 multi-dataset CSV 生成 Fleury2024 (compare5) 所需的 ecg+rr+label .npz（train/val/test，AF vs AFL 二分类）。
与主实验使用相同的患者级 8:1:1 划分。
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import noise_aware_ecg_af_afl_simple as base


def map_label_binary(row) -> int:
    """0=AF, 1=AFL，其他 -1 跳过。"""
    if "label_raw" in row and not pd.isna(row["label_raw"]):
        lab = str(row["label_raw"]).upper()
        if lab == "AF":
            return 0
        if lab == "AFL":
            return 1
        return -1
    if "label" in row and not pd.isna(row["label"]):
        lab = int(row["label"])
        if lab == 0:
            return 0
        if lab == 1:
            return 1
        return -1
    return -1


def to_1d_ecg(ecg: np.ndarray) -> np.ndarray:
    ecg = np.asarray(ecg, dtype=np.float32)
    if ecg.ndim == 1:
        return ecg
    if ecg.ndim == 2:
        T, C = ecg.shape
        if T < C:
            ecg = ecg.T
            T, C = ecg.shape
        return ecg[:, 0]
    raise ValueError(f"Unexpected ecg shape {ecg.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="生成 Fleury2024 所需的 ecg+rr+label .npz（train/val/test）"
    )
    parser.add_argument("--combined_csv", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="ablation_Af_AFL/fleury_ecg_rr_npz",
        help="输出根目录，下建 train/val/test",
    )
    parser.add_argument("--rr_key", type=str, default="rr_seq")
    parser.add_argument("--rr_len", type=int, default=128)
    args = parser.parse_args()

    csv_path = args.combined_csv
    df = pd.read_csv(csv_path)
    if "path" not in df.columns:
        raise ValueError("CSV 必须包含 'path' 列")

    print(f"Loading CSV: {csv_path}, total samples={len(df)}")
    train_df, val_df, test_df = base.split_dataset_csv(csv_path, 0.8, 0.1, 0.1)
    print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    out_root = Path(args.output_root)
    train_dir = out_root / "train"
    val_dir = out_root / "val"
    test_dir = out_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    def process_subset(sub_df: pd.DataFrame, out_dir: Path, split_name: str):
        count = 0
        for _, row in sub_df.iterrows():
            path = str(row["path"])
            label = map_label_binary(row)
            if label < 0:
                continue
            if not os.path.exists(path):
                print(f"[{split_name}] Skip missing: {path}")
                continue

            try:
                data = np.load(path, allow_pickle=False)
                if "ecg" not in data:
                    data.close()
                    continue
                ecg = np.asarray(data["ecg"], dtype=np.float32)
                data.close()
            except Exception as e:
                print(f"[{split_name}] Skip {path}: {e}")
                continue

            ecg_1d = to_1d_ecg(ecg)

            rr = None
            feat_path = row.get("feature_path", None)
            if isinstance(feat_path, str) and feat_path != "" and os.path.exists(feat_path):
                try:
                    feat = np.load(feat_path, allow_pickle=False)
                    if args.rr_key in feat:
                        rr = np.asarray(feat[args.rr_key], dtype=np.float32).reshape(-1)
                    feat.close()
                except Exception:
                    pass
            if rr is None:
                try:
                    data = np.load(path, allow_pickle=False)
                    ecg_raw = np.asarray(data["ecg"])
                    fs = float(data["fs"]) if "fs" in data else 250.0
                    data.close()
                    if ecg_raw.ndim == 1:
                        ecg_raw = ecg_raw[:, None]
                    elif ecg_raw.ndim == 2 and ecg_raw.shape[0] < ecg_raw.shape[1]:
                        ecg_raw = ecg_raw.T
                    rr = base.extract_rr_seq(
                        ecg_raw, fs=fs, max_intervals=args.rr_len, multi_lead_method="max_energy"
                    )
                except Exception as e:
                    print(f"[{split_name}] No RR for {path}: {e}")
                    rr = np.zeros(args.rr_len, dtype=np.float32)
            rr = np.asarray(rr, dtype=np.float32)
            if len(rr) > args.rr_len:
                rr = rr[: args.rr_len]
            elif len(rr) < args.rr_len:
                rr = np.pad(rr, (0, args.rr_len - len(rr)), constant_values=0.0)

            base_name = Path(path).stem
            out_path = out_dir / f"{base_name}_ecgrr.npz"
            np.savez_compressed(out_path, ecg=ecg_1d, rr=rr, label=int(label))
            count += 1

        print(f"[{split_name}] Saved {count} ecg+rr npz to {out_dir}")

    process_subset(train_df, train_dir, "train")
    process_subset(val_df, val_dir, "val")
    process_subset(test_df, test_dir, "test")
    print("Done.")


if __name__ == "__main__":
    main()
