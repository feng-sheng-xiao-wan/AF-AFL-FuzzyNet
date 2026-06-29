import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# 确保可以从任何工作目录导入项目根目录下的模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import noise_aware_ecg_af_afl_simple as base


def map_label(row) -> int:
    """
    映射为 ArNet2 二分类标签：
      0 -> AF
      1 -> AFL
    其他节律返回 -1（后续会跳过，不生成样本）
    """
    if "label_raw" in row and not pd.isna(row["label_raw"]):
        lab = str(row["label_raw"]).upper()
    elif "label" in row and not pd.isna(row["label"]):
        lab = int(row["label"])
        # 主脚本中：AF=0, AFL=1, PSVT=2, Normal=3
        if lab == 0:
            return 0
        if lab == 1:
            return 1
        return -1
    else:
        return -1

    if lab == "AF":
        return 0
    if lab == "AFL":
        return 1
    return -1


def main():
    parser = argparse.ArgumentParser(
        description="从 multi-dataset CSV 生成 ArNet2 所需的 RR .npz（train/val）"
    )
    parser.add_argument(
        "--combined_csv",
        type=str,
        required=True,
        help="包含 path / feature_path / label_raw 的 CSV（如 multi_dataset_with_afl_noise_aug.csv）",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/ablation_rr_npz",
        help="输出根目录，内部会创建 train / val 子目录",
    )
    parser.add_argument(
        "--rr_key",
        type=str,
        default="rr_seq",
        help="预计算特征文件中 RR 序列的 key（默认为 rr_seq）",
    )
    args = parser.parse_args()

    csv_path = args.combined_csv
    df = pd.read_csv(csv_path)
    if "path" not in df.columns:
        raise ValueError("CSV 必须包含 'path' 列")

    print(f"Loading CSV: {csv_path}, total samples={len(df)}")

    # 使用主脚本的患者级划分逻辑，保证与主实验一致（含 test 便于 ArNet2 也做 Test 评估）
    train_df, val_df, test_df = base.split_dataset_csv(csv_path, 0.8, 0.1, 0.1)
    print(f"Split by patient: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    out_root = Path(args.output_root)
    train_dir = out_root / "train"
    val_dir = out_root / "val"
    test_dir = out_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    def process_subset(sub_df: pd.DataFrame, out_dir: Path, split_name: str):
        count = 0
        for idx, row in sub_df.iterrows():
            path = str(row["path"])
            feat_path = row.get("feature_path", None)

            # 从 feature_path 里直接读取预计算的 RR 序列（如果存在）
            rr = None
            if isinstance(feat_path, str) and feat_path != "" and os.path.exists(feat_path):
                try:
                    feat = np.load(feat_path, allow_pickle=False)
                    if args.rr_key in feat:
                        rr = np.asarray(feat[args.rr_key], dtype=np.float32).reshape(-1)
                    feat.close()
                except Exception as e:
                    print(f"[{split_name}] Warning: failed to load {feat_path}: {e}")

            # 若没有 feature_path，则尝试在线从 ECG 重算（退而求其次）
            if rr is None:
                if not os.path.exists(path):
                    print(f"[{split_name}] Skip missing ECG npz: {path}")
                    continue
                try:
                    data = np.load(path, allow_pickle=False)
                    ecg = np.asarray(data["ecg"])
                    fs = float(data["fs"]) if "fs" in data else 250.0
                    data.close()
                except Exception as e:
                    print(f"[{split_name}] Skip invalid ECG npz {path}: {e}")
                    continue
                # 统一为 (T, C)
                if ecg.ndim == 1:
                    ecg = ecg[:, None]
                elif ecg.ndim == 2:
                    T, C = ecg.shape
                    if T < C:
                        ecg = ecg.T
                rr = base.extract_rr_seq(ecg, fs=fs, max_intervals=128, multi_lead_method="max_energy")

            label = map_label(row)
            if label < 0:
                continue  # 跳过非 AF/AFL

            base_name = Path(path).stem
            out_path = out_dir / f"{base_name}_rrlabel.npz"
            np.savez_compressed(
                out_path,
                rr=rr.astype(np.float32),
                label=int(label),
            )
            count += 1

        print(f"[{split_name}] Saved {count} RR npz files to {out_dir}")

    process_subset(train_df, train_dir, "train")
    process_subset(val_df, val_dir, "val")
    process_subset(test_df, test_dir, "test")

    print("Done.")


if __name__ == "__main__":
    main()

