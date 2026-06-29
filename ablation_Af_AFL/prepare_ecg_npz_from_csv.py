import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# 确保能从任何 cwd 导入项目根目录模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import noise_aware_ecg_af_afl_simple as base


def map_label_binary(row) -> int:
    """
    映射为 RawECGNet 二分类标签：
      0 -> AF
      1 -> AFL
    其他节律返回 -1（跳过）
    """
    if "label_raw" in row and not pd.isna(row["label_raw"]):
        lab = str(row["label_raw"]).upper()
        if lab == "AF":
            return 0
        if lab == "AFL":
            return 1
        return -1

    if "label" in row and not pd.isna(row["label"]):
        lab = int(row["label"])
        # 主脚本中：AF=0, AFL=1, PSVT=2, Normal=3
        if lab == 0:
            return 0
        if lab == 1:
            return 1
        return -1

    return -1


def to_1d_ecg(ecg: np.ndarray) -> np.ndarray:
    """
    将任意形状 ECG 转为单导联 1D。
    与主脚本风格保持一致：
      - 若为 (T,) 则直接返回
      - 若为 (T,C)，若 T<C 则转置，再取第一导联
    """
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
        description="从 multi-dataset CSV 生成 RawECGNet 所需的 ECG+label .npz（train/val/test，AF vs AFL 二分类）"
    )
    parser.add_argument(
        "--combined_csv",
        type=str,
        required=True,
        help="包含 path / label_raw 的 CSV（如 multi_dataset_with_afl_noise_aug.csv）",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="ablation_Af_AFL/rawecg_npz",
        help="输出根目录，内部会创建 train/val/test 子目录",
    )
    args = parser.parse_args()

    csv_path = args.combined_csv
    df = pd.read_csv(csv_path)
    if "path" not in df.columns:
        raise ValueError("CSV 必须包含 'path' 列")

    print(f"Loading CSV: {csv_path}, total samples={len(df)}")

    # 使用主脚本的患者级划分逻辑，保证与主实验一致
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
        for _, row in sub_df.iterrows():
            path = str(row["path"])
            label = map_label_binary(row)
            if label < 0:
                continue  # 跳过非 AF/AFL

            if not os.path.exists(path):
                print(f"[{split_name}] Skip missing ECG npz: {path}")
                continue

            try:
                data = np.load(path, allow_pickle=False)
                if "ecg" not in data:
                    print(f"[{split_name}] Skip: {path} missing 'ecg'")
                    data.close()
                    continue
                ecg = np.asarray(data["ecg"], dtype=np.float32)
                data.close()
            except Exception as e:
                print(f"[{split_name}] Skip invalid ECG npz {path}: {e}")
                continue

            ecg_1d = to_1d_ecg(ecg)

            base_name = Path(path).stem
            out_path = out_dir / f"{base_name}_morphlabel.npz"
            np.savez_compressed(
                out_path,
                ecg=ecg_1d,
                label=int(label),
            )
            count += 1

        print(f"[{split_name}] Saved {count} ECG npz files to {out_dir}")

    process_subset(train_df, train_dir, "train")
    process_subset(val_df, val_dir, "val")
    process_subset(test_df, test_dir, "test")

    print("Done.")


if __name__ == "__main__":
    main()

