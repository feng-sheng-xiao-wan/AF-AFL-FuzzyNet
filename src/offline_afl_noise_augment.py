import os
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

try:
    import wfdb
except ImportError:
    wfdb = None

from precompute_features import (
    precompute_features_for_sample,
    save_precomputed_features,
)


def load_nstdb_noise(nstdb_dir: str, target_fs: float = 250.0) -> dict:
    """
    读取 MIT-BIH Noise Stress Test Database 的三种噪声，并重采样到 target_fs。
    
    返回：
        {
            "bw": np.ndarray (T,),
            "em": np.ndarray (T,),
            "ma": np.ndarray (T,),
        }
    """
    if wfdb is None:
        raise ImportError("wfdb 未安装，无法读取 MIT-BIH NSTDB 噪声。请先 pip install wfdb")

    noise_records = ["bw", "em", "ma"]
    noises = {}

    for rec in noise_records:
        rec_path = os.path.join(nstdb_dir, rec)
        if not (os.path.exists(rec_path + ".dat") or os.path.exists(rec_path + ".hea")):
            raise FileNotFoundError(f"找不到噪声记录 {rec} (期望 {rec_path}.dat / .hea)")

        # 读取噪声记录（通常是 2 导联，fs=360Hz）
        sig, fields = wfdb.rdsamp(rec_path)
        # wfdb.rdsamp 返回的 fields 在不同版本中可能是 dict 或具名对象，这里统一用 dict 访问
        if isinstance(fields, dict):
            fs_src = float(fields.get("fs", 360.0))
        else:
            fs_src = float(getattr(fields, "fs", 360.0))
        # 转成单通道（简单取平均）
        if sig.ndim == 2 and sig.shape[1] > 1:
            sig_1d = sig.mean(axis=1)
        else:
            sig_1d = sig.squeeze()

        # 重采样到 target_fs
        if abs(fs_src - target_fs) < 1e-3:
            resampled = sig_1d.astype(np.float32)
        else:
            # 简单线性重采样
            import math

            duration = sig_1d.shape[0] / fs_src
            tgt_len = int(math.ceil(duration * target_fs))
            x_old = np.linspace(0.0, duration, num=sig_1d.shape[0], endpoint=False)
            x_new = np.linspace(0.0, duration, num=tgt_len, endpoint=False)
            resampled = np.interp(x_new, x_old, sig_1d).astype(np.float32)

        noises[rec] = resampled
        print(f"Loaded NSTDB noise '{rec}': src_fs={fs_src}, target_fs={target_fs}, len={len(resampled)}")

    return noises


def add_noise_with_snr(ecg: np.ndarray, noise: np.ndarray, fs: float, snr_db: float) -> np.ndarray:
    """
    按目标 SNR(dB) 把噪声叠加到 ecg 上。
    ecg: (T,) or (T,C) - 我们对每个导联叠加同一个 1D 噪声。
    noise: (T_noise,)
    """
    ecg_arr = np.asarray(ecg, dtype=np.float32)
    if ecg_arr.ndim == 1:
        T, C = ecg_arr.shape[0], 1
        ecg_arr = ecg_arr[:, None]
    else:
        T, C = ecg_arr.shape
        if T < C:
            ecg_arr = ecg_arr.T
            T, C = ecg_arr.shape

    if len(noise) < T:
        # 噪声不够长就循环使用
        reps = int(np.ceil(T / len(noise)))
        noise_long = np.tile(noise, reps)[:T]
    else:
        # 随机截取一段
        start = np.random.randint(0, len(noise) - T + 1)
        noise_long = noise[start:start + T]

    # 计算信号和噪声的 RMS
    sig_rms = np.sqrt(np.mean(ecg_arr ** 2))
    noise_rms = np.sqrt(np.mean(noise_long ** 2))
    if noise_rms < 1e-8:
        return ecg_arr  # 噪声太小，直接返回

    # 根据 SNR = 20 log10 (sig_rms / (k * noise_rms)) 求 k
    snr_linear = 10 ** (snr_db / 20.0)
    k = sig_rms / (snr_linear * noise_rms)
    noise_scaled = (k * noise_long).astype(np.float32)

    # 对每个导联叠加同一噪声
    ecg_noisy = ecg_arr + noise_scaled[:, None]
    return ecg_noisy.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="离线为 AFL 片段生成带噪版本，并重新预计算特征（基于 NSTDB 噪声）"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="原始多数据集 CSV（例如 data/precompute_features/multi_dataset_combined_with_features.csv）",
    )
    parser.add_argument(
        "--nstdb_dir",
        type=str,
        required=True,
        help="MIT-BIH Noise Stress Test Database 解压目录（包含 bw.dat/em.dat/ma.dat）",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="扩增后新的 CSV 路径（例如 data/precompute_features/multi_dataset_with_afl_noise_aug.csv）",
    )
    parser.add_argument(
        "--holter_root",
        type=str,
        default="data/holter",
        help="原始 ECG npz 所在根目录（用于在其下新建带噪目录）",
    )
    parser.add_argument(
        "--noise_subdir",
        type=str,
        default="afl_noise_aug",
        help="在 holter_root 下新建的带噪目录名（例如 afl_noise_aug）",
    )
    parser.add_argument(
        "--features_output_dir",
        type=str,
        default="data/precompute_features/afl_noise_aug_features",
        help="预计算特征输出目录",
    )
    parser.add_argument(
        "--snr_min",
        type=float,
        default=6.0,
        help="噪声叠加的最小 SNR(dB)",
    )
    parser.add_argument(
        "--snr_max",
        type=float,
        default=18.0,
        help="噪声叠加的最大 SNR(dB)",
    )
    parser.add_argument(
        "--num_aug_per_afl",
        type=int,
        default=2,
        help="每个 AFL 片段生成多少个带噪版本",
    )
    parser.add_argument(
        "--target_fs",
        type=float,
        default=250.0,
        help="训练用的目标采样率，用于把 NSTDB 噪声重采样到该频率",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "path" not in df.columns or "label_raw" not in df.columns:
        raise ValueError("CSV 必须包含 'path' 和 'label_raw' 两列")

    # 载入 NSTDB 噪声
    noises = load_nstdb_noise(args.nstdb_dir, target_fs=args.target_fs)
    noise_keys = list(noises.keys())

    holter_root = Path(args.holter_root)
    noise_ecg_root = holter_root / args.noise_subdir
    noise_ecg_root.mkdir(parents=True, exist_ok=True)

    features_output_dir = Path(args.features_output_dir)
    features_output_dir.mkdir(parents=True, exist_ok=True)

    new_rows = []

    print(f"总样本数: {len(df)}, 其中 AFL 样本: {(df['label_raw']=='AFL').sum()}")

    for idx, row in df.iterrows():
        label_raw = str(row["label_raw"])
        new_rows.append(row)  # 先保留原始行

        if label_raw.upper() != "AFL":
            continue

        orig_path = row["path"]
        if not os.path.exists(orig_path):
            print(f"[跳过] 找不到原始 npz: {orig_path}")
            continue

        # 读取原始 ecg 和 fs
        try:
            data = np.load(orig_path, allow_pickle=False)
        except Exception as e:
            print(f"[跳过] 无法读取 {orig_path}: {e}")
            continue

        if "ecg" not in data:
            print(f"[跳过] npz 中无 'ecg': {orig_path}")
            data.close()
            continue

        ecg = np.asarray(data["ecg"])
        if "fs" in data:
            fs = float(data["fs"])
        else:
            fs = args.target_fs
        data.close()

        # 如果 fs 与 target_fs 差异很大，简单提示一下（此处暂不对 ECG 重采样，假定原始就是 target_fs 对应的）
        if abs(fs - args.target_fs) > 1.0:
            print(f"Warning: {orig_path} fs={fs}, target_fs={args.target_fs}, 暂未对 ECG 做重采样")

        base_name = Path(orig_path).stem

        for aug_id in range(args.num_aug_per_afl):
            # 随机选择一种噪声类型
            noise_type = np.random.choice(noise_keys)
            noise_vec = noises[noise_type]
            # 随机 SNR
            snr_db = float(np.random.uniform(args.snr_min, args.snr_max))

            ecg_noisy = add_noise_with_snr(ecg, noise_vec, fs=fs, snr_db=snr_db)

            # 保存新的带噪 npz：放在 holter_root/noise_subdir 下，目录结构扁平化即可
            noisy_filename = f"{base_name}_noisy_{noise_type}_snr{int(round(snr_db))}_v{aug_id}.npz"
            noisy_path = noise_ecg_root / noisy_filename

            np.savez_compressed(
                str(noisy_path),
                ecg=ecg_noisy,
                fs=fs,
            )

            # 为新 npz 预计算特征
            feat = precompute_features_for_sample(
                str(noisy_path),
                max_rr_intervals=32,
                use_stft_dbscan=False,
                multi_lead_method="max_energy",
                flutter_method="fixed_lead",
                lead_names=None,
                output_dir=None,
                num_classes=2,
            )
            if not feat["success"]:
                print(f"[特征失败] {noisy_path}: {feat['error']}")
                continue

            feat_path = save_precomputed_features(
                str(noisy_path),
                feat["rr_seq"],
                feat["noise_feat"],
                feat["fuzzy_logits"],
                output_dir=str(features_output_dir),
            )

            # 复制一行 CSV，并修改 path / feature_path / record_name / dataset 等字段
            new_row = row.copy()
            new_row["path"] = str(noisy_path).replace("\\", "/")
            if "feature_path" in new_row.index:
                new_row["feature_path"] = str(feat_path).replace("\\", "/")
            # 每个噪声扩充视为新患者：用带噪文件名（不含 .npz）作为唯一 record_name，划分时不会与原始患者混在一起
            new_row["record_name"] = Path(noisy_path).stem  # 如 001_seg0001_noisy_em_snr15_v0
            if "dataset" in new_row.index:
                new_row["dataset"] = str(new_row["dataset"]) + "_AFL_NOISE"

            new_rows.append(new_row)

    df_out = pd.DataFrame(new_rows)
    out_dir = Path(args.output_csv).parent
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output_csv, index=False)
    print(f"扩增完成，原始样本数 {len(df)}，扩增后样本数 {len(df_out)}")
    print(f"输出 CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

