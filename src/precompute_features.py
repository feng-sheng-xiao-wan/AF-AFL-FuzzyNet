# precompute_features.py
"""
离线预计算特征脚本：
- 从CSV读取所有样本路径
- 计算 rr_seq, noise_feat, fuzzy_logits
- 保存到 npz 文件（与原ECG文件同目录或指定目录）
"""

import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

# 导入主脚本中的特征提取函数
from noise_aware_ecg_af_afl_simple import (
    extract_rr_seq,
    extract_noise_features,
    apply_fuzzy_rules,
    normalize_ecg,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


def precompute_features_for_sample(
    npz_path: str,
    max_rr_intervals: int = 32,
    use_stft_dbscan: bool = False,
    multi_lead_method: str = "max_energy",
    flutter_method: str = "fixed_lead",
    lead_names: list = None,
    output_dir: str = None,
    num_classes: int = 4,
) -> dict:
    """
    为单个样本预计算特征
    
    返回：
        {
            "success": bool,
            "rr_seq": np.ndarray or None,
            "noise_feat": np.ndarray or None,
            "fuzzy_logits": np.ndarray or None,
            "error": str or None,
        }
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(npz_path):
            return {"success": False, "error": f"File not found: {npz_path}"}
        
        # 检查文件大小（如果文件太小可能是损坏的）
        file_size = os.path.getsize(npz_path)
        if file_size < 100:  # NPZ文件至少应该有几百字节
            return {"success": False, "error": f"File too small ({file_size} bytes), possibly corrupted: {npz_path}"}
        
        # 尝试加载文件，使用更安全的参数
        try:
            data = np.load(npz_path, allow_pickle=False)
        except (OSError, ValueError, EOFError) as e:
            # CRC错误或其他文件损坏错误
            error_msg = str(e)
            if "CRC" in error_msg or "corrupt" in error_msg.lower():
                return {"success": False, "error": f"Corrupted NPZ file (CRC error): {npz_path}. File size: {file_size} bytes"}
            else:
                return {"success": False, "error": f"Failed to load NPZ file: {npz_path}. Error: {error_msg}"}
        
        if "ecg" not in data:
            available_keys = list(data.keys())
            data.close()  # 关闭文件
            return {"success": False, "error": f"No 'ecg' key in {npz_path}. Available keys: {available_keys}"}
        
        # 尝试读取数据，捕获可能的CRC错误
        try:
            ecg = np.asarray(data["ecg"])
        except (OSError, ValueError, EOFError) as e:
            error_msg = str(e)
            data.close()
            if "CRC" in error_msg:
                return {"success": False, "error": f"Corrupted NPZ file (CRC error reading 'ecg'): {npz_path}"}
            else:
                return {"success": False, "error": f"Failed to read 'ecg' from {npz_path}: {error_msg}"}
        
        # 尝试读取fs，如果失败则使用默认值
        try:
            if "fs" in data:
                fs = float(data["fs"])
            else:
                fs = 250.0
        except (OSError, ValueError, EOFError) as e:
            # fs读取失败，使用默认值并记录警告
            fs = 250.0
            # 不返回错误，因为fs有默认值，但记录警告
            print(f"  Warning: Failed to read 'fs' from {npz_path}, using default 250.0 Hz")
        
        data.close()  # 关闭文件以释放资源
        
        # 统一为 (T, C)
        if ecg.ndim == 1:
            ecg = ecg[:, None]
        elif ecg.ndim == 2:
            T, C = ecg.shape
            if T < C:  # (C, T)
                ecg = ecg.T
        
        # 1. 计算 RR 序列
        rr_seq = extract_rr_seq(
            ecg,
            fs=fs,
            max_intervals=max_rr_intervals,
            multi_lead_method=multi_lead_method,
        )
        
        # 2. 计算噪声特征（使用原始ECG，归一化前）
        noise_feat = extract_noise_features(ecg, fs, use_stft_dbscan=use_stft_dbscan)
        
        # 3. 计算模糊规则 logits
        fuzzy_logits = apply_fuzzy_rules(
            rr_seq,
            ecg=ecg,
            fs=fs,
            flutter_method=flutter_method,
            lead_names=lead_names,
        )
        fuzzy_logits_np = fuzzy_logits.detach().cpu().numpy()
        
        # 对于二分类任务（AF vs AFL），只保留前2个logits
        if num_classes == 2 and fuzzy_logits_np.shape[0] == 4:
            fuzzy_logits_np = fuzzy_logits_np[:2]  # 只保留 [AF_logit, AFL_logit]
        
        return {
            "success": True,
            "rr_seq": rr_seq,
            "noise_feat": noise_feat,
            "fuzzy_logits": fuzzy_logits_np,
            "error": None,
        }
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return {
            "success": False,
            "error": f"Error processing {npz_path}: {str(e)}\nDetails: {error_detail}",
        }


def save_precomputed_features(
    npz_path: str,
    rr_seq: np.ndarray,
    noise_feat: np.ndarray,
    fuzzy_logits: np.ndarray,
    output_dir: str = None,
) -> str:
    """
    保存预计算特征到文件
    
    如果 output_dir 为 None，则保存到与原 npz 文件同目录
    文件名：原文件名 + "_features.npz"
    """
    if output_dir is None:
        # 保存到原文件同目录
        base_path = Path(npz_path)
        output_path = base_path.parent / f"{base_path.stem}_features.npz"
    else:
        # 保存到指定目录
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(npz_path).stem
        output_path = Path(output_dir) / f"{base_name}_features.npz"
    
    np.savez_compressed(
        str(output_path),
        rr_seq=rr_seq,
        noise_feat=noise_feat,
        fuzzy_logits=fuzzy_logits,
    )
    
    return str(output_path)


def precompute_features_batch(
    csv_path: str,
    max_rr_intervals: int = 32,
    use_stft_dbscan: bool = False,
    multi_lead_method: str = "max_energy",
    flutter_method: str = "fixed_lead",
    lead_names: list = None,
    output_dir: str = None,
    num_workers: int = 0,
    batch_size: int = 100,
    num_classes: int = 2,
):
    """
    批量预计算特征
    """
    print("=" * 80)
    print("Precomputing Features for ECG Dataset")
    print("=" * 80)
    
    df = pd.read_csv(csv_path)
    if "path" not in df.columns:
        raise ValueError("CSV must contain 'path' column")
    
    print(f"\n[1/4] Loading CSV: {csv_path}")
    print(f"  - Total samples: {len(df)}")
    
    # 自动检测数据集类型（如果num_classes未指定或为默认值）
    if num_classes == 2:
        # 检查CSV中的标签，判断是否为二分类数据集
        label_col = None
        if "label_raw" in df.columns:
            label_col = "label_raw"
        elif "label" in df.columns:
            label_col = "label"
        
        if label_col:
            unique_labels = df[label_col].dropna().unique()
            # 如果只有AF和AFL，自动设置为二分类
            if set(unique_labels).issubset({"AF", "AFL", "af", "afl", 0, 1, "0", "1"}):
                num_classes = 2
                print(f"  - Auto-detected: Binary classification (AF vs AFL) based on labels: {unique_labels}")
            else:
                print(f"  - Auto-detected: Multi-class classification based on labels: {unique_labels}")
        else:
            print(f"  - No label column found, using default num_classes={num_classes}")
    
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"  - Output directory: {output_dir}")
    else:
        print(f"  - Features will be saved next to original .npz files")
    
    print(f"\n[2/4] Configuration:")
    print(f"  - max_rr_intervals: {max_rr_intervals}")
    print(f"  - use_stft_dbscan: {use_stft_dbscan}")
    print(f"  - multi_lead_method: {multi_lead_method}")
    print(f"  - flutter_method: {flutter_method}")
    print(f"  - num_classes: {num_classes} (Binary AF vs AFL)")
    if lead_names:
        print(f"  - lead_names: {lead_names}")
    
    print(f"\n[3/4] Processing samples...")
    
    success_count = 0
    error_count = 0
    corrupted_files = []  # 记录损坏的文件
    feature_paths = []
    
    # 处理每个样本
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Precomputing"):
        npz_path = row["path"]
        
        if not os.path.exists(npz_path):
            print(f"  Warning: File not found: {npz_path}")
            error_count += 1
            feature_paths.append("")
            continue
        
        result = precompute_features_for_sample(
            npz_path,
            max_rr_intervals=max_rr_intervals,
            use_stft_dbscan=use_stft_dbscan,
            multi_lead_method=multi_lead_method,
            flutter_method=flutter_method,
            lead_names=lead_names,
            output_dir=output_dir,
            num_classes=num_classes,
        )
        
        if result["success"]:
            # 保存特征
            try:
                feat_path = save_precomputed_features(
                    npz_path,
                    result["rr_seq"],
                    result["noise_feat"],
                    result["fuzzy_logits"],
                    output_dir=output_dir,
                )
                feature_paths.append(feat_path)
                success_count += 1
            except Exception as e:
                print(f"  Error saving features for {npz_path}: {str(e)}")
                error_count += 1
                feature_paths.append("")
        else:
            # 检查是否是CRC错误（文件损坏）
            error_msg = result.get('error', '')
            if 'CRC' in error_msg or 'corrupt' in error_msg.lower() or 'Corrupted' in error_msg:
                corrupted_files.append(npz_path)
            
            # 只打印前20个错误的详细信息，之后只打印简要信息
            if error_count < 20:
                print(f"  [{idx+1}/{len(df)}] Error: {result['error']}")
            elif error_count == 20:
                print(f"  ... (suppressing further error details, only showing count)")
            error_count += 1
            feature_paths.append("")
    
    # 更新CSV，添加特征路径列
    print(f"\n[4/4] Updating CSV with feature paths...")
    df["feature_path"] = feature_paths
    
    # 保存更新后的CSV
    output_csv = csv_path.replace(".csv", "_with_features.csv")
    if output_dir:
        output_csv = os.path.join(output_dir, os.path.basename(output_csv))
    df.to_csv(output_csv, index=False)
    
    print(f"\n" + "=" * 80)
    print(f"Precomputation Complete!")
    print(f"  - Success: {success_count}/{len(df)}")
    print(f"  - Errors: {error_count}/{len(df)}")
    if corrupted_files:
        print(f"  - Corrupted files: {len(corrupted_files)}")
        # 保存损坏文件列表
        corrupted_list_path = csv_path.replace(".csv", "_corrupted_files.txt")
        if output_dir:
            corrupted_list_path = os.path.join(output_dir, os.path.basename(corrupted_list_path))
        with open(corrupted_list_path, 'w', encoding='utf-8') as f:
            for fpath in corrupted_files:
                f.write(f"{fpath}\n")
        print(f"  - Corrupted files list saved to: {corrupted_list_path}")
    print(f"  - Updated CSV: {output_csv}")
    print("=" * 80)
    
    return output_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute features (rr_seq, noise_feat, fuzzy_logits) for ECG dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=str, required=True, help="Input CSV with 'path' column")
    parser.add_argument("--max_rr_intervals", type=int, default=32, help="Max RR intervals")
    parser.add_argument("--use_stft_dbscan", action="store_true", default=False, help="Enable STFT+DBSCAN")
    parser.add_argument(
        "--multi_lead_method",
        type=str,
        default="max_energy",
        choices=["max_energy", "voting"],
        help="Multi-lead RR detection method",
    )
    parser.add_argument(
        "--flutter_method",
        type=str,
        default="fixed_lead",
        choices=["fixed_lead", "attention"],
        help="Multi-lead flutter detection method",
    )
    parser.add_argument(
        "--lead_names",
        type=str,
        nargs="+",
        default=None,
        help="Lead names (e.g., I II V1) for fixed_lead method",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for features (if None, save next to original files)",
    )
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers (not used yet)")
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        choices=[2],
        help="Number of classes for AF/AFL binary features.",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    precompute_features_batch(
        csv_path=args.csv,
        max_rr_intervals=args.max_rr_intervals,
        use_stft_dbscan=args.use_stft_dbscan,
        multi_lead_method=args.multi_lead_method,
        flutter_method=args.flutter_method,
        lead_names=args.lead_names,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
    )


if __name__ == "__main__":
    main()





