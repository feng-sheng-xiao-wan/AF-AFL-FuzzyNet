# process_afdb_ltafdb.py
"""
处理 AFDB 和 LTAFDB 数据集，支持诊断级到动态数据的适配

数据集特点：
- AFDB (MIT-BIH Atrial Fibrillation Database): 诊断级，10小时记录，采样率250Hz，2导联
- LTAFDB (Long-Term AF Database): 动态监测，24-25小时记录，采样率128Hz，2导联

处理策略：
1. 诊断级数据（AFDB）：直接切分为10秒窗口
2. 动态数据（LTAFDB）：滑动窗口切分，增强噪声处理
3. 统一采样率、窗口长度
4. 生成CSV索引文件，支持域适应训练
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat
import warnings
warnings.filterwarnings('ignore')

# 尝试导入wfdb（MIT-BIH数据库读取）
try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    print("Warning: wfdb not available. Install with: pip install wfdb")


# ============================================================================
# 数据预处理函数
# ============================================================================

def resample_ecg(ecg: np.ndarray, orig_fs: float, target_fs: float = 250.0) -> np.ndarray:
    """
    重采样ECG信号到目标采样率
    
    Args:
        ecg: ECG信号，形状 (T,) 或 (T, C)
        orig_fs: 原始采样率
        target_fs: 目标采样率（默认250Hz）
    
    Returns:
        重采样后的ECG信号
    """
    if orig_fs == target_fs:
        return ecg
    
    ecg = np.asarray(ecg, dtype=np.float32)
    is_1d = ecg.ndim == 1
    
    if is_1d:
        ecg = ecg[:, None]
    
    T, C = ecg.shape
    num_samples = int(T * target_fs / orig_fs)
    
    resampled = np.zeros((num_samples, C), dtype=np.float32)
    for c in range(C):
        resampled[:, c] = signal.resample(ecg[:, c], num_samples)
    
    if is_1d:
        return resampled[:, 0]
    return resampled


def remove_baseline_drift(ecg: np.ndarray, fs: float, cutoff: float = 0.5) -> np.ndarray:
    """
    去除基线漂移（高通滤波）
    
    Args:
        ecg: ECG信号
        fs: 采样率
        cutoff: 截止频率（Hz），默认0.5Hz
    
    Returns:
        去基线后的ECG信号
    """
    ecg = np.asarray(ecg, dtype=np.float32)
    is_1d = ecg.ndim == 1
    
    if is_1d:
        ecg = ecg[:, None]
    
    # 设计高通滤波器（Butterworth）
    nyquist = fs / 2.0
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
    
    filtered = np.zeros_like(ecg)
    for c in range(ecg.shape[1]):
        filtered[:, c] = signal.filtfilt(b, a, ecg[:, c])
    
    if is_1d:
        return filtered[:, 0]
    return filtered


def remove_powerline_noise(ecg: np.ndarray, fs: float, freq: float = 50.0, Q: float = 30.0) -> np.ndarray:
    """
    去除工频干扰（陷波滤波）
    
    Args:
        ecg: ECG信号
        fs: 采样率
        freq: 工频频率（50Hz或60Hz）
        Q: 品质因数，控制带宽
    
    Returns:
        去工频后的ECG信号
    """
    ecg = np.asarray(ecg, dtype=np.float32)
    is_1d = ecg.ndim == 1
    
    if is_1d:
        ecg = ecg[:, None]
    
    # 设计陷波滤波器
    b, a = signal.iirnotch(freq, Q, fs)
    
    filtered = np.zeros_like(ecg)
    for c in range(ecg.shape[1]):
        filtered[:, c] = signal.filtfilt(b, a, ecg[:, c])
    
    if is_1d:
        return filtered[:, 0]
    return filtered


def preprocess_ecg(
    ecg: np.ndarray,
    fs: float,
    target_fs: float = 250.0,
    remove_baseline: bool = True,
    remove_powerline: bool = True,
    powerline_freq: float = 50.0,
) -> Tuple[np.ndarray, float]:
    """
    完整的ECG预处理流程
    
    Args:
        ecg: ECG信号
        fs: 原始采样率
        target_fs: 目标采样率
        remove_baseline: 是否去除基线漂移
        remove_powerline: 是否去除工频干扰
        powerline_freq: 工频频率
    
    Returns:
        (处理后的ECG, 最终采样率)
    """
    ecg = np.asarray(ecg, dtype=np.float32)
    
    # 1. 去除基线漂移（在重采样前，避免混叠）
    if remove_baseline:
        ecg = remove_baseline_drift(ecg, fs)
    
    # 2. 去除工频干扰
    if remove_powerline:
        ecg = remove_powerline_noise(ecg, fs, powerline_freq)
    
    # 3. 重采样到目标采样率
    if fs != target_fs:
        ecg = resample_ecg(ecg, fs, target_fs)
        fs = target_fs
    
    return ecg, fs


def segment_ecg(
    ecg: np.ndarray,
    fs: float,
    window_sec: float = 10.0,
    overlap_sec: float = 0.0,
    min_window_sec: float = 5.0,
) -> List[Tuple[np.ndarray, int, int]]:
    """
    将长ECG信号切分为固定长度的窗口
    
    Args:
        ecg: ECG信号，形状 (T,) 或 (T, C)
        fs: 采样率
        window_sec: 窗口长度（秒）
        overlap_sec: 重叠长度（秒），0表示无重叠
        min_window_sec: 最小窗口长度（秒），小于此长度的片段会被丢弃
    
    Returns:
        List of (segment, start_idx, end_idx)
    """
    ecg = np.asarray(ecg, dtype=np.float32)
    is_1d = ecg.ndim == 1
    
    if is_1d:
        ecg = ecg[:, None]
    
    T = ecg.shape[0]
    window_samples = int(window_sec * fs)
    overlap_samples = int(overlap_sec * fs)
    step_samples = window_samples - overlap_samples
    min_window_samples = int(min_window_sec * fs)
    
    segments = []
    start = 0
    
    while start + window_samples <= T:
        end = start + window_samples
        segment = ecg[start:end, :].copy()
        
        if is_1d:
            segment = segment[:, 0]
        
        segments.append((segment, start, end))
        start += step_samples
    
    # 处理最后一个片段（如果足够长）
    if start < T and (T - start) >= min_window_samples:
        segment = ecg[start:, :].copy()
        # 如果不够长，进行零填充
        if segment.shape[0] < window_samples:
            pad_len = window_samples - segment.shape[0]
            pad = np.zeros((pad_len, segment.shape[1]), dtype=np.float32)
            segment = np.concatenate([segment, pad], axis=0)
        
        if is_1d:
            segment = segment[:, 0]
        
        segments.append((segment, start, T))
    
    return segments


# ============================================================================
# AFDB 数据处理
# ============================================================================

def load_afdb_record(record_path: str, record_name: str) -> Tuple[np.ndarray, float, Dict]:
    """
    加载AFDB记录
    
    Args:
        record_path: 记录文件路径（不含扩展名）
        record_name: 记录名称（如 '04015'）
    
    Returns:
        (ecg信号, 采样率, 标注信息)
    """
    if not WFDB_AVAILABLE:
        raise ImportError("wfdb is required to load AFDB records")
    
    # 读取记录
    record = wfdb.rdrecord(record_path, channels=[0, 1])  # 读取2个导联
    annotation = wfdb.rdann(record_path, 'atr')  # 读取标注
    
    # 获取ECG数据
    ecg = record.p_signal.astype(np.float32)  # (T, 2)
    fs = record.fs
    
    # 解析标注
    # AFDB标注类型：'N'=正常, 'AFIB'=房颤, 'AFL'=房扑, 'J'=交界性, 'SVTA'=室上速等
    annotation_info = {
        'symbol': annotation.symbol,
        'sample': annotation.sample,
        'aux_note': annotation.aux_note if hasattr(annotation, 'aux_note') else None,
    }
    
    return ecg, fs, annotation_info


def process_afdb_record(
    record_path: str,
    record_name: str,
    output_dir: str,
    window_sec: float = 10.0,
    target_fs: float = 250.0,
    overlap_sec: float = 0.0,
) -> List[Dict]:
    """
    处理单个AFDB记录，切分为窗口并保存
    
    Returns:
        List of dict with segment info for CSV
    """
    print(f"  Processing AFDB record: {record_name}")
    
    # 加载记录
    ecg, fs, ann_info = load_afdb_record(record_path, record_name)
    
    # 预处理
    ecg, fs = preprocess_ecg(ecg, fs, target_fs=target_fs)
    
    # 切分窗口
    segments = segment_ecg(ecg, fs, window_sec=window_sec, overlap_sec=overlap_sec)
    
    # 为每个窗口分配标签
    segment_info = []
    ann_samples = ann_info['sample']
    ann_symbols = ann_info['symbol']
    
    for seg_idx, (segment, start_idx, end_idx) in enumerate(segments):
        # 确定窗口的主要标签（基于窗口中心点的标注）
        center_idx = (start_idx + end_idx) // 2
        
        # 找到最近的标注
        nearest_ann_idx = np.argmin(np.abs(ann_samples - center_idx))
        nearest_ann_sample = ann_samples[nearest_ann_idx]
        nearest_symbol = ann_symbols[nearest_ann_idx]
        
        # 映射标签
        label_map = {
            'N': 'Normal',
            'NSR': 'Normal',
            'AFIB': 'AF',
            'AFL': 'AFL',
            'SVTA': 'PSVT',
            'J': 'Normal',  # 交界性心律归为Normal
        }
        
        label_raw = label_map.get(nearest_symbol, 'Normal')
        
        # 如果窗口内标注变化频繁，可能需要更复杂的策略
        # 这里简化处理：使用中心点标注
        
        # 保存片段
        seg_name = f"{record_name}_seg{seg_idx:04d}"
        seg_path = os.path.join(output_dir, f"{seg_name}.npz")
        
        np.savez_compressed(
            seg_path,
            ecg=segment,
            fs=fs,
            record_name=record_name,
            start_idx=start_idx,
            end_idx=end_idx,
        )
        
        segment_info.append({
            'path': seg_path,
            'label_raw': label_raw,
            'record_name': record_name,
            'segment_idx': seg_idx,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'dataset': 'AFDB',
        })
    
    return segment_info


def process_afdb_dataset(
    afdb_dir: str,
    output_dir: str,
    window_sec: float = 10.0,
    target_fs: float = 250.0,
    overlap_sec: float = 0.0,
    record_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    处理整个AFDB数据集
    
    Args:
        afdb_dir: AFDB数据目录
        output_dir: 输出目录
        window_sec: 窗口长度（秒）
        target_fs: 目标采样率
        overlap_sec: 重叠长度（秒）
        record_list: 要处理的记录列表，None表示处理所有
    
    Returns:
        DataFrame with segment information
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有记录文件
    if record_list is None:
        # 自动查找所有.hea文件
        record_files = list(Path(afdb_dir).glob("*.hea"))
        record_list = [f.stem for f in record_files]
    
    all_segments = []
    
    for record_name in record_list:
        record_path = os.path.join(afdb_dir, record_name)
        
        if not os.path.exists(f"{record_path}.hea"):
            print(f"  Warning: Record {record_name} not found, skipping")
            continue
        
        try:
            segments = process_afdb_record(
                record_path, record_name, output_dir,
                window_sec=window_sec,
                target_fs=target_fs,
                overlap_sec=overlap_sec,
            )
            all_segments.extend(segments)
        except Exception as e:
            print(f"  Error processing {record_name}: {e}")
            continue
    
    df = pd.DataFrame(all_segments)
    return df


# ============================================================================
# LTAFDB 数据处理（动态数据，需要特殊处理）
# ============================================================================

def load_ltafdb_record(record_path: str, record_name: str) -> Tuple[np.ndarray, float, Dict]:
    """
    加载LTAFDB记录（MAT格式）
    
    Args:
        record_path: 记录文件路径（.mat文件）
        record_name: 记录名称
    
    Returns:
        (ecg信号, 采样率, 标注信息)
    """
    mat_data = loadmat(record_path)
    
    # LTAFDB的MAT文件结构可能不同，需要根据实际情况调整
    # 常见字段：'val', 'data', 'ecg'
    ecg = None
    fs = 128.0  # LTAFDB默认采样率
    
    # 尝试不同的字段名
    for key in ['val', 'data', 'ecg', 'ECG']:
        if key in mat_data:
            ecg = mat_data[key]
            break
    
    if ecg is None:
        # 如果找不到，尝试第一个非元数据的数组
        for key, val in mat_data.items():
            if not key.startswith('__') and isinstance(val, np.ndarray):
                if val.ndim >= 2 and val.size > 1000:
                    ecg = val
                    break
    
    if ecg is None:
        raise ValueError(f"Cannot find ECG data in {record_path}")
    
    # 确保是2D数组 (T, C)
    if ecg.ndim == 1:
        ecg = ecg[:, None]
    elif ecg.ndim > 2:
        ecg = ecg.reshape(ecg.shape[0], -1)
    
    # 检查采样率
    if 'fs' in mat_data:
        fs = float(mat_data['fs'])
    elif 'Fs' in mat_data:
        fs = float(mat_data['Fs'])
    elif 'sampling_rate' in mat_data:
        fs = float(mat_data['sampling_rate'])
    
    # 尝试读取标注（如果有）
    annotation_info = {}
    for key in ['annotation', 'ann', 'label', 'labels']:
        if key in mat_data:
            annotation_info[key] = mat_data[key]
            break
    
    return ecg.astype(np.float32), fs, annotation_info


def process_ltafdb_record(
    record_path: str,
    record_name: str,
    output_dir: str,
    window_sec: float = 10.0,
    target_fs: float = 250.0,
    overlap_sec: float = 5.0,  # 动态数据使用重叠以增加样本
    min_snr_db: float = 10.0,  # 最小信噪比阈值
) -> List[Dict]:
    """
    处理单个LTAFDB记录（动态数据，需要更鲁棒的处理）
    
    Args:
        min_snr_db: 最小信噪比（dB），低于此值的窗口会被标记为噪声
    
    Returns:
        List of dict with segment info for CSV
    """
    print(f"  Processing LTAFDB record: {record_name}")
    
    # 加载记录
    ecg, fs, ann_info = load_ltafdb_record(record_path, record_name)
    
    # 动态数据需要更强的预处理
    ecg, fs = preprocess_ecg(
        ecg, fs,
        target_fs=target_fs,
        remove_baseline=True,
        remove_powerline=True,
    )
    
    # 切分窗口（使用重叠）
    segments = segment_ecg(ecg, fs, window_sec=window_sec, overlap_sec=overlap_sec)
    
    segment_info = []
    
    for seg_idx, (segment, start_idx, end_idx) in enumerate(segments):
        # 计算信噪比（简单估计）
        # 使用频域方法：主频带能量 / 噪声频带能量
        ecg_1d = segment[:, 0] if segment.ndim == 2 else segment
        ecg_1d = ecg_1d - ecg_1d.mean()
        
        # FFT
        fft_vals = np.fft.rfft(ecg_1d)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(len(ecg_1d), d=1.0/fs)
        
        # 主频带（5-40 Hz）和噪声频带（0-0.5 Hz + 20-45 Hz）
        main_band = (freqs >= 5) & (freqs <= 40)
        noise_band = ((freqs >= 0) & (freqs <= 0.5)) | ((freqs >= 20) & (freqs <= 45))
        
        main_power = power[main_band].sum()
        noise_power = power[noise_band].sum() + 1e-10
        
        snr_db = 10 * np.log10(main_power / noise_power)
        
        # 如果信噪比太低，标记为噪声
        if snr_db < min_snr_db:
            noise_label = 1  # 噪声
        else:
            noise_label = 0  # 清洁
        
        # 动态数据的标签通常需要从标注文件获取
        # 这里简化处理，需要根据实际LTAFDB格式调整
        label_raw = 'Normal'  # 默认，需要根据标注更新
        
        # 如果有标注信息，使用标注
        if ann_info:
            # 根据窗口位置查找标注
            # 这里需要根据实际标注格式实现
            pass
        
        # 保存片段
        seg_name = f"{record_name}_seg{seg_idx:04d}"
        seg_path = os.path.join(output_dir, f"{seg_name}.npz")
        
        np.savez_compressed(
            seg_path,
            ecg=segment,
            fs=fs,
            record_name=record_name,
            start_idx=start_idx,
            end_idx=end_idx,
            snr_db=snr_db,
        )
        
        segment_info.append({
            'path': seg_path,
            'label_raw': label_raw,
            'noise_label': noise_label,
            'record_name': record_name,
            'segment_idx': seg_idx,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'snr_db': snr_db,
            'dataset': 'LTAFDB',
        })
    
    return segment_info


def process_ltafdb_dataset(
    ltafdb_dir: str,
    output_dir: str,
    window_sec: float = 10.0,
    target_fs: float = 250.0,
    overlap_sec: float = 5.0,
    min_snr_db: float = 10.0,
    record_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    处理整个LTAFDB数据集（动态数据）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有MAT文件
    if record_list is None:
        mat_files = list(Path(ltafdb_dir).glob("*.mat"))
        record_list = [f.stem for f in mat_files]
    
    all_segments = []
    
    for record_name in record_list:
        record_path = os.path.join(ltafdb_dir, f"{record_name}.mat")
        
        if not os.path.exists(record_path):
            print(f"  Warning: Record {record_name} not found, skipping")
            continue
        
        try:
            segments = process_ltafdb_record(
                record_path, record_name, output_dir,
                window_sec=window_sec,
                target_fs=target_fs,
                overlap_sec=overlap_sec,
                min_snr_db=min_snr_db,
            )
            all_segments.extend(segments)
        except Exception as e:
            print(f"  Error processing {record_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    df = pd.DataFrame(all_segments)
    return df


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Process AFDB and LTAFDB datasets for domain adaptation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--afdb_dir", type=str, default="", help="AFDB dataset directory")
    parser.add_argument("--ltafdb_dir", type=str, default="", help="LTAFDB dataset directory")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="Output directory")
    parser.add_argument("--window_sec", type=float, default=10.0, help="Window length in seconds")
    parser.add_argument("--target_fs", type=float, default=250.0, help="Target sampling rate (Hz)")
    parser.add_argument("--afdb_overlap", type=float, default=0.0, help="AFDB overlap in seconds")
    parser.add_argument("--ltafdb_overlap", type=float, default=5.0, help="LTAFDB overlap in seconds (for dynamic data)")
    parser.add_argument("--min_snr_db", type=float, default=10.0, help="Minimum SNR threshold for LTAFDB (dB)")
    parser.add_argument("--afdb_records", type=str, nargs="+", default=None, help="Specific AFDB records to process")
    parser.add_argument("--ltafdb_records", type=str, nargs="+", default=None, help="Specific LTAFDB records to process")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("AFDB & LTAFDB Dataset Processing")
    print("=" * 80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_dfs = []
    
    # 处理AFDB（诊断级数据）
    if args.afdb_dir:
        print(f"\n[1/2] Processing AFDB (Diagnostic-grade) dataset...")
        print(f"  Input: {args.afdb_dir}")
        afdb_output = os.path.join(args.output_dir, "afdb_segments")
        os.makedirs(afdb_output, exist_ok=True)
        
        afdb_df = process_afdb_dataset(
            args.afdb_dir,
            afdb_output,
            window_sec=args.window_sec,
            target_fs=args.target_fs,
            overlap_sec=args.afdb_overlap,
            record_list=args.afdb_records,
        )
        
        afdb_csv = os.path.join(args.output_dir, "afdb_segments.csv")
        afdb_df.to_csv(afdb_csv, index=False)
        print(f"  ✓ Saved {len(afdb_df)} segments to {afdb_csv}")
        all_dfs.append(afdb_df)
    
    # 处理LTAFDB（动态数据）
    if args.ltafdb_dir:
        print(f"\n[2/2] Processing LTAFDB (Dynamic/Holter) dataset...")
        print(f"  Input: {args.ltafdb_dir}")
        ltafdb_output = os.path.join(args.output_dir, "ltafdb_segments")
        os.makedirs(ltafdb_output, exist_ok=True)
        
        ltafdb_df = process_ltafdb_dataset(
            args.ltafdb_dir,
            ltafdb_output,
            window_sec=args.window_sec,
            target_fs=args.target_fs,
            overlap_sec=args.ltafdb_overlap,
            min_snr_db=args.min_snr_db,
            record_list=args.ltafdb_records,
        )
        
        ltafdb_csv = os.path.join(args.output_dir, "ltafdb_segments.csv")
        ltafdb_df.to_csv(ltafdb_csv, index=False)
        print(f"  ✓ Saved {len(ltafdb_df)} segments to {ltafdb_csv}")
        all_dfs.append(ltafdb_df)
    
    # 合并数据集
    if len(all_dfs) > 1:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_csv = os.path.join(args.output_dir, "combined_segments.csv")
        combined_df.to_csv(combined_csv, index=False)
        print(f"\n  ✓ Combined dataset: {len(combined_df)} segments -> {combined_csv}")
        
        # 统计信息
        print(f"\n  Dataset statistics:")
        print(f"    AFDB segments: {len(afdb_df) if args.afdb_dir else 0}")
        print(f"    LTAFDB segments: {len(ltafdb_df) if args.ltafdb_dir else 0}")
        if 'label_raw' in combined_df.columns:
            print(f"\n  Label distribution:")
            print(combined_df['label_raw'].value_counts().to_string())
        if 'noise_label' in combined_df.columns:
            print(f"\n  Noise label distribution:")
            print(combined_df['noise_label'].value_counts().to_string())
    
    print("\n" + "=" * 80)
    print("Processing completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

