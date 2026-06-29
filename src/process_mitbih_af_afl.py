# -*- coding: utf-8 -*-
"""
处理 MIT-BIH Arrhythmia Database，提取 AF 和 AFL 片段

MIT-BIH Arrhythmia Database:
- 采样率 360Hz，2导联（MLII, V1）
- 记录长度约30分钟
- 标注方式：点标注，标记某个时间点开始的状态
- 需要重采样到 250Hz，然后切分为 10 秒窗口
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    print("Error: wfdb is required. Install with: pip install wfdb")
    sys.exit(1)


def resample_ecg(ecg: np.ndarray, orig_fs: float, target_fs: float = 250.0) -> np.ndarray:
    """重采样ECG信号到目标采样率"""
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
    """去除基线漂移（高通滤波）"""
    ecg = np.asarray(ecg, dtype=np.float32)
    is_1d = ecg.ndim == 1
    
    if is_1d:
        ecg = ecg[:, None]
    
    nyquist = fs / 2.0
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
    
    filtered = np.zeros_like(ecg)
    for c in range(ecg.shape[1]):
        filtered[:, c] = signal.filtfilt(b, a, ecg[:, c])
    
    if is_1d:
        return filtered[:, 0]
    return filtered


def preprocess_ecg(ecg: np.ndarray, fs: float, target_fs: float = 250.0) -> Tuple[np.ndarray, float]:
    """
    预处理ECG信号：
    1. 重采样到目标采样率
    2. 去除基线漂移
    """
    # 重采样
    ecg = resample_ecg(ecg, fs, target_fs)
    
    # 去除基线漂移
    ecg = remove_baseline_drift(ecg, target_fs)
    
    return ecg, target_fs


def extract_af_afl_segments(
    ecg: np.ndarray,
    fs: float,
    ann_samples: np.ndarray,
    aux_note: List,
    ann_symbols: List,
    orig_fs: float,
    window_sec: float = 10.0,
    overlap_sec: float = 0.0,
    min_duration_sec: float = 10.0,
) -> List[Tuple[np.ndarray, str, int, int]]:
    """
    从ECG中提取AF和AFL片段
    
    Args:
        ecg: ECG信号，形状 (T, C) 或 (T,)，已重采样到target_fs
        fs: 采样率（重采样后，即target_fs）
        ann_samples: 标注样本点位置（原始采样率orig_fs下）
        aux_note: aux_note列表
        ann_symbols: symbol列表
        orig_fs: 原始采样率
        window_sec: 窗口长度（秒）
        overlap_sec: 重叠长度（秒）
        min_duration_sec: 最小持续时间（秒），过滤掉太短的标记
    
    Returns:
        List of (segment, label, start_idx, end_idx)
    """
    segments = []
    
    if ecg.ndim == 1:
        ecg = ecg[:, None]
    
    T, C = ecg.shape
    window_samples = int(window_sec * fs)
    overlap_samples = int(overlap_sec * fs)
    step_samples = window_samples - overlap_samples
    min_duration_samples = int(min_duration_sec * orig_fs)  # 在原始采样率下计算
    
    # 找到所有AF和AFL标记点（同时检查aux_note和symbol）
    af_afl_markers = []  # [(sample_idx_original, label, source, note_str)]
    
    # 1. 检查aux_note（主要来源）
    if aux_note is not None:
        for idx in range(len(ann_samples)):
            if idx < len(aux_note) and aux_note[idx]:
                aux_val = str(aux_note[idx]).strip().upper()
                if aux_val:
                    n_clean = aux_val.replace('(', '').replace(')', '').replace('[', '').replace(']', '').strip().upper()
                    
                    # 检测AF
                    if any(keyword in n_clean for keyword in ["AFIB", "(AF", "AFIB", "ATRIAL FIB", "ATRIAL_FIB"]):
                        af_afl_markers.append((ann_samples[idx], 'AF', 'aux_note', aux_val))
                    
                    # 检测AFL
                    if any(keyword in n_clean for keyword in ["AFL", "FLUTTER", "ATRIAL FLUTTER", "ATRIAL_FLUTTER"]):
                        af_afl_markers.append((ann_samples[idx], 'AFL', 'aux_note', aux_val))
    
    # 2. 检查symbol（MIT-BIH可能使用symbol='A'表示房性，但需要谨慎处理）
    # 注意：symbol='A'通常表示房性早搏，不是AF，但某些情况下可能表示AF段
    # 这里我们只提取aux_note中的标记，因为symbol中的'A'通常是单个心跳标注
    
    if len(af_afl_markers) == 0:
        return segments
    
    # 对标记点按样本位置排序
    af_afl_markers.sort(key=lambda x: x[0])
    
    # 计算每个标记点的持续时间（到下一个标记点或记录结束）
    # MIT-BIH是点标注：从标记点i到标记点i+1之间是同一种状态
    total_samples_orig = int(T * orig_fs / fs)  # 原始采样率下的总长度
    
    intervals = []  # [(start_sample_orig, end_sample_orig, label)]
    
    for marker_idx, (start_sample_orig, label, source, note) in enumerate(af_afl_markers):
        # 找到下一个标记点（或记录结束）
        if marker_idx + 1 < len(af_afl_markers):
            end_sample_orig = af_afl_markers[marker_idx + 1][0]
        else:
            end_sample_orig = total_samples_orig
        
        # 计算持续时间
        duration_orig = end_sample_orig - start_sample_orig
        if duration_orig < min_duration_samples:
            continue  # 过滤太短的标记
        
        intervals.append((start_sample_orig, end_sample_orig, label))
    
    # 转换到目标采样率并提取片段
    for start_sample_orig, end_sample_orig, label in intervals:
        # 转换到目标采样率
        start_sample = int(start_sample_orig * fs / orig_fs)
        end_sample = int(end_sample_orig * fs / orig_fs)
        end_sample = min(end_sample, T)  # 确保不超过ECG长度
        
        if end_sample <= start_sample:
            continue
        
        # 提取该区间内的所有10秒窗口
        current_start = start_sample
        
        while current_start + window_samples <= end_sample:
            segment_end = current_start + window_samples
            
            # 提取片段
            segment = ecg[current_start:segment_end, :].copy()
            
            # 确保片段形状正确
            if segment.shape[0] == window_samples:
                segments.append((segment, label, current_start, segment_end))
            
            current_start += step_samples
    
    return segments


def process_mitbih_record(
    record_path: str,
    record_name: str,
    output_dir: str,
    window_sec: float = 10.0,
    overlap_sec: float = 0.0,
    target_fs: float = 250.0,
) -> List[Dict]:
    """
    处理单条MIT-BIH记录
    
    Returns:
        List of segment info dicts
    """
    try:
        # 读取记录（MIT-BIH有2导联：MLII和V1）
        record = wfdb.rdrecord(record_path, channels=[0, 1])
        ecg = record.p_signal.astype(np.float32)  # (T, 2)
        orig_fs = record.fs
        
        # 读取标注
        try:
            ann = wfdb.rdann(record_path, 'atr')
        except:
            return []
        
        # 预处理ECG
        ecg_processed, fs = preprocess_ecg(ecg, orig_fs, target_fs)
        
        # 提取AF/AFL片段
        aux_note = list(ann.aux_note) if hasattr(ann, 'aux_note') and ann.aux_note is not None else []
        ann_symbols = list(ann.symbol) if hasattr(ann, 'symbol') and ann.symbol is not None else []
        segments = extract_af_afl_segments(
            ecg_processed,
            fs,
            ann.sample,
            aux_note,
            ann_symbols,
            orig_fs,  # 传入原始采样率
            window_sec,
            overlap_sec,
            min_duration_sec=10.0,  # 最小持续时间10秒（至少能提取一个10秒窗口）
        )
        
        # 保存片段
        segment_infos = []
        for seg_idx, (segment, label, start_idx, end_idx) in enumerate(segments):
            # 保存为NPZ文件
            filename = f"{record_name}_seg{seg_idx:04d}.npz"
            filepath = os.path.join(output_dir, filename)
            
            np.savez_compressed(
                filepath,
                ecg=segment,
                fs=fs,
                label=label,
                record_name=record_name,
                segment_idx=seg_idx,
                start_idx=start_idx,
                end_idx=end_idx,
            )
            
            # 记录信息（使用相对于项目根目录的路径）
            # 计算相对于output_dir的路径，然后转换为相对于项目根目录的路径
            rel_path = os.path.relpath(filepath, start=os.getcwd()).replace("\\", "/")
            segment_infos.append({
                'path': rel_path,
                'label_raw': label,
                'record_name': record_name,
                'segment_idx': seg_idx,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'dataset': 'MIT-BIH',
            })
        
        return segment_infos
        
    except Exception as e:
        print(f"  处理记录 {record_name} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    parser = argparse.ArgumentParser(
        description="从 MIT-BIH Arrhythmia Database 提取 AF 和 AFL 片段"
    )
    parser.add_argument(
        "--db_dir",
        type=str,
        required=True,
        help="MIT-BIH Arrhythmia 数据库目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/holter/mit-bih",
        help="输出目录",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="mitbih_af_afl_segments.csv",
        help="输出CSV文件名",
    )
    parser.add_argument(
        "--window_sec",
        type=float,
        default=10.0,
        help="窗口长度（秒）",
    )
    parser.add_argument(
        "--target_fs",
        type=float,
        default=250.0,
        help="目标采样率（Hz）",
    )
    parser.add_argument(
        "--overlap_sec",
        type=float,
        default=0.0,
        help="重叠长度（秒）",
    )
    parser.add_argument(
        "--records",
        type=str,
        nargs="+",
        default=None,
        help="指定要处理的记录（默认：所有记录）",
    )
    
    args = parser.parse_args()
    
    if not WFDB_AVAILABLE:
        print("Error: wfdb is required. Install with: pip install wfdb")
        sys.exit(1)
    
    db_dir = Path(args.db_dir)
    if not db_dir.exists():
        raise FileNotFoundError(f"数据库目录不存在: {db_dir}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有记录
    hea_files = sorted(db_dir.glob("*.hea"))
    if not hea_files:
        print(f"在目录 {db_dir} 下没有找到 .hea 文件")
        return
    
    # 如果指定了记录，只处理这些记录
    if args.records:
        hea_files = [h for h in hea_files if h.stem in args.records]
    
    print("=" * 80)
    print("MIT-BIH Arrhythmia Database AF/AFL 片段提取")
    print("=" * 80)
    print(f"数据库目录: {db_dir}")
    print(f"输出目录: {output_dir}")
    print(f"检测到 {len(hea_files)} 个记录")
    print(f"窗口长度: {args.window_sec} 秒")
    print(f"目标采样率: {args.target_fs} Hz")
    print("=" * 80)
    
    all_segments = []
    
    for hea in hea_files:
        record_name = hea.stem
        record_path = str(hea.with_suffix(""))
        
        print(f"\n处理记录: {record_name}")
        segment_infos = process_mitbih_record(
            record_path,
            record_name,
            str(output_dir),
            args.window_sec,
            args.overlap_sec,
            args.target_fs,
        )
        
        if len(segment_infos) > 0:
            print(f"  提取了 {len(segment_infos)} 个片段")
            all_segments.extend(segment_infos)
        else:
            print(f"  未找到AF/AFL片段")
    
    # 保存CSV
    if len(all_segments) > 0:
        df = pd.DataFrame(all_segments)
        # 如果output_csv是相对路径，创建目录；如果是绝对路径，直接使用
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        print("\n" + "=" * 80)
        print("处理完成！")
        print("=" * 80)
        print(f"总共提取了 {len(all_segments)} 个片段")
        print(f"标签分布:")
        print(df['label_raw'].value_counts())
        print(f"\nCSV文件已保存到: {csv_path}")
        print("=" * 80)
    else:
        print("\n未提取到任何AF/AFL片段")


if __name__ == "__main__":
    main()

