# process_ltafdb_segments.py
"""
处理 LTAFDB 数据集，拆分为 10 秒片段

LTAFDB (Long-Term Atrial Fibrillation Database):
- 动态监测数据，24-25小时记录
- 采样率 128Hz，2导联
- 需要重采样到 250Hz，然后切分为 10 秒窗口
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import Counter
import numpy as np
import pandas as pd
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# 尝试导入wfdb（MIT-BIH数据库读取）
try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    print("Warning: wfdb not available. Install with: pip install wfdb")


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


def segment_ecg(ecg: np.ndarray, fs: float, window_sec: float = 10.0, overlap_sec: float = 0.0) -> List[Tuple[np.ndarray, int, int]]:
    """
    将ECG信号切分为固定长度的窗口
    
    Args:
        ecg: ECG信号，形状 (T,) 或 (T, C)
        fs: 采样率
        window_sec: 窗口长度（秒）
        overlap_sec: 重叠长度（秒）
    
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
    
    segments = []
    start = 0
    
    while start + window_samples <= T:
        end = start + window_samples
        segment = ecg[start:end, :].copy()
        
        if is_1d:
            segment = segment[:, 0]
        
        segments.append((segment, start, end))
        start += step_samples
    
    # 处理最后一个片段（如果足够长，至少5秒）
    min_window_samples = int(5.0 * fs)
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


def load_ltafdb_record(record_path: str, record_name: str) -> Tuple[np.ndarray, float, Dict]:
    """
    加载LTAFDB记录
    
    Args:
        record_path: 记录文件路径（不含扩展名）
        record_name: 记录名称（如 '00'）
    
    Returns:
        (ecg信号, 采样率, 标注信息)
    """
    if not WFDB_AVAILABLE:
        raise ImportError("wfdb is required to load LTAFDB records")
    
    try:
        # 读取记录（LTAFDB通常是2导联）
        record = wfdb.rdrecord(record_path, channels=[0, 1])
    except Exception as e:
        # 某些记录可能只有1个导联，尝试读取所有可用导联
        try:
            record = wfdb.rdrecord(record_path)
            if record.p_signal.shape[1] >= 2:
                record.p_signal = record.p_signal[:, :2]
            elif record.p_signal.shape[1] == 1:
                # 如果只有1个导联，复制一份
                record.p_signal = np.column_stack([record.p_signal[:, 0], record.p_signal[:, 0]])
        except Exception as e2:
            raise ValueError(f"Cannot read record {record_name}: {e2}")
    
    # 读取标注
    try:
        annotation = wfdb.rdann(record_path, 'atr')
    except Exception:
        # 某些记录可能没有标注文件
        annotation = None
    
    ecg = record.p_signal.astype(np.float32)  # (T, 2)
    fs = record.fs
    
    # 解析标注
    annotation_info = {
        'symbol': annotation.symbol if annotation is not None else None,
        'sample': annotation.sample if annotation is not None else None,
        'aux_note': annotation.aux_note if (annotation is not None and hasattr(annotation, 'aux_note')) else None,
    }
    
    print(f"  [LTAFDB] Record {record_name}:")
    print(f"    - ECG length: {ecg.shape[0]} samples ({ecg.shape[0]/fs:.1f} seconds)")
    print(f"    - Sampling rate: {fs} Hz")
    if annotation is not None:
        unique_symbols = Counter(annotation.symbol)
        print(f"    - Annotation count: {len(annotation.symbol)}")
        print(f"    - Unique annotation symbols: {dict(unique_symbols)}")
        
        # 检查 aux_note 字段（LTAFDB 可能在 aux_note 中包含 AF/AFL 信息）
        if hasattr(annotation, 'aux_note') and annotation.aux_note is not None:
            aux_note_list = annotation.aux_note
            if isinstance(aux_note_list, np.ndarray):
                aux_note_list = aux_note_list.tolist()
            if isinstance(aux_note_list, list):
                # 过滤并统计 aux_note
                aux_notes = [str(n).strip() for n in aux_note_list if n and str(n).strip()]
                if aux_notes:
                    unique_aux = Counter([n.upper() for n in aux_notes if n])
                    print(f"    - Unique aux_note values: {dict(list(unique_aux.items())[:10])}")  # 只显示前10个
                    
                    # 统计 AF/AFL 相关的 aux_note
                    af_aux_count = sum(1 for n in aux_notes if 'AFIB' in n.upper() or ('AF' in n.upper() and 'AFL' not in n.upper() and 'FLUTTER' not in n.upper()))
                    afl_aux_count = sum(1 for n in aux_notes if 'AFL' in n.upper() or 'FLUTTER' in n.upper())
                    if af_aux_count > 0 or afl_aux_count > 0:
                        print(f"    - AF-related aux_note: {af_aux_count}, AFL-related aux_note: {afl_aux_count}")
    
    return ecg, fs, annotation_info


def infer_rhythm_from_rr_intervals(
    ecg_segment: np.ndarray,
    fs: float,
    ann_samples: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> Optional[str]:
    """
    根据RR间期特征推断心律类型（AF或AFL）
    
    AF特征：
    - 高度不规则（RMSSD高，CV高）
    - 无P波
    - RR间期变异大
    
    AFL特征：
    - 相对规律（RMSSD低，CV低）
    - 可能有flutter波
    - 心房率250-350 bpm
    """
    # 找到窗口内的R峰
    window_r_peaks = ann_samples[(ann_samples >= start_idx) & (ann_samples < end_idx)]
    
    if len(window_r_peaks) < 5:  # 至少需要5个R峰
        return None
    
    # 计算RR间期（秒）
    rr_intervals = np.diff(window_r_peaks) / fs
    
    if len(rr_intervals) < 4:
        return None
    
    # 计算统计特征
    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)
    cv = std_rr / (mean_rr + 1e-8)  # 变异系数
    
    # RMSSD (Root Mean Square of Successive Differences)
    diff_rr = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    
    # pNN50: 相邻RR间期差异>50ms的比例
    pnn50 = np.sum(np.abs(diff_rr) > 0.05) / len(diff_rr) * 100 if len(diff_rr) > 0 else 0
    
    # 心率（bpm）
    hr = 60.0 / mean_rr if mean_rr > 0 else 0
    
    # 判断规则
    # AF: 高度不规则
    is_af = (cv > 0.12) and (rmssd > 0.08) and (pnn50 > 10)
    
    # AFL: 相对规律，但可能有flutter
    # 如果规则性较高，但心率较快，可能是AFL
    is_afl = (cv < 0.15) and (rmssd < 0.12) and (hr > 100) and (hr < 200)
    
    if is_af:
        return 'AF'
    elif is_afl:
        return 'AFL'
    
    return None


def get_window_label_from_annotations(
    start_idx: int,
    end_idx: int,
    ann_samples: np.ndarray,
    ann_symbols: List[str],
    label_map: Dict[str, str],
    min_coverage: float = 0.5,
    aux_note: Optional[List] = None,
    ecg_segment: Optional[np.ndarray] = None,
    fs: Optional[float] = None,
) -> Optional[str]:
    """
    根据窗口内的标注确定标签（AF或AFL）
    
    规则：
    1. 优先检查 aux_note（如果可用），可能包含 'AFIB' 或 'AFL' 信息
    2. 如果窗口内同时存在AF和AFL两类标签，返回None（丢弃该片段）
    3. 如果窗口内只有一类标签（AF或AFL），且覆盖率>=min_coverage，返回该类标签
    4. 如果覆盖率<min_coverage或没有相关标注，返回None
    
    Args:
        start_idx: 窗口起始索引
        end_idx: 窗口结束索引
        ann_samples: 标注样本点位置
        ann_symbols: 标注符号列表
        label_map: 标签映射字典（只包含AF和AFL）
        min_coverage: 窗口内目标标注的最小覆盖率（0-1）
        aux_note: aux_note列表（可能包含心律类型信息）
    
    Returns:
        标签字符串（'AF'或'AFL'），如果无法确定或同时存在两类则返回None
    """
    if ann_samples is None or len(ann_samples) == 0:
        return None
    
    # 找到窗口内的所有标注
    window_mask = (ann_samples >= start_idx) & (ann_samples < end_idx)
    window_symbols = [ann_symbols[i] for i in np.where(window_mask)[0]]
    
    if len(window_symbols) == 0:
        return None
    
    # 方法1：优先检查 aux_note（LTAFDB 可能在 aux_note 中包含 AF/AFL 信息）
    if aux_note is not None and len(aux_note) > 0:
        window_ann_indices = np.where((ann_samples >= start_idx) & (ann_samples < end_idx))[0]
        if len(window_ann_indices) > 0:
            # 检查窗口内的 aux_note
            window_aux_notes = []
            for idx in window_ann_indices:
                if idx < len(aux_note) and aux_note[idx]:
                    aux_val = str(aux_note[idx]).strip().upper()
                    if aux_val:
                        window_aux_notes.append(aux_val)
            
            if len(window_aux_notes) > 0:
                # 统计 aux_note 中的 AF/AFL 信息
                # LTAFDB 的 aux_note 格式可能是 '(AFIB' 或 '(AFIB)' 等，需要去掉括号和空格
                af_in_aux = 0
                afl_in_aux = 0
                
                for n in window_aux_notes:
                    # 清理字符串：去掉括号、空格，转大写
                    n_clean = n.replace('(', '').replace(')', '').replace('[', '').replace(']', '').strip().upper()
                    
                    # 检查是否包含 AFIB 或 AF（但不包含 AFL 或 FLUTTER）
                    if 'AFIB' in n_clean or ('AF' in n_clean and 'AFL' not in n_clean and 'FLUTTER' not in n_clean):
                        af_in_aux += 1
                    # 检查是否包含 AFL 或 FLUTTER
                    elif 'AFL' in n_clean or 'FLUTTER' in n_clean:
                        afl_in_aux += 1
                
                # 如果同时存在，返回 None
                if af_in_aux > 0 and afl_in_aux > 0:
                    return None
                
                # 如果只有一类，返回该类
                if af_in_aux > 0:
                    return 'AF'
                elif afl_in_aux > 0:
                    return 'AFL'
    
    # 方法2：检查 symbol 标注
    # 统计窗口内的标注
    symbol_counts = Counter(window_symbols)
    
    # 只考虑AF和AFL相关的标注
    af_count = 0
    afl_count = 0
    
    for symbol, count in symbol_counts.items():
        mapped_label = label_map.get(symbol, None)
        if mapped_label == 'AF':
            af_count += count
        elif mapped_label == 'AFL':
            afl_count += count
    
    total_relevant = af_count + afl_count
    if total_relevant == 0:
        return None
    
    # 如果同时存在AF和AFL两类标签，丢弃该片段
    if af_count > 0 and afl_count > 0:
        return None
    
    # 计算覆盖率
    coverage = total_relevant / len(window_symbols)
    if coverage < min_coverage:
        return None
    
    # 返回标签（此时只有一类）
    if af_count > 0:
        return 'AF'
    elif afl_count > 0:
        return 'AFL'
    
    # 方法3：如果 symbol 中没有明确的 AF/AFL，但窗口内有 'A'（房性心律失常），
    # 尝试使用 RR 间期特征推断
    if ecg_segment is not None and fs is not None:
        # 检查是否有 'A' 符号（LTAFDB 使用 'A' 表示房性心律失常）
        has_atrial = any(sym == 'A' for sym in window_symbols)
        if has_atrial:
            inferred_label = infer_rhythm_from_rr_intervals(
                ecg_segment, fs, ann_samples, start_idx, end_idx
            )
            if inferred_label is not None:
                return inferred_label
    
    return None


def process_ltafdb_record(
    record_path: str,
    record_name: str,
    output_dir: str,
    window_sec: float = 10.0,
    target_fs: float = 250.0,
    overlap_sec: float = 0.0,
) -> List[Dict]:
    """
    处理单个LTAFDB记录，切分为窗口并保存
    
    Returns:
        List of dict with segment info for CSV
    """
    print(f"  Processing LTAFDB record: {record_name}")
    
    # 加载记录
    ecg, fs, ann_info = load_ltafdb_record(record_path, record_name)
    
    # 预处理
    ecg, fs = preprocess_ecg(ecg, fs, target_fs=target_fs)
    
    # 切分窗口
    segments = segment_ecg(ecg, fs, window_sec=window_sec, overlap_sec=overlap_sec)
    
    # 标签映射（只保留AF和AFL）
    # LTAFDB 使用 'A' 表示房性心律失常，需要从 aux_note 区分 AF 和 AFL
    label_map = {
        'AFIB': 'AF',
        'AF': 'AF',
        'AFL': 'AFL',
        'FLUTTER': 'AFL',
        # LTAFDB 可能使用 'A' 表示房性心律失常，但需要从 aux_note 区分
        # 这里先不映射 'A'，优先使用 aux_note
    }
    
    # 为每个窗口分配标签
    segment_info = []
    ann_samples = ann_info['sample']
    ann_symbols = ann_info['symbol']
    aux_note = ann_info.get('aux_note', None)
    
    # 处理 aux_note（转换为列表格式）
    if aux_note is not None:
        if isinstance(aux_note, np.ndarray):
            aux_note = aux_note.tolist()
        if isinstance(aux_note, list):
            # 过滤空值
            aux_note = [str(n).strip() if n else '' for n in aux_note]
        else:
            aux_note = None
    
    # 检查标注是否存在
    if ann_samples is None or len(ann_samples) == 0:
        print(f"    Warning: Record {record_name} has no annotations, skipping all segments")
        return segment_info
    
    # 统计信息
    total_segments = len(segments)
    skipped_no_label = 0
    skipped_mixed = 0
    skipped_low_coverage = 0
    kept_segments = 0
    
    for seg_idx, (segment, start_idx, end_idx) in enumerate(segments):
        # 确定窗口的标签
        label_raw = get_window_label_from_annotations(
            start_idx, end_idx,
            ann_samples, ann_symbols,
            label_map, min_coverage=0.3,
            aux_note=aux_note,
            ecg_segment=segment,
            fs=fs
        )
        
        # 如果无法确定标签，跳过该片段
        if label_raw is None:
            # 统计跳过原因（用于调试）
            if ann_samples is not None and len(ann_samples) > 0:
                window_mask = (ann_samples >= start_idx) & (ann_samples < end_idx)
                window_symbols = [ann_symbols[i] for i in np.where(window_mask)[0]]
                if len(window_symbols) > 0:
                    symbol_counts = Counter(window_symbols)
                    af_count = sum(count for sym, count in symbol_counts.items() if label_map.get(sym) == 'AF')
                    afl_count = sum(count for sym, count in symbol_counts.items() if label_map.get(sym) == 'AFL')
                    if af_count > 0 and afl_count > 0:
                        skipped_mixed += 1
                    elif af_count == 0 and afl_count == 0:
                        skipped_no_label += 1
                    else:
                        skipped_low_coverage += 1
                else:
                    skipped_no_label += 1
            else:
                skipped_no_label += 1
            continue
        
        kept_segments += 1
        
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
            'dataset': 'LTAFDB',
        })
    
    # 打印统计信息
    print(f"    Segments: total={total_segments}, kept={kept_segments}, skipped={total_segments - kept_segments}")
    if total_segments > 0:
        print(f"      - Skipped (no AF/AFL label): {skipped_no_label}")
        print(f"      - Skipped (mixed AF+AFL): {skipped_mixed}")
        print(f"      - Skipped (low coverage): {skipped_low_coverage}")
    
    return segment_info


def process_ltafdb_dataset(
    ltafdb_dir: str,
    output_dir: str,
    window_sec: float = 10.0,
    target_fs: float = 250.0,
    overlap_sec: float = 0.0,
    record_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    处理整个LTAFDB数据集
    
    Args:
        ltafdb_dir: LTAFDB数据目录
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
        record_files = list(Path(ltafdb_dir).glob("*.hea"))
        record_list = [f.stem for f in record_files]
        record_list = sorted(record_list)
    
    print(f"Found {len(record_list)} LTAFDB records")
    
    all_segments = []
    
    for record_name in record_list:
        record_path = os.path.join(ltafdb_dir, record_name)
        
        if not os.path.exists(f"{record_path}.hea"):
            print(f"  Warning: Record {record_name} not found, skipping")
            continue
        
        try:
            segments = process_ltafdb_record(
                record_path, record_name, output_dir,
                window_sec=window_sec,
                target_fs=target_fs,
                overlap_sec=overlap_sec,
            )
            all_segments.extend(segments)
            print(f"    Generated {len(segments)} segments from {record_name}")
        except Exception as e:
            print(f"    Error processing {record_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 创建DataFrame
    if len(all_segments) == 0:
        print("Warning: No segments generated!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_segments)
    
    # 统计标签分布
    print("\nLabel distribution:")
    print(df['label_raw'].value_counts())
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Process LTAFDB dataset into 10-second segments")
    parser.add_argument("--ltafdb_dir", type=str, required=True, help="LTAFDB data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for segments")
    parser.add_argument("--output_csv", type=str, default="ltafdb_segments.csv", help="Output CSV file")
    parser.add_argument("--window_sec", type=float, default=10.0, help="Window length in seconds")
    parser.add_argument("--target_fs", type=float, default=250.0, help="Target sampling rate (Hz)")
    parser.add_argument("--overlap_sec", type=float, default=0.0, help="Overlap length in seconds")
    parser.add_argument("--records", type=str, nargs="+", default=None, help="Specific records to process (default: all)")
    
    args = parser.parse_args()
    
    if not WFDB_AVAILABLE:
        print("Error: wfdb is required. Install with: pip install wfdb")
        sys.exit(1)
    
    print("=" * 80)
    print("Processing LTAFDB Dataset")
    print("=" * 80)
    print(f"LTAFDB directory: {args.ltafdb_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Window length: {args.window_sec} seconds")
    print(f"Target sampling rate: {args.target_fs} Hz")
    print(f"Overlap: {args.overlap_sec} seconds")
    print()
    
    # 处理数据集
    df = process_ltafdb_dataset(
        args.ltafdb_dir,
        args.output_dir,
        window_sec=args.window_sec,
        target_fs=args.target_fs,
        overlap_sec=args.overlap_sec,
        record_list=args.records,
    )
    
    if len(df) > 0:
        # 保存CSV
        if os.path.isabs(args.output_csv):
            output_csv_path = args.output_csv
        else:
            output_csv_path = os.path.join(args.output_dir, args.output_csv)
        
        # 确保输出目录存在
        output_csv_dir = os.path.dirname(output_csv_path)
        if output_csv_dir and not os.path.exists(output_csv_dir):
            os.makedirs(output_csv_dir, exist_ok=True)
        
        df.to_csv(output_csv_path, index=False)
        print(f"\nSaved {len(df)} segments to {output_csv_path}")
    else:
        print("\nNo segments generated!")


if __name__ == "__main__":
    main()

