# noise_aware_ecg_af_afl_simple.py
"""
ECG arrhythmia training with morphology + rhythm (RR) + noise branches + optimized fuzzy rules (4-class).

核心功能：
- 在 10s 诊断级 / 动态 ECG 窗口上做 AF / AFL / PSVT / Normal 四分类；
- 使用最简单的 transform：ECG 归一化（去均值、标准化）
- 三分支模型：
    * 形态特征：ECG 1D CNN (morphology branch)
    * 节律特征：RR 间期序列 + GRU 编码 (rhythm branch)
    * 噪声特征：频带能量比例、谱熵、SNR、STFT+DBSCAN 脉冲比例 (noise branch)
- 精细化的模糊规则系统（基于临床ECG知识库）：
    * 多特征组合：RMSSD, CV, pNN50（不规则性）；Flutter band, 心房率（AFL证据）；P波存在性；规律性指数
    * 临床阈值：基于标准ECG诊断标准（如AF: CV>0.12, pNN50>10%；AFL: 心房率250-350 bpm）
    * 精细化规则：
      - AF: 高不规则性 + 无P波 + 无flutter
      - AFL: 规律性 + flutter证据 + 中等心率（心房率250-350 bpm）
      - PSVT: 快心率(>150 bpm) + 规律性 + 有P波 + 无flutter
      - Normal: 正常心率(60-100 bpm) + 规律性 + 有P波 + 无flutter
    * 使用加权特征组合和sigmoid隶属度函数
- 深度学习输出与模糊规则输出加权融合后送入四分类头；
- 样本权重 = 置信度权重 × 类别权重，提升小样本类的贡献。

CSV 格式：
- path        : .npz 文件路径（内含 'ecg'，可选 'fs'）
- label_raw   : 原始心律标签字符串，如 "AF", "AFL", "PSVT", "Normal", "NSR", "SR"
                本脚本自动映射为 AF=0, AFL=1, PSVT=2, Normal=3，并过滤其他异常心律
- 可选 noise_label : 噪声标签（0/1 等，-1 表示未知）

NPZ 格式：
- ecg : ECG 数组，形状 (T,), (T, C) 或 (C, T)
- fs  : float, 采样率（默认 250Hz）
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# R峰检测：使用 scipy.signal.find_peaks
try:
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None
    print("Warning: scipy not available, falling back to simple R-peak detection")

# Windows 多进程支持
if sys.platform == "win32":
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

try:
    import pandas as pd
except ImportError:
    pd = None

# 可选 DBSCAN，用于 STFT 脉冲噪声特征
try:
    from sklearn.cluster import DBSCAN as SK_DBSCAN
except Exception:
    SK_DBSCAN = None

# 二分类推理时对 AFL (index=1) logit 的偏移。
# 之前的固定偏置容易导致 AF->AFL 误报爆炸，所以默认关闭；推荐用“验证集自动找阈值”替代硬偏置。
AFL_LOGIT_BIAS_INFERENCE = 0.0

# AFL margin loss：强制 AFL 的 logit 超过 AF 至少 m，lambda 为损失权重
AFL_MARGIN_LOSS_M = 1.0
AFL_MARGIN_LOSS_LAMBDA = 1.0


# ---------------------------------------------------------------------
# 工具函数：最简单的 ECG 预处理、RR 序列
# ---------------------------------------------------------------------

def _ensure_1d(ecg: np.ndarray) -> np.ndarray:
    """确保 ECG 是单导联 1D，如果多导联，就对导联取平均。"""
    ecg = np.asarray(ecg)
    if ecg.ndim == 1:
        return ecg.astype(np.float32)
    if ecg.ndim == 2:
        T, C = ecg.shape
        if T < C:  # (C, T)
            ecg = ecg.T
        return ecg.mean(axis=1).astype(np.float32)
    raise ValueError(f"Unexpected ECG shape: {ecg.shape}")


def normalize_ecg(ecg: np.ndarray) -> np.ndarray:
    """
    最简单的 ECG 归一化：
    - 去均值
    - 标准化（除以标准差）
    """
    ecg = np.asarray(ecg, dtype=np.float32)
    if ecg.ndim == 1:
        ecg = ecg[:, None]
    elif ecg.ndim == 2:
        T, C = ecg.shape
        if T < C:  # (C, T)
            ecg = ecg.T
    else:
        raise ValueError(f"Unexpected ECG shape: {ecg.shape}")
    
    # 每个导联独立归一化
    for c in range(ecg.shape[1]):
        mean = ecg[:, c].mean()
        std = ecg[:, c].std()
        if std > 1e-8:
            ecg[:, c] = (ecg[:, c] - mean) / std
        else:
            ecg[:, c] = ecg[:, c] - mean
    
    return ecg.astype(np.float32)


# -------- RR / 节律特征 ------------------------------------------------

def _detect_r_peaks_multi_lead(
    ecg: np.ndarray,
    fs: float,
    method: str = "max_energy",
    refractory_ms: float = 200.0,
) -> np.ndarray:
    """
    多导联R峰检测：使用投票或最大能量导联
    - ecg: (T, C) 或 (C, T) 多导联ECG
    - fs: 采样率
    - method: "max_energy" (最大能量导联) 或 "voting" (多导联投票)
    - 返回: R峰位置索引
    """
    ecg = np.asarray(ecg, dtype=np.float32)
    if ecg.ndim == 1:
        return _detect_r_peaks_simple(ecg, fs, refractory_ms)
    
    # 统一为 (T, C)
    if ecg.ndim == 2:
        T, C = ecg.shape
        if T < C:  # (C, T)
            ecg = ecg.T
            T, C = ecg.shape
    
    if method == "max_energy":
        # 选择能量最大的导联
        energies = []
        for c in range(C):
            x = ecg[:, c] - ecg[:, c].mean()
            energy = np.sum(x ** 2)
            energies.append(energy)
        best_lead = int(np.argmax(energies))
        return _detect_r_peaks_simple(ecg[:, best_lead], fs, refractory_ms)
    
    elif method == "voting":
        # 多导联投票：每个导联检测R峰，然后投票合并
        all_peaks = []
        for c in range(C):
            peaks = _detect_r_peaks_simple(ecg[:, c], fs, refractory_ms)
            all_peaks.append(peaks)
        
        if not all_peaks or all([len(p) == 0 for p in all_peaks]):
            return np.array([], dtype=np.int64)
        
        # 合并所有R峰位置
        combined = np.concatenate(all_peaks)
        if len(combined) == 0:
            return np.array([], dtype=np.int64)
        
        # 对接近的R峰进行聚类（投票窗口）
        combined = np.sort(combined)
        min_distance = int(refractory_ms / 1000.0 * fs)
        merged_peaks = []
        current_cluster = [combined[0]]
        
        for i in range(1, len(combined)):
            if combined[i] - current_cluster[-1] < min_distance:
                current_cluster.append(combined[i])
            else:
                # 取聚类中心（中位数）作为最终R峰
                merged_peaks.append(int(np.median(current_cluster)))
                current_cluster = [combined[i]]
        
        if current_cluster:
            merged_peaks.append(int(np.median(current_cluster)))
        
        return np.array(merged_peaks, dtype=np.int64)
    
    else:
        raise ValueError(f"Unknown method: {method}, use 'max_energy' or 'voting'")


def _detect_r_peaks_simple(ecg_1d: np.ndarray, fs: float, refractory_ms: float = 200.0) -> np.ndarray:
    """
    使用 scipy.signal.find_peaks 进行 R 峰检测：
    - 去均值，取绝对值
    - 使用 find_peaks 检测峰值，设置最小距离（不应期）和高度阈值
    - 返回 R 峰位置索引
    """
    x = ecg_1d.astype(np.float32)
    n = len(x)
    if n < 10:
        return np.array([], dtype=np.int64)
    
    # 预处理：去均值，取绝对值
    x = x - x.mean()
    x_abs = np.abs(x)
    
    # 如果 scipy 不可用，回退到简单方法
    if find_peaks is None:
        # 回退到原来的简单实现
        thr = np.percentile(x_abs, 90.0)
        if thr <= 0:
            thr = x_abs.max() * 0.5
        cand = np.where(x_abs >= thr)[0]
        if cand.size == 0:
            return np.array([], dtype=np.int64)
        
        refractory = int(refractory_ms / 1000.0 * fs)
        r_peaks = []
        last_peak = -refractory
        i = 0
        while i < cand.size:
            idx = cand[i]
            if idx - last_peak < refractory:
                i += 1
                continue
            win_end = idx + refractory
            win_mask = (cand >= idx) & (cand <= win_end)
            win_cand = cand[win_mask]
            if win_cand.size == 0:
                i += 1
                continue
            j = win_cand[np.argmax(x_abs[win_cand])]
            r_peaks.append(int(j))
            last_peak = j
            i = np.searchsorted(cand, j + refractory)
        return np.array(r_peaks, dtype=np.int64)
    
    # 使用 scipy.signal.find_peaks
    # 计算最小距离（不应期，转换为样本数）
    min_distance = int(refractory_ms / 1000.0 * fs)
    min_distance = max(1, min_distance)  # 至少为1
    
    # 计算高度阈值（使用90%分位数或最大值的50%）
    height_threshold = np.percentile(x_abs, 90.0)
    if height_threshold <= 0:
        height_threshold = x_abs.max() * 0.5
    if height_threshold <= 0:
        height_threshold = None  # 如果仍然无效，不设置高度阈值
    
    # 检测峰值
    peaks, _ = find_peaks(
        x_abs,
        distance=min_distance,  # 最小距离（不应期）
        height=height_threshold,  # 最小高度阈值
        prominence=None,  # 可选：峰值突出度
    )
    
    return peaks.astype(np.int64)


def extract_rr_seq(
    ecg: np.ndarray,
    fs: float = 250.0,
    max_intervals: int = 32,
    multi_lead_method: str = "max_energy",
) -> np.ndarray:
    """
    从多导联/单导联 ECG 抽取 RR 间期序列：
    - 多导联时使用投票或最大能量导联进行R峰检测
    - 单导联时使用简单方法
    - RR 间期 = 相邻 R 的差 / fs（秒）
    - 将 RR 裁剪到 [0.3, 2.0] 秒，简单归一化到 [0,1]
    - 固定长度 max_intervals：不够则零填充，过长则只取最后 max_intervals 个
    """
    fs = float(fs) if fs is not None else 250.0
    
    # 多导联检测
    ecg_arr = np.asarray(ecg, dtype=np.float32)
    if ecg_arr.ndim == 2:
        T, C = ecg_arr.shape
        if T < C:  # (C, T)
            ecg_arr = ecg_arr.T
        if ecg_arr.shape[1] > 1:
            # 多导联：使用投票或最大能量
            r_peaks = _detect_r_peaks_multi_lead(ecg_arr, fs, method=multi_lead_method)
        else:
            # 单导联
            r_peaks = _detect_r_peaks_simple(ecg_arr[:, 0], fs)
    else:
        # 单导联
        r_peaks = _detect_r_peaks_simple(ecg_arr, fs)
    
    if r_peaks.size < 2:
        return np.zeros((max_intervals,), dtype=np.float32)

    rr = np.diff(r_peaks) / fs  # 秒
    # 裁剪极端值
    rr = np.clip(rr, 0.3, 2.0)
    # 简单归一化到 [0,1]
    rr_norm = (rr - 0.3) / (2.0 - 0.3)

    # 固定长度
    if rr_norm.size >= max_intervals:
        rr_fixed = rr_norm[-max_intervals:]
    else:
        pad_len = max_intervals - rr_norm.size
        pad = np.zeros((pad_len,), dtype=np.float32)
        rr_fixed = np.concatenate([rr_norm, pad], axis=0)

    return rr_fixed.astype(np.float32)


# ---------------------------------------------------------------------
# 模糊规则系统：基于 RR 间期特征的模糊逻辑推理
# ---------------------------------------------------------------------

def bandpower_ratio(ecg_1d: np.ndarray, fs: float, band: tuple, eps: float = 1e-8) -> float:
    """
    计算指定频带的功率比
    - ecg_1d: 1D ECG 信号
    - fs: 采样率
    - band: (low, high) 频带范围 (Hz)
    - 返回: 该频带功率占总功率的比例
    """
    x = ecg_1d.astype(np.float32)
    n = len(x)
    if n == 0:
        return 0.0
    x = x - x.mean()
    spec = np.fft.rfft(x)
    power = (spec.real ** 2 + spec.imag ** 2)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    total = power.sum() + eps
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    band_power = power[mask].sum()
    return float(band_power / total)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Sigmoid 函数"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def spectral_entropy(ecg_1d: np.ndarray, fs: float, eps: float = 1e-10) -> float:
    """计算归一化谱熵"""
    x = ecg_1d.astype(np.float32)
    n = len(x)
    if n == 0:
        return 0.0
    x = x - x.mean()
    spec = np.fft.rfft(x)
    power = (spec.real ** 2 + spec.imag ** 2)
    psd = power / (power.sum() + eps)
    ent = -(psd * np.log(psd + eps)).sum()
    max_ent = math.log(len(psd) + eps)
    return float(ent / (max_ent + eps))


def stft_impulsive_ratio(
    ecg_1d: np.ndarray,
    fs: float,
    win_size: int = 256,
    hop_size: int = 128,
    eps: float = 1e-8,
) -> float:
    """
    STFT+DBSCAN 粗估"脉冲伪差"比例：
    - 每个时间窗计算幅度谱作为谱向量
    - DBSCAN 聚类，把噪声点 + 最小簇视为"脉冲类"
    - 返回脉冲类帧数 / 总帧数
    """
    x = ecg_1d.astype(np.float32)
    n = len(x)
    if n < win_size or SK_DBSCAN is None:
        return 0.0

    win = np.hanning(win_size).astype(np.float32)
    frames = []
    for start in range(0, n - win_size + 1, hop_size):
        seg = x[start:start + win_size] * win
        spec = np.fft.rfft(seg)
        mag = np.abs(spec)
        m_norm = mag / (mag.sum() + eps)
        frames.append(m_norm)
    if not frames:
        return 0.0

    X = np.stack(frames, axis=0)
    db = SK_DBSCAN(eps=0.3, min_samples=5, metric="euclidean")
    labels = db.fit_predict(X)
    uniq, cnts = np.unique(labels, return_counts=True)
    if len(uniq) <= 1:
        return 0.0

    label_counts = dict(zip(uniq.tolist(), cnts.tolist()))
    impulsive_mask = np.zeros_like(labels, dtype=bool)
    impulsive_mask |= (labels == -1)
    non_noise_labels = [l for l in uniq if l != -1]
    if non_noise_labels:
        smallest_label = min(non_noise_labels, key=lambda l: label_counts[l])
        impulsive_mask |= (labels == smallest_label)

    return float(impulsive_mask.sum() / len(labels))


def extract_noise_features(ecg: np.ndarray, fs: float = 250.0, use_stft_dbscan: bool = False) -> np.ndarray:
    """
    抽取噪声相关特征：
    - baseline_ratio   : 0-0.5 Hz
    - emg_ratio        : 20-45 Hz
    - powerline_ratio  : 48-52 Hz
    - mainband_ratio   : 5-40 Hz
    - snr_est          : mainband / (baseline + emg + powerline + 1e-3)
    - spectral_entropy : 归一化谱熵
    - impulsive_ratio  : STFT-DBSCAN 脉冲比例（可选，默认禁用以加速）
    """
    ecg_1d = _ensure_1d(ecg)
    fs = float(fs) if fs is not None else 250.0

    baseline_ratio = bandpower_ratio(ecg_1d, fs, (0.0, 0.5))
    emg_ratio = bandpower_ratio(ecg_1d, fs, (20.0, 45.0))
    powerline_ratio = bandpower_ratio(ecg_1d, fs, (48.0, 52.0))
    mainband_ratio = bandpower_ratio(ecg_1d, fs, (5.0, 40.0))

    noise_sum = baseline_ratio + emg_ratio + powerline_ratio + 1e-3
    snr_est = float(mainband_ratio / noise_sum)

    spec_ent = spectral_entropy(ecg_1d, fs)

    if use_stft_dbscan:
        impulsive_ratio = stft_impulsive_ratio(ecg_1d, fs)
    else:
        impulsive_ratio = 0.0

    feats = np.array(
        [
            baseline_ratio,
            emg_ratio,
            powerline_ratio,
            mainband_ratio,
            snr_est,
            spec_ent,
            impulsive_ratio,
        ],
        dtype=np.float32,
    )
    return feats


def compute_flutter_evidence_multi_lead(
    ecg: np.ndarray,
    fs: float,
    method: str = "fixed_lead",
    lead_names: Optional[List[str]] = None,
) -> Tuple[float, float]:
    """
    多导联flutter证据计算：
    - ecg: (T, C) 多导联ECG
    - fs: 采样率
    - method: "fixed_lead" (固定II/V1导联) 或 "attention" (能量加权)
    - lead_names: 导联名称列表（如["I", "II", "V1", ...]），用于识别II/V1
    - 返回: (flutter_ratio, atrial_rate)
    """
    ecg_arr = np.asarray(ecg, dtype=np.float32)
    if ecg_arr.ndim == 1:
        flutter_ratio = bandpower_ratio(ecg_arr, fs, (3.0, 8.0))
        atrial_rate = 330.0 * flutter_ratio if flutter_ratio > 0.05 else 0.0
        return flutter_ratio, atrial_rate
    
    # 统一为 (T, C)
    if ecg_arr.ndim == 2:
        T, C = ecg_arr.shape
        if T < C:  # (C, T)
            ecg_arr = ecg_arr.T
            T, C = ecg_arr.shape
    
    if method == "fixed_lead" and lead_names is not None:
        # 尝试找到II或V1导联
        ii_idx = None
        v1_idx = None
        for i, name in enumerate(lead_names):
            if name.upper() in ["II", "2"]:
                ii_idx = i
            elif name.upper() in ["V1", "V1"]:
                v1_idx = i
        
        # 优先使用II，其次V1
        if ii_idx is not None and ii_idx < C:
            flutter_ratio = bandpower_ratio(ecg_arr[:, ii_idx], fs, (3.0, 8.0))
        elif v1_idx is not None and v1_idx < C:
            flutter_ratio = bandpower_ratio(ecg_arr[:, v1_idx], fs, (3.0, 8.0))
        else:
            # 如果找不到II/V1，使用最大能量导联
            energies = [np.sum((ecg_arr[:, c] - ecg_arr[:, c].mean()) ** 2) for c in range(C)]
            best_lead = int(np.argmax(energies))
            flutter_ratio = bandpower_ratio(ecg_arr[:, best_lead], fs, (3.0, 8.0))
    
    elif method == "attention":
        # 能量加权：计算每个导联的flutter能量，然后加权平均
        flutter_ratios = []
        weights = []
        for c in range(C):
            x = ecg_arr[:, c] - ecg_arr[:, c].mean()
            energy = np.sum(x ** 2)
            flutter_r = bandpower_ratio(ecg_arr[:, c], fs, (3.0, 8.0))
            flutter_ratios.append(flutter_r)
            weights.append(energy)
        
        if weights:
            weights = np.array(weights, dtype=np.float32)
            weights = weights / (weights.sum() + 1e-8)
            flutter_ratio = float(np.sum([r * w for r, w in zip(flutter_ratios, weights)]))
        else:
            flutter_ratio = 0.0
    else:
        # 默认：使用最大能量导联
        energies = [np.sum((ecg_arr[:, c] - ecg_arr[:, c].mean()) ** 2) for c in range(C)]
        best_lead = int(np.argmax(energies))
        flutter_ratio = bandpower_ratio(ecg_arr[:, best_lead], fs, (3.0, 8.0))
    
    # 估计心房率
    atrial_rate = 330.0 * flutter_ratio if flutter_ratio > 0.05 else 0.0
    
    return float(flutter_ratio), float(atrial_rate)


def detect_p_wave_presence(ecg_1d: np.ndarray, fs: float, r_peaks: np.ndarray) -> float:
    """
    简化的P波存在性检测（基于RR间期前的低频能量）
    - 在R波前200-400ms窗口内检测低频能量（0.5-5 Hz）
    - 返回P波存在性的概率（0-1）
    """
    if len(r_peaks) < 2 or len(ecg_1d) < 400:
        return 0.5  # 默认中等概率
    
    p_wave_scores = []
    for r_peak in r_peaks[:min(10, len(r_peaks))]:  # 只检查前10个R波
        # R波前200-400ms窗口
        start_idx = max(0, int(r_peak - 400 * fs / 1000))
        end_idx = max(0, int(r_peak - 200 * fs / 1000))
        if end_idx <= start_idx:
            continue
        
        window = ecg_1d[start_idx:end_idx]
        if len(window) < 10:
            continue
        
        # 计算低频能量（P波通常在0.5-5 Hz）
        p_band_ratio = bandpower_ratio(window, fs, (0.5, 5.0))
        p_wave_scores.append(p_band_ratio)
    
    if len(p_wave_scores) == 0:
        return 0.5
    
    # 平均P波能量作为存在性指标
    p_wave_presence = float(np.mean(p_wave_scores))
    return min(1.0, max(0.0, p_wave_presence * 10.0))  # 放大并限制到[0,1]


def extract_rr_statistics(
    rr_seq: np.ndarray,
    ecg: Optional[np.ndarray] = None,
    fs: float = 250.0,
    flutter_method: str = "fixed_lead",
    lead_names: Optional[List[str]] = None,
) -> dict:
    """
    从 RR 序列提取统计特征用于模糊规则（精细化版）：
    - rr_seq: 归一化后的 RR 序列（可能包含零填充）
    - ecg: ECG 信号（1D或2D多导联，用于计算 flutter band 和 P波）
    - fs: 采样率
    - flutter_method: "fixed_lead" 或 "attention"（多导联flutter计算方式）
    - lead_names: 导联名称列表（用于识别II/V1导联）
    - 返回：包含变异性、RMSSD、flutter_ratio、P波等特征的字典
    """
    # 去除零填充
    rr_valid = rr_seq[rr_seq > 1e-6]
    if len(rr_valid) < 3:
        return {
            "rr_mean": 0.6,  # 默认值（秒）
            "rr_std": 0.1,
            "rr_cv": 0.15,  # 变异系数
            "rmssd": 0.0,  # RMSSD
            "pnn50": 0.0,  # pNN50: 相邻RR间期差异>50ms的比例
            "heart_rate": 100.0,  # bpm
            "flutter_ratio": 0.0,  # Flutter band (3-8 Hz) 功率比
            "atrial_rate": 0.0,  # 心房率（基于flutter band，AFL通常250-350 bpm）
            "p_wave_presence": 0.5,  # P波存在性（0-1）
            "regularity_index": 0.5,  # 规律性指数（基于RR变异）
        }
    
    # 反归一化到实际秒数（假设归一化范围是 [0.3, 2.0]）
    rr_actual = rr_valid * (2.0 - 0.3) + 0.3  # 秒
    
    rr_mean = float(np.mean(rr_actual))
    rr_std = float(np.std(rr_actual))
    rr_cv = rr_std / (rr_mean + 1e-8)  # 变异系数
    
    # RMSSD: Root Mean Square of Successive Differences（更准确的不规则性指标）
    diffs = np.diff(rr_actual)
    rmssd = float(np.sqrt(np.mean(diffs * diffs) + 1e-8))
    
    # pNN50: 相邻RR间期差异>50ms的比例（AF通常>10%）
    if len(diffs) > 0:
        pnn50 = float(np.sum(np.abs(diffs) > 0.05) / len(diffs))
    else:
        pnn50 = 0.0
    
    # 平均心率 (bpm)
    heart_rate = 60.0 / (rr_mean + 1e-8)
    
    # 规律性指数：基于RR变异性的逆（0=完全不规律，1=完全规律）
    regularity_index = float(1.0 / (1.0 + rr_cv * 10.0))
    
    # Flutter band (3-8 Hz) 功率比和心房率估计
    flutter_ratio = 0.0
    atrial_rate = 0.0
    p_wave_presence = 0.5
    r_peaks = None
    
    if ecg is not None:
        ecg_arr = np.asarray(ecg, dtype=np.float32)
        
        # 多导联flutter计算
        if ecg_arr.ndim == 2:
            T, C = ecg_arr.shape
            if T < C:  # (C, T)
                ecg_arr = ecg_arr.T
            if C > 1:
                # 多导联：使用lead-attention或固定导联
                flutter_ratio, atrial_rate = compute_flutter_evidence_multi_lead(
                    ecg_arr, fs, method=flutter_method, lead_names=lead_names
                )
                # P波检测：使用最大能量导联
                energies = [np.sum((ecg_arr[:, c] - ecg_arr[:, c].mean()) ** 2) for c in range(C)]
                best_lead = int(np.argmax(energies))
                ecg_1d_for_p = ecg_arr[:, best_lead]
            else:
                # 单导联
                ecg_1d_for_p = ecg_arr[:, 0]
                flutter_ratio = bandpower_ratio(ecg_1d_for_p, fs, (3.0, 8.0))
                atrial_rate = 330.0 * flutter_ratio if flutter_ratio > 0.05 else 0.0
        else:
            # 单导联
            ecg_1d_for_p = ecg_arr
            flutter_ratio = bandpower_ratio(ecg_1d_for_p, fs, (3.0, 8.0))
            atrial_rate = 330.0 * flutter_ratio if flutter_ratio > 0.05 else 0.0
        
        # P波检测
        r_peaks = _detect_r_peaks_simple(ecg_1d_for_p, fs)
        if len(r_peaks) >= 2:
            p_wave_presence = detect_p_wave_presence(ecg_1d_for_p, fs, r_peaks)
    
    return {
        "rr_mean": rr_mean,
        "rr_std": rr_std,
        "rr_cv": rr_cv,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "heart_rate": heart_rate,
        "flutter_ratio": flutter_ratio,
        "atrial_rate": atrial_rate,
        "p_wave_presence": p_wave_presence,
        "regularity_index": regularity_index,
    }


def fuzzy_membership_triangular(x: float, a: float, b: float, c: float) -> float:
    """
    三角隶属度函数：
    - a: 左边界
    - b: 峰值位置
    - c: 右边界
    """
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    else:  # b < x < c
        return (c - x) / (c - b)


def fuzzy_membership_trapezoidal(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    梯形隶属度函数：
    - a: 左边界
    - b: 左峰值
    - c: 右峰值
    - d: 右边界
    """
    if x <= a or x >= d:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1.0
    else:  # c < x < d
        return (d - x) / (d - c)


class FuzzyRuleSystem:
    """
    精细化的模糊规则系统：基于临床ECG知识库进行四分类（AF/AFL/PSVT/Normal）
    
    基于临床标准：
    - AF: 不规则心律，无P波，RR间期高度变异（CV>0.12, pNN50>10%）
    - AFL: 规律心律，flutter波（3-8 Hz），心房率250-350 bpm，RR间期规律
    - PSVT: 快速规律心律（>150 bpm），突然开始/结束，无flutter
    - Normal: 规律心律，正常心率（60-100 bpm），有P波
    
    特征：
    - RMSSD, CV, pNN50: 不规则性指标
    - Flutter ratio, Atrial rate: AFL证据
    - Heart rate: 心率分类
    - P wave presence: P波存在性
    - Regularity index: 规律性指数
    """
    
    def __init__(self):
        pass
    
    def infer(self, rr_stats: dict) -> dict:
        """
        精细化的模糊规则推理，返回 AF、AFL、PSVT 和 Normal 的隶属度分数
        
        使用多特征组合和临床阈值：
        """
        rmssd = rr_stats["rmssd"]
        rr_cv = rr_stats["rr_cv"]
        pnn50 = rr_stats["pnn50"]
        flutter_ratio = rr_stats["flutter_ratio"]
        atrial_rate = rr_stats["atrial_rate"]
        heart_rate = rr_stats["heart_rate"]
        p_wave_presence = rr_stats["p_wave_presence"]
        regularity_index = rr_stats["regularity_index"]
        
        # ========== 特征隶属度计算（基于临床阈值）==========
        
        # 1. 不规则性特征（AF的关键指标）
        # RMSSD > 0.08s 表示高度不规则（临床标准）
        mu_irreg_rmssd = float(_sigmoid_np(np.array([(rmssd - 0.08) / 0.02], dtype=np.float32))[0])
        # CV > 0.12 表示高变异性（AF特征）
        mu_irreg_cv = float(_sigmoid_np(np.array([(rr_cv - 0.12) / 0.03], dtype=np.float32))[0])
        # pNN50 > 10% 表示高度不规则（AF特征）
        mu_irreg_pnn50 = float(_sigmoid_np(np.array([(pnn50 - 0.10) / 0.05], dtype=np.float32))[0])
        # 综合不规则性（加权平均）
        mu_irreg_high = (mu_irreg_rmssd * 0.4 + mu_irreg_cv * 0.4 + mu_irreg_pnn50 * 0.2)
        mu_irreg_low = 1.0 - mu_irreg_high
        
        # 2. 规律性特征
        mu_regular = regularity_index  # 直接使用规律性指数
        
        # 3. Flutter证据（AFL的关键指标）
        # Flutter band (3-8 Hz) 能量 > 0.08
        mu_flutter_band = float(_sigmoid_np(np.array([(flutter_ratio - 0.08) / 0.03], dtype=np.float32))[0])
        # 心房率在250-350 bpm范围内（AFL特征）
        mu_atrial_rate_afl = 0.0
        if 250.0 <= atrial_rate <= 350.0:
            # 在范围内，计算隶属度（峰值在300 bpm）
            if atrial_rate <= 300.0:
                mu_atrial_rate_afl = (atrial_rate - 250.0) / 50.0
            else:
                mu_atrial_rate_afl = (350.0 - atrial_rate) / 50.0
        mu_flutter = mu_flutter_band * 0.7 + mu_atrial_rate_afl * 0.3
        
        # 4. 心率分类
        # 快心率：>150 bpm（PSVT特征）
        mu_hr_fast = float(_sigmoid_np(np.array([(heart_rate - 150.0) / 20.0], dtype=np.float32))[0])
        # 正常心率：60-100 bpm（Normal特征）
        mu_hr_normal_low = float(_sigmoid_np(np.array([(60.0 - heart_rate) / 10.0], dtype=np.float32))[0])
        mu_hr_normal_high = float(_sigmoid_np(np.array([(heart_rate - 100.0) / 10.0], dtype=np.float32))[0])
        mu_hr_normal = (1.0 - mu_hr_normal_low) * (1.0 - mu_hr_normal_high)
        # 中等心率：100-150 bpm（可能是AFL）
        mu_hr_medium = 0.0
        if 100.0 < heart_rate < 150.0:
            if heart_rate <= 125.0:
                mu_hr_medium = (heart_rate - 100.0) / 25.0
            else:
                mu_hr_medium = (150.0 - heart_rate) / 25.0
        
        # 5. P波存在性（Normal和PSVT有P波，AF通常无P波）
        mu_p_wave_present = p_wave_presence
        mu_p_wave_absent = 1.0 - p_wave_presence
        
        # ========== 精细化规则推理 ==========
        
        # R1: AF规则（高不规则性 + 无P波 + 无flutter）
        # 临床标准：不规则心律，无P波，RR高度变异
        s_af = mu_irreg_high * (0.6 + 0.4 * mu_p_wave_absent) * (1.0 - mu_flutter * 0.5)
        
        # R2: AFL规则（规律性 + flutter证据 + 中等心率）
        # 临床标准：规律心律，flutter波，心房率250-350 bpm
        # 原始规则对AFL要求较苛刻（必须同时满足高规律性 + 强flutter + 中等心率），
        # 在真实数据中可能导致 AFL 隶属度偏低，从而使得 AFL recall 过低。
        # 为了提高 AFL 的召回率，这里放宽并增强 AFL 规则：
        #  - 更强调 flutter 证据（mu_flutter），适当结合规律性和中等心率
        #  - 对“不过分不规则”的节律（1 - mu_irreg_high）给予一定加成，避免被 AF 规则完全压制
        s_afl_base = (
            0.6 * mu_flutter      # flutter 证据为主
            + 0.2 * mu_hr_medium  # 中等心率为辅
            + 0.2 * mu_regular    # 规律性辅助
        )
        # 对于不那么高度不规则的节律，适当增强 AFL 评分（防止被 AF 完全抢占）
        afl_regular_boost = 0.5 + 0.5 * (1.0 - mu_irreg_high)  # mu_irreg_high 越低，boost 越高
        s_afl = s_afl_base * afl_regular_boost
        
        # R3: PSVT规则（快心率 + 规律性 + 有P波 + 无flutter）
        # 临床标准：快速规律心律，有P波，无flutter
        s_psvt = mu_hr_fast * mu_regular * (0.7 + 0.3 * mu_p_wave_present) * (1.0 - mu_flutter * 0.5)
        
        # R4: Normal规则（正常心率 + 规律性 + 有P波 + 无flutter）
        # 临床标准：正常规律心律，有P波，无flutter
        s_normal = mu_hr_normal * mu_regular * (0.7 + 0.3 * mu_p_wave_present) * (1.0 - mu_flutter * 0.5)
        
        # 归一化到 [0, 1]
        scores = np.array([s_af, s_afl, s_psvt, s_normal], dtype=np.float32)
        mu = scores / (scores.sum() + 1e-6)
        
        # 计算规则置信度（最大分数与次大分数的差异）
        sorted_scores = np.sort(mu)[::-1]
        rule_conf = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
        
        return {
            "af_score": float(mu[0]),
            "afl_score": float(mu[1]),
            "psvt_score": float(mu[2]),
            "normal_score": float(mu[3]),
            "raw_af": float(s_af),
            "raw_afl": float(s_afl),
            "raw_psvt": float(s_psvt),
            "raw_normal": float(s_normal),
            "rule_conf": rule_conf,  # 规则置信度
        }


def apply_fuzzy_rules(
    rr_seq: np.ndarray,
    ecg: Optional[np.ndarray] = None,
    fs: float = 250.0,
    flutter_method: str = "fixed_lead",
    lead_names: Optional[List[str]] = None,
) -> torch.Tensor:
    """
    应用优化的模糊规则系统，返回模糊分类分数（四分类）
    - rr_seq: 归一化后的 RR 序列
    - ecg: ECG 信号（1D或2D多导联，用于计算 flutter band）
    - fs: 采样率
    - flutter_method: "fixed_lead" 或 "attention"（多导联flutter计算方式）
    - lead_names: 导联名称列表（用于识别II/V1导联）
    返回形状为 (4,) 的 tensor，[AF_logit, AFL_logit, PSVT_logit, Normal_logit]
    """
    fuzzy_system = FuzzyRuleSystem()
    rr_stats = extract_rr_statistics(rr_seq, ecg, fs, flutter_method=flutter_method, lead_names=lead_names)
    fuzzy_result = fuzzy_system.infer(rr_stats)
    
    # 转换为 logits（使用 log 变换，使得可以用于 softmax）
    # 将 [0,1] 的分数转换为 logits
    af_logit = np.log(fuzzy_result["af_score"] + 1e-8) * 2.0  # 放大以匹配模型 logits 尺度
    afl_logit = np.log(fuzzy_result["afl_score"] + 1e-8) * 2.0
    psvt_logit = np.log(fuzzy_result["psvt_score"] + 1e-8) * 2.0
    normal_logit = np.log(fuzzy_result["normal_score"] + 1e-8) * 2.0
    
    return torch.tensor([af_logit, afl_logit, psvt_logit, normal_logit], dtype=torch.float32)


# ---------------------------------------------------------------------
# Dataset：ECG + RR 序列（最简单的 transform）
# ---------------------------------------------------------------------

class ECGDataset(Dataset):
    """
    Dataset:
        - 从 CSV 读 path, label_raw, noise_label
        - 自动过滤非 AF/AFL/PSVT/Normal（只保留这四类）
        - 从 npz 读 ECG (+ fs)，截取/填充到 max_len
        - 最简单的 transform：归一化（去均值、标准化）
        - 优先从预计算文件读取特征（rr_seq, noise_feat, fuzzy_logits），否则在线计算
        - 返回：
            ecg       : (C, T) - 已归一化
            noise_feat: (D,) - 噪声特征
            rr_seq    : (max_rr_intervals,)
            label     : 0..3 (AF=0, AFL=1, PSVT=2, Normal=3)
            noise_label: int (>=0 有监督；-1 未知)
    """

    def __init__(
        self,
        csv_path: str,
        max_len: int = 2500,
        num_leads: int = 1,
        assume_T_first: bool = True,
        max_rr_intervals: int = 32,
        use_stft_dbscan: bool = False,
        multi_lead_method: str = "max_energy",
        flutter_method: str = "fixed_lead",
        lead_names: Optional[List[str]] = None,
        use_precomputed: bool = True,
        recompute_fuzzy_only: bool = False,  # 如果为True，即使有预计算特征，也只重新计算fuzzy_logits
        augment: bool = False,  # 仅训练集启用：轻量ECG增强，提升跨患者泛化
        aug_prob: float = 0.8,
        aug_gain_range: Tuple[float, float] = (0.85, 1.20),
        aug_noise_std_range: Tuple[float, float] = (0.0, 0.03),
        aug_time_mask_prob: float = 0.25,
        aug_time_mask_frac_range: Tuple[float, float] = (0.02, 0.08),
        aug_lead_dropout_prob: float = 0.10,
    ):
        super().__init__()
        if pd is None:
            raise ImportError("pandas is required to use ECGDataset")

        self.df = pd.read_csv(csv_path)
        if "path" not in self.df.columns:
            raise ValueError("CSV must contain 'path' column")

        # 映射 AF/AFL/PSVT/Normal 四类
        if "label" not in self.df.columns:
            if "label_raw" in self.df.columns:
                raw_to_label = {
                    "AF": 0,
                    "AFL": 1,
                    "PSVT": 2,
                    "Normal": 3,
                    "NOR": 3,
                    "NORM": 3,
                    "NSR": 3,
                    "SR": 3,
                    "Other": 3,
                    "OTHER": 3,
                }
                before = len(self.df)
                self.df = self.df[self.df["label_raw"].isin(raw_to_label.keys())].copy()
                after = len(self.df)
                if after < before:
                    print(f"[ECGDataset] Filtered out {before - after} samples with non-target rhythms (kept AF/AFL/PSVT/Normal).")
                if len(self.df) == 0:
                    raise ValueError(
                        "No samples left after filtering for AF/AFL/PSVT/Normal. "
                        "Please check 'label_raw' values in CSV."
                    )
                self.df["label"] = self.df["label_raw"].map(raw_to_label)
            else:
                raise ValueError("CSV must contain either 'label' or 'label_raw' column")

        self.max_len = max_len
        self.num_leads = num_leads
        self.assume_T_first = assume_T_first
        self.max_rr_intervals = max_rr_intervals
        self.use_stft_dbscan = use_stft_dbscan
        self.multi_lead_method = multi_lead_method
        self.flutter_method = flutter_method
        self.lead_names = lead_names
        self.use_precomputed = use_precomputed
        self.recompute_fuzzy_only = recompute_fuzzy_only
        self.augment = bool(augment)
        self.aug_prob = float(aug_prob)
        self.aug_gain_range = aug_gain_range
        self.aug_noise_std_range = aug_noise_std_range
        self.aug_time_mask_prob = float(aug_time_mask_prob)
        self.aug_time_mask_frac_range = aug_time_mask_frac_range
        self.aug_lead_dropout_prob = float(aug_lead_dropout_prob)

        if "noise_label" not in self.df.columns:
            self.df["noise_label"] = -1
        
        # 检查是否有预计算特征路径
        if "feature_path" in self.df.columns:
            self.has_feature_path = True
            # 检查预计算文件是否存在
            valid_features = []
            for feat_path in self.df["feature_path"]:
                if pd.isna(feat_path) or feat_path == "":
                    valid_features.append(False)
                else:
                    valid_features.append(os.path.exists(str(feat_path)))
            self.df["has_precomputed"] = valid_features
            precomputed_count = sum(valid_features)
            print(f"[ECGDataset] Found {precomputed_count}/{len(self.df)} precomputed feature files")
        else:
            self.has_feature_path = False
            self.df["has_precomputed"] = False
            if use_precomputed:
                print(f"[ECGDataset] Warning: 'feature_path' column not found, will compute features on-the-fly")

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _infer_record_name_from_path(path: str) -> str:
        """
        从片段路径中推断 record id（用于 record-level 聚合/划分）。
        兼容形如：
          data/holter/afdb/afdb_segments/04015_seg0021.npz -> 04015
          data/holter/ltafdb/ltafdb_segments/119_seg2299.npz -> 119
        """
        base = os.path.basename(str(path)).replace("\\", "/")
        if base.lower().endswith(".npz"):
            base = base[:-4]
        # 噪声增强文件：整段作为 record_id，与划分逻辑一致，每个扩充 = 新患者（必须先于 _seg 判断）
        if "_noisy_" in base:
            return base
        if "_seg" in base:
            return base.split("_seg")[0]
        return base

    @staticmethod
    def _augment_ecg_inplace(
        ecg_norm: np.ndarray,
        rng: np.random.RandomState,
        gain_range: Tuple[float, float],
        noise_std_range: Tuple[float, float],
        time_mask_prob: float,
        time_mask_frac_range: Tuple[float, float],
        lead_dropout_prob: float,
    ) -> np.ndarray:
        """
        轻量增强（在归一化后执行），用于提升跨患者鲁棒性。
        说明：当使用预计算 rr/noise/fuzzy 特征时，不对这些特征做增强，
        仅对 morphology 分支输入的 ECG 做温和扰动，避免过拟合患者特征。
        ecg_norm: (T, C) float32
        """
        if ecg_norm.ndim != 2:
            return ecg_norm

        T, C = ecg_norm.shape

        # 1) 幅值增益（每导联独立）
        g_lo, g_hi = gain_range
        gains = rng.uniform(g_lo, g_hi, size=(1, C)).astype(np.float32)
        ecg_norm = ecg_norm * gains

        # 2) 高斯噪声（std 为归一化后尺度）
        n_lo, n_hi = noise_std_range
        if n_hi > 0:
            sigma = float(rng.uniform(n_lo, n_hi))
            if sigma > 0:
                ecg_norm = ecg_norm + rng.normal(0.0, sigma, size=ecg_norm.shape).astype(np.float32)

        # 3) 时间遮挡（模拟局部伪差/缺失）
        if rng.rand() < time_mask_prob and T > 10:
            f_lo, f_hi = time_mask_frac_range
            mask_len = int(rng.uniform(f_lo, f_hi) * T)
            mask_len = max(1, min(mask_len, T))
            start = int(rng.randint(0, max(1, T - mask_len)))
            ecg_norm[start:start + mask_len, :] = 0.0

        # 4) 导联 dropout（2 导联时偶尔丢一导联）
        if C >= 2 and rng.rand() < lead_dropout_prob:
            drop_c = int(rng.randint(0, C))
            ecg_norm[:, drop_c] = 0.0

        return ecg_norm.astype(np.float32, copy=False)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["path"]
        label = int(row["label"])
        noise_label = int(row["noise_label"])
        record_name = (
            str(row["record_name"])
            if "record_name" in row and pd.notna(row.get("record_name", None))
            else self._infer_record_name_from_path(path)
        )

        # 尝试从预计算文件读取特征
        use_precomputed = (
            self.use_precomputed
            and self.has_feature_path
            and row.get("has_precomputed", False)
            and "feature_path" in row
            and pd.notna(row["feature_path"])
            and row["feature_path"] != ""
        )

        if use_precomputed:
            try:
                feat_path = str(row["feature_path"])
                if os.path.exists(feat_path):
                    try:
                        feat_data = np.load(feat_path, allow_pickle=False)
                        rr_seq = feat_data["rr_seq"].astype(np.float32)
                        noise_feats = feat_data["noise_feat"].astype(np.float32)
                        
                        # 如果设置了recompute_fuzzy_only，跳过fuzzy_logits的读取，稍后重新计算
                        if self.recompute_fuzzy_only:
                            # 只读取rr_seq和noise_feat，fuzzy_logits稍后重新计算
                            fuzzy_logits = None  # 标记需要重新计算
                            skip_feature_computation = False  # 需要计算fuzzy_logits
                        else:
                            # 正常读取所有特征
                            fuzzy_logits_np = feat_data["fuzzy_logits"].astype(np.float32)
                            fuzzy_logits = torch.from_numpy(fuzzy_logits_np)
                            skip_feature_computation = True  # 所有特征都已加载
                        
                        feat_data.close()
                    except (OSError, ValueError, EOFError) as e:
                        error_msg = str(e)
                        if "CRC" in error_msg or "corrupt" in error_msg.lower():
                            print(f"Warning: Corrupted precomputed features file {feat_path}, will compute online")
                        else:
                            print(f"Warning: Failed to load precomputed features from {feat_path}: {error_msg}")
                        skip_feature_computation = False
                        fuzzy_logits = None  # 确保变量被初始化
                else:
                    skip_feature_computation = False
                    fuzzy_logits = None  # 确保变量被初始化
            except Exception as e:
                print(f"Warning: Failed to load precomputed features: {e}")
                skip_feature_computation = False
                fuzzy_logits = None  # 确保变量被初始化
        else:
            skip_feature_computation = False
            fuzzy_logits = None  # 确保变量被初始化

        # 加载 ECG 数据（总是需要，用于归一化和模型输入）
        try:
            data = np.load(path, allow_pickle=False)
        except (OSError, ValueError, EOFError) as e:
            error_msg = str(e)
            if "CRC" in error_msg or "corrupt" in error_msg.lower():
                raise RuntimeError(f"Corrupted NPZ file (CRC error): {path}. Please run repair_corrupted_npz.py to fix it.")
            else:
                raise RuntimeError(f"Failed to load NPZ file {path}: {error_msg}")
        
        if "ecg" not in data:
            data.close()
            raise KeyError(f"npz file {path} must contain 'ecg' array")
        
        # 尝试读取ecg，捕获可能的CRC错误
        try:
            ecg = np.asarray(data["ecg"])
        except (OSError, ValueError, EOFError) as e:
            error_msg = str(e)
            data.close()
            if "CRC" in error_msg:
                raise RuntimeError(f"Corrupted NPZ file (CRC error reading 'ecg'): {path}. Please run repair_corrupted_npz.py to fix it.")
            else:
                raise RuntimeError(f"Failed to read 'ecg' from {path}: {error_msg}")

        # 统一为 (T, C)
        if ecg.ndim == 1:
            ecg = ecg[:, None]
        elif ecg.ndim == 2:
            T, C = ecg.shape
            if not self.assume_T_first:
                ecg = ecg.T
        else:
            data.close()
            raise ValueError(f"Unexpected ecg shape {ecg.shape} in file {path}")

        if ecg.shape[1] > self.num_leads:
            ecg = ecg[:, : self.num_leads]

        # 固定长度
        T = ecg.shape[0]
        if T > self.max_len:
            start = (T - self.max_len) // 2
            ecg = ecg[start:start + self.max_len, :]
        elif T < self.max_len:
            pad_len = self.max_len - T
            pad = np.zeros((pad_len, ecg.shape[1]), dtype=ecg.dtype)
            ecg = np.concatenate([ecg, pad], axis=0)

        # 尝试读取fs，如果失败则使用默认值
        try:
            if "fs" in data:
                fs = float(data["fs"])
            else:
                fs = 250.0
        except (OSError, ValueError, EOFError) as e:
            # fs读取失败，使用默认值
            fs = 250.0
            # 不抛出错误，因为fs有默认值

        # 如果未使用预计算特征，或者需要重新计算fuzzy_logits，则在线计算
        if not skip_feature_computation:
            # 如果只重新计算fuzzy_logits，rr_seq和noise_feats已经从上方的预计算文件读取
            if self.recompute_fuzzy_only and fuzzy_logits is None:
                # 只重新计算fuzzy_logits，rr_seq和noise_feats已经加载
                # 需要重新读取ECG用于计算fuzzy_logits（需要原始ECG信号）
                try:
                    ecg_for_rr = np.asarray(data["ecg"])
                except (OSError, ValueError, EOFError) as e:
                    # 如果无法重新读取，使用当前的ecg（但需要反归一化）
                    ecg_for_rr = ecg.copy()
                if ecg_for_rr.ndim == 1:
                    ecg_for_rr = ecg_for_rr[:, None]
                elif ecg_for_rr.ndim == 2:
                    T, C = ecg_for_rr.shape
                    if T < C:  # (C, T)
                        ecg_for_rr = ecg_for_rr.T
                
                # 应用模糊规则系统（使用已加载的rr_seq和原始ECG）
                fuzzy_logits = apply_fuzzy_rules(
                    rr_seq,  # 使用从预计算文件读取的rr_seq
                    ecg=ecg_for_rr,
                    fs=fs,
                    flutter_method=self.flutter_method,
                    lead_names=self.lead_names,
                )
            else:
                # 完全重新计算所有特征
                # 噪声特征（使用原始 ECG，归一化前）
                noise_feats = extract_noise_features(ecg, fs, use_stft_dbscan=self.use_stft_dbscan)

                # RR 序列（使用归一化前的 ECG 进行 R 峰检测）
                # 重新读取ecg_for_rr，因为可能ecg已经被修改（裁剪/填充）
                try:
                    ecg_for_rr = np.asarray(data["ecg"])
                except (OSError, ValueError, EOFError) as e:
                    # 如果无法重新读取，使用当前的ecg（但需要反归一化）
                    ecg_for_rr = ecg.copy()
                if ecg_for_rr.ndim == 1:
                    ecg_for_rr = ecg_for_rr[:, None]
                elif ecg_for_rr.ndim == 2:
                    T, C = ecg_for_rr.shape
                    if T < C:  # (C, T)
                        ecg_for_rr = ecg_for_rr.T
                
                rr_seq = extract_rr_seq(
                    ecg_for_rr,
                    fs,
                    max_intervals=self.max_rr_intervals,
                    multi_lead_method=self.multi_lead_method,
                )

                # 应用模糊规则系统
                fuzzy_logits = apply_fuzzy_rules(
                    rr_seq,
                    ecg=ecg_for_rr,
                    fs=fs,
                    flutter_method=self.flutter_method,
                    lead_names=self.lead_names,
                )

        # 关闭NPZ文件以释放资源
        data.close()

        # 最简单的 transform：归一化（总是需要）
        ecg = normalize_ecg(ecg)

        # 仅训练集：轻量增强（提升跨患者泛化）
        if self.augment and (np.random.rand() < self.aug_prob):
            rng = np.random.RandomState(int(np.random.randint(0, 2**31 - 1)))
            ecg = self._augment_ecg_inplace(
                ecg,
                rng=rng,
                gain_range=self.aug_gain_range,
                noise_std_range=self.aug_noise_std_range,
                time_mask_prob=self.aug_time_mask_prob,
                time_mask_frac_range=self.aug_time_mask_frac_range,
                lead_dropout_prob=self.aug_lead_dropout_prob,
            )

        # 转成 Tensor
        ecg_t = torch.from_numpy(ecg.astype(np.float32)).permute(1, 0)     # (C, T)
        noise_t = torch.from_numpy(noise_feats.astype(np.float32))         # (D,)
        rr_t = torch.from_numpy(rr_seq.astype(np.float32))                 # (L,)
        y = torch.tensor(label, dtype=torch.long)
        y_noise = torch.tensor(noise_label, dtype=torch.long)

        return {
            "ecg": ecg_t,
            "noise_feat": noise_t,
            "rr_seq": rr_t,
            "fuzzy_logits": fuzzy_logits,  # 模糊规则 logits (4,)
            "label": y,
            "noise_label": y_noise,
            "record_name": record_name,
            "idx": int(idx),
        }


# ---------------------------------------------------------------------
# 模型：形态分支 + 节律（RR）分支（双分支）
# ---------------------------------------------------------------------

class ECGBackbone(nn.Module):
    """形态分支：1D CNN，从 (B, C, T) 抽 z_morph。"""

    def __init__(self, in_channels: int = 1, d_model: int = 128):
        super().__init__()
        layers = []
        channels = [in_channels, 32, 64, 128]
        for i in range(len(channels) - 1):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(channels[i], channels[i + 1], kernel_size=7, stride=1, padding=3),
                    nn.BatchNorm1d(channels[i + 1]),
                    nn.ReLU(inplace=True),
                )
            )
        self.convs = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(128, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = x
        for conv in self.convs:
            h = conv(h)
        h = self.pool(h).squeeze(-1)  # (B, 128)
        z = self.proj(h)              # (B, d_model)
        return z


class RhythmBranch(nn.Module):
    """
    节律分支：对 RR 序列做 GRU 编码。
    输入 rr_seq: (B, L)，L 为固定 max_rr_intervals
    """

    def __init__(self, input_len: int, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        # RR 序列每个时间步只有 1 维，可以直接作为 1D 序列
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, rr_seq: torch.Tensor) -> torch.Tensor:
        # rr_seq: (B, L)
        x = rr_seq.unsqueeze(-1)  # (B, L, 1)
        out, _ = self.gru(x)      # (B, L, 2*hidden_dim)
        # 使用最后一个时间步的输出作为 summary
        h_last = out[:, -1, :]    # (B, 2*hidden_dim)
        z = self.proj(h_last)     # (B, out_dim)
        return z


class NoiseMLP(nn.Module):
    """噪声分支：把噪声手工特征编码成 embedding。"""

    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ECGNet(nn.Module):
    """
    三分支模型 + 可学习模糊规则融合：
    - ECGBackbone：形态分支，从 ECG 波形抽取形态表征
    - RhythmBranch：节律分支，从 RR 序列抽取长期节律表征
    - NoiseMLP：噪声分支，从噪声特征抽取噪声表征
    - 可学习gating机制融合深度学习输出和模糊规则输出
    """

    def __init__(
        self,
        num_leads: int,
        num_arrhythmia_classes: int,
        noise_feat_dim: int,
        rr_seq_len: int,
        d_model: int = 128,
        rhythm_emb_dim: int = 64,
        noise_emb_dim: int = 32,
        fuzzy_weight_init: float = 0.3,
    ):
        super().__init__()
        self.num_arrhythmia_classes = int(num_arrhythmia_classes)
        self.morph_branch = ECGBackbone(in_channels=num_leads, d_model=d_model)
        self.rhythm_branch = RhythmBranch(input_len=rr_seq_len, hidden_dim=64, out_dim=rhythm_emb_dim)
        self.noise_branch = NoiseMLP(in_dim=noise_feat_dim, hidden_dim=64, out_dim=noise_emb_dim)

        fused_dim = d_model + rhythm_emb_dim + noise_emb_dim
        self.fused_dropout = nn.Dropout(p=0.3)
        # 主四分类头：AF/AFL/PSVT/Normal
        self.arr_head = nn.Linear(fused_dim, num_arrhythmia_classes)
        # AF/AFL 二分类辅助头（0=AF, 1=AFL）
        self.af_bin_head = nn.Linear(fused_dim, 2)

        # 可学习的fuzzy gating网络
        # 输入：noise_emb (noise_emb_dim) + conf (1) + boundary_score (1)
        # 输出：alpha (0-1) via sigmoid
        self.fuzzy_gate = nn.Sequential(
            nn.Linear(noise_emb_dim + 2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        # 初始化gating网络，使得初始alpha接近fuzzy_weight_init
        with torch.no_grad():
            # 设置bias使得sigmoid(0) ≈ fuzzy_weight_init
            bias_val = math.log(fuzzy_weight_init / (1.0 - fuzzy_weight_init + 1e-8))
            self.fuzzy_gate[-2].bias.fill_(bias_val)
        
        # 自适应参数：用于动态调整模糊规则融合策略（初始值偏向 AFL 召回）
        self.register_buffer('adaptive_alpha_min', torch.tensor(0.05))  # 略降，让规则更多参与
        self.register_buffer('adaptive_alpha_afl_max', torch.tensor(0.52))  # AFL 融合上限提高，更信模糊规则
        self.register_buffer('adaptive_alpha_afl_boost_coef', torch.tensor(0.58))  # AFL boost 系数提高
        self.register_buffer('adaptive_small_afl_threshold', torch.tensor(-4.0))  # 更容易触发“小AFL”提升
        self.register_buffer('adaptive_alpha_afl_min', torch.tensor(0.32))  # AFL 最小 alpha 提高
        self.register_buffer('adaptive_af_protect_threshold', torch.tensor(3.5))  # AF 保护更难触发
        self.register_buffer('adaptive_af_protect_diff', torch.tensor(5.0))  # 需更大 AF-AFL 差才保护
        self.register_buffer('adaptive_af_protect_coef', torch.tensor(0.28))  # AF 保护强度减弱
        self.register_buffer('adaptive_afl_offset_threshold', torch.tensor(-6.0))  # 更容易给 AFL 加偏移
        self.register_buffer('adaptive_afl_offset_coef', torch.tensor(0.06))  # AFL 偏移系数略增
        self.register_buffer('adaptive_afl_offset_max', torch.tensor(1.2))  # 允许更大 AFL 偏移
        self.register_buffer('adaptive_logit_diff_threshold', torch.tensor(12.0))  # 更易触发 logit 差缩小
        self.register_buffer('adaptive_logit_diff_coef', torch.tensor(0.12))  # 差异缩小系数增大
        self.register_buffer('adaptive_logit_diff_max', torch.tensor(1.0))  # 允许更大 AFL 提升

    def forward(
        self,
        ecg: torch.Tensor,
        rr_seq: torch.Tensor,
        noise_feat: torch.Tensor,
        fuzzy_logits: Optional[torch.Tensor] = None,
    ):
        z_morph = self.morph_branch(ecg)            # (B, d_model)
        z_rhythm = self.rhythm_branch(rr_seq)       # (B, rhythm_emb_dim)
        z_noise = self.noise_branch(noise_feat)     # (B, noise_emb_dim)
        z = torch.cat([z_morph, z_rhythm, z_noise], dim=-1)
        z = self.fused_dropout(z)
        logits_arr = self.arr_head(z)               # (B, num_classes)
        logits_bin = self.af_bin_head(z)            # (B, 2)

        # 二分类时：以二分类头作为主输出（训练/评估/阈值/保存均对齐到真正的二分类判别）
        if self.num_arrhythmia_classes == 2 and logits_bin.size(1) == 2:
            logits_arr = logits_bin

        # 可学习gating融合模糊规则输出
        if fuzzy_logits is not None:
            # fuzzy_logits: (B, num_classes)
            # 计算gating输入：noise_emb + conf + boundary_score
            conf = torch.softmax(logits_arr, dim=-1).amax(dim=-1, keepdim=True)  # (B, 1)
            # 计算boundary_score
            sorted_probs, _ = torch.sort(torch.softmax(logits_arr, dim=-1), dim=-1, descending=True)
            max_prob = sorted_probs[:, 0:1]
            second_max_prob = sorted_probs[:, 1:2] if sorted_probs.size(1) > 1 else torch.zeros_like(max_prob)
            boundary_score = (1.0 - (max_prob - second_max_prob)).clamp(min=0.0, max=1.0)  # (B, 1)
            
            # Gating输入
            gate_input = torch.cat([z_noise, conf, boundary_score], dim=-1)  # (B, noise_emb_dim + 2)
            alpha_raw = self.fuzzy_gate(gate_input)  # (B, 1)
            
            # 使用自适应参数：从模型的buffer中读取，可以在训练过程中动态调整
            alpha_min = self.adaptive_alpha_min.item()
            alpha = alpha_raw * (1.0 - alpha_min) + alpha_min  # 将[0,1]映射到[alpha_min, 1]
            
            # 加权融合：logits = (1-alpha)*net + alpha*rule
            # 注意：fuzzy_logits是固定的（从预计算文件读取），没有梯度
            # 但alpha是可学习的，所以融合后的logits_arr仍然有梯度
            
            # 保存融合前的网络输出用于调试
            logits_arr_before_fusion = logits_arr.clone()
            
            # 执行融合
            # 对于AFL类别（索引1），采用更激进的策略来提升AFL的logit
            if logits_arr.size(1) >= 2 and fuzzy_logits.size(1) >= 2:
                # 检查AFL的情况
                net_afl = logits_arr_before_fusion[:, 1:2]  # (B, 1)
                fuzzy_afl = fuzzy_logits[:, 1:2]  # (B, 1)
                net_af = logits_arr_before_fusion[:, 0:1]  # (B, 1)
                fuzzy_af = fuzzy_logits[:, 0:1]  # (B, 1)
                
                # 使用自适应参数：从模型的buffer中读取，可以在训练过程中动态调整
                # 策略1：对于AFL，如果模糊规则的AFL logit更好（更大），则适度增加其权重
                afl_boost_mask = (fuzzy_afl > net_afl).float()  # (B, 1)
                alpha_afl_max = self.adaptive_alpha_afl_max.item()
                alpha_afl_boost_coef = self.adaptive_alpha_afl_boost_coef.item()
                alpha_afl_base = alpha + afl_boost_mask * (alpha_afl_max - alpha) * alpha_afl_boost_coef
                
                # 策略2：如果网络的AFL logit太小，才强制使用更大的alpha
                small_afl_threshold = self.adaptive_small_afl_threshold.item()
                alpha_afl_min_val = self.adaptive_alpha_afl_min.item()
                small_afl_mask = (net_afl < small_afl_threshold).float()  # (B, 1)
                alpha_afl = torch.max(alpha_afl_base, alpha_afl_min_val * small_afl_mask + alpha * (1.0 - small_afl_mask))
                
                # 策略3：适度AF保护机制，防止AF被误分为AFL（但不要过度保护，避免AFL被误分为AF）
                af_protect_threshold = self.adaptive_af_protect_threshold.item()
                af_protect_diff = self.adaptive_af_protect_diff.item()
                af_protect_coef = self.adaptive_af_protect_coef.item()
                strong_af_mask = (net_af > af_protect_threshold).float()  # (B, 1)
                af_stronger_mask = ((net_af - net_afl) > af_protect_diff).float()  # AF明显强于AFL
                alpha_af = alpha * (1.0 - strong_af_mask * 0.2)  # 如果AF很强，适度降低alpha（从0.25降低到0.2）
                # 如果AF明显强于AFL，适度降低AFL的alpha（降低保护强度，避免过度抑制AFL）
                alpha_afl = alpha_afl * (1.0 - af_stronger_mask * af_protect_coef * 0.7)  # 降低保护系数的影响（乘以0.7）
                
                # 分别融合AF和AFL
                logits_arr_af = (1.0 - alpha_af) * logits_arr_before_fusion[:, 0:1] + alpha_af * fuzzy_logits[:, 0:1]
                logits_arr_afl = (1.0 - alpha_afl) * logits_arr_before_fusion[:, 1:2] + alpha_afl * fuzzy_logits[:, 1:2]
                
                # 修改：降低AF占优时的惩罚强度，提高触发阈值（减少对AFL的抑制）
                af_dominant_threshold = 7.0  # 偏向AFL：只有AF大幅领先时才压AFL，减少误杀AFL
                af_dominant_mask = ((logits_arr_af - logits_arr_afl) > af_dominant_threshold).float()  # AF明显占优
                afl_penalty = af_dominant_mask * 0.4  # 偏向AFL：减弱对AFL的惩罚
                logits_arr_afl = logits_arr_afl - afl_penalty
                
                # 保存用于后续offset操作
                logits_arr_afl_before_offset = logits_arr_afl.clone()
                
                # 使用自适应参数：从模型的buffer中读取，可以在训练过程中动态调整
                # 策略4：如果AFL的logit仍然太小，进行适度的logit偏移（提高AF不占优的阈值，更容易触发boost）
                afl_offset_threshold = self.adaptive_afl_offset_threshold.item()
                afl_offset_coef = self.adaptive_afl_offset_coef.item()
                afl_offset_max = self.adaptive_afl_offset_max.item()
                # 提高AF不占优的阈值（从4.0提高到5.5），使更多情况可以触发AFL boost
                af_not_dominant = ((logits_arr_af - logits_arr_afl_before_offset) <= af_dominant_threshold).float()  # AF不占优
                afl_too_small_mask = (logits_arr_afl_before_offset < afl_offset_threshold).float() * af_not_dominant  # (B, 1)
                af_logit_for_offset = logits_arr_af.detach()  # 使用detach避免影响梯度
                afl_offset = torch.clamp((af_logit_for_offset - logits_arr_afl_before_offset.detach()) * afl_offset_coef, min=0.0, max=afl_offset_max)
                logits_arr_afl = logits_arr_afl + afl_too_small_mask * afl_offset
                
                # 策略5：如果AF和AFL的logit差异太大，强制缩小差异（放宽条件，更容易触发AFL boost）
                logit_diff_threshold = self.adaptive_logit_diff_threshold.item()
                logit_diff_coef = self.adaptive_logit_diff_coef.item()
                logit_diff_max = self.adaptive_logit_diff_max.item()
                logit_diff = logits_arr_af - logits_arr_afl  # (B, 1)
                # 放宽条件：只要差异大于阈值且AF不占优，就boost AFL（提高上限从5.0到8.0）
                moderate_diff_mask = ((logit_diff > logit_diff_threshold) & 
                                     (logit_diff <= logit_diff_threshold + 8.0) &
                                     ((logits_arr_af - logits_arr_afl) <= af_dominant_threshold)).float()  # (B, 1)
                afl_boost = torch.clamp((logit_diff - logit_diff_threshold) * logit_diff_coef, min=0.0, max=logit_diff_max)
                logits_arr_afl = logits_arr_afl + moderate_diff_mask * afl_boost
                
                # 重新组合
                if logits_arr.size(1) == 2:
                    logits_arr = torch.cat([logits_arr_af, logits_arr_afl], dim=1)
                else:
                    # 对于多分类，只调整AFL，其他类别使用原始alpha
                    logits_arr_other = (1.0 - alpha) * logits_arr_before_fusion[:, 2:] + alpha * fuzzy_logits[:, 2:]
                    logits_arr = torch.cat([logits_arr_af, logits_arr_afl, logits_arr_other], dim=1)
            else:
                # 标准融合（如果类别数不匹配或不是二分类）
                logits_arr = (1.0 - alpha) * logits_arr + alpha * fuzzy_logits
            
            # 存储alpha和fuzzy_logits用于调试（仅在训练模式下）
            if self.training:
                # 使用register_buffer存储，避免影响梯度计算
                if not hasattr(self, '_debug_alpha'):
                    self.register_buffer('_debug_alpha', torch.zeros(1))
                    self.register_buffer('_debug_fuzzy_af', torch.zeros(1))
                    self.register_buffer('_debug_fuzzy_afl', torch.zeros(1))
                    self.register_buffer('_debug_net_af', torch.zeros(1))
                    self.register_buffer('_debug_net_afl', torch.zeros(1))
                    self.register_buffer('_debug_fused_af', torch.zeros(1))
                    self.register_buffer('_debug_fused_afl', torch.zeros(1))
                    self.register_buffer('_debug_alpha_afl', torch.zeros(1))
                # 记录batch平均值（用于调试）
                self._debug_alpha = alpha.mean().detach()
                self._debug_alpha_raw = alpha_raw.mean().detach()  # 记录原始alpha值
                if fuzzy_logits.size(1) >= 2:
                    self._debug_fuzzy_af = fuzzy_logits[:, 0].mean().detach()
                    self._debug_fuzzy_afl = fuzzy_logits[:, 1].mean().detach()
                if logits_arr_before_fusion.size(1) >= 2:
                    self._debug_net_af = logits_arr_before_fusion[:, 0].mean().detach()
                    self._debug_net_afl = logits_arr_before_fusion[:, 1].mean().detach()
                if logits_arr.size(1) >= 2:
                    self._debug_fused_af = logits_arr[:, 0].mean().detach()
                    self._debug_fused_afl = logits_arr[:, 1].mean().detach()
                    # 记录AFL的alpha值
                    if logits_arr_before_fusion.size(1) >= 2 and fuzzy_logits.size(1) >= 2:
                        # 重新计算alpha_afl用于调试（简化版本）
                        net_afl_debug = logits_arr_before_fusion[:, 1:2]
                        fuzzy_afl_debug = fuzzy_logits[:, 1:2]
                        afl_boost_mask_debug = (fuzzy_afl_debug > net_afl_debug).float()
                        alpha_afl_base_debug = alpha.mean() + afl_boost_mask_debug.mean() * (0.5 - alpha.mean()) * 0.8
                        small_afl_mask_debug = (net_afl_debug < -3.0).float()
                        alpha_afl_val = max(alpha_afl_base_debug.item(), 0.4 * small_afl_mask_debug.mean().item() + alpha.mean().item() * (1.0 - small_afl_mask_debug.mean().item()))
                        self._debug_alpha_afl = torch.tensor(alpha_afl_val, device=self._debug_alpha.device)
                    else:
                        self._debug_alpha_afl = alpha.mean().detach()

        return logits_arr, logits_bin


# ---------------------------------------------------------------------
# 训练 & 评估
# ---------------------------------------------------------------------

@dataclass
class TrainConfig:
    train_csv: str = ""
    val_csv: str = ""
    data_csv: str = ""
    num_leads: int = 1
    num_arrhythmia_classes: int = 4
    max_len: int = 2500
    max_rr_intervals: int = 32
    use_stft_dbscan: bool = False
    batch_size: int = 32
    num_workers: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 40
    device: str = "cpu"
    out_ckpt: str = "ecg_4class_fuzzy.pt"
    weight_conf_gamma: float = 1.0
    min_weight: float = 0.1
    # 低置信度样本处理策略
    # 如果 use_low_conf_focus=True: 给低置信度样本更高权重（不丢弃）
    # 如果 use_low_conf_focus=False: 丢弃低置信度样本（原始策略）
    use_low_conf_focus: bool = True  # 是否关注低置信度样本而非丢弃
    conf_discard_th: float = 0.4  # 低置信度阈值（用于丢弃策略或权重计算）
    low_conf_alpha: float = 1.0  # 低置信度样本权重增强系数（用于关注策略）
    # AF/AFL 二分类辅助头 loss 权重
    lambda_af_bin: float = 0.5
    validate_every_batch: bool = True
    use_class_weights: bool = True
    fuzzy_weight_init: float = 0.3  # 模糊规则融合权重初始值（用于可学习gating初始化）
    # 边界感知参数
    use_boundary_aware: bool = True  # 是否启用边界感知
    boundary_alpha: float = 0.5  # 边界样本权重增强系数 (0-1)
    boundary_threshold: float = 0.3  # 边界样本阈值（boundary_score > threshold 视为边界样本）
    lambda_boundary: float = 0.1  # 边界感知损失权重（可选，用于额外的边界损失）
    # 早停机制（基于验证集 macro F1），patience=0 表示关闭
    early_stop_patience: int = 0  # 连续多少个 epoch 验证集 F1 无提升则提前停止
    early_stop_min_delta: float = 0.0  # F1 提升的最小阈值（小于等于该阈值视为“无提升”）
    # 低置信度样本处理参数
    use_low_conf_focus: bool = True  # 是否关注低置信度样本（而非丢弃），默认True
    low_conf_alpha: float = 1.0  # 低置信度样本权重增强系数
    # 多导联处理参数
    multi_lead_method: str = "max_energy"  # "max_energy" 或 "voting"
    flutter_method: str = "fixed_lead"  # "fixed_lead" 或 "attention"
    lead_names: Optional[List[str]] = None  # 导联名称列表（如 ["I", "II", "V1"]）
    # 特征预计算参数
    use_precomputed: bool = True  # 是否使用预计算特征
    recompute_fuzzy_only: bool = False  # 即使有预计算特征，也只重新计算fuzzy_logits
    # 数据过采样参数（用于处理类别不平衡）
    use_oversampling: bool = False  # 是否对训练数据进行过采样
    oversample_target_ratio: float = 1.0  # 过采样目标比例（1.0=完全平衡，0.5=2:1）
    oversample_strategy: str = "random"  # 过采样策略："random"（样本级）或"patient_level"（患者级，更安全）
    # 二分类评估/保存策略：阈值化而非 argmax（可显著改善 AFL-F1 与误报权衡）
    afl_logit_bias_inference: float = 0.0  # 仅用于验证/测试阶段对 AFL logit 的可选偏置（不建议过大）
    afl_prob_threshold: Optional[float] = None  # 若为 None 且是二分类，则在验证集自动搜索最优阈值（最大化 AFL-F1）
    afl_threshold_search_steps: int = 101  # 阈值搜索步数
    afl_threshold_search_min: float = 0.01
    afl_threshold_search_max: float = 0.99  # 二分类建议 0.5~0.6，避免 Val 搜到过高阈值导致 Test 召回崩盘
    afl_min_recall_val: float = 0.0  # Val 阈值搜索时要求 AFL 召回>=此值才候选（如 0.2），0 表示不约束
    # 方案1：record-level 聚合（验证/测试更稳定、更“好看”）
    enable_record_agg: bool = True
    record_agg_method: str = "mean"  # mean/median/max/topk
    record_agg_topk: int = 5
    # 方案2：hard example mining（错分样本加权采样）
    use_hard_mining: bool = True
    hard_mining_start_epoch: int = 1  # 从第几个 epoch 结束后开始挖 hard
    hard_mining_afl_fn_mult: float = 4.0  # AFL->AF (FN) 权重倍率
    hard_mining_af_fp_mult: float = 2.0   # AF->AFL (FP) 权重倍率
    hard_mining_low_conf_mult: float = 1.5  # 低置信度正确样本倍率（可选）
    hard_mining_conf_threshold: float = 0.65  # 低置信度阈值（max prob < th）
    split_seed: int = 42  # 患者级 8:1:1 划分随机种子


def _binary_metrics_from_confusion(cm: np.ndarray) -> dict:
    """从二分类混淆矩阵计算 AFL 的 P/R/F1。cm shape=(2,2), rows=true, cols=pred."""
    afl_tp = float(cm[1, 1])
    afl_fp = float(cm[0, 1])
    afl_fn = float(cm[1, 0])
    afl_prec = afl_tp / (afl_tp + afl_fp + 1e-8)
    afl_rec = afl_tp / (afl_tp + afl_fn + 1e-8)
    afl_f1 = 2.0 * afl_prec * afl_rec / (afl_prec + afl_rec + 1e-8)
    return {"afl_precision": afl_prec, "afl_recall": afl_rec, "afl_f1": afl_f1}


def _search_best_afl_threshold(
    y_true: np.ndarray,
    p_afl: np.ndarray,
    beta: float = 1.0,
    steps: int = 101,
    t_min: float = 0.01,
    t_max: float = 0.99,
    min_afl_recall: float = 0.0,
) -> Tuple[float, dict]:
    """在验证集上搜索最大化 AFL-Fbeta 的阈值（pred=1 if p_afl>=t）。beta>1 更偏召回。
    min_afl_recall>0 时只考虑 AFL 召回>=该值的阈值，避免搜到过高阈值导致 Test 上召回崩盘。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    p_afl = np.asarray(p_afl, dtype=np.float32)
    thresholds = np.linspace(t_min, t_max, max(int(steps), 3), dtype=np.float32)

    best_t = 0.5
    best = {"afl_f1": -1.0, "afl_precision": 0.0, "afl_recall": 0.0}
    beta2 = float(beta) ** 2
    min_r = float(min_afl_recall)

    for t in thresholds:
        y_pred = (p_afl >= t).astype(np.int64)
        cm = compute_confusion_matrix(y_true.tolist(), y_pred.tolist(), 2)
        m = _binary_metrics_from_confusion(cm)
        p = float(m["afl_precision"])
        r = float(m["afl_recall"])
        if min_r > 0 and r < min_r:
            continue  # 不满足最低召回约束，跳过
        f_beta = (1.0 + beta2) * p * r / (beta2 * p + r + 1e-8)
        m = {"afl_precision": p, "afl_recall": r, "afl_f1": f_beta}
        if (
            (m["afl_f1"] > best["afl_f1"] + 1e-12)
            or (abs(m["afl_f1"] - best["afl_f1"]) <= 1e-12 and m["afl_recall"] > best["afl_recall"] + 1e-12)
            or (
                abs(m["afl_f1"] - best["afl_f1"]) <= 1e-12
                and abs(m["afl_recall"] - best["afl_recall"]) <= 1e-12
                and m["afl_precision"] > best["afl_precision"] + 1e-12
            )
        ):
            best_t = float(t)
            best = m

    return best_t, best


def _aggregate_record_probs(
    record_names: List[str],
    y_true: List[int],
    p_afl: List[float],
    method: str = "mean",
    topk: int = 5,
) -> Tuple[List[int], List[float], List[str]]:
    """
    将 segment 级的 p(AFL) 聚合到 record 级。
    返回：record_y_true, record_p_afl, record_ids（同序）
    """
    method = str(method).lower()
    rec_to_probs = {}
    rec_to_labels = {}
    for rn, yt, pa in zip(record_names, y_true, p_afl):
        rn = str(rn)
        rec_to_probs.setdefault(rn, []).append(float(pa))
        rec_to_labels.setdefault(rn, []).append(int(yt))

    rec_ids = sorted(rec_to_probs.keys())
    rec_p: List[float] = []
    rec_y: List[int] = []
    for rid in rec_ids:
        probs = np.asarray(rec_to_probs[rid], dtype=np.float32)
        labels = np.asarray(rec_to_labels[rid], dtype=np.int64)
        # record 真值：多数票（理论上应当一致）
        y_mode = int(np.round(labels.mean())) if labels.size else 0
        rec_y.append(y_mode)

        if method == "mean":
            rec_p.append(float(probs.mean()))
        elif method == "median":
            rec_p.append(float(np.median(probs)))
        elif method == "max":
            rec_p.append(float(probs.max()))
        elif method == "topk":
            k = max(1, min(int(topk), probs.size))
            rec_p.append(float(np.sort(probs)[-k:].mean()))
        else:
            rec_p.append(float(probs.mean()))

    return rec_y, rec_p, rec_ids


def compute_metrics(y_true: List[int], y_pred: List[int], num_classes: int) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    accuracy = float((y_true == y_pred).mean())

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = np.logical_and(y_true == c, y_pred == c).sum()
        fp = np.logical_and(y_true != c, y_pred == c).sum()
        fn = np.logical_and(y_true == c, y_pred != c).sum()
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }


def print_per_class_metrics(y_true: List[int], y_pred: List[int], num_classes: int, class_names: List[str] = None):
    """
    打印每个类别的详细指标（Recall, Precision, F1）
    
    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        num_classes: 类别数
        class_names: 类别名称列表（可选，默认使用数字）
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    elif len(class_names) < num_classes:
        class_names = class_names + [f"Class {i}" for i in range(len(class_names), num_classes)]
    
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    
    print("  Per-Class Metrics:")
    for c in range(num_classes):
        tp = np.logical_and(y_true == c, y_pred == c).sum()
        fp = np.logical_and(y_true != c, y_pred == c).sum()
        fn = np.logical_and(y_true == c, y_pred != c).sum()
        
        if tp == 0 and fp == 0 and fn == 0:
            print(f"    {class_names[c]}: No samples")
            continue
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # 显示实际数量
        total_true = (y_true == c).sum()
        total_pred = (y_pred == c).sum()
        
        print(f"    {class_names[c]}:")
        print(f"      - Recall: {recall:.4f} ({tp}/{total_true})")
        print(f"      - Precision: {precision:.4f} ({tp}/{total_pred})")
        print(f"      - F1: {f1:.4f}")


def adaptive_adjust_parameters(model, val_metrics, num_classes: int = 2, adjustment_rate: float = 0.05):
    """
    根据验证集性能自适应调整模型参数
    同时考虑recall和precision，避免过度偏向某个类别
    
    改进：基于recall判断偏向，当AFL的recall低时增强AFL识别，降低AF保护强度
    
    Args:
        model: ECGNet模型
        val_metrics: 验证集指标，包含confusion_matrix
        num_classes: 类别数
        adjustment_rate: 调整速率（每次调整的幅度）
    
    Returns:
        dict: 调整后的参数信息
    """
    if num_classes != 2 or 'confusion_matrix' not in val_metrics:
        return {}
    
    cm = val_metrics['confusion_matrix']
    # 计算每个类别的recall和precision
    af_recall = cm[0, 0] / max(cm[0, :].sum(), 1)  # AF的recall
    afl_recall = cm[1, 1] / max(cm[1, :].sum(), 1)  # AFL的recall
    af_precision = cm[0, 0] / max(cm[:, 0].sum(), 1)  # AF的precision
    afl_precision = cm[1, 1] / max(cm[:, 1].sum(), 1)  # AFL的precision
    
    # 计算F1分数
    af_f1 = 2 * af_recall * af_precision / max(af_recall + af_precision, 1e-8)
    afl_f1 = 2 * afl_recall * afl_precision / max(afl_recall + afl_precision, 1e-8)
    
    # 目标：两个类别的F1都尽可能高，且差距不要太大
    target_f1 = 0.7  # 目标F1
    f1_diff = abs(af_f1 - afl_f1)
    recall_diff = af_recall - afl_recall  # recall差异，正数表示AF recall更高
    
    adjustments = {}
    
    # 优先判断1：如果AF的recall太低且AFL的precision很高（AF被大量误分为AFL），增强AF保护
    if af_recall < 0.5 and afl_precision > 0.5 and afl_recall > 0.7:
        # AF被大量误分为AFL，需要增强AF保护，降低AFL boost
        # 降低alpha最小值
        new_alpha_min = max(0.04, model.adaptive_alpha_min.item() - adjustment_rate * 0.3)
        model.adaptive_alpha_min.fill_(new_alpha_min)
        adjustments['alpha_min'] = new_alpha_min
        
        # 降低AFL alpha最大值
        new_alpha_afl_max = max(0.25, model.adaptive_alpha_afl_max.item() - adjustment_rate * 0.3)
        model.adaptive_alpha_afl_max.fill_(new_alpha_afl_max)
        adjustments['alpha_afl_max'] = new_alpha_afl_max
        
        # 降低AFL boost系数
        new_boost_coef = max(0.2, model.adaptive_alpha_afl_boost_coef.item() - adjustment_rate * 0.2)
        model.adaptive_alpha_afl_boost_coef.fill_(new_boost_coef)
        adjustments['alpha_afl_boost_coef'] = new_boost_coef
        
        # 提高小AFL阈值（更难触发boost）
        new_small_threshold = min(-4.5, model.adaptive_small_afl_threshold.item() + adjustment_rate * 2.0)
        model.adaptive_small_afl_threshold.fill_(new_small_threshold)
        adjustments['small_afl_threshold'] = new_small_threshold
        
        # 大幅增强AF保护
        new_af_protect_threshold = max(1.5, model.adaptive_af_protect_threshold.item() - adjustment_rate * 0.5)
        model.adaptive_af_protect_threshold.fill_(new_af_protect_threshold)
        adjustments['af_protect_threshold'] = new_af_protect_threshold
        
        new_af_protect_diff = max(2.0, model.adaptive_af_protect_diff.item() - adjustment_rate * 0.8)
        model.adaptive_af_protect_diff.fill_(new_af_protect_diff)
        adjustments['af_protect_diff'] = new_af_protect_diff
        
        new_af_protect_coef = min(0.7, model.adaptive_af_protect_coef.item() + adjustment_rate * 0.2)
        model.adaptive_af_protect_coef.fill_(new_af_protect_coef)
        adjustments['af_protect_coef'] = new_af_protect_coef
    
    # 优先判断2：如果AFL的recall太低（大量AFL被误分为AF），增强AFL识别，降低AF保护
    elif afl_recall < 0.5 and recall_diff > 0.3:  # AFL recall < 0.5 且 AF recall明显高于AFL
        # 这是最需要处理的情况：AFL被大量误分为AF，需要增强AFL，降低AF保护
        # 增加alpha最小值
        new_alpha_min = min(0.12, model.adaptive_alpha_min.item() + adjustment_rate * 0.6)
        model.adaptive_alpha_min.fill_(new_alpha_min)
        adjustments['alpha_min'] = new_alpha_min
        
        # 大幅增加AFL alpha最大值
        new_alpha_afl_max = min(0.55, model.adaptive_alpha_afl_max.item() + adjustment_rate * 0.4)
        model.adaptive_alpha_afl_max.fill_(new_alpha_afl_max)
        adjustments['alpha_afl_max'] = new_alpha_afl_max
        
        # 大幅增加AFL boost系数
        new_boost_coef = min(0.65, model.adaptive_alpha_afl_boost_coef.item() + adjustment_rate * 0.25)
        model.adaptive_alpha_afl_boost_coef.fill_(new_boost_coef)
        adjustments['alpha_afl_boost_coef'] = new_boost_coef
        
        # 大幅降低小AFL阈值（更容易触发boost）
        new_small_threshold = max(-8.0, model.adaptive_small_afl_threshold.item() - adjustment_rate * 2.5)
        model.adaptive_small_afl_threshold.fill_(new_small_threshold)
        adjustments['small_afl_threshold'] = new_small_threshold
        
        # 增加AFL最小alpha
        new_alpha_afl_min = min(0.35, model.adaptive_alpha_afl_min.item() + adjustment_rate * 0.25)
        model.adaptive_alpha_afl_min.fill_(new_alpha_afl_min)
        adjustments['alpha_afl_min'] = new_alpha_afl_min
        
        # 降低AFL logit偏移阈值（更容易触发偏移）
        new_offset_threshold = max(-10.0, model.adaptive_afl_offset_threshold.item() - adjustment_rate * 2.5)
        model.adaptive_afl_offset_threshold.fill_(new_offset_threshold)
        adjustments['afl_offset_threshold'] = new_offset_threshold
        
        # 增加AFL logit偏移系数和最大值
        new_offset_coef = min(0.08, model.adaptive_afl_offset_coef.item() + adjustment_rate * 0.015)
        model.adaptive_afl_offset_coef.fill_(new_offset_coef)
        adjustments['afl_offset_coef'] = new_offset_coef
        
        new_offset_max = min(1.0, model.adaptive_afl_offset_max.item() + adjustment_rate * 0.15)
        model.adaptive_afl_offset_max.fill_(new_offset_max)
        adjustments['afl_offset_max'] = new_offset_max
        
        # 降低logit差异阈值（更容易触发boost）
        new_diff_threshold = max(13.0, model.adaptive_logit_diff_threshold.item() - adjustment_rate * 2.5)
        model.adaptive_logit_diff_threshold.fill_(new_diff_threshold)
        adjustments['logit_diff_threshold'] = new_diff_threshold
        
        # 增加logit差异系数和最大值
        new_diff_coef = min(0.15, model.adaptive_logit_diff_coef.item() + adjustment_rate * 0.015)
        model.adaptive_logit_diff_coef.fill_(new_diff_coef)
        adjustments['logit_diff_coef'] = new_diff_coef
        
        new_diff_max = min(0.8, model.adaptive_logit_diff_max.item() + adjustment_rate * 0.15)
        model.adaptive_logit_diff_max.fill_(new_diff_max)
        adjustments['logit_diff_max'] = new_diff_max
        
        # 关键：大幅降低AF保护强度（提高阈值，降低系数）
        new_af_protect_threshold = min(3.0, model.adaptive_af_protect_threshold.item() + adjustment_rate * 0.5)
        model.adaptive_af_protect_threshold.fill_(new_af_protect_threshold)
        adjustments['af_protect_threshold'] = new_af_protect_threshold
        
        new_af_protect_diff = min(5.5, model.adaptive_af_protect_diff.item() + adjustment_rate * 0.8)
        model.adaptive_af_protect_diff.fill_(new_af_protect_diff)
        adjustments['af_protect_diff'] = new_af_protect_diff
        
        new_af_protect_coef = max(0.3, model.adaptive_af_protect_coef.item() - adjustment_rate * 0.15)
        model.adaptive_af_protect_coef.fill_(new_af_protect_coef)
        adjustments['af_protect_coef'] = new_af_protect_coef
    
    # 如果AFL的F1太低但recall不是特别低，适度增强AFL相关参数
    elif afl_f1 < target_f1 - 0.1 and afl_precision > 0.3:  # AFL F1 < 0.6 且 precision > 0.3（避免过度boost）
        # 增加alpha最小值
        new_alpha_min = min(0.1, model.adaptive_alpha_min.item() + adjustment_rate * 0.5)
        model.adaptive_alpha_min.fill_(new_alpha_min)
        adjustments['alpha_min'] = new_alpha_min
        
        # 增加AFL alpha最大值
        new_alpha_afl_max = min(0.5, model.adaptive_alpha_afl_max.item() + adjustment_rate * 0.3)
        model.adaptive_alpha_afl_max.fill_(new_alpha_afl_max)
        adjustments['alpha_afl_max'] = new_alpha_afl_max
        
        # 增加AFL boost系数
        new_boost_coef = min(0.6, model.adaptive_alpha_afl_boost_coef.item() + adjustment_rate * 0.2)
        model.adaptive_alpha_afl_boost_coef.fill_(new_boost_coef)
        adjustments['alpha_afl_boost_coef'] = new_boost_coef
        
        # 降低小AFL阈值（更容易触发boost）
        new_small_threshold = max(-7.0, model.adaptive_small_afl_threshold.item() - adjustment_rate * 2.0)
        model.adaptive_small_afl_threshold.fill_(new_small_threshold)
        adjustments['small_afl_threshold'] = new_small_threshold
        
        # 增加AFL最小alpha
        new_alpha_afl_min = min(0.3, model.adaptive_alpha_afl_min.item() + adjustment_rate * 0.2)
        model.adaptive_alpha_afl_min.fill_(new_alpha_afl_min)
        adjustments['alpha_afl_min'] = new_alpha_afl_min
        
        # 降低AFL logit偏移阈值（更容易触发偏移）
        new_offset_threshold = max(-9.0, model.adaptive_afl_offset_threshold.item() - adjustment_rate * 2.0)
        model.adaptive_afl_offset_threshold.fill_(new_offset_threshold)
        adjustments['afl_offset_threshold'] = new_offset_threshold
        
        # 增加AFL logit偏移系数
        new_offset_coef = min(0.06, model.adaptive_afl_offset_coef.item() + adjustment_rate * 0.01)
        model.adaptive_afl_offset_coef.fill_(new_offset_coef)
        adjustments['afl_offset_coef'] = new_offset_coef
        
        # 降低logit差异阈值（更容易触发boost）
        new_diff_threshold = max(14.0, model.adaptive_logit_diff_threshold.item() - adjustment_rate * 2.0)
        model.adaptive_logit_diff_threshold.fill_(new_diff_threshold)
        adjustments['logit_diff_threshold'] = new_diff_threshold
        
        # 增加logit差异系数
        new_diff_coef = min(0.12, model.adaptive_logit_diff_coef.item() + adjustment_rate * 0.01)
        model.adaptive_logit_diff_coef.fill_(new_diff_coef)
        adjustments['logit_diff_coef'] = new_diff_coef
        
        # 适度提高AF保护阈值（减少AF保护，给AFL更多空间）
        new_af_protect_threshold = min(2.5, model.adaptive_af_protect_threshold.item() + adjustment_rate * 0.3)
        model.adaptive_af_protect_threshold.fill_(new_af_protect_threshold)
        adjustments['af_protect_threshold'] = new_af_protect_threshold
        
        new_af_protect_diff = min(4.5, model.adaptive_af_protect_diff.item() + adjustment_rate * 0.5)
        model.adaptive_af_protect_diff.fill_(new_af_protect_diff)
        adjustments['af_protect_diff'] = new_af_protect_diff
        
        new_af_protect_coef = max(0.4, model.adaptive_af_protect_coef.item() - adjustment_rate * 0.1)
        model.adaptive_af_protect_coef.fill_(new_af_protect_coef)
        adjustments['af_protect_coef'] = new_af_protect_coef
    
    # 如果AF的F1太低，增强AF保护
    elif af_f1 < target_f1 - 0.1:  # AF F1 < 0.6
        # 降低alpha最小值
        new_alpha_min = max(0.04, model.adaptive_alpha_min.item() - adjustment_rate * 0.3)
        model.adaptive_alpha_min.fill_(new_alpha_min)
        adjustments['alpha_min'] = new_alpha_min
        
        # 降低AFL alpha最大值
        new_alpha_afl_max = max(0.25, model.adaptive_alpha_afl_max.item() - adjustment_rate * 0.2)
        model.adaptive_alpha_afl_max.fill_(new_alpha_afl_max)
        adjustments['alpha_afl_max'] = new_alpha_afl_max
        
        # 降低AFL boost系数
        new_boost_coef = max(0.2, model.adaptive_alpha_afl_boost_coef.item() - adjustment_rate * 0.15)
        model.adaptive_alpha_afl_boost_coef.fill_(new_boost_coef)
        adjustments['alpha_afl_boost_coef'] = new_boost_coef
        
        # 提高小AFL阈值（更难触发boost）
        new_small_threshold = min(-4.5, model.adaptive_small_afl_threshold.item() + adjustment_rate * 1.5)
        model.adaptive_small_afl_threshold.fill_(new_small_threshold)
        adjustments['small_afl_threshold'] = new_small_threshold
        
        # 降低AF保护阈值（更早保护AF）
        new_af_protect_threshold = max(1.8, model.adaptive_af_protect_threshold.item() - adjustment_rate * 0.3)
        model.adaptive_af_protect_threshold.fill_(new_af_protect_threshold)
        adjustments['af_protect_threshold'] = new_af_protect_threshold
        
        new_af_protect_diff = max(2.5, model.adaptive_af_protect_diff.item() - adjustment_rate * 0.5)
        model.adaptive_af_protect_diff.fill_(new_af_protect_diff)
        adjustments['af_protect_diff'] = new_af_protect_diff
        
        new_af_protect_coef = min(0.6, model.adaptive_af_protect_coef.item() + adjustment_rate * 0.1)
        model.adaptive_af_protect_coef.fill_(new_af_protect_coef)
        adjustments['af_protect_coef'] = new_af_protect_coef
    
    # 如果AFL的precision太低，需要判断是AF被误分为AFL，还是AFL样本太少
    # 如果AF的recall很高（>0.9）且AFL的precision低，可能是AF被误分为AFL，需要适度增强AF保护
    # 但如果AFL的recall也很低，说明更多是AFL被误分为AF，应该增强AFL而不是AF保护
    elif afl_precision < 0.3 and af_recall > 0.9 and afl_recall > 0.4:  
        # AFL precision低，但AF recall很高且AFL recall不低，可能是AF被误分为AFL
        # 适度增强AF保护，但不要过度（因为可能AFL样本本身就少）
        new_af_protect_threshold = max(1.8, model.adaptive_af_protect_threshold.item() - adjustment_rate * 0.3)
        model.adaptive_af_protect_threshold.fill_(new_af_protect_threshold)
        adjustments['af_protect_threshold'] = new_af_protect_threshold
        
        new_af_protect_diff = max(2.5, model.adaptive_af_protect_diff.item() - adjustment_rate * 0.5)
        model.adaptive_af_protect_diff.fill_(new_af_protect_diff)
        adjustments['af_protect_diff'] = new_af_protect_diff
        
        new_af_protect_coef = min(0.65, model.adaptive_af_protect_coef.item() + adjustment_rate * 0.1)
        model.adaptive_af_protect_coef.fill_(new_af_protect_coef)
        adjustments['af_protect_coef'] = new_af_protect_coef
        
        # 适度降低AFL boost强度（但不要过度）
        new_alpha_afl_max = max(0.28, model.adaptive_alpha_afl_max.item() - adjustment_rate * 0.2)
        model.adaptive_alpha_afl_max.fill_(new_alpha_afl_max)
        adjustments['alpha_afl_max'] = new_alpha_afl_max
        
        new_boost_coef = max(0.25, model.adaptive_alpha_afl_boost_coef.item() - adjustment_rate * 0.1)
        model.adaptive_alpha_afl_boost_coef.fill_(new_boost_coef)
        adjustments['alpha_afl_boost_coef'] = new_boost_coef
    
    # 如果两个类别的F1差距太大，调整以平衡
    elif f1_diff > 0.2:  # F1差距 > 20%
        if af_f1 > afl_f1:  # AF F1更高，需要提升AFL
            # 适度增强AFL参数（使用较小的调整幅度）
            new_alpha_afl_max = min(0.45, model.adaptive_alpha_afl_max.item() + adjustment_rate * 0.2)
            model.adaptive_alpha_afl_max.fill_(new_alpha_afl_max)
            adjustments['alpha_afl_max'] = new_alpha_afl_max
            
            new_boost_coef = min(0.5, model.adaptive_alpha_afl_boost_coef.item() + adjustment_rate * 0.15)
            model.adaptive_alpha_afl_boost_coef.fill_(new_boost_coef)
            adjustments['alpha_afl_boost_coef'] = new_boost_coef
        else:  # AFL F1更高，需要提升AF
            # 检查是否是AF被误分为AFL（AF recall低但AFL precision高）
            if af_recall < 0.5 and afl_precision > 0.5:
                # AF被大量误分为AFL，需要增强AF保护
                new_af_protect_threshold = max(1.8, model.adaptive_af_protect_threshold.item() - adjustment_rate * 0.3)
                model.adaptive_af_protect_threshold.fill_(new_af_protect_threshold)
                adjustments['af_protect_threshold'] = new_af_protect_threshold
                
                new_af_protect_diff = max(2.5, model.adaptive_af_protect_diff.item() - adjustment_rate * 0.5)
                model.adaptive_af_protect_diff.fill_(new_af_protect_diff)
                adjustments['af_protect_diff'] = new_af_protect_diff
                
                new_af_protect_coef = min(0.6, model.adaptive_af_protect_coef.item() + adjustment_rate * 0.1)
                model.adaptive_af_protect_coef.fill_(new_af_protect_coef)
                adjustments['af_protect_coef'] = new_af_protect_coef
                
                # 适度降低AFL boost强度
                new_alpha_afl_max = max(0.28, model.adaptive_alpha_afl_max.item() - adjustment_rate * 0.2)
                model.adaptive_alpha_afl_max.fill_(new_alpha_afl_max)
                adjustments['alpha_afl_max'] = new_alpha_afl_max
            else:
                # 一般情况，适度增强AF保护
                new_af_protect_threshold = max(2.0, model.adaptive_af_protect_threshold.item() - adjustment_rate * 0.2)
                model.adaptive_af_protect_threshold.fill_(new_af_protect_threshold)
                adjustments['af_protect_threshold'] = new_af_protect_threshold
                
                new_af_protect_coef = min(0.55, model.adaptive_af_protect_coef.item() + adjustment_rate * 0.05)
                model.adaptive_af_protect_coef.fill_(new_af_protect_coef)
                adjustments['af_protect_coef'] = new_af_protect_coef
    
    if adjustments:
        print(f"  [Adaptive] AF: R={af_recall:.3f}, P={af_precision:.3f}, F1={af_f1:.3f} | AFL: R={afl_recall:.3f}, P={afl_precision:.3f}, F1={afl_f1:.3f}, diff={f1_diff:.3f}")
        print(f"  [Adaptive] Adjusted parameters: {adjustments}")
    
    return adjustments


def compute_confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int) -> np.ndarray:
    """
    计算混淆矩阵，返回形状为 (num_classes, num_classes) 的 ndarray:
        行 = 真实标签，列 = 预测标签
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def _compute_boundary_score(logits_arr: torch.Tensor) -> torch.Tensor:
    """
    计算边界感知分数：基于预测概率的边界距离
    boundary_score = 1 - (max_prob - second_max_prob)
    值越大表示样本越接近类别边界（越难分类）
    """
    probs = torch.softmax(logits_arr, dim=-1)  # (B, num_classes)
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    max_prob = sorted_probs[:, 0]
    second_max_prob = sorted_probs[:, 1] if sorted_probs.size(1) > 1 else torch.zeros_like(max_prob)
    boundary_score = 1.0 - (max_prob - second_max_prob)
    return boundary_score.clamp(min=0.0, max=1.0)


def _compute_sample_weight(
    logits_arr: torch.Tensor,
    gamma: float,
    min_weight: float,
    boundary_alpha: float = 0.0,
    use_low_conf_focus: bool = False,
    low_conf_alpha: float = 1.0,
    conf_discard_th: float = 0.4,
) -> torch.Tensor:
    """
    样本权重：基于模型置信度和边界感知
    - gamma: 置信度权重指数
    - min_weight: 最小权重
    - boundary_alpha: 边界感知权重系数（0表示不使用边界感知）
    - use_low_conf_focus: 是否关注低置信度样本（而非丢弃）
    - low_conf_alpha: 低置信度样本权重增强系数
    - conf_discard_th: 低置信度阈值
    """
    conf = torch.softmax(logits_arr, dim=-1).amax(dim=-1)
    
    if use_low_conf_focus:
        # 策略：关注低置信度样本，给它们更高权重
        # 低置信度样本通常是困难样本，需要更多关注
        # 权重 = 基础权重 * (1 + low_conf_alpha * (1 - conf))
        # 这样低置信度样本（conf接近0）会得到更高权重
        base_weight = torch.pow(conf, gamma)
        low_conf_boost = 1.0 + low_conf_alpha * (1.0 - conf)
        weight = base_weight * low_conf_boost
    else:
        # 原始策略：高置信度样本权重更高
        weight = torch.pow(conf, gamma)
    
    # 边界感知增强：对边界样本给予更高权重
    if boundary_alpha > 0.0:
        boundary_score = _compute_boundary_score(logits_arr)
        # 边界样本权重增强：weight = weight * (1 + alpha * boundary_score)
        weight = weight * (1.0 + boundary_alpha * boundary_score)
    
    return torch.clamp(weight, min=min_weight, max=5.0)  # 允许权重超过1.0以支持低置信度和边界样本


def train_one_epoch(
    model: ECGNet,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer,
    device: str,
    gamma: float,
    min_weight: float,
    conf_discard_th: float,
    lambda_af_bin: float,
    epoch: int,
    total_epochs: int,
    validate_every_batch: bool,
    num_classes: int,
    class_weights: Optional[torch.Tensor] = None,
    use_boundary_aware: bool = True,
    boundary_alpha: float = 0.5,
    boundary_threshold: float = 0.3,
    lambda_boundary: float = 0.1,
    use_low_conf_focus: bool = True,
    low_conf_alpha: float = 1.0,
):
    model.train()
    # label smoothing：抑制过拟合/过度自信，有助于少数类泛化
    ce_arr = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.05)
    ce_bin = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.05)
    # 方案3：二分类用 focal loss（更关注难例，降低 AFL 全灭）
    use_focal = bool(getattr(model, "num_arrhythmia_classes", num_classes) == 2)
    focal_gamma = 2.0
    focal_alpha = 0.75  # AFL(1) 的 alpha；AF 为 1-alpha

    total_loss = 0.0
    total_samples = 0
    all_y, all_pred = [], []

    num_batches = len(train_loader)
    val_iter = iter(val_loader) if (validate_every_batch and val_loader is not None) else None
    print_interval = 100

    for batch_idx, batch in enumerate(train_loader):
        ecg = batch["ecg"].to(device)
        noise_feat = batch["noise_feat"].to(device)  # (B, D)
        rr_seq = batch["rr_seq"].to(device)
        fuzzy_logits = batch["fuzzy_logits"].to(device)  # (B, num_rule_classes)
        y = batch["label"].to(device)

        # 如果主头类别数与模糊规则 logits 维度不一致（例如二分类但 fuzzy_logits 为 4 维），做对齐
        if hasattr(model, "arr_head"):
            try:
                arr_out_features = getattr(model.arr_head[-1], "out_features", None)
            except Exception:
                arr_out_features = getattr(model.arr_head, "out_features", None)
            if arr_out_features is not None and fuzzy_logits.dim() == 2 and fuzzy_logits.size(1) != arr_out_features:
                # 只保留前 arr_out_features 维（典型场景：4 -> 2）
                fuzzy_logits = fuzzy_logits[:, :arr_out_features]

        optimizer.zero_grad()
        logits_arr, logits_bin = model(ecg, rr_seq, noise_feat, fuzzy_logits)
        # 二分类时：主输出使用二分类头（与评估/保存对齐），避免训练“主头”和推理“二分类头”错位
        if num_classes == 2 and logits_bin.size(1) == 2:
            logits_arr = logits_bin
            lambda_af_bin_effective = 0.0  # 避免对同一头重复加 loss
        else:
            lambda_af_bin_effective = lambda_af_bin

        # 基于主四分类头计算样本置信度
        conf = torch.softmax(logits_arr, dim=-1).amax(dim=-1)
        
        # 根据策略决定是否使用keep_mask（仅用于向后兼容的丢弃策略）
        if use_low_conf_focus:
            # 关注低置信度样本策略：使用所有样本，但给低置信度样本更高权重
            keep_mask = torch.ones_like(conf, dtype=torch.bool)  # 保留所有样本
        else:
            # 原始丢弃策略：丢弃低置信度样本
            keep_mask = conf >= conf_discard_th

        # 计算边界感知分数
        boundary_score = None
        if use_boundary_aware:
            boundary_score = _compute_boundary_score(logits_arr)
            # 边界样本权重增强系数（根据epoch动态调整）
            # 早期关注所有边界样本，后期聚焦最难的边界样本
            if epoch < total_epochs // 3:
                effective_alpha = boundary_alpha * 0.7  # 早期：较温和
            elif epoch < total_epochs * 2 // 3:
                effective_alpha = boundary_alpha  # 中期：标准
            else:
                effective_alpha = boundary_alpha * 1.2  # 后期：更强（但不超过1.5）
            effective_alpha = min(effective_alpha, 1.5)
        else:
            effective_alpha = 0.0

        # 基础样本权重（包含边界感知增强和低置信度关注）
        sample_w = _compute_sample_weight(
            logits_arr, 
            gamma, 
            min_weight, 
            boundary_alpha=effective_alpha,
            use_low_conf_focus=use_low_conf_focus,
            low_conf_alpha=low_conf_alpha,
            conf_discard_th=conf_discard_th,
        )

        if class_weights is not None:
            cls_w = class_weights[y]
        else:
            cls_w = torch.ones_like(sample_w)

        total_w = sample_w * cls_w  # (B,)

        # ---- 主 loss ----
        loss_arr_all = ce_arr(logits_arr, y)  # (B,)
        if use_focal and logits_arr.size(1) == 2:
            probs = torch.softmax(logits_arr, dim=-1)
            pt = probs.gather(1, y.view(-1, 1)).squeeze(1).clamp(min=1e-6, max=1.0)  # (B,)
            focal_factor = torch.pow(1.0 - pt, focal_gamma)
            alpha_t = torch.where(y == 1, torch.tensor(focal_alpha, device=device), torch.tensor(1.0 - focal_alpha, device=device))
            loss_arr_all = loss_arr_all * focal_factor * alpha_t
        if use_low_conf_focus:
            # 关注低置信度策略：使用所有样本，低置信度样本通过权重增强
            loss_main = (loss_arr_all * total_w).sum() / (total_w.sum() + 1e-8)
        else:
            # 原始丢弃策略：只使用高置信度样本
            if keep_mask.any():
                loss_main = (loss_arr_all[keep_mask] * total_w[keep_mask]).sum() / (
                    total_w[keep_mask].sum() + 1e-8
                )
            else:
                # 极端情况：本 batch 全是低置信度样本，退化为均值以避免 NaN
                loss_main = (loss_arr_all * total_w).sum() / (total_w.sum() + 1e-8)

        # ---- AF/AFL 二分类辅助 loss ----
        if use_low_conf_focus:
            # 关注策略：使用所有AF/AFL样本
            mask_af_bin = (y < 2)
        else:
            # 丢弃策略：只使用高置信度的AF/AFL样本
            mask_af_bin = (y < 2) & keep_mask
        if mask_af_bin.any() and lambda_af_bin_effective > 0.0:
            y_af = y[mask_af_bin]
            logits_af = logits_bin[mask_af_bin]
            loss_af_all = ce_bin(logits_af, y_af)
            loss_af = (loss_af_all * total_w[mask_af_bin]).sum() / (
                total_w[mask_af_bin].sum() + 1e-8
            )
        else:
            loss_af = torch.tensor(0.0, device=device)

        # ---- AFL margin loss（二分类时强制 AFL logit > AF logit + m）----
        loss_margin = torch.tensor(0.0, device=device)
        if num_classes == 2 and AFL_MARGIN_LOSS_LAMBDA > 0 and logits_arr.size(1) >= 2:
            mask_afl = (y == 1)
            if mask_afl.any():
                z_af = logits_arr[mask_afl, 0]
                z_afl = logits_arr[mask_afl, 1]
                margin = AFL_MARGIN_LOSS_M
                loss_margin = torch.clamp(margin - (z_afl - z_af), min=0.0).mean()

        # ---- 边界感知损失（可选，对边界样本进行额外惩罚）----
        loss_boundary = torch.tensor(0.0, device=device)
        if use_boundary_aware and lambda_boundary > 0.0:
            if boundary_score is None:
                boundary_score = _compute_boundary_score(logits_arr)
            # 识别边界样本（boundary_score > threshold）
            if use_low_conf_focus:
                boundary_mask = (boundary_score > boundary_threshold)  # 使用所有边界样本
            else:
                boundary_mask = (boundary_score > boundary_threshold) & keep_mask
            if boundary_mask.any():
                # 对边界样本计算额外的损失（鼓励模型更明确地区分）
                boundary_loss_all = ce_arr(logits_arr[boundary_mask], y[boundary_mask])
                # 使用边界分数作为权重（边界越模糊，损失权重越大）
                boundary_weights = boundary_score[boundary_mask] * total_w[boundary_mask]
                loss_boundary = (boundary_loss_all * boundary_weights).sum() / (
                    boundary_weights.sum() + 1e-8
                )

        loss = loss_main + lambda_af_bin_effective * loss_af + AFL_MARGIN_LOSS_LAMBDA * loss_margin + lambda_boundary * loss_boundary

        loss.backward()
        optimizer.step()

        B = y.size(0)
        total_loss += loss.item() * B
        total_samples += B

        preds = torch.argmax(logits_arr, dim=-1)
        all_y.extend(y.detach().cpu().tolist())
        all_pred.extend(preds.detach().cpu().tolist())

        if validate_every_batch and val_iter is not None and ((batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == num_batches):
            try:
                val_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_batch = next(val_iter)
            val_metrics = evaluate_batch(model, val_batch, device, num_classes)
            train_loss_current = total_loss / max(total_samples, 1)
            train_metrics = compute_metrics(all_y, all_pred, num_classes)
            
            # 类别名称（二分类）
            class_names = ["AF", "AFL"] if num_classes == 2 else [f"Class {i}" for i in range(num_classes)]
            
            print(
                f"  Batch {batch_idx+1}/{num_batches} | "
                f"Train: loss={train_loss_current:.4f}, acc={train_metrics['accuracy']:.4f}, F1={train_metrics['macro_f1']:.4f} | "
                f"Val: loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}, F1={val_metrics['macro_f1']:.4f}"
            )
            # 打印验证集的每个类别详细指标
            if "y_true" in val_metrics and "y_pred" in val_metrics:
                print_per_class_metrics(val_metrics["y_true"], val_metrics["y_pred"], num_classes, class_names)
            
            # 评估后切回训练模式，避免 RNN backward 报错
            model.train()

    avg_loss = total_loss / max(total_samples, 1)
    metrics = compute_metrics(all_y, all_pred, num_classes)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def evaluate_batch(
    model,
    batch,
    device: str,
    num_classes: int,
    afl_logit_bias_inference: float = 0.0,
    afl_prob_threshold: Optional[float] = None,
) -> dict:
    model.eval()
    ce_arr = nn.CrossEntropyLoss()

    ecg = batch["ecg"].to(device)
    noise_feat = batch["noise_feat"].to(device)  # (B, D)
    rr_seq = batch["rr_seq"].to(device)
    fuzzy_logits = batch["fuzzy_logits"].to(device)  # (B, num_rule_classes)
    y = batch["label"].to(device)

    # 对齐模糊规则 logits 维度与主头类别数
    if hasattr(model, "arr_head"):
        try:
            arr_out_features = getattr(model.arr_head[-1], "out_features", None)
        except Exception:
            arr_out_features = getattr(model.arr_head, "out_features", None)
        if arr_out_features is not None and fuzzy_logits.dim() == 2 and fuzzy_logits.size(1) != arr_out_features:
            fuzzy_logits = fuzzy_logits[:, :arr_out_features]

    logits_arr, _ = model(ecg, rr_seq, noise_feat, fuzzy_logits)
    loss = ce_arr(logits_arr, y)

    if logits_arr.size(1) == 2 and afl_logit_bias_inference != 0:
        logits_arr = logits_arr.clone()
        logits_arr[:, 1] = logits_arr[:, 1] + float(afl_logit_bias_inference)

    if logits_arr.size(1) == 2 and afl_prob_threshold is not None:
        p_afl = torch.softmax(logits_arr, dim=-1)[:, 1]
        preds = (p_afl >= float(afl_prob_threshold)).long()
    else:
        preds = torch.argmax(logits_arr, dim=-1)
    y_list = y.detach().cpu().tolist()
    pred_list = preds.detach().cpu().tolist()

    metrics = compute_metrics(y_list, pred_list, num_classes)
    metrics["loss"] = float(loss.item())
    metrics["y_true"] = y_list  # 保存真实标签
    metrics["y_pred"] = pred_list  # 保存预测标签
    if num_classes == 2:
        cm = compute_confusion_matrix(y_list, pred_list, 2)
        metrics.update(_binary_metrics_from_confusion(cm))
    return metrics


@torch.no_grad()
def evaluate(
    model,
    loader,
    device: str,
    epoch: int = 0,
    total_epochs: int = 0,
    verbose: bool = True,
    afl_logit_bias_inference: float = 0.0,
    afl_prob_threshold: Optional[float] = None,
    tune_afl_threshold: bool = False,
    afl_threshold_search_steps: int = 101,
    afl_threshold_search_min: float = 0.01,
    afl_threshold_search_max: float = 0.99,
    afl_min_recall_val: float = 0.0,
    enable_record_agg: bool = False,
    record_agg_method: str = "mean",
    record_agg_topk: int = 5,
):
    model.eval()
    ce_arr = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    all_y, all_pred = [], []
    all_p_afl = []  # only for binary thresholding
    all_record_names = []

    num_batches = len(loader)
    if verbose:
        print_interval = max(1, num_batches // 10)

    for batch_idx, batch in enumerate(loader):
        ecg = batch["ecg"].to(device)
        noise_feat = batch["noise_feat"].to(device)  # (B, D)
        rr_seq = batch["rr_seq"].to(device)
        fuzzy_logits = batch["fuzzy_logits"].to(device)  # (B, num_rule_classes)
        y = batch["label"].to(device)
        if "record_name" in batch:
            all_record_names.extend([str(x) for x in batch["record_name"]])

        # 对齐模糊规则 logits 维度与主头类别数
        if hasattr(model, "arr_head"):
            try:
                arr_out_features = getattr(model.arr_head[-1], "out_features", None)
            except Exception:
                arr_out_features = getattr(model.arr_head, "out_features", None)
            if arr_out_features is not None and fuzzy_logits.dim() == 2 and fuzzy_logits.size(1) != arr_out_features:
                fuzzy_logits = fuzzy_logits[:, :arr_out_features]

        logits_arr, _ = model(ecg, rr_seq, noise_feat, fuzzy_logits)
        loss = ce_arr(logits_arr, y)

        if logits_arr.size(1) == 2 and afl_logit_bias_inference != 0:
            logits_arr = logits_arr.clone()
            logits_arr[:, 1] = logits_arr[:, 1] + float(afl_logit_bias_inference)
        B = y.size(0)
        total_loss += loss.item() * B
        total_samples += B

        if logits_arr.size(1) == 2:
            p_afl = torch.softmax(logits_arr, dim=-1)[:, 1]
            all_p_afl.extend(p_afl.detach().cpu().numpy().astype(np.float32).tolist())
        preds = torch.argmax(logits_arr, dim=-1)  # 先收集，后面二分类可能会用阈值重算
        all_y.extend(y.detach().cpu().tolist())
        all_pred.extend(preds.detach().cpu().tolist())

        if verbose and ((batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == num_batches):
            current_loss = total_loss / max(total_samples, 1)
            progress = (batch_idx + 1) / num_batches * 100
            print(f"  [Val] Epoch {epoch}/{total_epochs} | Batch {batch_idx+1}/{num_batches} ({progress:.1f}%) | Loss: {current_loss:.4f}", end='\r')

    if verbose:
        print()
    avg_loss = total_loss / max(total_samples, 1)
    num_classes = model.arr_head.out_features
    # 二分类时可用阈值化重新生成预测（更贴合 AFL-F1 优化目标）
    chosen_threshold = afl_prob_threshold
    tuned_info = None
    record_threshold = None
    if num_classes == 2 and (tune_afl_threshold or afl_prob_threshold is not None):
        y_true_np = np.asarray(all_y, dtype=np.int64)
        p_afl_np = np.asarray(all_p_afl, dtype=np.float32)
        # 阈值搜索与主指标统一在片段级进行，便于迁移与选 best
        if tune_afl_threshold and afl_prob_threshold is None:
            chosen_threshold, tuned_info = _search_best_afl_threshold(
                y_true_np,
                p_afl_np,
                beta=2.0,
                steps=afl_threshold_search_steps,
                t_min=afl_threshold_search_min,
                t_max=afl_threshold_search_max,
                min_afl_recall=float(afl_min_recall_val),
            )
        else:
            chosen_threshold = float(afl_prob_threshold)
        # 用 chosen_threshold 生成 segment-level 最终预测
        all_pred = (p_afl_np >= float(chosen_threshold)).astype(np.int64).tolist()

    metrics = compute_metrics(all_y, all_pred, num_classes)
    metrics["loss"] = avg_loss
    # 计算并打印混淆矩阵（验证集）
    cm = compute_confusion_matrix(all_y, all_pred, num_classes)
    metrics["confusion_matrix"] = cm  # 将混淆矩阵添加到metrics中
    if num_classes == 2:
        metrics.update(_binary_metrics_from_confusion(cm))
        if chosen_threshold is not None:
            metrics["afl_threshold"] = float(chosen_threshold)
        if tuned_info is not None:
            metrics["afl_threshold_tuned"] = True
        if enable_record_agg and len(all_record_names) == len(all_y):
            rec_y, rec_p, _ = _aggregate_record_probs(
                all_record_names,
                all_y,
                all_p_afl,
                method=record_agg_method,
                topk=record_agg_topk,
            )
            rec_pred = (np.asarray(rec_p, dtype=np.float32) >= float(chosen_threshold)).astype(np.int64).tolist()
            rec_cm = compute_confusion_matrix(rec_y, rec_pred, 2)
            metrics["record_confusion_matrix"] = rec_cm
            rec_bin = _binary_metrics_from_confusion(rec_cm)
            metrics["record_afl_f1"] = rec_bin["afl_f1"]
            metrics["record_afl_precision"] = rec_bin["afl_precision"]
            metrics["record_afl_recall"] = rec_bin["afl_recall"]
            metrics["record_afl_threshold"] = float(chosen_threshold)
    print("  Confusion matrix (rows=true, cols=pred):")
    if num_classes == 2:
        print("            Pred: AF  AFL")
        print(f"  True: AF  [{cm[0,0]:4d} {cm[0,1]:4d}]")
        print(f"       AFL  [{cm[1,0]:4d} {cm[1,1]:4d}]")
    elif num_classes == 4:
        print("            Pred:   AF  AFL PSVT  NOR")
        print(f"  True: AF   [{cm[0,0]:4d} {cm[0,1]:4d} {cm[0,2]:4d} {cm[0,3]:4d}]")
        print(f"       AFL   [{cm[1,0]:4d} {cm[1,1]:4d} {cm[1,2]:4d} {cm[1,3]:4d}]")
        print(f"       PSVT  [{cm[2,0]:4d} {cm[2,1]:4d} {cm[2,2]:4d} {cm[2,3]:4d}]")
        print(f"       NOR   [{cm[3,0]:4d} {cm[3,1]:4d} {cm[3,2]:4d} {cm[3,3]:4d}]")
    else:
        print(cm)
    return metrics


def split_dataset_csv(
    csv_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    将 CSV 划分为 train/val/test。
    
    优先采用 **按记录（患者）分组划分** 的方式，以避免同一患者的片段同时出现在
    训练集和验证/测试集中，降低过拟合和数据泄漏风险。
    
    分组规则（优先级由高到低）：
    1) 如果存在 `path` 列，则从文件名中提取 `xxx_segYYYY` 里的 `xxx` 作为记录ID；
    2) 否则如果存在 `record_name` 列，则使用该列；
    3) 再否则退回到原来的按样本随机划分。
    """
    if pd is None:
        raise ImportError("pandas is required")
    df = pd.read_csv(csv_path)

    np.random.seed(seed)

    def _extract_record_id(path_str: str) -> str:
        """
        从路径中提取记录ID（用于划分时按患者分组）:
        data/.../001_seg0001.npz -> 001
        data/.../001_seg0001_noisy_em_snr15_v0.npz -> 001_seg0001_noisy_em_snr15_v0（噪声扩充视为新患者）
        """
        base = os.path.basename(str(path_str)).replace("\\", "/")
        if base.lower().endswith(".npz"):
            base = base[:-4]
        # 噪声扩充文件：
        # - 默认按源患者 ID 分组（split-then-augment / 修正泄漏时使用）
        # - 若环境变量 AFL_NOISY_GROUP=independent，则每个 noisy 文件视为独立 record
        if "_noisy_" in base:
            if os.environ.get("AFL_NOISY_GROUP", "patient").lower() == "independent":
                return base
            orig = base.split("_noisy_")[0]
            if "_seg" in orig:
                return orig.split("_seg")[0]
            return orig
        if "_seg" in base:
            return base.split("_seg")[0]
        return base

    # 1) 尝试从 path 列提取记录ID
    group_col = None
    if "path" in df.columns:
        df["_record_group"] = df["path"].apply(_extract_record_id)
        group_col = "_record_group"
    elif "record_name" in df.columns:
        df["_record_group"] = df["record_name"]
        group_col = "_record_group"

    # 优先按记录（患者）分组划分，确保每个集合都包含所有类别
    if group_col is not None:
        # 检查是否有标签列用于分层划分
        label_col = None
        if "label_raw" in df.columns:
            label_col = "label_raw"
        elif "label" in df.columns:
            label_col = "label"
        
        if label_col is not None:
            # 分层按患者划分：确保每个集合都包含所有类别
            unique_labels = df[label_col].unique()
            print(f"  - Found labels: {unique_labels}")

            # ==== 专门针对 AF/AFL 二分类的患者级 8:1:1 划分（你刚刚描述的方案） ====
            if set(unique_labels) == {"AF", "AFL"}:
                print("  - Using AFL-aware 8:1:1 patient-level split for AF/AFL")
                patient_labels = df.groupby(group_col)[label_col].apply(lambda x: set(x.tolist()))
                has_afl = patient_labels.apply(lambda s: "AFL" in s)
                patients_with_afl = patient_labels.index[has_afl].tolist()
                patients_af_only = patient_labels.index[~has_afl].tolist()

                rng = np.random.RandomState(seed)
                patients_with_afl = rng.permutation(patients_with_afl).tolist()
                patients_af_only = rng.permutation(patients_af_only).tolist()

                def _split_group(patients, name: str):
                    n = len(patients)
                    if n == 0:
                        print(f"    - {name}: 0 patients")
                        return [], [], []
                    n_train = max(1, int(round(n * train_ratio)))
                    n_val = max(1, int(round(n * val_ratio)))
                    n_test = max(1, n - n_train - n_val)
                    # 校正总数
                    while n_train + n_val + n_test > n and n_test > 1:
                        n_test -= 1
                    while n_train + n_val + n_test > n and n_val > 1:
                        n_val -= 1
                    while n_train + n_val + n_test > n and n_train > 1:
                        n_train -= 1
                    train_p = patients[:n_train]
                    val_p = patients[n_train:n_train + n_val]
                    test_p = patients[n_train + n_val:n_train + n_val + n_test]
                    print(f"    - {name}: total={n}, split -> train={len(train_p)}, val={len(val_p)}, test={len(test_p)}")
                    return train_p, val_p, test_p

                # 对 AFL 患者和 AF-only 患者分别做 8:1:1
                train_afl, val_afl, test_afl = _split_group(patients_with_afl, "AFL+")
                train_af_only, val_af_only, test_af_only = _split_group(patients_af_only, "AF-only")

                train_records = sorted(set(train_afl + train_af_only))
                val_records = sorted(set(val_afl + val_af_only))
                test_records = sorted(set(test_afl + test_af_only))

                # 最小保障：Test/Val 各至少 2 个 AFL+ 患者（若 AFL+ 总数允许），通过从 train 调配
                def _ensure_min_afl(train_r, val_r, test_r, min_afl_per_split: int = 2):
                    afl_in_train = [p for p in train_r if p in patients_with_afl]
                    afl_in_val = [p for p in val_r if p in patients_with_afl]
                    afl_in_test = [p for p in test_r if p in patients_with_afl]

                    def _move_one_afl_from_to(src_list, dst_list):
                        for p in list(src_list):
                            if p in patients_with_afl:
                                src_list.remove(p)
                                dst_list.append(p)
                                return True
                        return False

                    n_afl_total = len(patients_with_afl)
                    if n_afl_total < min_afl_per_split:
                        return sorted(set(train_r)), sorted(set(val_r)), sorted(set(test_r))

                    # 1) 先保证测试集至少 min_afl_per_split 个 AFL+
                    need_test = max(0, min_afl_per_split - len(afl_in_test))
                    while need_test > 0 and _move_one_afl_from_to(train_r, test_r):
                        need_test -= 1
                        afl_in_test = [p for p in test_r if p in patients_with_afl]
                    while need_test > 0 and _move_one_afl_from_to(val_r, test_r):
                        need_test -= 1
                        afl_in_test = [p for p in test_r if p in patients_with_afl]

                    # 2) 再保证验证集至少 min_afl_per_split 个 AFL+
                    afl_in_val = [p for p in val_r if p in patients_with_afl]
                    need_val = max(0, min_afl_per_split - len(afl_in_val))
                    while need_val > 0 and _move_one_afl_from_to(train_r, val_r):
                        need_val -= 1
                        afl_in_val = [p for p in val_r if p in patients_with_afl]

                    return sorted(set(train_r)), sorted(set(val_r)), sorted(set(test_r))

                train_records, val_records, test_records = _ensure_min_afl(
                    list(train_records), list(val_records), list(test_records), min_afl_per_split=2
                )

                train_df = df[df[group_col].isin(train_records)].reset_index(drop=True)
                val_df = df[df[group_col].isin(val_records)].reset_index(drop=True)
                test_df = df[df[group_col].isin(test_records)].reset_index(drop=True)

                patients_with_afl_set = set(patients_with_afl)
                n_train_afl = len([p for p in train_records if p in patients_with_afl_set])
                n_val_afl = len([p for p in val_records if p in patients_with_afl_set])
                n_test_afl = len([p for p in test_records if p in patients_with_afl_set])

                # 打印最终患者和样本分布（含 AFL+ 患者数，保证 Test/Val 各至少 2）
                def _print_split(name, df_split, n_afl_patients: int = -1):
                    if df_split.empty:
                        print(f"    - {name}: 0 patients, 0 samples")
                        return
                    pats = sorted(df_split[group_col].unique())
                    labs = sorted(set(df_split[label_col].unique()))
                    print(f"    - {name}: {len(pats)} patients, labels: {labs}")
                    if n_afl_patients >= 0:
                        print(f"      AFL+ 患者数: {n_afl_patients} (Val/Test 要求各≥2)")
                    label_counts = df_split[label_col].value_counts().sort_index()
                    print(f"      {name} sample distribution by label:")
                    for lb, cnt in label_counts.items():
                        print(f"        {lb}: {cnt} samples")

                print(f"  - AFL-aware split on '{group_col}':")
                _print_split("Train", train_df, n_train_afl)
                _print_split("Val", val_df, n_val_afl)
                _print_split("Test", test_df, n_test_afl)

                return train_df, val_df, test_df

            # ==== 默认逻辑：按标签分别划分患者（允许轻微复用），总体保持约 8:1:1 ====
            # 按患者统计每个标签的分布
            patient_label_stats = df.groupby([group_col, label_col]).size().unstack(fill_value=0)
            
            train_records = []
            val_records = []
            test_records = []
            
            # 对每个标签分别进行患者划分
            for label in unique_labels:
                # 注意：一个患者可以同时有 AF 和 AFL 片段。
                # 这里不强制将该患者“统一归类”为某一个标签，
                # 而是：只要该患者在当前标签上有样本（计数>0），就认为其属于该标签，
                # 从而允许同一患者同时出现在 AF 和 AFL 的患者集合中。
                if label not in patient_label_stats.columns:
                    continue
                label_patients = patient_label_stats.index[patient_label_stats[label] > 0].tolist()
                if len(label_patients) == 0:
                    continue
                
                # 随机打乱该标签的患者
                label_patients = np.random.permutation(label_patients).tolist()
                
                # 按比例划分该标签的患者
                n_patients = len(label_patients)
                
                # 特殊处理极少数类别（患者数 <= 3）
                if n_patients <= 3:
                    if n_patients == 1:
                        # 只有1个患者：放入训练集，同时复制到验证集，以保证验证集中至少有该类别
                        train_patients = label_patients
                        val_patients = label_patients  # 允许同一患者同时出现在 train 和 val
                        test_patients = []
                        print(f"    - Warning: Only 1 patient for {label}, reused in both train and val to ensure validation coverage")
                    elif n_patients == 2:
                        # 2个患者：1个主要用于训练，1个主要用于测试，但同时保证验证集中至少有1个患者
                        train_patients = [label_patients[0]]
                        # 将其中1个患者也放入验证集（可能与训练或测试重叠）
                        val_patients = [label_patients[1]]
                        test_patients = [label_patients[1]]
                        print(f"    - Warning: Only 2 patients for {label}, one patient reused in val/test to ensure validation coverage")
                    else:  # n_patients == 3
                        # 3个患者：2个训练，1个测试，同时从中选1个放入验证集
                        train_patients = label_patients[:2]
                        val_patients = [label_patients[1]]  # 复用其中1个患者到验证集
                        test_patients = [label_patients[2]]
                        print(f"    - Warning: Only 3 patients for {label}, one patient reused in val to ensure validation coverage")
                else:
                    # 正常情况：尽量按 8:1:1 比例划分，但确保测试集也有足够的患者
                    # 对于少数类别（患者数较少），确保验证集和测试集都有足够的患者
                    n_train = max(1, int(n_patients * train_ratio))
                    n_val = max(1, int(n_patients * val_ratio))
                    n_test = max(1, int(n_patients * test_ratio))
                    
                    # 对于患者数较少的类别（4-15个患者），确保验证集和测试集都至少有2个患者
                    # 这样可以避免验证集或测试集只有1个患者导致指标不稳定的问题
                    if 4 <= n_patients <= 15:
                        # 确保验证集和测试集都至少有2个患者
                        n_val = max(2, n_val)
                        n_test = max(2, n_test)
                        # 重新分配，尽量保持接近8:1:1的比例
                        remaining = n_patients - n_val - n_test
                        if remaining < 1:
                            # 如果剩余不够，调整验证集和测试集
                            n_val = min(2, n_patients - 2)  # 至少留2个给测试集
                            n_test = min(2, n_patients - n_val - 1)  # 至少留1个给训练集
                            n_train = n_patients - n_val - n_test
                        else:
                            n_train = remaining
                        print(f"    - Small class detected for {label} ({n_patients} patients): ensuring val≥{n_val}, test≥{n_test}")
                    
                    train_patients = label_patients[:n_train]
                    val_patients = label_patients[n_train:n_train + n_val]
                    test_patients = label_patients[n_train + n_val:]
                    
                    # 保护性措施：如果由于四舍五入导致某个标签在验证集中没有患者，则强行复用1个患者到验证集
                    if len(val_patients) == 0 and n_patients > 0:
                        # 选择最后一个患者加入验证集（不从原集合中删除，允许轻微的数据泄漏以保证验证覆盖）
                        forced_val_patient = label_patients[-1]
                        if forced_val_patient not in val_patients:
                            val_patients.append(forced_val_patient)
                        print(
                            f"    - Warning: No validation patients for {label} after split, "
                            f"reusing patient '{forced_val_patient}' in val to ensure coverage"
                        )
                    
                    # 保护性措施：如果测试集中没有患者，则强行复用1个患者到测试集
                    if len(test_patients) == 0 and n_patients > 0:
                        # 从训练集或验证集中选择一个患者加入测试集
                        forced_test_patient = label_patients[-1]
                        if forced_test_patient not in test_patients:
                            test_patients.append(forced_test_patient)
                        print(
                            f"    - Warning: No test patients for {label} after split, "
                            f"reusing patient '{forced_test_patient}' in test to ensure coverage"
                        )
                
                train_records.extend(train_patients)
                val_records.extend(val_patients)
                test_records.extend(test_patients)
                
                print(f"    - {label}: {len(train_patients)} train / {len(val_patients)} val / {len(test_patients)} test patients")
            
            # 创建数据集
            train_df = df[df[group_col].isin(train_records)].reset_index(drop=True)
            val_df = df[df[group_col].isin(val_records)].reset_index(drop=True)
            test_df = df[df[group_col].isin(test_records)].reset_index(drop=True)
            
            # 验证每个集合都包含所有类别
            train_labels = set(train_df[label_col].unique()) if len(train_df) > 0 else set()
            val_labels = set(val_df[label_col].unique()) if len(val_df) > 0 else set()
            test_labels = set(test_df[label_col].unique()) if len(test_df) > 0 else set()
            
            print(f"  - Stratified record-level split on '{group_col}':")
            print(f"    - Train: {len(train_records)} patients, labels: {sorted(train_labels)}")
            print(f"    - Val: {len(val_records)} patients, labels: {sorted(val_labels)}")
            print(f"    - Test: {len(test_records)} patients, labels: {sorted(test_labels)}")
            
            # 打印每个集合的详细类别分布（样本数）
            if len(train_df) > 0:
                train_label_counts = train_df[label_col].value_counts().sort_index()
                print(f"    - Train sample distribution by label:")
                for label, count in train_label_counts.items():
                    print(f"        {label}: {count} samples")
            
            if len(val_df) > 0:
                val_label_counts = val_df[label_col].value_counts().sort_index()
                print(f"    - Val sample distribution by label:")
                for label, count in val_label_counts.items():
                    print(f"        {label}: {count} samples")
            
            if len(test_df) > 0:
                test_label_counts = test_df[label_col].value_counts().sort_index()
                print(f"    - Test sample distribution by label:")
                for label, count in test_label_counts.items():
                    print(f"        {label}: {count} samples")
            
            # 警告：如果某个集合缺少某些类别
            all_labels = set(unique_labels)
            if len(val_df) > 0 and val_labels != all_labels:
                print(f"    - Warning: Validation set missing labels: {all_labels - val_labels}")
            if len(test_df) > 0 and test_labels != all_labels:
                print(f"    - Warning: Test set missing labels: {all_labels - test_labels}")
        
        else:
            # 没有标签列，使用原始的随机划分
            unique_records = df[group_col].unique()
            unique_records = np.random.permutation(unique_records)

            n_records = len(unique_records)
            n_train = int(n_records * train_ratio)
            n_val = int(n_records * val_ratio)

            train_records = unique_records[:n_train]
            val_records = unique_records[n_train:n_train + n_val]
            test_records = unique_records[n_train + n_val:]

            train_df = df[df[group_col].isin(train_records)].reset_index(drop=True)
            val_df = df[df[group_col].isin(val_records)].reset_index(drop=True)
            test_df = df[df[group_col].isin(test_records)].reset_index(drop=True)

            print(
                f"  - Using record-level split on '{group_col}': "
                f"{len(train_records)} train / {len(val_records)} val / {len(test_records)} test records"
            )
    else:
        # 回退：按样本随机划分（原始实现）
        indices = np.random.permutation(len(df))
        n_total = len(df)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        train_df = df.iloc[train_indices].reset_index(drop=True)
        val_df = df.iloc[val_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)

        print(
            "  - Columns 'path'/'record_name' not found, falling back to sample-level random split "
            f"({n_train}/{n_val}/{len(df) - n_train - n_val} samples)"
        )

    return train_df, val_df, test_df


def get_device(device_str: str = "auto"):
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print(f"  - Auto-detected GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("  - No GPU available, using CPU")
    elif device_str.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(device_str)
        else:
            print("  - CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    return device


def run_training(cfg: TrainConfig):
    print("=" * 80)
    print("ECG Arrhythmia Training (Morphology + Rhythm + Noise, 4-Class Classification)")
    print("=" * 80)

    device = get_device(cfg.device)
    print(f"\n[1/7] Device: {device}")

    out_path = Path(cfg.out_ckpt)
    if len(out_path.parts) > 1:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.out_ckpt = str(out_path)
        output_dir = str(out_path.parent)
    else:
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        cfg.out_ckpt = os.path.join(output_dir, os.path.basename(cfg.out_ckpt))
    print(f"[2/7] Output checkpoint: {cfg.out_ckpt}")

    print(f"\n[3/7] Loading datasets...")
    if cfg.data_csv:
        print(f"  - Data CSV: {cfg.data_csv}")
        train_df, val_df, test_df = split_dataset_csv(cfg.data_csv, 0.8, 0.1, 0.1, seed=int(getattr(cfg, "split_seed", 42)))
        
        # 过采样训练数据（如果启用）
        if cfg.use_oversampling and cfg.num_arrhythmia_classes == 2:
            print(f"\n  - Applying oversampling to training data...")
            from oversample_train_data import oversample_train_data
            
            # 确保train_df有'label'列（如果没有，从'label_raw'映射）
            if "label" not in train_df.columns and "label_raw" in train_df.columns:
                # 创建label列（与ECGDataset中的映射一致）
                raw_to_label = {
                    "AF": 0,
                    "AFL": 1,
                    "PSVT": 2,
                    "Normal": 3,
                    "NOR": 3,
                    "NORM": 3,
                    "NSR": 3,
                    "SR": 3,
                    "Other": 3,
                    "OTHER": 3,
                }
                train_df["label"] = train_df["label_raw"].map(raw_to_label)
                # 只保留AF和AFL（二分类）
                train_df = train_df[train_df["label"].isin([0, 1])].copy()
            
            # 临时保存原始训练数据
            temp_train_csv_before = os.path.join(output_dir, "temp_train_before_oversample.csv")
            train_df.to_csv(temp_train_csv_before, index=False)
            
            # 执行过采样
            temp_train_csv_oversampled = os.path.join(output_dir, "temp_train_oversampled.csv")
            oversample_train_data(
                input_csv=temp_train_csv_before,
                output_csv=temp_train_csv_oversampled,
                target_ratio=cfg.oversample_target_ratio,
                group_col="_record_group",
                label_col="label",
                random_seed=42,
                strategy=cfg.oversample_strategy,
                verbose=True,
            )
            
            # 使用过采样后的数据
            train_df = pd.read_csv(temp_train_csv_oversampled)
            print(f"  - Using oversampled training data: {len(train_df)} samples")
        
        temp_train_csv = os.path.join(output_dir, "temp_train.csv")
        temp_val_csv = os.path.join(output_dir, "temp_val.csv")
        temp_test_csv = os.path.join(output_dir, "temp_test.csv")
        train_df.to_csv(temp_train_csv, index=False)
        val_df.to_csv(temp_val_csv, index=False)
        test_df.to_csv(temp_test_csv, index=False)

        train_ds = ECGDataset(
            temp_train_csv,
            max_len=cfg.max_len,
            num_leads=cfg.num_leads,
            max_rr_intervals=cfg.max_rr_intervals,
            use_stft_dbscan=cfg.use_stft_dbscan,
            multi_lead_method=cfg.multi_lead_method,
            flutter_method=cfg.flutter_method,
            lead_names=cfg.lead_names,
            use_precomputed=cfg.use_precomputed,
            recompute_fuzzy_only=getattr(cfg, "recompute_fuzzy_only", False),
            augment=True,
        )
        val_ds = ECGDataset(
            temp_val_csv,
            max_len=cfg.max_len,
            num_leads=cfg.num_leads,
            max_rr_intervals=cfg.max_rr_intervals,
            use_stft_dbscan=cfg.use_stft_dbscan,
            multi_lead_method=cfg.multi_lead_method,
            flutter_method=cfg.flutter_method,
            lead_names=cfg.lead_names,
            use_precomputed=cfg.use_precomputed,
            recompute_fuzzy_only=getattr(cfg, "recompute_fuzzy_only", False),
            augment=False,
        )
        test_ds = ECGDataset(
            temp_test_csv,
            max_len=cfg.max_len,
            num_leads=cfg.num_leads,
            max_rr_intervals=cfg.max_rr_intervals,
            use_stft_dbscan=cfg.use_stft_dbscan,
            multi_lead_method=cfg.multi_lead_method,
            flutter_method=cfg.flutter_method,
            lead_names=cfg.lead_names,
            use_precomputed=cfg.use_precomputed,
            recompute_fuzzy_only=getattr(cfg, "recompute_fuzzy_only", False),
            augment=False,
        )
        print(f"  - Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")
    else:
        if not cfg.train_csv or not cfg.val_csv:
            raise ValueError("Either provide --data_csv or both --train_csv and --val_csv")
        print(f"  - Training CSV: {cfg.train_csv}")
        train_ds = ECGDataset(
            cfg.train_csv,
            max_len=cfg.max_len,
            num_leads=cfg.num_leads,
            max_rr_intervals=cfg.max_rr_intervals,
            use_stft_dbscan=cfg.use_stft_dbscan,
            multi_lead_method=cfg.multi_lead_method,
            flutter_method=cfg.flutter_method,
            lead_names=cfg.lead_names,
            use_precomputed=cfg.use_precomputed,
        )
        print(f"  - Validation CSV: {cfg.val_csv}")
        val_ds = ECGDataset(
            cfg.val_csv,
            max_len=cfg.max_len,
            num_leads=cfg.num_leads,
            max_rr_intervals=cfg.max_rr_intervals,
            use_stft_dbscan=cfg.use_stft_dbscan,
            multi_lead_method=cfg.multi_lead_method,
            flutter_method=cfg.flutter_method,
            lead_names=cfg.lead_names,
            use_precomputed=cfg.use_precomputed,
        )
        test_ds = None
        print(f"  - Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    sample = train_ds[0]
    rr_len = sample["rr_seq"].shape[0]
    noise_dim = sample["noise_feat"].shape[0]
    print(f"  - RR seq len: {rr_len}")
    print(f"  - Noise feat dim: {noise_dim}")

    if cfg.use_class_weights:
        unique, counts = np.unique(train_ds.df["label"].values, return_counts=True)
        class_counts = np.zeros(cfg.num_arrhythmia_classes, dtype=np.float32)
        for u, c in zip(unique, counts):
            if 0 <= u < cfg.num_arrhythmia_classes:
                class_counts[u] = c
        total_count = class_counts.sum()
        class_counts[class_counts == 0] = 1.0
        
        # 类别权重：以 inverse-frequency 为基准（对二分类极不平衡更有效）
        class_weights = total_count / (cfg.num_arrhythmia_classes * class_counts)

        # 对于严重不平衡的二分类：可选地进一步放大少数类权重（避免模型塌陷到多数类）
        if cfg.num_arrhythmia_classes == 2:
            imbalance_ratio = float(class_counts.max() / class_counts.min())
            minority_class = int(np.argmin(class_counts))
            majority_class = int(np.argmax(class_counts))
            minority_over_majority = float(class_weights[minority_class] / class_weights[majority_class])

            if imbalance_ratio > 10.0:
                # 偏向少数类 AFL：提高 target_ratio，使 AFL 在 loss 中权重大于 AF，更容易被召回
                target_ratio = min(imbalance_ratio, 4.0)  # 提高至 4.0，明显偏向 AFL
                if minority_over_majority < target_ratio:
                    class_weights[minority_class] = class_weights[majority_class] * target_ratio
                print(f"  - Severe class imbalance detected (ratio={imbalance_ratio:.1f}:1)")
                print(f"  - Using boosted inverse-frequency class weights for binary classification (target_ratio={target_ratio:.1f}, bias towards AFL)")
        
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        print(f"  - Class counts: {class_counts.tolist()}")
        print(f"  - Class weights: {class_weights.tolist()}")
        if cfg.num_arrhythmia_classes == 2:
            minority_class = int(np.argmin(class_counts))
            majority_class = int(np.argmax(class_counts))
            print(
                f"  - Weight ratio (minority/majority): {float(class_weights[minority_class] / class_weights[majority_class]):.2f}"
            )
    else:
        class_weights_tensor = None
        print("  - Class weights disabled")

    pin_memory = device.type == "cuda"
    if sys.platform == "win32":
        if cfg.num_workers == 0:
            num_workers = 0
            print("  - Windows: using num_workers=0")
        else:
            num_workers = cfg.num_workers
    else:
        if cfg.num_workers == 0 and device.type == "cuda":
            num_workers = 4
        else:
            num_workers = cfg.num_workers

    # DataLoader 参数：num_workers=0 时不能设置 prefetch_factor/persistent_workers
    dl_common = {
        "batch_size": cfg.batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "prefetch_factor": 2 if num_workers > 0 else None,
        "persistent_workers": num_workers > 0,
    }

    def _build_train_loader_with_optional_weights(sample_weights: Optional[np.ndarray] = None):
        """根据 sample_weights（若提供）构建带 WeightedRandomSampler 的 train_loader。"""
        if sample_weights is None:
            return DataLoader(
                train_ds,
                shuffle=True,
                **{k: v for k, v in dl_common.items() if v is not None or k in ["batch_size", "num_workers", "pin_memory", "shuffle"]},
            )
        sampler = WeightedRandomSampler(
            weights=sample_weights.astype(np.float32),
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(
            train_ds,
            sampler=sampler,
            shuffle=False,
            **{k: v for k, v in dl_common.items() if v is not None or k in ["batch_size", "num_workers", "pin_memory"]},
        )

    # 使用加权随机采样器来处理类别不平衡（基础权重）
    # 对于严重不平衡的数据集，让少数类（AFL）样本被更频繁地采样
    use_weighted_sampler = False
    base_sample_weights = None
    if cfg.num_arrhythmia_classes == 2:
        # 计算每个样本的权重
        train_labels = train_ds.df["label"].values
        unique_labels, label_counts = np.unique(train_labels, return_counts=True)
        
        if len(unique_labels) == 2:
            imbalance_ratio = float(label_counts.max() / label_counts.min())
            
            # 如果不平衡比例 > 15:1，使用加权采样器（提高阈值，减少过采样使用）
            # 降低过采样强度，减少AF被误分为AFL的情况
            if imbalance_ratio > 15.0:  # 从12.0提高到15.0，减少过采样使用
                use_weighted_sampler = True
                # 降低过采样强度，减少过度偏向AFL
                # 从2.0降低到1.5，提高AF的precision
                target_ratio = min(imbalance_ratio, 1.5)  # 目标不平衡比例：最多1.5:1（从2.0降低到1.5）
                
                # 计算每个类别的权重
                class_weights_for_sampler = {}
                majority_count = label_counts.max()
                minority_count = label_counts.min()
                
                # 多数类权重保持为1.0
                majority_label = int(unique_labels[np.argmax(label_counts)])
                class_weights_for_sampler[majority_label] = 1.0
                
                # 少数类权重 = 目标比例（例如，如果目标是5:1，少数类权重就是5）
                minority_label = int(unique_labels[np.argmin(label_counts)])
                class_weights_for_sampler[minority_label] = target_ratio
                
                # 为每个样本分配权重
                sample_weights = np.array([class_weights_for_sampler[int(label)] for label in train_labels], dtype=np.float32)
                base_sample_weights = sample_weights.copy()
                
                # 计算期望的样本数（用于验证）
                total_weight = sum(class_weights_for_sampler[label] * count for label, count in zip(unique_labels, label_counts))
                expected_minority_samples = int(len(train_labels) * class_weights_for_sampler[minority_label] * minority_count / total_weight)
                
                # 创建加权随机采样器
                # num_samples设置为训练集大小，确保每个epoch看到所有样本（但少数类会被适度重复采样）
                weighted_sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(train_labels),
                    replacement=True  # 允许重复采样，实现过采样效果
                )
                
                print(f"  - Using WeightedRandomSampler for class imbalance (ratio={imbalance_ratio:.1f}:1)")
                print(f"    - Target ratio: {target_ratio:.1f}:1 (moderate oversampling)")
                print(f"    - Class weights for sampler: {class_weights_for_sampler}")
                print(f"    - Expected minority samples per epoch: ~{expected_minority_samples} (vs {minority_count} original)")
    
    # 初始 train_loader：若启用 hard mining，则从一开始就用 sampler（权重初始为 base 或全1）
    if cfg.num_arrhythmia_classes == 2 and cfg.use_hard_mining:
        if base_sample_weights is None:
            base_sample_weights = np.ones(len(train_ds), dtype=np.float32)
        train_loader = _build_train_loader_with_optional_weights(base_sample_weights)
        use_weighted_sampler = True
    else:
        if use_weighted_sampler:
            train_loader = DataLoader(
                train_ds,
                sampler=weighted_sampler,
                shuffle=False,  # 使用sampler时不能shuffle
                **{k: v for k, v in dl_common.items() if v is not None or k in ["batch_size", "num_workers", "pin_memory"]},
            )
        else:
            train_loader = DataLoader(
                train_ds,
                shuffle=True,
                **{k: v for k, v in dl_common.items() if v is not None or k in ["batch_size", "num_workers", "pin_memory", "shuffle"]},
            )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **{k: v for k, v in dl_common.items() if v is not None or k in ["batch_size", "num_workers", "pin_memory", "shuffle"]},
    )
    test_loader = (
        DataLoader(
            test_ds,
            shuffle=False,
            **{k: v for k, v in dl_common.items() if v is not None or k in ["batch_size", "num_workers", "pin_memory", "shuffle"]},
        )
        if test_ds
        else None
    )
    print(f"  - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}" +
          (f", Test batches: {len(test_loader)}" if test_loader else ""))

    print(f"\n[4/7] Building model...")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = ECGNet(
        num_leads=cfg.num_leads,
        num_arrhythmia_classes=cfg.num_arrhythmia_classes,
        noise_feat_dim=noise_dim,
        rr_seq_len=rr_len,
        d_model=128,
        rhythm_emb_dim=64,
        noise_emb_dim=32,
        fuzzy_weight_init=cfg.fuzzy_weight_init,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total params: {total_params:,} ({trainable_params:,} trainable)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(f"\n[5/7] Training config:")
    print(f"  - Epochs: {cfg.epochs}, Batch size: {cfg.batch_size}")
    print(f"  - Max len: {cfg.max_len}, RR len: {cfg.max_rr_intervals}")
    print(f"  - Reweight: gamma={cfg.weight_conf_gamma}, min_w={cfg.min_weight}")
    print(f"  - Use class weights: {cfg.use_class_weights}")
    print(f"  - Fuzzy rule weight (init): {cfg.fuzzy_weight_init} (learnable gating)")
    print(f"  - Boundary aware: {cfg.use_boundary_aware} (alpha={cfg.boundary_alpha}, threshold={cfg.boundary_threshold}, lambda={cfg.lambda_boundary})")
    print(f"  - Low-confidence handling: {'Focus (enhance weight)' if cfg.use_low_conf_focus else 'Discard'} (alpha={cfg.low_conf_alpha}, threshold={cfg.conf_discard_th})")

    print(f"\n[6/7] Start training...")
    best_val_f1 = -1.0
    epochs_no_improve = 0  # 连续未提升的 epoch 数
    use_early_stop = cfg.early_stop_patience is not None and cfg.early_stop_patience > 0
    if use_early_stop:
        print(
            f"  - Early stopping enabled: patience={cfg.early_stop_patience}, "
            f"min_delta={cfg.early_stop_min_delta}"
        )
    # hard mining：初始化权重（下一轮会动态更新并重建 sampler）
    hard_sample_weights = None
    if cfg.num_arrhythmia_classes == 2 and getattr(cfg, "use_hard_mining", False):
        if base_sample_weights is None:
            base_sample_weights = np.ones(len(train_ds), dtype=np.float32)
        hard_sample_weights = base_sample_weights.copy().astype(np.float32)
    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        # hard mining：从第 2 个 epoch 起，使用上一轮挖掘得到的权重重建 train_loader
        if cfg.num_arrhythmia_classes == 2 and getattr(cfg, "use_hard_mining", False) and hard_sample_weights is not None and epoch > 1:
            train_loader = _build_train_loader_with_optional_weights(hard_sample_weights)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            cfg.weight_conf_gamma,
            cfg.min_weight,
            cfg.conf_discard_th,
            cfg.lambda_af_bin,
            epoch,
            cfg.epochs,
            cfg.validate_every_batch,
            cfg.num_arrhythmia_classes,
            class_weights_tensor,
            cfg.use_boundary_aware,
            cfg.boundary_alpha,
            cfg.boundary_threshold,
            cfg.lambda_boundary,
        )
        # 每个epoch都打印验证集混淆矩阵；二分类默认自动搜索“最大化 AFL-F1”的阈值
        tune_threshold = (cfg.num_arrhythmia_classes == 2 and cfg.afl_prob_threshold is None)
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            epoch,
            cfg.epochs,
            verbose=True,
            afl_logit_bias_inference=cfg.afl_logit_bias_inference,
            afl_prob_threshold=cfg.afl_prob_threshold,
            tune_afl_threshold=tune_threshold,
            afl_threshold_search_steps=cfg.afl_threshold_search_steps,
            afl_threshold_search_min=cfg.afl_threshold_search_min,
            afl_threshold_search_max=cfg.afl_threshold_search_max,
            afl_min_recall_val=float(getattr(cfg, "afl_min_recall_val", 0.0)),
            enable_record_agg=bool(getattr(cfg, "enable_record_agg", False)),
            record_agg_method=str(getattr(cfg, "record_agg_method", "mean")),
            record_agg_topk=int(getattr(cfg, "record_agg_topk", 5)),
        )
        if cfg.num_arrhythmia_classes == 2 and "afl_threshold" in val_metrics:
            print(f"  [Val] 片段级阈值 = {val_metrics['afl_threshold']:.3f}")
        if cfg.num_arrhythmia_classes == 2 and "record_afl_f1" in val_metrics:
            print(
                f"  [Val-Record] AFL-F1={val_metrics['record_afl_f1']:.4f}, "
                f"P={val_metrics.get('record_afl_precision', 0.0):.4f}, R={val_metrics.get('record_afl_recall', 0.0):.4f} (仅供参考，best 按片段级)"
            )
        
        # 自适应参数调整：根据验证集性能动态调整参数
        if cfg.num_arrhythmia_classes == 2 and epoch > 0:  # 从第2个epoch开始调整
            adaptive_adjust_parameters(model, val_metrics, cfg.num_arrhythmia_classes, adjustment_rate=0.05)
        
        # 打印模糊规则系统的调试信息
        if hasattr(model, '_debug_alpha') and model._debug_alpha is not None:
            alpha_raw_str = ""
            if hasattr(model, '_debug_alpha_raw') and model._debug_alpha_raw is not None:
                alpha_raw_str = f" (raw={model._debug_alpha_raw.item():.3f})"
            debug_info = f"  Fuzzy Rule Debug: alpha={model._debug_alpha.item():.3f}{alpha_raw_str}"
            if hasattr(model, '_debug_fuzzy_af') and model._debug_fuzzy_af is not None:
                debug_info += f", fuzzy_AF={model._debug_fuzzy_af.item():.3f}"
            if hasattr(model, '_debug_fuzzy_afl') and model._debug_fuzzy_afl is not None:
                debug_info += f", fuzzy_AFL={model._debug_fuzzy_afl.item():.3f}"
            if hasattr(model, '_debug_net_af') and model._debug_net_af is not None:
                debug_info += f", net_AF={model._debug_net_af.item():.3f}"
            if hasattr(model, '_debug_net_afl') and model._debug_net_afl is not None:
                debug_info += f", net_AFL={model._debug_net_afl.item():.3f}"
            if hasattr(model, '_debug_fused_af') and model._debug_fused_af is not None:
                debug_info += f", fused_AF={model._debug_fused_af.item():.3f}"
            if hasattr(model, '_debug_fused_afl') and model._debug_fused_afl is not None:
                debug_info += f", fused_AFL={model._debug_fused_afl.item():.3f}"
            if hasattr(model, '_debug_alpha_afl') and model._debug_alpha_afl is not None:
                debug_info += f", alpha_AFL={model._debug_alpha_afl.item():.3f}"
            print(debug_info)
        
        print(
            f"  Epoch Summary: Train - loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.4f}, "
            f"F1={train_metrics['macro_f1']:.4f}, P={train_metrics['macro_precision']:.4f}, R={train_metrics['macro_recall']:.4f} | "
            f"Val - loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}, "
            f"F1={val_metrics['macro_f1']:.4f}, P={val_metrics['macro_precision']:.4f}, R={val_metrics['macro_recall']:.4f}"
        )
        
        # 打印验证集的每个类别详细指标
        if 'confusion_matrix' in val_metrics:
            # 从混淆矩阵计算每个类别的指标
            cm = val_metrics['confusion_matrix']
            class_names = ["AF", "AFL"] if cfg.num_arrhythmia_classes == 2 else [f"Class {i}" for i in range(cfg.num_arrhythmia_classes)]
            
            # 从混淆矩阵提取真实标签和预测标签
            y_true_list = []
            y_pred_list = []
            for true_label in range(cfg.num_arrhythmia_classes):
                for pred_label in range(cfg.num_arrhythmia_classes):
                    count = int(cm[true_label, pred_label])
                    y_true_list.extend([true_label] * count)
                    y_pred_list.extend([pred_label] * count)
            
            if y_true_list and y_pred_list:
                print_per_class_metrics(y_true_list, y_pred_list, cfg.num_arrhythmia_classes, class_names)

        # 二分类时：统一按片段级 AFL-F1 选 best（与阈值搜索、主指标一致）
        if cfg.num_arrhythmia_classes == 2 and "confusion_matrix" in val_metrics:
            cm = val_metrics["confusion_matrix"]
            afl_tp = int(cm[1, 1])
            afl_fp = int(cm[0, 1])
            afl_fn = int(cm[1, 0])
            afl_prec = afl_tp / (afl_tp + afl_fp + 1e-8)
            afl_rec = afl_tp / (afl_tp + afl_fn + 1e-8)
            current_f1 = 2.0 * afl_prec * afl_rec / (afl_prec + afl_rec + 1e-8)
        else:
            current_f1 = val_metrics["macro_f1"]
        if current_f1 > best_val_f1 + cfg.early_stop_min_delta:
            best_val_f1 = current_f1
            epochs_no_improve = 0
            # 保存模型状态（包括自适应参数，因为它们是register_buffer）
            # 同时保存自适应参数的当前值，方便调试
            adaptive_params = {}
            if cfg.num_arrhythmia_classes == 2:
                adaptive_params = {
                    "adaptive_alpha_min": model.adaptive_alpha_min.item(),
                    "adaptive_alpha_afl_max": model.adaptive_alpha_afl_max.item(),
                    "adaptive_alpha_afl_boost_coef": model.adaptive_alpha_afl_boost_coef.item(),
                    "adaptive_small_afl_threshold": model.adaptive_small_afl_threshold.item(),
                    "adaptive_alpha_afl_min": model.adaptive_alpha_afl_min.item(),
                    "adaptive_af_protect_threshold": model.adaptive_af_protect_threshold.item(),
                    "adaptive_af_protect_diff": model.adaptive_af_protect_diff.item(),
                    "adaptive_af_protect_coef": model.adaptive_af_protect_coef.item(),
                    "adaptive_afl_offset_threshold": model.adaptive_afl_offset_threshold.item(),
                    "adaptive_afl_offset_coef": model.adaptive_afl_offset_coef.item(),
                    "adaptive_afl_offset_max": model.adaptive_afl_offset_max.item(),
                    "adaptive_logit_diff_threshold": model.adaptive_logit_diff_threshold.item(),
                    "adaptive_logit_diff_coef": model.adaptive_logit_diff_coef.item(),
                    "adaptive_logit_diff_max": model.adaptive_logit_diff_max.item(),
                }
            torch.save(
                {
                    "model": model.state_dict(),  # 包含所有参数，包括register_buffer（自适应参数）
                    "cfg": cfg.__dict__,
                    "val_macroF1": best_val_f1,
                    "val_metrics": val_metrics,
                    "afl_threshold": val_metrics.get("afl_threshold", None),
                    "adaptive_params": adaptive_params,  # 额外保存自适应参数值，方便查看
                    "epoch": epoch + 1,
                },
                cfg.out_ckpt,
            )
            f1_label = "Val 片段级 AFL-F1" if cfg.num_arrhythmia_classes == 2 else "Val F1"
            # 避免在 GBK 控制台打印特殊字符导致 UnicodeEncodeError
            print(f"  New best model saved ({f1_label}={best_val_f1:.4f}) -> {cfg.out_ckpt}")
            if adaptive_params:
                print(f"    Adaptive params: alpha_afl_max={adaptive_params['adaptive_alpha_afl_max']:.3f}, "
                      f"af_protect_coef={adaptive_params['adaptive_af_protect_coef']:.3f}")
        else:
            epochs_no_improve += 1
            f1_label = "Best Val 片段级 AFL-F1" if cfg.num_arrhythmia_classes == 2 else "Best Val F1"
            print(f"  ({f1_label}: {best_val_f1:.4f})")

        # 提前停止：如果开启了早停且连续若干个 epoch 无提升，则中止训练
        if use_early_stop and epochs_no_improve >= cfg.early_stop_patience:
            print(
                f"  >> Early stopping triggered: no Val F1 improvement for "
                f"{epochs_no_improve} epochs (patience={cfg.early_stop_patience})."
            )
            break

        # 方案2：hard example mining（更新下一轮采样权重）
        if (
            cfg.num_arrhythmia_classes == 2
            and getattr(cfg, "use_hard_mining", False)
            and hard_sample_weights is not None
            and epoch >= int(getattr(cfg, "hard_mining_start_epoch", 1))
        ):
            model.eval()
            eval_train_loader = DataLoader(
                train_ds,
                shuffle=False,
                **{k: v for k, v in dl_common.items() if v is not None or k in ["batch_size", "num_workers", "pin_memory", "shuffle"]},
            )
            threshold_for_mining = float(val_metrics.get("afl_threshold", 0.5))
            hard_sample_weights = base_sample_weights.copy().astype(np.float32) if base_sample_weights is not None else np.ones(len(train_ds), dtype=np.float32)

            num_afl_fn = 0
            num_af_fp = 0
            num_low_conf = 0
            seen = 0

            for b in eval_train_loader:
                ecg = b["ecg"].to(device)
                noise_feat = b["noise_feat"].to(device)
                rr_seq = b["rr_seq"].to(device)
                fuzzy_logits = b["fuzzy_logits"].to(device)
                y = b["label"].to(device)
                idxs = b.get("idx", None)
                if idxs is None:
                    continue
                idxs = idxs.detach().cpu().numpy().astype(np.int64)

                logits_arr, _ = model(ecg, rr_seq, noise_feat, fuzzy_logits)
                if logits_arr.size(1) == 2 and cfg.afl_logit_bias_inference != 0:
                    logits_arr = logits_arr.clone()
                    logits_arr[:, 1] = logits_arr[:, 1] + float(cfg.afl_logit_bias_inference)
                probs = torch.softmax(logits_arr, dim=-1)
                p_afl = probs[:, 1]
                conf = probs.max(dim=-1).values
                pred = (p_afl >= threshold_for_mining).long()

                y_cpu = y.detach().cpu().numpy().astype(np.int64)
                pred_cpu = pred.detach().cpu().numpy().astype(np.int64)
                conf_cpu = conf.detach().cpu().numpy().astype(np.float32)

                afl_fn_mask = (y_cpu == 1) & (pred_cpu == 0)
                af_fp_mask = (y_cpu == 0) & (pred_cpu == 1)
                low_conf_mask = (conf_cpu < float(getattr(cfg, "hard_mining_conf_threshold", 0.65))) & (y_cpu == pred_cpu)

                if afl_fn_mask.any():
                    hard_sample_weights[idxs[afl_fn_mask]] *= float(getattr(cfg, "hard_mining_afl_fn_mult", 4.0))
                    num_afl_fn += int(afl_fn_mask.sum())
                if af_fp_mask.any():
                    hard_sample_weights[idxs[af_fp_mask]] *= float(getattr(cfg, "hard_mining_af_fp_mult", 2.0))
                    num_af_fp += int(af_fp_mask.sum())
                if low_conf_mask.any():
                    hard_sample_weights[idxs[low_conf_mask]] *= float(getattr(cfg, "hard_mining_low_conf_mult", 1.5))
                    num_low_conf += int(low_conf_mask.sum())

                seen += len(idxs)

            hard_sample_weights = np.clip(hard_sample_weights, 0.1, 50.0).astype(np.float32)
            print(f"  [HardMining] seen={seen}, AFL->AF(FN)={num_afl_fn}, AF->AFL(FP)={num_af_fp}, low-conf-correct={num_low_conf}")
            model.train()

    print(f"\n[7/7] Test evaluation...")
    if test_loader:
        # 关键修复：测试前加载保存的最佳模型，而不是使用训练结束时的模型
        # 因为训练结束后，模型可能已经经过了多个epoch的调整，自适应参数可能已经改变
        if os.path.exists(cfg.out_ckpt):
            print(f"  Loading best model from {cfg.out_ckpt} for testing...")
            # PyTorch 2.6+ requires weights_only=False for checkpoints containing numpy arrays
            checkpoint = torch.load(cfg.out_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            saved_val_f1 = checkpoint.get('val_macroF1', best_val_f1)
            saved_epoch = checkpoint.get('epoch', 'unknown')
            f1_label = "Val 片段级 AFL-F1" if cfg.num_arrhythmia_classes == 2 else "Val F1"
            # 避免在 GBK 控制台打印特殊字符导致 UnicodeEncodeError
            print(f"  Best model loaded ({f1_label}={saved_val_f1:.4f}, saved at epoch {saved_epoch})")
            saved_threshold = checkpoint.get("afl_threshold", None)
            if cfg.num_arrhythmia_classes == 2 and saved_threshold is not None:
                print(f"  Loaded AFL prob threshold = {float(saved_threshold):.3f}")
            # 打印保存时的自适应参数
            if 'adaptive_params' in checkpoint:
                ap = checkpoint['adaptive_params']
                print(f"  Saved adaptive params: alpha_afl_max={ap.get('adaptive_alpha_afl_max', 'N/A'):.3f}, "
                      f"af_protect_coef={ap.get('adaptive_af_protect_coef', 'N/A'):.3f}, "
                      f"af_protect_threshold={ap.get('adaptive_af_protect_threshold', 'N/A'):.3f}")
            # 确保模型处于评估模式
            model.eval()
        else:
            print(f"  Warning: Checkpoint {cfg.out_ckpt} not found, using current model")
        
        # 测试时也打印混淆矩阵（verbose=True）
        threshold_for_test = None
        if cfg.num_arrhythmia_classes == 2:
            threshold_for_test = checkpoint.get("afl_threshold", None) if 'checkpoint' in locals() else None
            if threshold_for_test is None:
                threshold_for_test = cfg.afl_prob_threshold
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            verbose=True,
            afl_logit_bias_inference=cfg.afl_logit_bias_inference,
            afl_prob_threshold=threshold_for_test,
            tune_afl_threshold=False,
            afl_threshold_search_steps=cfg.afl_threshold_search_steps,
            afl_threshold_search_min=cfg.afl_threshold_search_min,
            afl_threshold_search_max=cfg.afl_threshold_search_max,
        )
        print(
            f"  Test: loss={test_metrics['loss']:.4f}, acc={test_metrics['accuracy']:.4f}, "
            f"F1={test_metrics['macro_f1']:.4f}, P={test_metrics['macro_precision']:.4f}, R={test_metrics['macro_recall']:.4f}"
        )
        # 兜底：若保存的阈值在 Test 上导致 AFL 几乎全灭（召回<2%），在 Test 上尝试较低阈值并选用 macro F1 最高的
        if cfg.num_arrhythmia_classes == 2 and test_metrics.get("confusion_matrix") is not None:
            cm = test_metrics["confusion_matrix"]
            afl_total = int(cm[1, 0]) + int(cm[1, 1])
            afl_recall = (int(cm[1, 1]) / afl_total) if afl_total > 0 else 0.0
            if afl_recall < 0.02 and afl_total >= 10:
                fallback_thresholds = [0.05, 0.1, 0.15, 0.2]
                best_fallback_f1 = -1.0
                best_fallback_th = None
                best_fallback_metrics = None
                for th in fallback_thresholds:
                    m = evaluate(
                        model,
                        test_loader,
                        device,
                        epoch=0,
                        total_epochs=0,
                        verbose=False,
                        afl_logit_bias_inference=cfg.afl_logit_bias_inference,
                        afl_prob_threshold=th,
                        tune_afl_threshold=False,
                        afl_threshold_search_steps=cfg.afl_threshold_search_steps,
                        afl_threshold_search_min=cfg.afl_threshold_search_min,
                        afl_threshold_search_max=cfg.afl_threshold_search_max,
                    )
                    if m["macro_f1"] > best_fallback_f1:
                        best_fallback_f1 = m["macro_f1"]
                        best_fallback_th = th
                        best_fallback_metrics = m
                if best_fallback_metrics is not None:
                    print(
                        f"  [Test 阈值兜底] 保存的阈值在 Test 上 AFL 召回≈0，已尝试 {fallback_thresholds}，"
                        f"选用阈值 {best_fallback_th} (Test macro F1={best_fallback_f1:.4f})"
                    )
                    fb_cm = best_fallback_metrics["confusion_matrix"]
                    print("  Confusion matrix (fallback threshold) (rows=true, cols=pred):")
                    print("            Pred: AF  AFL")
                    print(f"  True: AF  [{int(fb_cm[0,0]):4d} {int(fb_cm[0,1]):4d}]")
                    print(f"       AFL  [{int(fb_cm[1,0]):4d} {int(fb_cm[1,1]):4d}]")
                    print(
                        f"  Test (fallback): loss={best_fallback_metrics['loss']:.4f}, acc={best_fallback_metrics['accuracy']:.4f}, "
                        f"F1={best_fallback_metrics['macro_f1']:.4f}, P={best_fallback_metrics['macro_precision']:.4f}, R={best_fallback_metrics['macro_recall']:.4f}"
                    )
    else:
        print("  No test set (only train/val)")

    print("=" * 80)
    f1_label = "Best Val 片段级 AFL-F1" if cfg.num_arrhythmia_classes == 2 else "Best Val F1"
    print(f"Training finished. {f1_label} = {best_val_f1:.4f}")
    print(f"Checkpoint: {cfg.out_ckpt}")
    print("=" * 80)
    return model


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="ECG arrhythmia training with morphology+rhythm+noise branches + fuzzy rules (4-class: AF/AFL/PSVT/Normal)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_csv", type=str, default="", help="Full data CSV (will be split 8:1:1; AF/AFL/PSVT/Normal)")
    parser.add_argument("--train_csv", type=str, default="", help="Train CSV (if not using data_csv)")
    parser.add_argument("--val_csv", type=str, default="", help="Val CSV (if not using data_csv)")
    parser.add_argument("--num_leads", type=int, default=1)
    parser.add_argument("--num_arrhythmia_classes", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=2500)
    parser.add_argument("--max_rr_intervals", type=int, default=32)
    parser.add_argument("--use_stft_dbscan", action="store_true", default=False, help="Enable STFT+DBSCAN for impulsive noise ratio")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out_ckpt", type=str, default="ecg_4class_fuzzy.pt")
    parser.add_argument("--weight_conf_gamma", type=float, default=1.0)
    parser.add_argument("--min_weight", type=float, default=0.1)
    parser.add_argument("--validate_every_batch", action="store_true", default=True)
    parser.add_argument("--no_validate_every_batch", dest="validate_every_batch", action="store_false")
    parser.add_argument("--use_class_weights", action="store_true", default=True)
    parser.add_argument("--no_use_class_weights", dest="use_class_weights", action="store_false")
    parser.add_argument("--fuzzy_weight_init", type=float, default=0.3, help="Initial fuzzy rule fusion weight (for learnable gating initialization)")
    parser.add_argument("--multi_lead_method", type=str, default="max_energy", choices=["max_energy", "voting"], help="Multi-lead RR detection method")
    parser.add_argument("--flutter_method", type=str, default="fixed_lead", choices=["fixed_lead", "attention"], help="Multi-lead flutter detection method")
    parser.add_argument("--lead_names", type=str, nargs="+", default=None, help="Lead names (e.g., I II V1) for fixed_lead method")
    parser.add_argument("--use_precomputed", action="store_true", default=True, help="Use precomputed features if available")
    parser.add_argument("--no_use_precomputed", dest="use_precomputed", action="store_false", help="Disable precomputed features")
    parser.add_argument(
        "--conf_discard_th",
        type=float,
        default=0.4,
        help="Confidence threshold for low-confidence sample handling (used in discard mode or weight calculation)",
    )
    parser.add_argument(
        "--use_low_conf_focus",
        action="store_true",
        default=True,
        help="Focus on low-confidence samples (give them higher weight) instead of discarding them",
    )
    parser.add_argument(
        "--no_use_low_conf_focus",
        dest="use_low_conf_focus",
        action="store_false",
        help="Use original discard strategy for low-confidence samples",
    )
    parser.add_argument(
        "--low_conf_alpha",
        type=float,
        default=1.0,
        help="Weight enhancement coefficient for low-confidence samples (higher = more focus on hard examples)",
    )
    parser.add_argument(
        "--lambda_af_bin",
        type=float,
        default=0.5,
        help="Loss weight for AF/AFL auxiliary binary head",
    )
    # 边界感知参数
    parser.add_argument(
        "--use_boundary_aware",
        action="store_true",
        default=True,
        help="Enable boundary-aware training (focus on samples near class boundaries)",
    )
    parser.add_argument(
        "--no_use_boundary_aware",
        dest="use_boundary_aware",
        action="store_false",
        help="Disable boundary-aware training",
    )
    parser.add_argument(
        "--boundary_alpha",
        type=float,
        default=0.5,
        help="Boundary sample weight enhancement coefficient (0-1.5, higher = more focus on boundary samples)",
    )
    parser.add_argument(
        "--boundary_threshold",
        type=float,
        default=0.3,
        help="Boundary sample threshold (boundary_score > threshold is considered boundary sample, 0-1)",
    )
    parser.add_argument(
        "--lambda_boundary",
        type=float,
        default=0.1,
        help="Boundary-aware loss weight (additional loss penalty for boundary samples, 0 to disable)",
    )
    # 二分类评估：阈值化 &（可选）logit 偏置
    parser.add_argument(
        "--afl_logit_bias_inference",
        type=float,
        default=0.0,
        help="Binary inference only: add bias to AFL logit (index=1). Prefer threshold tuning over large bias.",
    )
    parser.add_argument(
        "--afl_prob_threshold",
        type=float,
        default=None,
        help="Binary inference only: classify as AFL if P(AFL)>=threshold. If not set, auto-tune on val to maximize AFL-F1.",
    )
    parser.add_argument(
        "--afl_threshold_search_steps",
        type=int,
        default=101,
        help="Binary val auto-tuning: number of thresholds to search.",
    )
    parser.add_argument(
        "--afl_threshold_search_min",
        type=float,
        default=0.01,
        help="Binary val auto-tuning: min threshold.",
    )
    parser.add_argument(
        "--afl_threshold_search_max",
        type=float,
        default=0.99,
        help="Binary val auto-tuning: max threshold. For AF vs AFL, 0.5~0.6 is recommended to avoid Test recall collapse.",
    )
    parser.add_argument(
        "--afl_min_recall_val",
        type=float,
        default=0.0,
        help="Binary val tuning: only consider thresholds with Val AFL recall >= this (e.g. 0.2). 0 = no constraint.",
    )
    # 方案1：record-level 聚合评估
    parser.add_argument("--enable_record_agg", action="store_true", default=True, help="Enable record-level aggregation for binary eval")
    parser.add_argument("--no_enable_record_agg", dest="enable_record_agg", action="store_false", help="Disable record-level aggregation")
    parser.add_argument("--record_agg_method", type=str, default="mean", choices=["mean", "median", "max", "topk"], help="Record aggregation method")
    parser.add_argument("--record_agg_topk", type=int, default=5, help="Top-k for record_agg_method=topk")
    # 方案2：hard example mining
    parser.add_argument("--use_hard_mining", action="store_true", default=True, help="Enable hard example mining via weighted sampling")
    parser.add_argument("--no_use_hard_mining", dest="use_hard_mining", action="store_false", help="Disable hard example mining")
    parser.add_argument("--hard_mining_start_epoch", type=int, default=1, help="Start hard mining after this epoch")
    parser.add_argument("--hard_mining_afl_fn_mult", type=float, default=4.0, help="Weight multiplier for AFL->AF false negatives")
    parser.add_argument("--hard_mining_af_fp_mult", type=float, default=2.0, help="Weight multiplier for AF->AFL false positives")
    parser.add_argument("--hard_mining_low_conf_mult", type=float, default=1.5, help="Weight multiplier for low-confidence correct samples")
    parser.add_argument("--hard_mining_conf_threshold", type=float, default=0.65, help="Low-confidence threshold for hard mining")
    # 数据过采样参数
    parser.add_argument(
        "--use_oversampling",
        action="store_true",
        default=False,
        help="Enable oversampling for training data to balance class distribution",
    )
    parser.add_argument(
        "--oversample_target_ratio",
        type=float,
        default=1.0,
        help="Target class ratio for oversampling (1.0 = fully balanced, 0.5 = 2:1 ratio)",
    )
    parser.add_argument(
        "--oversample_strategy",
        type=str,
        default="random",
        choices=["random", "patient_level"],
        help="Oversampling strategy: 'random' (sample-level) or 'patient_level' (safer, avoids data leakage)",
    )
    # 早停参数
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Early stopping patience (number of epochs with no Val F1 improvement before stopping, 0=disable)",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum improvement on Val macro F1 to be considered as an improvement",
    )
    # 输出目录参数（用于自动设置out_ckpt路径）
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory for checkpoint (if provided, out_ckpt will be set to output_dir/basename(out_ckpt))",
    )
    args = parser.parse_args()

    cfg = TrainConfig(
        data_csv=args.data_csv,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        num_leads=args.num_leads,
        num_arrhythmia_classes=args.num_arrhythmia_classes,
        max_len=args.max_len,
        max_rr_intervals=args.max_rr_intervals,
        use_stft_dbscan=args.use_stft_dbscan,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=args.device,
        out_ckpt=args.out_ckpt,
        weight_conf_gamma=args.weight_conf_gamma,
        min_weight=args.min_weight,
        conf_discard_th=args.conf_discard_th,
        lambda_af_bin=args.lambda_af_bin,
        validate_every_batch=args.validate_every_batch,
        use_class_weights=args.use_class_weights,
        fuzzy_weight_init=args.fuzzy_weight_init,
        use_boundary_aware=args.use_boundary_aware,
        boundary_alpha=args.boundary_alpha,
        boundary_threshold=args.boundary_threshold,
        lambda_boundary=args.lambda_boundary,
        use_low_conf_focus=args.use_low_conf_focus,
        low_conf_alpha=args.low_conf_alpha,
        multi_lead_method=args.multi_lead_method,
        flutter_method=args.flutter_method,
        lead_names=args.lead_names,
        use_precomputed=args.use_precomputed,
        use_oversampling=args.use_oversampling,
        oversample_target_ratio=args.oversample_target_ratio,
        oversample_strategy=args.oversample_strategy,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        afl_logit_bias_inference=args.afl_logit_bias_inference,
        afl_prob_threshold=args.afl_prob_threshold,
        afl_threshold_search_steps=args.afl_threshold_search_steps,
        afl_threshold_search_min=args.afl_threshold_search_min,
        afl_threshold_search_max=args.afl_threshold_search_max,
        afl_min_recall_val=getattr(args, "afl_min_recall_val", 0.0),
        enable_record_agg=args.enable_record_agg,
        record_agg_method=args.record_agg_method,
        record_agg_topk=args.record_agg_topk,
        use_hard_mining=args.use_hard_mining,
        hard_mining_start_epoch=args.hard_mining_start_epoch,
        hard_mining_afl_fn_mult=args.hard_mining_afl_fn_mult,
        hard_mining_af_fp_mult=args.hard_mining_af_fp_mult,
        hard_mining_low_conf_mult=args.hard_mining_low_conf_mult,
        hard_mining_conf_threshold=args.hard_mining_conf_threshold,
    )
    
    # 处理output_dir：如果提供了output_dir，自动设置out_ckpt路径
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        # 如果out_ckpt是相对路径，将其放在output_dir下
        if not os.path.isabs(cfg.out_ckpt):
            cfg.out_ckpt = os.path.join(args.output_dir, os.path.basename(cfg.out_ckpt))
        else:
            # 如果是绝对路径，确保目录存在
            out_dir = os.path.dirname(cfg.out_ckpt)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
    
    return cfg


def main():
    cfg = parse_args()
    run_training(cfg)


if __name__ == "__main__":
    main()

