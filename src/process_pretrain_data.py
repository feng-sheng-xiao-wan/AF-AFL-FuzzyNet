#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预处理预训练数据集：PTB-XL 和 ECG-Arrhythmia
生成与 Holter 数据格式一致的窗口切片，用于预训练
"""

import os
import glob
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly, butter, filtfilt, find_peaks
import ast
import argparse
import re

# 标签映射
LABELS = ["Other", "AF", "AFL", "PSVT"]
LABEL_MAP = {n: i for i, n in enumerate(LABELS)}

# PTB-XL 标签映射
AF_CODES = {"AFIB", "AF"}
AFL_CODES = {"AFLT", "AFL"}
PSVT_CODES = {"PSVT", "SVT", "SVTAC", "SVARR", "SVTACX"}
# 仅将以下视为 Normal，其它诊断直接丢弃
PTBXL_NORMAL_CODES = {"NORM", "SR", "SARRH", "SBRAD", "STACH"}

def parse_scp_codes(s):
    """解析PTB-XL的SCP代码"""
    if isinstance(s, dict):
        return s
    s = str(s).strip()
    s = s.replace("null", "None")
    try:
        d = ast.literal_eval(s)
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    return {}

def map_ptbxl_label(scp_dict):
    """将PTB-XL的SCP代码映射到我们的标签"""
    # 只考虑值>0的代码（值为0表示不存在该诊断）
    codes = set(k.upper() for k, v in scp_dict.items() if v > 0)
    if len(codes) == 0:
        return "Other"  # 无诊断，视为正常
    if len(codes & AF_CODES) > 0:
        return "AF"
    if len(codes & AFL_CODES) > 0:
        return "AFL"
    if len(codes & PSVT_CODES) > 0:
        return "PSVT"
    # 若全部代码属于窦性/正常相关，则标为 Normal(Other)，否则丢弃
    if len(codes - PTBXL_NORMAL_CODES) == 0:
        return "Other"
    return None  # 其他异常直接跳过

def map_mit_label(ann):
    """将MIT-BIH的标注映射到我们的标签"""
    # MIT-BIH 标注通常是 'N' (正常), 'A' (房性早搏), 'V' (室性早搏) 等
    # 这里简化处理，主要关注是否有AF相关标注
    # 实际使用中可能需要更复杂的规则
    aux = ann.aux_note if hasattr(ann, 'aux_note') else []
    labels = [s.replace('(', '').replace(')', '').strip() if s else '' for s in aux]
    
    # 检查是否有AF相关标注
    for lab in labels:
        lab_upper = lab.upper()
        if 'AF' in lab_upper or 'AFIB' in lab_upper:
            return "AF"
        if 'AFL' in lab_upper or 'AFLT' in lab_upper:
            return "AFL"
        if 'SVT' in lab_upper or 'PSVT' in lab_upper:
            return "PSVT"
    
    return "Other"


# SNOMED-CT 代码到类别的映射（用于 ECG-Arrhythmia 数据集）
# 仅映射 AF / AFL / PSVT 相关代码，其他异常将被过滤掉
SNOMED_TO_LABEL = {
    # 心房颤动 (Atrial Fibrillation)
    '164889003': 'AF',      # AFIB - Atrial Fibrillation
    
    # 心房扑动 (Atrial Flutter)
    '164890007': 'AFL',     # AFL - Atrial Flutter
    
    # 室上性心动过速相关 (Supraventricular Tachycardia)
    '426761007': 'PSVT',    # SVT - Supraventricular Tachycardia
    '713422000': 'PSVT',    # AT - Atrial Tachycardia
    '233896004': 'PSVT',    # AVNRT - Atrioventricular Node Reentrant Tachycardia
    '233897008': 'PSVT',    # AVRT - Atrioventricular Reentrant Tachycardia
}

# 窦性/轻微节律对应的 SNOMED 代码（数字串）
SINUS_SNOMED_CODES = {
    "426783006",  # Sinus rhythm
    "427393009",  # Sinus arrhythmia
    "427084000",  # Sinus bradycardia
    "427385000",  # Sinus tachycardia
}

# 兼容部分注释中出现的文字形式（若出现非数字）
SINUS_CODES = {"SR", "NSR", "SB", "SA", "ST"}

def map_ecg_arrhythmia_label(comments):
    """
    从 ECG-Arrhythmia 记录的 comments 中提取 SNOMED-CT 诊断代码并映射到类别。
    
    四分类策略：仅保留 AF / AFL / PSVT / Normal(Other) 四类
    - 没有 Dx 信息：视为 Normal(Other)
    - Dx 中包含 AF/AFL/PSVT 之一且仅一个：返回对应类别
    - Dx 中包含多个 AF/AFL/PSVT 类别：丢弃该样本（避免标签交叉）
    - Dx 存在但都不在映射表（如室性心律失常、传导阻滞等）：丢弃该样本
    
    返回值：
        - "AF": 心房颤动
        - "AFL": 心房扑动
        - "PSVT": 室上性心动过速
        - "Other": 正常（无异常诊断）
        - None: 其他异常或标签交叉，将被过滤掉
    """
    if not comments:
        return "Other"  # 无诊断信息，视为正常
    
    # 提取 Dx 诊断代码
    raw_codes = []
    for comment in comments:
        if isinstance(comment, str) and comment.startswith('Dx:'):
            codes_str = comment.replace('Dx:', '').strip()
            codes = [c.strip() for c in codes_str.split(',') if c.strip()]
            raw_codes.extend(codes)
            break

    # 规范化：优先提取数字串（SNOMED code），若没有数字则取大写文本
    dx_codes = []
    for c in raw_codes:
        m = re.findall(r"\d+", c)
        if m:
            dx_codes.append(m[0])
        else:
            dx_codes.append(c.upper())
    
    # 没有 Dx 代码，默认正常
    if not dx_codes:
        return "Other"
    
    # 如果 Dx 仅包含窦性/轻微节律（如 SR/NSR/SB/SA/ST），视为 Normal
    if all((c in SINUS_SNOMED_CODES or c in SINUS_CODES) for c in dx_codes):
        return "Other"

    # 提取所有匹配的标签（不按优先级，而是检查是否有多个）
    found_labels = set()
    for code in dx_codes:
        if code in SNOMED_TO_LABEL:
            found_labels.add(SNOMED_TO_LABEL[code])
    
    # 如果找到多个标签（AF、AFL、PSVT中的多个），丢弃该样本
    if len(found_labels) > 1:
        return None  # 标签交叉，丢弃
    
    # 如果只找到一个标签，返回该标签
    if len(found_labels) == 1:
        return list(found_labels)[0]
    
    # Dx 存在但不在映射表（其他诊断）：丢弃
    return None

def bandpass_filter(x, fs, band=(0.5, 40.0), order=4):
    b, a = butter(order, [band[0]/(fs/2.0), band[1]/(fs/2.0)], btype='band')
    return filtfilt(b, a, x, axis=-1)

def robust_zscore(x, axis=-1, eps=1e-6):
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True) + eps
    return (x - med) / (1.4826 * mad)

def read_fs_from_header(hea_path):
    with open(hea_path, 'r') as f:
        line = f.readline()
    parts = line.split()
    return int(float(parts[2]))

def extract_rr_from_ecg(ecg, fs):
    """从ECG信号中提取R峰并计算RR间期"""
    # 选择信号最强的导联
    if len(ecg.shape) == 2:
        c = int(np.argmax(np.std(ecg, axis=1)))
        sig = ecg[c]
    else:
        sig = ecg
    
    # 简单的R峰检测：差分+阈值
    dx = np.diff(sig, prepend=sig[0])
    env = dx ** 2
    win = max(1, int(0.12 * fs))
    ma = np.convolve(env, np.ones(win) / win, mode='same')
    thr = 0.3 * np.max(ma) + 1e-6
    distance = int(0.25 * fs)
    
    try:
        peaks, _ = find_peaks(ma, height=thr, distance=distance)
    except Exception:
        # 如果没有scipy，使用简单方法
        peaks = []
        for i in range(distance, len(ma) - distance):
            if ma[i] > thr and ma[i] == np.max(ma[i-distance:i+distance+1]):
                peaks.append(i)
        peaks = np.array(peaks)
    
    if len(peaks) < 2:
        return np.array([], dtype=np.float32)
    
    # 计算RR间期（秒）
    rr = np.diff(peaks) / float(fs)
    return rr.astype(np.float32)

def slice_ptbxl(
    data_root='data',
    out_root='data/slices_PTB-XL_2leads_II_V1',
    ptbxl_root=None,
    target_fs=250,
    leads=(0, 1),  # 备用：若未找到名称，则按索引
    ptbxl_lead_names=("II", "V1"),  # 优先按导联名称选择 II、V1
    window_sec=10.0
):
    """处理PTB-XL数据集"""
    print("\n" + "="*60)
    print("处理 PTB-XL 数据集")
    print("="*60)
    
    # 查找PTB-XL根目录
    if ptbxl_root is None:
        ptbxl_base = os.path.join(data_root, "ptb-xl")
        if os.path.exists(os.path.join(ptbxl_base, "ptbxl_database.csv")):
            ptbxl_root = ptbxl_base
        else:
            cands = glob.glob(os.path.join(ptbxl_base, "*"))
            cands = [d for d in cands if os.path.isdir(d)]
            for d in sorted(cands, reverse=True):
                if os.path.exists(os.path.join(d, "ptbxl_database.csv")):
                    ptbxl_root = d
                    break
    
    if ptbxl_root is None or not os.path.exists(os.path.join(ptbxl_root, "ptbxl_database.csv")):
        print(f"❌ PTB-XL 数据库文件未找到: {ptbxl_root}")
        return None
    
    meta_path = os.path.join(ptbxl_root, "ptbxl_database.csv")
    df_meta = pd.read_csv(meta_path)
    
    # 选择高分辨率文件（filename_hr）或低分辨率（filename_lr）
    fn_col = "filename_hr" if "filename_hr" in df_meta.columns else "filename_lr"
    
    windows_dir = os.path.join(out_root, 'windows_ptbxl')
    os.makedirs(windows_dir, exist_ok=True)
    out_manifest = os.path.join(out_root, 'windows_manifest_ptbxl.csv')
    
    rows = []
    skipped = 0
    lead_info_printed = False
    
    for i, row in df_meta.iterrows():
        if i % 1000 == 0 and i > 0:
            print(f"  已处理 {i}/{len(df_meta)} 条记录...")
        
        try:
            # 解析标签
            scp_dict = parse_scp_codes(row.get('scp_codes', '{}'))
            label = map_ptbxl_label(scp_dict)
            if label is None:
                skipped += 1
                continue  # 过滤掉非 AF/AFL/PSVT/Normal 的诊断
            
            # 获取文件路径
            filename = row.get(fn_col, '')
            if pd.isna(filename) or not filename:
                skipped += 1
                continue
            
            rec_path = os.path.join(ptbxl_root, filename)
            rec_base = rec_path.replace('.hea', '').replace('.dat', '')
            
            if not os.path.exists(rec_base + '.hea') or not os.path.exists(rec_base + '.dat'):
                skipped += 1
                continue
            
            # 读取信号
            fs = read_fs_from_header(rec_base + '.hea')
            sig, fields = wfdb.rdsamp(rec_base)
            sig_names = fields.get("sig_name", []) if isinstance(fields, dict) else []
            sig = sig.T  # (C, N)
            
            # 优先按导联名称选择（II、V1），否则回退索引
            selected = None
            selected_names = []
            if sig_names:
                name_to_idx = {n.upper(): idx for idx, n in enumerate(sig_names)}
                wanted = [n.upper() for n in ptbxl_lead_names] if ptbxl_lead_names else []
                idxs = [name_to_idx[n] for n in wanted if n in name_to_idx]
                if len(idxs) >= 2:
                    selected = idxs[:2]
                    selected_names = wanted[:2]
                elif len(idxs) == 1:
                    selected = idxs + [idxs[0]]
                    selected_names = wanted[:1] + wanted[:1]
            if selected is None:
                sel_idx = [c for c in leads if c < sig.shape[0]]
                if not sel_idx:
                    sel_idx = [0]
                selected = sel_idx
                selected_names = [f"idx_{i}" for i in selected]
            sig = sig[selected, :]
            
            if not lead_info_printed:
                print(f"  使用导联（PTB-XL）: {selected_names}")
                lead_info_printed = True
            
            # 重采样
            if int(fs) != int(target_fs):
                sig = resample_poly(sig, up=int(target_fs), down=int(fs), axis=-1)
                fs_eff = int(target_fs)
            else:
                fs_eff = int(fs)
            
            # 滤波和标准化
            sig = bandpass_filter(sig, fs_eff, band=(0.5, 40.0))
            sig = robust_zscore(sig, axis=-1)
            
            # PTB-XL记录通常是10秒，使用滑动窗口提取多个窗口（充分利用数据）
            win = int(window_sec * fs_eff)  # 窗口长度（默认10秒）
            step = int(2.0 * fs_eff)  # 2秒步长（与Holter数据一致）
            
            if sig.shape[1] < win:
                # 如果记录太短，跳过
                skipped += 1
                continue
            
            # 滑动窗口提取（10秒记录可以提取2个窗口：0-8秒和2-10秒）
            for st in range(0, sig.shape[1] - win + 1, step):
                ed = st + win
                t0 = st / float(fs_eff)
                t1 = ed / float(fs_eff)
                
                sig_win = sig[:, st:ed]
                
                # 提取RR间期
                rr = extract_rr_from_ecg(sig_win, fs_eff)
                
                # 保存
                rec_id = os.path.basename(rec_base)
                fname = f"{rec_id}_{t0:.3f}_{t1:.3f}.npz"
                fpath = os.path.join(windows_dir, fname)
                
                np.savez(
                    fpath,
                    ecg=sig_win.T.astype(np.float32),
                    rr=rr,
                    fs=fs_eff
                )
                
                rows.append({
                    'record_id': rec_id,
                    'path': fpath.replace('\\', '/'),
                    't0': t0,
                    't1': t1,
                    'label_raw': label,
                    'split': 'train',  # 预训练数据统一标记为train
                    'dataset': 'ptbxl'
                })
            
        except Exception as e:
            skipped += 1
            if i < 10:  # 只打印前10个错误
                print(f"  ⚠️  跳过记录 {i}: {e}")
            continue
    
    if len(rows) == 0:
        print("❌ 没有成功处理的记录!")
        return None
    
    df = pd.DataFrame(rows)
    df.to_csv(out_manifest, index=False)
    print(f"\n✅ PTB-XL 处理完成:")
    print(f"   成功: {len(df)} 窗口")
    print(f"   跳过: {skipped} 记录")
    print(f"   清单: {out_manifest}")
    
    # 显示标签分布
    print(f"\n📊 标签分布:")
    for lab in LABELS:
        count = len(df[df['label_raw'] == lab])
        pct = count / len(df) * 100 if len(df) > 0 else 0
        print(f"  {lab}: {count} ({pct:.1f}%)")
    
    return out_manifest

def slice_ecg_arrhythmia(
    data_root='data',
    out_root='data/slices',
    target_fs=250,
    window_sec=10.0,
    lead_names=('II', 'V1'),
):
    """处理ECG-Arrhythmia (MIT-BIH)数据集"""
    print("\n" + "="*60)
    print("处理 ECG-Arrhythmia (MIT-BIH) 数据集")
    print("="*60)
    
    ecg_arr_dir = os.path.join(data_root, "ecg-arrhythmia", "WFDBRecords")
    if not os.path.exists(ecg_arr_dir):
        print(f"❌ ECG-Arrhythmia 目录未找到: {ecg_arr_dir}")
        return None
    
    # 查找所有记录文件
    recs = []
    for root, dirs, files in os.walk(ecg_arr_dir):
        for f in files:
            if f.endswith('.hea'):
                recs.append(os.path.join(root, f[:-4]))
    
    recs = sorted(recs)
    print(f"  找到 {len(recs)} 条记录")
    
    windows_dir = os.path.join(out_root, 'windows_ecg_arrhythmia')
    os.makedirs(windows_dir, exist_ok=True)
    out_manifest = os.path.join(out_root, 'windows_manifest_ecg_arrhythmia.csv')
    
    rows = []
    skipped = 0
    
    for i, rec_path in enumerate(recs):
        if i % 100 == 0 and i > 0:
            print(f"  已处理 {i}/{len(recs)} 条记录...")
        
        try:
            rec_id = os.path.basename(rec_path)
            
            # 读取信号
            fs = read_fs_from_header(rec_path + '.hea')
            sig, rec_obj = wfdb.rdsamp(rec_path)
            sig = sig.T  # (C, N)

            # 选择指定导联（优先按名称匹配，如 II、V1；否则回退前两导联）
            selected = None
            if rec_obj is not None and hasattr(rec_obj, 'sig_name') and rec_obj.sig_name:
                names = list(rec_obj.sig_name)
                name_to_idx = {n.upper(): idx for idx, n in enumerate(names)}
                wanted = [n.upper() for n in lead_names] if lead_names else []
                idxs = [name_to_idx[n] for n in wanted if n in name_to_idx]
                if len(idxs) >= 2:
                    selected = idxs[:2]
                elif len(idxs) == 1:
                    selected = idxs + [idxs[0]]
            
            if selected is None:
                # 兜底：取前两导联
                selected = [0, 1 if sig.shape[0] > 1 else 0]

            sig = sig[selected, :]
            
            # 重采样
            if int(fs) != int(target_fs):
                sig = resample_poly(sig, up=int(target_fs), down=int(fs), axis=-1)
                fs_eff = int(target_fs)
            else:
                fs_eff = int(fs)
            
            # 滤波和标准化
            sig = bandpass_filter(sig, fs_eff, band=(0.5, 40.0))
            sig = robust_zscore(sig, axis=-1)
            
            # ECG-Arrhythmia 记录通常是短记录（约10秒），每条记录直接作为一个窗口
            # 不需要滑动窗口切分
            win_samples = int(window_sec * fs_eff)
            sig_len = sig.shape[1]
            
            # 读取标注（用于标签）
            # ECG-Arrhythmia 数据集的诊断代码存储在 .hea 文件的 comments 中
            try:
                rec = wfdb.rdrecord(rec_path)
                comments = rec.comments if hasattr(rec, 'comments') else []
                label = map_ecg_arrhythmia_label(comments)
                if label is None:
                    skipped += 1
                    continue  # 过滤掉非 AF/AFL/PSVT/Normal 的诊断
            except Exception as e:
                if i < 10:
                    print(f"  [Warning] Failed to read label for {rec_id}: {e}")
                label = "Other"
            
            # 处理窗口：如果记录长度 >= 8秒，截取前8秒；如果 < 8秒，补零到8秒
            if sig_len >= win_samples:
                # 记录足够长，截取前8秒
                sig_win = sig[:, :win_samples]
                t0 = 0.0
                t1 = window_sec
            else:
                # 记录较短，补零到8秒
                sig_win = np.zeros((sig.shape[0], win_samples), dtype=sig.dtype)
                sig_win[:, :sig_len] = sig
                t0 = 0.0
                t1 = window_sec
                # 如果记录太短（< 4秒），跳过
                if sig_len < int(4.0 * fs_eff):
                    skipped += 1
                    continue
            
            # 提取RR间隔
            rr = extract_rr_from_ecg(sig_win, fs_eff)
            
            # 保存
            fname = f"{rec_id}.npz"
            fpath = os.path.join(windows_dir, fname)
            
            np.savez(
                fpath,
                ecg=sig_win.T.astype(np.float32),
                rr=rr,
                fs=fs_eff
            )
            
            rows.append({
                'record_id': rec_id,
                'path': fpath.replace('\\', '/'),
                't0': t0,
                't1': t1,
                'label_raw': label,
                'split': 'train',
                'dataset': 'ecg_arrhythmia'
            })
            
        except Exception as e:
            skipped += 1
            if i < 10:
                print(f"  ⚠️  跳过记录 {rec_id}: {e}")
            continue
    
    if len(rows) == 0:
        print("❌ 没有成功处理的记录!")
        return None
    
    df = pd.DataFrame(rows)
    df.to_csv(out_manifest, index=False)
    print(f"\n✅ ECG-Arrhythmia 处理完成:")
    print(f"   成功: {len(df)} 窗口")
    print(f"   跳过: {skipped} 记录")
    print(f"   清单: {out_manifest}")
    
    # 显示标签分布
    print(f"\n📊 标签分布:")
    for lab in LABELS:
        count = len(df[df['label_raw'] == lab])
        pct = count / len(df) * 100 if len(df) > 0 else 0
        print(f"  {lab}: {count} ({pct:.1f}%)")
    
    return out_manifest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='预处理预训练数据集（PTB-XL 和 ECG-Arrhythmia）')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['ptbxl', 'ecg_arrhythmia', 'all'],
        default='all',
        help='要处理的数据集'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='data',
        help='数据根目录'
    )
    parser.add_argument(
        '--out_root',
        type=str,
        default='data/slices',
        help='通用输出根目录（备用，若未单独指定各数据集输出目录）'
    )
    parser.add_argument(
        '--ptbxl_out_root',
        type=str,
        default='data/slices_PTB-XL_2leads_II_V1',
        help='PTB-XL 输出目录（默认：data/slices_PTB-XL_2leads_II_V1）'
    )
    parser.add_argument(
        '--ecg_arr_out_root',
        type=str,
        default='data/slices_ecg_arrhythmia_2leads_II_V1',
        help='ECG-Arrhythmia 输出目录（默认：data/slices_ecg_arrhythmia_2leads_II_V1）'
    )
    parser.add_argument(
        '--target_fs',
        type=int,
        default=250,
        help='目标采样率（Hz）'
    )
    parser.add_argument(
        '--window_sec',
        type=float,
        default=10.0,
        help='窗口长度（秒）'
    )
    parser.add_argument(
        '--leads',
        type=int,
        nargs='+',
        default=[0, 1],
        help='PTB-XL使用的导联索引（备用，若按名称未找到则使用，默认0 1）'
    )
    parser.add_argument(
        '--ptbxl_leads',
        type=str,
        nargs='+',
        default=['II', 'V1'],
        help='PTB-XL优先使用的导联名称（默认 II 和 V1，大小写不敏感）'
    )
    parser.add_argument(
        '--ecg_arr_leads',
        type=str,
        nargs='+',
        default=['II', 'V1'],
        help='ECG-Arrhythmia 使用的导联名称（默认 II 和 V1，大小写不敏感）'
    )
    
    args = parser.parse_args()
    
    manifests = []
    
    if args.dataset in ['ptbxl', 'all']:
        ptb_out = args.ptbxl_out_root or args.out_root
        os.makedirs(ptb_out, exist_ok=True)
        manifest = slice_ptbxl(
            data_root=args.data_root,
            out_root=ptb_out,
            target_fs=args.target_fs,
            leads=tuple(args.leads),
            ptbxl_lead_names=tuple(args.ptbxl_leads),
            window_sec=args.window_sec
        )
        if manifest:
            manifests.append(manifest)
    
    if args.dataset in ['ecg_arrhythmia', 'all']:
        ecg_out = args.ecg_arr_out_root or args.out_root
        os.makedirs(ecg_out, exist_ok=True)
        manifest = slice_ecg_arrhythmia(
            data_root=args.data_root,
            out_root=ecg_out,
            target_fs=args.target_fs,
            window_sec=args.window_sec,
            lead_names=tuple(args.ecg_arr_leads),
        )
        if manifest:
            manifests.append(manifest)
    
    if len(manifests) > 0:
        print(f"\n✅ 预处理完成! 共生成 {len(manifests)} 个清单文件")
        print(f"   可以用于预训练: python holter4c.py pretrain --manifests {' '.join(manifests)}")

