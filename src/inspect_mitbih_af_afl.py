# -*- coding: utf-8 -*-
"""
检查 MIT-BIH Arrhythmia Database 中 AF / AFL 的详细情况。

用法示例（注意路径要改成你自己的实际路径）：

    python inspect_mitbih_af_afl.py ^
        --db_dir "E:\\医学研究\\mit-bih-arrhythmia-database-1.0.0\\mit-bih-arrhythmia-database-1.0.0"

说明：
- MIT-BIH Arrhythmia Database 主要是按心搏（beat）标注，而不是像 SHDB/LTAFDB 那样
  用 aux_note 做整段心律(AFIB/AFL)标注。
- 这里我们做详细检查：
  1）统计每条记录中 aux_note 里是否出现过 "(AFIB" / "(AFL" 等节律标记；
  2）统计 symbol 中与 AF/AFL 相关的节律/异常；
  3）计算每个标记的持续时间（从当前标记到下一个标记或记录结束）；
  4）显示标记的详细统计信息。

最终输出：每条记录的 AF / AFL 详细信息和全库汇总统计。
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import wfdb
except ImportError:
    wfdb = None
    raise ImportError("请先安装 wfdb: pip install wfdb")


def inspect_record_detailed(record_path: str, record_name: str, verbose: bool = False) -> Tuple[int, int, List[Dict], List[Dict]]:
    """
    详细检查单条 MIT-BIH 记录的 AF / AFL 标记。
    
    返回: (af_count, afl_count, af_details, afl_details)
    - af_details: 每个AF标记的详细信息 [{'sample': int, 'time_sec': float, 'duration_sec': float, 'source': str}, ...]
    - afl_details: 每个AFL标记的详细信息
    """
    ann_path = record_path
    try:
        ann = wfdb.rdann(ann_path, "atr")
        # 读取记录信息以获取采样率和总长度
        try:
            record = wfdb.rdrecord(record_path, channels=[0])
            fs = record.fs
            total_samples = len(record.p_signal)
        except:
            fs = 360.0  # MIT-BIH默认采样率
            total_samples = ann.sample[-1] if len(ann.sample) > 0 else 0
    except Exception as e:
        if verbose:
            print(f"  记录 {record_name}: 无法读取标注 ({e})")
        return 0, 0, [], []

    af_markers = []  # [(sample, source, aux_note_str)]
    afl_markers = []  # [(sample, source, aux_note_str)]

    # 1）aux_note 中查找 AFIB/AFL 等节律标记
    aux_notes = getattr(ann, "aux_note", None)
    if aux_notes is not None:
        for i, aux in enumerate(aux_notes):
            if not aux:
                continue
            s = str(aux).strip().upper()
            sample_idx = ann.sample[i] if i < len(ann.sample) else 0
            
            # 更全面的AF检测
            if any(keyword in s for keyword in ["AFIB", "(AF", "AFIB", "ATRIAL FIB", "ATRIAL_FIB"]):
                af_markers.append((sample_idx, "aux_note", s))
            
            # 更全面的AFL检测
            if any(keyword in s for keyword in ["AFL", "FLUTTER", "ATRIAL FLUTTER", "ATRIAL_FLUTTER"]):
                afl_markers.append((sample_idx, "aux_note", s))

    # 2）symbol 中尝试查找与 AF/AFL 相关的符号
    symbols = getattr(ann, "symbol", [])
    for i, sym in enumerate(symbols):
        if not sym:
            continue
        s = str(sym).strip().upper()
        sample_idx = ann.sample[i] if i < len(ann.sample) else 0
        
        # MIT-BIH可能使用的符号（虽然标准中没有，但检查一下）
        if s in ["AF", "AFIB", "A"]:  # 某些变体可能用A表示atrial
            af_markers.append((sample_idx, "symbol", s))
        if s in ["AFL", "FL"]:
            afl_markers.append((sample_idx, "symbol", s))

    # 计算每个标记的持续时间
    af_details = []
    afl_details = []
    
    # 合并所有标记点并排序
    all_markers = [(s, 'AF', src, note) for s, src, note in af_markers] + \
                  [(s, 'AFL', src, note) for s, src, note in afl_markers]
    all_markers.sort(key=lambda x: x[0])
    
    # 计算持续时间
    for idx, (sample, label_type, source, note) in enumerate(all_markers):
        # 找到下一个标记点（或记录结束）
        if idx + 1 < len(all_markers):
            next_sample = all_markers[idx + 1][0]
            duration_samples = next_sample - sample
        else:
            duration_samples = total_samples - sample
        
        duration_sec = duration_samples / fs if fs > 0 else 0
        time_sec = sample / fs if fs > 0 else 0
        
        detail = {
            'sample': sample,
            'time_sec': time_sec,
            'duration_sec': duration_sec,
            'duration_samples': duration_samples,
            'source': source,
            'note': note
        }
        
        if label_type == 'AF':
            af_details.append(detail)
        else:
            afl_details.append(detail)

    af_count = len(af_details)
    afl_count = len(afl_details)
    
    if verbose and (af_count > 0 or afl_count > 0):
        print(f"\n记录 {record_name} (采样率 {fs} Hz, 总长度 {total_samples/fs:.1f} 秒):")
        if af_count > 0:
            print(f"  AF 标记 {af_count} 次:")
            for d in af_details:
                print(f"    - 位置: {d['sample']} 样本点 ({d['time_sec']:.2f} 秒), "
                      f"持续时间: {d['duration_sec']:.2f} 秒 ({d['duration_samples']} 样本点), "
                      f"来源: {d['source']}, 标注: {d['note']}")
        if afl_count > 0:
            print(f"  AFL 标记 {afl_count} 次:")
            for d in afl_details:
                print(f"    - 位置: {d['sample']} 样本点 ({d['time_sec']:.2f} 秒), "
                      f"持续时间: {d['duration_sec']:.2f} 秒 ({d['duration_samples']} 样本点), "
                      f"来源: {d['source']}, 标注: {d['note']}")
    elif af_count > 0 or afl_count > 0:
        print(f"记录 {record_name}: AF 标记 {af_count} 次, AFL 标记 {afl_count} 次")
    
    return af_count, afl_count, af_details, afl_details


def inspect_record(record_path: str, record_name: str) -> Tuple[int, int]:
    """简化版本，只返回计数"""
    af_c, afl_c, _, _ = inspect_record_detailed(record_path, record_name, verbose=False)
    return af_c, afl_c


def main():
    parser = argparse.ArgumentParser(
        description="详细统计 MIT-BIH Arrhythmia Database 中 AF / AFL 的数量和持续时间"
    )
    parser.add_argument(
        "--db_dir",
        type=str,
        required=True,
        help="MIT-BIH Arrhythmia 数据库目录（包含 .hea/.dat/.atr 的目录）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示每条记录的详细标记信息（包括位置和持续时间）",
    )
    parser.add_argument(
        "--only_with_af_afl",
        action="store_true",
        help="只显示包含AF或AFL标记的记录",
    )
    parser.add_argument(
        "--show_all_annotations",
        action="store_true",
        help="显示所有唯一的aux_note和symbol值（用于发现可能的遗漏）",
    )
    args = parser.parse_args()

    db_dir = Path(args.db_dir)
    if not db_dir.exists():
        raise FileNotFoundError(f"数据库目录不存在: {db_dir}")

    # 查找所有 .hea 文件，视为一个记录
    hea_files = sorted(db_dir.glob("*.hea"))
    if not hea_files:
        print(f"在目录 {db_dir} 下没有找到 .hea 文件，请确认路径是否正确。")
        return

    print("=" * 80)
    print(f"MIT-BIH Arrhythmia Database 路径: {db_dir}")
    print(f"检测到 {len(hea_files)} 个记录（按 .hea 文件计）")
    print("=" * 80)

    total_af = 0
    total_afl = 0
    all_af_details = []
    all_afl_details = []
    records_with_af_afl = []
    
    # 用于收集所有标注值（检查遗漏）
    all_aux_notes = set()
    all_symbols = set()

    for hea in hea_files:
        record_name = hea.stem
        record_path = str(hea.with_suffix(""))  # 去掉扩展名，wfdb 使用基名
        
        # 如果需要收集所有标注值
        if args.show_all_annotations:
            try:
                ann = wfdb.rdann(record_path, "atr")
                aux_notes = getattr(ann, "aux_note", None)
                if aux_notes is not None:
                    for aux in aux_notes:
                        if aux:
                            all_aux_notes.add(str(aux).strip())
                symbols = getattr(ann, "symbol", None)
                if symbols is not None:
                    for sym in symbols:
                        if sym:
                            all_symbols.add(str(sym).strip())
            except:
                pass
        
        if args.verbose or args.only_with_af_afl:
            af_c, afl_c, af_details, afl_details = inspect_record_detailed(
                record_path, record_name, verbose=args.verbose
            )
            if af_c > 0 or afl_c > 0:
                all_af_details.extend(af_details)
                all_afl_details.extend(afl_details)
                records_with_af_afl.append((record_name, af_c, afl_c))
        else:
            af_c, afl_c = inspect_record(record_path, record_name)
            if af_c > 0 or afl_c > 0:
                records_with_af_afl.append((record_name, af_c, afl_c))
        
        total_af += af_c
        total_afl += afl_c

    print("\n" + "=" * 80)
    print("全库汇总：")
    print(f"  AF 标记总数 : {total_af}")
    print(f"  AFL 标记总数: {total_afl}")
    print(f"  包含AF/AFL的记录数: {len(records_with_af_afl)}")
    
    if len(records_with_af_afl) > 0:
        print(f"\n包含AF/AFL的记录列表:")
        for rec_name, af_c, afl_c in records_with_af_afl:
            print(f"  {rec_name}: AF={af_c}, AFL={afl_c}")
    
    # 统计持续时间信息
    if all_af_details:
        af_durations = [d['duration_sec'] for d in all_af_details]
        print(f"\nAF 标记持续时间统计:")
        print(f"  最短: {min(af_durations):.2f} 秒")
        print(f"  最长: {max(af_durations):.2f} 秒")
        print(f"  平均: {np.mean(af_durations):.2f} 秒")
        print(f"  中位数: {np.median(af_durations):.2f} 秒")
        print(f"  总持续时间: {sum(af_durations):.2f} 秒 ({sum(af_durations)/60:.2f} 分钟)")
    
    if all_afl_details:
        afl_durations = [d['duration_sec'] for d in all_afl_details]
        print(f"\nAFL 标记持续时间统计:")
        print(f"  最短: {min(afl_durations):.2f} 秒")
        print(f"  最长: {max(afl_durations):.2f} 秒")
        print(f"  平均: {np.mean(afl_durations):.2f} 秒")
        print(f"  中位数: {np.median(afl_durations):.2f} 秒")
        print(f"  总持续时间: {sum(afl_durations):.2f} 秒 ({sum(afl_durations)/60:.2f} 分钟)")
    
    # 显示所有标注值（用于发现可能的遗漏）
    if args.show_all_annotations:
        print(f"\n" + "=" * 80)
        print("所有唯一的标注值（用于检查可能的遗漏）:")
        if all_aux_notes:
            print(f"\naux_note 唯一值 ({len(all_aux_notes)} 个):")
            sorted_aux = sorted(all_aux_notes)
            for i, note in enumerate(sorted_aux, 1):
                # 标记可能相关的
                is_af_related = any(kw in note.upper() for kw in ["AF", "FIB", "ATRIAL"])
                is_afl_related = any(kw in note.upper() for kw in ["AFL", "FLUTTER"])
                marker = ""
                if is_af_related or is_afl_related:
                    marker = " <-- 可能与AF/AFL相关"
                print(f"  {i:3d}. {note}{marker}")
        else:
            print("\n未找到 aux_note 标注")
        
        if all_symbols:
            print(f"\nsymbol 唯一值 ({len(all_symbols)} 个):")
            sorted_syms = sorted(all_symbols)
            for i, sym in enumerate(sorted_syms, 1):
                is_af_related = any(kw in sym.upper() for kw in ["AF", "FIB", "ATRIAL"])
                is_afl_related = any(kw in sym.upper() for kw in ["AFL", "FLUTTER"])
                marker = ""
                if is_af_related or is_afl_related:
                    marker = " <-- 可能与AF/AFL相关"
                print(f"  {i:3d}. {sym}{marker}")
        else:
            print("\n未找到 symbol 标注")
        print("=" * 80)
    
    print("=" * 80)


if __name__ == "__main__":
    main()


