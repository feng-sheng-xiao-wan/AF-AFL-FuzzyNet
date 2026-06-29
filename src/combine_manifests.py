#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并 PTB-XL 和 ECG-Arrhythmia 数据集的 manifest 文件
生成 combined_windows_manifest.csv
"""

import pandas as pd
import os

def combine_manifests(
    ptbxl_manifest='data/slices_PTB-XL_2leads_II_V1/windows_manifest_ptbxl.csv',
    ecg_arr_manifest='data/slices_ecg_arrhythmia_2leads_II_V1/windows_manifest_ecg_arrhythmia.csv',
    output_path='data/combined_windows_manifest.csv'
):
    """合并两个 manifest 文件"""
    
    print("="*60)
    print("合并 Manifest 文件")
    print("="*60)
    
    # 读取 PTB-XL manifest
    if not os.path.exists(ptbxl_manifest):
        print(f"❌ PTB-XL manifest 未找到: {ptbxl_manifest}")
        return None
    
    print(f"\n读取 PTB-XL manifest: {ptbxl_manifest}")
    df_ptbxl = pd.read_csv(ptbxl_manifest)
    print(f"  PTB-XL 记录数: {len(df_ptbxl)}")
    
    # 读取 ECG-Arrhythmia manifest
    if not os.path.exists(ecg_arr_manifest):
        print(f"❌ ECG-Arrhythmia manifest 未找到: {ecg_arr_manifest}")
        return None
    
    print(f"\n读取 ECG-Arrhythmia manifest: {ecg_arr_manifest}")
    df_ecg_arr = pd.read_csv(ecg_arr_manifest)
    print(f"  ECG-Arrhythmia 记录数: {len(df_ecg_arr)}")
    
    # 合并
    print(f"\n合并两个数据集...")
    df_combined = pd.concat([df_ptbxl, df_ecg_arr], ignore_index=True)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存
    df_combined.to_csv(output_path, index=False)
    print(f"\n✅ 合并完成!")
    print(f"   输出文件: {output_path}")
    print(f"   总记录数: {len(df_combined)}")
    
    # 显示统计信息
    print(f"\n📊 数据集统计:")
    print(f"   PTB-XL: {len(df_ptbxl)} 条")
    print(f"   ECG-Arrhythmia: {len(df_ecg_arr)} 条")
    print(f"   总计: {len(df_combined)} 条")
    
    print(f"\n📊 标签分布:")
    for label in ["Other", "AF", "AFL", "PSVT"]:
        count = len(df_combined[df_combined['label_raw'] == label])
        pct = count / len(df_combined) * 100 if len(df_combined) > 0 else 0
        print(f"   {label}: {count} ({pct:.1f}%)")
    
    print("="*60)
    
    return output_path

if __name__ == "__main__":
    combine_manifests()







