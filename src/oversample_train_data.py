#!/usr/bin/env python3
"""
训练数据过采样脚本

对训练集中的少数类进行过采样，平衡类别分布。
支持基于患者级别的过采样（避免数据泄露）。

用法: python oversample_train_data.py --input_csv <训练CSV> --output_csv <输出CSV> [选项]
"""

import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path


def oversample_train_data(
    input_csv: str,
    output_csv: str,
    target_ratio: float = 1.0,
    group_col: str = "_record_group",
    label_col: str = "label",
    random_seed: int = 42,
    strategy: str = "random",
    verbose: bool = True,
):
    """
    对训练数据进行过采样
    
    Args:
        input_csv: 输入的训练CSV文件
        output_csv: 输出的过采样后CSV文件
        target_ratio: 目标类别比例（1.0表示完全平衡，0.5表示2:1，等等）
        group_col: 用于分组的列（通常是患者ID，避免数据泄露）
        label_col: 标签列名
        random_seed: 随机种子
        strategy: 过采样策略（'random' 或 'patient_level'）
    
    Returns:
        str: 输出CSV路径
    """
    if verbose:
        print("=" * 80)
        print("Training Data Oversampling")
        print("=" * 80)
    
    # 读取数据
    if verbose:
        print(f"\n[1/5] Loading data from: {input_csv}")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    
    df = pd.read_csv(input_csv)
    if verbose:
        print(f"  - Total samples: {len(df)}")
    
    # 检查必要的列
    if label_col not in df.columns:
        raise ValueError(f"CSV must contain '{label_col}' column")
    
    # 分析类别分布
    if verbose:
        print(f"\n[2/5] Analyzing class distribution...")
    label_counts = df[label_col].value_counts().sort_index()
    if verbose:
        print(f"  - Original class distribution:")
        for label, count in label_counts.items():
            print(f"    Label {label}: {count} samples")
    
    if len(label_counts) != 2:
        if verbose:
            print(f"  Warning: Expected 2 classes for binary classification, found {len(label_counts)}")
            print(f"  Will oversample the minority class to match majority class.")
    
    # 识别多数类和少数类
    majority_label = label_counts.idxmax()
    minority_label = label_counts.idxmin()
    majority_count = label_counts[majority_label]
    minority_count = label_counts[minority_label]
    imbalance_ratio = majority_count / minority_count
    
    if verbose:
        print(f"  - Majority class: {majority_label} ({majority_count} samples)")
        print(f"  - Minority class: {minority_label} ({minority_count} samples)")
        print(f"  - Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    # 计算目标样本数
    if target_ratio == 1.0:
        target_minority_count = majority_count  # 完全平衡
    else:
        target_minority_count = int(majority_count * target_ratio)  # 部分平衡
    
    samples_to_add = target_minority_count - minority_count
    if verbose:
        print(f"\n[3/5] Oversampling strategy: {strategy}")
        print(f"  - Target ratio: {target_ratio:.2f}")
        print(f"  - Target minority samples: {target_minority_count}")
        print(f"  - Samples to add: {samples_to_add}")
    
    if samples_to_add <= 0:
        if verbose:
            print(f"  - No oversampling needed (minority class already has enough samples)")
        df.to_csv(output_csv, index=False)
        if verbose:
            print(f"\n  ✓ Data saved to: {output_csv} (no changes)")
        return output_csv
    
    # 分离多数类和少数类
    majority_df = df[df[label_col] == majority_label].copy()
    minority_df = df[df[label_col] == minority_label].copy()
    
    # 过采样少数类
    if verbose:
        print(f"\n[4/5] Performing oversampling...")
    
    if strategy == "patient_level" and group_col in df.columns:
        # 基于患者级别的过采样（更安全，避免数据泄露）
        if verbose:
            print(f"  - Using patient-level oversampling (group_col: {group_col})")
        
        # 获取少数类的患者列表
        minority_patients = minority_df[group_col].unique()
        if verbose:
            print(f"  - Minority class patients: {len(minority_patients)}")
        
        # 计算每个患者需要被重复的次数
        samples_per_patient = minority_df.groupby(group_col).size()
        total_minority_samples = len(minority_df)
        
        # 计算需要添加的样本数
        if samples_to_add > 0:
            # 随机选择患者进行重复
            np.random.seed(random_seed)
            oversampled_rows = []
            
            # 计算每个患者应该被重复的次数
            # 策略：按比例重复，确保每个患者的样本都被均匀过采样
            repeat_times = samples_to_add // total_minority_samples
            remainder = samples_to_add % total_minority_samples
            
            # 对每个患者的所有样本重复
            for patient_id in minority_patients:
                patient_samples = minority_df[minority_df[group_col] == patient_id]
                
                # 基础重复次数
                for _ in range(repeat_times):
                    oversampled_rows.append(patient_samples)
                
                # 随机选择部分患者进行额外重复（处理余数）
                if remainder > 0 and np.random.rand() < (remainder / total_minority_samples):
                    oversampled_rows.append(patient_samples)
                    remainder -= len(patient_samples)
            
            # 如果还有余数，随机选择一些样本
            if remainder > 0:
                additional_samples = minority_df.sample(n=min(remainder, len(minority_df)), 
                                                        random_state=random_seed, 
                                                        replace=True)
                oversampled_rows.append(additional_samples)
            
            if oversampled_rows:
                oversampled_minority = pd.concat(oversampled_rows, ignore_index=True)
                if verbose:
                    print(f"  - Added {len(oversampled_minority)} oversampled samples")
            else:
                oversampled_minority = pd.DataFrame()
        else:
            oversampled_minority = pd.DataFrame()
        
        # 合并
        oversampled_df = pd.concat([majority_df, minority_df, oversampled_minority], ignore_index=True)
        
    else:
        # 简单的随机过采样（样本级别）
        print(f"  - Using random oversampling (sample-level)")
        np.random.seed(random_seed)
        
        # 随机采样少数类样本（有放回）
        oversampled_minority = minority_df.sample(
            n=samples_to_add,
            replace=True,
            random_state=random_seed
        )
        
        # 合并
        oversampled_df = pd.concat([majority_df, minority_df, oversampled_minority], ignore_index=True)
    
    # 打乱数据
    if verbose:
        print(f"\n[5/5] Shuffling and saving data...")
    oversampled_df = oversampled_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # 验证结果
    final_label_counts = oversampled_df[label_col].value_counts().sort_index()
    if verbose:
        print(f"  - Final class distribution:")
        for label, count in final_label_counts.items():
            print(f"    Label {label}: {count} samples")
    
    final_ratio = final_label_counts.max() / final_label_counts.min()
    if verbose:
        print(f"  - Final imbalance ratio: {final_ratio:.2f}:1")
    
    # 保存
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    oversampled_df.to_csv(output_csv, index=False)
    if verbose:
        # 避免在 GBK 控制台中输出无法编码的特殊符号
        print(f"\n  Oversampled data saved to: {output_csv}")
        print(f"    - Original samples: {len(df)}")
        print(f"    - Oversampled samples: {len(oversampled_df)}")
        print(f"    - Added samples: {len(oversampled_df) - len(df)}")
        print("\n" + "=" * 80)
        print("Oversampling Complete!")
        print("=" * 80)
    
    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description="Oversample training data to balance class distribution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Input training CSV file",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output oversampled CSV file",
    )
    parser.add_argument(
        "--target_ratio",
        type=float,
        default=1.0,
        help="Target class ratio (1.0 = fully balanced, 0.5 = 2:1 ratio)",
    )
    parser.add_argument(
        "--group_col",
        type=str,
        default="_record_group",
        help="Column name for patient grouping (for patient-level oversampling)",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="Column name for labels",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="random",
        choices=["random", "patient_level"],
        help="Oversampling strategy: 'random' (sample-level) or 'patient_level' (safer, avoids data leakage)",
    )
    
    args = parser.parse_args()
    
    oversample_train_data(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        target_ratio=args.target_ratio,
        group_col=args.group_col,
        label_col=args.label_col,
        random_seed=args.random_seed,
        strategy=args.strategy,
    )


if __name__ == "__main__":
    main()

