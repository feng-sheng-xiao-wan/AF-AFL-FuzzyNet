# -*- coding: utf-8 -*-
"""
从AFDB数据集中挑选测试集，确保AF和AFL都有

用法：
    python ablation_Af_AFL/create_afdb_test_set.py \
        --input_csv data/holter/afdb/combined_af_afl.csv \
        --output_csv data/holter/afdb/afdb_test_set.csv \
        --num_af 200 \
        --num_afl 200 \
        --seed 42
"""

import argparse
import os
import pandas as pd
import numpy as np


def create_test_set(
    input_csv: str,
    output_csv: str,
    num_af: int = 200,
    num_afl: int = 200,
    seed: int = 42,
):
    """
    从AFDB数据集中挑选测试集
    
    Args:
        input_csv: 输入的AFDB CSV文件
        output_csv: 输出的测试集CSV文件
        num_af: 挑选的AF样本数量
        num_afl: 挑选的AFL样本数量
        seed: 随机种子
    """
    print("=" * 80)
    print("Creating AFDB Test Set")
    print("=" * 80)
    
    # 读取数据
    print(f"\n[1/3] Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"  - Total samples: {len(df)}")
    print(f"  - Label distribution:")
    print(df['label_raw'].value_counts())
    
    # 分离AF和AFL
    df_af = df[df['label_raw'] == 'AF'].copy()
    df_afl = df[df['label_raw'] == 'AFL'].copy()
    
    print(f"\n[2/3] Sampling test set...")
    print(f"  - AF samples available: {len(df_af)}")
    print(f"  - AFL samples available: {len(df_afl)}")
    print(f"  - Requested: {num_af} AF, {num_afl} AFL")
    
    # 检查是否有足够的样本
    if len(df_af) < num_af:
        print(f"  Warning: Only {len(df_af)} AF samples available, using all")
        num_af = len(df_af)
    
    if len(df_afl) < num_afl:
        print(f"  Warning: Only {len(df_afl)} AFL samples available, using all")
        num_afl = len(df_afl)
    
    # 随机采样
    np.random.seed(seed)
    if num_af > 0:
        af_indices = np.random.choice(len(df_af), size=num_af, replace=False)
        df_af_test = df_af.iloc[af_indices].copy()
    else:
        df_af_test = pd.DataFrame()
    
    if num_afl > 0:
        afl_indices = np.random.choice(len(df_afl), size=num_afl, replace=False)
        df_afl_test = df_afl.iloc[afl_indices].copy()
    else:
        df_afl_test = pd.DataFrame()
    
    # 合并
    df_test = pd.concat([df_af_test, df_afl_test], ignore_index=True)
    
    # 打乱顺序
    df_test = df_test.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print(f"\n[3/3] Saving test set to: {output_csv}")
    print(f"  - Test set size: {len(df_test)}")
    print(f"  - Label distribution:")
    print(df_test['label_raw'].value_counts())
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # 保存
    df_test.to_csv(output_csv, index=False)
    
    print(f"\n[OK] Test set created successfully!")
    print("=" * 80)
    
    return df_test


def main():
    parser = argparse.ArgumentParser(
        description="Create test set from AFDB dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input_csv",
        type=str,
        default="data/holter/afdb/combined_af_afl.csv",
        help="Input AFDB CSV file",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/holter/afdb/afdb_test_set.csv",
        help="Output test set CSV file",
    )
    parser.add_argument(
        "--num_af",
        type=int,
        default=200,
        help="Number of AF samples to select",
    )
    parser.add_argument(
        "--num_afl",
        type=int,
        default=200,
        help="Number of AFL samples to select",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    create_test_set(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        num_af=args.num_af,
        num_afl=args.num_afl,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

