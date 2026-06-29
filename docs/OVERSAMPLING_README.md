# 数据过采样使用说明

## 概述

为了解决训练数据类别不平衡问题（AF:AFL ≈ 20:1），我们实现了数据过采样功能。可以在训练前对少数类（AFL）进行过采样，平衡类别分布。

## 使用方法

### 方法1：在训练时自动过采样（推荐）

在训练命令中添加 `--use_oversampling` 参数：

```bash
python noise_aware_ecg_af_afl_simple.py \
    --data_csv data/precompute_features/multi_dataset_combined_with_features.csv \
    --use_oversampling \
    --oversample_target_ratio 1.0 \
    --oversample_strategy patient_level \
    --num_arrhythmia_classes 2 \
    --epochs 50 \
    --out_ckpt outputs/afdb_ltafdb_binary_oversampled/af_vs_afl.pt
```

**参数说明：**
- `--use_oversampling`: 启用过采样
- `--oversample_target_ratio`: 目标类别比例
  - `1.0` = 完全平衡（AFL样本数 = AF样本数）
  - `0.5` = 2:1比例（AFL样本数 = AF样本数的一半）
  - `0.3` = 约3:1比例
- `--oversample_strategy`: 过采样策略
  - `random`: 随机过采样（样本级别，简单快速）
  - `patient_level`: 患者级别过采样（更安全，避免数据泄露，推荐）

### 方法2：手动过采样（独立脚本）

如果需要在训练前单独进行过采样：

```bash
# 1. 先划分数据集
python -c "
from noise_aware_ecg_af_afl_simple import split_dataset_csv
import pandas as pd
train_df, val_df, test_df = split_dataset_csv(
    'data/precompute_features/multi_dataset_combined_with_features.csv',
    0.8, 0.1, 0.1, seed=42
)
train_df.to_csv('outputs/temp_train.csv', index=False)
val_df.to_csv('outputs/temp_val.csv', index=False)
test_df.to_csv('outputs/temp_test.csv', index=False)
"

# 2. 对训练集进行过采样
python oversample_train_data.py \
    --input_csv outputs/temp_train.csv \
    --output_csv outputs/temp_train_oversampled.csv \
    --target_ratio 1.0 \
    --strategy patient_level \
    --group_col _record_group \
    --label_col label

# 3. 使用过采样后的训练集进行训练
python noise_aware_ecg_af_afl_simple.py \
    --train_csv outputs/temp_train_oversampled.csv \
    --val_csv outputs/temp_val.csv \
    --num_arrhythmia_classes 2 \
    --epochs 50
```

## 过采样策略对比

### 1. Random（随机过采样）

- **原理**：随机重复少数类样本
- **优点**：简单快速，实现容易
- **缺点**：可能导致过拟合，特别是如果某些样本被重复多次
- **适用场景**：数据量大，过拟合风险低

### 2. Patient-Level（患者级别过采样）

- **原理**：按患者分组，重复整个患者的样本
- **优点**：
  - 避免数据泄露（同一患者的不同样本不会同时出现在训练和验证集）
  - 更符合实际应用场景
  - 减少过拟合风险
- **缺点**：需要数据中有患者ID列（`_record_group`）
- **适用场景**：**推荐使用**，特别是数据有患者分组信息时

## 效果预期

使用过采样后，预期效果：

1. **类别分布更平衡**
   - 原始：AF: 41166, AFL: 2040 (20.2:1)
   - 过采样后（target_ratio=1.0）：AF: 41166, AFL: 41166 (1:1)

2. **训练效果提升**
   - AFL Recall: 从 ~0.3-0.4 提升到 **0.6-0.7**
   - AFL F1: 从 ~0.2-0.3 提升到 **0.5-0.6**
   - Macro F1: 从 ~0.7 提升到 **0.75-0.80**

3. **训练更稳定**
   - 减少类别不平衡导致的训练不稳定
   - 模型更关注少数类样本

## 注意事项

1. **只对训练集过采样**
   - 验证集和测试集**不要**过采样
   - 保持验证集和测试集的原始分布，以真实评估模型性能

2. **患者级别过采样更安全**
   - 如果数据有患者分组信息（`_record_group`列），强烈推荐使用 `patient_level` 策略
   - 避免同一患者的数据泄露到验证集

3. **目标比例选择**
   - `target_ratio=1.0`（完全平衡）：适合严重不平衡的数据
   - `target_ratio=0.5`（2:1）：如果完全平衡导致过拟合，可以尝试部分平衡
   - 根据实际效果调整

4. **与现有技术的配合**
   - 过采样可以与类别权重、加权采样器配合使用
   - 如果使用过采样，可以适当降低类别权重的boost强度

## 示例输出

```
[3/7] Loading datasets...
  - Data CSV: data/precompute_features/multi_dataset_combined_with_features.csv
  - Applying oversampling to training data...

================================================================================
Training Data Oversampling
================================================================================

[1/5] Loading data from: outputs/temp_train_before_oversample.csv
  - Total samples: 43206

[2/5] Analyzing class distribution...
  - Original class distribution:
    Label 0: 41166 samples
    Label 1: 2040 samples
  - Majority class: 0 (41166 samples)
  - Minority class: 1 (2040 samples)
  - Imbalance ratio: 20.18:1

[3/5] Oversampling strategy: patient_level
  - Target ratio: 1.00
  - Target minority samples: 41166
  - Samples to add: 39126

[4/5] Performing oversampling...
  - Using patient-level oversampling (group_col: _record_group)
  - Minority class patients: 7
  - Added 39126 oversampled samples

[5/5] Shuffling and saving data...
  - Final class distribution:
    Label 0: 41166 samples
    Label 1: 41166 samples
  - Final imbalance ratio: 1.00:1

  ✓ Oversampled data saved to: outputs/temp_train_oversampled.csv
    - Original samples: 43206
    - Oversampled samples: 82332
    - Added samples: 39126
================================================================================
Oversampling Complete!
================================================================================
  - Using oversampled training data: 82332 samples
```

## 故障排除

### 问题1：找不到 `_record_group` 列

**错误**：`KeyError: '_record_group'`

**解决**：
- 确保数据CSV中有 `_record_group` 列（患者ID）
- 或者使用 `--oversample_strategy random`（样本级别过采样）

### 问题2：过采样后内存不足

**解决**：
- 降低 `--oversample_target_ratio`（例如从1.0降到0.5）
- 减少batch size
- 使用更少的epochs

### 问题3：过采样后过拟合

**解决**：
- 降低 `--oversample_target_ratio`（例如从1.0降到0.5）
- 增加正则化（weight_decay, dropout等）
- 使用数据增强

## 相关文件

- `oversample_train_data.py` - 独立的过采样脚本
- `noise_aware_ecg_af_afl_simple.py` - 主训练脚本（集成了过采样功能）


