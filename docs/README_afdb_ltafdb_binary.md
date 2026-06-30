# AF vs AFL 二分类训练（多数据集：AFDB + LTAFDB + MIT-BIH）

基于 `noise_aware_ecg_af_afl_simple.py` 的二分类训练脚本，使用合并后的多数据集（AFDB、LTAFDB、MIT-BIH）进行 AF vs AFL 二分类任务。

**数据集信息**：
- 总样本数：52,248
- 数据集组成：AFDB (85.7%) + LTAFDB (12.0%) + MIT-BIH (2.3%)
- 类别分布：AF (96.1%) vs AFL (3.9%)
- 已包含预计算特征，可直接用于训练

## 文件说明

1. **`process_ltafdb_segments.py`**: 处理 LTAFDB 数据集，将其拆分为 10 秒片段
2. **`train_af_vs_afl_afdb_ltafdb.py`**: 二分类训练脚本，合并 AFDB 和 LTAFDB 数据进行训练

## 使用步骤

### 步骤 1: 运行二分类训练（使用已合并的数据集）

使用合并后的多数据集（包含 AFDB、LTAFDB、MIT-BIH）进行训练：

```bash
python train_af_vs_afl_afdb_ltafdb.py `
    --combined_csv "data/precompute_features/multi_dataset_combined_with_features.csv" `
    --output_dir outputs_multi_dataset `
    --num_leads 2 `
    --epochs 40 `
    --batch_size 128 `
    --lr 1e-3 `
    --multi_lead_method "voting" `
    --flutter_method "attention" `
    --use_boundary_aware `
    --boundary_alpha 0.6 `
    --device auto `
    --use_class_weights `
    --low_conf_alpha 1.5 `
    --weight_conf_gamma 0.5 `
    --lambda_af_bin 1.5
```

主要参数：
- `--combined_csv`: 合并后的数据集 CSV 路径（包含 AFDB、LTAFDB、MIT-BIH，已包含预计算特征）
- `--output_dir`: 输出目录
- `--num_leads`: 导联数，默认 2
- `--batch_size`: 批次大小，默认 128
- `--epochs`: 训练轮数，默认 40
- `--lr`: 学习率，默认 1e-3
- `--device`: 设备（auto/cpu/cuda），默认 auto
- `--multi_lead_method`: 多导联R峰检测方法（"max_energy" 或 "voting"）
- `--flutter_method`: Flutter检测方法（"fixed_lead" 或 "attention"）
- `--use_class_weights`: 使用类别权重（处理类别不平衡，强烈推荐）
- `--low_conf_alpha`: 低置信度样本权重增强系数（推荐 1.5）
- `--weight_conf_gamma`: 置信度权重指数（推荐 0.5）
- `--lambda_af_bin`: AF/AFL二分类辅助头权重（推荐 1.5）

其他训练参数（继承自基础脚本）：
- `--use_stft_dbscan`: 启用 STFT+DBSCAN 噪声特征（默认关闭）
- `--use_class_weights`: 使用类别权重（默认启用）
- `--fuzzy_weight_init`: 模糊规则融合权重初始值（默认 0.3）
- `--use_boundary_aware`: 启用边界感知训练（默认启用）
- `--use_precomputed`: 使用预计算特征（默认启用）

## 技术路线

与原始 `noise_aware_ecg_af_afl_simple.py` 保持一致：

1. **三分支模型**：
   - 形态分支：ECG 1D CNN
   - 节律分支：RR 间期序列 + GRU
   - 噪声分支：频带能量比例、谱熵、SNR 等

2. **模糊规则系统**：
   - 基于临床 ECG 知识库
   - 多特征组合：RMSSD, CV, pNN50（不规则性）；Flutter band, 心房率（AFL证据）
   - 针对二分类，只使用 AF 和 AFL 的模糊规则输出

3. **可学习融合**：
   - 深度学习输出与模糊规则输出通过可学习 gating 机制融合

## 数据格式

### 输入 CSV 格式

合并后的 CSV 文件（`multi_dataset_combined_with_features.csv`）应包含以下列：
- `path`: .npz 文件路径
- `label_raw`: 原始标签（'AF' 或 'AFL'）
- `record_name`: 记录名称
- `segment_idx`: 片段索引
- `start_idx`: 起始索引
- `end_idx`: 结束索引
- `dataset`: 数据集名称（'AFDB'、'LTAFDB' 或 'MIT-BIH'）
- `feature_path`: 预计算特征文件路径（包含 rr_seq、noise_feat、fuzzy_logits）

### NPZ 文件格式

每个 .npz 文件应包含：
- `ecg`: ECG 信号数组，形状 (T,) 或 (T, C)
- `fs`: 采样率（float，默认 250.0）

## 输出

训练完成后，会在 `output_dir` 目录下生成：
- `af_vs_afl_afdb_ltafdb.pt`: 最佳模型检查点
- `afdb_ltafdb_combined.csv`: 合并后的数据集 CSV

## 注意事项

1. **数据集组成**：合并数据集包含 AFDB、LTAFDB 和 MIT-BIH，共 52,248 样本
2. **类别不平衡**：AF:AFL 比例约为 24:1，已通过增强的类别权重和低置信度关注机制处理
3. **预计算特征**：数据集已包含预计算特征（rr_seq、noise_feat、fuzzy_logits），训练速度更快
4. **标签映射**：AF 映射为 0，AFL 映射为 1
5. **模糊规则**：输出二分类 logits（AF 和 AFL），与训练标签一致
6. **Windows 系统**：建议设置 `--num_workers 0` 以避免多进程问题
7. **训练策略**：
   - 使用 `--use_class_weights` 处理类别不平衡
   - 使用 `--low_conf_alpha 1.5` 关注低置信度样本（通常是AFL样本）
   - 使用 `--weight_conf_gamma 0.5` 降低置信度权重影响

## 示例完整流程

### 方式1: 使用已合并的数据集（推荐）

如果已有合并好的数据集（包含预计算特征）：
python noise_aware_ecg_af_afl_simple.py `
    --data_csv data/precompute_features/multi_dataset_combined_with_features.csv `
    --num_arrhythmia_classes 2 `
    --use_oversampling `
    --oversample_target_ratio 1.0 `
    --oversample_strategy patient_level `
    --output_dir outputs/afdb_ltafdb_binary_oversampled `
    --out_ckpt outputs/afdb_ltafdb_binary_oversampled/af_vs_afl_afdb_ltafdb.pt `
    --epochs 50 `
    --batch_size 128
    --lr 1e-3 `
    --device auto `
    --early_stop_patience 10 `
    --early_stop_min_delta 0.001
```bash
# 直接使用合并后的数据集进行训练
python train_af_vs_afl_afdb_ltafdb.py `
    --combined_csv "data/precompute_features/multi_dataset_combined_with_features.csv" `
    --output_dir outputs_multi_dataset `
    --num_leads 2 `
    --epochs 40 `
    --batch_size 128 `
    --lr 1e-3 `
    --multi_lead_method "voting" `
    --flutter_method "attention" `
    --use_boundary_aware `
    --boundary_alpha 0.6 `
    --device auto `
    --use_class_weights `
    --low_conf_alpha 1.5 `
    --weight_conf_gamma 0.5 `
    --lambda_af_bin 1.5
```

### 方式2: 分别处理数据集（如果需要重新处理）

如果需要重新处理数据：

```bash
# 1. 处理 LTAFDB
python process_ltafdb_segments.py `
    --ltafdb_dir data/ltafdb `
    --output_dir data/holter/ltafdb/ltafdb_segments `
    --output_csv data/holter/ltafdb/ltafdb_segments.csv

# 2. 训练模型（使用分离的CSV）
python train_af_vs_afl_afdb_ltafdb.py `
    --afdb_csv data/holter/afdb/afdb_segments.csv `
    --ltafdb_csv data/holter/ltafdb/ltafdb_segments.csv `
    --output_dir outputs/afdb_ltafdb_binary `
    --batch_size 32 `
    --epochs 40 `
    --lr 1e-3 `
    --num_workers 0
```
