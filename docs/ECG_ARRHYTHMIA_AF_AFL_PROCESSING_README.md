# ECG Arrhythmia AF/AFL 数据处理流程

本目录包含处理ECG Arrhythmia数据集中AF和AFL数据的完整流程，分为三个步骤。

**注意：** 所有ECG Arrhythmia相关的输出文件都统一存放在 `data/precompute_features/ablation_AF_AFL/` 目录下。

## 文件说明

- `step1_extract_ecg_arrhythmia_af_afl.py` - 第一步：从ECG Arrhythmia数据集中提取AF和AFL数据
- `step2_precompute_features_ecg_arrhythmia.py` - 第二步：为提取的数据预计算特征
- `step3_test_ecg_arrhythmia.py` - 第三步：在ECG Arrhythmia数据上测试保存的模型

## 快速开始

### 步骤1：提取AF和AFL数据

```bash
python step1_extract_ecg_arrhythmia_af_afl.py
```

**输出：**
- `data/precompute_features/ablation_AF_AFL/ecg_arrhythmia_af_afl.csv` - 只包含AF和AFL样本的CSV文件

### 步骤2：预计算特征

```bash
python step2_precompute_features_ecg_arrhythmia.py
```

**输出：**
- `data/precompute_features/ablation_AF_AFL/features/` - 预计算特征文件目录
- `data/precompute_features/ablation_AF_AFL/ecg_arrhythmia_af_afl_with_features.csv` - 包含特征路径的CSV文件

### 步骤3：测试模型

```bash
python step3_test_ecg_arrhythmia.py `
    --checkpoint outputs/afdb_ltafdb_binary_oversampled/af_vs_afl_afdb_ltafdb.pt `
    --recompute_fuzzy_only
```

## 参数说明

### step1_extract_ecg_arrhythmia_af_afl.py

- `--input_csv`: 输入的ECG Arrhythmia manifest CSV文件（默认：`data/slices_ecg_arrhythmia_2leads_II_V1/windows_manifest_ecg_arrhythmia.csv`）
- `--output_csv`: 输出的CSV文件路径（默认：`data/precompute_features/ablation_AF_AFL/ecg_arrhythmia_af_afl.csv`）

### step2_precompute_features_ecg_arrhythmia.py

- `--csv`: 输入CSV（从步骤1输出，默认：`data/precompute_features/ablation_AF_AFL/ecg_arrhythmia_af_afl.csv`）
- `--output_dir`: 特征输出目录（默认：`data/precompute_features/ablation_AF_AFL/features`）
- `--max_rr_intervals`: 最大RR间期数量（默认：32）
- `--lead_names`: 导联名称（ECG Arrhythmia使用II和V1，默认：`II V1`）
- `--use_stft_dbscan`: 启用STFT+DBSCAN噪声检测（默认：False）
- `--multi_lead_method`: 多导联RR检测方法（`max_energy` 或 `voting`，默认：`max_energy`）
- `--flutter_method`: 多导联flutter检测方法（`fixed_lead` 或 `attention`，默认：`fixed_lead`）

### step3_test_ecg_arrhythmia.py

- `--checkpoint`: 保存的模型checkpoint路径（必需）
- `--test_csv`: 测试CSV（从步骤2输出，默认：`data/precompute_features/ablation_AF_AFL/ecg_arrhythmia_af_afl_with_features.csv`）
- `--device`: 设备（`auto`, `cpu`, 或 `cuda:0`，默认：`auto`）
- `--batch_size`: 批次大小（默认：128）
- `--num_workers`: 数据加载工作进程数（默认：0）
- `--recompute_fuzzy_only`: 只重新计算模糊规则特征（推荐）
- `--no_use_precomputed`: 完全禁用预计算特征

## 输出文件结构

所有ECG Arrhythmia相关文件都统一存放在 `data/precompute_features/ablation_AF_AFL/` 目录下：

```
data/precompute_features/
└── ablation_AF_AFL/                        # ECG Arrhythmia专用目录
    ├── ecg_arrhythmia_af_afl.csv           # 步骤1输出：提取的AF/AFL数据
    ├── ecg_arrhythmia_af_afl_with_features.csv  # 步骤2输出：包含特征路径的CSV
    └── features/                            # 步骤2输出：特征文件目录
        ├── JS00001_hr_0.000_10.000_features.npz
        ├── JS00005_hr_0.000_10.000_features.npz
        └── ...
```

## 完整命令示例

### 完整流程（分步执行）

```bash
# 步骤1：提取AF和AFL
python step1_extract_ecg_arrhythmia_af_afl.py

# 步骤2：预计算特征
python step2_precompute_features_ecg_arrhythmia.py

# 步骤3：测试模型（只重新计算模糊规则）
python step3_test_ecg_arrhythmia.py `
    --checkpoint outputs/afdb_ltafdb_binary_oversampled/af_vs_afl_afdb_ltafdb.pt `
    --recompute_fuzzy_only
```

## 注意事项

1. **数据格式**：ECG Arrhythmia数据使用II和V1两个导联，确保`--lead_names`参数正确设置。

2. **特征预计算**：步骤2可能需要较长时间，取决于数据量。

3. **模型兼容性**：确保测试的模型checkpoint与ECG Arrhythmia数据的配置兼容（特别是导联数量和特征维度）。

4. **内存使用**：如果数据量很大，可能需要调整`--batch_size`参数以避免内存不足。





