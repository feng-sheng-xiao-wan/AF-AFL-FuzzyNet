# AF vs AFL 消融实验使用指南

本目录包含用于 AF vs AFL 二分类任务的消融实验脚本，用于验证各个组件的有效性。

## 📁 目录结构

```
ablation_Af_AFL/
├── README.md                    # 本文件
├── ablation_utils.py            # 通用消融工具函数
├── 01_baseline_morph_rr.py     # 消融实验 01：仅形态+RR
├── 02_add_noise_branch.py      # 消融实验 02：+ 噪声分支
├── 03_add_fuzzy_rules.py       # 消融实验 03：+ 模糊规则
└── 04_add_boundary_aware.py    # 消融实验 04：+ 边界感知（完整方案）
```

---

## 🎯 消融实验说明

### 实验设计

| 实验编号 | 组件配置 | 说明 |
|---------|---------|------|
| **01** | 形态 + RR | Baseline：仅使用ECG形态特征和RR间期序列 |
| **02** | + 噪声分支 | 添加噪声特征分支（频带能量、SNR、谱熵等） |
| **03** | + 模糊规则 | 添加临床先验驱动的模糊规则系统 |
| **04** | + 边界感知 | 添加边界感知训练机制（完整方案） |

### 组件说明

- **形态分支**：1D CNN 提取 ECG 波形形态特征
- **RR分支**：GRU 编码 RR 间期序列，捕获节律特征
- **噪声分支**：提取频带能量比、SNR、谱熵等噪声特征
- **模糊规则**：基于临床知识的规则系统（RMSSD、CV、flutter证据等）
- **边界感知**：针对难分类样本的增强训练机制

---

## 🚀 快速开始

### 前置条件

1. **数据准备**：推荐使用预计算特征的合并数据集
   - **多数据集合并CSV（推荐）**：`data/precompute_features/multi_dataset_combined_with_features.csv`
     - 包含三个数据集（AFDB、LTAFDB等）的合并数据
     - 已包含预计算特征（RR序列、噪声特征、模糊规则logits）
   - **双数据集合并CSV**：`data/precompute_features/af_afl_2leads/afdb_ltafdb_combined_with_features.csv`
   - **原始数据CSV（向后兼容）**：
     - AFDB CSV：`data/holter/afdb/afdb_segments.csv`
     - LTAFDB CSV：`data/holter/ltafdb/ltafdb_segments.csv`

2. **特征预计算**（如果使用原始数据）：
   ```bash
   python precompute_features.py --csv data/holter/afdb/afdb_segments.csv --output_dir data/precompute_features/afdb
   python precompute_features.py --csv data/holter/ltafdb/ltafdb_segments.csv --output_dir data/precompute_features/ltafdb
   ```

### 运行单个消融实验

#### 方式 1：使用多数据集合并CSV（最推荐，包含三个数据集的预计算特征）

```bash
# 实验 01：Baseline（形态 + RR）
python ablation_Af_AFL/01_baseline_morph_rr.py `
    --combined_csv data/precompute_features/multi_dataset_combined_with_features.csv `
    --output_dir outputs/ablation/01_morph_rr `
    --epochs 30 `
    --batch_size 128 `
    --lr 1e-3 `
    --use_oversampling `
    --oversample_target_ratio 1.0 `
    --oversample_strategy random
```

#### 方式 2：使用双数据集合并CSV（包含预计算特征）

```bash
# 实验 01：Baseline（形态 + RR）
python ablation_Af_AFL/01_baseline_morph_rr.py `
    --combined_csv data/precompute_features/af_afl_2leads/afdb_ltafdb_combined_with_features.csv `
    --output_dir outputs/ablation/01_morph_rr `
    --epochs 30 `
    --batch_size 32 `
    --lr 1e-3 `
    --use_oversampling `
    --oversample_target_ratio 1.0 `
    --oversample_strategy random
```

#### 方式 3：使用分离的CSV（向后兼容）

```bash
# 实验 01：Baseline（形态 + RR）
python ablation_Af_AFL/01_baseline_morph_rr.py `
    --afdb_csv data/holter/afdb/afdb_segments.csv `
    --ltafdb_csv data/holter/ltafdb/ltafdb_segments.csv `
    --output_dir outputs/ablation/01_morph_rr `
    --epochs 30 `
    --batch_size 32 `
    --lr 1e-3 `
    --use_oversampling `
    --oversample_target_ratio 1.0 `
    --oversample_strategy random
```

#### 实验 02：+ 噪声分支

```bash
# 使用多数据集合并CSV（推荐）
python ablation_Af_AFL/02_add_noise_branch.py `
    --combined_csv data/precompute_features/multi_dataset_combined_with_features.csv `
    --output_dir outputs/ablation/02_noise_only `
    --epochs 30 `
    --batch_size 32 `
    --lr 1e-3 `
    --use_oversampling `
    --oversample_target_ratio 1.0
```

#### 实验 03：+ 模糊规则

```bash
# 使用多数据集合并CSV（推荐）
python ablation_Af_AFL/03_add_fuzzy_rules.py `
    --combined_csv data/precompute_features/multi_dataset_combined_with_features.csv `
    --output_dir outputs/ablation/03_fuzzy `
    --epochs 30 `
    --batch_size 32 `
    --lr 1e-3 `
    --use_oversampling `
    --oversample_target_ratio 1.0 `
    --recompute_fuzzy_only  # 如果需要重新计算模糊规则特征
```

#### 实验 04：完整方案（+ 边界感知）

```bash
# 使用多数据集合并CSV（推荐）
python ablation_Af_AFL/04_add_boundary_aware.py `
    --combined_csv data/precompute_features/multi_dataset_combined_with_features.csv `
    --output_dir outputs/ablation/04_boundary `
    --epochs 30 `
    --batch_size 32 `
    --lr 1e-3 `
    --use_oversampling `
    --oversample_target_ratio 1.0
```

### 批量运行所有实验

**Windows PowerShell（推荐使用多数据集合并CSV）：**
```powershell
# 实验 01
python ablation_Af_AFL/01_baseline_morph_rr.py --combined_csv data/precompute_features/multi_dataset_combined_with_features.csv --output_dir outputs/ablation/01_morph_rr

# 实验 02
python ablation_Af_AFL/02_add_noise_branch.py --combined_csv data/precompute_features/multi_dataset_combined_with_features.csv --output_dir outputs/ablation/02_noise_only

# 实验 03
python ablation_Af_AFL/03_add_fuzzy_rules.py --combined_csv data/precompute_features/multi_dataset_combined_with_features.csv --output_dir outputs/ablation/03_fuzzy

# 实验 04
python ablation_Af_AFL/04_add_boundary_aware.py --combined_csv data/precompute_features/multi_dataset_combined_with_features.csv --output_dir outputs/ablation/04_boundary
```

**Linux/Mac（推荐使用多数据集合并CSV）：**
```bash
# 实验 01
python ablation_Af_AFL/01_baseline_morph_rr.py `
    --combined_csv data/precompute_features/multi_dataset_combined_with_features.csv `
    --output_dir outputs/ablation/01_morph_rr

# 实验 02
python ablation_Af_AFL/02_add_noise_branch.py `
    --combined_csv data/precompute_features/multi_dataset_combined_with_features.csv `
    --output_dir outputs/ablation/02_noise_only

# 实验 03
python ablation_Af_AFL/03_add_fuzzy_rules.py `
    --combined_csv data/precompute_features/multi_dataset_combined_with_features.csv `
    --output_dir outputs/ablation/03_fuzzy

# 实验 04
python ablation_Af_AFL/04_add_boundary_aware.py `
    --combined_csv data/precompute_features/multi_dataset_combined_with_features.csv `
    --output_dir outputs/ablation/04_boundary
```

**向后兼容（使用分离的CSV）：**
```bash
# 实验 01
python ablation_Af_AFL/01_baseline_morph_rr.py `
    --afdb_csv data/holter/afdb/afdb_segments.csv `
    --ltafdb_csv data/holter/ltafdb/ltafdb_segments.csv
```

---

## 📊 参数说明

### 数据输入参数（二选一）

| 参数 | 说明 | 示例 |
|------|------|------|
| `--combined_csv` | **推荐**：合并的CSV文件路径（包含预计算特征） | `data/precompute_features/multi_dataset_combined_with_features.csv`（三个数据集）<br>`data/precompute_features/af_afl_2leads/afdb_ltafdb_combined_with_features.csv`（两个数据集） |
| `--afdb_csv` | AFDB 分段 CSV 文件路径（向后兼容） | `data/holter/afdb/afdb_segments.csv` |
| `--ltafdb_csv` | LTAFDB 分段 CSV 文件路径（向后兼容） | `data/holter/ltafdb/ltafdb_segments.csv` |

**注意**：
- **最推荐**：使用 `--combined_csv data/precompute_features/multi_dataset_combined_with_features.csv`（包含三个数据集的预计算特征）
- 必须提供 `--combined_csv` 或同时提供 `--afdb_csv` 和 `--ltafdb_csv`

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_dir` | `outputs/ablation/XX_xxx` | 输出目录（模型权重、日志等） |
| `--epochs` | `30` | 训练轮数 |
| `--batch_size` | `32` | 批次大小 |
| `--lr` | `1e-3` | 学习率 |

### 过采样参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_oversampling` | `True` | 是否使用过采样（使用 `--no_use_oversampling` 禁用） |
| `--oversample_target_ratio` | `1.0` | 过采样目标比例（1.0=完全平衡，0.5=2:1） |
| `--oversample_strategy` | `random` | 过采样策略：`random`（样本级）或 `patient_level`（患者级，更安全） |

### 预计算特征参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_precomputed` | `True` | 是否使用预计算特征（使用 `--no_use_precomputed` 禁用） |
| `--recompute_fuzzy_only` | `False` | 是否仅重新计算模糊规则特征（即使有预计算特征） |

### CSV 文件格式要求

CSV 文件必须包含以下列：

- **`path`**：NPZ 文件路径（必需）
- **`label_raw`**：原始标签，值为 `"AF"` 或 `"AFL"`（必需）
- **`feature_path`**：预计算特征文件路径（可选，但推荐）

示例 CSV：
```csv
path,label_raw,feature_path
data/holter/afdb/00_seg001.npz,AF,data/precompute_features/afdb/00_seg001.npz
data/holter/afdb/00_seg002.npz,AFL,data/precompute_features/afdb/00_seg002.npz
```

---

## 🔧 添加新数据集

### 步骤 1：准备数据

1. **创建分段 CSV 文件**
   - 格式：`path,label_raw`（或包含 `feature_path`）
   - 确保 `label_raw` 列只包含 `"AF"` 或 `"AFL"`

2. **（可选）预计算特征**
   ```bash
   python precompute_features.py `
       --csv your_new_dataset.csv `
       --output_dir data/precompute_features/your_dataset
   ```

### 步骤 2：修改消融脚本

有两种方式添加新数据集：

#### 方式 A：修改 `ablation_utils.py`（推荐）

在 `ablation_utils.py` 中添加新的数据集合并函数：

```python
def combine_multiple_csvs(
    csv_list: List[str],
    output_csv: str,
) -> pd.DataFrame:
    """
    合并多个数据集的 CSV 文件
    
    Args:
        csv_list: CSV 文件路径列表
        output_csv: 输出合并后的 CSV 路径
    """
    dfs = []
    for csv_path in csv_list:
        df = pd.read_csv(csv_path)
        df = df[df['label_raw'].isin(['AF', 'AFL'])].copy()
        dfs.append(df)
    
    df_combined = pd.concat(dfs, ignore_index=True)
    df_combined.to_csv(output_csv, index=False)
    
    print(f"Combined {len(dfs)} datasets: {len(df_combined)} total samples")
    print(f"Label distribution:\n{df_combined['label_raw'].value_counts()}")
    
    return df_combined
```

然后修改 `run_ablation()` 函数签名：

```python
def run_ablation(
    csv_list: List[str],  # 改为列表
    output_dir: str,
    # ... 其他参数
):
    # 合并多个CSV
    os.makedirs(output_dir, exist_ok=True)
    combined_csv = os.path.join(output_dir, "combined_dataset.csv")
    combine_multiple_csvs(csv_list, combined_csv)
    
    # 后续使用 combined_csv
    # ...
```

#### 方式 B：修改每个消融脚本（简单直接）

在每个消融脚本（如 `01_baseline_morph_rr.py`）中添加新数据集参数：

```python
def main():
    parser = argparse.ArgumentParser(description="Ablation 01: Morph + RR only")
    parser.add_argument("--afdb_csv", type=str, required=True)
    parser.add_argument("--ltafdb_csv", type=str, required=True)
    parser.add_argument("--new_dataset_csv", type=str, default=None, help="新数据集 CSV（可选）")
    # ... 其他参数
    
    # 合并数据集
    csv_list = [args.afdb_csv, args.ltafdb_csv]
    if args.new_dataset_csv:
        csv_list.append(args.new_dataset_csv)
    
    # 创建临时合并CSV
    import pandas as pd
    dfs = []
    for csv_path in csv_list:
        df = pd.read_csv(csv_path)
        df = df[df['label_raw'].isin(['AF', 'AFL'])].copy()
        dfs.append(df)
    df_combined = pd.concat(dfs, ignore_index=True)
    combined_csv = os.path.join(args.output_dir, "combined.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    df_combined.to_csv(combined_csv, index=False)
    
    # 修改 run_ablation 调用，使用合并后的CSV
    # 注意：需要修改 ablation_utils.py 以支持单个CSV输入
```

### 步骤 3：运行实验

使用新数据集运行消融实验：

```bash
python ablation_Af_AFL/01_baseline_morph_rr.py `
    --afdb_csv data/holter/afdb/afdb_segments.csv `
    --ltafdb_csv data/holter/ltafdb/ltafdb_segments.csv `
    --new_dataset_csv data/holter/your_dataset/your_dataset_segments.csv `
    --output_dir ablation_Af_AFL/ablation/01_morph_rr_with_new_dataset
```

---

## 📈 结果解读

### 输出文件结构

每个实验的输出目录包含：

```
ablation_Af_AFL/ablation/XX_xxx/
├── afdb_ltafdb_combined.csv    # 合并后的数据集CSV
├── af_vs_afl_afdb_ltafdb.pt    # 训练好的模型权重
└── training_log.txt            # 训练日志（如果有）
```

### 关键指标

训练过程中会输出以下指标：

- **Accuracy**：整体准确率
- **F1 Score**：F1 分数（AF 和 AFL 的平均）
- **Precision (AF)**：AF 类别的精确率
- **Recall (AF)**：AF 类别的召回率
- **Precision (AFL)**：AFL 类别的精确率
- **Recall (AFL)**：AFL 类别的召回率
- **Confusion Matrix**：混淆矩阵

### 消融分析

对比不同实验的结果：

| 实验 | Accuracy | F1 (AF) | F1 (AFL) | 说明 |
|------|----------|---------|----------|------|
| 01 (Baseline) | X.XX | X.XX | X.XX | 仅形态+RR |
| 02 (+ Noise) | X.XX | X.XX | X.XX | + 噪声分支 |
| 03 (+ Fuzzy) | X.XX | X.XX | X.XX | + 模糊规则 |
| 04 (Full) | X.XX | X.XX | X.XX | + 边界感知 |

**分析要点**：
- 每个组件对性能的提升（Δ）
- AFL 类别的召回率变化（AFL 样本少，容易漏检）
- 混淆矩阵中 AF↔AFL 的误分类数量

---

## 🐛 常见问题

### Q1: 找不到 CSV 文件

**错误**：`FileNotFoundError: [Errno 2] No such file or directory: 'data/holter/afdb/afdb_segments.csv'`

**解决**：
1. 检查 CSV 文件路径是否正确
2. 使用绝对路径或确保相对路径正确
3. 检查文件是否存在：`ls data/holter/afdb/afdb_segments.csv`（Linux/Mac）或 `dir data\holter\afdb\afdb_segments.csv`（Windows）

### Q2: 内存不足（OOM）

**错误**：`RuntimeError: CUDA out of memory`

**解决**：
1. 减小 `batch_size`：`--batch_size 16` 或 `--batch_size 8`
2. 使用 CPU：`--device cpu`（如果支持）
3. 减少 `max_len`：修改 `train_af_vs_afl_afdb_ltafdb.py` 中的 `max_len` 参数

### Q3: 训练速度慢

**解决**：
1. 使用预计算特征：运行 `precompute_features.py`
2. 减小 `epochs`：`--epochs 20`
3. 使用 GPU：确保 CUDA 可用
4. 增加 `num_workers`：在 `ablation_utils.py` 中修改（Windows 可能不支持）

### Q4: 标签分布不平衡

**现象**：AFL 样本很少（如 1917 vs 49144 AF）

**解决**：
1. **使用过采样**（推荐）：`--use_oversampling --oversample_target_ratio 1.0`（默认已启用）
2. 使用类别权重：`--use_class_weights`（默认已启用）
3. 调整过采样策略：`--oversample_strategy patient_level`（患者级过采样，更安全）
4. 调整过采样比例：`--oversample_target_ratio 0.5`（2:1 比例，而非完全平衡）

### Q5: 如何只使用部分数据集？

**方法**：创建子集 CSV

```python
import pandas as pd

# 读取原始CSV
df = pd.read_csv('data/holter/afdb/afdb_segments.csv')

# 按标签采样
df_af = df[df['label_raw'] == 'AF'].sample(n=5000)  # 采样5000个AF
df_afl = df[df['label_raw'] == 'AFL']  # 全部AFL

# 合并
df_subset = pd.concat([df_af, df_afl], ignore_index=True)
df_subset.to_csv('data/holter/afdb/afdb_segments_subset.csv', index=False)
```

然后使用子集 CSV：
```bash
python ablation_Af_AFL/01_baseline_morph_rr.py `
    --afdb_csv data/holter/afdb/afdb_segments_subset.csv `
    --ltafdb_csv data/holter/ltafdb/ltafdb_segments.csv
```

---

## 📝 实验记录模板

建议为每次实验记录以下信息：

```markdown
## 实验记录 - [日期]

### 实验配置
- 数据集：AFDB + LTAFDB
- 训练轮数：30
- 批次大小：32
- 学习率：1e-3

### 结果

| 实验 | Accuracy | F1 (AF) | F1 (AFL) | Precision (AFL) | Recall (AFL) |
|------|----------|---------|----------|-----------------|-------------|
| 01   |          |         |          |                 |             |
| 02   |          |         |          |                 |             |
| 03   |          |         |          |                 |             |
| 04   |          |         |          |                 |             |

### 观察
- [ ] 噪声分支提升了 X%
- [ ] 模糊规则提升了 X%
- [ ] 边界感知提升了 X%
- [ ] AFL 召回率从 X% 提升到 Y%
```

---

## 🔗 相关文件

- **主训练脚本**：`train_af_vs_afl_afdb_ltafdb.py`
- **基础模型**：`noise_aware_ecg_af_afl_simple.py`
- **特征预计算**：`precompute_features.py`
- **技术路线文档**：`技术路线_AF_AFL二分类.md`

---

## 📧 问题反馈

如有问题或建议，请检查：
1. CSV 文件格式是否正确
2. NPZ 文件路径是否可访问
3. 依赖包是否安装完整
4. 日志文件中的错误信息

---

---

## 🆕 更新日志

### 2024-XX-XX：添加新功能

- ✅ **过采样支持**：添加 `--use_oversampling`、`--oversample_target_ratio`、`--oversample_strategy` 参数
- ✅ **合并CSV支持**：添加 `--combined_csv` 参数，支持使用预计算特征的合并CSV文件
- ✅ **预计算特征控制**：添加 `--use_precomputed` 和 `--recompute_fuzzy_only` 参数
- ✅ **PyTorch 2.6兼容性**：修复模型加载兼容性问题
- ✅ **每类指标输出**：训练过程中输出每个类别的详细指标（Recall、Precision、F1）

**最后更新**：2024-XX-XX
