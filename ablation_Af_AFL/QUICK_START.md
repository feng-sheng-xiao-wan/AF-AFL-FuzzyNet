# 快速参考卡片

## 🚀 一键运行所有实验

### Windows PowerShell

```powershell
# 设置数据路径
$AFDB_CSV = "data/holter/afdb/afdb_segments.csv"
$LTAFDB_CSV = "data/holter/ltafdb/ltafdb_segments.csv"

# 实验 01: Baseline
python ablation_Af_AFL/01_baseline_morph_rr.py --afdb_csv $AFDB_CSV --ltafdb_csv $LTAFDB_CSV

# 实验 02: + Noise
python ablation_Af_AFL/02_add_noise_branch.py --afdb_csv $AFDB_CSV --ltafdb_csv $LTAFDB_CSV

# 实验 03: + Fuzzy
python ablation_Af_AFL/03_add_fuzzy_rules.py --afdb_csv $AFDB_CSV --ltafdb_csv $LTAFDB_CSV

# 实验 04: Full
python ablation_Af_AFL/04_add_boundary_aware.py --afdb_csv $AFDB_CSV --ltafdb_csv $LTAFDB_CSV
```

### Linux/Mac

```bash
# 设置数据路径
export AFDB_CSV="data/holter/afdb/afdb_segments.csv"
export LTAFDB_CSV="data/holter/ltafdb/ltafdb_segments.csv"

# 实验 01: Baseline
python ablation_Af_AFL/01_baseline_morph_rr.py --afdb_csv $AFDB_CSV --ltafdb_csv $LTAFDB_CSV

# 实验 02: + Noise
python ablation_Af_AFL/02_add_noise_branch.py --afdb_csv $AFDB_CSV --ltafdb_csv $LTAFDB_CSV

# 实验 03: + Fuzzy
python ablation_Af_AFL/03_add_fuzzy_rules.py --afdb_csv $AFDB_CSV --ltafdb_csv $LTAFDB_CSV

# 实验 04: Full
python ablation_Af_AFL/04_add_boundary_aware.py --afdb_csv $AFDB_CSV --ltafdb_csv $LTAFDB_CSV
```

---

## 📋 实验配置对比

| 实验 | 形态 | RR | 噪声 | 模糊规则 | 边界感知 | 输出目录 |
|------|:----:|:--:|:----:|:--------:|:--------:|---------|
| 01   | ✅   | ✅ | ❌   | ❌       | ❌       | `outputs/ablation/01_morph_rr` |
| 02   | ✅   | ✅ | ✅   | ❌       | ❌       | `outputs/ablation/02_noise_only` |
| 03   | ✅   | ✅ | ✅   | ✅       | ❌       | `outputs/ablation/03_fuzzy` |
| 04   | ✅   | ✅ | ✅   | ✅       | ✅       | `outputs/ablation/04_boundary` |

---

## 🔧 常用参数

```bash
# 基本参数
--afdb_csv <path>          # AFDB CSV 路径（必需）
--ltafdb_csv <path>        # LTAFDB CSV 路径（必需）
--output_dir <path>         # 输出目录（默认：outputs/ablation/XX_xxx）

# 训练参数
--epochs <int>              # 训练轮数（默认：30）
--batch_size <int>          # 批次大小（默认：32）
--lr <float>                # 学习率（默认：1e-3）
```

---

## 📊 添加新数据集（3步）

### 步骤 1：准备 CSV
```csv
path,label_raw
your_data/seg001.npz,AF
your_data/seg002.npz,AFL
```

### 步骤 2：修改脚本（添加参数）
在 `01_baseline_morph_rr.py` 等文件中添加：
```python
parser.add_argument("--new_dataset_csv", type=str, default=None)
```

### 步骤 3：运行
```bash
python ablation_Af_AFL/01_baseline_morph_rr.py \
    --afdb_csv data/holter/afdb/afdb_segments.csv \
    --ltafdb_csv data/holter/ltafdb/ltafdb_segments.csv \
    --new_dataset_csv your_data/your_dataset.csv
```

---

## 📈 结果查看

训练完成后，检查输出目录：
```
outputs/ablation/XX_xxx/
├── afdb_ltafdb_combined.csv    # 合并数据集
├── af_vs_afl_afdb_ltafdb.pt    # 模型权重
└── training_log.txt            # 训练日志
```

---

## ⚡ 故障排除

| 问题 | 解决方案 |
|------|---------|
| 找不到 CSV | 检查路径，使用绝对路径 |
| OOM 错误 | 减小 `--batch_size 16` |
| 训练慢 | 使用预计算特征，减小 `--epochs` |
| 标签不平衡 | 启用 `--use_class_weights`（默认已启用） |

---

**详细文档**：请查看 `README.md`
