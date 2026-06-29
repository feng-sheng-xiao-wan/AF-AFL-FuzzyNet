# 基线对比实验：AF vs AFL 二分类

五个基线模型与主模型使用**相同数据划分**（患者级 8:1:1，与 `noise_aware_ecg_af_afl_simple.split_dataset_csv` 一致），统一评估 **Acc、Macro F1、AFL-F1**（片段级）。

---

## 一键：同时训练五个并保存指标到 CSV

在**项目根目录**执行下面一条命令即可依次完成：数据准备 → 五个基线训练 → 汇总 Val/Test 的准确率、F1、召回率等到 `ablation_Af_AFL/baseline_metrics.csv`（并带时间戳备份到 `baseline_metrics_YYYYMMDD_HHMMSS.csv`）。

```powershell
cd "e:\Study\FK-ECG_fuzzy_v1\aflike_v1_starter"
python ablation_Af_AFL/run_all_baselines.py
```

可选参数：
- `--csv "路径"`：数据 CSV，默认 `data/precompute_features/multi_dataset_with_afl_noise_aug.csv`
- `--epochs 30`、`--batch_size 32`、`--num_workers 0`
- `--print_interval 10`：每 N 个 batch 打印一次进度（当前 epoch、batch 序号、累计 loss），设为 `0` 则不打印
- `--skip_data_prep`：若已准备好 RR/ECG/ECG+RR 的 npz，可跳过数据准备只跑训练

输出说明：
- **CSV 列**：`baseline`, `error`, `val_loss`, `val_acc`, `val_macro_f1`, `val_afl_f1`, `test_loss`, `test_acc`, `test_macro_f1`, `test_afl_f1`, `test_AF_precision`, `test_AF_recall`, `test_AF_f1`, `test_AFL_precision`, `test_AFL_recall`, `test_AFL_f1`
- **日志**：每个基线的完整终端输出在 `ablation_Af_AFL/logs/YYYYMMDD_HHMMSS/<基线名>.log`

---

## 一、数据准备（只需跑一次）

在**项目根目录** `aflike_v1_starter` 下执行。请将 `DATA_CSV` 换成你的实际 CSV 路径（如预计算特征表）。

```powershell
cd "e:\Study\FK-ECG_fuzzy_v1\aflike_v1_starter"

# 1) ArNet2：RR 序列 .npz（train / val / test）
python ablation_Af_AFL/prepare_rr_npz_from_csv.py --combined_csv "data/precompute_features/multi_dataset_with_afl_noise_aug.csv" --output_root "data/ablation_rr_npz"

# 2) RawECGNet、Wang2021、Kraft2025：ECG + label .npz（train / val / test）
python ablation_Af_AFL/prepare_ecg_npz_from_csv.py --combined_csv "data/precompute_features/multi_dataset_with_afl_noise_aug.csv" --output_root "ablation_Af_AFL/rawecg_npz"

# 3) Fleury2024：ECG + RR + label .npz（train / val / test）
python ablation_Af_AFL/prepare_ecg_rr_npz_from_csv.py --combined_csv "data/precompute_features/multi_dataset_with_afl_noise_aug.csv" --output_root "ablation_Af_AFL/fleury_ecg_rr_npz"
```

---

## 二、训练与测试（五个基线）

以下命令均在**项目根目录**执行。训练结束后会用 best checkpoint 在 **test** 上评估（五个基线均支持 `--test_dir`，训练结束后用 best 模型跑 Test 评估）。

### 1. ArNet2（仅 RR 序列）

数据准备会生成 `train/val/test`，训练时传入 `--test_dir` 即可在结束后用 best 模型跑 Test 评估。

```powershell
python ablation_Af_AFL/compare1_train_arnet2.py --train_dir "data/ablation_rr_npz/train" --val_dir "data/ablation_rr_npz/val" --test_dir "data/ablation_rr_npz/test" --save_dir "ablation_Af_AFL/checkpoints_arnet2" --epochs 30 --batch_size 32 --num_workers 0
```

**若 Val/Test 上 AFL 指标全为 0**：多为「仅 RR 特征」的基线坍缩到多数类（全判 AF）。可先看启动时打印的 `Train/Val label distribution: AF(0)=..., AFL(1)=...` 确认 AFL 样本数>0；若数据正常则属该基线的已知局限。

### 2. RawECGNet（原始 ECG）

```powershell
python ablation_Af_AFL/compare2_train_rawecgnet.py --train_dir "ablation_Af_AFL/rawecg_npz/train" --val_dir "ablation_Af_AFL/rawecg_npz/val" --test_dir "ablation_Af_AFL/rawecg_npz/test" --save_dir "ablation_Af_AFL/checkpoints_rawecgnet" --epochs 30 --batch_size 32 --num_workers 0
```

### 3. Wang2021 BiLSTM（CNN + BiLSTM）

```powershell
python ablation_Af_AFL/compare3_train_wang2021_bilstm.py --train_dir "ablation_Af_AFL/rawecg_npz/train" --val_dir "ablation_Af_AFL/rawecg_npz/val" --test_dir "ablation_Af_AFL/rawecg_npz/test" --save_dir "ablation_Af_AFL/checkpoints_wang2021" --epochs 30 --batch_size 32 --num_workers 0
```

### 4. Kraft2025 ConvNeXt1D

```powershell
python ablation_Af_AFL/compare4_train_kraft2025_convnext1d.py --train_dir "ablation_Af_AFL/rawecg_npz/train" --val_dir "ablation_Af_AFL/rawecg_npz/val" --test_dir "ablation_Af_AFL/rawecg_npz/test" --save_dir "ablation_Af_AFL/checkpoints_kraft2025" --epochs 30 --batch_size 32 --num_workers 0
```

### 5. Fleury2024 Modular（ECG + RR 多分支）

```powershell
python ablation_Af_AFL/compare5_train_fleury2024_modular.py --train_dir "ablation_Af_AFL/fleury_ecg_rr_npz/train" --val_dir "ablation_Af_AFL/fleury_ecg_rr_npz/val" --test_dir "ablation_Af_AFL/fleury_ecg_rr_npz/test" --save_dir "ablation_Af_AFL/checkpoints_fleury2024" --epochs 30 --batch_size 32 --num_workers 0
```

---

## 三、单独在测试集上评估（test_baselines.py）

若训练脚本内 Test 阶段未成功运行（例如 `torch.load` 报错），可用**独立测试脚本**对已有 best checkpoint 在测试集上做评估。加载 checkpoint 时使用 `weights_only=False`，兼容 PyTorch 2.6+。

在**项目根目录**执行：

```powershell
# 测试单个基线（不写 --ckpt/--test_dir 则用该基线默认路径）
python ablation_Af_AFL/test_baselines.py --baseline rawecgnet
python ablation_Af_AFL/test_baselines.py --baseline arnet2 --ckpt ablation_Af_AFL/checkpoints_arnet2/best_arnet2.pth --test_dir data/ablation_rr_npz/test

# 测试全部五个基线，并把 Test 指标写入 CSV
python ablation_Af_AFL/test_baselines.py --baseline all --out_csv ablation_Af_AFL/baseline_test_metrics.csv
```

参数说明：
- `--baseline`：`arnet2` | `rawecgnet` | `wang2021` | `kraft2025` | `fleury2024` | `all`
- `--ckpt`：checkpoint 路径，不填则用该基线默认路径（见脚本内 `DEFAULTS`）
- `--test_dir`：测试集 npz 目录，不填则用该基线默认路径
- `--batch_size`、`--device`、`--out_csv`（可选，写入 CSV）

输出：终端打印 Test Loss、Acc、Macro F1、AFL-F1、classification_report、混淆矩阵；若指定 `--out_csv` 则追加/写入该 CSV。

---

## 四、汇总对比指标

| 基线           | 输入       | 保存目录                         | 输出指标（Val/Test）   |
|----------------|------------|----------------------------------|------------------------|
| compare1 ArNet2 | RR         | `checkpoints_arnet2`             | Acc, Macro F1, AFL-F1 |
| compare2 RawECGNet | ECG      | `checkpoints_rawecgnet`          | Acc, Macro F1, AFL-F1 |
| compare3 Wang2021 | ECG       | `checkpoints_wang2021`           | Macro F1, AFL-F1       |
| compare4 Kraft2025 | ECG      | `checkpoints_kraft2025`          | Macro F1, AFL-F1       |
| compare5 Fleury2024 | ECG+RR  | `checkpoints_fleury2024`         | Macro F1, AFL-F1       |

所有基线均为 **AF=0、AFL=1** 二分类，best 按 **Val Macro F1** 保存，Test 为加载 best 后的一次评估。

---

## 五、可选：一键跑齐数据准备 + 五条训练

```powershell
cd "e:\Study\FK-ECG_fuzzy_v1\aflike_v1_starter"
$csv = "data/precompute_features/multi_dataset_with_afl_noise_aug.csv"

python ablation_Af_AFL/prepare_rr_npz_from_csv.py --combined_csv $csv --output_root "data/ablation_rr_npz"
python ablation_Af_AFL/prepare_ecg_npz_from_csv.py --combined_csv $csv --output_root "ablation_Af_AFL/rawecg_npz"
python ablation_Af_AFL/prepare_ecg_rr_npz_from_csv.py --combined_csv $csv --output_root "ablation_Af_AFL/fleury_ecg_rr_npz"

python ablation_Af_AFL/compare1_train_arnet2.py --train_dir "data/ablation_rr_npz/train" --val_dir "data/ablation_rr_npz/val" --save_dir "ablation_Af_AFL/checkpoints_arnet2" --epochs 30 --batch_size 32
python ablation_Af_AFL/compare2_train_rawecgnet.py --train_dir "ablation_Af_AFL/rawecg_npz/train" --val_dir "ablation_Af_AFL/rawecg_npz/val" --test_dir "ablation_Af_AFL/rawecg_npz/test" --save_dir "ablation_Af_AFL/checkpoints_rawecgnet" --epochs 30 --batch_size 32
python ablation_Af_AFL/compare3_train_wang2021_bilstm.py --train_dir "ablation_Af_AFL/rawecg_npz/train" --val_dir "ablation_Af_AFL/rawecg_npz/val" --test_dir "ablation_Af_AFL/rawecg_npz/test" --save_dir "ablation_Af_AFL/checkpoints_wang2021" --epochs 30 --batch_size 32
python ablation_Af_AFL/compare4_train_kraft2025_convnext1d.py --train_dir "ablation_Af_AFL/rawecg_npz/train" --val_dir "ablation_Af_AFL/rawecg_npz/val" --test_dir "ablation_Af_AFL/rawecg_npz/test" --save_dir "ablation_Af_AFL/checkpoints_kraft2025" --epochs 30 --batch_size 32
python ablation_Af_AFL/compare5_train_fleury2024_modular.py --train_dir "ablation_Af_AFL/fleury_ecg_rr_npz/train" --val_dir "ablation_Af_AFL/fleury_ecg_rr_npz/val" --test_dir "ablation_Af_AFL/fleury_ecg_rr_npz/test" --save_dir "ablation_Af_AFL/checkpoints_fleury2024" --epochs 30 --batch_size 32
```
