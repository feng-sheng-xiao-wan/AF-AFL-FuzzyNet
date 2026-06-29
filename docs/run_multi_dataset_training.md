# 多数据集AF vs AFL训练指南

## 数据集组成

本训练使用三个数据集的组合：

1. **AFDB (MIT-BIH Atrial Fibrillation Database)**
   - 44,768 样本 (85.7%)
   - 23 患者 (22 AF + 1 AFL)

2. **LTAFDB (Long-Term AF Database)**  
   - 6,293 样本 (12.0%)
   - 83 患者 (全部AF)

3. **MIT-BIH Arrhythmia Database**
   - 1,187 样本 (2.3%)
   - 8 患者 (7 AF + 1 AFL)

**总计**: 52,248 样本，111 患者，AF:AFL = 96.1%:3.9%

## 患者级别分层划分结果

- **训练集**: 44,708 样本，88 患者 (包含AF和AFL)
- **验证集**: 2,389 样本，10 患者 (只有AF)
- **测试集**: 5,151 样本，13 患者 (包含AF和AFL)

AFL患者分布：训练集9个，测试集2个，验证集0个

## 训练命令

### 基础训练命令

```bash
python train_af_vs_afl_afdb_ltafdb.py `
  --combined_csv "data/precompute_features/multi_dataset_combined_with_features.csv" `
  --output_dir "outputs_multi_dataset" `
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

### 完整参数训练命令

```bash
python train_af_vs_afl_afdb_ltafdb.py `
  --combined_csv "data/precompute_features/multi_dataset_combined_with_features.csv" `
  --output_dir "outputs_multi_dataset_full" `
  --num_leads 2 `
  --max_len 2500 `
  --max_rr_intervals 32 `
  --batch_size 128 `
  --num_workers 0 `
  --lr 1e-3 `
  --weight_decay 1e-4 `
  --epochs 40 `
  --device auto `
  --weight_conf_gamma 0.5 `
  --min_weight 0.1 `
  --conf_discard_th 0.3 `
  --lambda_af_bin 1.5 `
  --validate_every_batch `
  --use_class_weights `
  --fuzzy_weight_init 0.3 `
  --multi_lead_method "voting" `
  --flutter_method "attention" `
  --use_precomputed `
  --use_boundary_aware `
  --boundary_alpha 0.6 `
  --boundary_threshold 0.3 `
  --lambda_boundary 0.1 `
  --low_conf_alpha 1.5
```

## 关键配置说明

### 数据配置
- `--num_leads 2`: 使用2导联ECG数据
- `--multi_lead_method "voting"`: 多导联R峰检测使用投票方法
- `--flutter_method "attention"`: Flutter检测使用能量加权方法
- `--use_precomputed`: 使用预计算特征加速训练

### 训练配置
- `--epochs 50`: 训练50个epoch
- `--batch_size 16`: 批大小16（可根据GPU内存调整）
- `--lr 1e-3`: 学习率1e-3
- `--use_class_weights`: 使用类别权重处理不平衡数据

### 边界感知训练
- `--use_boundary_aware`: 启用边界感知训练
- `--boundary_alpha 0.6`: 边界感知权重
- `--boundary_threshold 0.3`: 边界样本阈值
- `--lambda_boundary 0.1`: 边界损失权重

### 模糊规则系统
- `--fuzzy_weight_init 0.3`: 模糊规则初始权重
- `--lambda_af_bin 0.5`: 二分类损失权重

## 预期训练时间

使用预计算特征，每个epoch约需要：
- GPU (RTX 3060): ~2-3分钟
- CPU: ~10-15分钟

总训练时间（50 epochs）：
- GPU: ~2-2.5小时
- CPU: ~8-12小时

## 输出文件

训练完成后，在输出目录中会生成：
- `af_vs_afl_afdb_ltafdb.pt`: 训练好的模型（最佳验证F1）
- `temp_train.csv`, `temp_val.csv`, `temp_test.csv`: 临时划分文件
- 训练日志和性能指标（包括混淆矩阵）

## 数据验证

训练前可以检查数据集：

```bash
python diagnose_binary_classification.py --csv "data/precompute_features/multi_dataset_combined_with_features.csv"
```

## 注意事项

1. **数据不平衡**: AFL样本仅占3.9%，这是正常的临床分布。已通过以下机制处理：
   - 增强的类别权重（自动检测严重不平衡并放大少数类权重）
   - 低置信度样本关注（`--low_conf_alpha 1.5`）
   - 降低置信度权重影响（`--weight_conf_gamma 0.5`）
2. **患者级别划分**: 自动按患者划分，确保同一患者的样本不会同时出现在训练和测试集中
3. **预计算特征**: 数据集已包含预计算特征（rr_seq、noise_feat、fuzzy_logits），大幅加速训练
4. **边界感知训练**: 有助于处理类别边界模糊的样本
5. **模糊规则系统**: 结合临床知识提升分类性能，自动适配二分类任务
6. **训练策略**: 使用 `--use_low_conf_focus`（默认启用）关注低置信度样本而非丢弃，特别适合处理类别不平衡