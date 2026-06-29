# 模糊规则细化增强计划

## 目标
进一步细化模糊规则系统，加强过滤能力，提高AF和AFL的区分度，减少误分类。

## 当前问题分析

1. **特征维度不足**：当前主要使用RMSSD、CV、pNN50等基础统计特征，缺乏分布特征和频域细节
2. **规则过于简单**：规则组合方式较为线性，缺乏多维度交叉验证
3. **AF/AFL区分不够强**：两者在某些特征上相似，需要更精细的区分机制
4. **缺乏噪声过滤**：低质量样本可能影响规则判断
5. **规则置信度评估不足**：无法有效识别规则判断的可靠性

## 改进方案

### 1. 增强特征提取（extract_rr_statistics函数）

#### 1.1 增加RR间期分布特征
- **偏度（Skewness）**：RR间期分布的对称性，AF通常偏度较大
- **峰度（Kurtosis）**：RR间期分布的尖锐程度
- **分位数特征**：Q25, Q50, Q75, IQR（四分位距），用于描述分布形状
- **自相关特征**：RR序列的自相关函数，AF的自相关较低，AFL的自相关较高

#### 1.2 增强频域特征
- **多频段flutter分析**：不仅分析3-8Hz，还分析2-4Hz（慢flutter）和6-10Hz（快flutter）
- **频谱峰值检测**：检测flutter频段内的主要峰值频率
- **频谱能量分布**：分析flutter频段内能量的集中度

#### 1.3 增强P波检测
- **P波能量比**：P波频段（0.5-5Hz）的能量占比
- **P波一致性**：多个RR周期内P波的一致性
- **P波形态特征**：P波的宽度、幅度等

### 2. 细化AF规则

#### 2.1 多维度不规则性检查
```python
# 当前：简单的加权平均
mu_irreg_high = (mu_irreg_rmssd * 0.4 + mu_irreg_cv * 0.4 + mu_irreg_pnn50 * 0.2)

# 改进：多维度交叉验证
- 基础不规则性（RMSSD, CV, pNN50）
- 分布不规则性（偏度、峰度、IQR）
- 时序不规则性（自相关低、序列熵高）
- 综合评分：使用几何平均或最小值，确保所有维度都支持AF
```

#### 2.2 加强P波缺失证据
```python
# 当前：简单的1 - p_wave_presence
mu_p_wave_absent = 1.0 - p_wave_presence

# 改进：多证据融合
- P波存在性低
- P波能量比低
- P波一致性低
- 综合评分：使用加权平均，权重偏向P波存在性
```

#### 2.3 增加AF特异性检查
- **无flutter证据**：flutter_ratio必须很低
- **无规律性**：regularity_index必须很低
- **RR分布特征**：偏度大、峰度低、IQR大

#### 2.4 AF规则公式优化
```python
# 当前
s_af = mu_irreg_high * (0.6 + 0.4 * mu_p_wave_absent) * (1.0 - mu_flutter * 0.5)

# 改进：使用更严格的组合
s_af = (
    mu_irreg_high ** 1.2 *  # 提高不规则性的权重
    mu_irreg_distribution *  # 新增：分布不规则性
    (0.5 + 0.5 * mu_p_wave_absent) *  # 加强P波缺失权重
    (1.0 - mu_flutter * 0.8) *  # 加强flutter抑制
    (1.0 - mu_regular * 0.6)  # 新增：规律性抑制
)
```

### 3. 细化AFL规则

#### 3.1 加强flutter证据的多频段分析
```python
# 当前：单一flutter_ratio
mu_flutter = mu_flutter_band * 0.7 + mu_atrial_rate_afl * 0.3

# 改进：多频段融合
- 慢flutter（2-4Hz）：AFL的典型特征
- 标准flutter（3-8Hz）：当前使用的
- 快flutter（6-10Hz）：某些AFL变体
- 频谱峰值：检测flutter频段内的主要峰值
- 能量集中度：flutter能量是否集中在特定频率
- 综合评分：使用加权平均，权重偏向标准flutter
```

#### 3.2 细化心房率范围检查
```python
# 当前：简单的250-350 bpm范围
if 250.0 <= atrial_rate <= 350.0:
    mu_atrial_rate_afl = ...

# 改进：多范围检查
- 典型AFL：280-320 bpm（最高置信度）
- 可能AFL：250-280 bpm 或 320-350 bpm（中等置信度）
- 边缘AFL：240-250 bpm 或 350-360 bpm（低置信度）
- 使用三角隶属度函数，而非简单线性
```

#### 3.3 增加规律性细节检查
```python
# 当前：简单的regularity_index
mu_regular = regularity_index

# 改进：多维度规律性
- 基础规律性（regularity_index）
- 自相关规律性（自相关高）
- 周期一致性（RR周期的重复性）
- 综合评分：使用加权平均
```

#### 3.4 AFL规则公式优化
```python
# 当前
s_afl_base = (
    0.6 * mu_flutter      # flutter 证据为主
    + 0.2 * mu_hr_medium  # 中等心率为辅
    + 0.2 * mu_regular    # 规律性辅助
)
s_afl = s_afl_base * afl_regular_boost

# 改进：更精细的组合
s_afl = (
    mu_flutter_enhanced ** 1.1 *  # 增强flutter权重
    (0.4 + 0.6 * mu_regular_enhanced) *  # 增强规律性权重
    (0.3 + 0.7 * mu_atrial_rate_afl_enhanced) *  # 增强心房率权重
    (1.0 - mu_irreg_high * 0.7)  # 抑制高度不规则性
)
```

### 4. 增加规则互斥性检查

#### 4.1 AF和AFL冲突检测
```python
# 检测AF和AFL是否同时高置信度
conflict_threshold = 0.4
if s_af > conflict_threshold and s_afl > conflict_threshold:
    # 计算冲突强度
    conflict_strength = min(s_af, s_afl) / max(s_af, s_afl)
    
    # 根据特征优先级解决冲突
    if mu_irreg_high > 0.7 and mu_flutter < 0.3:
        # 高度不规则 + 无flutter -> 偏向AF
        s_afl *= (1.0 - conflict_strength * 0.5)
    elif mu_flutter > 0.6 and mu_regular > 0.6:
        # 强flutter + 规律性 -> 偏向AFL
        s_af *= (1.0 - conflict_strength * 0.5)
    else:
        # 其他情况：根据规则置信度决定
        if rule_conf_af > rule_conf_afl:
            s_afl *= (1.0 - conflict_strength * 0.3)
        else:
            s_af *= (1.0 - conflict_strength * 0.3)
```

#### 4.2 规则置信度计算
```python
# 基于特征一致性和规则匹配度
rule_conf = {
    "af": compute_rule_confidence(s_af, mu_irreg_high, mu_p_wave_absent, mu_flutter),
    "afl": compute_rule_confidence(s_afl, mu_flutter, mu_regular, mu_atrial_rate_afl),
    # ...
}

def compute_rule_confidence(score, *supporting_features):
    """计算规则置信度"""
    # 特征一致性：支持特征的平均值
    feature_consistency = np.mean(supporting_features)
    # 规则匹配度：规则分数本身
    rule_match = score
    # 综合置信度
    confidence = (feature_consistency * 0.6 + rule_match * 0.4)
    return confidence
```

### 5. 增加噪声过滤

#### 5.1 噪声质量评估
```python
# 在extract_rr_statistics中增加噪声特征
noise_features = extract_noise_features(ecg, fs)
noise_quality = compute_noise_quality(noise_features)

def compute_noise_quality(noise_features):
    """计算信号质量（0-1，1表示高质量）"""
    snr = noise_features[4]  # SNR估计
    spec_ent = noise_features[5]  # 谱熵
    impulsive = noise_features[6]  # 脉冲比例
    
    # SNR越高，质量越好
    snr_quality = min(snr / 10.0, 1.0)  # 假设SNR>10为高质量
    # 谱熵适中，质量好（过高或过低都不好）
    spec_ent_quality = 1.0 - abs(spec_ent - 0.5) * 2.0
    # 脉冲比例低，质量好
    impulsive_quality = 1.0 - min(impulsive * 2.0, 1.0)
    
    quality = (snr_quality * 0.5 + spec_ent_quality * 0.3 + impulsive_quality * 0.2)
    return quality
```

#### 5.2 基于质量的规则调整
```python
# 在FuzzyRuleSystem.infer中
noise_quality = rr_stats.get("noise_quality", 1.0)

# 低质量样本：降低规则置信度
if noise_quality < 0.5:
    # 降低所有规则的分数
    quality_factor = 0.5 + 0.5 * noise_quality  # 0.5-1.0
    s_af *= quality_factor
    s_afl *= quality_factor
    s_psvt *= quality_factor
    s_normal *= quality_factor
```

### 6. 优化规则权重和组合策略

#### 6.1 使用非线性组合
```python
# 当前：线性加权平均
mu_irreg_high = (mu_irreg_rmssd * 0.4 + mu_irreg_cv * 0.4 + mu_irreg_pnn50 * 0.2)

# 改进：使用几何平均或最小值（更严格）
# 选项1：几何平均（所有特征都必须支持）
mu_irreg_high_geo = (mu_irreg_rmssd * mu_irreg_cv * mu_irreg_pnn50) ** (1/3)

# 选项2：加权几何平均
mu_irreg_high_weighted_geo = (
    mu_irreg_rmssd ** 0.4 * 
    mu_irreg_cv ** 0.4 * 
    mu_irreg_pnn50 ** 0.2
)

# 选项3：最小值（最严格）
mu_irreg_high_min = min(mu_irreg_rmssd, mu_irreg_cv, mu_irreg_pnn50)

# 选项4：混合策略（根据置信度选择）
if max(mu_irreg_rmssd, mu_irreg_cv, mu_irreg_pnn50) > 0.8:
    # 高置信度：使用几何平均
    mu_irreg_high = mu_irreg_high_weighted_geo
else:
    # 低置信度：使用加权平均（更宽松）
    mu_irreg_high = (mu_irreg_rmssd * 0.4 + mu_irreg_cv * 0.4 + mu_irreg_pnn50 * 0.2)
```

#### 6.2 自适应权重调整
```python
# 根据特征可靠性动态调整权重
def compute_adaptive_weights(features, confidences):
    """根据特征置信度计算自适应权重"""
    # 置信度高的特征权重更大
    weights = confidences / (confidences.sum() + 1e-8)
    return weights
```

## 实施优先级

### 阶段1：核心增强（立即实施）
1. ✅ 增加RR间期分布特征（偏度、峰度、分位数）
2. ✅ 细化AF规则（多维度不规则性、加强P波缺失）
3. ✅ 细化AFL规则（多频段flutter、细化心房率范围）
4. ✅ 增加规则互斥性检查

### 阶段2：质量提升（后续优化）
5. ✅ 增加噪声过滤
6. ✅ 增加规则置信度评估
7. ✅ 优化规则权重和组合策略

## 预期效果

1. **提高AF召回率**：通过多维度不规则性检查，更准确识别AF
2. **提高AFL召回率**：通过多频段flutter分析和细化心房率检查，更准确识别AFL
3. **减少误分类**：通过规则互斥性检查和噪声过滤，减少AF和AFL之间的误分类
4. **提高规则可靠性**：通过置信度评估，识别规则判断的可靠性

## 注意事项

1. **计算复杂度**：新增特征会增加计算时间，需要平衡性能和效果
2. **参数调优**：新增的阈值和权重需要根据实际数据调优
3. **向后兼容**：确保改进后的规则系统与现有模型兼容
4. **测试验证**：需要在验证集上充分测试，确保改进有效

