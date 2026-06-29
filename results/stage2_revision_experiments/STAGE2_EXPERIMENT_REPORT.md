# Stage-2 Revision Experiments

All outputs were generated under `outputs/stage2_revision_experiments` using existing checkpoints and post-processing. No new performance values were invented.

## Protocol

- Main internal split: `data/precompute_features/multi_dataset_with_afl_noise_aug.csv`, `split_seed=42`, `AFL_NOISY_GROUP=independent`.
- This exactly reproduces `ablation_Af_AFL/revision_pack/temp_test_eval.csv` with 6,735 test segments.
- Baselines were evaluated on their available NPZ validation/test folders; their available test set has 6,734 segments because one AF segment is absent from the baseline preprocessing outputs.
- Threshold protocol: every model uses a validation-selected threshold that maximizes AFL-F1 on validation probabilities. Saved checkpoint thresholds are ignored for the threshold-harmonized table.
- CI protocol: sample-level nonparametric bootstrap, 1000 resamples, 95% percentile interval.

## Threshold-Harmonized Main Table

| model                                     | group               | n    | n_af | n_afl | threshold          | acc_ci95             | afl_precision_ci95   | afl_recall_ci95      | afl_f1_ci95          | macro_f1_ci95        | auroc_ci95           | auprc_ci95           |
| ----------------------------------------- | ------------------- | ---- | ---- | ----- | ------------------ | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| Ours                                      | main                | 6735 | 5217 | 1518  | 0.5303800106048584 | 0.918 [0.912, 0.924] | 0.972 [0.962, 0.982] | 0.656 [0.632, 0.679] | 0.783 [0.765, 0.800] | 0.866 [0.856, 0.876] | 0.861 [0.849, 0.874] | 0.833 [0.818, 0.849] |
| Ours neural-only (post-hoc No Gate)       | posthoc_sensitivity | 6735 | 5217 | 1518  | 0.7028599977493286 | 0.914 [0.907, 0.920] | 0.970 [0.958, 0.979] | 0.638 [0.615, 0.661] | 0.769 [0.752, 0.787] | 0.858 [0.847, 0.869] | 0.858 [0.845, 0.872] | 0.825 [0.809, 0.842] |
| Ours quality-masked (post-hoc No Quality) | posthoc_sensitivity | 6735 | 5217 | 1518  | 0.5705599784851074 | 0.924 [0.918, 0.931] | 0.965 [0.954, 0.975] | 0.689 [0.666, 0.713] | 0.804 [0.787, 0.821] | 0.879 [0.869, 0.889] | 0.847 [0.832, 0.863] | 0.824 [0.808, 0.842] |
| Ours fixed fusion alpha=0.30 (post-hoc)   | posthoc_sensitivity | 6735 | 5217 | 1518  | 0.5548800230026245 | 0.923 [0.917, 0.930] | 0.971 [0.961, 0.981] | 0.679 [0.657, 0.703] | 0.799 [0.783, 0.816] | 0.876 [0.866, 0.886] | 0.862 [0.847, 0.876] | 0.835 [0.819, 0.851] |
| Fuzzy logits only                         | rule_only           | 6735 | 5217 | 1518  | 0.2549999952316284 | 0.875 [0.868, 0.883] | 0.802 [0.778, 0.826] | 0.594 [0.570, 0.619] | 0.683 [0.661, 0.703] | 0.803 [0.791, 0.814] | 0.829 [0.816, 0.841] | 0.678 [0.651, 0.706] |
| ArNet2                                    | baseline            | 6734 | 5216 | 1518  | 0.0099999997764825 | 0.225 [0.216, 0.235] | 0.225 [0.216, 0.235] | 1.000 [1.000, 1.000] | 0.368 [0.355, 0.381] | 0.184 [0.178, 0.190] | 0.516 [0.500, 0.531] | 0.216 [0.205, 0.227] |
| RawECGNet                                 | baseline            | 6734 | 5216 | 1518  | 0.9900000095367432 | 0.581 [0.570, 0.592] | 0.347 [0.332, 0.361] | 0.974 [0.966, 0.982] | 0.512 [0.495, 0.527] | 0.572 [0.561, 0.584] | 0.876 [0.867, 0.884] | 0.550 [0.526, 0.574] |
| Wang2021 BiLSTM                           | baseline            | 6734 | 5216 | 1518  | 0.9753000140190125 | 0.592 [0.581, 0.603] | 0.355 [0.341, 0.369] | 0.991 [0.986, 0.995] | 0.523 [0.507, 0.538] | 0.583 [0.572, 0.594] | 0.910 [0.902, 0.917] | 0.735 [0.714, 0.757] |
| Kraft2025 ConvNeXt1D                      | baseline            | 6734 | 5216 | 1518  | 0.9900000095367432 | 0.493 [0.480, 0.505] | 0.308 [0.294, 0.321] | 0.996 [0.993, 0.999] | 0.470 [0.455, 0.486] | 0.492 [0.479, 0.504] | 0.890 [0.882, 0.898] | 0.632 [0.609, 0.654] |
| Fleury2024 modular                        | baseline            | 6734 | 5216 | 1518  | 0.9272800087928772 | 0.480 [0.468, 0.492] | 0.300 [0.286, 0.312] | 0.978 [0.970, 0.985] | 0.459 [0.442, 0.473] | 0.479 [0.467, 0.491] | 0.720 [0.705, 0.734] | 0.430 [0.405, 0.455] |

## Clean vs Noisy AFL

| subgroup               | n_afl | afl_recall_ci95      | mean_p_afl         | median_p_afl       |
| ---------------------- | ----- | -------------------- | ------------------ | ------------------ |
| Clean AFL              | 1112  | 0.603 [0.575, 0.633] | 0.5639169216156006 | 0.6148504614830017 |
| Noisy AFL augmentation | 406   | 0.803 [0.764, 0.840] | 0.6737990975379944 | 0.7683074474334717 |

## Fuzzy-Logit Summary

| group     | n    | fuzzy_logit_af_mean | fuzzy_logit_afl_mean | fuzzy_logit_afl_minus_af_mean | fuzzy_p_afl_mean   | fuzzy_p_afl_median |
| --------- | ---- | ------------------- | -------------------- | ----------------------------- | ------------------ | ------------------ |
| AF        | 5217 | -0.9461558510231428 | -3.5162390889005457  | -2.570083237877403            | 0.0975216308104204 | 0.0646327063441276 |
| AFL       | 1518 | -3.106230149371979  | -2.296649749944176   | 0.8095803994278035            | 0.4782659719153326 | 0.3120693117380142 |
| Clean AFL | 1112 | -3.424776770254905  | -2.430477230400919   | 0.9942995398539862            | 0.489176970280928  | 0.2324991896748542 |
| Noisy AFL | 406  | -2.233757631091649  | -1.9301074882005824  | 0.3036501428910664            | 0.4483816611208939 | 0.3356998115777969 |

## Reproducibility Details To Add

- Sampling rate and window: `fs=250 Hz`, `max_len=2500`, corresponding to 10-s windows.
- Window handling in `ECGDataset`: ECG arrays are read as `(T, C)`; segments longer than 2,500 samples are center-cropped; shorter segments are zero-padded.
- Lead handling: the main checkpoint uses `num_leads=2`. When more leads are present, the dataset keeps the first `num_leads` channels.
- Normalization: each lead is independently standardized as `(x - mean) / std`; if `std <= 1e-8`, only mean subtraction is applied.
- RR extraction: the main checkpoint uses `multi_lead_method=voting` and `max_rr_intervals=32`.
- Flutter descriptor: flutter evidence is the 3-8 Hz bandpower ratio. With `flutter_method=attention`, multilead flutter evidence is energy-weighted across leads.
- Atrial-rate surrogate: `atrial_rate = 330 * flutter_ratio` when `flutter_ratio > 0.05`, otherwise 0. This is a heuristic surrogate rather than a direct atrial-cycle detector.
- P-wave descriptor: `P_pres` is a P-wave-related heuristic score computed from 0.5-5 Hz energy in the 200-400 ms pre-R-window, not expert-validated P-wave detection.
- Fuzzy logits: rule scores are normalized and converted to logits by `log(score + 1e-8) * 2.0`; binary AF/AFL evaluation uses the AF and AFL fuzzy-logit entries.

## Text Caveats

- The post-hoc No Gate, No Quality, and Fixed Fusion rows are inference-time interventions on the existing checkpoint, not retrained architectural ablations.
- Use them as sensitivity analyses only. If the manuscript needs true ablation claims, new training runs with those components removed are still required.
- Baseline comparisons are threshold-harmonized by validation threshold, but the baseline NPZ test set has one fewer AF segment than the main model test CSV.
