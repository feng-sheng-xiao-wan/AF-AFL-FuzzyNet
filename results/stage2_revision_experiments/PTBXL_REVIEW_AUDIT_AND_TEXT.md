# PTB-XL Reviewer-Requirement Audit

## Short Answer

This requirement was only partially covered before. The reduced-lead PTB-XL comparison already had a paired bootstrap 95% CI, but the checkpoint-based PTB-XL external validation did not yet have a complete CI/label-source/multilabel/patient-deduplication/10-s-window statement. I added the missing audit files and text here.

## Generated Files

- `ptbxl_external_ci_summary.csv`: sample-level bootstrap 95% CI for checkpoint-based PTB-XL external validation.
- `ptbxl_external_split_first_with_ci.json`: split-first checkpoint PTB-XL metrics with CI.
- `ptbxl_external_legacy_with_ci.json`: legacy main checkpoint PTB-XL metrics with CI.
- `ptbxl_subset_metadata_audit.csv`: PTB-XL subset audit joined back to `ptbxl_database.csv`.

## PTB-XL External Validation With 95% CI

Bootstrap protocol: sample-level nonparametric bootstrap, 2,000 resamples, percentile 95% CI.

| Analysis | n | AF | AFL | AFL-F1 | Macro-F1 | Accuracy | AFL Precision | AFL Recall |
|---|---:|---:|---:|---|---|---|---|---|
| Split-first checkpoint external stress test | 104 | 48 | 56 | 0.638 [0.517, 0.745] | 0.670 [0.577, 0.759] | 0.673 [0.587, 0.760] | 0.789 [0.657, 0.914] | 0.536 [0.400, 0.667] |
| Legacy/main checkpoint external stress test | 104 | 48 | 56 | 0.680 [0.562, 0.783] | 0.701 [0.608, 0.787] | 0.702 [0.615, 0.788] | 0.805 [0.681, 0.920] | 0.589 [0.459, 0.720] |

Use the row that matches the manuscript experiment. If the manuscript currently reports AFL-F1 = 0.680, use the legacy/main checkpoint row and explicitly call it a legacy/main-checkpoint external stress test.

## PTB-XL Lead-Configuration Comparison CI

This was already available in `results/ptbxl_lead_comparison/ptbxl_lead_comparison_bootstrap_table.tex`.

| Comparison | Delta AFL-F1 | 95% CI | Interpretation |
|---|---:|---|---|
| II+V1 minus 12-lead | 0.010 | [-0.017, 0.035] | Preliminary comparable only |

This should not be mixed with the checkpoint-based external validation above, because the lead comparison was trained/evaluated within PTB-XL cross-validation folds.

## Label Source

Labels were traced to the PTB-XL metadata file:

- Source file: `data/ptb-xl/ptbxl_database.csv`
- Label field: `scp_codes`
- Mapping code: `process_pretrain_data.py`
- Positive SCP codes only: codes with value `> 0` were treated as active labels.
- AF mapping: `AFIB` or `AF`
- AFL mapping: `AFLT` or `AFL`

The final external-validation CSV `data/precompute_features/ptbxl/features/ptbxl_af_afl_with_features.csv` contains 104 ECG records: 48 AF and 56 AFL.

## Multilabel Situation

PTB-XL `scp_codes` can contain multiple diagnostic statements. In the audited 104-record AF/AFL subset:

- 6/104 records had more than one positive SCP code.
- No record had both a positive AF code and a positive AFL code.
- Some rows contained inactive codes with value `0.0`; these were not treated as active labels by the mapping function.

Therefore, the AF/AFL label assignment was single-target for this subset, although the original PTB-XL annotation field is multi-label.

## Patient Deduplication / Grouping

For the checkpoint-based external validation:

- The 104 PTB-XL ECG records corresponded to 81 unique patients.
- 8 patients contributed more than one ECG record.
- No patient had both AF and AFL labels in this audited subset.
- The external-validation result is record-level, not patient-deduplicated.

For the PTB-XL lead-configuration comparison:

- The prediction file contains 104 unique ECG records and 85 unique patient IDs.
- The comparison used `StratifiedGroupKFold`, so folds were grouped by patient ID.
- It is still better described as a limited PTB-XL feasibility analysis, not definitive generalization evidence.

## 10-s ECG Input

The audited external-validation CSV contains only 10-s PTB-XL windows:

- `t0 = 0.0`, `t1 = 10.0` for all 104 records.
- Each path contains `_0.000_10.000`.
- The NPZ ECG input uses 2,500 samples at 250 Hz, corresponding to 10 seconds.
- The external validation used the two-lead PTB-XL slices prepared under `data/slices_PTB-XL_2leads_II_V1/windows_ptbxl`.

## Manuscript Text To Add

For the PTB-XL external validation subsection:

```latex
The PTB-XL external stress-test subset was traced back to the PTB-XL `scp_codes` metadata field. AF labels were assigned from positive AFIB/AF SCP codes, and AFL labels were assigned from positive AFLT/AFL SCP codes; SCP codes with a value of 0 were not treated as active labels. Although PTB-XL uses a multi-label annotation format, no record in this AF/AFL subset had simultaneous positive AF and AFL codes. The evaluated subset contained 104 ECG records (48 AF and 56 AFL) from 81 unique patients; 8 patients contributed more than one ECG record, so the reported external result is record-level rather than patient-deduplicated. All evaluated PTB-XL inputs were 10-s ECG windows (0--10 s, 2,500 samples at 250 Hz).
```

For the CI sentence, choose the checkpoint row that matches the paper:

```latex
For the legacy/main checkpoint external stress test, AFL-F1 was 0.680 (95\% CI: 0.562--0.783), macro-F1 was 0.701 (95\% CI: 0.608--0.787), and accuracy was 0.702 (95\% CI: 0.615--0.788), using sample-level bootstrap resampling with 2,000 resamples.
```

Or, if using the split-first checkpoint:

```latex
For the split-first checkpoint external stress test, AFL-F1 was 0.638 (95\% CI: 0.517--0.745), macro-F1 was 0.670 (95\% CI: 0.577--0.759), and accuracy was 0.673 (95\% CI: 0.587--0.760), using sample-level bootstrap resampling with 2,000 resamples.
```

For the lead comparison:

```latex
The PTB-XL lead-configuration comparison was trained and evaluated within PTB-XL cross-validation folds using patient-grouped splits. It should therefore be interpreted as a limited lead-feasibility analysis rather than external testing of the AFDB/LTAFDB/MITDB-trained checkpoint. The paired bootstrap difference in AFL-F1 between II+V1 and 12-lead settings was 0.010 (95\% CI: -0.017 to 0.035).
```

## Reviewer Response Text

```text
We clarified the PTB-XL label source and evaluation unit. PTB-XL labels were derived from the `scp_codes` metadata field, using positive AFIB/AF codes for AF and positive AFLT/AFL codes for AFL. Because PTB-XL annotations are multi-label, we audited the selected AF/AFL subset: 6 of 104 records contained more than one positive SCP code, but none contained simultaneous positive AF and AFL codes. The checkpoint-based external evaluation was record-level rather than patient-deduplicated; the 104 records corresponded to 81 unique patients, with 8 patients contributing more than one ECG. All evaluated PTB-XL inputs were 10-s ECG windows. We also added sample-level bootstrap 95% confidence intervals for the PTB-XL external results and explicitly separated this external stress test from the PTB-XL internal lead-configuration feasibility analysis.
```
