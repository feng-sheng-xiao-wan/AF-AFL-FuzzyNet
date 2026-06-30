# Checkpoints

This folder contains the compact AF/AFL checkpoint uploaded with the code package.

| file | source run | note |
|---|---|---|
| `af_vs_afl_afdb_ltafdb.pt` | `outputs_thr_tuned_multi_afl_noise_aug_seg/af_vs_afl_afdb_ltafdb.pt` | Main AF/AFL checkpoint with segment-level validation thresholding. |

Checkpoint metadata from the saved file:

- best epoch: `2`
- saved validation AFL-F1: `0.8346213242395648`
- saved AFL probability threshold: `0.04919999837875366`
- record aggregation: disabled
- training epochs requested: `10`

Raw ECG data and precomputed `.npz` features are not included.
