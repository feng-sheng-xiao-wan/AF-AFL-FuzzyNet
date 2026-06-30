# Fuzzy-Rule-Guided AF/AFL Recognition from Short ECG Windows

This repository contains the code needed to reproduce the AF/AFL experiments, reviewer-requested post-processing analyses, and publication figures.

Raw ECG databases and precomputed feature files are not included. A compact main AF/AFL checkpoint is included for convenience; other checkpoints should be regenerated with the scripts in this repository.

## Repository Structure

```text
src/
  Main model, preprocessing, feature extraction, training, and testing scripts.

ablation_Af_AFL/
  Ablation and baseline-comparison code.

scripts/
  Reviewer-stage post-processing, PR/ROC/probability plots, and diagnostic-table generation.

checkpoints/
  Compact main AF/AFL checkpoint for inference/testing.

docs/
  Original project notes and experiment instructions.
```

## Main Scripts

- `src/noise_aware_ecg_af_afl_simple.py`: main three-branch ECG/RR/quality model with fuzzy-rule-guided fusion.
- `src/train_af_vs_afl_afdb_ltafdb.py`: AF/AFL binary training entry point.
- `src/precompute_features.py`: precompute RR, quality, and fuzzy-rule features.
- `src/process_pretrain_data.py`: PTB-XL slicing and label mapping from `scp_codes`.
- `checkpoints/af_vs_afl_afdb_ltafdb.pt`: uploaded compact AF/AFL checkpoint.
- `scripts/run_stage2_revision_experiments.py`: threshold-harmonized metrics, AUROC/AUPRC, bootstrap CI, clean/noisy AFL analysis, fuzzy-logit summary.
- `scripts/plot_stage2_revision_figures.py`: generates the two reviewer-stage diagnostic figures.

## Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data Expected But Not Included

The scripts expect local datasets such as:

```text
data/ptb-xl/ptbxl_database.csv
data/holter/
data/precompute_features/
```

These folders are intentionally excluded from GitHub because they may be large and/or subject to dataset licenses.

## Reproducing Reviewer-Stage Tables And Figures

After preparing data, features, and checkpoints, run:

```powershell
python scripts/run_stage2_revision_experiments.py --out_dir outputs/stage2_revision_experiments --device auto --batch_size 128
python scripts/plot_stage2_revision_figures.py
```

Generated result tables and figures are intentionally not tracked in this GitHub repository. They will be written to local output folders when the scripts are rerun.

## What Is Not Included

The following files are intentionally excluded:

- raw ECG waveforms
- `.npz` precomputed features
- additional `.pt`, `.pth`, `.pkl` checkpoints beyond the compact uploaded checkpoint
- generated result tables and figures
- local virtual environments
- local paper drafts and private reviewer files
