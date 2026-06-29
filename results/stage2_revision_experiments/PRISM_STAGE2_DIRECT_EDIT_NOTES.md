# Prism / Manuscript Direct Edit Notes

Use these results as replacement/additional material for the reviewer-requested second pass.

1. Replace the baseline-comparison caption with:

Test-set performance under a threshold-harmonized post-processing protocol. For each model, the AFL decision threshold was selected on the validation set by maximizing AFL-F1 and then fixed for test evaluation. AUROC and AUPRC are threshold-free metrics computed from test probabilities. Values in brackets are sample-level bootstrap 95% confidence intervals (1000 resamples). Baseline NPZ preprocessing contained 6,734 available test segments, one fewer AF segment than the 6,735-segment main test CSV.

2. Add a sentence after the table:

Under this protocol, the proposed model achieved favorable operating-point performance, but the comparison remains dependent on validation-threshold selection and on the available baseline preprocessing outputs.

3. Add the clean/noisy AFL paragraph:

Clean and noisy AFL performance was examined by separating AFL test segments according to whether the file path or dataset label indicated synthetic noise augmentation. AFL recall was computed as TP/(TP+FN) within each subgroup using the validation-selected threshold.

4. Add the post-hoc sensitivity caveat:

The No Gate, No Quality, and Fixed Fusion analyses are post-hoc inference-time interventions applied to the trained checkpoint. They are included only as sensitivity analyses and should not be interpreted as retrained architectural ablations.

5. Add the fuzzy-logit reproducibility sentence:

Fuzzy rule scores were normalized across rhythm classes and converted to logits as log(score + 1e-8) * 2.0. For binary AF/AFL experiments, only the AF and AFL fuzzy-logit entries were used.

6. Add the data-processing reproducibility sentence:

All segments were represented as 10-s windows at 250 Hz (2,500 samples). ECG channels were interpreted as (time, lead), the first two leads were retained for the main model, longer segments were center-cropped, shorter segments were zero-padded, and each lead was independently standardized by subtracting its mean and dividing by its standard deviation.
