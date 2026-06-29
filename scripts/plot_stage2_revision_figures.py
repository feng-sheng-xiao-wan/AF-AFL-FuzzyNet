from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[1]
STAGE2 = ROOT / "outputs" / "stage2_revision_experiments"
CURVES = STAGE2 / "curves"
PREDS = STAGE2 / "predictions"
FIGDIR = STAGE2 / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


MODEL_ORDER = [
    ("ours_full", "Ours"),
    ("arnet2", "ArNet2"),
    ("rawecgnet", "RawECGNet"),
    ("wang2021", "Wang2021 BiLSTM"),
    ("kraft2025", "Kraft2025 ConvNeXt1D"),
    ("fleury2024", "Fleury2024 Modular"),
]

COLORS = {
    "ours_full": "#1f5aa6",
    "arnet2": "#707070",
    "rawecgnet": "#d55e00",
    "wang2021": "#009e73",
    "kraft2025": "#cc79a7",
    "fleury2024": "#56b4e9",
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.45,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _ci_value(metrics: pd.DataFrame, model_name: str, metric: str) -> str:
    row = metrics.loc[metrics["model"].astype(str).str.lower() == model_name.lower()]
    if row.empty:
        return ""
    return str(row.iloc[0][f"{metric}_ci95"])


def _point_value(metrics: pd.DataFrame, model_name: str, metric: str) -> str:
    row = metrics.loc[metrics["model"].astype(str).str.lower() == model_name.lower()]
    if row.empty:
        return "NA"
    return f"{float(row.iloc[0][metric]):.3f}"


def plot_threshold_free() -> None:
    metrics = pd.read_csv(STAGE2 / "threshold_harmonized_metrics_with_ci.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)
    ax_roc, ax_pr = axes

    combined_rows = []
    for key, label in MODEL_ORDER:
        color = COLORS[key]
        lw = 2.8 if key == "ours_full" else 1.8
        z = 5 if key == "ours_full" else 3

        roc = pd.read_csv(CURVES / f"{key}_roc_curve.csv")
        pr = pd.read_csv(CURVES / f"{key}_pr_curve.csv")
        auroc = _point_value(metrics, label, "auroc")
        auprc = _point_value(metrics, label, "auprc")

        ax_roc.plot(
            roc["fpr"],
            roc["tpr"],
            color=color,
            lw=lw,
            zorder=z,
            label=f"{label}: {auroc}",
        )
        ax_pr.plot(
            pr["recall"],
            pr["precision"],
            color=color,
            lw=lw,
            zorder=z,
            label=f"{label}: {auprc}",
        )
        roc_out = roc.copy()
        roc_out["curve"] = "ROC"
        roc_out["model_key"] = key
        roc_out["display_model"] = label
        pr_out = pr.copy()
        pr_out["curve"] = "PR"
        pr_out["model_key"] = key
        pr_out["display_model"] = label
        combined_rows.extend([roc_out, pr_out])

    y = pd.read_csv(PREDS / "ours_full_test_predictions.csv")["y_true"].to_numpy()
    prevalence = float((y == 1).mean())

    ax_roc.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#9a9a9a", label="Chance")
    ax_pr.axhline(prevalence, ls="--", lw=1.0, color="#9a9a9a", label=f"AFL prevalence={prevalence:.3f}")

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, color="#e8e8e8")
        ax.set_aspect("equal", adjustable="box")

    ax_roc.set_title("A. ROC curves")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.legend(loc="lower right", frameon=False, title="AUROC")

    ax_pr.set_title("B. Precision-recall curves")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="lower left", frameon=False, title="AUPRC")

    fig.suptitle("Threshold-Free Diagnostic Curves for AF/AFL Recognition", fontsize=14, y=1.03)
    for ext in ["png", "pdf"]:
        fig.savefig(FIGDIR / f"figure_4_threshold_free_diagnostics.{ext}", bbox_inches="tight")
    plt.close(fig)

    pd.concat(combined_rows, ignore_index=True, sort=False).to_csv(
        FIGDIR / "figure_4_threshold_free_plot_data.csv", index=False
    )


def _afl_metrics_at_threshold(y_true: np.ndarray, p_afl: np.ndarray, threshold: float) -> dict:
    pred = (p_afl >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel().astype(float)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    return {
        "threshold": threshold,
        "afl_precision": precision,
        "afl_recall": recall,
        "afl_f1": f1,
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "accuracy": float((pred == y_true).mean()),
    }


def plot_probability_threshold() -> None:
    pred_df = pd.read_csv(PREDS / "ours_full_test_predictions.csv")
    with open(STAGE2 / "thresholds.json", "r", encoding="utf-8") as f:
        thresholds = json.load(f)
    selected_t = float(thresholds["ours_full"])

    y_true = pred_df["y_true"].to_numpy(dtype=np.int64)
    p_afl = pred_df["p_afl"].to_numpy(dtype=np.float32)

    bins = np.linspace(0, 1, 41)
    af = p_afl[y_true == 0]
    afl = p_afl[y_true == 1]
    af_hist, edges = np.histogram(af, bins=bins, density=True)
    afl_hist, _ = np.histogram(afl, bins=bins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "bin_center": bin_centers,
            "af_density": af_hist,
            "afl_density": afl_hist,
        }
    ).to_csv(FIGDIR / "figure_5_probability_distribution_bins.csv", index=False)

    sensitivity = pd.DataFrame(
        [_afl_metrics_at_threshold(y_true, p_afl, float(t)) for t in np.linspace(0.01, 0.99, 199)]
    )
    sensitivity.to_csv(FIGDIR / "figure_5_threshold_sensitivity_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    ax_dist, ax_thr = axes

    ax_dist.hist(
        af,
        bins=bins,
        weights=np.ones_like(af, dtype=float) * 100.0 / max(len(af), 1),
        color="#5b84c4",
        alpha=0.62,
        edgecolor="white",
        linewidth=0.35,
        label=f"AF (n={len(af)})",
    )
    ax_dist.hist(
        afl,
        bins=bins,
        weights=np.ones_like(afl, dtype=float) * 100.0 / max(len(afl), 1),
        color="#dd8452",
        alpha=0.62,
        edgecolor="white",
        linewidth=0.35,
        label=f"AFL (n={len(afl)})",
    )
    ax_dist.axvline(selected_t, color="#202020", lw=1.5, ls="--", label=f"Selected threshold={selected_t:.3f}")
    ax_dist.set_xlim(0, 1)
    ax_dist.set_xlabel("Predicted probability of AFL")
    ax_dist.set_ylabel("Within-class percentage (%)")
    ax_dist.set_title("A. Probability distributions by true class")
    ax_dist.grid(True, axis="y", color="#e8e8e8")
    ax_dist.legend(frameon=False, loc="upper center")

    ax_thr.plot(sensitivity["threshold"], sensitivity["afl_precision"], color="#1f5aa6", lw=2.0, label="AFL precision")
    ax_thr.plot(sensitivity["threshold"], sensitivity["afl_recall"], color="#d55e00", lw=2.0, label="AFL recall")
    ax_thr.plot(sensitivity["threshold"], sensitivity["afl_f1"], color="#009e73", lw=2.2, label="AFL F1")
    ax_thr.plot(sensitivity["threshold"], sensitivity["macro_f1"], color="#7b3294", lw=2.0, label="Macro F1")
    ax_thr.axvline(selected_t, color="#202020", lw=1.5, ls="--")
    ax_thr.set_xlim(0, 1)
    ax_thr.set_ylim(0, 1.02)
    ax_thr.set_xlabel("AFL decision threshold")
    ax_thr.set_ylabel("Metric value")
    ax_thr.set_title("B. Threshold sensitivity of the operating point")
    ax_thr.grid(True, color="#e8e8e8")
    ax_thr.legend(frameon=False, loc="lower center", ncol=2)

    fig.suptitle("Probability-Distribution and Threshold-Sensitivity Diagnostics", fontsize=14, y=1.03)
    for ext in ["png", "pdf"]:
        fig.savefig(FIGDIR / f"figure_5_probability_threshold_diagnostics.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _setup_style()
    plot_threshold_free()
    plot_probability_threshold()
    print(FIGDIR)


if __name__ == "__main__":
    main()
