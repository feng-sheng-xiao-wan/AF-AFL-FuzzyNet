"""Shared training/eval helpers for AF vs AFL baseline scripts."""
from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

import noise_aware_ecg_af_afl_simple as base


def _fast_labels(dataset) -> list[int]:
    if hasattr(dataset, "files"):
        labels = []
        for p in dataset.files:
            with np.load(p, allow_pickle=True) as data:
                labels.append(int(data["label"]))
        return labels
    return [int(dataset[i]["label"]) for i in range(len(dataset))]


def count_label_distribution(dataset, name: str = "") -> Counter:
    c = Counter(_fast_labels(dataset))
    if name:
        print(f"  {name} label distribution: AF(0)={c.get(0, 0)}, AFL(1)={c.get(1, 0)}")
    return c


def build_class_weights(dataset, power: float = 0.5) -> torch.Tensor:
    """Inverse-frequency weights; power<1 reduces minority-class over-correction."""
    labels = _fast_labels(dataset)
    counter = Counter(labels)
    num_classes = max(counter.keys()) + 1
    total = sum(counter.values())
    weights = []
    for c in range(num_classes):
        count = max(counter.get(c, 1), 1)
        w = (total / count) ** float(power)
        weights.append(w)
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)


def make_balanced_train_loader(
    dataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    labels = _fast_labels(dataset)
    counter = Counter(labels)
    sample_weights = [1.0 / counter[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


@torch.no_grad()
def collect_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    forward_fn: Callable[[nn.Module, dict], torch.Tensor],
    criterion: nn.Module | None = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    y_true, p_afl = [], []
    total_loss, total_n = 0.0, 0
    for batch in loader:
        label = batch["label"].to(device)
        logits = forward_fn(model, batch)
        if criterion is not None:
            loss = criterion(logits, label)
            total_n += label.size(0)
            total_loss += loss.item() * label.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1]
        y_true.extend(label.cpu().numpy().tolist())
        p_afl.extend(probs.cpu().numpy().tolist())
    avg_loss = total_loss / max(total_n, 1) if criterion is not None else 0.0
    return np.asarray(y_true, dtype=np.int64), np.asarray(p_afl, dtype=np.float32), avg_loss


def _cm_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    afl_p = tp / (tp + fp + 1e-12)
    afl_r = tp / (tp + fn + 1e-12)
    afl_f1 = 2 * afl_p * afl_r / (afl_p + afl_r + 1e-12)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["AF", "AFL"], digits=4, zero_division=0
    )
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "afl_precision": afl_p,
        "afl_recall": afl_r,
        "afl_f1": afl_f1,
        "cm": cm,
        "report": report,
    }


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    forward_fn: Callable[[nn.Module, dict], torch.Tensor],
    threshold: float | None = None,
    search_threshold: bool = False,
    min_afl_recall: float = 0.2,
) -> Dict:
    y_true, p_afl, avg_loss = collect_probs(model, loader, device, forward_fn, criterion)
    argmax_pred = (p_afl >= 0.5).astype(np.int64)
    m_argmax = _cm_metrics(y_true, argmax_pred)
    thr = 0.5 if threshold is None else float(threshold)
    if search_threshold:
        thr, _ = base._search_best_afl_threshold(y_true, p_afl, min_afl_recall=min_afl_recall)
    thr_pred = (p_afl >= thr).astype(np.int64)
    m_thr = _cm_metrics(y_true, thr_pred)
    return {
        "loss": avg_loss,
        "threshold": thr,
        "y_true": y_true,
        "p_afl": p_afl,
        **{f"argmax_{k}": v for k, v in m_argmax.items() if k not in ("cm", "report")},
        **{f"thr_{k}": v for k, v in m_thr.items() if k not in ("cm", "report")},
        "argmax_cm": m_argmax["cm"],
        "thr_cm": m_thr["cm"],
        "argmax_report": m_argmax["report"],
        "thr_report": m_thr["report"],
    }


def format_eval_line(prefix: str, m: Dict, use_thr: bool = True) -> str:
    if use_thr:
        return (
            f"{prefix} Loss: {m['loss']:.6f} | Acc: {m['thr_acc']:.6f} | "
            f"AFL-P: {m['thr_afl_precision']:.6f} | AFL-R: {m['thr_afl_recall']:.6f} | "
            f"AFL-F1: {m['thr_afl_f1']:.6f} | Macro-F1: {m['thr_macro_f1']:.6f} | "
            f"thr={m['threshold']:.4f}"
        )
    return (
        f"{prefix} Loss: {m['loss']:.6f} | Acc: {m['argmax_acc']:.6f} | "
        f"AFL-F1: {m['argmax_afl_f1']:.6f} | Macro-F1: {m['argmax_macro_f1']:.6f}"
    )
