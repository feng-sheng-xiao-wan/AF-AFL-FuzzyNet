# -*- coding: utf-8 -*-
"""
一键运行消融实验 05-10，并自动保存日志与四个总体指标汇总（Acc/Precision/Recall/F1，macro）。

覆盖实验：
- 05 取消模糊规则（Full - Fuzzy）
- 06 只用模糊规则（Fuzzy-only，no training）
- 07 逐条删除分支：去 RR
- 08 逐条删除分支：去形态（ECG）
- 09 取消重采样（仅 oversampling 关闭）
- 10 取消数据增强：无重采样 + 使用“不含噪声扩增”的 CSV

输出：
- 每个实验一个 log：ablation_Af_AFL/ablation_runs_05_10/<timestamp>/<exp_name>.log
- 汇总 CSV：ablation_Af_AFL/ablation_runs_05_10/<timestamp>/ablation_05_10_metrics.csv
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


TEST_LINE_RE = re.compile(
    r"Test:\s+loss=[0-9.]+,\s+acc=(?P<acc>[0-9.]+),\s+F1=(?P<f1>[0-9.]+),\s+P=(?P<p>[0-9.]+),\s+R=(?P<r>[0-9.]+)"
)

FUZZY_ONLY_RE = {
    "acc": re.compile(r"^Accuracy:\s+(?P<v>[0-9.]+)\s*$", re.MULTILINE),
    "p": re.compile(r"^Precision \(macro\):\s+(?P<v>[0-9.]+)\s*$", re.MULTILINE),
    "r": re.compile(r"^Recall \(macro\):\s+(?P<v>[0-9.]+)\s*$", re.MULTILINE),
    "f1": re.compile(r"^F1 \(macro\):\s+(?P<v>[0-9.]+)\s*$", re.MULTILINE),
}


def _run_and_log(cmd, log_path: Path, cwd: Path) -> tuple[int, str]:
    """
    运行子脚本：
    - 子进程 stdout/stderr 实时打印到当前终端
    - 同步写入 log_path
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out_chunks = []
    with open(log_path, "w", encoding="utf-8", newline="\n") as f:
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                out_chunks.append(line)
                f.write(line)
                f.flush()
                # 同时打印到终端（实现“打印 + 保存”）
                sys.stdout.write(line)
                sys.stdout.flush()
    rc = int(proc.wait())
    out_text = "".join(out_chunks)
    return rc, out_text


def _parse_metrics_from_log(text: str, exp_name: str) -> dict:
    """
    返回 dict: test_acc/test_precision/test_recall/test_f1（均为 macro 口径）或空字符串。
    """
    if exp_name == "06_fuzzy_only":
        def _pick(key: str) -> str:
            m = FUZZY_ONLY_RE[key].search(text)
            return m.group("v") if m else ""

        return {
            "test_acc": _pick("acc"),
            "test_precision": _pick("p"),
            "test_recall": _pick("r"),
            "test_f1": _pick("f1"),
        }

    last = None
    for m in TEST_LINE_RE.finditer(text):
        last = m
    if not last:
        return {"test_acc": "", "test_precision": "", "test_recall": "", "test_f1": ""}
    return {
        "test_acc": last.group("acc"),
        "test_precision": last.group("p"),
        "test_recall": last.group("r"),
        "test_f1": last.group("f1"),
    }


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments 05-10 (one click) and save metrics.")
    parser.add_argument(
        "--csv_aug",
        type=str,
        required=True,
        help="带噪声扩增的 CSV（用于 05/06/07/08/09），例如 data/precompute_features/multi_dataset_with_afl_noise_aug.csv",
    )
    parser.add_argument(
        "--csv_noaug",
        type=str,
        required=True,
        help="不含噪声扩增的 CSV（用于 10），例如 data/precompute_features/multi_dataset_combined_with_features.csv",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0, help="透传给训练脚本（如支持）")
    parser.add_argument("--device", type=str, default="auto", help="透传给训练脚本（如支持）")
    parser.add_argument(
        "--out_root",
        type=str,
        default="ablation_Af_AFL/ablation_runs_05_10",
        help="输出根目录（log 与汇总 CSV 放这里）",
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (PROJECT_ROOT / args.out_root / ts).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runner_log = out_dir / "runner.log"
    runner_fp = open(runner_log, "w", encoding="utf-8", newline="\n")

    exp_defs = [
        # 使用 -u 确保子进程 stdout 行缓冲，便于实时写 log
        (
            "05_no_fuzzy",
            [
                "python",
                "-u",
                "ablation_Af_AFL/05_full_no_fuzzy.py",
                "--combined_csv",
                args.csv_aug,
                "--output_dir",
                "outputs/ablation/05_full_no_fuzzy",
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--device",
                str(args.device),
            ],
        ),
        ("06_fuzzy_only", ["python", "-u", "ablation_Af_AFL/06_fuzzy_only_eval.py", "--combined_csv", args.csv_aug]),
        (
            "07_no_rr",
            [
                "python",
                "-u",
                "ablation_Af_AFL/07_full_no_rr_branch.py",
                "--combined_csv",
                args.csv_aug,
                "--output_dir",
                "outputs/ablation/07_full_no_rr",
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--device",
                str(args.device),
            ],
        ),
        (
            "08_no_morph",
            [
                "python",
                "-u",
                "ablation_Af_AFL/08_full_no_morph_branch.py",
                "--combined_csv",
                args.csv_aug,
                "--output_dir",
                "outputs/ablation/08_full_no_morph",
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--device",
                str(args.device),
            ],
        ),
        (
            "09_no_oversampling",
            [
                "python",
                "-u",
                "ablation_Af_AFL/09_full_no_oversampling.py",
                "--combined_csv",
                args.csv_aug,
                "--output_dir",
                "outputs/ablation/09_full_no_oversampling",
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--device",
                str(args.device),
            ],
        ),
        (
            "10_no_data_aug",
            [
                "python",
                "-u",
                "ablation_Af_AFL/10_full_no_data_aug.py",
                "--combined_csv",
                args.csv_noaug,
                "--output_dir",
                "outputs/ablation/10_full_no_data_aug",
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--device",
                str(args.device),
            ],
        ),
    ]

    def _log(msg: str):
        if not msg.endswith("\n"):
            msg += "\n"
        sys.stdout.write(msg)
        sys.stdout.flush()
        runner_fp.write(msg)
        runner_fp.flush()

    _log(f"Runner started. out_dir={out_dir}")
    _log(f"csv_aug={args.csv_aug}")
    _log(f"csv_noaug={args.csv_noaug}")
    _log(f"device={args.device}, epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

    rows = []
    for exp_name, cmd in exp_defs:
        log_path = out_dir / f"{exp_name}.log"
        _log(f"\n=== Running {exp_name} ===")
        _log("CMD: " + " ".join(cmd))
        rc, out_text = _run_and_log(cmd, log_path=log_path, cwd=PROJECT_ROOT)
        metrics = _parse_metrics_from_log(out_text, exp_name)
        row = {
            "exp": exp_name,
            "return_code": str(rc),
            "log_path": str(log_path),
            **metrics,
        }
        if rc != 0:
            row["error"] = "nonzero_return_code"
        else:
            row["error"] = ""
        rows.append(row)
        _log(
            f"[{exp_name}] rc={rc} | "
            f"acc={row['test_acc'] or 'N/A'} p={row['test_precision'] or 'N/A'} "
            f"r={row['test_recall'] or 'N/A'} f1={row['test_f1'] or 'N/A'}",
        )

    csv_path = out_dir / "ablation_05_10_metrics.csv"
    fieldnames = ["exp", "error", "return_code", "test_acc", "test_precision", "test_recall", "test_f1", "log_path"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    _log("\nDone.")
    _log("Logs & CSV saved to: " + str(out_dir))
    _log("Summary CSV: " + str(csv_path))
    runner_fp.close()


if __name__ == "__main__":
    main()

