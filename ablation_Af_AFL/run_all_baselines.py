"""
一键运行五个基线训练，并汇总 Val/Test 的 Acc、Macro F1、AFL-F1、每类 P/R/F1 到 CSV。
在项目根目录执行：python ablation_Af_AFL/run_all_baselines.py [--skip_data_prep]
"""
import os
import re
import sys
import csv
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

# 项目根目录
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 默认路径（相对项目根）
DEFAULT_CSV = "data/precompute_features/multi_dataset_with_afl_noise_aug.csv"
RR_NPZ_ROOT = "data/ablation_rr_npz"
ECG_NPZ_ROOT = "ablation_Af_AFL/rawecg_npz"
ECG_RR_NPZ_ROOT = "ablation_Af_AFL/fleury_ecg_rr_npz"
OUT_CSV = SCRIPT_DIR / "baseline_metrics.csv"
LOGS_DIR = SCRIPT_DIR / "logs"


def _run(cmd: list, name: str, log_path: Path) -> tuple[str, int]:
    """执行命令，流式写入日志，返回 (stdout+stderr, returncode)。"""
    os.makedirs(log_path.parent, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Command: {' '.join(cmd)}\n\n")
        f.flush()
        p = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        chunks: list[str] = []
        assert p.stdout is not None
        for line in p.stdout:
            chunks.append(line)
            f.write(line)
            f.flush()
        ret = p.wait()
    return "".join(chunks), ret


def _parse_val_metrics(text: str) -> dict:
    """从训练输出中解析最后一次 Val 指标。"""
    d = {
        "val_loss": "",
        "val_acc": "",
        "val_macro_f1": "",
        "val_afl_f1": "",
    }
    # Val   Loss: 0.xxx（取最后一次）
    for m in re.finditer(r"Val\s+Loss:\s*([\d.]+)", text):
        d["val_loss"] = m.group(1)
    # Val Acc: 0.xxx, Macro F1: 0.xxx, AFL-F1: 0.xxx（取最后一次）
    for m in re.finditer(r"Val\s+Acc:\s*([\d.]+).*?Macro F1:\s*([\d.]+).*?AFL-F1:\s*([\d.]+)", text, re.DOTALL):
        d["val_acc"], d["val_macro_f1"], d["val_afl_f1"] = m.group(1), m.group(2), m.group(3)
    # 仅有 Val Macro F1（compare3/4/5 的 epoch 行，取最后一次）
    if not d["val_macro_f1"]:
        for m in re.finditer(r"Val Macro F1:\s*([\d.]+),\s*AFL-F1:\s*([\d.]+)", text):
            d["val_macro_f1"], d["val_afl_f1"] = m.group(1), m.group(2)
    return d


def _parse_test_metrics(text: str) -> dict:
    """从训练输出中解析 Test 指标（AFL-threshold 协议）。"""
    d = {
        "test_loss": "",
        "test_acc": "",
        "test_macro_f1": "",
        "test_afl_f1": "",
        "test_afl_precision": "",
        "test_afl_recall": "",
        "test_threshold": "",
        "test_argmax_afl_f1": "",
        "test_AF_precision": "",
        "test_AF_recall": "",
        "test_AF_f1": "",
        "test_AFL_precision": "",
        "test_AFL_recall": "",
        "test_AFL_f1": "",
    }
    # Test  Loss: ... | Acc: ... | AFL-P: ... | AFL-R: ... | AFL-F1: ... | Macro-F1: ... | thr=...
    m = re.search(
        r"Test\s+Loss:\s*([\d.]+).*?Acc:\s*([\d.]+).*?AFL-P:\s*([\d.]+).*?AFL-R:\s*([\d.]+).*?AFL-F1:\s*([\d.]+).*?Macro-F1:\s*([\d.]+).*?thr=([\d.]+)",
        text,
        re.DOTALL,
    )
    if m:
        d["test_loss"], d["test_acc"] = m.group(1), m.group(2)
        d["test_afl_precision"], d["test_afl_recall"], d["test_afl_f1"] = m.group(3), m.group(4), m.group(5)
        d["test_macro_f1"], d["test_threshold"] = m.group(6), m.group(7)
        d["test_AFL_precision"], d["test_AFL_recall"], d["test_AFL_f1"] = m.group(3), m.group(4), m.group(5)
    m2 = re.search(r"Test \(argmax\) AFL-F1:\s*([\d.]+)", text)
    if m2:
        d["test_argmax_afl_f1"] = m2.group(1)
    # 解析 classification report AF/AFL 行（辅助）
    blocks = text.split("Test Confusion Matrix")
    if blocks:
        last_block = blocks[-1]
        for label in ("AF", "AFL"):
            pat = rf"\s+{label}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
            m = re.search(pat, last_block)
            if m and label == "AF":
                d["test_AF_precision"], d["test_AF_recall"], d["test_AF_f1"] = m.group(1), m.group(2), m.group(3)
    return d


def main():
    parser = argparse.ArgumentParser(description="依次训练五个基线并汇总指标到 CSV")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="数据 CSV 路径（相对项目根）")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--skip_data_prep", action="store_true", help="跳过数据准备，仅跑训练")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--print_interval", type=int, default=10, help="每 N 个 batch 打印一次进度，0 不打印")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path
    if not args.skip_data_prep and not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = LOGS_DIR / ts
    os.makedirs(logs_dir, exist_ok=True)

    rows = []
    configs = [
        ("ArNet2", "compare1_train_arnet2.py", [
            "python", "ablation_Af_AFL/prepare_rr_npz_from_csv.py",
            "--combined_csv", str(csv_path), "--output_root", RR_NPZ_ROOT,
        ], [
            "python", "ablation_Af_AFL/compare1_train_arnet2.py",
            "--train_dir", f"{RR_NPZ_ROOT}/train", "--val_dir", f"{RR_NPZ_ROOT}/val",
            "--test_dir", f"{RR_NPZ_ROOT}/test",
            "--save_dir", "ablation_Af_AFL/checkpoints_arnet2",
            "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers), "--print_interval", str(args.print_interval),
        ], False),
        ("RawECGNet", "compare2_train_rawecgnet.py", [
            "python", "ablation_Af_AFL/prepare_ecg_npz_from_csv.py",
            "--combined_csv", str(csv_path), "--output_root", ECG_NPZ_ROOT,
        ], [
            "python", "ablation_Af_AFL/compare2_train_rawecgnet.py",
            "--train_dir", f"{ECG_NPZ_ROOT}/train", "--val_dir", f"{ECG_NPZ_ROOT}/val",
            "--test_dir", f"{ECG_NPZ_ROOT}/test", "--save_dir", "ablation_Af_AFL/checkpoints_rawecgnet",
            "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers), "--print_interval", str(args.print_interval),
        ], True),
        ("Wang2021_BiLSTM", "compare3_train_wang2021_bilstm.py", None, [
            "python", "ablation_Af_AFL/compare3_train_wang2021_bilstm.py",
            "--train_dir", f"{ECG_NPZ_ROOT}/train", "--val_dir", f"{ECG_NPZ_ROOT}/val",
            "--test_dir", f"{ECG_NPZ_ROOT}/test", "--save_dir", "ablation_Af_AFL/checkpoints_wang2021",
            "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers), "--print_interval", str(args.print_interval),
        ], True),
        ("Kraft2025_ConvNeXt1D", "compare4_train_kraft2025_convnext1d.py", None, [
            "python", "ablation_Af_AFL/compare4_train_kraft2025_convnext1d.py",
            "--train_dir", f"{ECG_NPZ_ROOT}/train", "--val_dir", f"{ECG_NPZ_ROOT}/val",
            "--test_dir", f"{ECG_NPZ_ROOT}/test", "--save_dir", "ablation_Af_AFL/checkpoints_kraft2025",
            "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers), "--print_interval", str(args.print_interval),
        ], True),
        ("Fleury2024_Modular", "compare5_train_fleury2024_modular.py", [
            "python", "ablation_Af_AFL/prepare_ecg_rr_npz_from_csv.py",
            "--combined_csv", str(csv_path), "--output_root", ECG_RR_NPZ_ROOT,
        ], [
            "python", "ablation_Af_AFL/compare5_train_fleury2024_modular.py",
            "--train_dir", f"{ECG_RR_NPZ_ROOT}/train", "--val_dir", f"{ECG_RR_NPZ_ROOT}/val",
            "--test_dir", f"{ECG_RR_NPZ_ROOT}/test", "--save_dir", "ablation_Af_AFL/checkpoints_fleury2024",
            "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers), "--print_interval", str(args.print_interval),
        ], True),
    ]

    # 数据准备：只跑一次 ECG 和 ECG+RR（RR 在 ArNet2 前跑）
    data_prep_done = {"rr": False, "ecg": False, "ecg_rr": False}
    for name, script, prep_cmd, train_cmd, needs_ecg in configs:
        print(f"\n{'='*60}\n[{name}]\n{'='*60}")
        log_path = logs_dir / f"{name.replace(' ', '_')}.log"
        full_out = ""

        if not args.skip_data_prep:
            if "arnet2" in script.lower() or "compare1" in script:
                if not data_prep_done["rr"]:
                    out, ret = _run(prep_cmd, "prepare_rr", logs_dir / "prep_rr.log")
                    full_out += out
                    data_prep_done["rr"] = True
                    if ret != 0:
                        print(f"  [FAIL] prepare_rr_npz returncode={ret}")
                        rows.append({"baseline": name, "error": "data_prep_rr_failed"})
                        continue
            elif "fleury" in name.lower() or "compare5" in script:
                if not data_prep_done["ecg_rr"]:
                    out, ret = _run(prep_cmd, "prepare_ecg_rr", logs_dir / "prep_ecg_rr.log")
                    full_out += out
                    data_prep_done["ecg_rr"] = True
                    if ret != 0:
                        print(f"  [FAIL] prepare_ecg_rr returncode={ret}")
                        rows.append({"baseline": name, "error": "data_prep_ecg_rr_failed"})
                        continue
            elif needs_ecg and not data_prep_done["ecg"]:
                # RawECGNet 的 prep 会生成 ECG npz，后面 Wang/Kraft 共用
                prep_cmd_ecg = [
                    "python", "ablation_Af_AFL/prepare_ecg_npz_from_csv.py",
                    "--combined_csv", str(csv_path), "--output_root", ECG_NPZ_ROOT,
                ]
                out, ret = _run(prep_cmd_ecg, "prepare_ecg", logs_dir / "prep_ecg.log")
                full_out += out
                data_prep_done["ecg"] = True
                if ret != 0:
                    print(f"  [FAIL] prepare_ecg returncode={ret}")
                    rows.append({"baseline": name, "error": "data_prep_ecg_failed"})
                    continue

        out, ret = _run(train_cmd, name, log_path)
        full_out += out
        if ret != 0:
            print(f"  [FAIL] training returncode={ret}. See {log_path}")
            rows.append({"baseline": name, "error": f"train_exit_{ret}"})
        else:
            val_d = _parse_val_metrics(full_out)
            test_d = _parse_test_metrics(full_out)
            row = {
                "baseline": name,
                "error": "",
                **val_d,
                **test_d,
            }
            rows.append(row)
            print(f"  Val  -> Acc: {val_d['val_acc'] or 'N/A'}, Macro F1: {val_d['val_macro_f1'] or 'N/A'}, AFL-F1: {val_d['val_afl_f1'] or 'N/A'}")
            print(f"  Test -> Acc: {test_d['test_acc'] or 'N/A'}, Macro F1: {test_d['test_macro_f1'] or 'N/A'}, AFL-F1: {test_d['test_afl_f1'] or 'N/A'}")

    # 写 CSV
    fieldnames = [
        "baseline", "error",
        "val_loss", "val_acc", "val_macro_f1", "val_afl_f1",
        "test_loss", "test_acc", "test_macro_f1", "test_afl_f1",
        "test_afl_precision", "test_afl_recall", "test_threshold", "test_argmax_afl_f1",
        "test_AF_precision", "test_AF_recall", "test_AF_f1",
        "test_AFL_precision", "test_AFL_recall", "test_AFL_f1",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # 同时写一份带时间戳的备份，便于多次运行对比
    backup_csv = SCRIPT_DIR / f"baseline_metrics_{ts}.csv"
    with open(backup_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\n{'='*60}")
    print(f"指标已保存: {OUT_CSV}")
    print(f"各基线完整日志: {logs_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
