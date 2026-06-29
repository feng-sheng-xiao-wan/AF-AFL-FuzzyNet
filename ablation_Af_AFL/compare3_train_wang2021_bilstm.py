import os
import sys
import argparse
import random
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import baseline_utils as bu


# =========================================================
# 1. Utils
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fix_length_1d(x, target_len):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) >= target_len:
        return x[:target_len]
    pad = np.zeros(target_len - len(x), dtype=np.float32)
    return np.concatenate([x, pad], axis=0)


# =========================================================
# 2. Dataset
# =========================================================
class ECGNPZDataset(Dataset):
    """
    每个 .npz 文件至少包含：
        ecg: [L] 或 [1, L]
        label: int  (AF vs AFL 二分类)
            0 -> AF
            1 -> AFL
    """

    def __init__(self, root_dir, ecg_len=2500):
        self.root_dir = root_dir
        self.ecg_len = ecg_len
        self.files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".npz")
        ]
        self.files.sort()

        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path, allow_pickle=True)

        if "ecg" not in data:
            raise KeyError(f"{path} missing key 'ecg'")
        if "label" not in data:
            raise KeyError(f"{path} missing key 'label'")

        ecg = np.asarray(data["ecg"], dtype=np.float32)

        if ecg.ndim == 1:
            ecg = fix_length_1d(ecg, self.ecg_len)[None, :]
        elif ecg.ndim == 2:
            if ecg.shape[0] != 1 and ecg.shape[1] == 1:
                ecg = ecg.T
            if ecg.shape[0] != 1:
                ecg = ecg[:1, :]
            ecg = fix_length_1d(ecg[0], self.ecg_len)[None, :]
        else:
            raise ValueError(f"{path} ecg shape invalid: {ecg.shape}")

        label = int(data["label"])

        return {
            "ecg": torch.tensor(ecg, dtype=torch.float32),   # [1, L]
            "label": torch.tensor(label, dtype=torch.long),
            "path": path
        }


# =========================================================
# 3. Model: Wang2021-style CNN + BiLSTM
# =========================================================
class ConvFeatureExtractor(nn.Module):
    def __init__(self, in_ch=1, hidden_ch=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=15, stride=1, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(32, hidden_ch, kernel_size=11, stride=1, padding=5, bias=False),
            nn.BatchNorm1d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(hidden_ch, hidden_ch, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, 1, L]
        return self.net(x)  # [B, C, T]


class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Linear(in_dim, 1)

    def forward(self, x):
        # x: [B, T, D]
        w = torch.softmax(self.attn(x), dim=1)  # [B, T, 1]
        out = torch.sum(w * x, dim=1)           # [B, D]
        return out


class Wang2021BiLSTM(nn.Module):
    """
    ECG -> CNN -> BiLSTM -> Attention -> FC
    """

    def __init__(
        self,
        num_classes=3,
        cnn_hidden=64,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.feature_extractor = ConvFeatureExtractor(
            in_ch=1,
            hidden_ch=cnn_hidden,
            dropout=dropout * 0.5
        )

        self.bilstm = nn.LSTM(
            input_size=cnn_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        self.attn_pool = AttentionPooling(lstm_hidden * 2)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, ecg):
        # ecg: [B, 1, L]
        x = self.feature_extractor(ecg)   # [B, C, T]
        x = x.transpose(1, 2)             # [B, T, C]
        x, _ = self.bilstm(x)             # [B, T, 2H]
        x = self.attn_pool(x)             # [B, 2H]
        logits = self.classifier(x)
        return logits


# =========================================================
# 4. Metrics / Loss
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch=0, print_interval=10):
    model.train()

    total_loss = 0.0
    total_num = 0
    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        ecg = batch["ecg"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(ecg)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        bs = label.size(0)
        total_loss += loss.item() * bs
        total_num += bs

        if print_interval > 0 and (
            (batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == num_batches
        ):
            avg_so_far = total_loss / max(total_num, 1)
            pct = 100.0 * (batch_idx + 1) / num_batches
            print(f"  [Epoch {epoch}] Batch {batch_idx + 1}/{num_batches} ({pct:.1f}%) | loss={avg_so_far:.4f}")

    return total_loss / max(total_num, 1)


# =========================================================
# 5. Main
# =========================================================
def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = ECGNPZDataset(args.train_dir, ecg_len=args.ecg_len)
    val_dataset = ECGNPZDataset(args.val_dir, ecg_len=args.ecg_len)
    test_dataset = ECGNPZDataset(args.test_dir, ecg_len=args.ecg_len) if getattr(args, "test_dir", None) else None

    if args.balanced_sampler:
        train_loader = bu.make_balanced_train_loader(
            train_dataset, args.batch_size, num_workers=args.num_workers, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

    model = Wang2021BiLSTM(
        num_classes=2,  # AF vs AFL
        cnn_hidden=args.cnn_hidden,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout
    ).to(device)

    class_weights = bu.build_class_weights(train_dataset, power=args.class_weight_power).to(device)
    print("Class weights:", class_weights.detach().cpu().numpy())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_afl_f1 = -1.0
    best_threshold = 0.5

    def _forward(model, batch):
        return model(batch["ecg"].to(device))

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, print_interval=getattr(args, "print_interval", 10)
        )
        val_m = bu.evaluate_split(
            model, val_loader, criterion, device, _forward,
            search_threshold=True, min_afl_recall=args.min_afl_recall,
        )
        scheduler.step(val_m["thr_afl_f1"])

        print(f"\nEpoch [{epoch}/{args.epochs}]")
        print(f"Train Loss: {train_loss:.6f}")
        print(bu.format_eval_line("Val  ", val_m, use_thr=True))
        print(f"Val  (argmax) AFL-F1: {val_m['argmax_afl_f1']:.6f}, Macro-F1: {val_m['argmax_macro_f1']:.6f}")
        print(val_m["thr_report"])
        print("Confusion Matrix (AFL-threshold):")
        print(val_m["thr_cm"])

        if val_m["thr_afl_f1"] > best_afl_f1:
            best_afl_f1 = val_m["thr_afl_f1"]
            best_threshold = val_m["threshold"]
            save_path = os.path.join(args.save_dir, "best_wang2021_bilstm.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_afl_f1": best_afl_f1,
                    "afl_threshold": best_threshold,
                    "args": vars(args),
                },
                save_path,
            )
            print(f"Saved best model to: {save_path} (val AFL-F1={best_afl_f1:.6f}, thr={best_threshold:.4f})")

    print(f"\nTraining finished. Best Val AFL-F1 (threshold) = {best_afl_f1:.6f}, thr={best_threshold:.4f}")

    if test_loader is not None:
        best_path = os.path.join(args.save_dir, "best_wang2021_bilstm.pth")
        if os.path.exists(best_path):
            print("\n[TEST] Evaluating best model on test set...")
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            thr = float(ckpt.get("afl_threshold", best_threshold))
            test_m = bu.evaluate_split(model, test_loader, criterion, device, _forward, threshold=thr)
            print(bu.format_eval_line("Test ", test_m, use_thr=True))
            print(f"Test (argmax) AFL-F1: {test_m['argmax_afl_f1']:.6f}, Macro-F1: {test_m['argmax_macro_f1']:.6f}")
            print(test_m["thr_report"])
            print("Test Confusion Matrix (AFL-threshold, rows=true, cols=pred; 0=AF, 1=AFL):")
            print(test_m["thr_cm"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, default=None, help="Optional test set directory")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_wang2021")

    parser.add_argument("--ecg_len", type=int, default=2500)

    parser.add_argument("--cnn_hidden", type=int, default=64)
    parser.add_argument("--lstm_hidden", type=int, default=128)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--print_interval", type=int, default=10, help="每 N 个 batch 打印一次进度，0 表示不打印")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class_weight_power", type=float, default=0.5, help="类别权重指数，<1 减轻少数类过拟合")
    parser.add_argument("--min_afl_recall", type=float, default=0.2, help="验证集阈值搜索时 AFL 最低召回")
    parser.add_argument("--balanced_sampler", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    main(args)