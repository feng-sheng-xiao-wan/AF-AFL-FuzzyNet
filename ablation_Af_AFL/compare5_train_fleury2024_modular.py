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
class ModularECGDataset(Dataset):
    """
    每个 .npz 文件至少包含：
        ecg: [L] 或 [1, L]
        rr: [T] (可选，没有则补零)
        label: int  (AF vs AFL 二分类)
            0 -> AF
            1 -> AFL
    """

    def __init__(self, root_dir, ecg_len=2500, rr_len=128):
        self.root_dir = root_dir
        self.ecg_len = ecg_len
        self.rr_len = rr_len
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

        if "rr" in data:
            rr = fix_length_1d(data["rr"], self.rr_len)
        else:
            rr = np.zeros(self.rr_len, dtype=np.float32)

        label = int(data["label"])

        return {
            "ecg": torch.tensor(ecg, dtype=torch.float32),   # [1, L]
            "rr": torch.tensor(rr, dtype=torch.float32),     # [T]
            "label": torch.tensor(label, dtype=torch.long),
            "path": path
        }


# =========================================================
# 3. Building Blocks
# =========================================================
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=None, dropout=0.0):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Linear(in_dim, 1)

    def forward(self, x):
        # x: [B, T, D]
        w = torch.softmax(self.attn(x), dim=1)
        out = torch.sum(w * x, dim=1)
        return out


# =========================================================
# 4. Branch 1: Rhythm Regularity Module
# =========================================================
class RhythmBranch(nn.Module):
    """
    RR sequence -> BiGRU -> Attention -> feature
    """
    def __init__(self, rr_hidden=64, out_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=rr_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn = AttentionPooling(rr_hidden * 2)
        self.fc = nn.Sequential(
            nn.Linear(rr_hidden * 2, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, rr):
        # rr: [B, T]
        x = rr.unsqueeze(-1)       # [B, T, 1]
        x, _ = self.gru(x)         # [B, T, 2H]
        x = self.attn(x)           # [B, 2H]
        x = self.fc(x)             # [B, out_dim]
        return x


# =========================================================
# 5. Branch 2: Atrial Activity Module
# =========================================================
class AtrialActivityBranch(nn.Module):
    """
    强调较细粒度房性活动纹理
    """
    def __init__(self, out_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(1, 32, kernel_size=5, stride=1, dropout=dropout),
            ConvBNAct(32, 64, kernel_size=5, stride=1, dropout=dropout),
            nn.MaxPool1d(2),

            ConvBNAct(64, 128, kernel_size=3, stride=1, dropout=dropout),
            ConvBNAct(128, 128, kernel_size=3, stride=1, dropout=dropout),
            nn.MaxPool1d(2),

            ConvBNAct(128, 128, kernel_size=3, stride=1, dropout=dropout),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, ecg):
        x = self.net(ecg)
        x = self.fc(x)
        return x


# =========================================================
# 6. Branch 3: Raw Voltage Temporal Module
# =========================================================
class RawVoltageBranch(nn.Module):
    """
    原始 ECG 粗粒度时序形态
    """
    def __init__(self, out_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(1, 32, kernel_size=15, stride=2, dropout=dropout),
            ConvBNAct(32, 64, kernel_size=11, stride=1, dropout=dropout),
            nn.MaxPool1d(2),

            ConvBNAct(64, 128, kernel_size=9, stride=1, dropout=dropout),
            nn.MaxPool1d(2),

            ConvBNAct(128, 128, kernel_size=7, stride=1, dropout=dropout),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, ecg):
        x = self.net(ecg)
        x = self.fc(x)
        return x


# =========================================================
# 7. Fleury2024-style Modular Model
# =========================================================
class Fleury2024Modular(nn.Module):
    """
    Rhythm + Atrial Activity + Raw Voltage
    """
    def __init__(
        self,
        num_classes=3,
        rhythm_dim=128,
        atrial_dim=128,
        raw_dim=128,
        rr_hidden=64,
        dropout=0.2
    ):
        super().__init__()

        self.rhythm_branch = RhythmBranch(
            rr_hidden=rr_hidden,
            out_dim=rhythm_dim,
            num_layers=2,
            dropout=dropout
        )
        self.atrial_branch = AtrialActivityBranch(
            out_dim=atrial_dim,
            dropout=dropout * 0.5
        )
        self.raw_branch = RawVoltageBranch(
            out_dim=raw_dim,
            dropout=dropout * 0.5
        )

        fusion_dim = rhythm_dim + atrial_dim + raw_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, ecg, rr):
        z_rhythm = self.rhythm_branch(rr)
        z_atrial = self.atrial_branch(ecg)
        z_raw = self.raw_branch(ecg)

        z = torch.cat([z_rhythm, z_atrial, z_raw], dim=1)
        logits = self.classifier(z)
        return logits


# =========================================================
# 8. Metrics / Loss
# =========================================================
def build_class_weights(dataset):
    labels = []
    for i in range(len(dataset)):
        labels.append(int(dataset[i]["label"]))
    counter = Counter(labels)

    num_classes = max(counter.keys()) + 1
    total = sum(counter.values())

    weights = []
    for c in range(num_classes):
        count = counter.get(c, 1)
        weights.append(total / count)

    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_num = 0
    y_true = []
    y_pred = []

    for batch in loader:
        ecg = batch["ecg"].to(device)
        rr = batch["rr"].to(device)
        label = batch["label"].to(device)

        logits = model(ecg, rr)
        loss = criterion(logits, label)
        pred = torch.argmax(logits, dim=1)

        bs = label.size(0)
        total_loss += loss.item() * bs
        total_num += bs

        y_true.extend(label.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())

    avg_loss = total_loss / max(total_num, 1)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    labels_binary = [0, 1]
    afl_f1 = f1_score(y_true, y_pred, labels=[1], average="macro")
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_binary,
        target_names=["AF", "AFL"],
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_binary)

    return avg_loss, macro_f1, afl_f1, report, cm


def train_one_epoch(model, loader, criterion, optimizer, device, epoch=0, print_interval=10):
    model.train()

    total_loss = 0.0
    total_num = 0
    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        ecg = batch["ecg"].to(device)
        rr = batch["rr"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(ecg, rr)
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
# 9. Main
# =========================================================
def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = ModularECGDataset(
        args.train_dir,
        ecg_len=args.ecg_len,
        rr_len=args.rr_len
    )
    val_dataset = ModularECGDataset(
        args.val_dir,
        ecg_len=args.ecg_len,
        rr_len=args.rr_len
    )
    test_dataset = None
    if getattr(args, "test_dir", None):
        test_dataset = ModularECGDataset(
            args.test_dir,
            ecg_len=args.ecg_len,
            rr_len=args.rr_len
        )

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

    model = Fleury2024Modular(
        num_classes=2,  # AF vs AFL
        rhythm_dim=args.rhythm_dim,
        atrial_dim=args.atrial_dim,
        raw_dim=args.raw_dim,
        rr_hidden=args.rr_hidden,
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
        return model(batch["ecg"].to(device), batch["rr"].to(device))

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
            save_path = os.path.join(args.save_dir, "best_fleury2024_modular.pth")
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
        best_path = os.path.join(args.save_dir, "best_fleury2024_modular.pth")
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
    parser.add_argument("--save_dir", type=str, default="./checkpoints_fleury2024")

    parser.add_argument("--ecg_len", type=int, default=2500)
    parser.add_argument("--rr_len", type=int, default=128)

    parser.add_argument("--rhythm_dim", type=int, default=128)
    parser.add_argument("--atrial_dim", type=int, default=128)
    parser.add_argument("--raw_dim", type=int, default=128)
    parser.add_argument("--rr_hidden", type=int, default=64)
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