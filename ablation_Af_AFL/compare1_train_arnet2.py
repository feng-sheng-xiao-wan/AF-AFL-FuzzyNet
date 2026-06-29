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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

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
class RRNPZDataset(Dataset):
    """
    每个 .npz 文件至少包含：
        rr: shape [T]
        label: int  (AF vs AFL 二分类，与 prepare_rr_npz 一致)
            0 -> AF
            1 -> AFL
    """

    def __init__(self, root_dir, rr_len=128):
        self.root_dir = root_dir
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

        if "rr" not in data:
            raise KeyError(f"{path} missing key 'rr'")
        if "label" not in data:
            raise KeyError(f"{path} missing key 'label'")

        rr = fix_length_1d(data["rr"], self.rr_len)
        label = int(data["label"])

        return {
            "rr": torch.tensor(rr, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "path": path
        }


# =========================================================
# 3. Model: ArNet2-style
# =========================================================
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, T, H]
        weights = torch.softmax(self.attn(x), dim=1)   # [B, T, 1]
        context = torch.sum(weights * x, dim=1)        # [B, H]
        return context


class ArNet2(nn.Module):
    """
    ArNet2-style RR interval baseline
    Input:
        rr: [B, T]
    Output:
        logits: [B, num_classes]
    """

    def __init__(
        self,
        input_dim=1,
        hidden_dim=128,
        num_layers=2,
        num_classes=3,
        dropout=0.2,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = AttentionLayer(hidden_dim * 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, rr):
        # rr: [B, T]
        x = rr.unsqueeze(-1)           # [B, T, 1]
        out, _ = self.gru(x)           # [B, T, 2H]
        context = self.attention(out)  # [B, 2H]
        logits = self.classifier(context)
        return logits


# =========================================================
# 4. Metrics / Loss
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
        rr = batch["rr"].to(device)
        label = batch["label"].to(device)

        logits = model(rr)
        loss = criterion(logits, label)

        pred = torch.argmax(logits, dim=1)

        bs = label.size(0)
        total_loss += loss.item() * bs
        total_num += bs

        y_true.extend(label.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())

    avg_loss = total_loss / max(total_num, 1)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    # AF vs AFL 二分类：0=AF, 1=AFL
    labels = [0, 1]
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=["AF", "AFL"],
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    afl_f1 = f1_score(y_true, y_pred, labels=[1], average="macro")

    return avg_loss, acc, macro_f1, afl_f1, report, cm


def train_one_epoch(model, loader, criterion, optimizer, device, epoch=0, print_interval=10):
    model.train()

    total_loss = 0.0
    total_num = 0
    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        rr = batch["rr"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(rr)
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

    train_dataset = RRNPZDataset(args.train_dir, rr_len=args.rr_len)
    val_dataset = RRNPZDataset(args.val_dir, rr_len=args.rr_len)
    test_dataset = RRNPZDataset(args.test_dir, rr_len=args.rr_len) if getattr(args, "test_dir", None) else None

    # 打印标签分布，便于核对 AFL 是否有样本（若 Val AFL 全 0 多为 RR 基线坍缩到多数类或数据问题）
    bu.count_label_distribution(train_dataset, "Train")
    bu.count_label_distribution(val_dataset, "Val")

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
        bu.count_label_distribution(test_dataset, "Test")

    model = ArNet2(
        input_dim=1,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=2,  # AF vs AFL
        dropout=args.dropout,
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

    def _forward(model, batch):
        return model(batch["rr"].to(device))

    os.makedirs(args.save_dir, exist_ok=True)
    best_afl_f1 = -1.0
    best_threshold = 0.5

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
            save_path = os.path.join(args.save_dir, "best_arnet2.pth")
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
        best_path = os.path.join(args.save_dir, "best_arnet2.pth")
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
    parser.add_argument("--test_dir", type=str, default=None, help="Optional test set directory (RR npz)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_arnet2")

    parser.add_argument("--rr_len", type=int, default=128)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
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