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
import torch.nn.functional as F
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
# 3. Model: ConvNeXt1D-style
# =========================================================
class LayerNorm1D(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, L]
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.weight[:, None] * x + self.bias[:, None]
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class GRN1D(nn.Module):
    """
    Global Response Normalization, ConvNeXtV2 风格简化版
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        # x: [B, C, L]
        gx = torch.norm(x, p=2, dim=2, keepdim=True)          # [B, C, 1]
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)       # [B, C, 1]
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtBlock1D(nn.Module):
    def __init__(self, dim, kernel_size=7, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim
        )
        self.norm = LayerNorm1D(dim)
        self.pwconv1 = nn.Conv1d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN1D(4 * dim)
        self.pwconv2 = nn.Conv1d(4 * dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = identity + self.drop_path(x)
        return x


class DownsampleLayer1D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer = nn.Sequential(
            LayerNorm1D(in_dim),
            nn.Conv1d(in_dim, out_dim, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)


class ConvNeXt1D(nn.Module):
    """
    ECG -> ConvNeXt1D -> classifier
    """

    def __init__(
        self,
        num_classes=3,
        dims=(32, 64, 128, 256),
        depths=(2, 2, 3, 2),
        drop_path_rate=0.1,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size=7, stride=2, padding=3),
            LayerNorm1D(dims[0])
        )

        total_blocks = sum(depths)
        dp_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        dp_idx = 0

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(len(dims)):
            blocks = []
            for _ in range(depths[i]):
                blocks.append(
                    ConvNeXtBlock1D(
                        dim=dims[i],
                        kernel_size=7,
                        drop_path=dp_rates[dp_idx]
                    )
                )
                dp_idx += 1
            self.stages.append(nn.Sequential(*blocks))

            if i < len(dims) - 1:
                self.downsamples.append(DownsampleLayer1D(dims[i], dims[i + 1]))

        self.head_norm = LayerNorm1D(dims[-1])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dims[-1], 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, ecg):
        # ecg: [B, 1, L]
        x = self.stem(ecg)

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        x = self.head_norm(x)
        x = self.pool(x)
        logits = self.fc(x)
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

    model = ConvNeXt1D(
        num_classes=2,  # AF vs AFL
        dims=(args.dim1, args.dim2, args.dim3, args.dim4),
        depths=(args.depth1, args.depth2, args.depth3, args.depth4),
        drop_path_rate=args.drop_path_rate
    ).to(device)

    class_weights = bu.build_class_weights(train_dataset, power=args.class_weight_power).to(device)
    print("Class weights:", class_weights.detach().cpu().numpy())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
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
            save_path = os.path.join(args.save_dir, "best_kraft2025_convnext1d.pth")
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
        best_path = os.path.join(args.save_dir, "best_kraft2025_convnext1d.pth")
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
    parser.add_argument("--save_dir", type=str, default="./checkpoints_kraft2025")

    parser.add_argument("--ecg_len", type=int, default=2500)

    parser.add_argument("--dim1", type=int, default=32)
    parser.add_argument("--dim2", type=int, default=64)
    parser.add_argument("--dim3", type=int, default=128)
    parser.add_argument("--dim4", type=int, default=256)

    parser.add_argument("--depth1", type=int, default=2)
    parser.add_argument("--depth2", type=int, default=2)
    parser.add_argument("--depth3", type=int, default=3)
    parser.add_argument("--depth4", type=int, default=2)

    parser.add_argument("--drop_path_rate", type=float, default=0.1)

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