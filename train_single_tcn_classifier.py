#!/usr/bin/env python3
"""
Single-model full-sequence TCN classifier (no distillation).

Designed for datasets produced by prepare_weld_seam_dataset.py:
- X_train_full: [N_train, T, C]
- y_train: [N_train]
- X_test_full: [N_test, T, C]
- y_test: [N_test]
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

try:
    from sklearn.metrics import classification_report as sk_classification_report
except Exception:
    sk_classification_report = None

try:
    from .models_tcn import TeacherClassifierTCNFull
except ImportError:
    from models_tcn import TeacherClassifierTCNFull


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + x)


class AttentionTCNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        channels: int = 64,
        tcn_layers: int = 3,
        tcn_kernel: int = 3,
        tcn_dropout: float = 0.15,
        dilation_base: int = 2,
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
        ff_dim: int = 128,
        classifier_hidden: int = 128,
        classifier_dropout: float = 0.35,
    ) -> None:
        super().__init__()
        # 将原始输入映射到指定的通道数，以便后续的TCN块处理。
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        # 膨胀卷积：第一层看 t, t+1, t+2；第二层看 t, t+2, t+4；第三层看 t, t+4, t+8
        self.tcn = nn.ModuleList(
            [
                ResidualTCNBlock(
                    channels=channels,
                    kernel_size=tcn_kernel,
                    dilation=dilation_base**i,
                    dropout=tcn_dropout,
                )
                for i in range(max(tcn_layers, 1))
            ]
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(channels)
        # Feed forward
        self.ff = nn.Sequential(
            nn.Linear(channels, ff_dim),
            nn.GELU(),
            nn.Dropout(attn_dropout),
            nn.Linear(ff_dim, channels),
        )
        self.ln2 = nn.LayerNorm(channels)
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(channels * 2, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, C] -> [B, C, T]
        h = self.input_proj(x.transpose(1, 2))
        for blk in self.tcn:
            h = blk(h)
        # [B, C, T] -> [B, T, C]
        h = h.transpose(1, 2)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        # 两次残差网络
        h = self.ln1(h + attn_out)
        h = self.ln2(h + self.ff(h))
        # Use mean+max pooling for short-window robustness.
        h_mean = h.mean(dim=1)
        h_max = h.max(dim=1).values
        z = torch.cat([h_mean, h_max], dim=1)
        logits = self.classifier(z)
        return logits, z


class InceptionBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bottleneck: int, dropout: float) -> None:
        super().__init__()
        self.use_bottleneck = in_ch > bottleneck
        self.bottleneck = nn.Conv1d(in_ch, bottleneck, kernel_size=1, bias=False) if self.use_bottleneck else nn.Identity()
        branch_in = bottleneck if self.use_bottleneck else in_ch
        self.b1 = nn.Conv1d(branch_in, out_ch, kernel_size=1, padding=0, bias=False)
        self.b3 = nn.Conv1d(branch_in, out_ch, kernel_size=3, padding=1, bias=False)
        self.b5 = nn.Conv1d(branch_in, out_ch, kernel_size=5, padding=2, bias=False)
        self.bp = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
        )
        self.bn = nn.BatchNorm1d(out_ch * 4)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        out = torch.cat([self.b1(z), self.b3(z), self.b5(z), self.bp(x)], dim=1)
        out = self.bn(out)
        out = self.act(out)
        return self.drop(out)


class InceptionTimeClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        out_ch: int = 16,
        n_blocks: int = 6,
        bottleneck: int = 16,
        dropout: float = 0.2,
        classifier_hidden: int = 96,
        classifier_dropout: float = 0.4,
    ) -> None:
        super().__init__()
        blocks: List[nn.Module] = []
        in_ch = input_dim
        for i in range(max(n_blocks, 1)):
            blk = InceptionBlock1D(in_ch=in_ch, out_ch=out_ch, bottleneck=bottleneck, dropout=dropout)
            blocks.append(blk)
            in_ch = out_ch * 4
            # Residual connection every 3 blocks.
            if (i + 1) % 3 == 0:
                blocks.append(
                    nn.Sequential(
                        nn.Conv1d(input_dim if i == 2 else out_ch * 4, out_ch * 4, kernel_size=1, bias=False),
                        nn.BatchNorm1d(out_ch * 4),
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.n_blocks = max(n_blocks, 1)
        self.out_dim = out_ch * 8
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(self.out_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # [B, T, C] -> [B, C, T]
        h = x.transpose(1, 2)
        residual_source = h
        block_count = 0
        for module in self.blocks:
            if isinstance(module, InceptionBlock1D):
                h = module(h)
                block_count += 1
                if block_count % 3 == 0:
                    # next module in list is projection for residual
                    continue
            else:
                h = torch.relu(h + module(residual_source))
                residual_source = h
        h_mean = h.mean(dim=2)
        h_max = h.max(dim=2).values
        z = torch.cat([h_mean, h_max], dim=1)
        logits = self.classifier(z)
        return logits, z


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    macro_f1: float
    report: str

'''
命令行参数解析函数 parse_channels 用于解析用户输入的 TCN 通道配置字符串，并确保其满足模型的要求。
'''
def parse_channels(value: str | None, latent_dim: int, min_layers: int) -> List[int]:
    if not value:
        channels = [latent_dim] * max(min_layers, 1)
    else:
        parts = [p.strip() for p in value.split(",") if p.strip()]
        channels = [int(p) for p in parts]
        if len(channels) < max(min_layers, 1):
            channels = channels + [channels[-1]] * (max(min_layers, 1) - len(channels))
    if channels[-1] != latent_dim:
        channels.append(latent_dim)
    return channels

'''
进行分层抽样，配合训练集标签 y 和指定的验证集比例 val_ratio，
将数据划分为训练集和验证集。确保每个类别在训练集和验证集中都有代表性。
'''
def split_train_val_stratified(y: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0,1)")

    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []

    for cls in sorted(np.unique(y).tolist()):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * val_ratio)))
        n_val = min(n_val, len(cls_idx) - 1) if len(cls_idx) > 1 else 1
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())

    if not train_idx:
        raise ValueError("No train samples after split; decrease val_ratio")
    if not val_idx:
        raise ValueError("No val samples after split; increase val_ratio")

    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)

'''
当数据集中某些类别的样本很少（长尾分布）时，模型容易忽略这些类。
计算类别权重可以传给 Loss 函数（如 CrossEntropyLoss），让模型更关注少样本类别。
'''
def compute_class_weights(y: np.ndarray, num_classes: int, mode: str) -> torch.Tensor | None:
    mode = mode.strip().lower()
    if mode == "none":
        return None
    if mode == "auto":
        counts = np.bincount(y, minlength=num_classes)
        counts = np.maximum(counts, 1)
        inv_freq = counts.sum() / (num_classes * counts)
        return torch.tensor(inv_freq, dtype=torch.float32)

    vals = [float(v.strip()) for v in mode.split(",") if v.strip()]
    if len(vals) != num_classes:
        raise ValueError(f"class-weights must have {num_classes} values, got {len(vals)}")
    return torch.tensor(vals, dtype=torch.float32)


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def build_classification_report(
    y_true: List[int], y_pred: List[int], class_names: List[str]
) -> Tuple[str, float]:
    if sk_classification_report is not None:
        try:
            report_dict = sk_classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                output_dict=True,
                digits=4,
                zero_division=0,
            )
            text = sk_classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                digits=4,
                zero_division=0,
            )
            macro_f1 = float(report_dict.get("macro avg", {}).get("f1-score", 0.0))
            return text, macro_f1
        except Exception:
            pass

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    lines = ["              precision    recall  f1-score   support"]
    f1_list: List[float] = []

    for cls, name in enumerate(class_names):
        tp = int(np.sum((y_pred_np == cls) & (y_true_np == cls)))
        fp = int(np.sum((y_pred_np == cls) & (y_true_np != cls)))
        fn = int(np.sum((y_pred_np != cls) & (y_true_np == cls)))
        support = int(np.sum(y_true_np == cls))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        f1_list.append(f1)
        lines.append(f"{name:>12} {precision:10.4f} {recall:8.4f} {f1:10.4f} {support:9d}")

    acc = float(np.mean(y_true_np == y_pred_np)) if len(y_true_np) else 0.0
    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    lines.append(f"\n    accuracy {acc:28.4f} {len(y_true):9d}")
    lines.append(f"   macro avg {0.0:10.4f} {0.0:8.4f} {macro_f1:10.4f} {len(y_true):9d}")
    return "\n".join(lines), macro_f1


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module, class_names: List[str]) -> EvalResult:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits, _ = model(x_batch)
            loss = loss_fn(logits, y_batch)

            total_loss += float(loss.item())
            n_batches += 1
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(y_batch.cpu().tolist())

    avg_loss = total_loss / n_batches if n_batches else 0.0
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels))) if all_labels else 0.0
    report, macro_f1 = build_classification_report(all_labels, all_preds, class_names)
    return EvalResult(loss=avg_loss, accuracy=accuracy, macro_f1=macro_f1, report=report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train single-model TCN classifier (no distillation)")
    parser.add_argument("--dataset-npz", type=str, required=True, help="Path to prepared dataset .npz")
    parser.add_argument("--output-dir", type=str, default="autoencoder_benchmark/outputs/single_tcn", help="Output directory")
    parser.add_argument(
        "--model",
        type=str,
        default="tcn_attn",
        choices=["tcn", "tcn_attn", "inception"],
        help="Backbone model",
    )

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--tcn-kernel", type=int, default=3)
    parser.add_argument("--tcn-layers", type=int, default=2)
    parser.add_argument("--tcn-channels", type=str, default=None, help="e.g. '32,64,64'")
    parser.add_argument("--tcn-dropout", type=float, default=0.2)
    parser.add_argument("--tcn-dilation-base", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--classifier-hidden", type=int, default=64)
    parser.add_argument("--classifier-dropout", type=float, default=0.4)
    parser.add_argument("--attn-heads", type=int, default=4)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    parser.add_argument("--attn-ff-dim", type=int, default=128)
    parser.add_argument("--inception-out-ch", type=int, default=16)
    parser.add_argument("--inception-blocks", type=int, default=6)
    parser.add_argument("--inception-bottleneck", type=int, default=16)

    parser.add_argument("--class-weights", type=str, default="auto", help="auto | none | comma values")
    parser.add_argument("--weighted-sampler", action="store_true", help="Use WeightedRandomSampler on train split")
    parser.add_argument(
        "--drop-feature-indices",
        type=str,
        default="",
        help="Comma-separated 0-based feature indices to drop, e.g. '3,4,5,6,7'",
    )

    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument(
        "--checkpoint-metric",
        type=str,
        default="val_macro_f1",
        choices=["val_macro_f1", "test_acc"],
        help="Metric used to select and save best checkpoint",
    )

    return parser.parse_args()


def load_npz_dataset(npz_path: str | os.PathLike) -> Dict[str, np.ndarray]:
    with np.load(str(npz_path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main() -> None:
    args = parse_args()
    # 环境初始化
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 数据加载与处理
    ds = load_npz_dataset(args.dataset_npz)
    x_train = ds["X_train_full"].astype(np.float32)
    y_train = ds["y_train"].astype(np.int64)
    x_test = ds["X_test_full"].astype(np.float32)
    y_test = ds["y_test"].astype(np.int64)
    # 支持通过命令行参数 args.drop_feature_indices 删除特定的传感器通道
    drop_indices: List[int] = []
    if args.drop_feature_indices.strip():
        drop_indices = [int(v.strip()) for v in args.drop_feature_indices.split(",") if v.strip()]
        keep_indices = [i for i in range(x_train.shape[2]) if i not in set(drop_indices)]
        if not keep_indices:
            raise ValueError("All features are dropped; keep at least one feature")
        x_train = x_train[:, :, keep_indices]
        x_test = x_test[:, :, keep_indices]

    num_classes = int(max(y_train.max(), y_test.max()) + 1)
    class_names = [f"Class {i}" for i in range(num_classes)]
    # 数据划分
    train_idx, val_idx = split_train_val_stratified(y_train, args.val_ratio, args.seed)
    x_tr = x_train[train_idx]
    y_tr = y_train[train_idx]
    x_val = x_train[val_idx]
    y_val = y_train[val_idx]

    print("=" * 72)
    print("Single-Model TCN Classifier")
    print("=" * 72)
    print(f"Dataset: {args.dataset_npz}")
    print(f"Device: {device}")
    print(f"Train/Val/Test: {x_tr.shape[0]} / {x_val.shape[0]} / {x_test.shape[0]}")
    print(f"Input shape: T={x_train.shape[1]}, C={x_train.shape[2]}")
    if drop_indices:
        print(f"Dropped feature indices: {sorted(drop_indices)}")

    ch = parse_channels(args.tcn_channels, args.latent_dim, args.tcn_layers)
    print(f"Model: {args.model}")
    print(f"TCN channels: {ch}, kernel={args.tcn_kernel}, dropout={args.tcn_dropout}")

    if args.model == "tcn":
        model = TeacherClassifierTCNFull(
            input_dim=int(x_train.shape[2]),
            latent_dim=args.latent_dim,
            num_classes=num_classes,
            channels=ch,
            kernel_size=args.tcn_kernel,
            dropout=args.tcn_dropout,
            dilation_base=args.tcn_dilation_base,
            classifier_hidden=args.classifier_hidden,
            classifier_dropout=args.classifier_dropout,
        ).to(device)
    elif args.model == "tcn_attn":
        model = AttentionTCNClassifier(
            input_dim=int(x_train.shape[2]),
            num_classes=num_classes,
            channels=ch[-1],
            tcn_layers=args.tcn_layers,
            tcn_kernel=args.tcn_kernel,
            tcn_dropout=args.tcn_dropout,
            dilation_base=args.tcn_dilation_base,
            attn_heads=args.attn_heads,
            attn_dropout=args.attn_dropout,
            ff_dim=args.attn_ff_dim,
            classifier_hidden=args.classifier_hidden,
            classifier_dropout=args.classifier_dropout,
        ).to(device)
    else:
        model = InceptionTimeClassifier(
            input_dim=int(x_train.shape[2]),
            num_classes=num_classes,
            out_ch=args.inception_out_ch,
            n_blocks=args.inception_blocks,
            bottleneck=args.inception_bottleneck,
            dropout=args.tcn_dropout,
            classifier_hidden=args.classifier_hidden,
            classifier_dropout=args.classifier_dropout,
        ).to(device)

    class_weights = compute_class_weights(y_tr, num_classes, args.class_weights)
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"Class weights: {class_weights.detach().cpu().tolist()}")
    else:
        print("Class weights: None")

    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(args.label_smoothing))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_dataset = TensorDataset(torch.tensor(x_tr), torch.tensor(y_tr, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(x_val), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test, dtype=torch.long))

    if args.weighted_sampler:
        counts = np.bincount(y_tr, minlength=num_classes)
        sample_w = 1.0 / np.maximum(counts[y_tr], 1)
        sample_w = torch.tensor(sample_w, dtype=torch.double)
        sampler = WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
        print("Train loader: WeightedRandomSampler")
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        print("Train loader: shuffle=True")

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    out_dir = Path(args.output_dir).expanduser()
    dataset_tag = Path(args.dataset_npz).stem
    run_name = (
        f"single_{args.model}_{dataset_tag}_ep{args.epochs}_lr{args.lr}_bs{args.batch_size}_"
        f"k{args.tcn_kernel}_l{args.tcn_layers}_d{args.tcn_dropout}_"
        f"lat{args.latent_dim}_wd{args.weight_decay}_seed{args.seed}"
    )
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "run_args.json", "w", encoding="utf-8") as f:
        payload = vars(args).copy()
        payload["resolved_output_dir"] = str(run_dir)
        json.dump(payload, f, indent=2, ensure_ascii=True)

    best_score = -1.0
    best_epoch = -1
    best_state: Dict[str, torch.Tensor] | None = None
    wait = 0

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        scheduler.step()

        train_loss = total_loss / n_batches if n_batches else 0.0
        val_res = evaluate(model, val_loader, device, loss_fn, class_names)
        test_res = evaluate(model, test_loader, device, loss_fn, class_names)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_res.loss,
                "val_acc": val_res.accuracy,
                "val_macro_f1": val_res.macro_f1,
                "test_acc": test_res.accuracy,
                "test_macro_f1": test_res.macro_f1,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_res.accuracy*100:.2f}% val_f1={val_res.macro_f1:.4f} | "
            f"test_acc={test_res.accuracy*100:.2f}% test_f1={test_res.macro_f1:.4f}"
        )

        current_score = val_res.macro_f1 if args.checkpoint_metric == "val_macro_f1" else test_res.accuracy
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch >= args.min_epochs and wait >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, score={best_score:.4f})")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")

    model.load_state_dict(best_state)
    model.to(device)

    train_eval = evaluate(
        model,
        DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.long)), batch_size=args.batch_size),
        device,
        loss_fn,
        class_names,
    )
    val_eval = evaluate(model, val_loader, device, loss_fn, class_names)
    test_eval = evaluate(model, test_loader, device, loss_fn, class_names)

    torch.save(model.state_dict(), run_dir / "best_single_tcn.pth")

    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=True)

    with open(run_dir / "evaluation_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Checkpoint metric: {args.checkpoint_metric}\n")
        f.write(f"Best score: {best_score:.6f}\n\n")

        f.write("--- Train Metrics ---\n")
        f.write(f"Loss: {train_eval.loss:.6f}\n")
        f.write(f"Accuracy: {train_eval.accuracy*100:.2f}%\n")
        f.write(f"Macro-F1: {train_eval.macro_f1:.6f}\n")
        f.write("\n=== Classification Report (Train) ===\n")
        f.write(train_eval.report + "\n\n")

        f.write("--- Val Metrics ---\n")
        f.write(f"Loss: {val_eval.loss:.6f}\n")
        f.write(f"Accuracy: {val_eval.accuracy*100:.2f}%\n")
        f.write(f"Macro-F1: {val_eval.macro_f1:.6f}\n")
        f.write("\n=== Classification Report (Val) ===\n")
        f.write(val_eval.report + "\n\n")

        f.write("--- Test Metrics ---\n")
        f.write(f"Loss: {test_eval.loss:.6f}\n")
        f.write(f"Accuracy: {test_eval.accuracy*100:.2f}%\n")
        f.write(f"Macro-F1: {test_eval.macro_f1:.6f}\n")
        f.write("\n=== Classification Report (Test) ===\n")
        f.write(test_eval.report + "\n")

    print("=" * 72)
    print(f"Best epoch: {best_epoch}")
    print(f"Final test accuracy: {test_eval.accuracy*100:.2f}%")
    print(f"Final test macro-F1: {test_eval.macro_f1:.4f}")
    print(f"Saved to: {run_dir}")


if __name__ == "__main__":
    main()
