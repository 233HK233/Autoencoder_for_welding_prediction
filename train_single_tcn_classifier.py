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
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

try:
    from .models_tcn import AttentionTCNClassifier, InceptionTimeClassifier, TeacherClassifierTCNFull
    from .training_utils import compute_class_weights, evaluate, parse_channels, split_train_val_stratified
except ImportError:
    from models_tcn import AttentionTCNClassifier, InceptionTimeClassifier, TeacherClassifierTCNFull
    from training_utils import compute_class_weights, evaluate, parse_channels, split_train_val_stratified


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
