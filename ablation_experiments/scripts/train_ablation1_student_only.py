#!/usr/bin/env python3
"""Ablation-1: train 13D student-only model for horizon=1 future-step prediction."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models_tcn import AttentionTCNClassifier  # noqa: E402
from train_distill_single_tcn_student import (  # noqa: E402
    build_test_prediction_rows,
    write_prediction_csv,
)
from training_utils import (  # noqa: E402
    compute_class_weights,
    evaluate,
    parse_channels,
    split_train_val_stratified,
    validate_forecast_dataset_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation-1: 13D Student-only future-step prediction with TCN+Attention backbone"
    )
    parser.add_argument(
        "--dataset-npz",
        type=Path,
        default=PROJECT_ROOT / "Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz",
        help="Path to the prepared horizon=1 dataset (.npz)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/h1/results/ablation1_student_only_tcn_attn",
    )
    parser.add_argument(
        "--drop-feature-indices",
        type=str,
        default="3,4,5,6,7",
        help="Comma-separated 0-based feature indices to drop from full 18D input",
    )

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--tcn-kernel", type=int, default=3)
    parser.add_argument("--tcn-layers", type=int, default=3)
    parser.add_argument("--tcn-channels", type=str, default="80,80,80")
    parser.add_argument(
        "--student-latent-dim",
        type=int,
        default=80,
        help="Final TCN channel count used by the student-only latent representation.",
    )
    parser.add_argument("--tcn-dropout", type=float, default=0.12)
    parser.add_argument("--tcn-dilation-base", type=int, default=2)
    parser.add_argument("--classifier-hidden", type=int, default=128)
    parser.add_argument("--classifier-dropout", type=float, default=0.35)
    parser.add_argument("--attn-heads", type=int, default=4)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    parser.add_argument("--attn-ff-dim", type=int, default=128)

    parser.add_argument("--class-weights", type=str, default="auto", help="auto | none | comma values")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--early-stop-patience", type=int, default=16)
    parser.add_argument("--min-epochs", type=int, default=16)
    parser.add_argument(
        "--checkpoint-metric",
        type=str,
        default="val_macro_f1",
        choices=["val_macro_f1", "test_acc"],
    )
    return parser.parse_args()


def load_npz_dataset(path: Path) -> Dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def parse_drop_indices(value: str, input_dim: int) -> List[int]:
    if not value.strip():
        return []
    drop = sorted({int(v.strip()) for v in value.split(",") if v.strip()})
    for idx in drop:
        if idx < 0 or idx >= input_dim:
            raise ValueError(f"drop index out of range: {idx} for input_dim={input_dim}")
    return drop


def predict_classifier_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_pred: List[int] = []
    all_prob: List[List[float]] = []
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            logits, _ = model(x_batch)
            probabilities = torch.softmax(logits, dim=1)
            all_pred.extend(logits.argmax(dim=1).cpu().tolist())
            all_prob.extend(probabilities.cpu().tolist())
    return np.asarray(all_pred, dtype=np.int64), np.asarray(all_prob, dtype=np.float64)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validate_forecast_dataset_contract(args.dataset_npz)

    ds = load_npz_dataset(args.dataset_npz)
    x_train = ds["X_train_full"].astype(np.float32)
    y_train = ds["y_train"].astype(np.int64)
    x_test = ds["X_test_full"].astype(np.float32)
    y_test = ds["y_test"].astype(np.int64)

    input_dim_full = int(x_train.shape[2])
    drop_indices = parse_drop_indices(args.drop_feature_indices, input_dim=input_dim_full)
    keep_indices = [i for i in range(input_dim_full) if i not in set(drop_indices)]
    if not keep_indices:
        raise ValueError("All features are dropped; keep at least one feature")

    x_train = x_train[:, :, keep_indices]
    x_test = x_test[:, :, keep_indices]

    if int(x_train.shape[2]) != 13:
        raise ValueError(
            f"Ablation-1 expects 13D input, got {x_train.shape[2]}D. "
            f"drop_feature_indices={drop_indices}"
        )

    num_classes = int(max(y_train.max(), y_test.max()) + 1)
    class_names = [f"Class {i}" for i in range(num_classes)]
    train_idx, val_idx = split_train_val_stratified(y_train, args.val_ratio, args.seed)

    x_tr = x_train[train_idx]
    y_tr = y_train[train_idx]
    x_val = x_train[val_idx]
    y_val = y_train[val_idx]

    channels = parse_channels(args.tcn_channels, latent_dim=args.student_latent_dim, min_layers=args.tcn_layers)
    model = AttentionTCNClassifier(
        input_dim=int(x_train.shape[2]),
        num_classes=num_classes,
        channels=int(channels[-1]),
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

    class_weights = compute_class_weights(y_tr, num_classes, args.class_weights)
    if class_weights is not None:
        class_weights = class_weights.to(device)

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
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_tag = args.dataset_npz.stem
    run_name = (
        f"ablation1_student_only_tcn_attn_{dataset_tag}_ep{args.epochs}_lr{args.lr}_"
        f"bs{args.batch_size}_seed{args.seed}"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    run_payload = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
    }
    run_payload["resolved_output_dir"] = str(run_dir)
    run_payload["keep_feature_indices"] = keep_indices
    with (run_dir / "run_args.json").open("w", encoding="utf-8") as f:
        json.dump(run_payload, f, indent=2, ensure_ascii=True)

    best_score = -1.0
    best_epoch = -1
    best_state: Dict[str, torch.Tensor] | None = None
    wait = 0
    history: List[Dict[str, float]] = []

    print("=" * 80)
    print("Ablation-1 | Student-only (13D) future-step prediction with TCN+Attention")
    print("=" * 80)
    print(f"Dataset: {args.dataset_npz}")
    print(f"Device: {device}")
    print(f"Train/Val/Test: {len(x_tr)} / {len(x_val)} / {len(x_test)}")
    print(f"Drop features: {drop_indices}")

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
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
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
            print(f"Early stopping at epoch {epoch} (best={best_epoch}, score={best_score:.6f})")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")

    model.load_state_dict(best_state)
    model.to(device)

    train_eval = evaluate(
        model,
        DataLoader(
            TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.long)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
        device,
        loss_fn,
        class_names,
    )
    val_eval = evaluate(model, val_loader, device, loss_fn, class_names)
    test_eval = evaluate(model, test_loader, device, loss_fn, class_names)

    torch.save(model.state_dict(), run_dir / "best_student_only.pth")
    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=True)

    y_pred_test, probabilities_test = predict_classifier_probabilities(
        model=model,
        loader=test_loader,
        device=device,
    )
    prediction_rows = build_test_prediction_rows(
        ds=ds,
        y_pred=y_pred_test,
        probabilities=probabilities_test,
    )
    write_prediction_csv(run_dir / "test_predictions.csv", prediction_rows)

    with (run_dir / "evaluation_metrics.txt").open("w", encoding="utf-8") as f:
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Checkpoint metric: {args.checkpoint_metric}\n")
        f.write(f"Best score: {best_score:.6f}\n\n")

        f.write("--- Train Metrics (Student-only) ---\n")
        f.write(f"Loss: {train_eval.loss:.6f}\n")
        f.write(f"Accuracy: {train_eval.accuracy*100:.2f}%\n")
        f.write(f"Macro-F1: {train_eval.macro_f1:.6f}\n")
        f.write("\n=== Classification Report (Train) ===\n")
        f.write(train_eval.report + "\n\n")

        f.write("--- Val Metrics (Student-only) ---\n")
        f.write(f"Loss: {val_eval.loss:.6f}\n")
        f.write(f"Accuracy: {val_eval.accuracy*100:.2f}%\n")
        f.write(f"Macro-F1: {val_eval.macro_f1:.6f}\n")
        f.write("\n=== Classification Report (Val) ===\n")
        f.write(val_eval.report + "\n\n")

        f.write("--- Test Metrics (Student-only) ---\n")
        f.write(f"Loss: {test_eval.loss:.6f}\n")
        f.write(f"Accuracy: {test_eval.accuracy*100:.2f}%\n")
        f.write(f"Macro-F1: {test_eval.macro_f1:.6f}\n")
        f.write("\n=== Classification Report (Test) ===\n")
        f.write(test_eval.report + "\n")

    print("=" * 80)
    print(f"Best epoch: {best_epoch}")
    print(f"Best score ({args.checkpoint_metric}): {best_score:.6f}")
    print(f"Final test accuracy: {test_eval.accuracy*100:.2f}%")
    print(f"Saved to: {run_dir}")


if __name__ == "__main__":
    main()
