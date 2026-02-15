#!/usr/bin/env python3
"""
Distill a 13D student from a frozen single-model TCN-attention teacher.

Teacher is restored from:
- run_args.json (architecture config)
- best_single_tcn.pth (weights)

Student consumes dropped-feature inputs only (default drop indices: 3,4,5,6,7).
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
    from .train_single_tcn_classifier import (
        AttentionTCNClassifier,
        build_classification_report,
        compute_class_weights,
        parse_channels,
        split_train_val_stratified,
    )
except ImportError:
    from train_single_tcn_classifier import (
        AttentionTCNClassifier,
        build_classification_report,
        compute_class_weights,
        parse_channels,
        split_train_val_stratified,
    )


@dataclass
class EvalResult:
    total_loss: float
    ce_loss: float
    kd_loss: float
    feat_loss: float
    accuracy: float
    macro_f1: float
    teacher_agreement: float
    report: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill 13D student from frozen single TCN-attention teacher"
    )
    parser.add_argument("--dataset-npz", type=str, required=True, help="Path to prepared dataset .npz")
    parser.add_argument("--teacher-ckpt", type=str, required=True, help="Path to teacher best_single_tcn.pth")
    parser.add_argument("--teacher-run-args", type=str, required=True, help="Path to teacher run_args.json")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="autoencoder_benchmark/outputs/distill_single_tcn",
        help="Output directory",
    )

    parser.add_argument(
        "--drop-feature-indices",
        type=str,
        default="3,4,5,6,7",
        help="Comma-separated 0-based feature indices to drop",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--class-weights", type=str, default="auto", help="auto | none | comma values")
    parser.add_argument("--weighted-sampler", action="store_true", help="Use WeightedRandomSampler")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument(
        "--checkpoint-metric",
        type=str,
        default="val_teacher_agreement",
        choices=["val_teacher_agreement", "val_macro_f1"],
        help="Metric used to select and save best checkpoint",
    )

    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lambda-ce", type=float, default=1.0)
    parser.add_argument("--lambda-kd", type=float, default=0.7)
    parser.add_argument("--lambda-feat", type=float, default=0.2)

    # Optional student overrides. Defaults inherit teacher run args.
    parser.add_argument("--student-latent-dim", type=int, default=None)
    parser.add_argument("--student-tcn-kernel", type=int, default=None)
    parser.add_argument("--student-tcn-layers", type=int, default=None)
    parser.add_argument("--student-tcn-channels", type=int, default=None)
    parser.add_argument("--student-tcn-dropout", type=float, default=None)
    parser.add_argument("--student-tcn-dilation-base", type=int, default=None)
    parser.add_argument("--student-classifier-hidden", type=int, default=None)
    parser.add_argument("--student-classifier-dropout", type=float, default=None)
    parser.add_argument("--student-attn-heads", type=int, default=None)
    parser.add_argument("--student-attn-dropout", type=float, default=None)
    parser.add_argument("--student-attn-ff-dim", type=int, default=None)

    return parser.parse_args()


def load_npz_dataset(npz_path: str | os.PathLike) -> Dict[str, np.ndarray]:
    with np.load(str(npz_path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def parse_drop_indices(value: str, input_dim: int) -> List[int]:
    if not value.strip():
        return []
    drop = sorted({int(v.strip()) for v in value.split(",") if v.strip()})
    for idx in drop:
        if idx < 0 or idx >= input_dim:
            raise ValueError(f"drop index out of range: {idx} for input_dim={input_dim}")
    return drop


def read_teacher_run_args(path: str | os.PathLike) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("teacher run_args.json must decode to a JSON object")
    return payload


def resolve_teacher_config(teacher_args: Dict[str, object]) -> Dict[str, int | float]:
    model_name = str(teacher_args.get("model", ""))
    if model_name != "tcn_attn":
        raise ValueError(
            f"Only teacher model='tcn_attn' is supported by this script, got '{model_name}'"
        )

    latent_dim = int(teacher_args.get("latent_dim", 64))
    tcn_layers = int(teacher_args.get("tcn_layers", 3))
    ch = parse_channels(
        teacher_args.get("tcn_channels"),
        latent_dim=latent_dim,
        min_layers=tcn_layers,
    )
    channels = int(ch[-1])

    return {
        "latent_dim": latent_dim,
        "tcn_kernel": int(teacher_args.get("tcn_kernel", 3)),
        "tcn_layers": tcn_layers,
        "tcn_channels": channels,
        "tcn_dropout": float(teacher_args.get("tcn_dropout", 0.15)),
        "tcn_dilation_base": int(teacher_args.get("tcn_dilation_base", 2)),
        "classifier_hidden": int(teacher_args.get("classifier_hidden", 128)),
        "classifier_dropout": float(teacher_args.get("classifier_dropout", 0.35)),
        "attn_heads": int(teacher_args.get("attn_heads", 4)),
        "attn_dropout": float(teacher_args.get("attn_dropout", 0.1)),
        "attn_ff_dim": int(teacher_args.get("attn_ff_dim", 128)),
    }


def build_attention_model(
    input_dim: int,
    num_classes: int,
    cfg: Dict[str, int | float],
) -> AttentionTCNClassifier:
    return AttentionTCNClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        channels=int(cfg["tcn_channels"]),
        tcn_layers=int(cfg["tcn_layers"]),
        tcn_kernel=int(cfg["tcn_kernel"]),
        tcn_dropout=float(cfg["tcn_dropout"]),
        dilation_base=int(cfg["tcn_dilation_base"]),
        attn_heads=int(cfg["attn_heads"]),
        attn_dropout=float(cfg["attn_dropout"]),
        ff_dim=int(cfg["attn_ff_dim"]),
        classifier_hidden=int(cfg["classifier_hidden"]),
        classifier_dropout=float(cfg["classifier_dropout"]),
    )


def evaluate_distill(
    teacher: nn.Module,
    student: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    ce_loss_fn: nn.Module,
    kl_loss_fn: nn.Module,
    temperature: float,
    lambda_ce: float,
    lambda_kd: float,
    lambda_feat: float,
) -> EvalResult:
    teacher.eval()
    student.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_feat = 0.0
    n_batches = 0
    all_preds: List[int] = []
    all_labels: List[int] = []
    teacher_preds: List[int] = []

    with torch.no_grad():
        for x_full, x_subset, y_batch in loader:
            x_full = x_full.to(device)
            x_subset = x_subset.to(device)
            y_batch = y_batch.to(device)

            logits_t, z_t = teacher(x_full)
            logits_s, z_s = student(x_subset)

            loss_ce = ce_loss_fn(logits_s, y_batch)
            loss_kd = kl_loss_fn(
                torch.log_softmax(logits_s / temperature, dim=1),
                torch.softmax(logits_t / temperature, dim=1),
            ) * (temperature * temperature)
            loss_feat = torch.mean((z_s - z_t) ** 2)
            loss_total = lambda_ce * loss_ce + lambda_kd * loss_kd + lambda_feat * loss_feat

            total_loss += float(loss_total.item())
            total_ce += float(loss_ce.item())
            total_kd += float(loss_kd.item())
            total_feat += float(loss_feat.item())
            n_batches += 1

            s_pred = logits_s.argmax(dim=1)
            t_pred = logits_t.argmax(dim=1)
            all_preds.extend(s_pred.cpu().tolist())
            teacher_preds.extend(t_pred.cpu().tolist())
            all_labels.extend(y_batch.cpu().tolist())

    avg_total = total_loss / n_batches if n_batches else 0.0
    avg_ce = total_ce / n_batches if n_batches else 0.0
    avg_kd = total_kd / n_batches if n_batches else 0.0
    avg_feat = total_feat / n_batches if n_batches else 0.0
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels))) if all_labels else 0.0
    teacher_agreement = (
        float(np.mean(np.array(all_preds) == np.array(teacher_preds))) if all_preds else 0.0
    )
    report, macro_f1 = build_classification_report(all_labels, all_preds, class_names)
    return EvalResult(
        total_loss=avg_total,
        ce_loss=avg_ce,
        kd_loss=avg_kd,
        feat_loss=avg_feat,
        accuracy=accuracy,
        macro_f1=macro_f1,
        teacher_agreement=teacher_agreement,
        report=report,
    )


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_npz_dataset(args.dataset_npz)
    x_train_full = ds["X_train_full"].astype(np.float32)
    y_train = ds["y_train"].astype(np.int64)
    x_test_full = ds["X_test_full"].astype(np.float32)
    y_test = ds["y_test"].astype(np.int64)

    input_dim_full = int(x_train_full.shape[2])
    drop_indices = parse_drop_indices(args.drop_feature_indices, input_dim=input_dim_full)
    keep_indices = [i for i in range(input_dim_full) if i not in set(drop_indices)]
    if not keep_indices:
        raise ValueError("All features are dropped; keep at least one feature")

    x_train_subset = x_train_full[:, :, keep_indices]
    x_test_subset = x_test_full[:, :, keep_indices]
    if int(x_train_subset.shape[2]) != 13:
        raise ValueError(
            f"Student input must be 13D for this setup, got {x_train_subset.shape[2]}D. "
            f"drop_feature_indices={drop_indices}"
        )

    num_classes = int(max(y_train.max(), y_test.max()) + 1)
    class_names = [f"Class {i}" for i in range(num_classes)]

    train_idx, val_idx = split_train_val_stratified(y_train, args.val_ratio, args.seed)
    x_tr_full = x_train_full[train_idx]
    x_tr_subset = x_train_subset[train_idx]
    y_tr = y_train[train_idx]
    x_val_full = x_train_full[val_idx]
    x_val_subset = x_train_subset[val_idx]
    y_val = y_train[val_idx]

    # 读取Teacher的训练Json配置
    teacher_args = read_teacher_run_args(args.teacher_run_args)
    teacher_cfg = resolve_teacher_config(teacher_args)
    # 构建Student配置，基于Teacher配置并应用可选覆盖
    student_cfg = dict(teacher_cfg)
    if args.student_latent_dim is not None:
        student_cfg["latent_dim"] = int(args.student_latent_dim)
    if args.student_tcn_kernel is not None:
        student_cfg["tcn_kernel"] = int(args.student_tcn_kernel)
    if args.student_tcn_layers is not None:
        student_cfg["tcn_layers"] = int(args.student_tcn_layers)
    if args.student_tcn_channels is not None:
        student_cfg["tcn_channels"] = int(args.student_tcn_channels)
    if args.student_tcn_dropout is not None:
        student_cfg["tcn_dropout"] = float(args.student_tcn_dropout)
    if args.student_tcn_dilation_base is not None:
        student_cfg["tcn_dilation_base"] = int(args.student_tcn_dilation_base)
    if args.student_classifier_hidden is not None:
        student_cfg["classifier_hidden"] = int(args.student_classifier_hidden)
    if args.student_classifier_dropout is not None:
        student_cfg["classifier_dropout"] = float(args.student_classifier_dropout)
    if args.student_attn_heads is not None:
        student_cfg["attn_heads"] = int(args.student_attn_heads)
    if args.student_attn_dropout is not None:
        student_cfg["attn_dropout"] = float(args.student_attn_dropout)
    if args.student_attn_ff_dim is not None:
        student_cfg["attn_ff_dim"] = int(args.student_attn_ff_dim)
    # 确保Teacher和Student latent dim一致，否则无法进行直接的特征MSE对齐
    if int(student_cfg["latent_dim"]) != int(teacher_cfg["latent_dim"]):
        raise ValueError(
            "student latent_dim must equal teacher latent_dim for direct feature MSE alignment. "
            f"teacher={teacher_cfg['latent_dim']}, student={student_cfg['latent_dim']}"
        )
    # 重建Teacher和Student模型
    teacher = build_attention_model(
        input_dim=input_dim_full,
        num_classes=num_classes,
        cfg=teacher_cfg,
    ).to(device)
    student = build_attention_model(
        input_dim=int(x_train_subset.shape[2]),
        num_classes=num_classes,
        cfg=student_cfg,
    ).to(device)
    # 加载Teacher权重并冻结
    teacher_ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
    teacher.load_state_dict(teacher_ckpt, strict=True)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    class_weights = compute_class_weights(y_tr, num_classes, args.class_weights)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_dataset = TensorDataset(
        torch.tensor(x_tr_full, dtype=torch.float32),
        torch.tensor(x_tr_subset, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(x_val_full, dtype=torch.float32),
        torch.tensor(x_val_subset, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(x_test_full, dtype=torch.float32),
        torch.tensor(x_test_subset, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

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

    out_dir = Path(args.output_dir).expanduser()
    dataset_tag = Path(args.dataset_npz).stem
    run_name = (
        f"distill_tcn_attn_{dataset_tag}_ep{args.epochs}_lr{args.lr}_bs{args.batch_size}_"
        f"T{args.temperature}_lce{args.lambda_ce}_lkd{args.lambda_kd}_lf{args.lambda_feat}_seed{args.seed}"
    )
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    args_payload = vars(args).copy()
    args_payload["resolved_output_dir"] = str(run_dir)
    args_payload["teacher_config_resolved"] = teacher_cfg
    args_payload["student_config_resolved"] = student_cfg
    args_payload["keep_feature_indices"] = keep_indices
    with open(run_dir / "run_args.json", "w", encoding="utf-8") as f:
        json.dump(args_payload, f, indent=2, ensure_ascii=True)

    teacher_ref = {
        "teacher_ckpt": str(Path(args.teacher_ckpt).expanduser().resolve()),
        "teacher_run_args": str(Path(args.teacher_run_args).expanduser().resolve()),
    }
    with open(run_dir / "teacher_reference.json", "w", encoding="utf-8") as f:
        json.dump(teacher_ref, f, indent=2, ensure_ascii=True)

    print("=" * 80)
    print("Frozen-Teacher Distillation (13D Student)")
    print("=" * 80)
    print(f"Dataset: {args.dataset_npz}")
    print(f"Teacher ckpt: {args.teacher_ckpt}")
    print(f"Teacher run args: {args.teacher_run_args}")
    print(f"Device: {device}")
    print(
        f"Train/Val/Test: {x_tr_full.shape[0]} / {x_val_full.shape[0]} / {x_test_full.shape[0]}"
    )
    print(f"Input full/subset dims: {x_train_full.shape[2]} / {x_train_subset.shape[2]}")
    print(f"Dropped feature indices: {drop_indices}")
    print(f"Temperature: {args.temperature}")
    print(
        f"Loss weights (ce/kd/feat): {args.lambda_ce} / {args.lambda_kd} / {args.lambda_feat}"
    )
    print(f"Checkpoint metric: {args.checkpoint_metric}")
    if class_weights is None:
        print("Class weights: None")
    else:
        print(f"Class weights: {class_weights.detach().cpu().tolist()}")

    best_score = -1.0
    best_epoch = -1
    best_state: Dict[str, torch.Tensor] | None = None
    wait = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        student.train()
        total_train = 0.0
        total_ce = 0.0
        total_kd = 0.0
        total_feat = 0.0
        n_batches = 0

        for x_full_batch, x_subset_batch, y_batch in train_loader:
            # 数据加载
            x_full_batch = x_full_batch.to(device)
            x_subset_batch = x_subset_batch.to(device)
            y_batch = y_batch.to(device)
            # Teacher做infer, logits为分类头输入，z为分类头输出
            with torch.no_grad():
                logits_t, z_t = teacher(x_full_batch)
            # Student做infer
            logits_s, z_s = student(x_subset_batch)
            # total_loss = 分类损失 + KD损失 + 特征对齐损失
            loss_ce = ce_loss_fn(logits_s, y_batch)
            loss_kd = kl_loss_fn(
                torch.log_softmax(logits_s / args.temperature, dim=1),
                torch.softmax(logits_t / args.temperature, dim=1),
            ) * (args.temperature * args.temperature)
            loss_feat = torch.mean((z_s - z_t) ** 2)
            loss_total = (
                args.lambda_ce * loss_ce
                + args.lambda_kd * loss_kd
                + args.lambda_feat * loss_feat
            )
            # 反向传播和优化
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            total_train += float(loss_total.item())
            total_ce += float(loss_ce.item())
            total_kd += float(loss_kd.item())
            total_feat += float(loss_feat.item())
            n_batches += 1

        scheduler.step()
        # 做validation
        train_loss = total_train / n_batches if n_batches else 0.0
        train_ce = total_ce / n_batches if n_batches else 0.0
        train_kd = total_kd / n_batches if n_batches else 0.0
        train_feat = total_feat / n_batches if n_batches else 0.0
        val_res = evaluate_distill(
            teacher=teacher,
            student=student,
            loader=val_loader,
            device=device,
            class_names=class_names,
            ce_loss_fn=ce_loss_fn,
            kl_loss_fn=kl_loss_fn,
            temperature=args.temperature,
            lambda_ce=args.lambda_ce,
            lambda_kd=args.lambda_kd,
            lambda_feat=args.lambda_feat,
        )
        test_res = evaluate_distill(
            teacher=teacher,
            student=student,
            loader=test_loader,
            device=device,
            class_names=class_names,
            ce_loss_fn=ce_loss_fn,
            kl_loss_fn=kl_loss_fn,
            temperature=args.temperature,
            lambda_ce=args.lambda_ce,
            lambda_kd=args.lambda_kd,
            lambda_feat=args.lambda_feat,
        )

        history.append(
            {
                "epoch": epoch,
                "train_total_loss": train_loss,
                "train_ce_loss": train_ce,
                "train_kd_loss": train_kd,
                "train_feat_loss": train_feat,
                "val_total_loss": val_res.total_loss,
                "val_acc": val_res.accuracy,
                "val_macro_f1": val_res.macro_f1,
                "val_teacher_agreement": val_res.teacher_agreement,
                "test_total_loss": test_res.total_loss,
                "test_acc": test_res.accuracy,
                "test_macro_f1": test_res.macro_f1,
                "test_teacher_agreement": test_res.teacher_agreement,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.4f} (ce={train_ce:.4f}, kd={train_kd:.4f}, feat={train_feat:.4f}) | "
            f"val_acc={val_res.accuracy*100:.2f}% val_f1={val_res.macro_f1:.4f} "
            f"val_agree={val_res.teacher_agreement*100:.2f}% | "
            f"test_acc={test_res.accuracy*100:.2f}% test_agree={test_res.teacher_agreement*100:.2f}%"
        )
        # 优化目标选择F1 score或teacher agreement
        current_score = (
            val_res.teacher_agreement
            if args.checkpoint_metric == "val_teacher_agreement"
            else val_res.macro_f1
        )
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            wait = 0
        else:
            wait += 1
        # early stopping
        if epoch >= args.min_epochs and wait >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch} "
                f"(best epoch {best_epoch}, score={best_score:.6f})"
            )
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")

    student.load_state_dict(best_state)
    student.to(device)

    full_train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train_full, dtype=torch.float32),
            torch.tensor(x_train_subset, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    train_eval = evaluate_distill(
        teacher=teacher,
        student=student,
        loader=full_train_loader,
        device=device,
        class_names=class_names,
        ce_loss_fn=ce_loss_fn,
        kl_loss_fn=kl_loss_fn,
        temperature=args.temperature,
        lambda_ce=args.lambda_ce,
        lambda_kd=args.lambda_kd,
        lambda_feat=args.lambda_feat,
    )
    val_eval = evaluate_distill(
        teacher=teacher,
        student=student,
        loader=val_loader,
        device=device,
        class_names=class_names,
        ce_loss_fn=ce_loss_fn,
        kl_loss_fn=kl_loss_fn,
        temperature=args.temperature,
        lambda_ce=args.lambda_ce,
        lambda_kd=args.lambda_kd,
        lambda_feat=args.lambda_feat,
    )
    test_eval = evaluate_distill(
        teacher=teacher,
        student=student,
        loader=test_loader,
        device=device,
        class_names=class_names,
        ce_loss_fn=ce_loss_fn,
        kl_loss_fn=kl_loss_fn,
        temperature=args.temperature,
        lambda_ce=args.lambda_ce,
        lambda_kd=args.lambda_kd,
        lambda_feat=args.lambda_feat,
    )

    torch.save(student.state_dict(), run_dir / "best_student_distill.pth")

    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=True)

    with open(run_dir / "evaluation_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Checkpoint metric: {args.checkpoint_metric}\n")
        f.write(f"Best score: {best_score:.6f}\n\n")

        f.write("--- Train Metrics (Student) ---\n")
        f.write(f"Total Loss: {train_eval.total_loss:.6f}\n")
        f.write(f"CE Loss: {train_eval.ce_loss:.6f}\n")
        f.write(f"KD Loss: {train_eval.kd_loss:.6f}\n")
        f.write(f"Feature Loss: {train_eval.feat_loss:.6f}\n")
        f.write(f"Accuracy: {train_eval.accuracy*100:.2f}%\n")
        f.write(f"Macro-F1: {train_eval.macro_f1:.6f}\n")
        f.write(f"Train Teacher-Agreement: {train_eval.teacher_agreement*100:.2f}%\n")
        f.write("\n=== Classification Report (Train) ===\n")
        f.write(train_eval.report + "\n\n")

        f.write("--- Val Metrics (Student) ---\n")
        f.write(f"Total Loss: {val_eval.total_loss:.6f}\n")
        f.write(f"CE Loss: {val_eval.ce_loss:.6f}\n")
        f.write(f"KD Loss: {val_eval.kd_loss:.6f}\n")
        f.write(f"Feature Loss: {val_eval.feat_loss:.6f}\n")
        f.write(f"Accuracy: {val_eval.accuracy*100:.2f}%\n")
        f.write(f"Macro-F1: {val_eval.macro_f1:.6f}\n")
        f.write(f"Val Teacher-Agreement: {val_eval.teacher_agreement*100:.2f}%\n")
        f.write("\n=== Classification Report (Val) ===\n")
        f.write(val_eval.report + "\n\n")

        f.write("--- Test Metrics (Student) ---\n")
        f.write(f"Total Loss: {test_eval.total_loss:.6f}\n")
        f.write(f"CE Loss: {test_eval.ce_loss:.6f}\n")
        f.write(f"KD Loss: {test_eval.kd_loss:.6f}\n")
        f.write(f"Feature Loss: {test_eval.feat_loss:.6f}\n")
        f.write(f"Accuracy: {test_eval.accuracy*100:.2f}%\n")
        f.write(f"Macro-F1: {test_eval.macro_f1:.6f}\n")
        f.write(f"Test Teacher-Agreement: {test_eval.teacher_agreement*100:.2f}%\n")
        f.write("\n=== Classification Report (Test) ===\n")
        f.write(test_eval.report + "\n")

    print("=" * 80)
    print(f"Best epoch: {best_epoch}")
    print(f"Best score ({args.checkpoint_metric}): {best_score:.6f}")
    print(f"Final test accuracy: {test_eval.accuracy*100:.2f}%")
    print(f"Final test teacher agreement: {test_eval.teacher_agreement*100:.2f}%")
    print(f"Saved to: {run_dir}")


if __name__ == "__main__":
    main()
