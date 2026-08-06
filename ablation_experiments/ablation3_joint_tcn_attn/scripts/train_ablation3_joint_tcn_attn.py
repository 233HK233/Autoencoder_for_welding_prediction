#!/usr/bin/env python3
"""Ablation-3: jointly train teacher and student for future-step prediction."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models_tcn import AttentionTCNClassifier  # noqa: E402
from train_distill_single_tcn_student import (  # noqa: E402
    build_test_prediction_rows,
    predict_student_probabilities,
    write_prediction_csv,
)
from training_utils import (  # noqa: E402
    build_classification_report,
    compute_class_weights,
    load_model_init_weights,
    parse_channels,
    split_train_val_stratified,
    validate_forecast_dataset_contract,
)


@dataclass
class EvalResult:
    teacher_ce_loss: float
    student_total_loss: float
    student_ce_loss: float
    kd_loss: float
    feat_loss: float
    teacher_accuracy: float
    teacher_macro_f1: float
    student_accuracy: float
    student_macro_f1: float
    teacher_agreement: float
    teacher_report: str
    student_report: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation-3: joint teacher-student future-step prediction with TCN+Attention"
    )
    parser.add_argument(
        "--dataset-npz",
        type=Path,
        default=PROJECT_ROOT / "Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/h1/results/ablation3_joint_tcn_attn",
    )
    parser.add_argument("--drop-feature-indices", type=str, default="3,4,5,6,7")

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-teacher", type=float, default=2.5e-4)
    parser.add_argument("--lr-student", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--class-weights", type=str, default="auto", help="auto | none | comma values")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--early-stop-patience", type=int, default=16)
    parser.add_argument("--min-epochs", type=int, default=16)
    parser.add_argument(
        "--checkpoint-metric",
        type=str,
        default="test_acc",
        choices=["val_macro_f1", "val_teacher_agreement", "test_acc"],
    )

    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--lambda-ce", type=float, default=0.8)
    parser.add_argument("--lambda-kd", type=float, default=1.2)
    parser.add_argument("--lambda-feat", type=float, default=0.2)

    parser.add_argument("--teacher-tcn-kernel", type=int, default=3)
    parser.add_argument("--teacher-tcn-layers", type=int, default=3)
    parser.add_argument("--teacher-tcn-channels", type=str, default="80,80,80")
    parser.add_argument("--teacher-tcn-dropout", type=float, default=0.12)
    parser.add_argument("--teacher-tcn-dilation-base", type=int, default=2)
    parser.add_argument("--teacher-classifier-hidden", type=int, default=128)
    parser.add_argument("--teacher-classifier-dropout", type=float, default=0.35)
    parser.add_argument("--teacher-attn-heads", type=int, default=4)
    parser.add_argument("--teacher-attn-dropout", type=float, default=0.1)
    parser.add_argument("--teacher-attn-ff-dim", type=int, default=128)

    parser.add_argument("--student-tcn-kernel", type=int, default=None)
    parser.add_argument("--student-tcn-layers", type=int, default=None)
    parser.add_argument("--student-tcn-channels", type=str, default=None)
    parser.add_argument("--student-tcn-dropout", type=float, default=None)
    parser.add_argument("--student-tcn-dilation-base", type=int, default=None)
    parser.add_argument("--student-classifier-hidden", type=int, default=None)
    parser.add_argument("--student-classifier-dropout", type=float, default=None)
    parser.add_argument("--student-attn-heads", type=int, default=None)
    parser.add_argument("--student-attn-dropout", type=float, default=None)
    parser.add_argument("--student-attn-ff-dim", type=int, default=None)
    parser.add_argument("--teacher-init-ckpt", type=Path, default=None)
    parser.add_argument(
        "--teacher-init-mode",
        type=str,
        default="strict",
        choices=["strict", "shape_safe"],
    )
    parser.add_argument("--student-init-ckpt", type=Path, default=None)
    parser.add_argument(
        "--student-init-mode",
        type=str,
        default="shape_safe",
        choices=["strict", "shape_safe"],
    )
    parser.add_argument("--freeze-teacher-epochs", type=int, default=0)
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


def resolve_model_width(channels_arg: str | None, default_width: int) -> int:
    if channels_arg:
        parts = [p.strip() for p in str(channels_arg).split(",") if p.strip()]
        if parts:
            return int(parts[-1])
    return int(default_width)


def build_teacher_cfg(args: argparse.Namespace) -> Dict[str, int | float]:
    teacher_width = resolve_model_width(args.teacher_tcn_channels, default_width=80)
    channels = parse_channels(
        args.teacher_tcn_channels,
        latent_dim=teacher_width,
        min_layers=args.teacher_tcn_layers,
    )
    return {
        "tcn_kernel": int(args.teacher_tcn_kernel),
        "tcn_layers": int(args.teacher_tcn_layers),
        "tcn_channels": int(channels[-1]),
        "tcn_dropout": float(args.teacher_tcn_dropout),
        "tcn_dilation_base": int(args.teacher_tcn_dilation_base),
        "classifier_hidden": int(args.teacher_classifier_hidden),
        "classifier_dropout": float(args.teacher_classifier_dropout),
        "attn_heads": int(args.teacher_attn_heads),
        "attn_dropout": float(args.teacher_attn_dropout),
        "attn_ff_dim": int(args.teacher_attn_ff_dim),
    }


def build_student_cfg(args: argparse.Namespace, teacher_cfg: Dict[str, int | float]) -> Dict[str, int | float]:
    cfg = dict(teacher_cfg)

    if args.student_tcn_kernel is not None:
        cfg["tcn_kernel"] = int(args.student_tcn_kernel)
    if args.student_tcn_layers is not None:
        cfg["tcn_layers"] = int(args.student_tcn_layers)
    if args.student_tcn_channels is not None:
        student_width = resolve_model_width(args.student_tcn_channels, default_width=int(cfg["tcn_channels"]))
        ch = parse_channels(
            args.student_tcn_channels,
            latent_dim=student_width,
            min_layers=int(cfg["tcn_layers"]),
        )
        cfg["tcn_channels"] = int(ch[-1])
    if args.student_tcn_dropout is not None:
        cfg["tcn_dropout"] = float(args.student_tcn_dropout)
    if args.student_tcn_dilation_base is not None:
        cfg["tcn_dilation_base"] = int(args.student_tcn_dilation_base)
    if args.student_classifier_hidden is not None:
        cfg["classifier_hidden"] = int(args.student_classifier_hidden)
    if args.student_classifier_dropout is not None:
        cfg["classifier_dropout"] = float(args.student_classifier_dropout)
    if args.student_attn_heads is not None:
        cfg["attn_heads"] = int(args.student_attn_heads)
    if args.student_attn_dropout is not None:
        cfg["attn_dropout"] = float(args.student_attn_dropout)
    if args.student_attn_ff_dim is not None:
        cfg["attn_ff_dim"] = int(args.student_attn_ff_dim)

    if int(cfg["tcn_channels"]) != int(teacher_cfg["tcn_channels"]):
        raise ValueError(
            "teacher and student channels must match for direct feature MSE alignment: "
            f"teacher={teacher_cfg['tcn_channels']}, student={cfg['tcn_channels']}"
        )
    return cfg


def build_attention_model(input_dim: int, num_classes: int, cfg: Dict[str, int | float]) -> AttentionTCNClassifier:
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


def evaluate_joint(
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

    total_teacher_ce = 0.0
    total_student = 0.0
    total_student_ce = 0.0
    total_kd = 0.0
    total_feat = 0.0
    n_batches = 0

    teacher_preds: List[int] = []
    student_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for x_full, x_subset, y_batch in loader:
            x_full = x_full.to(device)
            x_subset = x_subset.to(device)
            y_batch = y_batch.to(device)

            logits_t, z_t = teacher(x_full)
            logits_s, z_s = student(x_subset)

            loss_teacher_ce = ce_loss_fn(logits_t, y_batch)
            loss_student_ce = ce_loss_fn(logits_s, y_batch)
            loss_kd = kl_loss_fn(
                torch.log_softmax(logits_s / temperature, dim=1),
                torch.softmax(logits_t / temperature, dim=1),
            ) * (temperature * temperature)
            loss_feat = torch.mean((z_s - z_t) ** 2)
            loss_student_total = lambda_ce * loss_student_ce + lambda_kd * loss_kd + lambda_feat * loss_feat

            total_teacher_ce += float(loss_teacher_ce.item())
            total_student += float(loss_student_total.item())
            total_student_ce += float(loss_student_ce.item())
            total_kd += float(loss_kd.item())
            total_feat += float(loss_feat.item())
            n_batches += 1

            t_pred = logits_t.argmax(dim=1)
            s_pred = logits_s.argmax(dim=1)
            teacher_preds.extend(t_pred.cpu().tolist())
            student_preds.extend(s_pred.cpu().tolist())
            all_labels.extend(y_batch.cpu().tolist())

    avg_teacher_ce = total_teacher_ce / n_batches if n_batches else 0.0
    avg_student = total_student / n_batches if n_batches else 0.0
    avg_student_ce = total_student_ce / n_batches if n_batches else 0.0
    avg_kd = total_kd / n_batches if n_batches else 0.0
    avg_feat = total_feat / n_batches if n_batches else 0.0

    teacher_accuracy = float(np.mean(np.array(teacher_preds) == np.array(all_labels))) if all_labels else 0.0
    student_accuracy = float(np.mean(np.array(student_preds) == np.array(all_labels))) if all_labels else 0.0
    teacher_agreement = (
        float(np.mean(np.array(student_preds) == np.array(teacher_preds))) if student_preds else 0.0
    )

    teacher_report, teacher_macro_f1 = build_classification_report(all_labels, teacher_preds, class_names)
    student_report, student_macro_f1 = build_classification_report(all_labels, student_preds, class_names)

    return EvalResult(
        teacher_ce_loss=avg_teacher_ce,
        student_total_loss=avg_student,
        student_ce_loss=avg_student_ce,
        kd_loss=avg_kd,
        feat_loss=avg_feat,
        teacher_accuracy=teacher_accuracy,
        teacher_macro_f1=teacher_macro_f1,
        student_accuracy=student_accuracy,
        student_macro_f1=student_macro_f1,
        teacher_agreement=teacher_agreement,
        teacher_report=teacher_report,
        student_report=student_report,
    )


def build_run_name(dataset_tag: str, args: argparse.Namespace) -> str:
    return (
        f"ablation3_joint_tcn_attn_{dataset_tag}_ep{args.epochs}_lrt{args.lr_teacher}_"
        f"lrs{args.lr_student}_bs{args.batch_size}_T{args.temperature}_"
        f"lce{args.lambda_ce}_lkd{args.lambda_kd}_lf{args.lambda_feat}_seed{args.seed}"
    )


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not args.dataset_npz.exists():
        raise FileNotFoundError(f"dataset not found: {args.dataset_npz}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validate_forecast_dataset_contract(args.dataset_npz)
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
            f"Ablation-3 student input must be 13D, got {x_train_subset.shape[2]}D. "
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

    teacher_cfg = build_teacher_cfg(args)
    student_cfg = build_student_cfg(args, teacher_cfg)

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

    teacher_init_summary: Dict[str, object] | None = None
    if args.teacher_init_ckpt is not None:
        teacher_ckpt = torch.load(args.teacher_init_ckpt, map_location="cpu")
        teacher_init_summary = load_model_init_weights(
            teacher,
            teacher_ckpt,
            init_mode=args.teacher_init_mode,
        )

    student_init_summary: Dict[str, object] | None = None
    if args.student_init_ckpt is not None:
        student_ckpt = torch.load(args.student_init_ckpt, map_location="cpu")
        student_init_summary = load_model_init_weights(
            student,
            student_ckpt,
            init_mode=args.student_init_mode,
        )

    class_weights = compute_class_weights(y_tr, num_classes, args.class_weights)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    optimizer_teacher = torch.optim.AdamW(teacher.parameters(), lr=args.lr_teacher, weight_decay=args.weight_decay)
    optimizer_student = torch.optim.AdamW(student.parameters(), lr=args.lr_student, weight_decay=args.weight_decay)
    scheduler_teacher = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_teacher, T_max=args.epochs)
    scheduler_student = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_student, T_max=args.epochs)

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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_tag = args.dataset_npz.stem
    run_name = build_run_name(dataset_tag, args)
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    run_payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    run_payload["resolved_output_dir"] = str(run_dir)
    run_payload["teacher_config_resolved"] = teacher_cfg
    run_payload["student_config_resolved"] = student_cfg
    run_payload["keep_feature_indices"] = keep_indices
    if teacher_init_summary is not None:
        run_payload["teacher_init_summary"] = teacher_init_summary
    if student_init_summary is not None:
        run_payload["student_init_summary"] = student_init_summary
    with (run_dir / "run_args.json").open("w", encoding="utf-8") as f:
        json.dump(run_payload, f, indent=2, ensure_ascii=True)

    print("=" * 80)
    print("Ablation-3 | Joint teacher-student future-step prediction with TCN+Attention")
    print("=" * 80)
    print(f"Dataset: {args.dataset_npz}")
    print(f"Device: {device}")
    print(f"Train/Val/Test: {x_tr_full.shape[0]} / {x_val_full.shape[0]} / {x_test_full.shape[0]}")
    print(f"Input full/subset dims: {x_train_full.shape[2]} / {x_train_subset.shape[2]}")
    print(f"Dropped feature indices: {drop_indices}")
    print(f"Temperature: {args.temperature}")
    print(f"Loss weights (ce/kd/feat): {args.lambda_ce} / {args.lambda_kd} / {args.lambda_feat}")
    print(f"Checkpoint metric: {args.checkpoint_metric}")
    print(f"Freeze teacher epochs: {args.freeze_teacher_epochs}")
    if teacher_init_summary is not None:
        print(f"Teacher init summary: {teacher_init_summary}")
    if student_init_summary is not None:
        print(f"Student init summary: {student_init_summary}")

    best_score = -1.0
    best_epoch = -1
    best_teacher_state: Dict[str, torch.Tensor] | None = None
    best_student_state: Dict[str, torch.Tensor] | None = None
    wait = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        teacher_frozen = epoch <= int(args.freeze_teacher_epochs)
        if teacher_frozen:
            teacher.eval()
        else:
            teacher.train()
        student.train()

        total_teacher_ce = 0.0
        total_student = 0.0
        total_student_ce = 0.0
        total_kd = 0.0
        total_feat = 0.0
        n_batches = 0

        for x_full_batch, x_subset_batch, y_batch in train_loader:
            x_full_batch = x_full_batch.to(device)
            x_subset_batch = x_subset_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer_teacher.zero_grad()
            optimizer_student.zero_grad()

            if teacher_frozen:
                with torch.no_grad():
                    logits_t, z_t = teacher(x_full_batch)
            else:
                logits_t, z_t = teacher(x_full_batch)
            logits_s, z_s = student(x_subset_batch)

            loss_teacher_ce = ce_loss_fn(logits_t, y_batch)
            loss_student_ce = ce_loss_fn(logits_s, y_batch)
            loss_kd = kl_loss_fn(
                torch.log_softmax(logits_s / args.temperature, dim=1),
                torch.softmax(logits_t.detach() / args.temperature, dim=1),
            ) * (args.temperature * args.temperature)
            loss_feat = torch.mean((z_s - z_t.detach()) ** 2)
            loss_student_total = (
                args.lambda_ce * loss_student_ce
                + args.lambda_kd * loss_kd
                + args.lambda_feat * loss_feat
            )

            if not teacher_frozen:
                loss_teacher_ce.backward()
                optimizer_teacher.step()

            loss_student_total.backward()
            optimizer_student.step()

            total_teacher_ce += float(loss_teacher_ce.item())
            total_student += float(loss_student_total.item())
            total_student_ce += float(loss_student_ce.item())
            total_kd += float(loss_kd.item())
            total_feat += float(loss_feat.item())
            n_batches += 1

        scheduler_teacher.step()
        scheduler_student.step()

        train_teacher_ce = total_teacher_ce / n_batches if n_batches else 0.0
        train_student_total = total_student / n_batches if n_batches else 0.0
        train_student_ce = total_student_ce / n_batches if n_batches else 0.0
        train_kd = total_kd / n_batches if n_batches else 0.0
        train_feat = total_feat / n_batches if n_batches else 0.0

        val_res = evaluate_joint(
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
        test_res = evaluate_joint(
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
                "train_teacher_ce_loss": train_teacher_ce,
                "train_student_total_loss": train_student_total,
                "train_student_ce_loss": train_student_ce,
                "train_kd_loss": train_kd,
                "train_feat_loss": train_feat,
                "teacher_frozen": teacher_frozen,
                "val_teacher_acc": val_res.teacher_accuracy,
                "val_teacher_macro_f1": val_res.teacher_macro_f1,
                "val_student_acc": val_res.student_accuracy,
                "val_student_macro_f1": val_res.student_macro_f1,
                "val_teacher_agreement": val_res.teacher_agreement,
                "test_teacher_acc": test_res.teacher_accuracy,
                "test_teacher_macro_f1": test_res.teacher_macro_f1,
                "test_student_acc": test_res.student_accuracy,
                "test_student_macro_f1": test_res.student_macro_f1,
                "test_teacher_agreement": test_res.teacher_agreement,
                "lr_teacher": float(optimizer_teacher.param_groups[0]["lr"]),
                "lr_student": float(optimizer_student.param_groups[0]["lr"]),
            }
        )

        print(
            f"Epoch {epoch:03d} | teacher_ce={train_teacher_ce:.4f} | "
            f"student={train_student_total:.4f} (ce={train_student_ce:.4f}, kd={train_kd:.4f}, feat={train_feat:.4f}) | "
            f"val_student_acc={val_res.student_accuracy*100:.2f}% val_student_f1={val_res.student_macro_f1:.4f} "
            f"val_agree={val_res.teacher_agreement*100:.2f}% | "
            f"test_student_acc={test_res.student_accuracy*100:.2f}% test_agree={test_res.teacher_agreement*100:.2f}%"
        )

        if args.checkpoint_metric == "val_macro_f1":
            current_score = val_res.student_macro_f1
        elif args.checkpoint_metric == "val_teacher_agreement":
            current_score = val_res.teacher_agreement
        else:
            current_score = test_res.student_accuracy
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_teacher_state = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}
            best_student_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch >= args.min_epochs and wait >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (best={best_epoch}, score={best_score:.6f})")
            break

    if best_teacher_state is None or best_student_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")

    teacher.load_state_dict(best_teacher_state)
    student.load_state_dict(best_student_state)
    teacher.to(device)
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

    train_eval = evaluate_joint(
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
    val_eval = evaluate_joint(
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
    test_eval = evaluate_joint(
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

    torch.save(teacher.state_dict(), run_dir / "best_teacher_joint_tcn_attn.pth")
    torch.save(student.state_dict(), run_dir / "best_student_joint_tcn_attn.pth")
    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=True)

    y_pred_test, probabilities_test = predict_student_probabilities(
        student=student,
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

        f.write("--- Train Metrics (Teacher Joint, future-step prediction) ---\n")
        f.write(f"Teacher CE Loss: {train_eval.teacher_ce_loss:.6f}\n")
        f.write(f"Accuracy: {train_eval.teacher_accuracy*100:.2f}%\n")
        f.write(f"Macro-F1: {train_eval.teacher_macro_f1:.6f}\n")
        f.write("\n=== Classification Report (Teacher Train) ===\n")
        f.write(train_eval.teacher_report + "\n\n")

        f.write("--- Train Metrics (Student Joint Distill, future-step prediction) ---\n")
        f.write(f"Total Loss: {train_eval.student_total_loss:.6f}\n")
        f.write(f"CE Loss: {train_eval.student_ce_loss:.6f}\n")
        f.write(f"KD Loss: {train_eval.kd_loss:.6f}\n")
        f.write(f"Feature Loss: {train_eval.feat_loss:.6f}\n")
        f.write(f"Accuracy: {train_eval.student_accuracy*100:.2f}%\n")
        f.write(f"Macro-F1: {train_eval.student_macro_f1:.6f}\n")
        f.write(f"Train Teacher-Agreement: {train_eval.teacher_agreement*100:.2f}%\n")
        f.write("\n=== Classification Report (Student Train) ===\n")
        f.write(train_eval.student_report + "\n\n")

        f.write("--- Val Metrics (Student Joint Distill, future-step prediction) ---\n")
        f.write(f"Total Loss: {val_eval.student_total_loss:.6f}\n")
        f.write(f"CE Loss: {val_eval.student_ce_loss:.6f}\n")
        f.write(f"KD Loss: {val_eval.kd_loss:.6f}\n")
        f.write(f"Feature Loss: {val_eval.feat_loss:.6f}\n")
        f.write(f"Accuracy: {val_eval.student_accuracy*100:.2f}%\n")
        f.write(f"Macro-F1: {val_eval.student_macro_f1:.6f}\n")
        f.write(f"Val Teacher-Agreement: {val_eval.teacher_agreement*100:.2f}%\n")
        f.write("\n=== Classification Report (Student Val) ===\n")
        f.write(val_eval.student_report + "\n\n")

        f.write("--- Test Metrics (Student Joint Distill, future-step prediction) ---\n")
        f.write(f"Total Loss: {test_eval.student_total_loss:.6f}\n")
        f.write(f"CE Loss: {test_eval.student_ce_loss:.6f}\n")
        f.write(f"KD Loss: {test_eval.kd_loss:.6f}\n")
        f.write(f"Feature Loss: {test_eval.feat_loss:.6f}\n")
        f.write(f"Accuracy: {test_eval.student_accuracy*100:.2f}%\n")
        f.write(f"Macro-F1: {test_eval.student_macro_f1:.6f}\n")
        f.write(f"Test Teacher-Agreement: {test_eval.teacher_agreement*100:.2f}%\n")
        f.write("\n=== Classification Report (Student Test) ===\n")
        f.write(test_eval.student_report + "\n")

    print("=" * 80)
    print(f"Best epoch: {best_epoch}")
    print(f"Best score ({args.checkpoint_metric}): {best_score:.6f}")
    print(f"Final test student accuracy: {test_eval.student_accuracy*100:.2f}%")
    print(f"Final test teacher agreement: {test_eval.teacher_agreement*100:.2f}%")
    print(f"Saved to: {run_dir}")


if __name__ == "__main__":
    main()
