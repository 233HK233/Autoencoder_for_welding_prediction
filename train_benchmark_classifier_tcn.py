#!/usr/bin/env python3
"""
基准对齐的 Teacher (18D) / Student (14D) 全序列 TCN 分类蒸馏训练。

特点：
1) 仅分类头（无 Decoder），减少模型容量
2) Teacher/Student 的 latent_dim 统一为 32，便于对齐损失直接计算
"""

import json
import os
import random
import time
from typing import List, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.metrics import classification_report as sk_classification_report
except Exception:
    sk_classification_report = None

try:
    from .data_utils import load_sequences_from_folder, standardize_train_test, load_npz_dataset
    from .framework import TeacherStudentFramework, plot_training_history, train_model
    from .models_tcn import StudentClassifierTCNFull, TeacherClassifierTCNFull
except ImportError:
    from data_utils import load_sequences_from_folder, standardize_train_test, load_npz_dataset
    from framework import TeacherStudentFramework, plot_training_history, train_model
    from models_tcn import StudentClassifierTCNFull, TeacherClassifierTCNFull


def build_classification_report(
    y_true: List[int], y_pred: List[int], class_names: Optional[List[str]] = None
) -> str:
    # 兼容 sklearn 不可用时的简化报告
    if class_names is None:
        class_names = [f"Class {i}" for i in sorted(set(y_true) | set(y_pred))]
    if sk_classification_report:
        try:
            return sk_classification_report(y_true, y_pred, target_names=class_names, digits=2)
        except Exception:
            pass

    n_classes = len(class_names)
    metrics = []
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    total = len(y_true)

    for cls in range(n_classes):
        tp = np.sum((y_pred_np == cls) & (y_true_np == cls))
        fp = np.sum((y_pred_np == cls) & (y_true_np != cls))
        fn = np.sum((y_pred_np != cls) & (y_true_np == cls))
        support = np.sum(y_true_np == cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics.append((precision, recall, f1, support))

    accuracy = float(np.sum(y_true_np == y_pred_np) / total) if total > 0 else 0.0
    macro = np.mean([[p, r, f] for p, r, f, _ in metrics], axis=0) if metrics else [0.0, 0.0, 0.0]
    supports = np.array([s for _, _, _, s in metrics], dtype=float)
    weights = supports / supports.sum() if supports.sum() > 0 else np.zeros_like(supports)
    weighted = (
        np.average([p for p, _, _, _ in metrics], weights=weights) if weights.size else 0.0,
        np.average([r for _, r, _, _ in metrics], weights=weights) if weights.size else 0.0,
        np.average([f for _, _, f, _ in metrics], weights=weights) if weights.size else 0.0,
    )

    lines = ["              precision    recall  f1-score   support"]
    for name, (p, r, f, s) in zip(class_names, metrics):
        lines.append(f"{name:>12} {p:10.2f} {r:8.2f} {f:10.2f} {s:9d}")
    lines.append(f"\n    accuracy {accuracy:28.2f} {total:9d}")
    lines.append(
        f"   macro avg {macro[0]:10.2f} {macro[1]:8.2f} {macro[2]:10.2f} {int(supports.sum()):9d}"
    )
    lines.append(
        f"weighted avg {weighted[0]:10.2f} {weighted[1]:8.2f} {weighted[2]:10.2f} {int(supports.sum()):9d}"
    )
    return "\n".join(lines)


def evaluate_loader_with_report(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    use_teacher: bool = False,
):
    # 统一评估入口：返回 CE 损失、准确率与分类报告
    ce_loss = nn.CrossEntropyLoss()
    total_class = 0.0
    num_batches = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    model.eval()
    with torch.no_grad():
        for x_full_batch, x_subset_batch, y_batch in loader:
            x_full_batch = x_full_batch.to(device)
            x_subset_batch = x_subset_batch.to(device)
            y_batch = y_batch.to(device)

            # Teacher 使用全特征输入，Student 使用子集特征
            if use_teacher:
                logits, _ = model(x_full_batch)
            else:
                logits, _ = model(x_subset_batch)

            total_class += ce_loss(logits, y_batch).item()
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().tolist())
            num_batches += 1

    avg_class = total_class / num_batches if num_batches else 0.0
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) if all_labels else 0.0
    report = build_classification_report(all_labels, all_preds, class_names=class_names)
    return avg_class, accuracy, report


import argparse


def parse_channels(value: Optional[str], latent_dim: int) -> List[int]:
    # 解析逗号分隔的通道配置；保证最后一层与 latent_dim 对齐
    if not value:
        return [latent_dim, latent_dim]
    parts = [p.strip() for p in value.split(",") if p.strip()]
    channels = [int(p) for p in parts]
    if channels[-1] != latent_dim:
        channels.append(latent_dim)
    return channels


def parse_args():
    # 命令行参数配置
    parser = argparse.ArgumentParser(description="Train Teacher-Student Classifier (Full-Sequence TCN)")
    parser.add_argument("--epochs", type=int, default=250, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save outputs")
    parser.add_argument(
        "--dataset-npz",
        type=str,
        default=None,
        help="Path to prepared .npz (from prepare_weld_seam_dataset.py).",
    )
    parser.add_argument("--tcn-kernel", type=int, default=3, help="TCN kernel size (odd)")
    parser.add_argument("--tcn-layers", type=int, default=2, help="TCN layer count")
    parser.add_argument("--tcn-dropout", type=float, default=0.1, help="TCN dropout")
    parser.add_argument("--tcn-dilation-base", type=int, default=2, help="TCN dilation base")
    parser.add_argument(
        "--tcn-channels",
        type=str,
        default=None,
        help="Comma-separated channels; last will be forced to latent_dim",
    )
    parser.add_argument(
        "--class-weights",
        type=str,
        default="auto",
        help="Class weights, e.g. '1,2,3' or 'auto' for inverse frequency",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--lambda-class", type=float, default=1.0, help="Classification loss weight")
    parser.add_argument("--lambda-align", type=float, default=0.5, help="Alignment loss weight")
    parser.add_argument("--lambda-kl", type=float, default=0.3, help="KL distillation loss weight")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 固定随机种子，便于复现实验
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.lr
    # 统一输出目录：按参数组合创建子目录
    base_output_dir = Path(args.output_dir).expanduser()
    output_dir = base_output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 与 LSTM-MLP 基线保持一致的窗口配置
    window_size = 10
    target_offset = 5
    teacher_features = 18
    student_features = 13
    # Drop columns 4-8 (1-based) -> indices 3-7 (0-based)
    drop_feature_indices = [3, 4, 5, 6, 7]
    num_classes = 3
    latent_dim = 32

    # 构造 TCN 通道序列（必要时补齐到指定层数）
    tcn_channels = parse_channels(args.tcn_channels, latent_dim)
    if args.tcn_layers > 0:
        if len(tcn_channels) < args.tcn_layers:
            tcn_channels = tcn_channels + [latent_dim] * (args.tcn_layers - len(tcn_channels))
        else:
            tcn_channels = tcn_channels[: args.tcn_layers]
            if tcn_channels[-1] != latent_dim:
                tcn_channels.append(latent_dim)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_data_path = os.path.join(script_dir, "Data", "train_data")
    test_data_path = os.path.join(script_dir, "Data", "test_data")

    print("=" * 60)
    print("Benchmark-aligned Teacher (18D) / Student (14D) Full-Sequence TCN")
    print("=" * 60)
    print(
        f"Config: window_size={window_size}, target_offset={target_offset}, "
        f"teacher_feats={teacher_features}, student_feats={student_features}"
    )
    print(f"Dropped indices for student: {drop_feature_indices}")
    print(
        f"Latent dim={latent_dim}, TCN kernel={args.tcn_kernel}, layers={args.tcn_layers}, "
        f"dropout={args.tcn_dropout}, dilation_base={args.tcn_dilation_base}"
    )
    print(f"TCN channels: {tcn_channels}")
    print(f"Hyperparams: LR={learning_rate}, Batch={batch_size}, Output={output_dir}")
    print("=" * 60)

    # 数据加载：优先使用已准备好的 .npz
    if args.dataset_npz:
        ds = load_npz_dataset(args.dataset_npz)
        X_train_full = ds["X_train_full"].astype(np.float32)
        y_train = ds["y_train"].astype(np.int64)
        X_test_full = ds["X_test_full"].astype(np.float32)
        y_test = ds["y_test"].astype(np.int64)
    else:
        # 否则从原始 CSV 滑窗构造训练/测试集
        print("\nLoading training data...")
        train_sequences, train_labels = load_sequences_from_folder(train_data_path, window_size, target_offset)
        if len(train_sequences) == 0:
            raise ValueError(f"No valid training data found in {train_data_path}")
        X_train_raw = np.array(train_sequences, dtype=np.float32)
        y_train = np.array(train_labels, dtype=np.int64)

        print("\nLoading test data...")
        test_sequences, test_labels = load_sequences_from_folder(test_data_path, window_size, target_offset)
        if len(test_sequences) == 0:
            raise ValueError(f"No valid test data found in {test_data_path}")
        X_test_raw = np.array(test_sequences, dtype=np.float32)
        y_test = np.array(test_labels, dtype=np.int64)

        # 训练集拟合标准化器，避免数据泄露
        X_train_full, X_test_full = standardize_train_test(X_train_raw, X_test_raw)

    # 生成学生端特征子集
    keep_indices: List[int] = [i for i in range(teacher_features) if i not in drop_feature_indices]
    X_train_subset = X_train_full[:, :, keep_indices]
    X_test_subset = X_test_full[:, :, keep_indices]

    print(f"Train shapes: full {X_train_full.shape}, subset {X_train_subset.shape}, labels {y_train.shape}")
    print(f"Test shapes: full {X_test_full.shape}, subset {X_test_subset.shape}, labels {y_test.shape}")

    # 组装 TensorDataset
    train_dataset = TensorDataset(
        torch.tensor(X_train_full, dtype=torch.float32),
        torch.tensor(X_train_subset, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_full, dtype=torch.float32),
        torch.tensor(X_test_subset, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 构建 Teacher / Student 全序列 TCN
    teacher = TeacherClassifierTCNFull(
        input_dim=teacher_features,
        latent_dim=latent_dim,
        num_classes=num_classes,
        channels=tcn_channels,
        kernel_size=args.tcn_kernel,
        dropout=args.tcn_dropout,
        dilation_base=args.tcn_dilation_base,
    )
    student = StudentClassifierTCNFull(
        input_dim=student_features,
        latent_dim=latent_dim,
        num_classes=num_classes,
        channels=tcn_channels,
        kernel_size=args.tcn_kernel,
        dropout=args.tcn_dropout,
        dilation_base=args.tcn_dilation_base,
    )
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student parameters: {student_params:,}")

    # 分类蒸馏：仅使用 CE + 对齐 + KL，保持与基线一致
    class ClassifierFramework(TeacherStudentFramework):
        def train_step(self, x_full: torch.Tensor, x_subset: torch.Tensor, labels: torch.Tensor) -> dict:
            self.teacher.train()
            self.student.train()
            self.optimizer_teacher.zero_grad()
            self.optimizer_student.zero_grad()

            logits_teacher, z_teacher = self.teacher(x_full)
            loss_class_teacher = self.ce_loss(logits_teacher, labels)

            logits_student, z_student = self.student(x_subset)
            loss_class_student = self.ce_loss(logits_student, labels)

            # 对齐损失：约束 Student 的潜变量接近 Teacher
            loss_align = self.mse_loss(z_student, z_teacher.detach())
            teacher_probs = F.softmax(z_teacher.detach(), dim=1)
            student_log_probs = F.log_softmax(z_student, dim=1)
            loss_kl = self.kl_loss(student_log_probs, teacher_probs)

            total_loss_teacher = self.lambda_class * loss_class_teacher
            total_loss_student = (
                self.lambda_class * loss_class_student + self.lambda_align * loss_align + self.lambda_kl * loss_kl
            )

            total_loss_teacher.backward()
            self.optimizer_teacher.step()
            total_loss_student.backward()
            self.optimizer_student.step()

            return {
                "loss_class_teacher": loss_class_teacher.item(),
                "loss_class_student": loss_class_student.item(),
                "loss_align": loss_align.item(),
                "loss_kl": loss_kl.item(),
                "total_loss_teacher": total_loss_teacher.item(),
                "total_loss_student": total_loss_student.item(),
            }

    framework = ClassifierFramework(
        teacher=teacher,
        student=student,
        lambda_recon=0.0,
        lambda_class=args.lambda_class,
        lambda_align=args.lambda_align,
        lambda_kl=args.lambda_kl,
        learning_rate=learning_rate,
    )

    # 计算并设置类别权重（用于训练损失）
    if args.class_weights.lower() == "auto":
        counts = np.bincount(y_train, minlength=num_classes)
        counts = np.maximum(counts, 1)
        inv_freq = counts.sum() / (num_classes * counts)
        class_weights = torch.tensor(inv_freq, dtype=torch.float32, device=device)
    else:
        weight_values = [float(v.strip()) for v in args.class_weights.split(",") if v.strip()]
        if len(weight_values) != num_classes:
            raise ValueError(f"class-weights must have {num_classes} values, got {len(weight_values)}")
        class_weights = torch.tensor(weight_values, dtype=torch.float32, device=device)

    framework.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights: {class_weights.tolist()}")

    # 根据参数生成实验子目录，避免不同参数结果互相覆盖
    safe_weights = args.class_weights.replace(",", "-")
    run_name = (
        f"epochs{epochs}_lr{learning_rate}_bs{batch_size}_"
        f"k{args.tcn_kernel}_l{args.tcn_layers}_d{args.tcn_dropout}_db{args.tcn_dilation_base}_"
        f"wd{args.weight_decay}_cw{safe_weights}_seed{seed}_"
        f"lc{args.lambda_class}_la{args.lambda_align}_lk{args.lambda_kl}"
    )
    run_output_dir = output_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # 保存本次运行的全部命令行参数
    args_payload = vars(args).copy()
    args_payload["resolved_output_dir"] = str(run_output_dir)
    with open(run_output_dir / "run_args.json", "w") as f:
        json.dump(args_payload, f, indent=2, ensure_ascii=True)

    # 正则化：使用 AdamW 并设置 weight decay
    framework.optimizer_teacher = torch.optim.AdamW(
        framework.teacher.parameters(), lr=learning_rate, weight_decay=args.weight_decay
    )
    framework.optimizer_student = torch.optim.AdamW(
        framework.student.parameters(), lr=learning_rate, weight_decay=args.weight_decay
    )

    start_time = time.time()
    history = train_model(framework, train_loader, epochs=epochs, device=device)
    print(f"\nTotal training time: {time.time() - start_time:.2f} seconds")

    # 保存训练曲线
    plot_training_history(history, save_path=str(run_output_dir / "training_history.png"))

    print("\n" + "=" * 60)
    print("Final Evaluation (Teacher/Student on TRAIN & TEST Sets)")
    print("=" * 60)
    framework.teacher.to(device)
    framework.student.to(device)
    class_names = ["Class 0", "Class 1", "Class 2"]

    train_class_s, train_acc_s, train_report_s = evaluate_loader_with_report(
        framework.student, train_loader, device, class_names, use_teacher=False
    )
    test_class_s, test_acc_s, test_report_s = evaluate_loader_with_report(
        framework.student, test_loader, device, class_names, use_teacher=False
    )

    train_class_t, train_acc_t, train_report_t = evaluate_loader_with_report(
        framework.teacher, train_loader, device, class_names, use_teacher=True
    )
    test_class_t, test_acc_t, test_report_t = evaluate_loader_with_report(
        framework.teacher, test_loader, device, class_names, use_teacher=True
    )

    # 将评估结果写入文件，便于与基线对比
    with open(run_output_dir / "evaluation_metrics.txt", "w") as f:
        f.write("--- Student Train Metrics ---\n")
        f.write(f"Classification Loss (CE): {train_class_s:.6f}\n")
        f.write(f"Accuracy: {train_acc_s * 100:.2f}%\n")
        f.write("\n=== Classification Report (Student Train) ===\n")
        f.write(train_report_s + "\n")

        f.write("\n--- Student Test Metrics ---\n")
        f.write(f"Classification Loss (CE): {test_class_s:.6f}\n")
        f.write(f"Accuracy: {test_acc_s * 100:.2f}%\n")
        f.write("\n=== Classification Report (Student Test) ===\n")
        f.write(test_report_s + "\n")

        f.write("\n--- Teacher Train Metrics ---\n")
        f.write(f"Classification Loss (CE): {train_class_t:.6f}\n")
        f.write(f"Accuracy: {train_acc_t * 100:.2f}%\n")
        f.write("\n=== Classification Report (Teacher Train) ===\n")
        f.write(train_report_t + "\n")

        f.write("\n--- Teacher Test Metrics ---\n")
        f.write(f"Classification Loss (CE): {test_class_t:.6f}\n")
        f.write(f"Accuracy: {test_acc_t * 100:.2f}%\n")
        f.write("\n=== Classification Report (Teacher Test) ===\n")
        f.write(test_report_t + "\n")

    print("\n--- Student Train Metrics ---")
    print(f"Classification Loss (CE): {train_class_s:.6f}")
    print(f"Accuracy: {train_acc_s * 100:.2f}%")
    print("\n=== Classification Report (Student Train) ===")
    print(train_report_s)

    print("\n--- Student Test Metrics ---")
    print(f"Classification Loss (CE): {test_class_s:.6f}")
    print(f"Accuracy: {test_acc_s * 100:.2f}%")
    print("\n=== Classification Report (Student Test) ===")
    print(test_report_s)

    print("\n--- Teacher Train Metrics ---")
    print(f"Classification Loss (CE): {train_class_t:.6f}")
    print(f"Accuracy: {train_acc_t * 100:.2f}%")
    print("\n=== Classification Report (Teacher Train) ===")
    print(train_report_t)

    print("\n--- Teacher Test Metrics ---")
    print(f"Classification Loss (CE): {test_class_t:.6f}")
    print(f"Accuracy: {test_acc_t * 100:.2f}%")
    print("\n=== Classification Report (Teacher Test) ===")
    print(test_report_t)

    # 保存模型权重
    torch.save(teacher.state_dict(), run_output_dir / "teacher_classifier_benchmark_tcn.pth")
    torch.save(student.state_dict(), run_output_dir / "student_classifier_benchmark_tcn.pth")
    print(f"\nModel weights saved to {run_output_dir}")


if __name__ == "__main__":
    main()
