#!/usr/bin/env python3
"""
Benchmark-aligned Teacher (18D) / Student (14D) classifier-only training.
Uses classifier-only models (no decoder) for reduced capacity.
"""

# use gpu 3 only
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random

try:
    from sklearn.metrics import classification_report as sk_classification_report
except Exception:
    sk_classification_report = None

try:
    from .data_utils import load_sequences_from_folder, standardize_train_test, load_npz_dataset
    from .framework import TeacherStudentFramework, plot_training_history, train_model
    from .models_slim_classifier_only_lstm import StudentClassifier, TeacherClassifier
except ImportError:
    from data_utils import load_sequences_from_folder, standardize_train_test, load_npz_dataset
    from framework import TeacherStudentFramework, plot_training_history, train_model
    from models_slim_classifier_only_lstm import StudentClassifier, TeacherClassifier


def build_classification_report(
    y_true: List[int], y_pred: List[int], class_names: Optional[List[str]] = None
) -> str:
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


# ... (imports remain the same)
import argparse

# ... (reporting functions remain the same)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Teacher-Student Classifier")
    parser.add_argument("--epochs", type=int, default=250, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save outputs")
    parser.add_argument("--dataset-npz", type=str, default=None, help="Path to prepared .npz (from prepare_weld_seam_dataset.py).")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use args or defaults
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    OUTPUT_DIR = args.output_dir
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    WINDOW_SIZE = 10  # match LSTM-MLP
    TARGET_OFFSET = 5  # match LSTM-MLP
    TEACHER_FEATURES = 18
    STUDENT_FEATURES = 14
    DROP_FEATURE_INDICES = [1, 3, 5, 11]
    NUM_CLASSES = 3
    # Use the same latent dimension for teacher/student to avoid size mismatch in alignment.
    LATENT_DIM_TEACHER = 32
    LATENT_DIM_STUDENT = 32
    NUM_SEGMENTS = 2

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_DATA_PATH = os.path.join(SCRIPT_DIR, "Data", "train_data")
    TEST_DATA_PATH = os.path.join(SCRIPT_DIR, "Data", "test_data")

    print("=" * 60)
    print("Benchmark-aligned Teacher (18D) / Student (14D) Classifier-Only")
    print("=" * 60)
    print(f"Config: window_size={WINDOW_SIZE}, target_offset={TARGET_OFFSET}, teacher_feats={TEACHER_FEATURES}, student_feats={STUDENT_FEATURES}")
    print(f"Dropped indices for student: {DROP_FEATURE_INDICES}")
    print(f"Latent dims: teacher={LATENT_DIM_TEACHER}, student={LATENT_DIM_STUDENT}, Segments: {NUM_SEGMENTS}, Epochs: {EPOCHS}")
    print(f"Hyperparams: LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Output={OUTPUT_DIR}")
    print("=" * 60)

    # Load data (sliding window, stride=1)
    # Load data
    if args.dataset_npz:
        ds = load_npz_dataset(args.dataset_npz)
        X_train_full = ds["X_train_full"].astype(np.float32)
        y_train = ds["y_train"].astype(np.int64)
        X_test_full = ds["X_test_full"].astype(np.float32)
        y_test = ds["y_test"].astype(np.int64)
    else:
        print("\nLoading training data...")
        train_sequences, train_labels = load_sequences_from_folder(TRAIN_DATA_PATH, WINDOW_SIZE, TARGET_OFFSET)
        if len(train_sequences) == 0:
            raise ValueError(f"No valid training data found in {TRAIN_DATA_PATH}")
        X_train_raw = np.array(train_sequences, dtype=np.float32)
        y_train = np.array(train_labels, dtype=np.int64)

        print("\nLoading test data...")
        test_sequences, test_labels = load_sequences_from_folder(TEST_DATA_PATH, WINDOW_SIZE, TARGET_OFFSET)
        if len(test_sequences) == 0:
            raise ValueError(f"No valid test data found in {TEST_DATA_PATH}")
        X_test_raw = np.array(test_sequences, dtype=np.float32)
        y_test = np.array(test_labels, dtype=np.int64)

        X_train_full, X_test_full = standardize_train_test(X_train_raw, X_test_raw)
    keep_indices: List[int] = [i for i in range(TEACHER_FEATURES) if i not in DROP_FEATURE_INDICES]
    X_train_subset = X_train_full[:, :, keep_indices]
    X_test_subset = X_test_full[:, :, keep_indices]

    print(f"Train shapes: full {X_train_full.shape}, subset {X_train_subset.shape}, labels {y_train.shape}")
    print(f"Test shapes: full {X_test_full.shape}, subset {X_test_subset.shape}, labels {y_test.shape}")

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    teacher = TeacherClassifier(
        input_dim=TEACHER_FEATURES,
        latent_dim=LATENT_DIM_TEACHER,
        num_classes=NUM_CLASSES,
        num_segments=NUM_SEGMENTS,
        sequence_length=WINDOW_SIZE,
    )
    student = StudentClassifier(
        input_dim=STUDENT_FEATURES,
        latent_dim=LATENT_DIM_STUDENT,
        num_classes=NUM_CLASSES,
        num_segments=NUM_SEGMENTS,
        sequence_length=WINDOW_SIZE,
    )
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student parameters: {student_params:,}")

    # Simple framework wrapper for classification-only
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
        lambda_recon=0.0,  # not used
        lambda_class=1.0,
        lambda_align=0.5,
        lambda_kl=0.3,
        learning_rate=LEARNING_RATE,
    )

    start_time = time.time()
    history = train_model(framework, train_loader, epochs=EPOCHS, device=device)
    print(f"\nTotal training time: {time.time() - start_time:.2f} seconds")
    
    # Save history plot
    plot_training_history(history, save_path=os.path.join(OUTPUT_DIR, "training_history.png"))

    print("\n" + "=" * 60)
    print("Final Evaluation (Teacher/Student on TRAIN & TEST Sets)")
    print("=" * 60)
    framework.teacher.to(device)
    framework.student.to(device)
    class_names = ["Class 0", "Class 1", "Class 2"]

    # Student metrics
    train_class_s, train_acc_s, train_report_s = evaluate_loader_with_report(
        framework.student, train_loader, device, class_names, use_teacher=False
    )
    test_class_s, test_acc_s, test_report_s = evaluate_loader_with_report(
        framework.student, test_loader, device, class_names, use_teacher=False
    )

    # Teacher metrics
    train_class_t, train_acc_t, train_report_t = evaluate_loader_with_report(
        framework.teacher, train_loader, device, class_names, use_teacher=True
    )
    test_class_t, test_acc_t, test_report_t = evaluate_loader_with_report(
        framework.teacher, test_loader, device, class_names, use_teacher=True
    )

    # Save metrics to text file
    with open(os.path.join(OUTPUT_DIR, "evaluation_metrics.txt"), "w") as f:
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

    torch.save(teacher.state_dict(), os.path.join(OUTPUT_DIR, "teacher_classifier_benchmark.pth"))
    torch.save(student.state_dict(), os.path.join(OUTPUT_DIR, "student_classifier_benchmark.pth"))
    print(f"\nModel weights saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
