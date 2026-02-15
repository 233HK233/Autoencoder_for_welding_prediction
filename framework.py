"""
Training framework for Teacher-Student distillation (benchmark).
"""
from typing import Dict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Optional type hints; this repo may not ship autoencoder-style models.py
    from torch.nn import Module as TeacherAutoencoder
    from torch.nn import Module as StudentAutoencoder

# Teacher-Student框架，用于训练教师模型和学生模型，实现知识蒸馏
class TeacherStudentFramework:
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        lambda_recon: float = 1.0,
        lambda_class: float = 1.0,
        lambda_align: float = 0.5,
        lambda_kl: float = 0.3,
        learning_rate: float = 1e-4,
    ):
        self.teacher = teacher
        self.student = student
        # 重构损失权重(有Decoder才用)
        self.lambda_recon = lambda_recon
        self.lambda_class = lambda_class
        # student teacher 对齐损失权重
        self.lambda_align = lambda_align
        # KL散度损失权重，用于衡量student teacher的分布差异(概率分布)
        self.lambda_kl = lambda_kl

        self.optimizer_teacher = optim.Adam(teacher.parameters(), lr=learning_rate)
        self.optimizer_student = optim.Adam(student.parameters(), lr=learning_rate)

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def train_step(self, x_full: torch.Tensor, x_subset: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        self.teacher.train()
        self.student.train()
        self.optimizer_teacher.zero_grad()
        self.optimizer_student.zero_grad()

        x_recon_teacher, logits_teacher, z_teacher = self.teacher(x_full, labels)
        loss_recon_teacher = self.mse_loss(x_recon_teacher, x_full)
        loss_class_teacher = self.ce_loss(logits_teacher, labels)

        x_recon_student, logits_student, z_student = self.student(x_subset)
        loss_recon_student = self.mse_loss(x_recon_student, x_subset)
        loss_class_student = self.ce_loss(logits_student, labels)

        loss_align = self.mse_loss(z_student, z_teacher.detach())
        teacher_probs = F.softmax(z_teacher.detach(), dim=1)
        student_log_probs = F.log_softmax(z_student, dim=1)
        loss_kl = self.kl_loss(student_log_probs, teacher_probs)

        total_loss_teacher = self.lambda_recon * loss_recon_teacher + self.lambda_class * loss_class_teacher
        total_loss_student = (
            self.lambda_recon * loss_recon_student
            + self.lambda_class * loss_class_student
            + self.lambda_align * loss_align
            + self.lambda_kl * loss_kl
        )

        total_loss_teacher.backward()
        self.optimizer_teacher.step()
        total_loss_student.backward()
        self.optimizer_student.step()

        return {
            "loss_recon_teacher": loss_recon_teacher.item(),
            "loss_class_teacher": loss_class_teacher.item(),
            "loss_recon_student": loss_recon_student.item(),
            "loss_class_student": loss_class_student.item(),
            "loss_align": loss_align.item(),
            "loss_kl": loss_kl.item(),
            "total_loss_teacher": total_loss_teacher.item(),
            "total_loss_student": total_loss_student.item(),
        }

    @torch.no_grad()
    def evaluate_student(self, x_subset: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        self.student.eval()
        x_recon, logits, _ = self.student(x_subset)
        preds = logits.argmax(dim=1)
        accuracy = (preds == labels).float().mean().item()
        return {
            "accuracy": accuracy,
            "loss_recon": self.mse_loss(x_recon, x_subset).item(),
            "loss_class": self.ce_loss(logits, labels).item(),
        }


def train_model(framework: TeacherStudentFramework, dataloader: DataLoader, epochs: int, device: torch.device) -> Dict[str, list]:
    history = {
        "loss_recon_teacher": [],
        "loss_class_teacher": [],
        "loss_recon_student": [],
        "loss_class_student": [],
        "loss_align": [],
        "loss_kl": [],
        "total_loss_teacher": [],
        "total_loss_student": [],
    }

    framework.teacher.to(device)
    framework.student.to(device)

    print("=" * 60)
    print("Starting Teacher-Student Training (Benchmark windowing)")
    print("=" * 60)

    for epoch in range(epochs):
        epoch_losses = {k: 0.0 for k in history.keys()}
        num_batches = 0

        for x_full, x_subset, y_batch in dataloader:
            x_full = x_full.to(device)
            x_subset = x_subset.to(device)
            y_batch = y_batch.to(device)

            losses = framework.train_step(x_full, x_subset, y_batch)
            for k, v in losses.items():
                epoch_losses[k] += v
            num_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= num_batches
            history[k].append(epoch_losses[k])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}]")
            print(
                f"  Teacher - Recon: {epoch_losses['loss_recon_teacher']:.6f}, "
                f"Class: {epoch_losses['loss_class_teacher']:.6f}"
            )
            print(
                f"  Student - Recon: {epoch_losses['loss_recon_student']:.6f}, "
                f"Class: {epoch_losses['loss_class_student']:.6f}, "
                f"Align: {epoch_losses['loss_align']:.6f}, "
                f"KL: {epoch_losses['loss_kl']:.6f}"
            )

    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    return history


def plot_training_history(history: Dict[str, list], save_path: str = "training_history.png") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(history["loss_recon_teacher"], label="Teacher", color="blue")
    ax.plot(history["loss_recon_student"], label="Student", color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Reconstruction Loss")
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(history["loss_class_teacher"], label="Teacher", color="blue")
    ax.plot(history["loss_class_student"], label="Student", color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CrossEntropy Loss")
    ax.set_title("Classification Loss")
    ax.legend()
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(history["loss_align"], label="Alignment (MSE)", color="green")
    ax.plot(history["loss_kl"], label="KL (Student||Teacher)", color="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Latent Alignment Loss")
    ax.legend()
    ax.grid(True)

    ax = axes[1, 1]
    ax.plot(history["total_loss_teacher"], label="Teacher Total", color="blue")
    ax.plot(history["total_loss_student"], label="Student Total", color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history saved to {save_path}")
