"""
Knowledge Distillation Loss Functions
Includes KL Divergence, MSE, and combined loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining multiple loss components:
    - KL Divergence between teacher and student outputs
    - MSE loss for output alignment
    - Task loss (ground truth)
    """
    
    def __init__(self, temperature=3.0, alpha=0.5, beta=0.3):
        """
        Initialize the knowledge distillation loss.
        
        Args:
            temperature (float): Temperature for softening probability distributions
            alpha (float): Weight for KL divergence loss (default: 0.5)
            beta (float): Weight for MSE loss (default: 0.3)
            The weight for task loss is (1 - alpha - beta)
        """
        super(KnowledgeDistillationLoss, self).__init__()
        
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.task_weight = 1.0 - alpha - beta
        
        assert self.task_weight >= 0, "alpha + beta should not exceed 1.0"
        
        self.mse_loss = nn.MSELoss()
        self.task_loss = nn.MSELoss()  # For regression task
        
    def forward(self, student_output, teacher_output, target=None):
        """
        Compute the combined knowledge distillation loss.
        
        Args:
            student_output (torch.Tensor): Student model predictions
            teacher_output (torch.Tensor): Teacher model predictions
            target (torch.Tensor, optional): Ground truth labels
        
        Returns:
            dict: Dictionary containing total loss and individual loss components
        """
        # KL Divergence Loss with temperature scaling
        # For regression, we use soft targets
        kl_loss = F.kl_div(
            F.log_softmax(student_output / self.temperature, dim=1),
            F.softmax(teacher_output / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # MSE Loss between teacher and student outputs
        mse_loss = self.mse_loss(student_output, teacher_output)
        
        # Task Loss (if ground truth is provided)
        if target is not None:
            task_loss = self.task_loss(student_output, target)
            total_loss = (self.alpha * kl_loss + 
                         self.beta * mse_loss + 
                         self.task_weight * task_loss)
        else:
            total_loss = self.alpha * kl_loss + self.beta * mse_loss
            task_loss = torch.tensor(0.0)
        
        return {
            'total_loss': total_loss,
            'kl_loss': kl_loss,
            'mse_loss': mse_loss,
            'task_loss': task_loss
        }


class RegressionDistillationLoss(nn.Module):
    """
    Simplified distillation loss for regression tasks.
    Combines MSE between teacher-student and student-target.
    """
    
    def __init__(self, alpha=0.7):
        """
        Initialize regression distillation loss.
        
        Args:
            alpha (float): Weight for distillation loss (1-alpha for task loss)
        """
        super(RegressionDistillationLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        
    def forward(self, student_output, teacher_output, target):
        """
        Compute regression distillation loss.
        
        Args:
            student_output (torch.Tensor): Student predictions
            teacher_output (torch.Tensor): Teacher predictions
            target (torch.Tensor): Ground truth
        
        Returns:
            dict: Loss components
        """
        # Distillation loss (student learns from teacher)
        distill_loss = self.mse_loss(student_output, teacher_output)
        
        # Task loss (student learns from ground truth)
        task_loss = self.mse_loss(student_output, target)
        
        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * task_loss
        
        return {
            'total_loss': total_loss,
            'distill_loss': distill_loss,
            'task_loss': task_loss
        }
