"""
Complete Demo: Knowledge Distillation for Welding Quality Prediction
This script demonstrates the complete workflow from data generation to model evaluation.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from models import TeacherModel, StudentModel
from utils import (
    create_synthetic_welding_data,
    prepare_data_loaders,
    preprocess_data,
    RegressionDistillationLoss
)


def demo_complete_workflow():
    """
    Demonstrate the complete knowledge distillation workflow.
    """
    print("\n" + "="*70)
    print("KNOWLEDGE DISTILLATION DEMO - COMPLETE WORKFLOW")
    print("="*70)
    
    # ========== Step 1: Data Generation ==========
    print("\n[Step 1] Generating synthetic welding data...")
    n_samples = 800
    features, labels = create_synthetic_welding_data(n_samples=n_samples)
    print(f"  Generated {n_samples} samples with 18 features")
    print(f"  Label range: [{labels.min():.4f}, {labels.max():.4f}]")
    
    # ========== Step 2: Data Preprocessing ==========
    print("\n[Step 2] Preprocessing data...")
    features, scaler = preprocess_data(features)
    
    # Split data
    n_train = int(0.8 * len(features))
    train_data, val_data = features[:n_train], features[n_train:]
    train_labels, val_labels = labels[:n_train], labels[n_train:]
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    
    # ========== Step 3: Prepare Data Loaders ==========
    print("\n[Step 3] Preparing data loaders...")
    student_feature_indices = list(range(13))  # Use first 13 features
    batch_size = 32
    
    data_loaders = prepare_data_loaders(
        train_data, train_labels,
        val_data, val_labels,
        student_feature_indices=student_feature_indices,
        batch_size=batch_size
    )
    print(f"  Batch size: {batch_size}")
    print(f"  Student using {len(student_feature_indices)} out of 18 features")
    
    # ========== Step 4: Initialize Models ==========
    print("\n[Step 4] Initializing models...")
    teacher_model = TeacherModel(input_dim=18, hidden_dims=[64, 32, 16], output_dim=1)
    student_model = StudentModel(input_dim=13, hidden_dims=[32, 16], output_dim=1)
    
    teacher_params = teacher_model.get_num_params()
    student_params = student_model.get_num_params()
    compression_ratio = teacher_params / student_params
    
    print(f"  Teacher parameters: {teacher_params:,}")
    print(f"  Student parameters: {student_params:,}")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Parameter reduction: {(1 - 1/compression_ratio)*100:.1f}%")
    
    # ========== Step 5: Train Teacher Model ==========
    print("\n[Step 5] Training teacher model...")
    teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=0.001)
    teacher_criterion = torch.nn.MSELoss()
    
    num_epochs_teacher = 5
    teacher_train_losses = []
    teacher_val_losses = []
    
    for epoch in range(num_epochs_teacher):
        # Training
        teacher_model.train()
        train_loss = 0.0
        for features_batch, labels_batch in data_loaders['train_teacher']:
            teacher_optimizer.zero_grad()
            outputs = teacher_model(features_batch)
            loss = teacher_criterion(outputs, labels_batch)
            loss.backward()
            teacher_optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(data_loaders['train_teacher'])
        
        # Validation
        teacher_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features_batch, labels_batch in data_loaders['val_teacher']:
                outputs = teacher_model(features_batch)
                loss = teacher_criterion(outputs, labels_batch)
                val_loss += loss.item()
        
        val_loss /= len(data_loaders['val_teacher'])
        
        teacher_train_losses.append(train_loss)
        teacher_val_losses.append(val_loss)
        
        print(f"  Epoch {epoch+1}/{num_epochs_teacher}: "
              f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    # ========== Step 6: Train Student with Distillation ==========
    print("\n[Step 6] Training student model with knowledge distillation...")
    student_optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    distillation_criterion = RegressionDistillationLoss(alpha=0.7)
    teacher_model.eval()
    
    num_epochs_student = 5
    student_train_losses = []
    student_val_losses = []
    distill_losses = []
    task_losses = []
    
    for epoch in range(num_epochs_student):
        # Training
        student_model.train()
        total_loss = 0.0
        total_distill = 0.0
        total_task = 0.0
        
        for (student_batch, student_labels), (teacher_batch, _) in zip(
            data_loaders['train_student'], 
            data_loaders['train_teacher']
        ):
            # Get teacher predictions (frozen)
            with torch.no_grad():
                teacher_outputs = teacher_model(teacher_batch)
            
            # Train student
            student_optimizer.zero_grad()
            student_outputs = student_model(student_batch)
            
            loss_dict = distillation_criterion(
                student_outputs, 
                teacher_outputs, 
                student_labels
            )
            
            loss = loss_dict['total_loss']
            loss.backward()
            student_optimizer.step()
            
            total_loss += loss.item()
            total_distill += loss_dict['distill_loss'].item()
            total_task += loss_dict['task_loss'].item()
        
        avg_train_loss = total_loss / len(data_loaders['train_student'])
        avg_distill = total_distill / len(data_loaders['train_student'])
        avg_task = total_task / len(data_loaders['train_student'])
        
        # Validation
        student_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features_batch, labels_batch in data_loaders['val_student']:
                outputs = student_model(features_batch)
                loss = torch.nn.MSELoss()(outputs, labels_batch)
                val_loss += loss.item()
        
        val_loss /= len(data_loaders['val_student'])
        
        student_train_losses.append(avg_train_loss)
        student_val_losses.append(val_loss)
        distill_losses.append(avg_distill)
        task_losses.append(avg_task)
        
        print(f"  Epoch {epoch+1}/{num_epochs_student}: "
              f"Total={avg_train_loss:.4f}, "
              f"Distill={avg_distill:.4f}, "
              f"Task={avg_task:.4f}, "
              f"Val={val_loss:.4f}")
    
    # ========== Step 7: Evaluation ==========
    print("\n[Step 7] Evaluating models...")
    
    def evaluate_model(model, data_loader):
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features_batch, labels_batch in data_loader:
                outputs = model(features_batch)
                all_preds.append(outputs)
                all_labels.append(labels_batch)
        
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        mse = torch.nn.functional.mse_loss(all_preds, all_labels)
        mae = torch.nn.functional.l1_loss(all_preds, all_labels)
        
        return {
            'mse': mse.item(),
            'rmse': np.sqrt(mse.item()),
            'mae': mae.item()
        }
    
    teacher_metrics = evaluate_model(teacher_model, data_loaders['val_teacher'])
    student_metrics = evaluate_model(student_model, data_loaders['val_student'])
    
    print("\n  Teacher Model Performance:")
    print(f"    MSE:  {teacher_metrics['mse']:.6f}")
    print(f"    RMSE: {teacher_metrics['rmse']:.6f}")
    print(f"    MAE:  {teacher_metrics['mae']:.6f}")
    
    print("\n  Student Model Performance:")
    print(f"    MSE:  {student_metrics['mse']:.6f}")
    print(f"    RMSE: {student_metrics['rmse']:.6f}")
    print(f"    MAE:  {student_metrics['mae']:.6f}")
    
    # Performance comparison
    performance_ratio = student_metrics['mse'] / teacher_metrics['mse']
    print(f"\n  Student MSE is {performance_ratio:.2f}x of Teacher MSE")
    print(f"  Model size reduction: {compression_ratio:.2f}x")
    
    # ========== Summary ==========
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Successfully trained teacher model (18D input, {teacher_params:,} params)")
    print(f"✓ Successfully trained student model (13D input, {student_params:,} params)")
    print(f"✓ Achieved {compression_ratio:.2f}x model compression")
    print(f"✓ Student model performance: {performance_ratio:.2f}x teacher MSE")
    print("="*70)
    
    return {
        'teacher_model': teacher_model,
        'student_model': student_model,
        'teacher_metrics': teacher_metrics,
        'student_metrics': student_metrics
    }


if __name__ == '__main__':
    results = demo_complete_workflow()
    print("\nDemo completed successfully!")
