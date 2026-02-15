"""
Training script for knowledge distillation
Trains teacher and student models for welding quality prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import numpy as np
import os
import argparse
from tqdm import tqdm

from models import TeacherModel, StudentModel
from utils import (
    RegressionDistillationLoss,
    create_synthetic_welding_data,
    prepare_data_loaders,
    preprocess_data
)


def train_teacher(model, train_loader, val_loader, num_epochs=50, 
                 learning_rate=0.001, device='cpu', save_path='teacher_model.pth'):
    """
    Train the teacher model on the full 18D feature set.
    
    Args:
        model: Teacher model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on (cpu/cuda)
        save_path: Path to save the best model
    
    Returns:
        dict: Training history
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    print("Training Teacher Model...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for features, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f'Model saved to {save_path}')
    
    return history


def train_student_with_distillation(student_model, teacher_model, 
                                   train_loader_student, val_loader_student,
                                   train_loader_teacher, val_loader_teacher,
                                   num_epochs=50, learning_rate=0.001, 
                                   device='cpu', save_path='student_model.pth',
                                   alpha=0.7):
    """
    Train the student model using knowledge distillation from teacher.
    
    Args:
        student_model: Student model
        teacher_model: Pre-trained teacher model
        train_loader_student: Training data loader for student (13D)
        val_loader_student: Validation data loader for student (13D)
        train_loader_teacher: Training data loader for teacher (18D)
        val_loader_teacher: Validation data loader for teacher (18D)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on (cpu/cuda)
        save_path: Path to save the best model
        alpha: Weight for distillation loss
    
    Returns:
        dict: Training history
    """
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # Teacher is frozen
    
    distillation_criterion = RegressionDistillationLoss(alpha=alpha)
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 
        'val_loss': [],
        'train_distill_loss': [],
        'train_task_loss': []
    }
    
    print("\nTraining Student Model with Knowledge Distillation...")
    for epoch in range(num_epochs):
        # Training phase
        student_model.train()
        total_loss = 0.0
        total_distill_loss = 0.0
        total_task_loss = 0.0
        
        # Iterate through both student and teacher data loaders
        iterator = zip(
            tqdm(train_loader_student, desc=f'Epoch {epoch+1}/{num_epochs}'),
            train_loader_teacher
        )
        
        for (student_features, student_labels), (teacher_features, teacher_labels) in iterator:
            student_features = student_features.to(device)
            student_labels = student_labels.to(device)
            teacher_features = teacher_features.to(device)
            
            # Get teacher predictions (no gradient needed)
            with torch.no_grad():
                teacher_outputs = teacher_model(teacher_features)
            
            # Get student predictions
            optimizer.zero_grad()
            student_outputs = student_model(student_features)
            
            # Compute distillation loss
            loss_dict = distillation_criterion(
                student_outputs, 
                teacher_outputs, 
                student_labels
            )
            
            loss = loss_dict['total_loss']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_distill_loss += loss_dict['distill_loss'].item()
            total_task_loss += loss_dict['task_loss'].item()
        
        avg_train_loss = total_loss / len(train_loader_student)
        avg_distill_loss = total_distill_loss / len(train_loader_student)
        avg_task_loss = total_task_loss / len(train_loader_student)
        
        # Validation phase
        student_model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for student_features, student_labels in val_loader_student:
                student_features = student_features.to(device)
                student_labels = student_labels.to(device)
                
                student_outputs = student_model(student_features)
                loss = nn.MSELoss()(student_outputs, student_labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader_student)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['train_distill_loss'].append(avg_distill_loss)
        history['train_task_loss'].append(avg_task_loss)
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, '
              f'Distill Loss: {avg_distill_loss:.4f}, '
              f'Task Loss: {avg_task_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(student_model.state_dict(), save_path)
            print(f'Model saved to {save_path}')
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Train welding prediction models with knowledge distillation')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs-teacher', type=int, default=50, help='Number of epochs for teacher')
    parser.add_argument('--epochs-student', type=int, default=50, help='Number of epochs for student')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.7, help='Weight for distillation loss')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of synthetic samples')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Generate synthetic data
    print("Generating synthetic welding data...")
    features, labels = create_synthetic_welding_data(n_samples=args.n_samples)
    
    # Preprocess data
    features, scaler = preprocess_data(features)
    
    # Split data into train and validation
    n_train = int(0.8 * len(features))
    train_data, val_data = features[:n_train], features[n_train:]
    train_labels, val_labels = labels[:n_train], labels[n_train:]
    
    # Define student feature indices (13 out of 18 features)
    # Select the most important features (for demonstration, use first 13)
    student_feature_indices = list(range(13))
    
    # Prepare data loaders
    data_loaders = prepare_data_loaders(
        train_data, train_labels,
        val_data, val_labels,
        student_feature_indices=student_feature_indices,
        batch_size=args.batch_size
    )
    
    # Initialize models
    teacher_model = TeacherModel(input_dim=18, hidden_dims=[64, 32, 16], output_dim=1)
    student_model = StudentModel(input_dim=13, hidden_dims=[32, 16], output_dim=1)
    
    print(f"\nTeacher Model Parameters: {teacher_model.get_num_params()}")
    print(f"Student Model Parameters: {student_model.get_num_params()}")
    
    # Train teacher model
    teacher_save_path = os.path.join(args.save_dir, 'teacher_model.pth')
    teacher_history = train_teacher(
        teacher_model,
        data_loaders['train_teacher'],
        data_loaders['val_teacher'],
        num_epochs=args.epochs_teacher,
        learning_rate=args.lr,
        device=device,
        save_path=teacher_save_path
    )
    
    # Load best teacher model
    teacher_model.load_state_dict(torch.load(teacher_save_path))
    
    # Train student model with knowledge distillation
    student_save_path = os.path.join(args.save_dir, 'student_model.pth')
    student_history = train_student_with_distillation(
        student_model,
        teacher_model,
        data_loaders['train_student'],
        data_loaders['val_student'],
        data_loaders['train_teacher'],
        data_loaders['val_teacher'],
        num_epochs=args.epochs_student,
        learning_rate=args.lr,
        device=device,
        save_path=student_save_path,
        alpha=args.alpha
    )
    
    print("\nTraining completed!")
    print(f"Teacher model saved to: {teacher_save_path}")
    print(f"Student model saved to: {student_save_path}")
    

if __name__ == '__main__':
    main()
