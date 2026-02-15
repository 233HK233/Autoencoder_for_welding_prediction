#!/usr/bin/env python
"""
Quick start example for knowledge distillation training
"""

import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
from models import TeacherModel, StudentModel
from utils import (
    create_synthetic_welding_data,
    prepare_data_loaders,
    preprocess_data
)


def main():
    """
    Quick start example demonstrating knowledge distillation workflow.
    """
    print("=" * 60)
    print("Knowledge Distillation for Welding Quality Prediction")
    print("Quick Start Example")
    print("=" * 60)
    
    # Configuration
    n_samples = 500
    batch_size = 32
    device = 'cpu'
    
    print(f"\n1. Generating synthetic welding data ({n_samples} samples)...")
    features, labels = create_synthetic_welding_data(n_samples=n_samples)
    print(f"   Features shape: {features.shape}")
    print(f"   Labels shape: {labels.shape}")
    
    # Preprocess
    print("\n2. Preprocessing data (standardization)...")
    features, scaler = preprocess_data(features)
    
    # Split data
    n_train = int(0.8 * len(features))
    train_data, val_data = features[:n_train], features[n_train:]
    train_labels, val_labels = labels[:n_train], labels[n_train:]
    print(f"   Training samples: {len(train_data)}")
    print(f"   Validation samples: {len(val_data)}")
    
    # Prepare data loaders
    print("\n3. Preparing data loaders...")
    student_feature_indices = list(range(13))  # First 13 features for student
    data_loaders = prepare_data_loaders(
        train_data, train_labels,
        val_data, val_labels,
        student_feature_indices=student_feature_indices,
        batch_size=batch_size
    )
    
    # Initialize models
    print("\n4. Initializing models...")
    teacher_model = TeacherModel(input_dim=18, hidden_dims=[64, 32, 16], output_dim=1)
    student_model = StudentModel(input_dim=13, hidden_dims=[32, 16], output_dim=1)
    
    print(f"   Teacher model parameters: {teacher_model.get_num_params():,}")
    print(f"   Student model parameters: {student_model.get_num_params():,}")
    print(f"   Compression ratio: {teacher_model.get_num_params() / student_model.get_num_params():.2f}x")
    
    # Display model architectures
    print("\n5. Model Architectures:")
    print("\n   Teacher Model (18D input):")
    print(f"   {teacher_model}")
    print("\n   Student Model (13D input):")
    print(f"   {student_model}")
    
    # Test forward pass
    print("\n6. Testing forward pass...")
    sample_teacher_input = torch.randn(4, 18)
    sample_student_input = torch.randn(4, 13)
    
    teacher_output = teacher_model(sample_teacher_input)
    student_output = student_model(sample_student_input)
    
    print(f"   Teacher output shape: {teacher_output.shape}")
    print(f"   Student output shape: {student_output.shape}")
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)
    print("\nTo train the models, run:")
    print("  python train.py --batch-size 32 --epochs-teacher 50 --epochs-student 50")
    print("\nTo evaluate trained models, run:")
    print("  python evaluate.py --model-type teacher --model-path ./checkpoints/teacher_model.pth")
    print("  python evaluate.py --model-type student --model-path ./checkpoints/student_model.pth")
    print("=" * 60)


if __name__ == '__main__':
    main()
