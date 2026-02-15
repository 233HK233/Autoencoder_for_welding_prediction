"""
Data preprocessing and loading utilities for welding prediction
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd


class WeldingDataset(Dataset):
    """
    Dataset for welding quality prediction with support for both
    18D (teacher) and 13D (student) features.
    """
    
    def __init__(self, data, labels, feature_indices=None, transform=None):
        """
        Initialize the welding dataset.
        
        Args:
            data (np.ndarray): Feature data
            labels (np.ndarray): Target labels (quality scores)
            feature_indices (list, optional): Indices of features to use (for student)
            transform (callable, optional): Optional transform to apply to features
        """
        self.data = data
        self.labels = labels
        self.feature_indices = feature_indices
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = self.data[idx]
        
        # Select specific features if indices provided (for student model)
        if self.feature_indices is not None:
            features = features[self.feature_indices]
        
        # Apply transform if provided
        if self.transform:
            features = self.transform(features)
        
        # Convert to tensors
        features = torch.FloatTensor(features)
        label = torch.FloatTensor([self.labels[idx]])
        
        return features, label


def create_synthetic_welding_data(n_samples=1000, n_features=18, random_state=42):
    """
    Create synthetic welding data for demonstration and testing.
    
    Args:
        n_samples (int): Number of samples to generate
        n_features (int): Number of features (should be 18 for teacher)
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (features, labels)
    """
    np.random.seed(random_state)
    
    # Generate random features representing welding parameters
    # Features could represent: current, voltage, speed, gas flow, temperature, etc.
    features = np.random.randn(n_samples, n_features)
    
    # Generate labels based on a simple relationship
    # Quality score is influenced by multiple features
    weights = np.random.randn(n_features)
    labels = np.dot(features, weights) + np.random.randn(n_samples) * 0.5
    
    # Normalize labels to [0, 1] range
    labels = (labels - labels.min()) / (labels.max() - labels.min())
    
    return features, labels


def prepare_data_loaders(train_data, train_labels, val_data, val_labels,
                        student_feature_indices=None, batch_size=32, 
                        num_workers=0):
    """
    Prepare data loaders for teacher and student models.
    
    Args:
        train_data (np.ndarray): Training features
        train_labels (np.ndarray): Training labels
        val_data (np.ndarray): Validation features
        val_labels (np.ndarray): Validation labels
        student_feature_indices (list): Feature indices for student model
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
    
    Returns:
        dict: Dictionary containing data loaders
    """
    # Create datasets for teacher (uses all 18 features)
    train_dataset_teacher = WeldingDataset(train_data, train_labels)
    val_dataset_teacher = WeldingDataset(val_data, val_labels)
    
    # Create datasets for student (uses 13 selected features)
    train_dataset_student = WeldingDataset(
        train_data, train_labels, 
        feature_indices=student_feature_indices
    )
    val_dataset_student = WeldingDataset(
        val_data, val_labels,
        feature_indices=student_feature_indices
    )
    
    # Create data loaders
    train_loader_teacher = DataLoader(
        train_dataset_teacher, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader_teacher = DataLoader(
        val_dataset_teacher,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    train_loader_student = DataLoader(
        train_dataset_student,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader_student = DataLoader(
        val_dataset_student,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return {
        'train_teacher': train_loader_teacher,
        'val_teacher': val_loader_teacher,
        'train_student': train_loader_student,
        'val_student': val_loader_student
    }


def preprocess_data(data, scaler=None):
    """
    Preprocess welding data with standardization.
    
    Args:
        data (np.ndarray): Raw feature data
        scaler (StandardScaler, optional): Fitted scaler, if None creates new one
    
    Returns:
        tuple: (preprocessed_data, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        preprocessed_data = scaler.fit_transform(data)
    else:
        preprocessed_data = scaler.transform(data)
    
    return preprocessed_data, scaler
