"""
Configuration file for knowledge distillation models
"""

# Model Configuration
TEACHER_CONFIG = {
    'input_dim': 18,
    'hidden_dims': [64, 32, 16],
    'output_dim': 1,
    'dropout': 0.2
}

STUDENT_CONFIG = {
    'input_dim': 13,
    'hidden_dims': [32, 16],
    'output_dim': 1,
    'dropout': 0.2
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'teacher_epochs': 50,
    'student_epochs': 50,
    'weight_decay': 1e-5
}

# Distillation Configuration
DISTILLATION_CONFIG = {
    'temperature': 3.0,
    'alpha': 0.7,  # Weight for distillation loss
    'beta': 0.3,   # Weight for MSE loss (only for KL-based distillation)
}

# Data Configuration
DATA_CONFIG = {
    'n_samples': 1000,
    'train_ratio': 0.8,
    'random_state': 42,
    'student_feature_indices': list(range(13)),  # Use first 13 features
}

# Paths
PATHS = {
    'checkpoints': './checkpoints',
    'data': './data',
    'logs': './logs'
}
