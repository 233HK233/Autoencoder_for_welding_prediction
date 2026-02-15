"""
Teacher Model for Welding Quality Prediction
18-dimensional input model
"""

import torch
import torch.nn as nn


class TeacherModel(nn.Module):
    """
    Teacher model with 18-dimensional input for welding quality prediction.
    This model serves as the teacher in knowledge distillation.
    """
    
    def __init__(self, input_dim=18, hidden_dims=[64, 32, 16], output_dim=1):
        """
        Initialize the teacher model.
        
        Args:
            input_dim (int): Input dimension (default: 18)
            hidden_dims (list): List of hidden layer dimensions
            output_dim (int): Output dimension (default: 1 for quality score)
        """
        super(TeacherModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x, return_features=False):
        """
        Forward pass through the teacher model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 18)
            return_features (bool): If True, return intermediate features for distillation
        
        Returns:
            torch.Tensor: Output predictions
            torch.Tensor (optional): Intermediate features if return_features=True
        """
        if return_features:
            features = []
            for i, layer in enumerate(self.model):
                x = layer(x)
                # Store features after each ReLU activation
                if isinstance(layer, nn.ReLU):
                    features.append(x)
            return x, features
        else:
            return self.model(x)
    
    def get_num_params(self):
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
