"""
Inference and evaluation script for welding quality prediction
"""

import torch
import numpy as np
import argparse
import os

from models import TeacherModel, StudentModel
from utils import create_synthetic_welding_data, preprocess_data


def evaluate_model(model, features, labels, device='cpu'):
    """
    Evaluate model performance on given data.
    
    Args:
        model: Trained model
        features: Input features
        labels: Ground truth labels
        device: Device to use
    
    Returns:
        dict: Evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    features_tensor = torch.FloatTensor(features).to(device)
    labels_tensor = torch.FloatTensor(labels).to(device)
    
    with torch.no_grad():
        predictions = model(features_tensor)
    
    # Compute metrics
    mse = torch.nn.functional.mse_loss(predictions, labels_tensor.unsqueeze(1))
    mae = torch.nn.functional.l1_loss(predictions, labels_tensor.unsqueeze(1))
    
    # R-squared
    ss_res = torch.sum((labels_tensor.unsqueeze(1) - predictions) ** 2)
    ss_tot = torch.sum((labels_tensor.unsqueeze(1) - labels_tensor.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    return {
        'mse': mse.item(),
        'mae': mae.item(),
        'rmse': np.sqrt(mse.item()),
        'r2': r2.item()
    }


def predict(model, features, device='cpu'):
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained model
        features: Input features
        device: Device to use
    
    Returns:
        np.ndarray: Predictions
    """
    model = model.to(device)
    model.eval()
    
    features_tensor = torch.FloatTensor(features).to(device)
    
    with torch.no_grad():
        predictions = model(features_tensor)
    
    return predictions.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Evaluate welding prediction models')
    parser.add_argument('--model-type', type=str, required=True, 
                       choices=['teacher', 'student'],
                       help='Type of model to evaluate')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')
    parser.add_argument('--n-samples', type=int, default=200,
                       help='Number of test samples to generate')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Generate test data
    print("Generating test data...")
    features, labels = create_synthetic_welding_data(n_samples=args.n_samples, random_state=123)
    features, _ = preprocess_data(features)
    
    # Initialize model
    if args.model_type == 'teacher':
        model = TeacherModel(input_dim=18, hidden_dims=[64, 32, 16], output_dim=1)
        test_features = features
    else:
        model = StudentModel(input_dim=13, hidden_dims=[32, 16], output_dim=1)
        # Use only first 13 features for student
        test_features = features[:, :13]
    
    # Load model weights
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded {args.model_type} model from {args.model_path}")
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, test_features, labels, device=device)
    
    print(f"\n{args.model_type.capitalize()} Model Performance:")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  R²:   {metrics['r2']:.6f}")
    
    # Make sample predictions
    print("\nSample Predictions (first 5):")
    predictions = predict(model, test_features[:5], device=device)
    for i in range(5):
        print(f"  Sample {i+1}: True={labels[i]:.4f}, Predicted={predictions[i][0]:.4f}")


if __name__ == '__main__':
    main()
