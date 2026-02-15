"""
Model comparison utility
Compare performance of teacher and student models
"""

import os
import sys
import argparse
import torch
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from models import TeacherModel, StudentModel
from utils import create_synthetic_welding_data, preprocess_data


def compare_models(teacher_path, student_path, n_samples=200, device='cpu'):
    """
    Compare teacher and student model performance.
    
    Args:
        teacher_path: Path to teacher model checkpoint
        student_path: Path to student model checkpoint
        n_samples: Number of test samples
        device: Device to use
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    # Load models
    print("\nLoading models...")
    teacher_model = TeacherModel(input_dim=18, hidden_dims=[64, 32, 16], output_dim=1)
    student_model = StudentModel(input_dim=13, hidden_dims=[32, 16], output_dim=1)
    
    if not os.path.exists(teacher_path):
        print(f"Error: Teacher model not found at {teacher_path}")
        return
    if not os.path.exists(student_path):
        print(f"Error: Student model not found at {student_path}")
        return
    
    teacher_model.load_state_dict(torch.load(teacher_path, map_location=device))
    student_model.load_state_dict(torch.load(student_path, map_location=device))
    
    teacher_model.eval()
    student_model.eval()
    
    print(f"✓ Teacher model loaded from {teacher_path}")
    print(f"✓ Student model loaded from {student_path}")
    
    # Generate test data
    print(f"\nGenerating {n_samples} test samples...")
    features, labels = create_synthetic_welding_data(n_samples=n_samples, random_state=999)
    features, _ = preprocess_data(features)
    
    # Prepare data for both models
    teacher_features = torch.FloatTensor(features)
    student_features = torch.FloatTensor(features[:, :13])  # First 13 features
    labels_tensor = torch.FloatTensor(labels).unsqueeze(1)
    
    # Make predictions
    print("\nMaking predictions...")
    with torch.no_grad():
        teacher_preds = teacher_model(teacher_features)
        student_preds = student_model(student_features)
    
    # Calculate metrics
    def calculate_metrics(predictions, targets):
        mse = torch.nn.functional.mse_loss(predictions, targets)
        mae = torch.nn.functional.l1_loss(predictions, targets)
        rmse = torch.sqrt(mse)
        
        # R-squared
        ss_res = torch.sum((targets - predictions) ** 2)
        ss_tot = torch.sum((targets - targets.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        return {
            'MSE': mse.item(),
            'RMSE': rmse.item(),
            'MAE': mae.item(),
            'R²': r2.item()
        }
    
    teacher_metrics = calculate_metrics(teacher_preds, labels_tensor)
    student_metrics = calculate_metrics(student_preds, labels_tensor)
    
    # Display comparison
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    print(f"\n{'Metric':<15} {'Teacher':<20} {'Student':<20} {'Ratio':<15}")
    print("-" * 70)
    
    for metric in ['MSE', 'RMSE', 'MAE', 'R²']:
        teacher_val = teacher_metrics[metric]
        student_val = student_metrics[metric]
        
        if metric == 'R²':
            ratio_str = "N/A"
        else:
            ratio = student_val / teacher_val if teacher_val != 0 else float('inf')
            ratio_str = f"{ratio:.2f}x"
        
        print(f"{metric:<15} {teacher_val:<20.6f} {student_val:<20.6f} {ratio_str:<15}")
    
    # Model size comparison
    print("\n" + "="*70)
    print("MODEL SIZE COMPARISON")
    print("="*70)
    
    teacher_params = teacher_model.get_num_params()
    student_params = student_model.get_num_params()
    compression_ratio = teacher_params / student_params
    
    print(f"\n{'Model':<15} {'Parameters':<20} {'Input Dim':<15} {'Size':<15}")
    print("-" * 70)
    print(f"{'Teacher':<15} {teacher_params:<20,} {18:<15} {'100%':<15}")
    print(f"{'Student':<15} {student_params:<20,} {13:<15} {f'{(student_params/teacher_params)*100:.1f}%':<15}")
    
    print(f"\nCompression Ratio: {compression_ratio:.2f}x")
    print(f"Parameter Reduction: {(1 - student_params/teacher_params)*100:.1f}%")
    
    # Sample predictions
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS (First 5)")
    print("="*70)
    print(f"\n{'Sample':<10} {'True':<15} {'Teacher':<15} {'Student':<15} {'T-Error':<15} {'S-Error':<15}")
    print("-" * 90)
    
    for i in range(min(5, n_samples)):
        true_val = labels[i]
        teacher_pred = teacher_preds[i].item()
        student_pred = student_preds[i].item()
        teacher_error = abs(true_val - teacher_pred)
        student_error = abs(true_val - student_pred)
        
        print(f"{i+1:<10} {true_val:<15.4f} {teacher_pred:<15.4f} {student_pred:<15.4f} "
              f"{teacher_error:<15.4f} {student_error:<15.4f}")
    
    print("\n" + "="*70)
    
    # Summary
    performance_gap = (student_metrics['MSE'] / teacher_metrics['MSE'] - 1) * 100
    
    print("\nSUMMARY:")
    print(f"  • Student model is {compression_ratio:.2f}x smaller than teacher")
    print(f"  • Student MSE is {performance_gap:+.1f}% compared to teacher")
    print(f"  • Student uses only 13 out of 18 features (72% of features)")
    
    if performance_gap < 50:
        print("  ✓ Good knowledge distillation performance!")
    else:
        print("  ⚠ Consider tuning distillation parameters for better performance")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Compare teacher and student models')
    parser.add_argument('--teacher-path', type=str, 
                       default='./checkpoints/teacher_model.pth',
                       help='Path to teacher model')
    parser.add_argument('--student-path', type=str,
                       default='./checkpoints/student_model.pth',
                       help='Path to student model')
    parser.add_argument('--n-samples', type=int, default=200,
                       help='Number of test samples')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    compare_models(
        args.teacher_path,
        args.student_path,
        args.n_samples,
        args.device
    )


if __name__ == '__main__':
    main()
