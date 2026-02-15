# Knowledge Distillation for Welding Quality Prediction

基于知识蒸馏的焊缝质量预测深度学习模型

## 项目简介 (Project Overview)

本项目实现了一个基于知识蒸馏（Knowledge Distillation）的深度学习框架，用于焊缝质量预测。该框架包含：

- **Teacher Model**: 使用18维输入特征的深度神经网络
- **Student Model**: 使用13维输入特征的轻量级神经网络
- **知识蒸馏**: 利用KL散度、MSE等损失函数对齐Teacher和Student模型的输出
- **目标**: 获得一个性能优良且计算高效的焊缝质量预测模型

This project implements a knowledge distillation-based deep learning framework for welding quality prediction, featuring:

- **Teacher Model**: Deep neural network with 18-dimensional input features
- **Student Model**: Lightweight neural network with 13-dimensional input features  
- **Knowledge Distillation**: Aligns teacher and student outputs using KL divergence, MSE loss
- **Goal**: Achieve a high-performance, computationally efficient welding quality prediction model

## 项目结构 (Project Structure)

```
Autoencoder_for_welding_prediction/
├── models/
│   ├── __init__.py
│   ├── teacher_model.py      # 18D Teacher model
│   └── student_model.py      # 13D Student model
├── utils/
│   ├── __init__.py
│   ├── loss.py              # Knowledge distillation losses
│   └── data_utils.py        # Data loading and preprocessing
├── data/                     # Data directory (for custom datasets)
├── checkpoints/             # Saved model checkpoints
├── scripts/                 # Additional utility scripts
├── config.py               # Configuration file
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 安装依赖 (Installation)

```bash
pip install -r requirements.txt
```

### 依赖包 (Dependencies)

- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- tqdm >= 4.65.0

## 使用方法 (Usage)

### 1. 训练模型 (Training)

#### 基本训练 (Basic Training)

```bash
python train.py --batch-size 32 --epochs-teacher 50 --epochs-student 50
```

#### 自定义参数训练 (Training with Custom Parameters)

```bash
python train.py \
    --batch-size 64 \
    --epochs-teacher 100 \
    --epochs-student 80 \
    --lr 0.001 \
    --alpha 0.7 \
    --device cuda \
    --n-samples 2000 \
    --save-dir ./checkpoints
```

#### 参数说明 (Parameters)

- `--batch-size`: 批次大小 (default: 32)
- `--epochs-teacher`: Teacher模型训练轮数 (default: 50)
- `--epochs-student`: Student模型训练轮数 (default: 50)
- `--lr`: 学习率 (default: 0.001)
- `--alpha`: 蒸馏损失权重 (default: 0.7)
- `--device`: 训练设备 cpu/cuda (default: cpu)
- `--n-samples`: 合成数据样本数 (default: 1000)
- `--save-dir`: 模型保存目录 (default: ./checkpoints)

### 2. 评估模型 (Evaluation)

#### 评估Teacher模型 (Evaluate Teacher Model)

```bash
python evaluate.py \
    --model-type teacher \
    --model-path ./checkpoints/teacher_model.pth \
    --device cpu
```

#### 评估Student模型 (Evaluate Student Model)

```bash
python evaluate.py \
    --model-type student \
    --model-path ./checkpoints/student_model.pth \
    --device cpu
```

### 3. 使用自定义数据 (Using Custom Data)

如果您有自己的焊接数据，可以修改 `utils/data_utils.py` 中的数据加载函数：

```python
from utils.data_utils import WeldingDataset, prepare_data_loaders
import numpy as np

# 加载您的数据
train_data = np.load('your_train_data.npy')  # Shape: (n_samples, 18)
train_labels = np.load('your_train_labels.npy')  # Shape: (n_samples,)

# 准备数据加载器
student_feature_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 选择13个特征
data_loaders = prepare_data_loaders(
    train_data, train_labels,
    val_data, val_labels,
    student_feature_indices=student_feature_indices
)
```

## 模型架构 (Model Architecture)

### Teacher Model (18维输入)

```
Input (18) → Linear(64) → BatchNorm → ReLU → Dropout(0.2) →
Linear(32) → BatchNorm → ReLU → Dropout(0.2) →
Linear(16) → BatchNorm → ReLU → Dropout(0.2) →
Linear(1) → Output
```

### Student Model (13维输入)

```
Input (13) → Linear(32) → BatchNorm → ReLU → Dropout(0.2) →
Linear(16) → BatchNorm → ReLU → Dropout(0.2) →
Linear(1) → Output
```

## 知识蒸馏损失 (Knowledge Distillation Loss)

项目实现了两种蒸馏损失：

### 1. RegressionDistillationLoss (推荐用于回归任务)

```
L_total = α * L_distill + (1-α) * L_task

其中:
- L_distill = MSE(student_output, teacher_output)
- L_task = MSE(student_output, ground_truth)
- α: 蒸馏损失权重 (default: 0.7)
```

### 2. KnowledgeDistillationLoss (包含KL散度)

```
L_total = α * L_KL + β * L_MSE + (1-α-β) * L_task

其中:
- L_KL = KL_divergence(student || teacher) with temperature
- L_MSE = MSE(student_output, teacher_output)
- L_task = MSE(student_output, ground_truth)
```

## 性能对比 (Performance Comparison)

训练完成后，您可以比较Teacher和Student模型的性能：

| Model | Parameters | Input Dim | MSE | RMSE | R² |
|-------|-----------|-----------|-----|------|-----|
| Teacher | ~3K | 18 | TBD | TBD | TBD |
| Student | ~1K | 13 | TBD | TBD | TBD |

## 引用 (Citation)

如果您在研究中使用了本项目，请引用：

```bibtex
@misc{welding_distillation,
  title={Knowledge Distillation for Welding Quality Prediction},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/233HK233/Autoencoder_for_welding_prediction}
}
```

## 许可证 (License)

MIT License

## 联系方式 (Contact)

如有问题或建议，请提交Issue或Pull Request。