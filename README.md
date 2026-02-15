# 焊缝质量分类与知识蒸馏系统

基于 **时序卷积网络 (TCN)** 的焊缝质量三分类系统，支持 **LUPI (Learning Using Privileged Information)** 知识蒸馏和 **单调时序后处理** 推理。

## 目录

- [项目概述](#项目概述)
- [项目结构](#项目结构)
- [环境依赖](#环境依赖)
- [数据说明](#数据说明)
- [使用流程](#使用流程)
  - [1. 数据预处理](#1-数据预处理)
  - [2. 训练教师模型](#2-训练教师模型)
  - [3. 知识蒸馏训练学生模型](#3-知识蒸馏训练学生模型)
  - [4. 推理与后处理](#4-推理与后处理)
  - [5. 超参数搜索](#5-超参数搜索)
- [模型架构](#模型架构)
- [标签定义](#标签定义)

---

## 项目概述

本项目旨在对焊接过程中的焊缝状态进行实时分类，将焊缝质量划分为三种状态：**准稳态 (quasistable)**、**非稳态 (nonstationary)** 和 **不稳定 (instability)**。

核心思路是利用 **LUPI 知识蒸馏** 框架：
- **教师模型**：使用完整的 **18 维传感器特征**（包含部分仅在离线/实验环境下可用的特权特征）进行训练
- **学生模型**：仅使用 **13 维可部署特征**（去除特权特征），通过蒸馏学习教师的知识

这样，学生模型在实际部署时只需要可实时获取的传感器信号，同时继承了教师模型利用特权信息学到的判别能力。

---

## 项目结构

```
autoencoder_benchmark/
├── Data/                           # 数据目录
│   ├── raw_data/                   # 原始焊缝 CSV 数据 (a01.csv, b01.csv, ...)
│   └── processed_data/             # 预处理后的 .npz 数据集
├── outputs/                        # 训练输出 (模型权重、日志、指标)
│
├── data_utils.py                   # 数据加载与标准化工具函数
├── prepare_weld_seam_dataset.py    # 数据预处理：标签安全窗口化 + 覆盖约束采样
├── models_tcn.py                   # TCN 编码器 + 分类器模型定义 (Teacher/Student)
├── framework.py                    # Teacher-Student 蒸馏训练框架
│
├── train_single_tcn_classifier.py  # 单模型 TCN 分类器训练 (含 Attention 和 InceptionTime 变体)
├── train_distill_single_tcn_student.py  # LUPI 知识蒸馏：从教师蒸馏学生
│
├── run_tcn_sweep.py                # TCN 超参数网格搜索
├── run_distill_sweep.py            # 蒸馏超参数网格搜索
│
├── infer_with_monotonic_postprocess.py  # 推理 + 单调时序解码后处理
└── inspect_npz.py                  # 数据集检查与可视化工具
```

---

## 环境依赖

- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy
- Pandas
- scikit-learn（可选，回落到内置 `StandardScaler`）
- matplotlib

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

---

## 数据说明

### 原始数据格式

原始数据存放在 `Data/raw_data/` 下，每条焊缝对应一个 CSV 文件（如 `a01.csv`、`b01.csv`、`c01.csv`、`c02.csv`）。

每个 CSV 文件包含 **19 列**：
- **前 18 列**：传感器特征（浮点数）
- **第 19 列**：焊缝状态标签（字符串）

### 预处理后数据格式

预处理后以 `.npz` 格式保存，主要包含：

| 键名 | 形状 | 说明 |
|------|------|------|
| `X_train_full` | `[N_train, T, 18]` | 训练集全特征窗口 |
| `y_train` | `[N_train]` | 训练集标签 |
| `seam_id_train` | `[N_train]` | 训练集焊缝 ID |
| `start_idx_train` | `[N_train]` | 训练集窗口起始索引 |
| `X_test_full` | `[N_test, T, 18]` | 测试集全特征窗口 |
| `y_test` | `[N_test]` | 测试集标签 |
| `seam_id_test` | `[N_test]` | 测试集焊缝 ID |
| `start_idx_test` | `[N_test]` | 测试集窗口起始索引 |
| `seam_name_order` | `[n_seams]` | 焊缝名称到 ID 的映射顺序 |

---

## 使用流程

### 1. 数据预处理

将原始 CSV 转换为标签安全的滑动窗口数据集。核心特性：

- **标签安全窗口化**：窗口不跨越标签边界
- **时间切分 + Purge Gap**：在 train/test 切分边界留出间隔，防止时间泄漏
- **覆盖约束采样**：确保每条焊缝的每种标签在 train/test 中都有代表

```bash
python prepare_weld_seam_dataset.py \
    --data-dir Data/raw_data \
    --output-dir Data/processed_data \
    --window-size 5 \
    --stride 1 \
    --train-frac 0.75 \
    --purge-gap 0
```

### 2. 训练教师模型

使用全部 18 维特征训练 TCN 分类器。支持两种架构变体：

- **AttentionTCNClassifier**：TCN + 多头自注意力（默认）
- **InceptionTimeClassifier**：InceptionTime 架构

```bash
python train_single_tcn_classifier.py \
    --dataset-npz Data/processed_data/weld_seam_windows.npz \
    --output-dir outputs/teacher_run \
    --epochs 50 \
    --lr 1e-4 \
    --batch-size 96 \
    --tcn-kernel 3 \
    --tcn-layers 3 \
    --class-weights auto \
    --seed 42
```

训练完成后，`outputs/teacher_run/` 目录下会生成：
- `best_single_tcn.pth` — 最佳模型权重
- `run_args.json` — 运行参数记录
- `evaluation_metrics.txt` — 评估指标报告
- `training_curves.png` — 训练曲线图

### 3. 知识蒸馏训练学生模型

从冻结的教师模型蒸馏出使用 13 维特征的学生模型（去除第 3–7 列特权特征）。

蒸馏损失包含三部分：
- **CE Loss**：学生自身的交叉熵分类损失
- **KD Loss**：学生与教师的 KL 散度（软标签蒸馏）
- **Feature Loss**：学生与教师隐空间的 MSE 对齐损失

```bash
python train_distill_single_tcn_student.py \
    --dataset-npz Data/processed_data/weld_seam_windows.npz \
    --teacher-ckpt outputs/teacher_run/best_single_tcn.pth \
    --teacher-run-args outputs/teacher_run/run_args.json \
    --output-dir outputs/distill_run \
    --drop-feature-indices 3,4,5,6,7 \
    --epochs 60 \
    --lr 2e-4 \
    --temperature 2.0 \
    --lambda-ce 1.0 \
    --lambda-kd 0.5 \
    --lambda-feat 0.2 \
    --checkpoint-metric val_teacher_agreement
```

### 4. 推理与后处理

支持多模型集成推理，并提供两种时序后处理策略：

- **单调解码 (monotonic)**：利用动态规划强制类别序列单调递增（0→1→2）
- **三段解码 (three_segment)**：最优化将序列划分为连续的三个阶段

```bash
python infer_with_monotonic_postprocess.py \
    --dataset-npz Data/processed_data/weld_seam_windows.npz \
    --checkpoints outputs/teacher_run/best_single_tcn.pth \
    --decode both
```

### 5. 超参数搜索

#### TCN 模型超参搜索

在 `lambda_align` 和 `lambda_kl` 的网格上搜索最优超参：

```bash
python run_tcn_sweep.py \
    --dataset-npz Data/processed_data/weld_seam_windows.npz \
    --output-dir outputs/tcn_sweep \
    --stop-acc 0.92 \
    --max-runs 50
```

#### 蒸馏超参搜索

在 `lambda_kd` 和 `lambda_feat` 的网格上搜索：

```bash
python run_distill_sweep.py \
    --dataset-npz Data/processed_data/weld_seam_windows.npz \
    --teacher-ckpt outputs/teacher_run/best_single_tcn.pth \
    --teacher-run-args outputs/teacher_run/run_args.json \
    --output-dir outputs/distill_sweep \
    --stop-val-agreement 0.98 \
    --max-runs 16
```

搜索完成后会生成 `sweep_summary.json` 和 `sweep_summary.csv` 汇总报告。

---

## 模型架构

### TCN 编码器

```
输入 [B, T, C] → 1×1 Conv 投影 → N × TCNResidualBlock (空洞卷积) → 全局平均池化 → 潜在表征 z
```

- 使用 **残差连接** 稳定训练
- **指数增长的空洞率** (`dilation_base^i`) 逐层扩大感受野
- **对称 padding** 保持时间维度长度不变

### 分类器

- **TCN + Attention**：TCN 编码后接多头自注意力层，使用注意力加权和均值池化的拼接作为分类特征
- **InceptionTime**：多尺度 1D 卷积并行提取特征
- **MLP 分类头**：两层全连接 + BatchNorm + Dropout

### 知识蒸馏架构

```
教师 (18D 全特征) ──冻结──→ 软标签 + 隐空间表征
                                ↓
学生 (13D 可部署特征) ──────→ CE + KD + Feature 三项损失联合优化
```

---

## 标签定义

| 标签值 | 名称 | 含义 |
|--------|------|------|
| 0 | `quasistable` | 准稳态（正常） |
| 1 | `nonstationary` | 非稳态（过渡/预警） |
| 2 | `instability` | 不稳定（故障） |

---

## 辅助工具

### 数据集检查

使用 `inspect_npz.py` 检查预处理后数据集的形状、分布和样本预览：

```bash
python inspect_npz.py
```

会输出数据集形状、各焊缝各标签的样本分布交叉表，并生成 CSV 预览文件。
