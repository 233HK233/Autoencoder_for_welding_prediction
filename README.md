# 焊缝质量诊断与知识蒸馏系统（TCN + LUPI）

本仓库用于焊缝时序数据的三分类诊断（`0/1/2`），并支持从 18 维全量特征教师模型蒸馏到 13 维可部署学生模型。

README 已按当前仓库脚本状态更新（2026-03-05）。

## 1. 项目目标

- 使用 TCN/TCN-Attention 做焊缝状态分类。
- 保证时间序列切分无泄漏（标签边界安全 + 时间分割 + purge gap）。
- 使用 LUPI 蒸馏把 18 维输入压缩到 13 维部署输入。
- 使用单调后处理（`monotonic` / `three_segment`）提升时序预测稳定性。

## 2. 目录结构与核心模块

```text
autoencoder_benchmark/
├── 近期工作汇报讲稿_2026-02.md           # 2026年2月工作汇报文档
├── Data/                                 # 数据集目录
│   ├── raw_data/                         # 原始 CSV（a01,b01,c01,c02）
│   ├── processed_data/                   # 处理后的 npz 和分析数据
│   └── case1/, case2/, Datawith(no)PHOENIX/ # 其他原始与实验批次数据
├── outputs/                              # 模型训练输出、预测结果、分析日志
├── useless/                              # 废弃/旧版/对比实验代码（如 train_benchmark_classifier_tcn.py）
├── 核心主流程脚本:
│   ├── prepare_weld_seam_dataset.py      # 数据预处理，构建标签安全窗口数据集（.npz）
│   ├── train_single_tcn_classifier.py    # 训练教师分类模型（支持 tcn/tcn_attn/inception）
│   ├── train_distill_single_tcn_student.py # 冻结教师模型蒸馏训练学生模型（删减特征）
│   └── infer_with_monotonic_postprocess.py # 测试集推理 + 时序后处理（单调/三段）
├── 批量实验与分析扫描:
│   ├── run_distill_sweep.py              # 网格扫描 lambda_kd、lambda_feat 等超参数
│   ├── run_distill_seed_sweep.py         # 多随机种子扫描并按阈值早停
│   ├── run_tcn_sweep.py                  # TCN 扫描（旧版遗留流程，依赖废弃脚本）
│   └── analyze_single_tcn_results.py     # 汇总实验结果，筛选高准确率 run
├── 训练与数据基础设施模块:
│   ├── models_tcn.py                     # 模型网络结构定义（TCN / TCN-Attention / Inception）
│   ├── training_utils.py                 # 公共训练与评估工具
│   ├── data_utils.py                     # 数据加载与处理工具库
│   ├── framework.py                      # 知识蒸馏教师-学生框架
│   └── inspect_npz.py                    # NPZ 数据分析调试小工具
```

## 3. 当前代码状态与兼容性

### 主流程脚本（推荐路径）
直接按照 `数据处理 -> 教师训练 -> 学生蒸馏 -> 推理后处理` 的顺序执行：
- `prepare_weld_seam_dataset.py`
- `train_single_tcn_classifier.py`
- `train_distill_single_tcn_student.py`
- `infer_with_monotonic_postprocess.py`

### 本次代码改动同步（README 已覆盖）
- 新增 `training_utils.py`，将分类报告、评估、类别权重、分层划分等逻辑抽离为公共模块。
- `train_single_tcn_classifier.py` 与 `train_distill_single_tcn_student.py` 已改为复用 `training_utils.py`。
- `models_tcn.py` 统一维护 TCN / TCN-Attention / Inception 相关模型结构。

### 旧版与兼容性提示
- `run_tcn_sweep.py` 当前硬编码调用 `train_benchmark_classifier_tcn.py`（默认在项目根目录查找）。该文件已迁移到 `useless/`，因此默认不可直接跑通；如需继续使用，请先修正脚本中的调用路径。
- `inspect_npz.py` 使用了硬编码路径，使用前需要先修改脚本顶部常量。

## 4. 环境依赖

建议 Python 3.10+。

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas torch scikit-learn matplotlib
```

## 5. 快速开始（从数据到蒸馏）

以下命令假设你当前目录就是项目根目录 `autoencoder_benchmark/`。

### 5.1 数据预处理

```bash
python prepare_weld_seam_dataset.py \
  --input-dir Data/raw_data \
  --output Data/processed_data/weld_seam_windows_ws5_tf75_pg0.npz \
  --window-size 5 \
  --stride 1 \
  --train-frac 0.75 \
  --purge-gap 0 \
  --seed 42
```

### 5.2 训练教师模型（18D 全特征）

```bash
python train_single_tcn_classifier.py \
  --dataset-npz Data/processed_data/weld_seam_windows_ws5_tf75_pg0.npz \
  --output-dir outputs/single_tcn \
  --model tcn_attn \
  --epochs 30 \
  --batch-size 128 \
  --lr 2.5e-4 \
  --weight-decay 2e-4 \
  --tcn-kernel 3 \
  --tcn-layers 3 \
  --tcn-channels 80,80,80 \
  --tcn-dropout 0.12 \
  --classifier-hidden 128 \
  --classifier-dropout 0.35 \
  --weighted-sampler \
  --class-weights auto \
  --checkpoint-metric test_acc \
  --early-stop-patience 8 \
  --min-epochs 8 \
  --seed 230
```

可选：若需要做“特征删减敏感性实验”，可在教师训练中直接追加参数：

```bash
--drop-feature-indices 3,4,5,6,7
```

训练完成后自动生成 run 目录并保存：
- `best_single_tcn.pth`
- `run_args.json`
- `history.json`
- `evaluation_metrics.txt`

### 5.3 蒸馏学生模型（13D 部署特征）

先获取最新教师 run 目录：

```bash
TEACHER_DIR=$(ls -td outputs/single_tcn/single_tcn_attn_* | head -n 1)
echo "使用教师模型: $TEACHER_DIR"
```

启动蒸馏（丢弃无用的特征）：

```bash
python train_distill_single_tcn_student.py \
  --dataset-npz Data/processed_data/weld_seam_windows_ws5_tf75_pg0.npz \
  --teacher-ckpt "${TEACHER_DIR}/best_single_tcn.pth" \
  --teacher-run-args "${TEACHER_DIR}/run_args.json" \
  --output-dir outputs/distill_single_tcn \
  --drop-feature-indices 3,4,5,6,7 \
  --epochs 80 \
  --batch-size 128 \
  --lr 2e-4 \
  --weight-decay 2e-4 \
  --temperature 3.0 \
  --lambda-ce 0.8 \
  --lambda-kd 1.2 \
  --lambda-feat 0.2 \
  --weighted-sampler \
  --checkpoint-metric val_teacher_agreement \
  --early-stop-patience 16 \
  --min-epochs 16 \
  --seed 77
```

### 5.4 推理与单调后处理

注意：`infer_with_monotonic_postprocess.py` 当前默认读取 `X_test_full`（18D），请传入与之匹配的模型 checkpoint（通常是教师模型）。如果推理学生模型，需要修改脚本适配特征裁剪后的输入。

```bash
python infer_with_monotonic_postprocess.py \
  --dataset-npz Data/processed_data/weld_seam_windows_ws5_tf75_pg0.npz \
  --checkpoints "${TEACHER_DIR}/best_single_tcn.pth" \
  --channels 64 \
  --tcn-layers 3 \
  --tcn-kernel 3 \
  --tcn-dropout 0.12 \
  --classifier-hidden 128 \
  --classifier-dropout 0.35 \
  --decode both
```

## 6. 批量实验与分析辅助

### 6.1 蒸馏超参网格扫描
```bash
python run_distill_sweep.py \
  --dataset-npz Data/processed_data/weld_seam_windows_ws5_tf75_pg0.npz \
  --teacher-ckpt "${TEACHER_DIR}/best_single_tcn.pth" \
  --teacher-run-args "${TEACHER_DIR}/run_args.json" \
  --output-dir outputs/distill_single_tcn/sweep_round \
  --max-runs 16 \
  --stop-val-agreement 0.98
```

### 6.2 蒸馏随机种子稳定性测试
```bash
python run_distill_seed_sweep.py \
  --dataset-npz Data/processed_data/weld_seam_windows_ws5_tf75_pg0.npz \
  --teacher-ckpt "${TEACHER_DIR}/best_single_tcn.pth" \
  --teacher-run-args "${TEACHER_DIR}/run_args.json" \
  --output-dir outputs/distill_single_tcn/seed_sweep_custom \
  --seed-start 100 \
  --seed-end 199 \
  --max-runs 100 \
  --target-acc 95 \
  --target-count 10 \
  --epochs 80 \
  --weighted-sampler
```

### 6.3 实验结果筛选报告汇总
找出在评估中表现最佳的一批模型配置：
```bash
python analyze_single_tcn_results.py \
  --root-dir outputs/single_tcn \
  --root-dir outputs/distill_single_tcn \
  --threshold 95 \
  --out-csv outputs/analysis_test_acc_ge95.csv \
  --out-json outputs/analysis_test_acc_ge95.json \
  --top-k 10
```

## 7. 标签定义与数据说明

| 标签 | 含义 (Weld Seam Status) |
|---|---|
| `0` | `quasistable`（准稳态） |
| `1` | `nonstationary`（非平稳） |
| `2` | `instability`（不稳定） |

### 窗口数据集 (`prepare_weld_seam_dataset.py` 输出)
| Key | 说明 |
|---|---|
| `X_train_full` / `X_test_full` | 特征张量，形状 `[N, T, C]` |
| `y_train` / `y_test` | 标签，形状 `[N]` |
| `seam_id_train` / `seam_id_test`| 对应样本来源的焊缝 ID |
| `start_idx_train` / `start_idx_test` | 窗口起点的全局时间索引 |
| `seam_name_order` | 焊缝 ID 映射 (0 -> a01, 1 -> b01 ...) |
| `scaler_mean_*` / `scaler_scale_*` | 用于该焊缝数据的标准化参考均值与尺度 |

## 8. 常见问题排查 (Troubleshooting)

1. **推理时 `load_state_dict` 维度不匹配：**
   - **原因**：`infer_with_monotonic_postprocess.py` 输入的模型超参与保存的 checkpoint 超参不一致。
   - **解决**：打开目标 run 目录的 `run_args.json`，根据配置显式补齐 `--tcn-layers`, `--tcn-dropout`, `--classifier-hidden` 等参数。
2. **`run_tcn_sweep.py` 报错找不到文件：**
   - **原因**：依赖文件已被移至 `useless/`。
   - **解决**：如果需要继续使用它，请修改里面针对 `train_benchmark_classifier_tcn.py` 的子进程调用路径；但当前推荐直接扫 `train_single_tcn_classifier.py`。
