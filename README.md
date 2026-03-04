# 焊缝质量诊断与知识蒸馏系统（TCN + LUPI）

本仓库用于焊缝时序数据的三分类诊断（`0/1/2`），并支持从 18 维全量特征教师模型蒸馏到 13 维可部署学生模型。

README 已按当前仓库脚本状态更新（2026-03-04）。

## 1. 项目目标

- 使用 TCN/TCN-Attention 做焊缝状态分类。
- 保证时间序列切分无泄漏（标签边界安全 + 时间分割 + purge gap）。
- 使用 LUPI 蒸馏把 18 维输入压缩到 13 维部署输入。
- 使用单调后处理（`monotonic` / `three_segment`）提升时序预测稳定性。

## 2. 当前代码状态（重要）

### 主流程脚本（可直接使用）

- `prepare_weld_seam_dataset.py`：构建标签安全窗口数据集（`.npz`）。
- `train_single_tcn_classifier.py`：训练教师模型（支持 `tcn` / `tcn_attn` / `inception`）。
- `train_distill_single_tcn_student.py`：冻结教师蒸馏学生（默认删特征 `3,4,5,6,7`）。
- `infer_with_monotonic_postprocess.py`：推理 + 时序后处理。
- `run_distill_sweep.py`：网格扫描 `lambda_kd`、`lambda_feat`。
- `run_distill_seed_sweep.py`：多随机种子扫描并按阈值早停。
- `analyze_single_tcn_results.py`：汇总实验结果，筛选准确率阈值以上 run。

### 兼容性说明

- `run_tcn_sweep.py` 目前依赖 `train_benchmark_classifier_tcn.py`（仓库中不存在），属于旧脚本，默认不可直接跑通。
- `inspect_npz.py` 使用了硬编码路径，使用前需要先修改脚本顶部常量。

## 3. 目录结构

```text
autoencoder_benchmark/
├── Data/
│   ├── raw_data/                       # 原始 CSV（a01,b01,c01,c02）
│   └── processed_data/                 # 处理后的 npz
├── outputs/                            # 训练输出与分析结果
├── prepare_weld_seam_dataset.py
├── train_single_tcn_classifier.py
├── train_distill_single_tcn_student.py
├── infer_with_monotonic_postprocess.py
├── run_distill_sweep.py
├── run_distill_seed_sweep.py
└── analyze_single_tcn_results.py
```

## 4. 环境依赖

建议 Python 3.10+。

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas torch scikit-learn matplotlib
```

## 5. 快速开始（从数据到蒸馏）

以下命令假设你当前目录就是 `autoencoder_benchmark/`。

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

### 5.2 训练教师模型（18D）

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

训练完成后自动生成 run 目录并保存：

- `best_single_tcn.pth`
- `run_args.json`
- `history.json`
- `evaluation_metrics.txt`

### 5.3 蒸馏学生模型（13D）

先取最新教师 run 目录：

```bash
TEACHER_DIR=$(ls -td outputs/single_tcn/single_tcn_attn_* | head -n 1)
echo "$TEACHER_DIR"
```

再启动蒸馏：

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

注意：`infer_with_monotonic_postprocess.py` 当前读取 `X_test_full`（18D），请传入与之匹配的模型 checkpoint（通常是教师模型）。

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

## 6. 已记录结果（来自 `outputs/`）

### 6.1 教师模型（最佳测试准确率）

- Run: `single_tcn_attn_weld_seam_windows_ws5_tf75_pg0_ep30_lr0.00025_bs128_k3_l3_d0.12_lat64_wd0.0002_seed230`
- 来源: `outputs/single_tcn/best_record/.../seed230/`
- Test Accuracy: `98.64%`
- Test Macro-F1: `0.9828`
- Best Epoch: `2`

### 6.2 学生模型（蒸馏）

- 官方 best_record 摘要（`BEST_RECORD_SUMMARY.txt`）：
  - Run: `...seed77`
  - Test Accuracy: `95.92%`
  - Test Teacher Agreement: `96.20%`
  - Test Macro-F1: `0.9474`
- 已记录最高蒸馏测试准确率：
  - Run: `...seed132`
  - Test Accuracy: `96.74%`
  - Test Teacher Agreement: `97.01%`
  - Test Macro-F1: `0.9552`

### 6.3 后处理效果（示例）

`outputs/postprocess_reports/seed14_postprocess_report.txt`：

- Raw Accuracy: `96.47%`
- Monotonic Accuracy: `98.64%`
- Three-segment Accuracy: `98.64%`

## 7. 批量实验与分析

### 7.1 蒸馏超参扫描

```bash
python run_distill_sweep.py \
  --dataset-npz Data/processed_data/weld_seam_windows_ws5_tf75_pg0.npz \
  --teacher-ckpt "${TEACHER_DIR}/best_single_tcn.pth" \
  --teacher-run-args "${TEACHER_DIR}/run_args.json" \
  --output-dir outputs/distill_single_tcn/sweep_round \
  --max-runs 16 \
  --stop-val-agreement 0.98
```

### 7.2 蒸馏多随机种子扫描

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

### 7.3 实验结果筛选与汇总

```bash
python analyze_single_tcn_results.py \
  --root-dir outputs/single_tcn \
  --root-dir outputs/distill_single_tcn \
  --threshold 95 \
  --out-csv outputs/analysis_test_acc_ge95.csv \
  --out-json outputs/analysis_test_acc_ge95.json \
  --top-k 10
```

## 8. 标签定义

- `0`：`quasistable`
- `1`：`nonstationary`
- `2`：`instability`

## 9. 数据格式（`prepare_weld_seam_dataset.py` 输出）

| Key | 说明 |
|---|---|
| `X_train_full` | 训练特征，形状 `[N_train, T, C]` |
| `y_train` | 训练标签，形状 `[N_train]` |
| `X_test_full` | 测试特征，形状 `[N_test, T, C]` |
| `y_test` | 测试标签，形状 `[N_test]` |
| `seam_id_train` | 训练样本焊缝 ID |
| `seam_id_test` | 测试样本焊缝 ID |
| `start_idx_train` | 训练窗口起始时间索引 |
| `start_idx_test` | 测试窗口起始时间索引 |
| `seam_name_order` | 焊缝名顺序（ID 到文件名映射） |
| `scaler_mean_<seam>` | 对应焊缝标准化均值 |
| `scaler_scale_<seam>` | 对应焊缝标准化方差尺度 |

当前常用数据集 `weld_seam_windows_ws5_tf75_pg0.npz` 的统计：

- `X_train_full`: `(1177, 5, 18)`
- `X_test_full`: `(368, 5, 18)`
- 训练标签分布: `{0: 268, 1: 285, 2: 624}`
- 测试标签分布: `{0: 82, 1: 87, 2: 199}`

## 10. 常见问题

- 推理时 `load_state_dict` 维度不匹配：
  - 原因通常是 `infer_with_monotonic_postprocess.py` 的模型超参和 checkpoint 训练超参不一致。
  - 解决：以对应 run 的 `run_args.json` 为准，显式传入 `--tcn-layers`、`--tcn-dropout`、`--classifier-hidden` 等参数。
- 运行 `run_tcn_sweep.py` 报找不到训练脚本：
  - 这是旧流程遗留，当前主线请用 `train_single_tcn_classifier.py` + `run_distill_sweep.py` / `run_distill_seed_sweep.py`。
