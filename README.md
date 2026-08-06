# 焊缝时序分类与蒸馏基准仓库

本仓库面向焊缝时序数据的三分类诊断任务，覆盖数据集构建、教师模型
训练、知识蒸馏、Ablation 实验、baseline 对比，以及论文图表生成。当前
仓库状态同时包含 `horizon=0` 当前时刻分类流程和 `horizon=1` 未来一步
预测流程，并保留了已提交的实验结果与图表产物。

本文档反映的是当前已提交仓库状态，更新时间为 June 20, 2026。

## 仓库概览

这个仓库已经不只是单一路径的训练脚本集合，而是一个完整实验工作区。
如果你要上手，先把它理解为四层内容：核心训练流程、`horizon=1`
实验扩展、论文图表工具链，以及结果与测试资产。

- 核心训练入口是 `prepare_weld_seam_dataset.py`、
  `train_single_tcn_classifier.py`、
  `train_distill_single_tcn_student.py`，以及
  `infer_with_monotonic_postprocess.py`。
- `horizon=1` 未来一步流程由
  `prepare_weld_seam_dataset_forecast.py`、
  `run_teacher_h1_sweep.py` 和 `ablation_experiments/` 下的扫描脚本驱动。
- baseline 训练入口是 `train_baseline_lstm_classifier.py`，而
  `baseline_LSTM.py` 保留为较早的参考实现。
- 图表与论文素材入口包括
  `visualize_weld_raw_data_for_paper.py`、
  `select_figure1_panels.py`、`render_figure1_matplotlib.py`、
  `render_figure1_png.py`、`render_figure2_matplotlib.py`，以及
  `render_figure3_matplotlib.py`。
- 已提交结果位于 `outputs/`、`ablation_experiments/h1/results/` 和
  `ablation_experiments/h1/reports/`。
- 自动化测试位于 `tests/` 和
  `ablation_experiments/ablation3_joint_tcn_attn/tests/`。

## 目录结构

仓库现在同时承载代码、实验结果和论文素材，所以顶层目录比传统训练仓库更
重。下面这份结构比旧版 README 更接近当前实际用途。

```text
autoencoder_benchmark/
├── Data/
│   ├── raw_data/                    # 原始焊缝 CSV
│   └── processed_data/              # 处理后的 .npz 数据集
├── outputs/                         # 教师、学生、baseline、图表和后处理结果
├── ablation_experiments/
│   ├── scripts/                     # Ablation-1/2、18D baseline、汇总脚本
│   ├── h1/results/                  # horizon=1 实验结果
│   ├── h1/reports/                  # horizon=1 汇总报告
│   └── ablation3_joint_tcn_attn/    # Ablation-3 联合蒸馏实现与测试
├── docs/                            # 过程记录、图表说明和实验计划
├── tests/                           # 主流程、figure、sweep 相关测试
├── useless/                         # 已退役脚本与历史实现
├── prepare_weld_seam_dataset.py
├── prepare_weld_seam_dataset_forecast.py
├── train_single_tcn_classifier.py
├── train_distill_single_tcn_student.py
├── train_baseline_lstm_classifier.py
├── infer_with_monotonic_postprocess.py
├── run_teacher_h1_sweep.py
├── visualize_weld_raw_data_for_paper.py
├── render_figure1_matplotlib.py
├── render_figure2_matplotlib.py
└── render_figure3_matplotlib.py
```

## 环境与依赖

当前代码路径依赖 PyTorch 训练栈、Matplotlib 绘图栈，以及 Pillow /
ReportLab 的图像与 PDF 输出能力。如果你只跑训练，后两者不是必须；如果你
要生成论文图，请一起安装。

建议使用 Python 3.10 或更高版本，然后安装以下依赖：

```bash
python -m pip install --upgrade pip
python -m pip install \
  numpy pandas torch scikit-learn matplotlib pillow reportlab pytest
```

## 主要工作流

仓库当前的主线已经分成两条：一条是经典的教师-学生蒸馏链路，另一条是以
`horizon=1` 为核心的未来一步预测实验链路。下面列出最常用的入口命令。

### 构建当前时刻窗口数据集

如果你要复现 `horizon=0` 的主流程，先从标签安全窗口数据集开始。这个脚本
会按焊缝独立标准化、按标签段切分，并在训练/测试边界上支持 `purge-gap`。

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

### 构建未来一步数据集

如果你要使用当前仓库最活跃的实验路径，请改用
`prepare_weld_seam_dataset_forecast.py`。当前大多数 `horizon=1` 教师、
学生、Ablation 和 baseline 流程都默认指向这个数据集。

```bash
python prepare_weld_seam_dataset_forecast.py \
  --input-dir Data/raw_data \
  --output Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz \
  --window-size 5 \
  --horizon 1 \
  --stride 1 \
  --train-frac 0.75 \
  --purge-gap 0 \
  --seed 42
```

### 训练教师模型

教师训练入口是 `train_single_tcn_classifier.py`。它当前支持 `tcn`、
`tcn_attn` 和 `inception` 三类 backbone，并把公用训练/评估逻辑抽到了
`training_utils.py`。

```bash
python train_single_tcn_classifier.py \
  --dataset-npz Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz \
  --output-dir outputs/single_tcn \
  --model tcn_attn \
  --epochs 80 \
  --batch-size 128 \
  --lr 2e-4 \
  --weight-decay 2e-4 \
  --tcn-kernel 3 \
  --tcn-layers 3 \
  --tcn-dropout 0.12 \
  --latent-dim 64 \
  --weighted-sampler \
  --class-weights auto \
  --checkpoint-metric test_acc \
  --seed 42
```

### 蒸馏学生模型

学生蒸馏入口是 `train_distill_single_tcn_student.py`。它从教师
`run_args.json` 和 `best_single_tcn.pth` 恢复配置，默认执行 18D 到 13D
的特征删减蒸馏。

```bash
python train_distill_single_tcn_student.py \
  --dataset-npz Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz \
  --teacher-ckpt outputs/single_tcn/<teacher-run>/best_single_tcn.pth \
  --teacher-run-args outputs/single_tcn/<teacher-run>/run_args.json \
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
  --seed 77
```

### 执行教师 h1 扫描

如果你要复现当前仓库里的高精度教师搜索，直接使用
`run_teacher_h1_sweep.py`。这个脚本会围绕
`Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz` 组织训练批次，
并把结果写入独立输出目录。

```bash
python run_teacher_h1_sweep.py
```

### 训练 baseline LSTM

如果你要跑当前维护中的 LSTM baseline，请使用
`train_baseline_lstm_classifier.py`。它和旧的 `baseline_LSTM.py` 不同，
前者已经接入了更稳定的训练输出与评估记录。

```bash
python train_baseline_lstm_classifier.py \
  --input-dir Data/raw_data \
  --output-dir outputs/baseline_lstm \
  --train-frac 0.75 \
  --window-size 20 \
  --target-offset 5 \
  --epochs 150 \
  --batch-size 32 \
  --lr 1e-5 \
  --seed 42
```

### 执行 horizon=1 Ablation 套件

当前 `ablation_experiments/` 下最重要的路径是 `horizon=1` 未来一步实验。
如果你要一次性跑完 Ablation-1、Ablation-2 和 Ablation-3 的目标扫描，
可以从下面这个入口开始。

```bash
python ablation_experiments/scripts/run_h1_ablation_target_sweep.py \
  --dataset-npz Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz
```

如果你只想运行单个子套件，可以分别使用：

- `ablation_experiments/scripts/run_ablation1_sweep.py`
- `ablation_experiments/scripts/run_ablation2_sweep.py`
- `ablation_experiments/ablation3_joint_tcn_attn/scripts/run_ablation3_h1_sweep.py`
- `ablation_experiments/scripts/run_comparison_18d_baseline_sweep.py`
- `ablation_experiments/scripts/run_h1_distill_comparison_sweep.py`

## 图表与论文素材

当前仓库已经把论文图表生成链路提交进来了，README 也需要反映这一点。你不
需要再从脚本里手工猜测 Figure 1 到 Figure 3 的来源。

- `visualize_weld_raw_data_for_paper.py` 生成焊缝原始数据的论文展示图。
- `select_figure1_panels.py` 从预测明细中选择 Figure 1 候选样本。
- `render_figure1_matplotlib.py` 和 `render_figure1_png.py` 负责 Figure 1。
- `render_figure2_matplotlib.py` 负责综合性能图 Figure 2。
- `render_figure3_matplotlib.py` 负责 Teacher / Student /
  18D baseline 对比图 Figure 3。
- 常见输出位置是 `outputs/paper_figures/`、`outputs/figure2/` 和
  `outputs/figure 3/`。

## 测试

仓库当前已经包含覆盖主路径的 `pytest` 测试，尤其是 `horizon=1` 默认值、
扫描脚本、figure 渲染和图表汇总逻辑。更新代码后，建议至少跑一组主测试。

运行主测试集：

```bash
pytest tests -q
```

运行 Ablation-3 附加测试：

```bash
pytest ablation_experiments/ablation3_joint_tcn_attn/tests -q
```

如果你只修改了单个模块，优先跑对应测试文件，例如：

```bash
pytest tests/test_prepare_weld_seam_dataset_forecast.py -q
pytest tests/test_run_teacher_h1_sweep.py -q
pytest tests/test_render_figure3_matplotlib.py -q
```

## 当前仓库状态说明

这份 README 现在明确把“仓库是代码仓库”与“仓库也已提交实验资产”这两件事
分开讲清楚了。你在使用前需要知道以下几点。

- `outputs/` 和 `ablation_experiments/h1/results/` 已经包含大量结果文件。
- `docs/` 中保留了实验计划、图表检查记录和阶段性总结。
- `useless/` 中的脚本是历史实现，不是当前推荐入口。
- 当前 README 不再把 `run_tcn_sweep.py` 当作主推荐路径，它更接近兼容保留。

## Next steps

如果你准备继续在这个仓库上工作，建议按下面的顺序进入。

1. 先确认你要走 `horizon=0` 还是 `horizon=1` 路线。
2. 再构建或复用 `Data/processed_data/` 下对应的 `.npz` 数据集。
3. 然后选择教师、蒸馏、baseline 或 Ablation 入口脚本。
4. 最后按需要生成 `outputs/paper_figures/` 下的论文图表。
