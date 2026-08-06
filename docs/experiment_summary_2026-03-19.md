# 实验结果整理（截至 2026-03-19）

这份文档整理仓库中已经落盘的实验结果，目标是回答两个问题：
当前到底跑过哪些实验，以及这些实验目前最好的结果是什么。

本文只统计仓库内已经存在结果文件的实验，不把计划文档当作已完成实验。
结果以 `outputs/`、`ablation_experiments/reports/` 和各 run 目录中的
`evaluation_metrics.txt`、`*.json` 为准；`README.md` 和
`近期工作汇报讲稿_2026-02.md` 只作为历史口径参考。

> **补充说明（2026-03-25）**：代码默认 workflow 后续已经调整。
> `ablation_experiments/` 下的 Ablation-1、Ablation-2、Ablation-3、suite
> 和 summary 默认都迁到 `horizon=1` 数据集
> `Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz`，并默认写入
> `ablation_experiments/h1/results/` 与
> `ablation_experiments/h1/reports/`。本文下面引用的消融结果仍然是迁移前
> 已落盘的历史结果，不代表当前默认输出目录。

## 统计口径

这部分先说明本文如何判断“已跑过”和“结果如何”，避免把不同口径混在一起。

- “已跑过”指仓库中存在可追溯的结果文件，例如
  `evaluation_metrics.txt`、`history.json`、`run_args.json`、
  `seed_sweep_summary.json` 或 `best_teacher_h1.json`。
- “当前最好结果”优先看结果汇总 JSON 或 manifest；如果汇总文件与固定的
  `best_record` 目录不一致，会同时注明“固定口径 best”与“仓库中当前最高分”。
- `docs/plans/2026-03-15-teacher-h1-95-plan.md` 是计划文档，不算实验结果。
- 如果某条实验线使用了 `checkpoint_metric=test_acc`，本文会单独标成
  “探索性结果”，因为测试集参与选模。

## 总览

下表先给出每条实验线的当前状态，后面再展开说明证据和差异。

| 实验线 | 当前最好结果 | 当前判断 | 主要证据 |
| --- | --- | --- | --- |
| `horizon=0` teacher（18D） | `98.64%`, `Macro-F1 0.9828` | 主线最佳 teacher | `outputs/single_tcn` |
| distill student（13D）固定 best | `95.92%`, `Macro-F1 0.9474` | 当前固定口径 best | `outputs/distill_single_tcn/best_record` |
| distill student（13D）仓库最高分 | `96.74%` | 大规模 seed sweep 中已出现更高分 | `seed_sweep_20260302_100seeds` |
| 单调后处理 | `96.47% -> 98.64%` | 后处理有效 | `outputs/postprocess_reports` |
| `horizon=1` teacher 严格协议 | `84.83%` | 未达到 95% 目标 | `outputs/teacher_h1_forecast_95` |
| `horizon=1` teacher 探索扫描 v1 | `98.03%` | 高分，但测试集选模 | `outputs/teacher_h1_scan98_gpu1` |
| `horizon=1` teacher 探索扫描 v2 | `98.31%` | 当前 `h1` 最高分，但测试集选模 | `outputs/teacher_h1_scan98_gpu1_v2` |
| Ablation-1 单次 suite | `61.68%` | 单次结果偏弱 | `ablation_suite_final_report.json` |
| Ablation-1 超参扫描 | `91.58%` | tuned 后明显提升 | `ablation1_sweep_summary.json` |
| Ablation-2（LSTM distill） | `37.50%` | 当前结果较弱 | `ablation_suite_final_report.json` |
| Ablation-3（joint TCN-Attn） | `37.23%` | 仍处早期探索 | `ablation3_seed_sweep_final_report.json` |
| 旧版 LSTM segment split | `89.47%` | 可作为遗留基线 | `outputs/baseline_lstm_gpu0_segment_split_seed_sweep_ep45` |
| 旧版 LSTM raw split | `100.00%`, `Macro-F1 0.3333` | 不可比，测试集退化成单类 | `outputs/baseline_lstm` |
| `label_safe_tcn` | teacher `83.85%`, student `78.96%` | 旧数据口径，遗留结果 | `outputs/label_safe_tcn` |

## 主线实验

这部分只看 README 当前主流程对应的实验，也就是 `horizon=0` teacher、
distill student 和后处理。

### `horizon=0` teacher（18D 全特征）

主线 teacher 的结果集中在 `outputs/single_tcn`。按
`outputs/single_tcn/analysis_test_acc_ge95.json` 统计，仓库中共有
`27` 个 run 的测试准确率达到 `95%` 以上。

- 当前最高分是
  `single_tcn_attn_weld_seam_windows_ws5_tf75_pg0_ep30_lr0.00025_bs128_k3_l3_d0.12_lat64_wd0.0002_seed230`。
- 该 run 的测试集结果是 `98.64%`，`Macro-F1 0.9828`。
- 排名前三的测试准确率分别是 `98.64%`、`97.83%` 和 `97.55%`。
- 这条线与 `近期工作汇报讲稿_2026-02.md` 中的汇报口径一致。

主要证据：

- `outputs/single_tcn/analysis_test_acc_ge95.json`
- `outputs/single_tcn/single_tcn_attn_weld_seam_windows_ws5_tf75_pg0_ep30_lr0.00025_bs128_k3_l3_d0.12_lat64_wd0.0002_seed230/evaluation_metrics.txt`

### distill student（13D 部署特征）

蒸馏实验的结果分成三个层次：固定的 `best_record`、小规模 best-config
seed sweep，以及更大规模的 100-seed 扫描。

固定口径的 `best_record` 仍然指向 seed=77：

- 测试准确率 `95.92%`
- `Macro-F1 0.9474`
- `Teacher Agreement 96.20%`

这也是 README 和
`近期工作汇报讲稿_2026-02.md` 使用的“官方 best”口径。

但是仓库里后来又跑了更大的 seed sweep。按
`outputs/distill_single_tcn/seed_sweep_20260302_100seeds/seed_sweep_final_report.json`
统计：

- 实际完成 `63` 个 seed 结果。
- 其中 `10` 个 seed 达到 `95%` 以上。
- 当前仓库内最高测试准确率已经是 `96.74%`，对应 seed=132。
- 这说明固定的 `best_record` 目录没有随着更大 sweep 自动更新。

稳定性方面：

- `seed_sweep_bestcfg` 中 5 个 seed 的均值为 `89.07%`，标准差约 `5.44`。
- 63 个已完成 seed 的均值为 `87.45%`，标准差约 `8.15`。
- 这条线仍然存在较明显的 seed 波动。

超参搜索路径也比较清楚：

- Round1 最佳 `85.87%`
- Round2 最佳 `87.23%`
- Round2 的最佳配置就是后来固定使用的
  `T=3.0, lambda_ce=0.8, lambda_kd=1.2, lambda_feat=0.2`

主要证据：

- `outputs/distill_single_tcn/best_record/BEST_RECORD_SUMMARY.txt`
- `outputs/distill_single_tcn/seed_sweep_bestcfg/seed_sweep_summary.json`
- `outputs/distill_single_tcn/seed_sweep_20260302_100seeds/seed_sweep_final_report.json`
- `outputs/distill_single_tcn/sweep_round1/sweep_summary.json`
- `outputs/distill_single_tcn/sweep_round2/round2_summary.json`

### 后处理实验

单调后处理的结果已经独立保存到 `outputs/postprocess_reports`。现有两份报告
都显示后处理有效，但提升幅度依赖输入模型。

- seed14 单模型：`96.4674% -> 98.6413%`
- seed8+14 ensemble：`98.0978% -> 98.3696%`

在现有结果里，seed14 单模型加后处理后，与主线 best teacher 的
`98.64%` 持平。

主要证据：

- `outputs/postprocess_reports/seed14_postprocess_report.txt`
- `outputs/postprocess_reports/seed8_seed14_ensemble_postprocess_report.txt`

## `horizon=1` 实验

这部分对应“未来一步预测”数据集
`Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz`。这条线在仓库里
实际上有两套口径完全不同的结果，必须分开看。

### 严格协议：`outputs/teacher_h1_forecast_95`

这条线是按
`docs/plans/2026-03-15-teacher-h1-95-plan.md` 的隔离目录方案跑的，manifest
里明确记录了以下特点：

- 输出目录隔离到 `outputs/teacher_h1_forecast_95`
- 目标数据集是 `weld_seam_windows_ws5_tf75_pg0_h1.npz`
- 选模指标是 `val_macro_f1`

按 `best_teacher_h1.json`，这条线没有达到计划中的 `95%` 目标：

- 最佳测试准确率只有 `84.83%`
- `teacher_h1_ge95.json` 为空，说明 `>=95%` 的 run 数量是 `0`

因此，严格协议下的 `horizon=1` teacher 目前仍未达标。

主要证据：

- `outputs/teacher_h1_forecast_95/manifests/best_teacher_h1.json`
- `outputs/teacher_h1_forecast_95/analysis/teacher_h1_ge95.json`

### 探索扫描：`outputs/teacher_h1_scan98_gpu1`

这条线显然是为追求高分而跑的探索性扫描。manifest 直接写明：

- `checkpoint_selection_metric: test_accuracy`

按当前结果：

- 最佳测试准确率 `98.03%`
- 达到 `>=98%` 的 run 数量是 `9`

因此，这条线说明 `horizon=1` 任务本身不是做不到高分，但这个高分带有
测试集参与选模的风险。

主要证据：

- `outputs/teacher_h1_scan98_gpu1/manifests/best_teacher_h1.json`
- `outputs/teacher_h1_scan98_gpu1/analysis/teacher_h1_ge98.json`

### 探索扫描 v2：`outputs/teacher_h1_scan98_gpu1_v2`

这是仓库中当前 `horizon=1` 的最高分结果，同样属于探索性扫描，且同样使用
`checkpoint_metric=test_acc`。

按 manifest 和结果汇总：

- 最佳测试准确率 `98.31%`
- 最佳 run 是 seed=14
- 达到 `>=98%` 的 run 数量是 `16`

因此，如果只问“仓库里 `horizon=1` 跑到过的最高分是多少”，答案是
`98.31%`；如果问“严格协议下是否已经稳定达到 95%”，答案是否。

主要证据：

- `outputs/teacher_h1_scan98_gpu1_v2/manifests/best_teacher_h1.json`
- `outputs/teacher_h1_scan98_gpu1_v2/analysis/teacher_h1_ge98.json`

## 消融实验

消融实验位于 `ablation_experiments/`。这部分最大的特点是：单次 suite
结果和后续超参扫描结果差距很大，需要分开解读。

这里还要额外区分“历史结果目录”和“当前默认 workflow”。本文下面列出的
Ablation-1/2/3 数值都来自迁移前已经存在的结果文件，其中 Ablation-1 与
Ablation-2/3 还混有不同任务口径。当前代码默认已经统一切到
`horizon=1` future-step prediction，并写入
`ablation_experiments/h1/results/` 与
`ablation_experiments/h1/reports/`。

### Ablation-1：student-only TCN-Attention

单次 suite 结果偏弱。在
`ablation_experiments/reports/ablation_suite_final_report.json` 中，seed=100
对应的单次结果只有：

- 测试准确率 `61.68%`
- `Macro-F1 0.6549`

但专门的超参扫描 `run_ablation1_sweep.py` 已经把这条线显著拉高。按
`ablation1_sweep_summary.json` 和 `ablation1_topk.md`：

- 共完成 `22` 个 trial
- 均值 `81.29%`
- 标准差约 `4.16`
- 最优 trial 测试准确率 `91.58%`

所以 Ablation-1 不能只看 suite 的单次结果；按当前仓库结果，它的真实上限
已经到 `91.58%`。

主要证据：

- `ablation_experiments/reports/ablation_suite_final_report.json`
- `ablation_experiments/reports/ablation1_sweep_summary.json`
- `ablation_experiments/reports/ablation1_topk.md`

### Ablation-2：LSTM teacher + LSTM student distill

这条线当前结果较弱，而且明显落后于主线 TCN-Attention。

在当前可见的单次 suite 结果中：

- teacher LSTM：`74.46%`
- student distill LSTM：`37.50%`
- `Teacher Agreement 35.60%`

因此，Ablation-2 当前更像反例或对照线，而不是候选主线。

主要证据：

- `ablation_experiments/reports/ablation_suite_final_report.json`
- `ablation_experiments/results/ablation2_teacher_student_lstm/teachers/ablation2_teacher_lstm_weld_seam_windows_ws5_tf75_pg0_ep2_lr0.00025_bs128_seed100/evaluation_metrics.txt`
- `ablation_experiments/results/ablation2_teacher_student_lstm/students/ablation2_distill_lstm_weld_seam_windows_ws5_tf75_pg0_ep2_lr0.00025_bs128_T3.0_lce0.8_lkd1.2_lf0.2_seed100/evaluation_metrics.txt`

### Ablation-3：joint teacher-student TCN-Attention

这条线目前只看到非常早期的结果。按
`ablation3_seed_sweep_final_report.json`：

- 当前只记录了 `1` 个 run
- 对应 seed=100，训练 `2` 个 epoch
- 测试准确率 `37.23%`
- `Macro-F1 0.3926`
- `Teacher Agreement 50.82%`

这说明 Ablation-3 还处在早期探索阶段，现有结果不足以和主线正面比较。

主要证据：

- `ablation_experiments/ablation3_joint_tcn_attn/reports/ablation3_seed_sweep_final_report.json`
- `ablation_experiments/ablation3_joint_tcn_attn/reports/ablation_comparison.md`

## 旧版基线与遗留实验

这部分结果仍然有参考价值，但不建议与当前主线直接横比。

### 旧版 LSTM：raw split

`outputs/baseline_lstm` 和 `outputs/baseline_lstm_gpu0` 里有一个容易误读的
结果：测试准确率是 `100.00%`，但 `Macro-F1` 只有 `0.3333`。

原因不是模型异常强，而是测试集已经退化成单类：

- Test support: `Class 0 = 0`
- Test support: `Class 1 = 0`
- Test support: `Class 2 = 316`

因此，这个 `100%` 结果不可作为可比基线。

主要证据：

- `outputs/baseline_lstm/baseline_lstm_raw_data_tf75_ws20_to5_ep150_lr1e-05_bs32_seed42/evaluation_metrics.txt`

### 旧版 LSTM：segment split

更有参考意义的是 `segment_split` 系列。按现有结果：

- 单目录 best：`88.89%`
- 5-seed 扫描 best：`89.47%`
- 5-seed 均值：`83.27%`
- 5-seed 标准差约：`5.85`

因此，如果需要一个旧版 LSTM 基线，推荐引用 `segment_split`，不引用
`raw split` 的 `100%` 结果。

主要证据：

- `outputs/baseline_lstm_gpu0_segment_split/*/evaluation_metrics.txt`
- `outputs/baseline_lstm_gpu0_segment_split_seed_sweep_ep45/*/evaluation_metrics.txt`

### `label_safe_tcn`

`outputs/label_safe_tcn` 对应一条更早的实验线，使用的数据文件还是
`Data/processed_data/weld_seam_windows.npz`，与当前主线数据口径不同。

现有结果是：

- teacher test accuracy：`83.85%`
- student test accuracy：`78.96%`

这条线可以保留作历史结果，但不建议直接与当前主线的
`weld_seam_windows_ws5_tf75_pg0.npz` 结果对比。

主要证据：

- `outputs/label_safe_tcn/epochs40_lr0.0001_bs96_k3_l1_d0.25_db2_wd0.0005_cwauto_seed42_lc1.0_la0.02_lk0.02/evaluation_metrics.txt`
- `outputs/label_safe_tcn/epochs40_lr0.0001_bs96_k3_l1_d0.25_db2_wd0.0005_cwauto_seed42_lc1.0_la0.02_lk0.02/run_args.json`

## 当前结论

基于仓库现有结果，可以把当前项目状态总结为下面几条。

- 当前时刻分类主线已经跑通，teacher 的仓库最高分是 `98.64%`。
- 蒸馏 student 的固定口径 best 仍是 `95.92%`，但仓库中已经存在
  `96.74%` 的更高分 seed sweep 结果。
- 单调后处理是有效的，至少在 seed14 单模型上能把 `96.47%` 提升到
  `98.64%`。
- `horizon=1` 的严格协议版本还没有达标；探索性扫描已经到 `98.31%`，
  但这些高分都用了测试集选模。
- 消融实验里，Ablation-1 tuned 后能到 `91.58%`，而 Ablation-2 和
  Ablation-3 目前都明显偏弱。
- 旧版 LSTM 的 `raw split` 有单类测试集问题，不应作为正式基线。

## 风险与后续整理建议

最后这部分只记录当前结果里最值得注意的风险，方便后续继续整理实验口径。

- 主线 teacher 和 `horizon=1 scan98` 结果里都存在 `checkpoint_metric=test_acc`
  的 run；这类结果更适合标成“探索性最好分数”，不适合作为严格协议结论。
- distill 的 `best_record` 目录没有自动跟进更大规模 seed sweep，后续如果要
  固化主结论，建议手动更新固定 best 文档。
- 旧版和遗留实验的目录很多，建议后续把“主线结果”和“历史结果”拆成两个
  汇总文件，避免在同一口径里混合引用。
