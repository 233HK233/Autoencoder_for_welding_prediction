# `horizon=1` 三分类时序预测结果图详细清单（2026-05-20）

这份文档把当前 `horizon=1`、三分类、未来一步状态预测任务的结果图方案整合成一份
可执行清单。目标不是再讨论“能画什么”，而是直接回答五张大结果图分别画什么、每张图
里 9 或 16 个子图怎么排、每张图需要什么数据字段、哪些字段现在仓库里已经有、哪些字段
必须先补导出。

本文默认你当前的主任务是焊缝状态按时间单调演化的三分类预测，且数据集是
`Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz`。本文同时区分两套结果口径：
一套是严格协议 `outputs/teacher_h1_forecast_95`，另一套是探索性高分扫描
`outputs/teacher_h1_scan98_gpu1_v2`。这两套结果不能混在同一主图里直接比较。

## 任务定义与标签口径

这一节先固定论文和图表使用的任务描述，避免后续每张图都换一种说法。

- 任务名称：`future-step state classification`
- 输入：长度为 `window_size=5` 的时序窗口
- 输出：窗口末端后 `horizon=1` 步对应的状态标签
- 主标签：`S0`、`S1`、`S2`
- 推荐显示名称：
  - `S0 = stable stage`
  - `S1 = transition stage`
  - `S2 = severe stage`
- 标签属性：有序三分类，不是三个彼此独立的无序类别
- 误差解释：
  - `S0 ↔ S1` 和 `S1 ↔ S2` 属于相邻误判
  - `S0 ↔ S2` 属于跨阶段重错

为了让五张图自然成型，建议保留 `3` 个主类别不变，同时给每个测试窗口再补两个分析
标签。这样不会改动训练目标，但会显著提升结果图的解释力。

- 分析标签 1：`seam_id`
  - 值域：`a01`、`b01`、`c01`、`c02`
- 分析标签 2：`boundary_distance_bin`
  - 定义：目标时刻 `t+h` 到最近状态边界的距离
  - 推荐分箱：
    - `Near`: `0-5` 个采样步
    - `Mid`: `6-15` 个采样步
    - `Far`: `>=16` 个采样步
- 分析标签 3：`case_type`
  - 推荐定义：
    - `S0-core`
    - `0→1-boundary`
    - `S1-core`
    - `1→2-boundary`
- 分析标签 4：`error_type`
  - 推荐定义：
    - `correct`
    - `adjacent_error`
    - `cross_stage_error`

## 现有数据与缺口

这一节说明哪些图现在就能画，哪些图还差逐样本预测明细。你后面如果先画“现成图”，
可以直接从这里开始执行。

### 现在仓库里已经有的汇总数据

这部分数据已经足够支撑多数组间比较图、箱线图、扫描图和总览图。

- 严格协议主汇总：
  - `outputs/teacher_h1_forecast_95/analysis/teacher_h1_seed_summary.csv`
  - `outputs/teacher_h1_forecast_95/analysis/teacher_h1_coarse_summary.csv`
  - `outputs/teacher_h1_forecast_95/analysis/teacher_h1_focus_summary.csv`
  - `outputs/teacher_h1_forecast_95/analysis/teacher_h1_fallback_summary.csv`
- 探索性扫描主汇总：
  - `outputs/teacher_h1_scan98_gpu1_v2/analysis/teacher_h1_seed_sweep_summary.csv`
  - `outputs/teacher_h1_scan98_gpu1_v2/analysis/teacher_h1_peak_replay_summary.csv`
  - `outputs/teacher_h1_scan98_gpu1_v2/analysis/teacher_h1_ge98.csv`
- 消融和基线主汇总：
  - `ablation_experiments/h1/reports/run_20260327_131013/final_summary/ablation_summary.csv`
  - `ablation_experiments/h1/reports/comparison_18d_baselines/checkpoints/round*/comparison_summary.csv`
- 逐 run 指标：
  - `*/evaluation_metrics.txt`
  - `*/history.json`
  - `*/run_args.json`

### 现在仓库里已经有的逐样本预测

这部分文件说明你们的代码路径已经支持输出逐样本预测，只是 `horizon=1` 主实验目前没有
统一落盘。

- 已有示例：
  - `outputs/baseline_lstm/*/test_predictions.csv`
  - `outputs/baseline_lstm*/confusion_matrix_test.csv`

### 当前缺口

如果你要完成本文推荐的五张主图，其中有两张必须先补统一导出。

- 缺少 `horizon=1` teacher 主实验的 `test_predictions.csv`
- 缺少按焊缝拆分的 `per-seam confusion matrix`
- 缺少带 `seam_id`、`target_idx`、`boundary_distance_bin` 的逐样本评估表

### 建议新增的统一评估明细表

建议后续为每个入选 run 统一导出一个 `test_predictions_detailed.csv`。后面所有精细图都以
这张表为唯一数据源，避免每张图临时拼字段。

推荐字段如下：

| 字段名 | 含义 | 是否现成可得 |
| --- | --- | --- |
| `run_name` | 模型运行名 | 是 |
| `protocol` | `strict` 或 `exploratory` | 是 |
| `model_family` | teacher / student / ablation / baseline | 部分现成 |
| `seam_id` | `a01/b01/c01/c02` | 数据集可得 |
| `window_start_idx` | 窗口起点 | 数据集可得 |
| `target_idx` | `t+h` 目标时刻索引 | 数据集可得 |
| `y_true` | 真实标签 | 是 |
| `y_pred` | 预测标签 | 需导出 |
| `p_s0` | 对 `S0` 的预测概率 | 需导出 |
| `p_s1` | 对 `S1` 的预测概率 | 需导出 |
| `p_s2` | 对 `S2` 的预测概率 | 需导出 |
| `nearest_boundary_dist` | 到最近边界距离 | 需计算 |
| `boundary_distance_bin` | `Near/Mid/Far` | 需计算 |
| `case_type` | `S0-core/0→1-boundary/S1-core/1→2-boundary` | 需计算 |
| `error_type` | `correct/adjacent/cross-stage` | 需计算 |

## 五张大图总览

这一节先给出五张图的一句话定位。后面的章节再展开到子图级别。

| 图号 | 子图数 | 核心问题 | 现在能否直接画 |
| --- | --- | --- | --- |
| 图1 | 16 | 模型在不同焊缝和不同典型场景下具体预测成什么样 | 否，需要逐样本预测 |
| 图2 | 9 | 三个状态里哪个最难，难在精确率、召回率还是 F1 | 部分可画，最佳做法需逐样本预测 |
| 图3 | 9 | 性能下降是否集中在状态边界附近 | 否，需要逐样本预测 |
| 图4 | 16 | 不同焊缝上的混淆模式是否一致 | 否，需要逐样本预测 |
| 图5 | 9 | 严格协议、探索协议、消融与扫描的稳定性和敏感性如何 | 是 |

## 图 1：16 子图定性案例总图

这一图负责讲“模型到底看到了什么、错在什么地方”，它应该是整篇实验部分最直观的一张
图。它不承担总分比较任务，而承担模式展示任务。

### 图 1 的推荐布局

推荐使用 `4×4` 面板。行固定为焊缝，列固定为场景类型。这样读者先沿行看跨焊缝一致性，
再沿列看同类场景的一般规律。

- 行：
  - `a01`
  - `b01`
  - `c01`
  - `c02`
- 列：
  - `S0-core`
  - `0→1-boundary`
  - `S1-core`
  - `1→2-boundary`

### 每个子图画什么

每个子图内部建议固定成“三层结构”，不要每个 panel 自己发明样式。

1. 顶部：选 `3-5` 个代表性通道的原始波形
2. 中部：真实状态条带，颜色表示 `S0/S1/S2`
3. 底部：模型预测概率曲线或预测标签条带

如果版面过挤，底部保留 `argmax` 标签条即可，但至少要对入选的 `4` 个关键 panel 保留
概率曲线。

### 图 1 的 panel 选择规则

每个 panel 必须按统一规则选，不要人工挑“最好看”的样本。

- `S0-core`：
  - 从 `boundary_distance_bin=Far` 的 `S0` 正确样本中取置信度最高的样本
- `0→1-boundary`：
  - 从靠近 `0→1` 边界的 `Near` 样本中取一个正确样本
- `S1-core`：
  - 从 `boundary_distance_bin=Far` 的 `S1` 样本中取一个典型样本
- `1→2-boundary`：
  - 从靠近 `1→2` 边界的 `Near` 样本中优先取一个误判样本

### 图 1 的数据来源

这一图必须基于 `test_predictions_detailed.csv`，并且需要能回溯到原始时间轴。

- 必需字段：
  - `seam_id`
  - `target_idx`
  - `y_true`
  - `y_pred`
  - `p_s0/p_s1/p_s2`
  - `case_type`
  - `boundary_distance_bin`
- 额外原始数据：
  - 原始 CSV 或标准化后的完整序列

### 图 1 的标题模板

推荐主标题：

`Qualitative prediction cases across seams and state-transition contexts.`

推荐子图标题模板：

`a01 | S0-core`

### 图 1 的 caption 首句模板

`Figure X shows that the model remains stable in core regions, while most
prediction ambiguity concentrates near the 0→1 and 1→2 transition boundaries.`

### 图 1 的完成检查

- 每个 panel 的时间范围一致
- 每个 panel 的颜色映射一致
- `S0/S1/S2` 的显示顺序固定
- 至少包含 `2` 个正确边界案例和 `2` 个错误边界案例
- 不混入严格协议和探索协议的样本

## 图 2：9 子图分类性能分解图

这一图负责讲“哪个状态难”。它是最标准、最容易被审稿人快速理解的一张定量图。

### 图 2 的推荐布局

推荐使用 `3×3` 面板。行是状态，列是评价指标。

- 行：
  - `S0`
  - `S1`
  - `S2`
- 列：
  - `Precision`
  - `Recall`
  - `F1`

### 每个子图画什么

每个 panel 内部使用分组柱状图，横轴是模型，纵轴是该状态下的对应指标。

推荐比较对象最多 `4-5` 个，避免图例过长。建议优先选：

- `Strict Teacher Best`
- `Exploratory Teacher Best`
- `Best Student Distill`
- `Best Ablation-1`
- `Best LSTM baseline` 或 `Best GRU baseline`

### 图 2 的最小数据表

建议整理成 `class_metrics_summary.csv`，每行对应“一个模型在一个类上的一个指标”。

推荐字段如下：

| 字段名 | 含义 |
| --- | --- |
| `model_name` | 图例名称 |
| `protocol` | `strict/exploratory/baseline/ablation` |
| `class_id` | `0/1/2` |
| `class_name` | `S0/S1/S2` |
| `metric` | `precision/recall/f1` |
| `value` | 数值 |

### 图 2 的现有数据来源

这张图的粗版可以先从 `evaluation_metrics.txt` 里的分类报告手工抽数。正式版建议从统一导出
脚本生成 `class_metrics_summary.csv`。

- 可直接抽取的来源：
  - `outputs/teacher_h1_scan98_gpu1_v2/runs/*/evaluation_metrics.txt`
  - `outputs/teacher_h1_forecast_95/runs/*/evaluation_metrics.txt`
- 汇总对比来源：
  - `ablation_experiments/h1/reports/.../ablation_summary.csv`

### 图 2 的标题模板

推荐主标题：

`Per-class precision, recall, and F1 reveal that the transition state is the
dominant source of classification difficulty.`

### 图 2 的 caption 首句模板

`Figure X shows that class-wise difficulty is highly imbalanced, with S1
consistently underperforming S0 and S2 across most model variants.`

### 图 2 的完成检查

- 同一列的纵轴范围统一
- 模型顺序在所有 panel 中完全一致
- 最优模型放在最右侧或图例首位
- 明确注明 `strict` 和 `exploratory` 口径

## 图 3：9 子图边界难度分解图

这一图负责讲“是不是边界导致了性能下降”。如果你想把三分类时序问题讲成“状态转移预测”
而不是普通分类，这张图最关键。

### 图 3 的推荐布局

推荐使用 `3×3` 面板。行是状态，列是到边界距离的难度档。

- 行：
  - `S0`
  - `S1`
  - `S2`
- 列：
  - `Far`
  - `Mid`
  - `Near`

### 每个子图画什么

每个 panel 内部仍然使用分组柱状图。纵轴推荐用 `Accuracy` 或 `F1`，二选一即可，不要混用。
如果你更关心类别不平衡，优先用 `F1`。

推荐每个 panel 比较 `3-4` 个模型：

- `Strict Teacher Best`
- `Exploratory Teacher Best`
- `Best Student Distill`
- `Best Baseline`

### 图 3 的最小数据表

建议整理成 `boundary_bin_metrics.csv`。

推荐字段如下：

| 字段名 | 含义 |
| --- | --- |
| `model_name` | 图例名称 |
| `class_name` | `S0/S1/S2` |
| `boundary_distance_bin` | `Far/Mid/Near` |
| `metric` | `accuracy` 或 `f1` |
| `value` | 数值 |
| `support` | 样本数 |

### 图 3 的计算规则

这一图的可信度取决于你是否固定了边界定义。建议在文中写死，不要临时调整。

- `Near`: `nearest_boundary_dist <= 5`
- `Mid`: `6 <= nearest_boundary_dist <= 15`
- `Far`: `nearest_boundary_dist >= 16`

### 图 3 的标题模板

推荐主标题：

`Performance degradation concentrates near state-transition boundaries rather
than in the stable core regions.`

### 图 3 的 caption 首句模板

`Figure X shows that most performance loss appears in Near-boundary samples,
while core-region prediction remains substantially easier across all models.`

### 图 3 的完成检查

- 三档边界定义在正文和图注中一致
- 每个 panel 标明 `support`
- 不使用仅凭肉眼挑出的边界样本

## 图 4：16 子图按焊缝混淆模式图

这一图负责讲“不同焊缝上的混淆结构是否一致”。它和图 2 的差异在于，图 2 讲类别难度，
图 4 讲域内差异和跨焊缝稳定性。

### 图 4 的推荐布局

推荐使用 `4×4` 面板。行固定为焊缝，列固定为模型。

- 行：
  - `a01`
  - `b01`
  - `c01`
  - `c02`
- 列：
  - `Strict Teacher Best`
  - `Exploratory Teacher Best`
  - `Best Student Distill`
  - `Best Baseline` 或 `Best Ablation-1`

### 每个子图画什么

每个 panel 画一个 `3×3` confusion matrix，必须做行归一化。你的任务是比较混淆模式，不是
比较样本总量。

推荐显示规则：

- 单元格显示百分比
- 对角线加粗或用更深色
- 横轴：预测标签
- 纵轴：真实标签

### 图 4 的最小数据表

建议整理成 `per_seam_confusion.csv`。

推荐字段如下：

| 字段名 | 含义 |
| --- | --- |
| `model_name` | 列索引 |
| `seam_id` | 行索引 |
| `y_true` | 真实标签 |
| `y_pred` | 预测标签 |
| `count` | 计数 |
| `ratio` | 行归一化比例 |

### 图 4 的关键观察点

这张图在写 caption 和正文时，重点不在“谁最高”，而在“谁错得一致、谁错得离谱”。

建议优先观察：

- 哪个焊缝的 `S1` 最容易被预测成 `S2`
- 是否存在某个焊缝上 `S0→S2` 的跨阶段重错
- 严格协议与探索协议的混淆形态是否相同

### 图 4 的标题模板

推荐主标题：

`Per-seam confusion patterns show whether errors are globally consistent or
concentrated on specific weld seams.`

### 图 4 的 caption 首句模板

`Figure X shows that seam-wise error patterns are not uniform, and most severe
cross-stage confusion concentrates on a subset of seams rather than appearing
equally across all four seams.`

### 图 4 的完成检查

- 所有 confusion matrix 都做同一种归一化
- 色条范围对所有 panel 完全一致
- 每个 panel 标出样本数

## 图 5：9 子图扫描与消融稳定性总图

这一图负责讲“结果是否稳定、哪些超参最敏感、严格协议和探索性高分差在哪里”。这张图
现在就可以做，而且最适合先完成。

### 图 5 的推荐布局

推荐使用 `3×3` 面板。每个 panel 对应一个问题，而不是强行固定“行列同义”。因为这张图
本质上是实验管理总览图。

推荐九个 panel 如下：

1. `strict seed effect`
2. `exploratory seed effect`
3. `strict vs exploratory`
4. `lr sensitivity`
5. `dropout sensitivity`
6. `weight decay sensitivity`
7. `layer depth sensitivity`
8. `teacher vs student vs ablation`
9. `runtime vs accuracy`

### 每个子图的图型建议

这张图不适合全用柱状图。建议按问题选图型。

- `strict seed effect`：箱线图或 strip plot
- `exploratory seed effect`：箱线图或 strip plot
- `strict vs exploratory`：并列箱线图
- `lr sensitivity`：折线图
- `dropout sensitivity`：折线图
- `weight decay sensitivity`：折线图
- `layer depth sensitivity`：折线图
- `teacher vs student vs ablation`：分组柱状图
- `runtime vs accuracy`：散点图

### 图 5 的现有数据来源

这张图可以几乎完全从现有 CSV 直接生成。

- 严格协议：
  - `outputs/teacher_h1_forecast_95/analysis/teacher_h1_seed_summary.csv`
- 探索协议：
  - `outputs/teacher_h1_scan98_gpu1_v2/analysis/teacher_h1_seed_sweep_summary.csv`
- 消融总览：
  - `ablation_experiments/h1/reports/run_20260327_131013/final_summary/ablation_summary.csv`
- 训练时长和参数：
  - 汇总 CSV 中的 `elapsed_sec`、`lr`、`tcn_dropout`、`weight_decay`、
    `tcn_layers`

### 图 5 的 panel 明细

下面给出每个 panel 的横纵轴和对应来源。

| Panel | 图型 | 横轴 | 纵轴 | 数据来源 |
| --- | --- | --- | --- | --- |
| P1 | box/strip | `seed` | `test_accuracy_percent` | strict seed summary |
| P2 | box/strip | `seed` | `test_accuracy_percent` | exploratory seed summary |
| P3 | box | `protocol` | `test_accuracy_percent` | strict + exploratory |
| P4 | line | `lr` | `test_accuracy_percent` | exploratory seed sweep |
| P5 | line | `tcn_dropout` | `test_accuracy_percent` | exploratory seed sweep |
| P6 | line | `weight_decay` | `test_accuracy_percent` | strict + exploratory |
| P7 | line | `tcn_layers` | `test_accuracy_percent` | strict + exploratory |
| P8 | bar | `model_family` | `best_test_acc` | ablation summary |
| P9 | scatter | `elapsed_sec` | `test_accuracy_percent` | strict + exploratory |

### 图 5 的标题模板

推荐主标题：

`Performance stability depends more on protocol and hyperparameter choice than
on the nominal three-class formulation itself.`

### 图 5 的 caption 首句模板

`Figure X shows that the large gap between strict and exploratory results is
mainly driven by selection protocol and hyperparameter sensitivity, rather than
by an intrinsic inability of the horizon=1 task to reach high accuracy.`

### 图 5 的完成检查

- `strict` 与 `exploratory` 用不同颜色
- `test_acc` 选模结果必须明确标注为 exploratory
- 不把 `best run` 和 `distribution` 画在同一纵轴上误导读者

## 统一绘图风格

这一节固定所有结果图的样式，避免五张图像五个人画出来的。

### 颜色与图例

建议在整篇论文中固定以下颜色映射。

- 状态颜色：
  - `S0`: 蓝色
  - `S1`: 橙色
  - `S2`: 红色
- 协议颜色：
  - `strict`: 深灰
  - `exploratory`: 深绿
- 模型颜色：
  - `teacher`: 深蓝
  - `student`: 棕色
  - `baseline`: 中灰
  - `ablation`: 紫灰或橄榄色

### 版式

所有结果图必须统一下列约束。

- 导出格式优先 `PDF` 和 `SVG`
- 字号不低于 `8 pt`
- 线宽建议 `1.2-1.8 pt`
- 小图内图例尽量放外部
- 所有 panel label 使用 `A, B, C, ...`

### 坐标轴

所有分类指标图必须统一度量范围和命名。

- 百分比统一显示为 `0-100%`
- 同类 panel 的纵轴范围一致
- 类名一律写 `S0/S1/S2`，不要混写 `Class 0/1/2`

## 出图执行顺序

这一节给出最现实的执行顺序。你不需要等所有数据都补齐，再一起开画。

### 第一阶段：现在就能完成

你可以先完成图 5，再完成图 2 的粗版。

1. 先画图 5 的 9 子图扫描总览
2. 从 `evaluation_metrics.txt` 抽每类 `precision/recall/f1`
3. 先画图 2 的粗版，用于验证“`S1` 最难”这个主结论

### 第二阶段：补统一逐样本导出

这一阶段的目标是补出图 1、图 3、图 4 所需的唯一数据源。

1. 为入选模型统一导出 `test_predictions_detailed.csv`
2. 为每个样本计算 `nearest_boundary_dist`
3. 生成 `case_type` 与 `error_type`
4. 汇总 `per_seam_confusion.csv`
5. 汇总 `boundary_bin_metrics.csv`

### 第三阶段：补最终主图

在统一明细表落盘后，再完成其余三张主图。

1. 画图 3 的边界难度分解图
2. 画图 4 的按焊缝混淆模式图
3. 最后画图 1 的定性案例总图

## 每张图的最终产出文件建议

这一节固定输出文件名，避免后面目录混乱。

建议输出到 `outputs/paper_figures/h1_results/`，并统一使用如下文件名：

- `fig1_h1_case_gallery_4x4.pdf`
- `fig1_h1_case_gallery_4x4.svg`
- `fig2_h1_class_metrics_3x3.pdf`
- `fig2_h1_class_metrics_3x3.svg`
- `fig3_h1_boundary_difficulty_3x3.pdf`
- `fig3_h1_boundary_difficulty_3x3.svg`
- `fig4_h1_per_seam_confusion_4x4.pdf`
- `fig4_h1_per_seam_confusion_4x4.svg`
- `fig5_h1_sweep_ablation_overview_3x3.pdf`
- `fig5_h1_sweep_ablation_overview_3x3.svg`

## 最终建议

如果你只能优先做两张图，建议先做图 5 和图 2。它们现在就能基本成型，而且足够支撑
“任务能做高分，但结果口径差异和状态难度不均衡都很明显”这条主线。

如果你要把论文里的三分类任务讲得更像“状态转移预测”，那就必须补图 3 和图 1。原因很
直接：不把边界附近的难例单独拿出来，读者很容易把这个任务理解成普通的静态分类问题，
而不是一个有明显时间结构和转移结构的预测问题。

## 下一步

如果下一步继续执行，最合理的顺序是先补一份统一的 `test_predictions_detailed.csv`
导出脚本，然后再一次性生成图 1、图 3、图 4 所需的中间表。这样后续改模型、换 run 或
补 baseline 时，不需要重写每张图的统计逻辑。
