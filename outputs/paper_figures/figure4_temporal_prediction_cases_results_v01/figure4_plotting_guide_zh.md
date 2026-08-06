# Figure 4 跨焊缝时序预测案例图绘图说明

本文档说明如何使用当前结果包绘制 `Figure 4：跨焊缝的时序预测案例图`。
结果包已经包含三模型概率、连续滑窗推理、attention ribbon、局部经济特征
摘要、真实状态条带和确定性选样信息。绘图阶段不需要重新训练模型。

## 结果目录

所有绘图源数据位于：

`outputs/paper_figures/figure4_temporal_prediction_cases_results_v01/`

如果需要重新生成结果包，运行：

```bash
python build_figure4_temporal_case_results.py --device auto
```

脚本会创建新的版本目录，例如 `figure4_temporal_prediction_cases_results_v02`。
不要手动覆盖旧目录，便于追溯不同版本。

## 推荐图形结构

这张图推荐使用 `4 x 4` 布局。行表示焊缝，列表示时序场景。每个子图内部
使用固定的四层结构，避免不同 panel 使用不同视觉语法。

行顺序固定为：

- `a01`
- `b01`
- `c01`
- `c02`

列顺序固定为：

- `S0-core`
- `0->1-boundary`
- `S1-core`
- `1->2-boundary`

每个 panel 建议从上到下画：

1. **Attention ribbon**：使用三模型的 5 个输入步 attention 权重。
2. **经济特征信号摘要**：画 `economic_pc1` 或 `economic_mean_z`。
3. **真实状态条带和目标时刻**：用状态颜色条展示 `S0/S1/S2`，并在
   `relative_time_s = 0` 标记目标预测时刻。
4. **三模型概率曲线**：画 Teacher、SEAL-Weld Student 和 Student-only 的
   预测概率轨迹。

## 主要文件用途

绘图通常只需要读取 `figure4_panel_selection.csv`、
`figure4_panel_timeseries.csv` 和 `figure4_panel_attention.csv`。其他文件用于
审计、图注和补充分析。

| 文件 | 用途 |
| --- | --- |
| `figure4_panel_selection.csv` | 16 个 panel 的确定性选样表。 |
| `figure4_panel_timeseries.csv` | 每个 panel 的局部时序信号、状态和概率轨迹。 |
| `figure4_panel_attention.csv` | 每个 panel 的 attention ribbon 数据。 |
| `figure4_highlight_cases.csv` | transfer-gain 和 SEAL-Weld failure 代表案例。 |
| `figure4_test_window_predictions.csv` | H1 测试窗口级完整预测明细。 |
| `figure4_continuous_predictions.csv` | 全焊缝连续滑窗推理结果。 |
| `figure4_validation_summary.csv` | 连续推理与已有测试 CSV 的一致性校验。 |
| `figure4_result_metadata.json` | 数据来源、模型 checkpoint 和选择规则。 |

## Panel 选择规则

图中案例不能人工挑选。当前结果包使用以下确定性规则生成
`figure4_panel_selection.csv`。

- `S0-core` 和 `S1-core`：
  - 在同一焊缝和同一场景内，优先选择 Teacher 与 SEAL-Weld Student 均预测
    正确的测试窗口。
  - 从这些窗口中选择距离最近状态边界最远的窗口。
  - 若距离相同，选择 SEAL-Weld Student 置信度更高的窗口。
  - 若仍相同，选择 `sample_index` 更小的窗口。

- `0->1-boundary` 和 `1->2-boundary`：
  - 在同一焊缝和同一场景内，优先选择 Teacher 与 SEAL-Weld Student 均预测
    正确的测试窗口。
  - 从这些窗口中选择距离状态边界最近的窗口。
  - 若距离相同，选择 SEAL-Weld Student 置信度更高的窗口。
  - 若仍相同，选择 `sample_index` 更小的窗口。

- `transfer_gain` 高亮案例：
  - 从全测试集选择 SEAL-Weld Student 正确、Student-only 错误的窗口。
  - 优先选择距离边界最近的窗口。

- `failure case` 高亮案例：
  - 从全测试集选择 SEAL-Weld Student 仍然错误的窗口。
  - 优先选择距离边界最近的窗口。

当前结果中，`c01|S0-core` 在 H1 测试集中没有同缝候选窗口，因此该 panel
使用 cross-seam backfill，并在 `panel_status` 中标记为 `backfilled`。其来源
写在 `source_seam_name` 中，不能在图注中写成 c01 的原生 S0-core 样本。

## `figure4_panel_selection.csv` 字段

这个文件定义 16 个 panel 画哪一个目标窗口。绘图时先按 `panel_key` 读取
该表，再到其他表中筛选同一 `panel_key` 的局部时序数据。

关键字段如下：

| 字段 | 含义 |
| --- | --- |
| `panel_key` | panel 标识，格式为 `seam|case_type`。 |
| `seam_name` | 图中显示的行焊缝。 |
| `source_seam_name` | 实际数据来源焊缝。backfill 时可能不同于 `seam_name`。 |
| `case_type` | 场景类型。 |
| `panel_status` | `selected` 或 `backfilled`。 |
| `selection_rule` | 当前 panel 使用的选择规则。 |
| `sample_index` | H1 测试集样本编号。 |
| `start_idx` | 输入窗口起点。 |
| `target_idx` | 目标预测时刻。 |
| `y_true` | 目标时刻真实状态。 |
| `teacher_pred` | Teacher 预测类别。 |
| `seal_student_pred` | SEAL-Weld Student 预测类别。 |
| `student_only_pred` | Student-only 预测类别。 |
| `nearest_boundary_dist` | 目标时刻到最近状态边界的采样步距离。 |
| `boundary_distance_bin` | `Near`、`Mid` 或 `Far`。 |

## `figure4_panel_timeseries.csv` 字段

这个文件是绘制每个 panel 主体内容的核心数据表。每一行对应某个 panel 中的
一个局部时间点。

关键字段如下：

| 字段 | 含义 |
| --- | --- |
| `panel_key` | 与 `figure4_panel_selection.csv` 对齐。 |
| `time_idx` | 原始焊缝时间轴索引。 |
| `relative_step` | 相对目标时刻的采样步，目标时刻为 0。 |
| `relative_time_s` | 相对目标时刻的秒数，采样间隔为 `0.01 s`。 |
| `state_label` | 当前时间点真实状态，`0=S0`、`1=S1`、`2=S2`。 |
| `economic_mean_z` | 经济特征 z-score 均值。 |
| `economic_pc1` | 经济特征一维 PCA 摘要。 |
| `teacher_prob_s0/s1/s2` | Teacher 对三个状态的概率。 |
| `seal_student_prob_s0/s1/s2` | SEAL-Weld Student 对三个状态的概率。 |
| `student_only_prob_s0/s1/s2` | Student-only 对三个状态的概率。 |

建议默认画 `economic_pc1` 作为经济特征摘要。如果该曲线在某些 panel 中
视觉过尖，可以改画 `economic_mean_z`。

概率曲线有两种推荐画法：

- **紧凑版**：每个模型只画目标真实类别 `y_true` 对应的概率曲线。这样 16 个
  panel 更清晰，适合主文图。
- **完整版**：每个模型画 `S0/S1/S2` 三条概率曲线，但需要用线型或透明度区分
  类别，容易拥挤，更适合补充图。

## `figure4_panel_attention.csv` 字段

这个文件用于画 panel 顶部的浅色 attention ribbon。每个 panel 有 15 行数据，
即 3 个模型乘以 5 个输入步。

关键字段如下：

| 字段 | 含义 |
| --- | --- |
| `panel_key` | 与 panel 对齐。 |
| `method` | `Teacher-18D`、`SEAL-Weld Student-13D` 或 `Student-only-13D`。 |
| `method_slug` | 简短模型名。 |
| `input_step` | 输入窗口内部步号，范围为 `0..4`。 |
| `raw_idx` | 输入步在原始焊缝时间轴上的索引。 |
| `relative_step` | 输入步相对目标时刻的位置。 |
| `relative_time_s` | 输入步相对目标时刻的秒数。 |
| `attention_weight` | 归一化 attention 权重。 |

绘制 ribbon 时，可以把每个模型画成一条 5 格热图。推荐顺序为 Teacher、
SEAL-Weld Student、Student-only。为了避免喧宾夺主，ribbon 使用浅色透明
色阶即可。

## 状态和模型的推荐视觉编码

状态条带建议使用固定颜色，并在全图共享一个 legend。

| 状态 | 含义 | 推荐颜色 |
| --- | --- | --- |
| `S0` | stable stage | 浅蓝色 |
| `S1` | transition stage | 金黄色 |
| `S2` | severe stage | 红橙色 |

模型概率曲线建议使用固定颜色，并在全图共享一个 legend。

| 模型 | 推荐颜色 | 线型 |
| --- | --- | --- |
| Teacher-18D | 深灰色 | 实线 |
| SEAL-Weld Student-13D | 蓝色 | 实线或较粗实线 |
| Student-only-13D | 橙色 | 虚线 |

目标时刻用竖线标记在 `relative_time_s = 0`。边界 panel 中，如果状态切换点
出现在当前局部窗口内，可以额外用细虚线标出状态边界。

## 推荐绘图流程

下面的流程适合用 Matplotlib 实现。

1. 读取 `figure4_panel_selection.csv`，固定行列顺序生成 `4 x 4` axes。
2. 对每个 `panel_key`，从 `figure4_panel_timeseries.csv` 取局部时间序列。
3. 在 panel 背景或中层画 `state_label` 状态条带。
4. 在信号层画 `economic_pc1` 或 `economic_mean_z`。
5. 在目标时刻 `relative_time_s = 0` 画竖线。
6. 从 `figure4_panel_attention.csv` 取同一 `panel_key` 的 attention 权重，
   在顶部画 3 条浅色 ribbon。
7. 在概率层画 Teacher、SEAL-Weld Student 和 Student-only 的概率曲线。
8. 用 `figure4_panel_selection.csv` 的 `panel_status` 和 `selection_rule`
   给 backfill 或特殊案例加小标注。
9. 用 `figure4_highlight_cases.csv` 标出 transfer-gain 和 SEAL-Weld failure
   的代表窗口。

## 结果完整性检查

当前结果包已经通过完整性检查：

- `figure4_panel_selection.csv` 包含 16 个 panel。
- `figure4_panel_attention.csv` 包含 240 行，即 16 个 panel 乘以 3 个模型
  乘以 5 个输入步。
- `figure4_panel_timeseries.csv` 包含 717 行局部时序数据。
- 连续滑窗推理与已有测试预测的 argmax 完全一致。
- 最大概率差为 `8.06e-05`，在绘图和数值复核上可以接受。
- `c01|S0-core` 是唯一 backfilled panel。

## 图注建议

可以在图注中使用以下中文表述：

“Figure 4 展示了四条焊缝在核心状态区和状态边界附近的未来一步预测案例。
每个 panel 按确定性规则选择：核心区选择距离最近状态边界最远且 Teacher 与
SEAL-Weld Student 均预测正确的窗口；边界区选择距离边界最近且两者均预测
正确的窗口。若同一焊缝不存在对应测试窗口，则使用同一场景下的 cross-seam
backfill 并在图中标注。顶部 ribbon 表示输入窗口内的 attention 权重，中部
显示经济特征摘要和真实状态条带，底部显示 Teacher、SEAL-Weld Student 和
Student-only 的预测概率轨迹。transfer-gain 和 failure case 按全测试集的
确定性规则额外标出。”

## 注意事项

绘图时需要区分 `seam_name` 和 `source_seam_name`。`seam_name` 是图中布局
位置，`source_seam_name` 是实际数据来源。对于 backfilled panel，两者不同。

不要人工替换 panel 样本。若需要改变选择规则，先修改
`build_figure4_temporal_case_results.py` 并重新生成结果包，再更新本文档中的
规则说明。
