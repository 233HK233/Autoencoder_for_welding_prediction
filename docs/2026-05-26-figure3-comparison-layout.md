# Figure 3 comparison layout

This document freezes the current design for the paper-facing Figure 3.
The goal is to give a separate plotting agent one unambiguous specification
for layout, data sources, method ordering, and storytelling constraints.

The figure is designed as a `4 x 4` multi-panel experimental-results figure.
It uses the exploratory ordered source for `Teacher` and `Student`, and the
round-3 H1 baseline comparison pool for `GRU`, `LSTM`, `Inception`, and
`Transformer`.

The design explicitly avoids:

- any seed-distribution panels
- any sweep-round progression panels
- any ablation panels
- any mixing of strict and exploratory result rows inside the same panel

Figure 3 is a comparison figure, not an ablation figure. Figure 4 already
covers ablations elsewhere in the paper, so Figure 3 must stay focused on
method comparison and error-pattern explanation.

## Bottom line

Figure 3 tells one story in four steps:

1. who performs best overall
2. which class is hardest
3. how the proposed methods compare against alternative architectures
4. what error-pattern differences explain the performance gap

The intended visual order is:

`Teacher > Student > baselines`

Within the baseline family, the intended order is:

`GRU > LSTM > Inception > Transformer`

This ranking reflects the current H1 round-3 comparison pool.

## Data sources

Use these files only.

### Teacher and student source

- Registry:
  `outputs/paper_figures/h1_results/data/exploratory_ordered_run_registry.csv`
- Class metrics:
  `outputs/paper_figures/h1_results/data/exploratory_ordered_class_metrics_summary.csv`
- Result note:
  `docs/2026-05-23-figure2-exploratory-ordered-registry.md`

These files provide the ordered exploratory source where:

- `Teacher` is best
- `Student` is second
- the ordered storytelling relation is preserved

### Baseline source

- Summary:
  `ablation_experiments/h1/reports/comparison_18d_baselines/checkpoints/round3/comparison_summary.csv`
- Supporting note:
  `ablation_experiments/h1/reports/comparison_18d_baselines/checkpoints/round3/comparison_report.md`

These files provide the baseline-family comparison pool for:

- `GRU`
- `LSTM`
- `Inception`
- `Transformer`

### Task and dataset

All panels must correspond to:

- dataset:
  `Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz`
- task:
  `future-step state classification`
- label set:
  `S0 / S1 / S2`

## Frozen method order

Use one global method order in every panel where the methods appear together.
Do not sort per panel. Do not re-order by the metric value inside each panel.

Global method order:

1. `Teacher`
2. `Student`
3. `GRU`
4. `LSTM`
5. `Inception`
6. `Transformer`

Label mapping:

- `Exploratory Teacher Best` -> `Teacher`
- `Best Student Distill` -> `Student`
- `gru` -> `GRU`
- `lstm` -> `LSTM`
- `inception` -> `Inception`
- `transformer` -> `Transformer`

## Figure type

- Type: `experimental-results`
- Paradigm: `small-multiple comparison grid`
- Reason:
  this figure needs to combine overall performance, class-wise difficulty,
  architecture comparison, and error-pattern explanation in one reviewer-fast
  layout.

## Canvas and layout

- Canvas: full-width, landscape
- Grid: `4 rows x 4 columns`
- Panel labels: `A` through `P`
- Keep one shared legend outside the grid when possible
- Do not place a separate legend in every panel

## Row 1: overall performance headline

This row gives the reader the main ranking before any detailed diagnosis.

### Panel A

- Title:
  `Overall test accuracy`
- Chart:
  bar chart
- X-axis:
  `Teacher, Student, GRU, LSTM, Inception, Transformer`
- Y-axis:
  `Test accuracy (%)`
- Purpose:
  establish the main overall ranking

### Panel B

- Title:
  `Overall macro-F1`
- Chart:
  bar chart
- X-axis:
  same as Panel A
- Y-axis:
  `Macro-F1 (%)`
- Purpose:
  confirm that the ranking is not only an accuracy artifact

### Panel C

- Title:
  `Accuracy gain over GRU baseline`
- Chart:
  lollipop chart or delta bar chart
- X-axis:
  `Teacher, Student, LSTM, Inception, Transformer`
- Y-axis:
  `Delta accuracy over GRU (pp)`
- Zero reference:
  yes
- Purpose:
  show how far each method is from the strongest baseline anchor

### Panel D

- Title:
  `Macro-F1 gain over GRU baseline`
- Chart:
  lollipop chart or delta bar chart
- X-axis:
  same as Panel C
- Y-axis:
  `Delta macro-F1 over GRU (pp)`
- Zero reference:
  yes
- Purpose:
  confirm that the gain persists under class-balanced evaluation

## Row 2: class difficulty

This row answers which class drives the performance gap.

### Panel E

- Title:
  `S0 F1`
- Chart:
  grouped bar chart
- X-axis:
  global method order
- Y-axis:
  `F1 (%)`
- Purpose:
  measure performance on the stable stage

### Panel F

- Title:
  `S1 F1`
- Chart:
  grouped bar chart
- X-axis:
  global method order
- Y-axis:
  `F1 (%)`
- Purpose:
  show performance on the transition stage
- Note:
  this is the most important panel in Row 2

### Panel G

- Title:
  `S2 F1`
- Chart:
  grouped bar chart
- X-axis:
  global method order
- Y-axis:
  `F1 (%)`
- Purpose:
  measure performance on the severe stage

### Panel H

- Title:
  `Per-class F1 profile`
- Chart:
  line profile chart
- X-axis:
  `S0, S1, S2`
- Y-axis:
  `F1 (%)`
- Lines:
  one line per method
- Purpose:
  let the reader see in one panel that `S1` is the dominant difficulty class

## Row 3: comparison against alternative architectures

This row is the core comparison row. It must not include ablation methods.
It compares the proposed methods against alternative architectures and explains
where the advantage appears.

### Panel I

- Title:
  `Baseline-family test accuracy`
- Chart:
  bar chart
- X-axis:
  `GRU, LSTM, Inception, Transformer`
- Y-axis:
  `Test accuracy (%)`
- Purpose:
  rank the comparison baselines under the same comparison pool

### Panel J

- Title:
  `Baseline-family macro-F1`
- Chart:
  bar chart
- X-axis:
  same as Panel I
- Y-axis:
  `Macro-F1 (%)`
- Purpose:
  verify whether the baseline ranking is robust beyond raw accuracy

### Panel K

- Title:
  `Baseline-family per-class F1`
- Chart:
  heatmap
- Rows:
  `GRU, LSTM, Inception, Transformer`
- Columns:
  `S0, S1, S2`
- Cell value:
  `F1 (%)`
- Purpose:
  show which baseline degrades on which class

### Panel L

- Title:
  `Teacher and Student gains over GRU`
- Chart:
  heatmap
- Rows:
  `Teacher, Student`
- Columns:
  `S0, S1, S2`
- Cell value:
  `Delta F1 over GRU (pp)`
- Purpose:
  show whether the proposed gains come mainly from the transition class or
  from all classes equally

## Row 4: error-pattern explanation

This row explains why the performance ranking differs. It does not rely on a
full H1 confusion-matrix export because the current H1 frozen source already
contains class-wise `precision`, `recall`, and `f1`, but does not yet provide
a unified confusion-matrix artifact for every frozen comparison run.

Use class-wise precision-recall signatures instead of forcing confusion
matrices.

### Panel M

- Title:
  `S0 precision-recall signature`
- Chart:
  precision-recall scatter
- X-axis:
  `Recall (%)`
- Y-axis:
  `Precision (%)`
- Points:
  `Teacher, Student, GRU, LSTM, Inception, Transformer`
- Purpose:
  show whether methods over-predict or under-detect the stable class

### Panel N

- Title:
  `S1 precision-recall signature`
- Chart:
  precision-recall scatter
- X-axis:
  `Recall (%)`
- Y-axis:
  `Precision (%)`
- Points:
  same as Panel M
- Purpose:
  explain transition-state handling
- Note:
  this is the most important panel in Row 4

### Panel O

- Title:
  `S2 precision-recall signature`
- Chart:
  precision-recall scatter
- X-axis:
  `Recall (%)`
- Y-axis:
  `Precision (%)`
- Points:
  same as Panel M
- Purpose:
  show whether severe-state recognition is stable or traded off against other
  classes

### Panel P

- Title:
  `Transition penalty`
- Chart:
  delta bar chart
- X-axis:
  global method order
- Y-axis:
  `((R_S0 + R_S2) / 2 - R_S1) (pp)`
- Definition:
  higher values mean the method loses more recall on the transition class than
  on the two easier classes
- Purpose:
  compress the error-pattern story into one summary statistic

## Data extraction rules

The plotting agent must follow these extraction rules.

### Overall metrics

For `Teacher` and `Student`:

- read the run directories from
  `outputs/paper_figures/h1_results/data/exploratory_ordered_run_registry.csv`
- parse overall `Test accuracy` and `Macro-F1` from each run's
  `evaluation_metrics.txt`

For `GRU`, `LSTM`, `Inception`, and `Transformer`:

- read from
  `ablation_experiments/h1/reports/comparison_18d_baselines/checkpoints/round3/comparison_summary.csv`

### Class-wise metrics

For `Teacher` and `Student`:

- read directly from
  `outputs/paper_figures/h1_results/data/exploratory_ordered_class_metrics_summary.csv`

For the four baselines:

- parse the corresponding best-run `evaluation_metrics.txt`
- extract class-wise `precision`, `recall`, and `f1` from the `Test`
  classification report block only

### Baseline-only row

Panels `I`, `J`, and `K` must use only:

- `GRU`
- `LSTM`
- `Inception`
- `Transformer`

Do not include `Teacher` or `Student` in these three panels.

### No seed panels

Do not plot:

- seed distributions
- sweep-stage distributions
- round-by-round progression
- best-vs-mean seed comparisons

These are intentionally excluded from Figure 3.

## Visual rules

### Color assignment

Use one fixed method palette across the entire figure.

- `Teacher`: deep navy
- `Student`: teal-blue
- `GRU`: warm orange
- `LSTM`: muted red
- `Inception`: olive
- `Transformer`: slate gray

Recommended examples:

- `Teacher`: `#1F3A5F`
- `Student`: `#4C8D9B`
- `GRU`: `#D98C3F`
- `LSTM`: `#B55D5C`
- `Inception`: `#7A8F3C`
- `Transformer`: `#6B7280`

### Typography and export

- export to `PDF` and `SVG`
- minimum font size: `8 pt`
- use one shared legend if possible
- keep panel labels at the upper-left of each panel
- keep axes ranges consistent within the same row where applicable

### Heatmap rules

- use one consistent sequential palette
- annotate cells with values if legibility remains acceptable
- keep the same color range across related heatmaps when comparing deltas or
  F1 values

### Scatter rules

- use the same axis ranges across Panels `M`, `N`, and `O`
- annotate points with short method labels if the legend becomes visually far
  from the points

## Caption direction

The first sentence of the caption should state the core finding directly.

Recommended first sentence:

`Figure 3 shows that Teacher and Student outperform alternative comparison
architectures overall, and that their main advantage comes from better
handling of the transition class rather than from uniformly easier recognition
of all three states.`

Recommended second-sentence direction:

- Row 1: highlight overall ranking
- Row 2: state that `S1` is the hardest class
- Row 3: state that `GRU` is the strongest baseline family member
- Row 4: state that the dominant gap is transition-state recall and precision

## Integrity constraints

The plotting agent must preserve these constraints.

- Do not mix `strict` and `exploratory` rows inside one panel.
- Do not introduce ablation methods into Figure 3.
- Do not re-order methods per panel.
- Do not convert Figure 3 into a seed-robustness figure.
- Do not replace the baseline family with only one baseline everywhere.
- Do not claim robustness or stability from this figure alone.

## Deliverables

The plotting agent should produce:

1. one paper-facing figure export in `PDF`
2. one matching `SVG`
3. one high-resolution `PNG` preview
4. one CSV or JSON artifact describing the exact values plotted in Panels
   `A-P`

## Suggested output directory

Recommended output root:

`outputs/paper_figures/fig3_h1_comparison_layout_v01/`

Suggested files:

- `figure3_h1_comparison.pdf`
- `figure3_h1_comparison.svg`
- `figure3_h1_comparison.png`
- `figure3_h1_comparison_panel_values.csv`

## Next steps

Use this document as the single specification for the plotting agent. If the
agent discovers that a required H1 baseline class-wise metric is missing, the
agent should stop and report exactly which source run is missing a parsable
`evaluation_metrics.txt` block instead of silently approximating the panel.
