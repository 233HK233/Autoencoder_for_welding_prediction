# Figure 2 ready result sources

This document summarizes the experiment results that are already ready to use
for drawing Figure 2 as of May 23, 2026. The goal is to make the plotting
input unambiguous and to prevent mixing strict and exploratory protocols in one
figure.

## Bottom line

The repository now contains two complete Figure 2 result sources:

- one `strict` source for the official paper figure
- one `exploratory` source for the high-score comparison figure

Both sources are already frozen and already contain the class-wise `Test`
metrics needed for plotting. No additional reruns are required before drawing
Figure 2.

Use exactly one source per figure. Do not mix rows from the `strict` and
`exploratory` files in the same plot.

## Strict official source

Use this source when Figure 2 is meant to represent the official paper result
under one unified protocol.

### Files

- Registry:
  `outputs/paper_figures/h1_results/data/official_run_registry.csv`
- Class metrics:
  `outputs/paper_figures/h1_results/data/class_metrics_summary.csv`
- Result note:
  `docs/2026-05-23-figure2-strict-rerun-results.md`

### Frozen model slots

The strict source contains these five entries:

1. `Strict Teacher Best`
2. `Best Student Distill`
3. `Best Ablation-1`
4. `Best Ablation-2`
5. `Best LSTM baseline`

### Protocol summary

The strict source uses one unified protocol:

- dataset:
  `Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz`
- task: `future-step state classification`
- protocol: `strict`
- label set: `S0 / S1 / S2`
- selection metric: `val_macro_f1`
- baseline family: `LSTM`

### Output size

The strict source is complete for plotting:

- registry rows: `5`
- class-metric rows: `45`
- classes per model: `3`
- metrics per class: `precision`, `recall`, `f1`

## Exploratory source

Use this source when Figure 2 is meant to show the strongest available
exploratory-compatible comparison from the current repository.

### Files

- Registry:
  `outputs/paper_figures/h1_results/data/exploratory_run_registry.csv`
- Class metrics:
  `outputs/paper_figures/h1_results/data/exploratory_class_metrics_summary.csv`
- Result note:
  `docs/2026-05-23-figure2-exploratory-gapfill-results.md`

### Frozen model slots

The exploratory source contains these five entries:

1. `Exploratory Teacher Best`
2. `Best Student Distill`
3. `Best Ablation-1`
4. `Best Ablation-2`
5. `Best GRU baseline`

### Protocol summary

The exploratory source keeps the existing high-score selection logic already
present in the repository:

- dataset:
  `Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz`
- task: `future-step state classification`
- protocol: `exploratory`
- label set: `S0 / S1 / S2`
- teacher checkpoint metric: `test_acc`
- student checkpoint metric: `val_teacher_agreement`
- ablation checkpoint metric: `test_acc`
- baseline family: `GRU`

### Output size

The exploratory source is complete for plotting:

- registry rows: `5`
- class-metric rows: `45`
- classes per model: `3`
- metrics per class: `precision`, `recall`, `f1`

## What the plotting code should read

If the plotting code only needs class-wise bar heights, error grouping, and
model names, read the class-metrics CSV directly.

- strict:
  `outputs/paper_figures/h1_results/data/class_metrics_summary.csv`
- exploratory:
  `outputs/paper_figures/h1_results/data/exploratory_class_metrics_summary.csv`

If the plotting code also needs the frozen run identity, protocol label, or
source run directory, read the matching registry CSV as metadata.

Each class-metrics CSV already contains:

- `model_name`
- `model_family`
- `protocol`
- `class_id`
- `class_name`
- `metric`
- `value`
- `source_run_dir`

## Which source to use

Choose the source based on the story you want Figure 2 to tell.

- Use `strict` if Figure 2 is the official main-paper comparison figure.
- Use `exploratory` if Figure 2 is the stronger high-score comparison figure.
- Do not combine `Strict Teacher Best` with `Best GRU baseline`.
- Do not combine `Exploratory Teacher Best` with `Best LSTM baseline` unless
  you intentionally rebuild the exploratory export with a different baseline
  choice.

## Optional exploratory baseline override

The default exploratory source uses `GRU` because it is the stronger available
baseline in the exploratory pool. If you need architectural continuity with the
strict figure, you can rebuild the exploratory export with `LSTM` instead.

Use this command:

```bash
python ablation_experiments/scripts/build_figure2_exploratory_registry.py \
  --baseline-method lstm
```

After that export, use the regenerated exploratory CSV files as a matched pair.
