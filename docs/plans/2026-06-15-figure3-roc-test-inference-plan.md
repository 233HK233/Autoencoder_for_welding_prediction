# Figure 3 ROC test inference plan

This plan defines the implementation work needed to generate ROC curves for
the current Figure 3 comparison experiments. The goal is to reuse the six runs
already selected by Figure 3, export test-set probabilities for each run, and
plot threshold-independent ROC curves without retraining or reselection.

## Goal

Generate a publication-ready ROC figure for the current Figure 3 comparison
set, using exactly the runs listed in:

`outputs/figure 3/source_tables/comparison_summary.csv`

The implementation must produce one test prediction CSV per method, a ROC AUC
summary table, a ROC curve point table, and vector figure exports.

## Non-goals

- Do not retrain any model.
- Do not run a new sweep.
- Do not choose models by ROC AUC.
- Do not modify existing Figure 2 or Figure 3 source tables.
- Do not mix train or validation predictions into ROC outputs.

## Required Figure 3 methods

Use the six methods currently listed in Figure 3:

1. `teacher-student(student)`
2. `teacher(18D upper bound)`
3. `lstm`
4. `gru`
5. `transformer`
6. `inception`

All methods must be evaluated on the same test split from:

`Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz`

The dataset contains `356` test samples. Each exported prediction CSV must
therefore contain exactly `356` data rows.

## Files to create

Create these implementation files:

- `ablation_experiments/scripts/export_figure3_roc_predictions.py`
- `ablation_experiments/scripts/plot_figure3_roc.py`
- `tests/test_export_figure3_roc_predictions.py`
- `tests/test_plot_figure3_roc.py`

Create these output directories at runtime:

- `outputs/roc_figure3/predictions/`
- `outputs/roc_figure3/tables/`
- `outputs/roc_figure3/figure_exports/`

## Output contract

The export script must write six prediction CSV files under:

`outputs/roc_figure3/predictions/`

Use stable slugs for filenames:

- `teacher_student_student_test_predictions.csv`
- `teacher_18d_upper_bound_test_predictions.csv`
- `lstm_test_predictions.csv`
- `gru_test_predictions.csv`
- `transformer_test_predictions.csv`
- `inception_test_predictions.csv`

Each prediction CSV must contain these columns:

- `sample_index`
- `seam_id`
- `seam_name`
- `start_idx`
- `target_idx`
- `y_true`
- `y_pred`
- `logit_class_0`
- `logit_class_1`
- `logit_class_2`
- `prob_class_0`
- `prob_class_1`
- `prob_class_2`

The plotting script must write:

- `outputs/roc_figure3/tables/roc_auc_summary.csv`
- `outputs/roc_figure3/tables/roc_curve_points.csv`
- `outputs/roc_figure3/figure_exports/figure3_roc_macro_average.svg`
- `outputs/roc_figure3/figure_exports/figure3_roc_macro_average.pdf`
- `outputs/roc_figure3/figure_exports/figure3_roc_macro_average.png`
- `outputs/roc_figure3/figure_exports/figure3_roc_per_class.svg`
- `outputs/roc_figure3/figure_exports/figure3_roc_per_class.pdf`
- `outputs/roc_figure3/figure_exports/figure3_roc_per_class.png`

`roc_auc_summary.csv` must contain one row per method and ROC view:

- `S0 vs rest`
- `S1 vs rest`
- `S2 vs rest`
- `macro-average`
- `micro-average`

Expected row count: `6 methods * 5 views = 30`.

## Task 1: implement prediction export

Implement `ablation_experiments/scripts/export_figure3_roc_predictions.py`.

Required command-line interface:

```bash
python ablation_experiments/scripts/export_figure3_roc_predictions.py \
  --registry "outputs/figure 3/source_tables/comparison_summary.csv" \
  --output-dir outputs/roc_figure3 \
  --device auto
```

Implementation requirements:

1. Read the registry with `csv.DictReader`. Do not use ad hoc string splitting,
   because `outputs/figure 3` contains a space in the path.
2. Resolve relative `run_path` values against the repository root.
3. Load each run's `run_args.json`.
4. Load the checkpoint from the run directory:
   - `teacher-student(student)`: `best_student_distill.pth`
   - `teacher(18D upper bound)`: `best_single_tcn.pth`
   - `lstm`, `gru`, `transformer`, `inception`: `best_model.pth`
5. Build models from the exact architecture recorded in `run_args.json`.
6. Use `torch.load(..., map_location=device)` and `load_state_dict(...,
   strict=True)`.
7. Run inference only on `X_test_full` and `y_test`.
8. For the 13D student method, use `keep_feature_indices` from `run_args.json`.
   If it is missing, compute it from `drop_feature_indices`.
9. Preserve test order. Use a `DataLoader` with `shuffle=False`.
10. Compute `prob_class_*` with `torch.softmax(logits, dim=1)`.
11. Write a companion summary file:
    `outputs/roc_figure3/tables/prediction_export_summary.csv`.

Model construction guidance:

- For `teacher(18D upper bound)`, reproduce the `tcn_attn` construction from
  `train_single_tcn_classifier.py` with `AttentionTCNClassifier` and
  `parse_channels`.
- For `teacher-student(student)`, build `AttentionTCNClassifier` using
  `student_config_resolved` from the run args. The input dimension must be
  `13`.
- For baselines, reuse `build_model` from
  `ablation_experiments/scripts/train_comparison_18d_baseline.py` with a
  `types.SimpleNamespace` made from the run args.

The script must fail loudly if:

- a run directory is missing;
- a checkpoint is missing;
- a checkpoint cannot load strictly;
- the test sample count is not `356`;
- the class count is not `3`;
- probability rows do not sum to `1.0` within tolerance;
- exported accuracy differs from `best_test_acc` in the registry by more than
  `0.0015`.

## Task 2: implement ROC plotting

Implement `ablation_experiments/scripts/plot_figure3_roc.py`.

Required command-line interface:

```bash
python ablation_experiments/scripts/plot_figure3_roc.py \
  --predictions-dir outputs/roc_figure3/predictions \
  --output-dir outputs/roc_figure3 \
  --title "Figure 3 ROC comparison"
```

Implementation requirements:

1. Read the six prediction CSV files in the fixed Figure 3 order.
2. Verify that all files share identical `sample_index`, `seam_id`,
   `start_idx`, `target_idx`, and `y_true` columns.
3. Compute one-vs-rest ROC curves for classes `0`, `1`, and `2`.
4. Compute micro-average ROC by flattening one-hot labels and probabilities.
5. Compute macro-average ROC by interpolating per-class TPR values over the
   union of all per-class FPR points, then averaging the TPR values.
6. Write AUC values to `roc_auc_summary.csv`.
7. Write plotted ROC coordinates to `roc_curve_points.csv`.
8. Plot one macro-average panel with six curves.
9. Plot one per-class figure with three panels, one for each class.
10. Use a color-blind-safe palette and include AUC values in the legend.

Use `sklearn.metrics.roc_curve` and `sklearn.metrics.auc` if available. If
`sklearn` is not installed, stop with a clear error message instead of silently
using a different implementation.

## Task 3: add tests

Add focused tests for helper logic rather than full GPU inference.

For `tests/test_export_figure3_roc_predictions.py`, test:

- registry path resolution for absolute and relative `run_path` values;
- checkpoint filename selection by `method_name`;
- 13D keep-index fallback from `drop_feature_indices`;
- exported prediction row validation catches non-normalized probabilities;
- exported prediction row validation catches wrong sample counts.

For `tests/test_plot_figure3_roc.py`, test:

- identity-column validation catches mismatched `y_true` order;
- ROC table generation returns `30` AUC rows for six synthetic methods;
- macro-average and micro-average rows are present for every method;
- output filenames are deterministic.

Run the focused tests with:

```bash
python -m pytest \
  tests/test_export_figure3_roc_predictions.py \
  tests/test_plot_figure3_roc.py \
  -q
```

## Task 4: run export and plotting

After tests pass, run the export script:

```bash
python ablation_experiments/scripts/export_figure3_roc_predictions.py \
  --registry "outputs/figure 3/source_tables/comparison_summary.csv" \
  --output-dir outputs/roc_figure3 \
  --device auto
```

Then run the plotting script:

```bash
python ablation_experiments/scripts/plot_figure3_roc.py \
  --predictions-dir outputs/roc_figure3/predictions \
  --output-dir outputs/roc_figure3 \
  --title "Figure 3 ROC comparison"
```

## Task 5: verify results

Verify the outputs with these checks:

1. Confirm six prediction CSV files exist.
2. Confirm each prediction CSV has `356` data rows.
3. Confirm all six files have identical test identity columns.
4. Confirm each probability row sums to `1.0` within `1e-5`.
5. Confirm exported accuracies match Figure 3 registry values within `0.0015`.
6. Confirm `roc_auc_summary.csv` has `30` rows plus a header.
7. Confirm all ROC AUC values are in `[0, 1]`.
8. Confirm SVG and PDF figure exports are non-empty.

Recommended shell checks:

```bash
find outputs/roc_figure3/predictions -name '*_test_predictions.csv' | wc -l
wc -l outputs/roc_figure3/predictions/*_test_predictions.csv
sed -n '1,40p' outputs/roc_figure3/tables/prediction_export_summary.csv
sed -n '1,40p' outputs/roc_figure3/tables/roc_auc_summary.csv
ls -lh outputs/roc_figure3/figure_exports
```

## Acceptance criteria

The work is complete only when all of these conditions hold:

- The focused pytest command passes.
- Six Figure 3 methods have exported test prediction CSVs.
- `prediction_export_summary.csv` reports accuracy values consistent with the
  Figure 3 source table.
- `roc_auc_summary.csv` contains per-class, macro-average, and micro-average
  AUC values for all six methods.
- `figure3_roc_macro_average.svg` and `figure3_roc_macro_average.pdf` exist.
- `figure3_roc_per_class.svg` and `figure3_roc_per_class.pdf` exist.
- No existing Figure 2 or Figure 3 source table is modified.

## Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Wrong architecture reconstruction | High | Build every model from its own `run_args.json` and load checkpoints strictly. |
| Test-set mismatch between methods | High | Validate identity columns across all prediction files before plotting. |
| Path with spaces breaks commands | Medium | Use `Path` objects and `csv.DictReader`; quote CLI paths in docs. |
| Macro ROC definition ambiguity | Medium | Document macro as interpolation over the union of per-class FPR points. |
| ROC figure looks crowded | Medium | Put macro-average in the main figure and per-class ROC in a separate figure. |
| `sklearn` missing | Low | Fail with a clear dependency error before generating partial outputs. |

## Suggested final report

When the coding agent finishes, it must report:

- the created script paths;
- the six prediction file paths;
- the ROC figure paths;
- the focused pytest command and result;
- the exported accuracy values compared with Figure 3 registry values;
- any AUC values that look suspicious, especially values exactly `1.0`.
