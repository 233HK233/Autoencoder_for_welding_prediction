# Figure 2 distillation ablation plan

This document defines how to draw the proposed Figure 2, "Distillation
components and training strategy ablation," and identifies which additional
experiments are required before the figure can support the paper's method
claims. The figure must prove the value of attention, logit knowledge
distillation, latent alignment, and a frozen teacher separately.

## Bottom line

The current repository can support a draft version of this figure, but it
cannot yet support the full paper claim without more experiments. The available
results cover the full SEAL-Weld student, student-only training, LSTM-based
distillation, joint teacher-student training, hyperparameter response surfaces,
and training dynamics. The current results do not contain strict matched
ablations for `w/o attention`, `w/o logit KD`, or `latent alignment only`.

For the paper version, add the missing matched ablations before drawing final
Figure 2. A matched ablation means changing one factor while keeping the teacher,
dataset, seed policy, optimizer, training schedule, and other loss weights
fixed.

## Figure purpose

Figure 2 must answer three reviewer questions.

- Does each method component help, especially on the transition class `S1`?
- Is the reported gain robust across reasonable distillation hyperparameters?
- Does frozen-teacher distillation train differently from student-only and
  jointly trained teacher-student optimization?

Do not use accuracy as the only metric. Accuracy hides important differences
on `S1`, which is the transition class and the main class where distillation
can improve temporal-state discrimination.

## Recommended layout

Use a `3x3` layout. Each row has a different evidential role.

### Row 1: Component ablation

This row compares final test-set behavior across model variants. Use bars or
lollipop bars with `Full SEAL-Weld` highlighted and unavailable variants marked
as `N/A`, not estimated.

- Panel `(a)`: test Macro-F1.
- Panel `(b)`: test `S1` recall.
- Panel `(c)`: severe cross-stage error rate.

The severe cross-stage error rate is:

```text
count(abs(y_true - y_pred) >= 2) / count(test samples)
```

For the ordered labels `S0=0`, `S1=1`, and `S2=2`, this captures direct
`S0 <-> S2` mistakes. It is more relevant than generic error rate because it
penalizes jumps across the transition class.

### Row 2: Hyperparameter response surfaces

This row shows that the selected result is not a single lucky hyperparameter
point. Use heatmaps with numeric cell annotations.

- Panel `(d)`: `Temperature x lambda_kd` heatmap for test Macro-F1.
- Panel `(e)`: `lambda_kd x lambda_feat` heatmap for test `S1` recall.
- Panel `(f)`: `Temperature x lambda_feat` heatmap for test teacher agreement.

Use only valid sweep rows with real test metrics. Exclude rows where metrics
are `-1` or missing.

### Row 3: Training dynamics

This row explains why the training strategy matters. Plot `Full SEAL-Weld`,
`Student-only`, and `Joint teacher-student` on the same axes where the metric is
shared.

- Panel `(g)`: CE loss curve.
- Panel `(h)`: KD loss and latent-MSE curve.
- Panel `(i)`: validation Macro-F1 and teacher agreement curve.

For `Student-only`, plot its supervised training loss as the CE-equivalent
curve and omit KD, latent-MSE, and teacher-agreement curves unless those values
are actually exported.

## Current data status

The table below separates results that are already available from results that
must be added before the paper version.

| Figure item | Current status | Action |
| --- | --- | --- |
| `Full SEAL-Weld` | Available | Use the ordered full student run. |
| `CE only / Student-only` | Available | Use the ordered ablation-1 student-only run. |
| `LSTM-based KD` | Available | Use the ordered ablation-2 LSTM distillation run. |
| `Jointly trained Teacher-Student` | Available | Use the best valid ablation-3 joint-training run, but align protocol before final paper use. |
| `w/o latent alignment` | Partially available | Current `lambda_feat=0` sweep rows exist, but a strict matched rerun is recommended. |
| `logit KD only` | Partially available | Same evidence as `lambda_feat=0`; strict matched rerun is recommended. |
| `w/o attention` | Missing | Add a student architecture ablation without attention. |
| `w/o logit KD` | Missing | Add `lambda_kd=0` while keeping latent alignment active. |
| `latent alignment only` | Missing | Add `lambda_kd=0` and `lambda_feat>0`, with the same CE setting as the full model. |
| Severe cross-stage error for all methods | Partially available | Export per-sample predictions or confusion matrices for every plotted method. |

## Known available result sources

Use these files as the starting point for the draft figure.

- Ordered run registry:
  `outputs/paper_figures/h1_results/data/exploratory_ordered_run_registry.csv`
- Ordered class metrics:
  `outputs/paper_figures/h1_results/data/exploratory_ordered_class_metrics_summary.csv`
- Student distillation sweep:
  `ablation_experiments/h1/reports/comparison_18d_baselines/teacher_student_reference/student_h1_sweep/student_h1_scan96_20260326/distill_sweep_progress.csv`
- Joint-training sweep:
  `ablation_experiments/h1/reports/run_20260327_131013/ablation3_joint_tcn_attn/ablation3_h1_scan90_run_20260327_131013_rerun/ablation3_h1_sweep_progress.csv`
- Existing confusion matrices:
  `outputs/paper_figures/figure1_performance_error_structure_v02/source_tables/source_confusion_matrices.csv`
- Existing prediction exports:
  `outputs/roc_figure3/predictions/teacher_student_student_test_predictions.csv`
  and
  `outputs/paper_figures/figure1_performance_error_structure_v02/source_predictions/student_only_13d_test_predictions.csv`

## Anchor runs for the draft figure

These runs are useful for the draft figure because they already exist in the
current repository.

| Method | Run source | Key current metric |
| --- | --- | --- |
| `Full SEAL-Weld` | `student_h1_scan96_20260326/stage3/s3_s2_s1_l3_o2_c2_r5/...seed14` | Macro-F1 `0.973757`, `S1` recall `1.0000` |
| `Student-only` | `run_20260327_131013/ablation1_sweep/trials/trial_0031_A/...seed144` | Macro-F1 `0.936082`, `S1` recall `0.8434` |
| `LSTM-based KD` | `run_20260327_131013/ablation2_teacher_student_lstm/...seed_77/...` | Macro-F1 `0.894788`, `S1` recall `0.9518` |
| `Joint teacher-student` | `ablation3_h1_scan90_run_20260327_131013_rerun` best valid rows | Best observed Macro-F1 `0.976478` |

The draft figure can include `N/A` markers for missing component ablations.
The final paper figure must replace those markers with matched experimental
results.

## Required additional experiments

This section lists the minimum experiments needed for a defensible paper
version. Run these under the same dataset and evaluation protocol as the full
SEAL-Weld student.

### Required experiment 1: without attention

This experiment tests whether the attention block contributes beyond the TCN
backbone and distillation losses.

- Student architecture: TCN student without attention.
- Teacher: same frozen teacher checkpoint as the full model.
- Losses: CE + logit KD + latent alignment.
- Hyperparameters: match the full model as closely as implementation allows.
- Required outputs: `evaluation_metrics.txt`, `history.json`, and
  `test_predictions.csv`.

### Required experiment 2: without logit KD

This experiment isolates the contribution of soft teacher logits.

- Student architecture: same TCN-attention student as the full model.
- Teacher: frozen and used for latent features if needed.
- Losses: CE + latent alignment.
- Set `lambda_kd=0`.
- Keep `lambda_feat` equal to the full model.
- Required outputs: `evaluation_metrics.txt`, `history.json`, and
  `test_predictions.csv`.

### Required experiment 3: without latent alignment

This experiment isolates the contribution of feature-space alignment.

- Student architecture: same TCN-attention student as the full model.
- Teacher: same frozen teacher.
- Losses: CE + logit KD.
- Set `lambda_feat=0`.
- Keep `lambda_kd` equal to the full model.
- Required outputs: `evaluation_metrics.txt`, `history.json`, and
  `test_predictions.csv`.

The repository already has `lambda_feat=0` sweep rows, but those rows vary
other hyperparameters. Treat them as exploratory evidence, not as the final
strict component ablation.

### Required experiment 4: latent alignment only

This experiment is needed only if the paper or figure explicitly includes a
`latent alignment only` bar.

- Student architecture: same TCN-attention student as the full model.
- Teacher: same frozen teacher.
- Losses: CE + latent alignment.
- Set `lambda_kd=0`.
- Set `lambda_feat>0`.
- Required outputs: `evaluation_metrics.txt`, `history.json`, and
  `test_predictions.csv`.

If space is limited, the main paper can omit this variant and keep it in the
appendix. The main component claim is already covered by `w/o logit KD` and
`w/o latent alignment`.

### Required experiment 5: aligned joint-training comparison

This experiment tests the frozen-teacher design choice.

- Train teacher and student jointly under the same dataset split.
- Use the same student architecture as the full model.
- Use the same evaluation script and test set as full SEAL-Weld.
- Record whether the teacher is frozen at each epoch.
- Required outputs: `evaluation_metrics.txt`, `history.json`, and
  `test_predictions.csv`.

Existing ablation-3 results are useful, but the final paper comparison should
use a protocol-aligned run or clearly state that the joint-training row comes
from the ablation-3 sweep.

## Recommended repeat policy

For a draft figure, one seed is enough to inspect the story. For the final
paper, use at least three seeds for each component ablation if compute permits.

Recommended seed policy:

- Minimum draft: `seed14`.
- Minimum paper: `seed14`, `seed132`, and `seed230`.
- Strong paper version: use the same seed set as the selected full
  SEAL-Weld multi-seed group.

If only one seed is feasible, state that Figure 2 uses matched single-seed
diagnostic ablations and move multi-seed robustness to a supplemental table.

## Required output files per run

Every run used in Figure 2 must export the same evidence files.

- `evaluation_metrics.txt`: final train, validation, and test metrics.
- `history.json`: per-epoch losses, validation Macro-F1, and teacher
  agreement when applicable.
- `test_predictions.csv`: one row per test sample with `y_true`, `y_pred`, and
  class probabilities.
- `run_args.json`: dataset, seed, architecture, and loss weights.

The recommended `test_predictions.csv` columns are:

| Column | Purpose |
| --- | --- |
| `sample_index` | Stable sample key for joins and audits. |
| `seam_id` | Enables per-seam diagnostics. |
| `seam_name` | Human-readable seam identifier. |
| `start_idx` | Window start index. |
| `target_idx` | Future-step target index. |
| `y_true` | Ground-truth class id. |
| `y_pred` | Predicted class id. |
| `prob_class_0` | Predicted probability for `S0`. |
| `prob_class_1` | Predicted probability for `S1`. |
| `prob_class_2` | Predicted probability for `S2`. |

## Plotting implementation notes

Create a new renderer rather than changing the existing composite Figure 2
script.

Recommended script:

```text
render_figure2_distillation_ablation.py
```

Recommended output directory:

```text
outputs/paper_figures/figure2_distillation_ablation_vXX/
```

The renderer should also export source tables:

- `source_tables/component_ablation_metrics.csv`
- `source_tables/hyperparameter_response_macro_f1.csv`
- `source_tables/hyperparameter_response_s1_recall.csv`
- `source_tables/hyperparameter_response_teacher_agreement.csv`
- `source_tables/training_dynamics.csv`
- `qa_notes.txt`

The `qa_notes.txt` file must list every `N/A` item and explain whether it is
missing because the run does not exist, the prediction export is missing, or
the result is exploratory rather than matched.

## Visual design rules

Use a paper-facing Matplotlib style, not spreadsheet-style defaults.

- Export vector formats: `PDF` and `SVG`.
- Export `PNG` only as a preview.
- Use color-blind-safe colors.
- Use a neutral gray for missing `N/A` bars.
- Highlight `Full SEAL-Weld` with the same color in all panels.
- Keep all row-1 metrics in percent units.
- Label heatmap cells with values rounded to one decimal point in percent.
- Use identical x-axis epoch ranges for row-3 training curves.
- Use line style plus color for row-3 methods, so the figure remains readable
  in grayscale.

## Caption draft

Use this caption as the starting point after the missing ablations are complete.

```text
Figure 2. Distillation components and training strategy ablation for
SEAL-Weld. Full frozen-teacher distillation improves Macro-F1 and transition
state recall over student-only training and LSTM-based distillation, while
matched ablations show the separate contributions of attention, logit
distillation, and latent alignment. Hyperparameter response surfaces show that
the improvement is stable across temperature and loss-weight choices. Training
dynamics further show that frozen-teacher distillation provides steadier
validation Macro-F1 and teacher agreement than jointly trained teacher-student
optimization.
```

If any component ablation remains unavailable, revise the caption so it does
not claim that the missing component has been proven.

## Completion checklist

Use this checklist before treating Figure 2 as paper-ready.

- The row-1 bars include all claimed components or mark unavailable components
  as `N/A`.
- `w/o attention` is a true architecture ablation.
- `w/o logit KD` uses `lambda_kd=0`.
- `w/o latent alignment` uses `lambda_feat=0`.
- `latent alignment only`, if shown, uses `lambda_kd=0` and `lambda_feat>0`.
- Full, ablated, student-only, LSTM KD, and joint-training rows use the same
  test protocol.
- Every row-1 method has `S1` recall.
- Every row-1 method has severe cross-stage error computed from predictions or
  confusion matrices.
- Heatmaps exclude failed or placeholder rows.
- Training curves use synchronized epoch axes.
- The figure exports `PDF`, `SVG`, source tables, and `qa_notes.txt`.
