# Figure 2 strict rerun results

This document records the strict-protocol reruns completed on May 23, 2026 for
the official Figure 2 data source. It covers the protocol used, code changes,
rerun outputs, best-result summaries, and the final files that now serve as the
frozen Figure 2 inputs.

## Scope

This rerun batch follows the plan in
`docs/plans/2026-05-22-figure2-official-unified-rerun.md`. The goal is to
freeze Figure 2 to one protocol and remove the previous mix of exploratory and
strict selection logic.

The rerun scope is:

- `Best Student Distill`
- `Best Ablation-1`
- `Best Ablation-2`

The direct-reuse scope is:

- `Strict Teacher Best` from
  `outputs/teacher_h1_forecast_95/manifests/best_teacher_h1.json`
- `Best LSTM baseline` from the existing 18D baseline pool

## Frozen protocol

All Figure 2 official entries now follow the same protocol:

- Dataset: `Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz`
- Task: `future-step state classification`
- Protocol: `strict`
- Label set: `S0 / S1 / S2`
- Selection metric: `val_macro_f1`
- Evaluation source: raw classifier output, no monotonic postprocess
- 13D crop: `drop_feature_indices=3,4,5,6,7`

The following were explicitly excluded from the official Figure 2 source:

- `test_acc` as the orchestration selection metric
- `val_teacher_agreement` as the orchestration selection metric
- exploratory teacher and exploratory student runs derived from
  `outputs/teacher_h1_scan98_gpu1_v2`

## Code changes

This rerun required three orchestration fixes and one export utility.

### Strict student comparison sweep

The script
`ablation_experiments/scripts/run_h1_distill_comparison_sweep.py` now:

- defaults to
  `outputs/teacher_h1_forecast_95/manifests/best_teacher_h1.json`
- uses `ranking_metric=val_macro_f1`
- removes the `val_teacher_agreement` stage-0 branch
- writes new outputs under
  `ablation_experiments/h1/results/comparison_18d_baselines/teacher_student_reference/student_h1_strict_valf1_20260522/`
- writes new reports under
  `ablation_experiments/h1/reports/comparison_18d_baselines/teacher_student_reference/student_h1_strict_valf1_20260522/`

### Ablation-1 sweep

The script `ablation_experiments/scripts/run_ablation1_sweep.py` now:

- accepts `--checkpoint-metric`
- defaults to `val_macro_f1`
- ranks candidates by validation macro-F1 instead of test accuracy
- disables early stop on `target_acc` by default so the staged budget completes
- writes new outputs under
  `ablation_experiments/h1/results/ablation1_sweep_strict_valf1_20260522/`
- writes new reports under
  `ablation_experiments/h1/reports/ablation1_sweep_strict_valf1_20260522/`

### Ablation-2 sweep

The script `ablation_experiments/scripts/run_ablation2_sweep.py` now:

- accepts `--checkpoint-metric`
- defaults to `val_macro_f1`
- ranks staged candidates by validation macro-F1
- uses the new strict output root
- uses the new strict report root
- fixes stage-2 and stage-3 output paths so they do not duplicate the
  experiment tag

### Figure 2 export

The new script
`ablation_experiments/scripts/build_figure2_official_registry.py` now:

- builds `official_run_registry.csv`
- builds `class_metrics_summary.csv`
- extracts class-wise `precision / recall / f1` from the `Test` classification
  report block only

## Execution summary

The reruns were executed on May 23, 2026 with the following allocation:

- `GPU1`: strict student distillation sweep
- `GPU0`: strict ablation-1 sweep, then strict ablation-2 sweep
- `GPU2` and `GPU3`: left untouched because they were occupied by other work

The completed budgets were:

- Student strict sweep: `108` runs
- Ablation-1 strict sweep: `60` trials
- Ablation-2 strict sweep: `56` runs

## Best results

This section records the strict-protocol best entries selected for the official
Figure 2 source.

### Strict teacher best

- Source manifest:
  `outputs/teacher_h1_forecast_95/manifests/best_teacher_h1.json`
- Run dir:
  `outputs/teacher_h1_forecast_95/runs/single_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep80_lr0.0003_bs128_k3_l2_d0.1_lat64_wd0.0005_seed183`
- `checkpoint_metric`: `val_macro_f1`

### Best student distill

- Summary:
  `ablation_experiments/h1/reports/comparison_18d_baselines/teacher_student_reference/student_h1_strict_valf1_20260522/distill_sweep_summary.json`
- Best run:
  `ablation_experiments/h1/results/comparison_18d_baselines/teacher_student_reference/student_h1_strict_valf1_20260522/extension/extension_s3_s2_s1_l4_o1_c2_r1/distill_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep80_lr0.0001_bs128_T2.0_lce0.9_lkd1.1_lf0.0_seed132`
- `checkpoint_metric`: `val_macro_f1`
- `val_macro_f1`: `1.000000`
- `test_acc`: `0.904500`
- `test_macro_f1`: `0.863536`
- `test_teacher_agreement`: `0.898900`

### Best ablation-1

- Summary:
  `ablation_experiments/h1/reports/ablation1_sweep_strict_valf1_20260522/ablation1_sweep_runtime_summary.json`
- Best run:
  `ablation_experiments/h1/results/ablation1_sweep_strict_valf1_20260522/trials/trial_0037_A/ablation1_student_only_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep80_lr0.0003_bs64_seed128`
- `checkpoint_metric`: `val_macro_f1`
- `val_macro_f1`: `1.000000`
- `test_acc`: `0.887600`
- `test_macro_f1`: `0.860760`

### Best ablation-2

- Summary:
  `ablation_experiments/h1/reports/ablation2_teacher_student_lstm/ablation2_strict_valf1_20260522/ablation2_sweep_summary.json`
- Best run:
  `ablation_experiments/h1/results/ablation2_teacher_student_lstm/students/ablation2_strict_valf1_20260522/ablation2_strict_valf1_20260522/stage2/s2_s1_l4_o3_c0_r1/ablation2_distill_lstm_weld_seam_windows_ws5_tf75_pg0_h1_ep80_lr0.00024999999999999995_bs128_T3.5_lce0.7_lkd1.4_lf0.3_seed14`
- `checkpoint_metric`: `val_macro_f1`
- `val_macro_f1`: `0.986227`
- `test_acc`: `0.893300`
- `test_macro_f1`: `0.853736`
- `test_teacher_agreement`: `0.980300`

### Best LSTM baseline

- Direct reuse run:
  `ablation_experiments/h1/results/comparison_18d_baselines/lstm/lstm_weld_seam_windows_ws5_tf75_pg0_h1_h64_l2_bi_do0.3_ep80_lr0.0001_wd1e-05_bs128_seed42`
- `checkpoint_metric`: `val_macro_f1`
- `test_acc`: `0.907300`

## Final deliverables

The official Figure 2 data source is now frozen in these files:

- Registry:
  `outputs/paper_figures/h1_results/data/official_run_registry.csv`
- Class metrics:
  `outputs/paper_figures/h1_results/data/class_metrics_summary.csv`

The registry contains five official entries:

1. `Strict Teacher Best`
2. `Best Student Distill`
3. `Best Ablation-1`
4. `Best Ablation-2`
5. `Best LSTM baseline`

The class metrics file contains:

- `45` rows total
- `5` models
- `3` classes
- `3` metrics per class: `precision`, `recall`, and `f1`

## Verification

The rerun and export workflow was verified in the following ways:

- Targeted `unittest` coverage was updated and passed for:
  - strict distill sweep defaults and ranking
  - strict ablation-1 ranking and defaults
  - strict ablation-2 ranking and path handling
  - official Figure 2 registry and class-metrics export
- The export utility was checked against the real `Test` classification report
  blocks, not the `Train` or `Val` blocks.
- The final output counts were checked:
  - `official_run_registry.csv`: `5` rows
  - `class_metrics_summary.csv`: `45` rows

## Notes

One `skipped_existing` record appears in the strict distill progress log. The
strict rerun still completed its full `108` run budget, and the best strict
student run selected for the official registry comes from the new strict output
root.

The strict ablation-2 rerun completed successfully, but its best run path still
contains a duplicated `ablation2_strict_valf1_20260522` segment because that
path was produced before the final path cleanup patch. The orchestration script
has now been corrected so future runs do not repeat that nesting.

## Next steps

You can now point the Figure 2 plotting workflow directly at:

- `outputs/paper_figures/h1_results/data/official_run_registry.csv`
- `outputs/paper_figures/h1_results/data/class_metrics_summary.csv`

If you want, the next step can be either:

1. update the Figure 2 plotting script to consume `class_metrics_summary.csv`
   only, or
2. generate a paper-ready Figure 2 from the frozen strict data source.
