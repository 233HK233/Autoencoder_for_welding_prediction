# Figure 2 exploratory gap-fill results

This document records the exploratory Figure 2 gap-fill execution completed on
May 23, 2026. It follows the plan in
`docs/plans/2026-05-23-figure2-exploratory-gapfill-plan.md`.

## Scope

This batch targets the exploratory Figure 2 source only. The goal is to reuse
the strongest exploratory-compatible runs already present in the repository and
rerun only if a required slot is missing a usable metrics artifact.

The five frozen exploratory slots are:

1. `Exploratory Teacher Best`
2. `Best Student Distill`
3. `Best Ablation-1`
4. `Best Ablation-2`
5. `Best GRU baseline`

## Gap-fill outcome

The repository state on May 23, 2026 already satisfied the reuse-first rule
for all five default exploratory slots.

Each slot was verified to have:

- a real run directory
- a real `run_args.json`
- a real `evaluation_metrics.txt`
- a parsable `Test` classification report
- the expected dataset:
  `Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz`

The final gap count was:

- missing model slots: `0`
- missing `run_args.json`: `0`
- missing `evaluation_metrics.txt`: `0`
- missing parsable class reports: `0`

Because no slot failed validation, no reruns were launched on `GPU0` or
`GPU1`.

## Frozen exploratory entries

This section records the exploratory entries frozen for Figure 2.

### Exploratory teacher best

- Source manifest:
  `outputs/teacher_h1_scan98_gpu1_v2/manifests/best_teacher_h1.json`
- Run dir:
  `outputs/teacher_h1_scan98_gpu1_v2/runs/single_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep60_lr0.0003_bs128_k3_l3_d0.08_lat64_wd0.00015_seed14`
- `checkpoint_metric`: `test_acc`
- `best_test_acc`: `0.9831`

### Best student distill

- Source summary:
  `ablation_experiments/h1/reports/comparison_18d_baselines/checkpoints/round3/comparison_summary.csv`
- Run dir:
  `ablation_experiments/h1/results/comparison_18d_baselines/teacher_student_reference/student_h1_sweep/student_h1_scan96_20260326/extension/extension_s3_s2_s1_l4_o4_c7_r3/distill_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep80_lr0.0003_bs128_T2.5_lce0.9_lkd1.1_lf0.0_seed132`
- `checkpoint_metric`: `val_teacher_agreement`
- `best_test_acc`: `0.9888`

### Best ablation-1

- Source summary:
  `ablation_experiments/h1/reports/run_20260327_131013/final_summary/ablation_summary.csv`
- Run dir:
  `ablation_experiments/h1/results/run_20260327_131013/ablation1_sweep/trials/trial_0004_A/ablation1_student_only_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep80_lr0.00015_bs96_seed156`
- `checkpoint_metric`: `test_acc`
- `best_test_acc`: `0.9831`

### Best ablation-2

- Source summary:
  `ablation_experiments/h1/reports/run_20260327_131013/final_summary/ablation_summary.csv`
- Run dir:
  `ablation_experiments/h1/results/run_20260327_131013/ablation2_teacher_student_lstm/students/ablation2_h1_scan90_run_20260327_131013/stage3/s3_s2_s1_l3_o2_c1_r2/seed_230/ablation2_distill_lstm_weld_seam_windows_ws5_tf75_pg0_h1_ep80_lr0.00025_bs128_T3.0_lce0.8_lkd1.3_lf0.30000000000000004_seed230`
- `checkpoint_metric`: `test_acc`
- `best_test_acc`: `0.9944`

### Best GRU baseline

- Source summary:
  `ablation_experiments/h1/reports/comparison_18d_baselines/checkpoints/round3/comparison_summary.csv`
- Run dir:
  `ablation_experiments/h1/results/comparison_18d_baselines/gru/gru_weld_seam_windows_ws5_tf75_pg0_h1_h64_l3_uni_do0.3_ep80_lr0.0001_wd0.0001_bs128_seed14`
- `checkpoint_metric`: `val_macro_f1`
- `best_test_acc`: `0.9157`

## Deliverables

The exploratory Figure 2 source is now frozen in these files:

- Registry:
  `outputs/paper_figures/h1_results/data/exploratory_run_registry.csv`
- Class metrics:
  `outputs/paper_figures/h1_results/data/exploratory_class_metrics_summary.csv`

The registry contains `5` rows. The class-metrics file contains `45` rows:

- `5` models
- `3` classes
- `3` metrics per class: `precision`, `recall`, and `f1`

## Implementation notes

The repository did not previously contain an exploratory Figure 2 export
utility. This batch added:

- `ablation_experiments/scripts/build_figure2_exploratory_registry.py`
- `tests/test_build_figure2_exploratory_registry.py`

The export utility:

- reuses the exploratory teacher manifest
- reuses the exploratory comparison summary
- reuses the exploratory ablation summary
- defaults to the stronger `GRU` baseline
- supports `--baseline-method lstm` as the fallback export choice

## Verification

The exploratory freeze was verified in the following ways:

- direct artifact checks confirmed all five default slots were reusable
- the new export test suite passed:
  `python -m unittest tests.test_build_figure2_exploratory_registry`
- the existing strict export test suite still passed:
  `python -m unittest tests.test_build_figure2_official_registry`
- the real exploratory export completed successfully
- final output counts were checked:
  - `exploratory_run_registry.csv`: `5` rows
  - `exploratory_class_metrics_summary.csv`: `45` rows

## Next steps

You can now point the Figure 2 plotting workflow directly at:

- `outputs/paper_figures/h1_results/data/exploratory_run_registry.csv`
- `outputs/paper_figures/h1_results/data/exploratory_class_metrics_summary.csv`

If you want architectural continuity with the strict Figure 2 baseline slot,
rerun the export once with:

`python ablation_experiments/scripts/build_figure2_exploratory_registry.py --baseline-method lstm`
