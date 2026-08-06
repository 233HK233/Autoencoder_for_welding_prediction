# Figure 2 exploratory gap-fill plan

This document defines the exploratory-protocol plan for a second Figure 2
result source. The goal is not to retrain everything again. The goal is to
reuse the best existing exploratory-compatible runs wherever they already
exist, and only rerun experiments if a Figure 2 slot is still missing a usable
run or a usable metrics file.

This plan is intentionally different from the official strict Figure 2 plan.
The strict plan freezes one publication-safe protocol. This exploratory plan
freezes one high-score comparison source that is allowed to inherit the
selection logic already used in the exploratory result pools.

## Summary

You want an exploratory Figure 2 that uses the strongest available results in
the current repository, not a fully unified validation-only protocol. Under
that rule, the repository already contains usable high-score runs for all core
Figure 2 model families:

- exploratory teacher
- exploratory student distill
- exploratory ablation-1
- exploratory ablation-2
- baseline

The working assumption of this plan is:

- if a model family already has an exploratory-compatible best run with a real
  `evaluation_metrics.txt` test classification report, reuse it;
- if a slot lacks a usable run artifact or lacks a parsable class-wise test
  report, rerun only that slot;
- if two baseline candidates are available, prefer the stronger one by
  `best_test_acc` unless the paper story explicitly needs a specific baseline
  family.

Under the current repository state, the expected rerun count is likely `0`.
The main work is to freeze an exploratory registry and exploratory class-metric
table.

## Scope

This plan targets only Figure 2. It does not attempt to redefine the paper's
main official protocol.

The exploratory Figure 2 source will be built from these five comparison slots:

1. `Exploratory Teacher Best`
2. `Best Student Distill`
3. `Best Ablation-1`
4. `Best Ablation-2`
5. `Best baseline`

The fifth slot has two baseline choices already available:

- `Best GRU baseline`
- `Best LSTM baseline`

By raw test accuracy, the current exploratory pool favors `GRU` over `LSTM`.
Therefore, this plan treats `Best GRU baseline` as the default exploratory
baseline unless you explicitly want architectural continuity with the strict
Figure 2 version.

## Exploratory protocol definition

This plan defines `exploratory` operationally, using the current repository's
existing high-score pools.

An exploratory Figure 2 entry is valid if all of the following hold:

- it uses the same dataset:
  `Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz`
- it solves the same task:
  `future-step state classification`
- it has a real run directory with `run_args.json`
- it has a real `evaluation_metrics.txt`
- its `evaluation_metrics.txt` contains a parsable `Test` classification report
- its run belongs to an exploratory or exploratory-compatible result pool

Unlike the strict version, this plan does **not** require:

- `checkpoint_metric=val_macro_f1`
- a strict teacher source
- validation-only orchestration ranking

This means the exploratory Figure 2 is a high-score comparison figure, not a
single-protocol fairness figure.

## Reuse-first rule

You must decide each Figure 2 slot with the following order:

1. Look for an existing best run in the current repository.
2. Verify that the run has `run_args.json` and `evaluation_metrics.txt`.
3. Verify that the `Test` classification report is parsable into 9 class-metric
   rows.
4. Reuse that run if all checks pass.
5. Rerun only if one of those checks fails.

This rule applies slot by slot. Do not launch a full sweep simply because one
slot is missing.

## Current best available candidates

This section records the current best candidates already present in the
repository.

### Slot 1: exploratory teacher

- Source:
  `outputs/teacher_h1_scan98_gpu1_v2/manifests/best_teacher_h1.json`
- Current best run:
  `outputs/teacher_h1_scan98_gpu1_v2/runs/single_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep60_lr0.0003_bs128_k3_l3_d0.08_lat64_wd0.00015_seed14`
- `best_test_acc`: `0.9831`
- `checkpoint_metric`: `test_acc`
- Status: reusable now

### Slot 2: exploratory student distill

- Source summary:
  `ablation_experiments/h1/reports/comparison_18d_baselines/checkpoints/round3/comparison_summary.csv`
- Current best run:
  `ablation_experiments/h1/results/comparison_18d_baselines/teacher_student_reference/student_h1_sweep/student_h1_scan96_20260326/extension/extension_s3_s2_s1_l4_o4_c7_r3/distill_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep80_lr0.0003_bs128_T2.5_lce0.9_lkd1.1_lf0.0_seed132`
- `best_test_acc`: `0.9888`
- `checkpoint_metric`: `val_teacher_agreement`
- Status: reusable now

### Slot 3: exploratory ablation-1

- Source summary:
  `ablation_experiments/h1/reports/run_20260327_131013/final_summary/ablation_summary.csv`
- Current best run:
  `ablation_experiments/h1/results/run_20260327_131013/ablation1_sweep/trials/trial_0004_A/ablation1_student_only_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep80_lr0.00015_bs96_seed156`
- `best_test_acc`: `0.9831`
- `checkpoint_metric`: `test_acc`
- Status: reusable now

### Slot 4: exploratory ablation-2

- Source summary:
  `ablation_experiments/h1/reports/run_20260327_131013/final_summary/ablation_summary.csv`
- Current best run:
  `ablation_experiments/h1/results/run_20260327_131013/ablation2_teacher_student_lstm/students/ablation2_h1_scan90_run_20260327_131013/stage3/s3_s2_s1_l3_o2_c1_r2/seed_230/ablation2_distill_lstm_weld_seam_windows_ws5_tf75_pg0_h1_ep80_lr0.00025_bs128_T3.0_lce0.8_lkd1.3_lf0.30000000000000004_seed230`
- `best_test_acc`: `0.9944`
- `checkpoint_metric`: `test_acc`
- Status: reusable now

### Slot 5: exploratory baseline

Two baseline candidates are already usable:

- `Best GRU baseline`
  - source summary:
    `ablation_experiments/h1/reports/comparison_18d_baselines/checkpoints/round3/comparison_summary.csv`
  - run:
    `ablation_experiments/h1/results/comparison_18d_baselines/gru/gru_weld_seam_windows_ws5_tf75_pg0_h1_h64_l3_uni_do0.3_ep80_lr0.0001_wd0.0001_bs128_seed14`
  - `best_test_acc`: `0.9157`
  - `checkpoint_metric`: `val_macro_f1`

- `Best LSTM baseline`
  - source summary:
    `ablation_experiments/h1/reports/comparison_18d_baselines/checkpoints/round3/comparison_summary.csv`
  - run:
    `ablation_experiments/h1/results/comparison_18d_baselines/lstm/lstm_weld_seam_windows_ws5_tf75_pg0_h1_h64_l2_bi_do0.3_ep80_lr0.0001_wd1e-05_bs128_seed42`
  - `best_test_acc`: `0.9073`
  - `checkpoint_metric`: `val_macro_f1`

Default exploratory choice:

- use `Best GRU baseline`

Fallback choice:

- use `Best LSTM baseline` if you want direct visual comparability with the
  strict Figure 2 baseline slot.

## Gap analysis

Under the current repository state, all five default exploratory slots already
have:

- a run directory
- a `run_args.json`
- an `evaluation_metrics.txt`
- a parsable `Test` classification report

That means the current expected gap count is:

- missing model slots: `0`
- missing metrics files: `0`
- missing parsable class reports: `0`

Therefore, the default plan is:

- do not rerun teacher
- do not rerun student
- do not rerun ablation-1
- do not rerun ablation-2
- do not rerun baselines

## Rerun triggers

Only rerun a slot if at least one of the following becomes true during export:

- the referenced run directory no longer exists
- `run_args.json` is missing
- `evaluation_metrics.txt` is missing
- the `Test` classification report cannot be parsed into class-wise metrics
- the slot is intentionally replaced by a different family choice

### Trigger T1: exploratory teacher missing

If the exploratory teacher manifest or best run is unusable, rerun:

- script: `run_teacher_h1_sweep.py`
- output root: `outputs/teacher_h1_scan98_gpu1_v2/`
- selection logic: keep existing exploratory behavior

### Trigger T2: exploratory student missing

If the exploratory best student run is unusable, rerun:

- script:
  `ablation_experiments/scripts/run_h1_distill_comparison_sweep.py`
- teacher manifest:
  `outputs/teacher_h1_scan98_gpu1_v2/manifests/best_teacher_h1.json`
- experiment tag:
  reuse `student_h1_scan96_20260326` logic or write a new exploratory tag
- selection logic:
  keep existing exploratory behavior, including the current ranking regime

### Trigger T3: exploratory ablation-1 missing

If the exploratory best ablation-1 run is unusable, rerun:

- script: `ablation_experiments/scripts/run_ablation1_sweep.py`
- output root:
  `ablation_experiments/h1/results/run_20260327_131013/ablation1_sweep/`
  pattern or a new exploratory-tagged sibling
- selection logic:
  keep the exploratory `test_acc`-oriented behavior

### Trigger T4: exploratory ablation-2 missing

If the exploratory best ablation-2 run is unusable, rerun:

- script: `ablation_experiments/scripts/run_ablation2_sweep.py`
- teacher reference:
  keep the existing exploratory LSTM-teacher reference
- selection logic:
  keep the exploratory `test_acc`-oriented behavior

### Trigger T5: exploratory baseline slot replacement

If you decide not to use `GRU`, then the fallback slot is:

- `Best LSTM baseline`

No rerun is needed unless both baseline candidates become unusable.

## Figure 2 slot policy

This plan recommends one default exploratory lineup and one compatibility
lineup.

### Default exploratory lineup

Use this lineup if your priority is strongest available results:

1. `Exploratory Teacher Best`
2. `Best Student Distill`
3. `Best Ablation-1`
4. `Best Ablation-2`
5. `Best GRU baseline`

### Compatibility lineup

Use this lineup if your priority is side-by-side comparability with the strict
Figure 2 narrative:

1. `Exploratory Teacher Best`
2. `Best Student Distill`
3. `Best Ablation-1`
4. `Best Ablation-2`
5. `Best LSTM baseline`

The only difference is the baseline family in slot 5.

## Required export changes

The current export script
`ablation_experiments/scripts/build_figure2_official_registry.py` is strict-only.
It hardcodes:

- strict default inputs
- strict display names
- strict protocol labels
- the strict LSTM baseline choice

Before exporting exploratory Figure 2 tables, you must do one of the following:

- extend the existing script with `--mode strict|exploratory`, or
- create a sibling script such as
  `ablation_experiments/scripts/build_figure2_exploratory_registry.py`

The second option is safer if you want to avoid touching the already-frozen
strict export path.

## Required deliverables

The exploratory Figure 2 workflow must produce two tables, parallel to the
strict version but stored separately.

### Deliverable D1: exploratory registry

Write:

`outputs/paper_figures/h1_results_exploratory/data/exploratory_run_registry.csv`

Required columns:

| 字段名 | 含义 |
| --- | --- |
| `model_name` | Figure 2 图例名称 |
| `model_family` | teacher / student / ablation / baseline |
| `protocol` | 固定写 `exploratory` |
| `run_dir` | 来源 run |
| `teacher_source` | 仅 distill 或 teacher-derived ablation 需要填写 |
| `checkpoint_metric` | 原样记录当前 run 的选模指标 |
| `selection_note` | `direct reuse` 或 `gap-fill rerun` |
| `included_in_fig2` | `yes/no` |

### Deliverable D2: exploratory class metrics

Write:

`outputs/paper_figures/h1_results_exploratory/data/class_metrics_summary_exploratory.csv`

Required columns:

| 字段名 | 含义 |
| --- | --- |
| `model_name` | 图例名称 |
| `model_family` | teacher / student / ablation / baseline |
| `protocol` | 固定写 `exploratory` |
| `class_id` | `0/1/2` |
| `class_name` | `S0/S1/S2` |
| `metric` | `precision/recall/f1` |
| `value` | 数值 |
| `source_run_dir` | 来源 run |

These rows must be extracted from the `Test` classification report only.

## Suggested execution order

Follow this order to minimize unnecessary training:

1. Freeze the exploratory slot policy:
   choose `GRU` or `LSTM` for the baseline slot.
2. Verify the five chosen run directories still exist.
3. Verify each chosen run has a parsable `evaluation_metrics.txt`.
4. Build the exploratory registry.
5. Build the exploratory class-metrics table.
6. Only if a slot fails verification, rerun that single slot.
7. Rebuild the two export tables after any gap-fill rerun.

## Acceptance checklist

The exploratory plan is complete only if all of the following are true:

- exactly five Figure 2 slots are frozen
- each slot has a real run directory
- each slot has a parsable `Test` classification report
- the exploratory registry has five rows
- the exploratory class-metrics table has `45` rows
- no unnecessary full sweep was launched
- any rerun that did happen is justified by a concrete missing-artifact trigger

## Notes

This exploratory Figure 2 is useful for showing the upper-end empirical
behavior already present in the repository. It is not a fairness-constrained
single-protocol comparison like the strict version.

That means you must not blur the two stories in the paper:

- strict Figure 2:
  fairer protocol control, lower scores
- exploratory Figure 2:
  stronger attainable scores, looser orchestration logic

If both are shown in the same manuscript, label them explicitly as different
result protocols.

## Next steps

The next concrete action is not training. The next concrete action is to build
the exploratory registry and exploratory class-metrics export from the existing
best runs.

Only after that export step fails on a specific slot should you schedule any
new exploratory rerun.
