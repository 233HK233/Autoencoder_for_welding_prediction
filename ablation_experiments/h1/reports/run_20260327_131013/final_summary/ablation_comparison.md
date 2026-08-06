# Ablation Comparison Report

## Group Summary

| Group | N | Mean Acc | Std Acc | Mean Macro-F1 | Std Macro-F1 | Mean Teacher Agreement | Best Seed | Best Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ablation1_student_only_tcn_attn | 60 | 89.84% | 4.13% | 0.8701 | 0.0533 | N/A | 156 | 98.31% |
| ablation2_teacher_student_lstm | 56 | 90.19% | 2.59% | 0.8687 | 0.0363 | 94.96% | 230 | 99.44% |
| ablation3_joint_teacher_student_tcn_attn | 40 | 94.82% | 3.91% | 0.9310 | 0.0469 | 94.88% | 14 | 98.31% |

## Notes

- Ablation-1: Student-only (13D, TCN+Attention) future-step prediction
- Ablation-2: Teacher+Student distillation with LSTM backbone for future-step prediction
- Ablation-3: Joint online teacher-student distillation with TCN+Attention for future-step prediction
- Metrics are parsed from each run's evaluation_metrics.txt
