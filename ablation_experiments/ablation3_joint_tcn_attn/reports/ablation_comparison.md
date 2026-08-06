# Ablation Comparison Report

## Group Summary

| Group | N | Mean Acc | Std Acc | Mean Macro-F1 | Std Macro-F1 | Mean Teacher Agreement | Best Seed | Best Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ablation1_student_only_tcn_attn | 1 | 61.68% | 0.00% | 0.6549 | 0.0000 | N/A | 100 | 61.68% |
| ablation2_teacher_student_lstm | 1 | 37.50% | 0.00% | 0.2852 | 0.0000 | 35.60% | 100 | 37.50% |
| ablation3_joint_teacher_student_tcn_attn | 1 | 37.23% | 0.00% | 0.3926 | 0.0000 | 50.82% | 100 | 37.23% |

## Notes

- Ablation-1: Student-only (13D, TCN+Attention)
- Ablation-2: Teacher+Student distillation with LSTM backbone
- Ablation-3: Joint online teacher-student distillation with TCN+Attention backbone
- Metrics are parsed from each run's evaluation_metrics.txt
