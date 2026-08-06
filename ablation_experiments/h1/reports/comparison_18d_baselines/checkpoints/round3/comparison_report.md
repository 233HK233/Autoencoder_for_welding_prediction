# H1 Comparison Report

| Method | Train Scheme | Deploy Input | Teacher Input | Best Test Acc | Best Macro-F1 | Seed | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| teacher-student(student) | frozen_teacher_distill | 13 | 18 | 98.88% | 0.9847 | 132 | 13D deploy student distilled from 18D teacher |
| lstm | baseline_18d | 18 |  | 90.73% | 0.8753 | 42 | Best 18D lstm baseline run |
| gru | baseline_18d | 18 |  | 91.57% | 0.8872 | 14 | Best 18D gru baseline run |
| transformer | baseline_18d | 18 |  | 88.76% | 0.8620 | 14 | Best 18D transformer baseline run |
| inception | baseline_18d | 18 |  | 90.17% | 0.8732 | 14 | Best 18D inception baseline run |
| teacher(18D upper bound) | teacher_only | 18 | 18 | 98.31% | 0.9793 | 14 | Best h1 teacher reference from manifest |
