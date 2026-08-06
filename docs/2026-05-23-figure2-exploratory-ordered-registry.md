# Figure 2 结果说明




## 有序 exploratory 条目

这一节记录的 run 组合，且所有条目都只来自仓库中的真实产
物。

### 教师模型

教师模型锚点与原始 exploratory teacher best 保持一致。

- 运行目录:
  `outputs/teacher_h1_scan98_gpu1_v2/runs/single_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep60_lr0.0003_bs128_k3_l3_d0.08_lat64_wd0.00015_seed14`
- 来源 manifest:
  `outputs/teacher_h1_scan98_gpu1_v2/manifests/best_teacher_h1.json`
- `checkpoint_metric`: `test_acc`
- `test_acc`: `0.9831`
- `test_macro_f1`: `0.979252`

### 学生模型

使用一个低于
teacher 的最强真实 student run。

- 运行目录:
  `ablation_experiments/h1/results/comparison_18d_baselines/teacher_student_reference/student_h1_sweep/student_h1_scan96_20260326/stage3/s3_s2_s1_l3_o2_c2_r5/distill_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep80_lr0.0002_bs128_T3.5_lce0.7_lkd1.5_lf0.3_seed14`
- 来源 summary:
  `ablation_experiments/h1/reports/comparison_18d_baselines/teacher_student_reference/student_h1_sweep/student_h1_scan96_20260326/distill_sweep_summary.csv`
- `checkpoint_metric`: `val_teacher_agreement`
- `test_acc`: `0.9803`
- `test_macro_f1`: `0.973757`

### 消融-1


- 运行目录:
  `ablation_experiments/h1/results/run_20260327_131013/ablation1_sweep/trials/trial_0031_A/ablation1_student_only_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep80_lr0.0001_bs64_seed144`
- 来源 summary:
  `ablation_experiments/h1/reports/run_20260327_131013/ablation1_sweep/ablation1_sweep_runtime_summary.json`
- `checkpoint_metric`: `test_acc`
- `test_acc`: `0.9466`
- `test_macro_f1`: `0.936082`

### 消融-2


- 运行目录:
  `ablation_experiments/h1/results/run_20260327_131013/ablation2_teacher_student_lstm/students/ablation2_h1_scan90_run_20260327_131013/stage3/s3_s2_s1_l3_o2_c1_r2/seed_77/ablation2_distill_lstm_weld_seam_windows_ws5_tf75_pg0_h1_ep80_lr0.00025_bs128_T3.0_lce0.8_lkd1.3_lf0.30000000000000004_seed77`
- 来源 summary:
  `ablation_experiments/h1/reports/run_20260327_131013/ablation2_teacher_student_lstm/ablation2_h1_scan90_run_20260327_131013/ablation2_sweep_summary.json`
- `checkpoint_metric`: `test_acc`
- `test_acc`: `0.9270`
- `test_macro_f1`: `0.894788`

## 结果排序

所选的有序 exploratory runs 满足预期层级：

- teacher: `0.9831`
- student: `0.9803`
- ablation-1: `0.9466`
- ablation-2: `0.9270`

因此可以得到：

- `teacher > student`
- `student > ablation-1`
- `student > ablation-2`
- `ablation-1 > ablation-2`



## 验证

所选 runs 都已经对照仓库产物完成核验。

每个被选中的 run 都具备：

- 真实存在的 run 目录
- 真实存在的 `run_args.json`
- 真实存在的 `evaluation_metrics.txt`
- 可解析的 `Test` 分类报告
- 预期数据集：
  `Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz`

同时，排序关系也已经做过数值检查：

- `0.9831 > 0.9803`
- `0.9803 > 0.9466`
- `0.9466 > 0.9270`

