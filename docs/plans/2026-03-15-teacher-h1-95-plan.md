 # Teacher window_size=5, horizon=1 95%+ plan (isolated outputs)

  ## Summary

  - 目标：在 --window-size 5 --horizon 1 数据集上重训 teacher，达到 test_accuracy > 95%（最佳单次达标）。
  - 约束：尽量复用现有训练代码，不改 train_single_tcn_classifier.py 核心。
  - 新要求（已纳入）：所有新实验结果必须写入全新目录，与原分类任务结果完全隔离。

  ## Output isolation (hard rule)

  - 统一新根目录：outputs/teacher_h1_forecast_95/
  - 目录结构固定：
      - outputs/teacher_h1_forecast_95/runs/：所有 teacher 训练 run
      - outputs/teacher_h1_forecast_95/analysis/：汇总 CSV/JSON
      - outputs/teacher_h1_forecast_95/logs/：批量运行日志
      - outputs/teacher_h1_forecast_95/manifests/：数据与最优配置清单
  - 禁止写入旧目录：outputs/single_tcn/、outputs/distill_single_tcn/
  - 验收必须包含：本次新增 run 全部位于 outputs/teacher_h1_forecast_95/

  ## Key changes

  1. 数据冻结（单一数据源）

  - 用 prepare_weld_seam_dataset_forecast.py 生成并固定：
      - Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz
  - 本轮全部 teacher run 只使用该数据文件。

  2. 新增最小自动化脚本（仅 orchestration）

  - 新增 run_teacher_h1_sweep.py（不改训练器核心）：
      - 默认 --dataset-npz Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz
      - 默认 --output-root outputs/teacher_h1_forecast_95
      - 默认模型骨干 tcn_attn
      - 分阶段执行：coarse、focus、seed、fallback
      - 每次调用都显式传 --output-dir outputs/teacher_h1_forecast_95/runs
      - 每轮写 summary.csv/json 到 analysis/

  3. 搜索策略（高预算，决策固定）

  - coarse（18 runs）
      - 固定：epochs=80, batch=128, weighted-sampler, class-weights=auto, checkpoint-metric=val_macro_f1, early-stop=16, min-epochs=16
      - 网格：lr={2e-4,2.5e-4,3e-4} × layers={2,3} × dropout={0.10,0.12,0.15}
  - focus（8 runs）
      - 以 coarse 前2配置为中心，扩展 weight_decay={1e-4,2e-4,3e-4,5e-4}、label_smoothing={0.0,0.05}
  - seed（10 runs）
      - 最优2配置，各跑 seeds {14,21,42,77,183}


  - 用 analyze_single_tcn_results.py 仅扫描 outputs/teacher_h1_forecast_95/runs
  - 输出：
      - outputs/teacher_h1_forecast_95/analysis/teacher_h1_ge95.csv
      - outputs/teacher_h1_forecast_95/analysis/teacher_h1_ge95.json
  - 在 manifests/best_teacher_h1.json 固化：
      - best run 路径
      - 对应 run_args.json
      - evaluation_metrics.txt 核心指标

  ## Test plan / acceptance

  1. 数据验收

  - NPZ 包含 target_horizon_steps=1
  - 满足 target_idx - start_idx = 5

  2. 训练验收

  - 所有新 run 都在 outputs/teacher_h1_forecast_95/runs
  - 每个 run 完整产物：run_args.json, history.json, evaluation_metrics.txt, best_single_tcn.pth
  - checkpoint 选择统一 val_macro_f1

  3. 目标验收

  - 至少 1 个 run test_accuracy > 95%
  - 生成 analysis 和 manifests 两类最终文件
  - 零混淆检查：旧目录无本轮新增 run

  ## Assumptions

  - 采样率 100Hz 已确认，horizon=1 对应 +0.01s
  - 本轮只做 teacher，不启动 distill
  - “严格协议”按当前代码可执行版本落地：训练决策按 val_macro_f1，测试指标仅用于最终达标判断