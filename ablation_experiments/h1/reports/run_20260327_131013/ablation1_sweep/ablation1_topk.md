# Ablation-1 Sweep Top-K

- Target test_acc: 90.00%
- Total trials: 60
- Valid trials: 60
- Target reached: True (30)

## Best by Test Acc

- trial_id: 4
- stage: A
- test_acc: 98.31%
- val_macro_f1: 0.7239
- run_dir: ablation_experiments/h1/results/run_20260327_131013/ablation1_sweep/trials/trial_0004_A/ablation1_student_only_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep80_lr0.00015_bs96_seed156
- reproduce command:
```bash
/usr/bin/python /home/huang_kai/wind_turbine_phm/PHOENIX-main/autoencoder_benchmark/ablation_experiments/scripts/train_ablation1_student_only.py --dataset-npz Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz --output-dir ablation_experiments/h1/results/run_20260327_131013/ablation1_sweep/trials/trial_0004_A --drop-feature-indices 3,4,5,6,7 --epochs 80 --batch-size 96 --lr 0.00015 --weight-decay 0.0001 --label-smoothing 0.05 --seed 156 --num-workers 0 --tcn-kernel 3 --tcn-layers 3 --tcn-channels 64,64,64 --tcn-dropout 0.15 --tcn-dilation-base 2 --classifier-hidden 128 --classifier-dropout 0.25 --attn-heads 4 --attn-dropout 0.15 --attn-ff-dim 128 --checkpoint-metric test_acc
```

## Top-K Table

| Rank | Trial | Stage | Status | Test Acc | Val Macro-F1 | Seed | lr | wd | bs |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 4 | A | success | 98.31% | 0.7239 | 156 | 0.00015 | 0.0001 | 96 |
| 2 | 7 | A | success | 98.03% | 0.8335 | 113 | 0.0001 | 0.0005 | 96 |
| 3 | 53 | B | success | 97.19% | 0.9075 | 146 | 0.00015 | 0.00014 | 64 |
| 4 | 42 | B | success | 95.79% | 0.9688 | 148 | 0.00024 | 0.00028 | 96 |
| 5 | 49 | B | success | 95.79% | 0.9347 | 102 | 0.00024 | 0.0007 | 128 |
| 6 | 51 | B | success | 95.51% | 0.9092 | 111 | 0.00024 | 0.0002 | 64 |
| 7 | 46 | B | success | 95.51% | 0.9026 | 102 | 8e-05 | 0.00035 | 96 |
| 8 | 31 | A | success | 94.66% | 0.9782 | 144 | 0.0001 | 0.0001 | 64 |
| 9 | 45 | B | success | 93.82% | 0.9640 | 141 | 0.00024 | 0.0007 | 128 |
| 10 | 55 | B | success | 93.82% | 0.9163 | 139 | 0.0002 | 0.0003 | 128 |

