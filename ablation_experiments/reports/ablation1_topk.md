# Ablation-1 Sweep Top-K

- Target test_acc: 90.00%
- Total trials: 22
- Valid trials: 22
- Target reached: True (1)

## Best by Test Acc

- trial_id: 22
- stage: A
- test_acc: 91.58%
- val_macro_f1: 1.0000
- run_dir: /home/huang_kai/wind_turbine_phm/PHOENIX-main/autoencoder_benchmark/ablation_experiments/results/ablation1_sweep/trials/trial_0022_A/ablation1_student_only_tcn_attn_weld_seam_windows_ws5_tf75_pg0_ep80_lr0.0002_bs128_seed122
- reproduce command:
```bash
/usr/bin/python /home/huang_kai/wind_turbine_phm/PHOENIX-main/autoencoder_benchmark/ablation_experiments/scripts/train_ablation1_student_only.py --dataset-npz /home/huang_kai/wind_turbine_phm/PHOENIX-main/autoencoder_benchmark/Data/processed_data/weld_seam_windows_ws5_tf75_pg0.npz --output-dir /home/huang_kai/wind_turbine_phm/PHOENIX-main/autoencoder_benchmark/ablation_experiments/results/ablation1_sweep/trials/trial_0022_A --drop-feature-indices 3,4,5,6,7 --epochs 80 --batch-size 128 --lr 0.0002 --weight-decay 0.0003 --label-smoothing 0.0 --seed 122 --num-workers 0 --tcn-kernel 3 --tcn-layers 4 --tcn-channels 96,96,96 --tcn-dropout 0.12 --tcn-dilation-base 2 --classifier-hidden 128 --classifier-dropout 0.35 --attn-heads 4 --attn-dropout 0.15 --attn-ff-dim 128 --checkpoint-metric val_macro_f1 --weighted-sampler
```

## Top-K Table

| Rank | Trial | Stage | Status | Test Acc | Val Macro-F1 | Seed | lr | wd | bs |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 22 | A | success | 91.58% | 1.0000 | 122 | 0.0002 | 0.0003 | 128 |
| 2 | 10 | A | success | 87.77% | 1.0000 | 146 | 0.0001 | 0.0001 | 96 |
| 3 | 16 | A | success | 86.68% | 1.0000 | 120 | 0.0004 | 0.0002 | 128 |
| 4 | 1 | A | success | 86.41% | 1.0000 | 154 | 0.0003 | 0.0002 | 128 |
| 5 | 7 | A | success | 82.61% | 1.0000 | 113 | 0.0001 | 0.0005 | 96 |
| 6 | 14 | A | success | 82.61% | 1.0000 | 144 | 0.0004 | 0.0002 | 64 |
| 7 | 18 | A | success | 82.61% | 1.0000 | 117 | 0.00015 | 0.0001 | 64 |
| 8 | 20 | A | success | 82.34% | 0.9944 | 155 | 0.00025 | 0.0001 | 96 |
| 9 | 5 | A | success | 81.79% | 1.0000 | 105 | 0.00025 | 0.0005 | 64 |
| 10 | 13 | A | success | 81.25% | 1.0000 | 131 | 0.00015 | 0.0001 | 128 |

