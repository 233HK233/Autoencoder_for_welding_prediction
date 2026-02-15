#!/usr/bin/env python3
"""Optional sweep runner for KD/feature-loss weights."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def parse_metric(metrics_text: str, label: str) -> float:
    pattern = rf"{re.escape(label)}:\s*([0-9.]+)%"
    m = re.search(pattern, metrics_text)
    if not m:
        return -1.0
    return float(m.group(1)) / 100.0


def parse_section_metric(metrics_text: str, section: str, label: str) -> float:
    pattern = rf"{re.escape(section)}.*?{re.escape(label)}:\s*([0-9.]+)%"
    m = re.search(pattern, metrics_text, flags=re.S)
    if not m:
        return -1.0
    return float(m.group(1)) / 100.0


def build_run_name(params: Dict[str, object]) -> str:
    return (
        f"distill_tcn_attn_{params['dataset_tag']}_ep{params['epochs']}_lr{params['lr']}_"
        f"bs{params['batch_size']}_T{params['temperature']}_lce{params['lambda_ce']}_"
        f"lkd{params['lambda_kd']}_lf{params['lambda_feat']}_seed{params['seed']}"
    )


def build_command(params: Dict[str, object], train_script: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(train_script),
        "--dataset-npz",
        str(params["dataset_npz"]),
        "--teacher-ckpt",
        str(params["teacher_ckpt"]),
        "--teacher-run-args",
        str(params["teacher_run_args"]),
        "--output-dir",
        str(params["output_dir"]),
        "--epochs",
        str(params["epochs"]),
        "--batch-size",
        str(params["batch_size"]),
        "--lr",
        str(params["lr"]),
        "--weight-decay",
        str(params["weight_decay"]),
        "--seed",
        str(params["seed"]),
        "--temperature",
        str(params["temperature"]),
        "--lambda-ce",
        str(params["lambda_ce"]),
        "--lambda-kd",
        str(params["lambda_kd"]),
        "--lambda-feat",
        str(params["lambda_feat"]),
        "--checkpoint-metric",
        str(params["checkpoint_metric"]),
        "--val-ratio",
        str(params["val_ratio"]),
        "--early-stop-patience",
        str(params["early_stop_patience"]),
        "--min-epochs",
        str(params["min_epochs"]),
        "--class-weights",
        str(params["class_weights"]),
        "--drop-feature-indices",
        str(params["drop_feature_indices"]),
    ]
    if bool(params["weighted_sampler"]):
        cmd.append("--weighted-sampler")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep lambda_kd and lambda_feat for distillation")
    parser.add_argument("--output-dir", type=str, default="autoencoder_benchmark/outputs/distill_single_tcn/sweep")
    parser.add_argument(
        "--dataset-npz",
        type=str,
        required=True,
    )
    parser.add_argument("--teacher-ckpt", type=str, required=True)
    parser.add_argument("--teacher-run-args", type=str, required=True)
    parser.add_argument("--stop-val-agreement", type=float, default=0.98)
    parser.add_argument("--max-runs", type=int, default=16)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train_distill_single_tcn_student.py"
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_tag = Path(args.dataset_npz).stem
    base_params: Dict[str, object] = {
        "dataset_npz": args.dataset_npz,
        "dataset_tag": dataset_tag,
        "teacher_ckpt": args.teacher_ckpt,
        "teacher_run_args": args.teacher_run_args,
        "output_dir": str(output_dir),
        "epochs": 60,
        "batch_size": 128,
        "lr": 2e-4,
        "weight_decay": 2e-4,
        "seed": 42,
        "temperature": 2.0,
        "lambda_ce": 1.0,
        "checkpoint_metric": "val_teacher_agreement",
        "val_ratio": 0.15,
        "early_stop_patience": 12,
        "min_epochs": 12,
        "class_weights": "auto",
        "drop_feature_indices": "3,4,5,6,7",
        "weighted_sampler": True,
    }

    lambda_kd_values = [0.3, 0.5, 0.7, 1.0]
    lambda_feat_values = [0.0, 0.1, 0.2, 0.3]

    results = []
    run_count = 0

    for lambda_kd in lambda_kd_values:
        for lambda_feat in lambda_feat_values:
            if run_count >= args.max_runs:
                break
            params = dict(base_params)
            params["lambda_kd"] = lambda_kd
            params["lambda_feat"] = lambda_feat
            run_name = build_run_name(params)
            run_dir = output_dir / run_name
            cmd = build_command(params, train_script)

            print("=" * 80)
            print(
                f"Run {run_count + 1}: lambda_kd={lambda_kd}, lambda_feat={lambda_feat}"
            )
            print(f"Output: {run_dir}")
            print("Command:", " ".join(cmd))
            print("=" * 80)

            start = time.time()
            subprocess.run(cmd, check=True)
            elapsed = time.time() - start

            metrics_path = run_dir / "evaluation_metrics.txt"
            val_agreement = -1.0
            test_agreement = -1.0
            test_acc = -1.0
            if metrics_path.exists():
                text = metrics_path.read_text(encoding="utf-8")
                val_agreement = parse_metric(text, "Val Teacher-Agreement")
                test_agreement = parse_metric(text, "Test Teacher-Agreement")
                test_acc = parse_section_metric(
                    text, "--- Test Metrics (Student) ---", "Accuracy"
                )

            entry = {
                "run_name": run_name,
                "lambda_kd": lambda_kd,
                "lambda_feat": lambda_feat,
                "val_teacher_agreement": val_agreement,
                "test_teacher_agreement": test_agreement,
                "test_accuracy": test_acc,
                "elapsed_sec": round(elapsed, 2),
            }
            results.append(entry)
            run_count += 1

            print(
                f"Val agreement={val_agreement:.4f}, "
                f"Test agreement={test_agreement:.4f}, "
                f"Test acc={test_acc:.4f}"
            )
            if val_agreement >= args.stop_val_agreement:
                print(
                    f"Stop early: val teacher agreement {val_agreement:.4f} "
                    f">= {args.stop_val_agreement:.4f}"
                )
                break
        if run_count >= args.max_runs:
            break

    json_path = output_dir / "sweep_summary.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    csv_path = output_dir / "sweep_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "lambda_kd",
                "lambda_feat",
                "val_teacher_agreement",
                "test_teacher_agreement",
                "test_accuracy",
                "elapsed_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("Sweep finished.")
    print(f"Summary: {json_path}")
    print(f"Summary: {csv_path}")


if __name__ == "__main__":
    main()
