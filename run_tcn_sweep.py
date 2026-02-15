#!/usr/bin/env python3

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def parse_student_test_accuracy(metrics_text: str) -> float:
    match = re.search(
        r"--- Student Test Metrics ---.*?Accuracy:\s*([0-9.]+)%",
        metrics_text,
        flags=re.S,
    )
    if not match:
        return -1.0
    return float(match.group(1)) / 100.0


def build_run_name(params: dict) -> str:
    safe_weights = params["class_weights"].replace(",", "-")
    return (
        f"epochs{params['epochs']}_lr{params['lr']}_bs{params['batch_size']}_"
        f"k{params['tcn_kernel']}_l{params['tcn_layers']}_d{params['tcn_dropout']}_"
        f"db{params['tcn_dilation_base']}_wd{params['weight_decay']}_"
        f"cw{safe_weights}_seed{params['seed']}_"
        f"lc{params['lambda_class']}_la{params['lambda_align']}_lk{params['lambda_kl']}"
    )


def build_command(params: dict, train_script: Path) -> list:
    return [
        sys.executable,
        str(train_script),
        "--epochs",
        str(params["epochs"]),
        "--lr",
        str(params["lr"]),
        "--batch-size",
        str(params["batch_size"]),
        "--output-dir",
        str(params["output_dir"]),
        "--dataset-npz",
        str(params["dataset_npz"]),
        "--tcn-kernel",
        str(params["tcn_kernel"]),
        "--tcn-layers",
        str(params["tcn_layers"]),
        "--tcn-dropout",
        str(params["tcn_dropout"]),
        "--tcn-dilation-base",
        str(params["tcn_dilation_base"]),
        "--class-weights",
        str(params["class_weights"]),
        "--weight-decay",
        str(params["weight_decay"]),
        "--seed",
        str(params["seed"]),
        "--lambda-class",
        str(params["lambda_class"]),
        "--lambda-align",
        str(params["lambda_align"]),
        "--lambda-kl",
        str(params["lambda_kl"]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="TCN sweep for lambda_align/lambda_kl")
    parser.add_argument("--output-dir", type=str, default="outputs/label_safe_tcn")
    parser.add_argument(
        "--dataset-npz",
        type=str,
        default="Data/processed_data/weld_seam_windows.npz",
    )
    parser.add_argument("--stop-acc", type=float, default=0.92)
    parser.add_argument("--max-runs", type=int, default=50)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train_benchmark_classifier_tcn.py"

    base_params = {
        "epochs": 40,
        "lr": 0.0001,
        "batch_size": 96,
        "output_dir": args.output_dir,
        "dataset_npz": args.dataset_npz,
        "tcn_kernel": 3,
        "tcn_layers": 1,
        "tcn_dropout": 0.25,
        "tcn_dilation_base": 2,
        "class_weights": "auto",
        "weight_decay": 0.0005,
        "seed": 42,
        "lambda_class": 1.0,
    }

    lambda_align_values = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.12, 0.15]
    lambda_kl_values = [0.0, 0.005, 0.01, 0.02, 0.05]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    run_count = 0

    for la in lambda_align_values:
        for lk in lambda_kl_values:
            if run_count >= args.max_runs:
                break

            params = dict(base_params)
            params["lambda_align"] = la
            params["lambda_kl"] = lk

            cmd = build_command(params, train_script)
            run_name = build_run_name(params)
            run_dir = output_dir / run_name

            print("=" * 60)
            print(f"Run {run_count + 1}: lambda_align={la}, lambda_kl={lk}")
            print(f"Output: {run_dir}")
            print("Command:", " ".join(cmd))
            print("=" * 60)

            start = time.time()
            subprocess.run(cmd, check=True)
            elapsed = time.time() - start

            metrics_path = run_dir / "evaluation_metrics.txt"
            accuracy = -1.0
            if metrics_path.exists():
                metrics_text = metrics_path.read_text(encoding="utf-8")
                accuracy = parse_student_test_accuracy(metrics_text)

            results.append(
                {
                    "run_name": run_name,
                    "lambda_align": la,
                    "lambda_kl": lk,
                    "student_test_accuracy": accuracy,
                    "elapsed_sec": round(elapsed, 2),
                }
            )

            print(f"Student test accuracy: {accuracy:.4f}")
            run_count += 1

            if accuracy >= args.stop_acc:
                print(f"Stop early: reached {accuracy:.4f} >= {args.stop_acc}")
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
                "lambda_align",
                "lambda_kl",
                "student_test_accuracy",
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
