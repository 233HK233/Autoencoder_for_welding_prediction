#!/usr/bin/env python3
"""Run the horizon=1 ablation target sweep suite end-to-end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
A1_SWEEP = SCRIPT_DIR / "run_ablation1_sweep.py"
A2_SWEEP = SCRIPT_DIR / "run_ablation2_sweep.py"
A3_SWEEP = PROJECT_ROOT / "ablation_experiments/ablation3_joint_tcn_attn/scripts/run_ablation3_h1_sweep.py"
SUMMARY_SCRIPT = SCRIPT_DIR / "summarize_ablation_results.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all h1 ablation target sweeps")
    parser.add_argument(
        "--dataset-npz",
        type=Path,
        default=PROJECT_ROOT / "Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/h1/results",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/h1/reports",
    )
    parser.add_argument("--target-acc", type=float, default=90.0)
    parser.add_argument("--a1-stage-a-runs", type=int, default=40)
    parser.add_argument("--a1-stage-b-runs", type=int, default=20)
    parser.add_argument("--a1-top-k", type=int, default=6)
    parser.add_argument("--a2-max-runs", type=int, default=56)
    parser.add_argument("--a3-max-runs", type=int, default=40)
    parser.add_argument("--a2-experiment-tag", type=str, default="ablation2_h1_scan90_20260327")
    parser.add_argument("--a3-experiment-tag", type=str, default="ablation3_h1_scan90_20260327")
    parser.add_argument("--print-command", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str], print_command: bool) -> None:
    if print_command:
        print("Command:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    args = parse_args()

    run_cmd(
        [
            sys.executable,
            str(A1_SWEEP),
            "--dataset-npz",
            str(args.dataset_npz),
            "--output-root",
            str(args.results_root / "ablation1_sweep"),
            "--report-dir",
            str(args.report_dir / "ablation1_sweep"),
            "--stage-a-runs",
            str(args.a1_stage_a_runs),
            "--stage-b-runs",
            str(args.a1_stage_b_runs),
            "--top-k",
            str(args.a1_top_k),
            "--target-acc",
            str(args.target_acc),
            "--no-stop-on-target",
        ],
        args.print_command,
    )
    run_cmd(
        [
            sys.executable,
            str(A2_SWEEP),
            "--dataset-npz",
            str(args.dataset_npz),
            "--output-dir",
            str(args.results_root / "ablation2_teacher_student_lstm/students"),
            "--report-dir",
            str(args.report_dir / "ablation2_teacher_student_lstm"),
            "--max-runs",
            str(args.a2_max_runs),
            "--experiment-tag",
            str(args.a2_experiment_tag),
            "--target-acc",
            str(args.target_acc),
        ],
        args.print_command,
    )
    run_cmd(
        [
            sys.executable,
            str(A3_SWEEP),
            "--dataset-npz",
            str(args.dataset_npz),
            "--output-dir",
            str(args.results_root / "ablation3_joint_tcn_attn"),
            "--report-dir",
            str(args.report_dir / "ablation3_joint_tcn_attn"),
            "--student-results-dir",
            str(args.results_root / "ablation1_sweep/trials"),
            "--max-runs",
            str(args.a3_max_runs),
            "--experiment-tag",
            str(args.a3_experiment_tag),
            "--target-acc",
            str(args.target_acc),
        ],
        args.print_command,
    )
    run_cmd(
        [
            sys.executable,
            str(SUMMARY_SCRIPT),
            "--ablation1-dir",
            str(args.results_root / "ablation1_sweep/trials"),
            "--ablation2-dir",
            str(args.results_root / "ablation2_teacher_student_lstm/students"),
            "--ablation3-dir",
            str(args.results_root / "ablation3_joint_tcn_attn"),
            "--out-dir",
            str(args.report_dir),
        ],
        args.print_command,
    )


if __name__ == "__main__":
    main()
