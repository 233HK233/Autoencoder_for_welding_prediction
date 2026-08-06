#!/usr/bin/env python3
"""Seed sweep runner for Ablation-3 future-step prediction training."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

AB3_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training_utils import validate_forecast_dataset_contract  # noqa: E402

TEST_ACC_PATTERN = re.compile(
    r"---\s*Test Metrics.*?Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%",
    flags=re.S,
)
TEST_F1_PATTERN = re.compile(
    r"---\s*Test Metrics.*?Macro-F1:\s*([0-9]+(?:\.[0-9]+)?)",
    flags=re.S,
)
TEST_AGREE_PATTERN = re.compile(r"Test Teacher-Agreement:\s*([0-9]+(?:\.[0-9]+)?)%")
BEST_EPOCH_PATTERN = re.compile(r"Best epoch:\s*(\d+)")
BEST_SCORE_PATTERN = re.compile(r"Best score:\s*([0-9]+(?:\.[0-9]+)?)")


@dataclass
class RunOutcome:
    run_name: str
    run_dir: str
    seed: int
    status: str
    attempts: int
    elapsed_sec: float
    test_accuracy: float
    test_macro_f1: float
    test_teacher_agreement: float
    best_epoch: int
    best_score: float


def parse_metrics_file(metrics_path: Path) -> Dict[str, float | int]:
    if not metrics_path.exists():
        return {
            "test_accuracy": -1.0,
            "test_macro_f1": -1.0,
            "test_teacher_agreement": -1.0,
            "best_epoch": -1,
            "best_score": -1.0,
        }

    text = metrics_path.read_text(encoding="utf-8", errors="ignore")

    m_acc = TEST_ACC_PATTERN.search(text)
    m_f1 = TEST_F1_PATTERN.search(text)
    m_agree = TEST_AGREE_PATTERN.search(text)
    m_epoch = BEST_EPOCH_PATTERN.search(text)
    m_score = BEST_SCORE_PATTERN.search(text)

    return {
        "test_accuracy": float(m_acc.group(1)) / 100.0 if m_acc else -1.0,
        "test_macro_f1": float(m_f1.group(1)) if m_f1 else -1.0,
        "test_teacher_agreement": float(m_agree.group(1)) / 100.0 if m_agree else -1.0,
        "best_epoch": int(m_epoch.group(1)) if m_epoch else -1,
        "best_score": float(m_score.group(1)) if m_score else -1.0,
    }


def build_run_name(dataset_tag: str, args: argparse.Namespace, seed: int) -> str:
    return (
        f"ablation3_joint_tcn_attn_{dataset_tag}_ep{args.epochs}_lrt{args.lr_teacher}_"
        f"lrs{args.lr_student}_bs{args.batch_size}_T{args.temperature}_"
        f"lce{args.lambda_ce}_lkd{args.lambda_kd}_lf{args.lambda_feat}_seed{seed}"
    )


def build_train_command(args: argparse.Namespace, seed: int, train_script: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(train_script),
        "--dataset-npz",
        str(args.dataset_npz),
        "--output-dir",
        str(args.output_dir),
        "--drop-feature-indices",
        str(args.drop_feature_indices),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr-teacher",
        str(args.lr_teacher),
        "--lr-student",
        str(args.lr_student),
        "--weight-decay",
        str(args.weight_decay),
        "--seed",
        str(seed),
        "--num-workers",
        str(args.num_workers),
        "--class-weights",
        str(args.class_weights),
        "--val-ratio",
        str(args.val_ratio),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--min-epochs",
        str(args.min_epochs),
        "--checkpoint-metric",
        str(args.checkpoint_metric),
        "--temperature",
        str(args.temperature),
        "--lambda-ce",
        str(args.lambda_ce),
        "--lambda-kd",
        str(args.lambda_kd),
        "--lambda-feat",
        str(args.lambda_feat),
        "--teacher-tcn-kernel",
        str(args.teacher_tcn_kernel),
        "--teacher-tcn-layers",
        str(args.teacher_tcn_layers),
        "--teacher-tcn-channels",
        str(args.teacher_tcn_channels),
        "--teacher-tcn-dropout",
        str(args.teacher_tcn_dropout),
        "--teacher-tcn-dilation-base",
        str(args.teacher_tcn_dilation_base),
        "--teacher-classifier-hidden",
        str(args.teacher_classifier_hidden),
        "--teacher-classifier-dropout",
        str(args.teacher_classifier_dropout),
        "--teacher-attn-heads",
        str(args.teacher_attn_heads),
        "--teacher-attn-dropout",
        str(args.teacher_attn_dropout),
        "--teacher-attn-ff-dim",
        str(args.teacher_attn_ff_dim),
    ]

    if args.weighted_sampler:
        cmd.append("--weighted-sampler")

    if args.student_tcn_kernel is not None:
        cmd += ["--student-tcn-kernel", str(args.student_tcn_kernel)]
    if args.student_tcn_layers is not None:
        cmd += ["--student-tcn-layers", str(args.student_tcn_layers)]
    if args.student_tcn_channels is not None:
        cmd += ["--student-tcn-channels", str(args.student_tcn_channels)]
    if args.student_tcn_dropout is not None:
        cmd += ["--student-tcn-dropout", str(args.student_tcn_dropout)]
    if args.student_tcn_dilation_base is not None:
        cmd += ["--student-tcn-dilation-base", str(args.student_tcn_dilation_base)]
    if args.student_classifier_hidden is not None:
        cmd += ["--student-classifier-hidden", str(args.student_classifier_hidden)]
    if args.student_classifier_dropout is not None:
        cmd += ["--student-classifier-dropout", str(args.student_classifier_dropout)]
    if args.student_attn_heads is not None:
        cmd += ["--student-attn-heads", str(args.student_attn_heads)]
    if args.student_attn_dropout is not None:
        cmd += ["--student-attn-dropout", str(args.student_attn_dropout)]
    if args.student_attn_ff_dim is not None:
        cmd += ["--student-attn-ff-dim", str(args.student_attn_ff_dim)]

    return cmd


def write_progress(report_dir: Path, outcomes: List[RunOutcome]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = [item.__dict__ for item in outcomes]

    json_path = report_dir / "ablation3_seed_sweep_progress.json"
    csv_path = report_dir / "ablation3_seed_sweep_progress.csv"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(RunOutcome.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run seed sweep for Ablation-3 future-step prediction"
    )
    parser.add_argument(
        "--dataset-npz",
        type=Path,
        default=PROJECT_ROOT / "Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/h1/results/ablation3_joint_tcn_attn",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/h1/reports/ablation3_joint_tcn_attn",
    )

    parser.add_argument("--drop-feature-indices", type=str, default="3,4,5,6,7")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-teacher", type=float, default=2.5e-4)
    parser.add_argument("--lr-student", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--class-weights", type=str, default="auto")
    parser.add_argument("--weighted-sampler", dest="weighted_sampler", action="store_true")
    parser.add_argument("--no-weighted-sampler", dest="weighted_sampler", action="store_false")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--early-stop-patience", type=int, default=16)
    parser.add_argument("--min-epochs", type=int, default=16)
    parser.add_argument(
        "--checkpoint-metric",
        choices=["val_macro_f1", "val_teacher_agreement"],
        default="val_macro_f1",
    )
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--lambda-ce", type=float, default=0.8)
    parser.add_argument("--lambda-kd", type=float, default=1.2)
    parser.add_argument("--lambda-feat", type=float, default=0.2)

    parser.add_argument("--teacher-tcn-kernel", type=int, default=3)
    parser.add_argument("--teacher-tcn-layers", type=int, default=3)
    parser.add_argument("--teacher-tcn-channels", type=str, default="80,80,80")
    parser.add_argument("--teacher-tcn-dropout", type=float, default=0.12)
    parser.add_argument("--teacher-tcn-dilation-base", type=int, default=2)
    parser.add_argument("--teacher-classifier-hidden", type=int, default=128)
    parser.add_argument("--teacher-classifier-dropout", type=float, default=0.35)
    parser.add_argument("--teacher-attn-heads", type=int, default=4)
    parser.add_argument("--teacher-attn-dropout", type=float, default=0.1)
    parser.add_argument("--teacher-attn-ff-dim", type=int, default=128)

    parser.add_argument("--student-tcn-kernel", type=int, default=None)
    parser.add_argument("--student-tcn-layers", type=int, default=None)
    parser.add_argument("--student-tcn-channels", type=str, default=None)
    parser.add_argument("--student-tcn-dropout", type=float, default=None)
    parser.add_argument("--student-tcn-dilation-base", type=int, default=None)
    parser.add_argument("--student-classifier-hidden", type=int, default=None)
    parser.add_argument("--student-classifier-dropout", type=float, default=None)
    parser.add_argument("--student-attn-heads", type=int, default=None)
    parser.add_argument("--student-attn-dropout", type=float, default=None)
    parser.add_argument("--student-attn-ff-dim", type=int, default=None)

    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-end", type=int, default=109)
    parser.add_argument("--max-runs", type=int, default=10)
    parser.add_argument("--retry-on-fail", type=int, default=1)
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--print-command", action="store_true")
    parser.set_defaults(weighted_sampler=True, skip_existing=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed_end < args.seed_start:
        raise SystemExit("seed-end must be >= seed-start")

    train_script = Path(__file__).resolve().parent / "train_ablation3_joint_tcn_attn.py"
    if not train_script.exists():
        raise SystemExit(f"training script not found: {train_script}")
    if not args.dataset_npz.exists():
        raise SystemExit(f"dataset not found: {args.dataset_npz}")
    validate_forecast_dataset_contract(args.dataset_npz)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    dataset_tag = args.dataset_npz.stem
    seeds = list(range(args.seed_start, args.seed_end + 1))[: args.max_runs]
    outcomes: List[RunOutcome] = []

    print("=" * 88)
    print("Ablation-3 future-step prediction seed sweep")
    print("=" * 88)
    print(f"Output dir: {args.output_dir}")
    print(f"Report dir: {args.report_dir}")
    print(f"Seeds planned: {len(seeds)} ({seeds[0]}..{seeds[-1]})")

    for idx, seed in enumerate(seeds, start=1):
        run_name = build_run_name(dataset_tag, args, seed)
        run_dir = args.output_dir / run_name
        metrics_path = run_dir / "evaluation_metrics.txt"

        if args.skip_existing and metrics_path.exists():
            parsed = parse_metrics_file(metrics_path)
            outcome = RunOutcome(
                run_name=run_name,
                run_dir=str(run_dir),
                seed=seed,
                status="skipped_existing",
                attempts=0,
                elapsed_sec=0.0,
                test_accuracy=float(parsed["test_accuracy"]),
                test_macro_f1=float(parsed["test_macro_f1"]),
                test_teacher_agreement=float(parsed["test_teacher_agreement"]),
                best_epoch=int(parsed["best_epoch"]),
                best_score=float(parsed["best_score"]),
            )
            outcomes.append(outcome)
            write_progress(args.report_dir, outcomes)
            print(
                f"[{idx:03d}/{len(seeds)}] seed={seed} skipped_existing "
                f"acc={outcome.test_accuracy*100:.2f}% f1={outcome.test_macro_f1:.4f}"
            )
            continue

        cmd = build_train_command(args, seed, train_script)
        if args.print_command:
            print("Command:", " ".join(cmd))

        status = "failed"
        elapsed = 0.0
        attempts = 0
        for attempt in range(args.retry_on_fail + 1):
            attempts = attempt + 1
            print(f"[{idx:03d}/{len(seeds)}] seed={seed} attempt={attempts}")
            start_t = time.time()
            proc = subprocess.run(cmd, check=False)
            elapsed += time.time() - start_t
            if proc.returncode == 0:
                status = "success"
                break
            print(f"  run failed with return code {proc.returncode}")

        parsed = parse_metrics_file(metrics_path)
        outcome = RunOutcome(
            run_name=run_name,
            run_dir=str(run_dir),
            seed=seed,
            status=status,
            attempts=attempts,
            elapsed_sec=round(elapsed, 2),
            test_accuracy=float(parsed["test_accuracy"]),
            test_macro_f1=float(parsed["test_macro_f1"]),
            test_teacher_agreement=float(parsed["test_teacher_agreement"]),
            best_epoch=int(parsed["best_epoch"]),
            best_score=float(parsed["best_score"]),
        )
        outcomes.append(outcome)
        write_progress(args.report_dir, outcomes)

        print(
            f"  status={status} test_acc={outcome.test_accuracy*100:.2f}% "
            f"test_f1={outcome.test_macro_f1:.4f} "
            f"test_agree={outcome.test_teacher_agreement*100:.2f}%"
        )

    final_json = args.report_dir / "ablation3_seed_sweep_final_report.json"
    report = {
        "output_dir": str(args.output_dir),
        "report_dir": str(args.report_dir),
        "total_records": len(outcomes),
        "planned_total": len(seeds),
        "records": [item.__dict__ for item in outcomes],
    }
    final_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 88)
    print(f"Finished records: {len(outcomes)} / {len(seeds)}")
    print(f"Progress JSON: {args.report_dir / 'ablation3_seed_sweep_progress.json'}")
    print(f"Final report: {final_json}")


if __name__ == "__main__":
    main()
