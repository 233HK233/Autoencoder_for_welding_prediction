#!/usr/bin/env python3
"""Seed-only sweep runner for distill single TCN student."""

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


TEST_ACC_PATTERN = re.compile(
    r"---\s*Test Metrics \(Student\)\s*---.*?Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%",
    flags=re.S,
)
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
    best_epoch: int
    best_score: float
    reached_target: bool


def parse_metric_file(metrics_path: Path) -> Dict[str, float | int]:
    if not metrics_path.exists():
        return {"test_accuracy": -1.0, "best_epoch": -1, "best_score": -1.0}
    text = metrics_path.read_text(encoding="utf-8", errors="ignore")

    m_acc = TEST_ACC_PATTERN.search(text)
    test_acc = float(m_acc.group(1)) / 100.0 if m_acc else -1.0

    m_epoch = BEST_EPOCH_PATTERN.search(text)
    best_epoch = int(m_epoch.group(1)) if m_epoch else -1

    m_score = BEST_SCORE_PATTERN.search(text)
    best_score = float(m_score.group(1)) if m_score else -1.0

    return {
        "test_accuracy": test_acc,
        "best_epoch": best_epoch,
        "best_score": best_score,
    }


def count_reached_target(outcomes: Iterable[RunOutcome]) -> int:
    return sum(1 for item in outcomes if item.reached_target)


def build_run_name(dataset_tag: str, args: argparse.Namespace, seed: int) -> str:
    return (
        f"distill_tcn_attn_{dataset_tag}_ep{args.epochs}_lr{args.lr}_bs{args.batch_size}_"
        f"T{args.temperature}_lce{args.lambda_ce}_lkd{args.lambda_kd}_lf{args.lambda_feat}_seed{seed}"
    )


def build_train_command(args: argparse.Namespace, seed: int, train_script: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(train_script),
        "--dataset-npz",
        str(args.dataset_npz),
        "--teacher-ckpt",
        str(args.teacher_ckpt),
        "--teacher-run-args",
        str(args.teacher_run_args),
        "--output-dir",
        str(args.output_dir),
        "--drop-feature-indices",
        str(args.drop_feature_indices),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
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
    ]
    if args.weighted_sampler:
        cmd.append("--weighted-sampler")
    return cmd


def write_progress(out_dir: Path, outcomes: List[RunOutcome]) -> None:
    payload = [item.__dict__ for item in outcomes]
    json_path = out_dir / "seed_sweep_summary.json"
    csv_path = out_dir / "seed_sweep_summary.csv"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(RunOutcome.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(payload)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_dataset = script_dir / "Data/processed_data/weld_seam_windows_ws5_tf75_pg0.npz"
    default_teacher_dir = (
        script_dir
        / "outputs/single_tcn/best_record/"
        "single_tcn_attn_weld_seam_windows_ws5_tf75_pg0_ep30_lr0.00025_bs128_k3_l3_d0.12_lat64_wd0.0002_seed230"
    )
    default_output = script_dir / "outputs/distill_single_tcn/seed_sweep_20260302_100seeds"

    parser = argparse.ArgumentParser(
        description="Run seed-only sweep for distill student and stop after enough >= target test accuracy."
    )
    parser.add_argument("--dataset-npz", type=Path, default=default_dataset)
    parser.add_argument("--teacher-ckpt", type=Path, default=default_teacher_dir / "best_single_tcn.pth")
    parser.add_argument("--teacher-run-args", type=Path, default=default_teacher_dir / "run_args.json")
    parser.add_argument("--output-dir", type=Path, default=default_output)

    parser.add_argument("--drop-feature-indices", type=str, default="3,4,5,6,7")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
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
        choices=["val_teacher_agreement", "val_macro_f1"],
        default="val_teacher_agreement",
    )
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--lambda-ce", type=float, default=0.8)
    parser.add_argument("--lambda-kd", type=float, default=1.2)
    parser.add_argument("--lambda-feat", type=float, default=0.2)

    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-end", type=int, default=199)
    parser.add_argument("--max-runs", type=int, default=100)
    parser.add_argument("--retry-on-fail", type=int, default=1)
    parser.add_argument("--target-acc", type=float, default=95.0, help="Use percent if >1.0 else ratio")
    parser.add_argument("--target-count", type=int, default=10)
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--print-command", action="store_true")
    parser.set_defaults(weighted_sampler=True, skip_existing=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed_end < args.seed_start:
        raise SystemExit("seed-end must be >= seed-start")

    train_script = Path(__file__).resolve().parent / "train_distill_single_tcn_student.py"
    if not train_script.exists():
        raise SystemExit(f"training script not found: {train_script}")

    for required in (args.dataset_npz, args.teacher_ckpt, args.teacher_run_args):
        if not required.exists():
            raise SystemExit(f"required path not found: {required}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    threshold = args.target_acc / 100.0 if args.target_acc > 1.0 else args.target_acc
    dataset_tag = args.dataset_npz.stem
    seeds = list(range(args.seed_start, args.seed_end + 1))[: args.max_runs]

    outcomes: List[RunOutcome] = []
    reached_total = 0
    planned_total = len(seeds)

    print("=" * 88)
    print("Distill Seed Sweep")
    print("=" * 88)
    print(f"Output dir: {args.output_dir}")
    print(f"Seeds planned: {planned_total} ({seeds[0]}..{seeds[-1]})")
    print(f"Threshold: {threshold*100:.2f}%")
    print(f"Target count: {args.target_count}")

    for idx, seed in enumerate(seeds, start=1):
        run_name = build_run_name(dataset_tag, args, seed)
        run_dir = args.output_dir / run_name
        metrics_path = run_dir / "evaluation_metrics.txt"

        if args.skip_existing and metrics_path.exists():
            parsed = parse_metric_file(metrics_path)
            reached = float(parsed["test_accuracy"]) >= threshold
            outcome = RunOutcome(
                run_name=run_name,
                run_dir=str(run_dir),
                seed=seed,
                status="skipped_existing",
                attempts=0,
                elapsed_sec=0.0,
                test_accuracy=float(parsed["test_accuracy"]),
                best_epoch=int(parsed["best_epoch"]),
                best_score=float(parsed["best_score"]),
                reached_target=reached,
            )
            outcomes.append(outcome)
            reached_total = count_reached_target(outcomes)
            print(
                f"[{idx:03d}/{planned_total}] seed={seed} skipped_existing "
                f"test_acc={outcome.test_accuracy*100:.2f}% reached={reached_total}"
            )
            write_progress(args.output_dir, outcomes)
            if reached_total >= args.target_count:
                print("Stop condition reached by existing runs.")
                break
            continue

        cmd = build_train_command(args, seed, train_script)
        if args.print_command:
            print("Command:", " ".join(cmd))

        status = "failed"
        elapsed = 0.0
        attempts = 0
        for attempt in range(args.retry_on_fail + 1):
            attempts = attempt + 1
            print(f"[{idx:03d}/{planned_total}] seed={seed} attempt={attempts}")
            start_t = time.time()
            proc = subprocess.run(cmd, check=False)
            elapsed += time.time() - start_t
            if proc.returncode == 0:
                status = "success"
                break
            print(f"  run failed with return code {proc.returncode}")

        parsed = parse_metric_file(metrics_path)
        reached = float(parsed["test_accuracy"]) >= threshold
        outcome = RunOutcome(
            run_name=run_name,
            run_dir=str(run_dir),
            seed=seed,
            status=status,
            attempts=attempts,
            elapsed_sec=round(elapsed, 2),
            test_accuracy=float(parsed["test_accuracy"]),
            best_epoch=int(parsed["best_epoch"]),
            best_score=float(parsed["best_score"]),
            reached_target=reached,
        )
        outcomes.append(outcome)
        reached_total = count_reached_target(outcomes)
        print(
            f"  status={status} test_acc={outcome.test_accuracy*100:.2f}% "
            f"best_epoch={outcome.best_epoch} reached={reached_total}/{args.target_count}"
        )
        write_progress(args.output_dir, outcomes)

        if reached_total >= args.target_count:
            print("Stop condition reached.")
            break

    final_json = args.output_dir / "seed_sweep_final_report.json"
    report = {
        "output_dir": str(args.output_dir),
        "total_records": len(outcomes),
        "target_accuracy_threshold": threshold,
        "target_count": args.target_count,
        "reached_count": reached_total,
        "goal_met": reached_total >= args.target_count,
        "planned_total": planned_total,
        "records": [item.__dict__ for item in outcomes],
    }
    final_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=" * 88)
    print(f"Finished records: {len(outcomes)} / {planned_total}")
    print(f"Reached >= {threshold*100:.2f}%: {reached_total}")
    print(f"Goal met: {report['goal_met']}")
    print(f"Progress JSON: {args.output_dir / 'seed_sweep_summary.json'}")
    print(f"Final report: {final_json}")


if __name__ == "__main__":
    main()
