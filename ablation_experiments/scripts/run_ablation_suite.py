#!/usr/bin/env python3
"""Run ablation-1 and ablation-2 future-step prediction experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
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
TEST_AGREE_PATTERN = re.compile(
    r"Test Teacher-Agreement:\s*([0-9]+(?:\.[0-9]+)?)%",
)


@dataclass
class SeedRecord:
    seed: int
    ablation1_status: str
    ablation1_test_acc: float
    ablation1_test_f1: float
    teacher_status: str
    teacher_test_acc: float
    teacher_test_f1: float
    ablation2_status: str
    ablation2_test_acc: float
    ablation2_test_f1: float
    ablation2_test_teacher_agreement: float
    elapsed_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run h1 ablation suite")
    parser.add_argument(
        "--dataset-npz",
        type=Path,
        default=PROJECT_ROOT / "Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/h1/results",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/h1/reports",
    )

    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-end", type=int, default=109)
    parser.add_argument("--max-runs", type=int, default=10)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--lambda-ce", type=float, default=0.8)
    parser.add_argument("--lambda-kd", type=float, default=1.2)
    parser.add_argument("--lambda-feat", type=float, default=0.2)
    parser.add_argument("--drop-feature-indices", type=str, default="3,4,5,6,7")

    parser.add_argument("--weighted-sampler", dest="weighted_sampler", action="store_true")
    parser.add_argument("--no-weighted-sampler", dest="weighted_sampler", action="store_false")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--print-command", action="store_true")
    parser.set_defaults(weighted_sampler=True, skip_existing=True)
    return parser.parse_args()


def parse_metrics(metrics_path: Path) -> Dict[str, float]:
    if not metrics_path.exists():
        return {"test_acc": -1.0, "test_f1": -1.0, "test_agreement": -1.0}

    text = metrics_path.read_text(encoding="utf-8", errors="ignore")
    m_acc = TEST_ACC_PATTERN.search(text)
    m_f1 = TEST_F1_PATTERN.search(text)
    m_agree = TEST_AGREE_PATTERN.search(text)
    return {
        "test_acc": float(m_acc.group(1)) / 100.0 if m_acc else -1.0,
        "test_f1": float(m_f1.group(1)) if m_f1 else -1.0,
        "test_agreement": float(m_agree.group(1)) / 100.0 if m_agree else -1.0,
    }


def run_cmd(cmd: List[str], print_command: bool) -> str:
    if print_command:
        print("Command:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return "success" if proc.returncode == 0 else f"failed_rc_{proc.returncode}"


def write_progress(report_dir: Path, records: List[SeedRecord]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "ablation_suite_progress.json"
    csv_path = report_dir / "ablation_suite_progress.csv"

    payload = [asdict(r) for r in records]
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(SeedRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(payload)


def main() -> None:
    args = parse_args()
    if args.seed_end < args.seed_start:
        raise SystemExit("seed-end must be >= seed-start")

    scripts = {
        "ablation1": SCRIPT_DIR / "train_ablation1_student_only.py",
        "teacher": SCRIPT_DIR / "train_ablation2_teacher_lstm.py",
        "ablation2": SCRIPT_DIR / "train_ablation2_distill_lstm_student.py",
    }
    for name, path in scripts.items():
        if not path.exists():
            raise SystemExit(f"script not found for {name}: {path}")

    if not args.dataset_npz.exists():
        raise SystemExit(f"dataset not found: {args.dataset_npz}")
    validate_forecast_dataset_contract(args.dataset_npz)

    dataset_tag = args.dataset_npz.stem
    seeds = list(range(args.seed_start, args.seed_end + 1))[: args.max_runs]

    ablation1_out = args.output_root / "ablation1_student_only_tcn_attn"
    teacher_out = args.output_root / "ablation2_teacher_student_lstm/teachers"
    ablation2_out = args.output_root / "ablation2_teacher_student_lstm/students"

    records: List[SeedRecord] = []
    print("=" * 88)
    print("Ablation suite | future-step prediction (horizon=1)")
    print("=" * 88)
    print(f"Dataset: {args.dataset_npz}")
    print(f"Seeds: {seeds[0]}..{seeds[-1]} (count={len(seeds)})")

    for idx, seed in enumerate(seeds, start=1):
        t0 = time.time()
        print(f"\n[{idx:02d}/{len(seeds)}] seed={seed}")

        a1_run_name = (
            f"ablation1_student_only_tcn_attn_{dataset_tag}_ep{args.epochs}_lr{args.lr}_"
            f"bs{args.batch_size}_seed{seed}"
        )
        teacher_run_name = (
            f"ablation2_teacher_lstm_{dataset_tag}_ep{args.epochs}_lr{args.lr}_"
            f"bs{args.batch_size}_seed{seed}"
        )
        a2_run_name = (
            f"ablation2_distill_lstm_{dataset_tag}_ep{args.epochs}_lr{args.lr}_bs{args.batch_size}_"
            f"T{args.temperature}_lce{args.lambda_ce}_lkd{args.lambda_kd}_lf{args.lambda_feat}_seed{seed}"
        )

        a1_dir = ablation1_out / a1_run_name
        teacher_dir = teacher_out / teacher_run_name
        a2_dir = ablation2_out / a2_run_name

        a1_metrics_path = a1_dir / "evaluation_metrics.txt"
        teacher_metrics_path = teacher_dir / "evaluation_metrics.txt"
        a2_metrics_path = a2_dir / "evaluation_metrics.txt"

        if args.skip_existing and a1_metrics_path.exists():
            a1_status = "skipped_existing"
        else:
            cmd = [
                sys.executable,
                str(scripts["ablation1"]),
                "--dataset-npz",
                str(args.dataset_npz),
                "--output-dir",
                str(ablation1_out),
                "--drop-feature-indices",
                args.drop_feature_indices,
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
            ]
            if args.weighted_sampler:
                cmd.append("--weighted-sampler")
            a1_status = run_cmd(cmd, args.print_command)

        if args.skip_existing and teacher_metrics_path.exists():
            teacher_status = "skipped_existing"
        else:
            cmd = [
                sys.executable,
                str(scripts["teacher"]),
                "--dataset-npz",
                str(args.dataset_npz),
                "--output-dir",
                str(teacher_out),
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
            ]
            if args.weighted_sampler:
                cmd.append("--weighted-sampler")
            teacher_status = run_cmd(cmd, args.print_command)

        teacher_ckpt = teacher_dir / "best_teacher_lstm.pth"
        teacher_run_args = teacher_dir / "run_args.json"
        if args.skip_existing and a2_metrics_path.exists():
            a2_status = "skipped_existing"
        elif not teacher_ckpt.exists() or not teacher_run_args.exists():
            a2_status = "failed_missing_teacher_artifacts"
        else:
            cmd = [
                sys.executable,
                str(scripts["ablation2"]),
                "--dataset-npz",
                str(args.dataset_npz),
                "--teacher-ckpt",
                str(teacher_ckpt),
                "--teacher-run-args",
                str(teacher_run_args),
                "--output-dir",
                str(ablation2_out),
                "--drop-feature-indices",
                args.drop_feature_indices,
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
            a2_status = run_cmd(cmd, args.print_command)

        a1_metrics = parse_metrics(a1_metrics_path)
        teacher_metrics = parse_metrics(teacher_metrics_path)
        a2_metrics = parse_metrics(a2_metrics_path)

        record = SeedRecord(
            seed=seed,
            ablation1_status=a1_status,
            ablation1_test_acc=a1_metrics["test_acc"],
            ablation1_test_f1=a1_metrics["test_f1"],
            teacher_status=teacher_status,
            teacher_test_acc=teacher_metrics["test_acc"],
            teacher_test_f1=teacher_metrics["test_f1"],
            ablation2_status=a2_status,
            ablation2_test_acc=a2_metrics["test_acc"],
            ablation2_test_f1=a2_metrics["test_f1"],
            ablation2_test_teacher_agreement=a2_metrics["test_agreement"],
            elapsed_sec=round(time.time() - t0, 2),
        )
        records.append(record)
        write_progress(args.report_dir, records)

        print(
            f"seed={seed} | a1={record.ablation1_status} ({record.ablation1_test_acc*100:.2f}%) | "
            f"teacher={record.teacher_status} ({record.teacher_test_acc*100:.2f}%) | "
            f"a2={record.ablation2_status} ({record.ablation2_test_acc*100:.2f}%)"
        )

    final_report = {
        "dataset": str(args.dataset_npz),
        "output_root": str(args.output_root),
        "report_dir": str(args.report_dir),
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "max_runs": args.max_runs,
        "records": [asdict(r) for r in records],
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    final_json = args.report_dir / "ablation_suite_final_report.json"
    final_json.write_text(json.dumps(final_report, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print(f"Finished {len(records)} records")
    print(f"Progress CSV: {args.report_dir / 'ablation_suite_progress.csv'}")
    print(f"Final report: {final_json}")


if __name__ == "__main__":
    main()
