#!/usr/bin/env python3
"""Run a staged horizon=1 sweep for Ablation-2 LSTM distillation."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "ablation_experiments/scripts/train_ablation2_distill_lstm_student.py"
COMPARISON_LSTM_ROOT = PROJECT_ROOT / "ablation_experiments/h1/results/comparison_18d_baselines/lstm"
DEFAULT_EXPERIMENT_TAG = "ablation2_strict_valf1_20260522"

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
VAL_F1_PATTERN = re.compile(
    r"---\s*Val Metrics.*?Macro-F1:\s*([0-9]+(?:\.[0-9]+)?)",
    flags=re.S,
)
TEST_AGREE_PATTERN = re.compile(r"Test Teacher-Agreement:\s*([0-9]+(?:\.[0-9]+)?)%")
BEST_EPOCH_PATTERN = re.compile(r"Best epoch:\s*(-?[0-9]+)")
BEST_SCORE_PATTERN = re.compile(r"Best score:\s*([0-9]+(?:\.[0-9]+)?)")

LOSS_PRESETS: Sequence[tuple[str, Dict[str, float]]] = (
    ("l1", {"temperature": 2.0, "lambda_ce": 0.9, "lambda_kd": 1.1, "lambda_feat": 0.0}),
    ("l2", {"temperature": 2.5, "lambda_ce": 0.9, "lambda_kd": 1.1, "lambda_feat": 0.0}),
    ("l3", {"temperature": 3.0, "lambda_ce": 0.8, "lambda_kd": 1.2, "lambda_feat": 0.2}),
    ("l4", {"temperature": 3.5, "lambda_ce": 0.7, "lambda_kd": 1.4, "lambda_feat": 0.3}),
)
OPT_PRESETS: Sequence[tuple[str, Dict[str, Any]]] = (
    ("o0", {"lr": 1.0e-4, "weight_decay": 1e-5, "weighted_sampler": True}),
    ("o1", {"lr": 1.5e-4, "weight_decay": 1e-5, "weighted_sampler": True}),
    ("o2", {"lr": 2.0e-4, "weight_decay": 1e-4, "weighted_sampler": True}),
    ("o3", {"lr": 3.0e-4, "weight_decay": 2e-4, "weighted_sampler": False}),
)
CAPACITY_PRESETS: Sequence[tuple[str, Dict[str, Any]]] = (
    (
        "c0",
        {
            "student_lstm_layers": 1,
            "student_lstm_dropout": 0.2,
            "student_classifier_hidden": 64,
            "student_classifier_dropout": 0.30,
        },
    ),
    (
        "c1",
        {
            "student_lstm_layers": 2,
            "student_lstm_dropout": 0.3,
            "student_classifier_hidden": 64,
            "student_classifier_dropout": 0.35,
        },
    ),
)
LOCAL_REFINEMENTS: Sequence[tuple[str, Dict[str, Any]]] = (
    ("r1", {"lr_delta": -5e-5, "lambda_kd_delta": 0.0, "lambda_feat_delta": 0.0}),
    ("r2", {"lr_delta": 5e-5, "lambda_kd_delta": 0.1, "lambda_feat_delta": 0.1}),
)
SEARCH_SEEDS = (14, 42)
STAGE3_SEEDS = (77, 122, 183, 230)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a staged Ablation-2 h1 student sweep")
    parser.add_argument(
        "--dataset-npz",
        type=Path,
        default=PROJECT_ROOT / "Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz",
    )
    parser.add_argument(
        "--teacher-results-dir",
        type=Path,
        default=COMPARISON_LSTM_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "ablation_experiments/h1/results/ablation2_teacher_student_lstm/students/ablation2_strict_valf1_20260522",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT
        / "ablation_experiments/h1/reports/ablation2_teacher_student_lstm/ablation2_strict_valf1_20260522",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--retry-on-fail", type=int, default=0)
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--target-acc", type=float, default=90.0)
    parser.add_argument("--max-runs", type=int, default=56)
    parser.add_argument("--experiment-tag", type=str, default=DEFAULT_EXPERIMENT_TAG)
    parser.add_argument(
        "--checkpoint-metric",
        type=str,
        default="val_macro_f1",
        choices=("val_macro_f1", "test_acc", "val_teacher_agreement"),
    )
    parser.add_argument(
        "--student-init-source",
        type=str,
        choices=("none", "teacher"),
        default="teacher",
    )
    parser.set_defaults(skip_existing=True)
    return parser.parse_args()


def parse_threshold(value: float) -> float:
    return value / 100.0 if value > 1.0 else value


def parse_metrics_file(metrics_path: Path) -> Dict[str, float | int]:
    if not metrics_path.exists():
        return {
            "val_macro_f1": -1.0,
            "test_acc": -1.0,
            "test_macro_f1": -1.0,
            "test_teacher_agreement": -1.0,
            "best_epoch": -1,
            "best_score": -1.0,
        }

    text = metrics_path.read_text(encoding="utf-8", errors="ignore")
    m_val_f1 = VAL_F1_PATTERN.search(text)
    m_acc = TEST_ACC_PATTERN.search(text)
    m_f1 = TEST_F1_PATTERN.search(text)
    m_agree = TEST_AGREE_PATTERN.search(text)
    m_epoch = BEST_EPOCH_PATTERN.search(text)
    m_score = BEST_SCORE_PATTERN.search(text)
    return {
        "val_macro_f1": float(m_val_f1.group(1)) if m_val_f1 else -1.0,
        "test_acc": float(m_acc.group(1)) / 100.0 if m_acc else -1.0,
        "test_macro_f1": float(m_f1.group(1)) if m_f1 else -1.0,
        "test_teacher_agreement": float(m_agree.group(1)) / 100.0 if m_agree else -1.0,
        "best_epoch": int(m_epoch.group(1)) if m_epoch else -1,
        "best_score": float(m_score.group(1)) if m_score else -1.0,
    }


def make_job(
    args: argparse.Namespace,
    *,
    stage: str,
    config_id: str,
    base_config_id: str | None,
    loss_cfg: Dict[str, float],
    opt_cfg: Dict[str, Any],
    capacity_cfg: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    return {
        "stage": stage,
        "config_id": config_id,
        "base_config_id": base_config_id,
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "checkpoint_metric": str(args.checkpoint_metric),
        "job_output_root": args.output_dir / stage / config_id,
        **loss_cfg,
        **opt_cfg,
        **capacity_cfg,
    }


def build_stage0_jobs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for loss_name, loss_cfg in LOSS_PRESETS[:2]:
        for opt_name, opt_cfg in OPT_PRESETS[:2]:
            for cap_name, capacity_cfg in CAPACITY_PRESETS:
                config_id = f"s0_{loss_name}_{opt_name}_{cap_name}"
                jobs.append(
                    make_job(
                        args,
                        stage="stage0",
                        config_id=config_id,
                        base_config_id=None,
                        loss_cfg=loss_cfg,
                        opt_cfg=opt_cfg,
                        capacity_cfg=capacity_cfg,
                        seed=SEARCH_SEEDS[0],
                    )
                )
    return jobs


def build_stage1_jobs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for loss_name, loss_cfg in LOSS_PRESETS:
        for opt_name, opt_cfg in OPT_PRESETS:
            for cap_name, capacity_cfg in CAPACITY_PRESETS:
                config_id = f"s1_{loss_name}_{opt_name}_{cap_name}"
                jobs.append(
                    make_job(
                        args,
                        stage="stage1",
                        config_id=config_id,
                        base_config_id=None,
                        loss_cfg=loss_cfg,
                        opt_cfg=opt_cfg,
                        capacity_cfg=capacity_cfg,
                        seed=SEARCH_SEEDS[0],
                    )
                )
    return jobs


def build_stage2_jobs(args: argparse.Namespace, ranked_records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for record in ranked_records[:4]:
        for refine_name, refine in LOCAL_REFINEMENTS:
            config_id = f"s2_{record['config_id']}_{refine_name}"
            jobs.append(
                {
                    **record,
                    "stage": "stage2",
                    "config_id": config_id,
                    "base_config_id": record["config_id"],
                    "lr": max(5e-5, float(record["lr"]) + float(refine.get("lr_delta", 0.0))),
                    "lambda_kd": max(0.5, float(record["lambda_kd"]) + float(refine.get("lambda_kd_delta", 0.0))),
                    "lambda_feat": max(
                        0.0,
                        float(record["lambda_feat"]) + float(refine.get("lambda_feat_delta", 0.0)),
                    ),
                    "job_output_root": args.output_dir / "stage2" / config_id,
                    "seed": SEARCH_SEEDS[len(jobs) % len(SEARCH_SEEDS)],
                }
            )
    return jobs


def build_stage3_jobs(args: argparse.Namespace, ranked_records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for record in ranked_records[:2]:
        base_config_id = record["config_id"]
        for seed in STAGE3_SEEDS:
            config_id = f"s3_{base_config_id}"
            jobs.append(
                {
                    **record,
                    "stage": "stage3",
                    "config_id": config_id,
                    "base_config_id": base_config_id,
                    "job_output_root": args.output_dir / "stage3" / config_id / f"seed_{seed}",
                    "seed": seed,
                }
            )
    return jobs


def rank_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = [
        r
        for r in records
        if r.get("status") in {"success", "skipped_existing"} and float(r.get("val_macro_f1", -1.0)) >= 0.0
    ]
    return sorted(
        valid,
        key=lambda r: (
            float(r.get("val_macro_f1", -1.0)),
            float(r.get("test_macro_f1", -1.0)),
            float(r.get("test_acc", -1.0)),
            float(r.get("test_teacher_agreement", -1.0)),
        ),
        reverse=True,
    )


def resolve_best_teacher_run(results_dir: Path) -> Dict[str, str]:
    best: Dict[str, str] | None = None
    best_acc = -1.0
    for run_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        metrics = parse_metrics_file(run_dir / "evaluation_metrics.txt")
        run_args = run_dir / "run_args.json"
        ckpt_candidates = [run_dir / "best_model.pth", run_dir / "best_teacher_lstm.pth"]
        ckpt_path = next((path for path in ckpt_candidates if path.exists()), None)
        if ckpt_path is None or not run_args.exists():
            continue
        if float(metrics["test_acc"]) > best_acc:
            best_acc = float(metrics["test_acc"])
            best = {
                "run_dir": str(run_dir),
                "teacher_ckpt": str(ckpt_path),
                "teacher_run_args": str(run_args),
                "test_acc": f"{best_acc:.6f}",
            }
    if best is None:
        raise FileNotFoundError(f"no compatible teacher run found under {results_dir}")
    return best


def build_command(
    args: argparse.Namespace,
    job: Dict[str, Any],
    teacher_ckpt: Path,
    teacher_run_args: Path,
) -> List[str]:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset-npz",
        str(args.dataset_npz),
        "--teacher-ckpt",
        str(teacher_ckpt),
        "--teacher-run-args",
        str(teacher_run_args),
        "--output-dir",
        str(job["job_output_root"]),
        "--epochs",
        str(job["epochs"]),
        "--batch-size",
        str(job["batch_size"]),
        "--lr",
        str(job["lr"]),
        "--weight-decay",
        str(job["weight_decay"]),
        "--seed",
        str(job["seed"]),
        "--num-workers",
        str(args.num_workers),
        "--checkpoint-metric",
        str(job["checkpoint_metric"]),
        "--temperature",
        str(job["temperature"]),
        "--lambda-ce",
        str(job["lambda_ce"]),
        "--lambda-kd",
        str(job["lambda_kd"]),
        "--lambda-feat",
        str(job["lambda_feat"]),
        "--student-lstm-layers",
        str(job["student_lstm_layers"]),
        "--student-lstm-dropout",
        str(job["student_lstm_dropout"]),
        "--student-classifier-hidden",
        str(job["student_classifier_hidden"]),
        "--student-classifier-dropout",
        str(job["student_classifier_dropout"]),
    ]
    if bool(job.get("weighted_sampler", False)):
        cmd.append("--weighted-sampler")
    if args.student_init_source == "teacher":
        cmd.extend(
            [
                "--student-init-ckpt",
                str(teacher_ckpt),
                "--student-init-mode",
                "shape_safe",
            ]
        )
    return cmd


def find_result_run(job_output_root: Path) -> Path:
    run_dirs = [path for path in job_output_root.iterdir() if path.is_dir()]
    if not run_dirs:
        return job_output_root
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def write_progress(report_root: Path, records: Sequence[Dict[str, Any]]) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    json_path = report_root / "ablation2_sweep_progress.json"
    csv_path = report_root / "ablation2_sweep_progress.csv"
    json_path.write_text(json.dumps(list(records), indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        if not records:
            writer = csv.writer(f)
            writer.writerow(["stage", "config_id", "status", "test_acc"])
            return
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def execute_jobs(
    args: argparse.Namespace,
    jobs: Sequence[Dict[str, Any]],
    teacher_ckpt: Path,
    teacher_run_args: Path,
    records: List[Dict[str, Any]],
    target_acc: float,
) -> List[Dict[str, Any]]:
    stage_records: List[Dict[str, Any]] = []
    for job in jobs:
        job_output_root = Path(job["job_output_root"])
        job_output_root.mkdir(parents=True, exist_ok=True)
        metrics_path = find_result_run(job_output_root) / "evaluation_metrics.txt"

        if args.skip_existing and metrics_path.exists():
            status = "skipped_existing"
            elapsed_sec = 0.0
        else:
            cmd = build_command(args, job, teacher_ckpt, teacher_run_args)
            if args.print_command:
                print("Command:", " ".join(cmd))
            status = "failed"
            elapsed_sec = 0.0
            for _ in range(args.retry_on_fail + 1):
                start_t = time.time()
                proc = subprocess.run(cmd, check=False)
                elapsed_sec += time.time() - start_t
                if proc.returncode == 0:
                    status = "success"
                    break
                status = f"failed_rc_{proc.returncode}"

        run_dir = find_result_run(job_output_root)
        metrics = parse_metrics_file(run_dir / "evaluation_metrics.txt")
        record = {
            **job,
            "job_output_root": str(job_output_root),
            "run_dir": str(run_dir),
            "status": status,
            "elapsed_sec": round(elapsed_sec, 2),
            "val_macro_f1": float(metrics["val_macro_f1"]),
            "test_acc": float(metrics["test_acc"]),
            "test_macro_f1": float(metrics["test_macro_f1"]),
            "test_teacher_agreement": float(metrics["test_teacher_agreement"]),
            "best_epoch": int(metrics["best_epoch"]),
            "best_score": float(metrics["best_score"]),
            "target_reached": float(metrics["test_acc"]) >= target_acc,
        }
        stage_records.append(record)
        records.append(record)
        write_progress(args.report_dir, records)
    return stage_records


def write_summary(
    args: argparse.Namespace,
    records: Sequence[Dict[str, Any]],
    teacher_ref: Dict[str, str],
    target_acc: float,
) -> Path:
    report_root = args.report_dir
    report_root.mkdir(parents=True, exist_ok=True)
    ranked = rank_records(records)
    best = ranked[0] if ranked else None
    payload = {
        "dataset_npz": str(args.dataset_npz),
        "teacher_reference": teacher_ref,
        "target_test_acc": target_acc,
        "num_runs": len(records),
        "target_met": bool(best and float(best["test_acc"]) >= target_acc),
        "best_run": best,
        "records": list(records),
    }
    summary_path = report_root / "ablation2_sweep_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    args = parse_args()
    if not TRAIN_SCRIPT.exists():
        raise SystemExit(f"training script not found: {TRAIN_SCRIPT}")
    if not args.dataset_npz.exists():
        raise SystemExit(f"dataset not found: {args.dataset_npz}")
    if not args.teacher_results_dir.exists():
        raise SystemExit(f"teacher results dir not found: {args.teacher_results_dir}")

    validate_forecast_dataset_contract(args.dataset_npz)
    target_acc = parse_threshold(float(args.target_acc))
    teacher_ref = resolve_best_teacher_run(args.teacher_results_dir)
    teacher_ckpt = Path(teacher_ref["teacher_ckpt"])
    teacher_run_args = Path(teacher_ref["teacher_run_args"])

    records: List[Dict[str, Any]] = []
    all_jobs = build_stage0_jobs(args) + build_stage1_jobs(args)
    stage0_jobs = all_jobs[: len(build_stage0_jobs(args))]
    stage1_jobs = all_jobs[len(stage0_jobs) :]

    print("=" * 88)
    print("Ablation-2 h1 staged sweep")
    print("=" * 88)
    print(f"Dataset: {args.dataset_npz}")
    print(f"Teacher run: {teacher_ref['run_dir']}")
    print(f"Target test_acc: {target_acc*100:.2f}%")

    execute_jobs(args, stage0_jobs[: args.max_runs], teacher_ckpt, teacher_run_args, records, target_acc)
    if len(records) < args.max_runs:
        execute_jobs(
            args,
            stage1_jobs[: max(0, args.max_runs - len(records))],
            teacher_ckpt,
            teacher_run_args,
            records,
            target_acc,
        )

    ranked = rank_records(records)
    if len(records) < args.max_runs and ranked:
        stage2_jobs = build_stage2_jobs(args, ranked)
        execute_jobs(
            args,
            stage2_jobs[: max(0, args.max_runs - len(records))],
            teacher_ckpt,
            teacher_run_args,
            records,
            target_acc,
        )

    ranked = rank_records(records)
    if len(records) < args.max_runs and ranked:
        stage3_jobs = build_stage3_jobs(args, ranked)
        execute_jobs(
            args,
            stage3_jobs[: max(0, args.max_runs - len(records))],
            teacher_ckpt,
            teacher_run_args,
            records,
            target_acc,
        )

    summary_path = write_summary(args, records, teacher_ref, target_acc)
    print(f"Finished runs: {len(records)}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
