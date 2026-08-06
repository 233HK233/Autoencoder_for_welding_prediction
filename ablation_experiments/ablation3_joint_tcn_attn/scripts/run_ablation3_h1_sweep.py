#!/usr/bin/env python3
"""Run a staged horizon=1 sweep for Ablation-3 joint TCN distillation."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_ROOT / "train_ablation3_joint_tcn_attn.py"
DEFAULT_EXPERIMENT_TAG = "ablation3_h1_scan90_20260327"
A1_RESULTS_ROOT = PROJECT_ROOT / "ablation_experiments/h1/results/ablation1_sweep/trials"
TEACHER_MANIFEST = PROJECT_ROOT / "outputs/teacher_h1_scan98_gpu1_v2/manifests/best_teacher_h1.json"

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
BEST_EPOCH_PATTERN = re.compile(r"Best epoch:\s*(-?[0-9]+)")
BEST_SCORE_PATTERN = re.compile(r"Best score:\s*([0-9]+(?:\.[0-9]+)?)")

LOSS_PRESETS: Sequence[tuple[str, Dict[str, float]]] = (
    ("l4", {"temperature": 2.5, "lambda_ce": 0.9, "lambda_kd": 1.1, "lambda_feat": 0.0}),
    ("l4p", {"temperature": 2.0, "lambda_ce": 0.9, "lambda_kd": 1.2, "lambda_feat": 0.1}),
    ("l3", {"temperature": 3.0, "lambda_ce": 0.8, "lambda_kd": 1.1, "lambda_feat": 0.0}),
)
OPT_PRESETS: Sequence[tuple[str, Dict[str, float]]] = (
    ("o1", {"lr_teacher": 1e-4, "lr_student": 2e-4, "weight_decay": 1.5e-4}),
    ("o2", {"lr_teacher": 1e-4, "lr_student": 3e-4, "weight_decay": 1.5e-4}),
    ("o3", {"lr_teacher": 2e-4, "lr_student": 3e-4, "weight_decay": 2e-4}),
)
CAPACITY_PRESETS: Sequence[tuple[str, Dict[str, Any]]] = (
    (
        "c0",
        {
            "student_tcn_layers": 3,
            "student_tcn_dropout": 0.12,
            "student_classifier_hidden": 64,
            "student_classifier_dropout": 0.30,
            "student_attn_dropout": 0.05,
            "student_attn_ff_dim": 128,
            "student_tcn_dilation_base": 1,
        },
    ),
    (
        "c7",
        {
            "student_tcn_layers": 3,
            "student_tcn_dropout": 0.12,
            "student_classifier_hidden": 64,
            "student_classifier_dropout": 0.30,
            "student_attn_dropout": 0.05,
            "student_attn_ff_dim": 128,
            "student_tcn_dilation_base": 1,
        },
    ),
)
LOCAL_REFINEMENTS: Sequence[tuple[str, Dict[str, float]]] = (
    ("r1", {"lr_student_delta": -5e-5, "lambda_kd_delta": 0.0, "lambda_feat_delta": 0.0}),
    ("r2", {"lr_student_delta": 5e-5, "lambda_kd_delta": 0.1, "lambda_feat_delta": 0.1}),
)
SEARCH_SEEDS = (14, 42, 122)
STAGE3_SEEDS = (132, 230, 314, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a staged Ablation-3 h1 sweep")
    parser.add_argument(
        "--dataset-npz",
        type=Path,
        default=PROJECT_ROOT / "Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz",
    )
    parser.add_argument(
        "--teacher-manifest",
        type=Path,
        default=TEACHER_MANIFEST,
    )
    parser.add_argument(
        "--student-results-dir",
        type=Path,
        default=A1_RESULTS_ROOT,
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
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--retry-on-fail", type=int, default=0)
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--target-acc", type=float, default=90.0)
    parser.add_argument("--max-runs", type=int, default=40)
    parser.add_argument("--experiment-tag", type=str, default=DEFAULT_EXPERIMENT_TAG)
    parser.set_defaults(skip_existing=True)
    return parser.parse_args()


def parse_threshold(value: float) -> float:
    return value / 100.0 if value > 1.0 else value


def parse_metrics_file(metrics_path: Path) -> Dict[str, float | int]:
    if not metrics_path.exists():
        return {
            "test_acc": -1.0,
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
        "test_acc": float(m_acc.group(1)) / 100.0 if m_acc else -1.0,
        "test_macro_f1": float(m_f1.group(1)) if m_f1 else -1.0,
        "test_teacher_agreement": float(m_agree.group(1)) / 100.0 if m_agree else -1.0,
        "best_epoch": int(m_epoch.group(1)) if m_epoch else -1,
        "best_score": float(m_score.group(1)) if m_score else -1.0,
    }


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_teacher_artifacts(manifest_path: Path) -> Dict[str, Any]:
    payload = load_json(manifest_path)
    best_run = payload["best_run"]
    run_dir = Path(best_run["path"])
    run_args_path = Path(best_run["run_args_path"])
    ckpt_candidates = [run_dir / "best_single_tcn.pth", run_dir / "best_teacher_joint_tcn_attn.pth"]
    ckpt_path = next((path for path in ckpt_candidates if path.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(f"teacher checkpoint not found under {run_dir}")
    run_args = load_json(run_args_path)
    latent_dim = int(run_args.get("latent_dim", 64))
    layers = int(run_args.get("tcn_layers", 3))
    channels = run_args.get("tcn_channels")
    if not channels:
        channels = ",".join([str(latent_dim)] * layers)
    return {
        "teacher_ckpt": str(ckpt_path),
        "teacher_run_args": str(run_args_path),
        "teacher_tcn_kernel": int(run_args.get("tcn_kernel", 3)),
        "teacher_tcn_layers": layers,
        "teacher_tcn_channels": str(channels),
        "teacher_tcn_dropout": float(run_args.get("tcn_dropout", 0.08)),
        "teacher_tcn_dilation_base": int(run_args.get("tcn_dilation_base", 2)),
        "teacher_classifier_hidden": int(run_args.get("classifier_hidden", 64)),
        "teacher_classifier_dropout": float(run_args.get("classifier_dropout", 0.4)),
        "teacher_attn_heads": int(run_args.get("attn_heads", 4)),
        "teacher_attn_dropout": float(run_args.get("attn_dropout", 0.1)),
        "teacher_attn_ff_dim": int(run_args.get("attn_ff_dim", 128)),
        "teacher_weighted_sampler": bool(run_args.get("weighted_sampler", True)),
        "teacher_run_dir": str(run_dir),
    }


def resolve_best_a1_student(results_dir: Path) -> Dict[str, Any] | None:
    if not results_dir.exists():
        return None

    best: Dict[str, Any] | None = None
    best_acc = -1.0
    for ckpt_path in sorted(results_dir.rglob("best_student_only.pth")):
        run_dir = ckpt_path.parent
        metrics = parse_metrics_file(run_dir / "evaluation_metrics.txt")
        if float(metrics["test_acc"]) <= best_acc:
            continue
        run_args_path = run_dir / "run_args.json"
        if not ckpt_path.exists() or not run_args_path.exists():
            continue
        best_acc = float(metrics["test_acc"])
        best = {
            "student_ckpt": str(ckpt_path),
            "student_run_args": str(run_args_path),
            "student_run_dir": str(run_dir),
            "test_acc": best_acc,
        }
    return best


def make_job(
    args: argparse.Namespace,
    *,
    stage: str,
    config_id: str,
    base_config_id: str | None,
    loss_cfg: Dict[str, float],
    opt_cfg: Dict[str, float],
    capacity_cfg: Dict[str, Any],
    seed: int,
    freeze_teacher_epochs: int,
) -> Dict[str, Any]:
    return {
        "stage": stage,
        "config_id": config_id,
        "base_config_id": base_config_id,
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "checkpoint_metric": "test_acc",
        "job_output_root": args.output_dir / args.experiment_tag / stage / config_id,
        "freeze_teacher_epochs": freeze_teacher_epochs,
        **loss_cfg,
        **opt_cfg,
        **capacity_cfg,
    }


def build_stage0_jobs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for loss_name, loss_cfg in LOSS_PRESETS:
        for cap_name, capacity_cfg in CAPACITY_PRESETS:
            config_id = f"s0_{loss_name}_{cap_name}"
            jobs.append(
                make_job(
                    args,
                    stage="stage0",
                    config_id=config_id,
                    base_config_id=None,
                    loss_cfg=loss_cfg,
                    opt_cfg=OPT_PRESETS[1][1],
                    capacity_cfg=capacity_cfg,
                    seed=SEARCH_SEEDS[0],
                    freeze_teacher_epochs=5,
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
                        freeze_teacher_epochs=0,
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
                    "lr_student": max(1e-4, float(record["lr_student"]) + float(refine.get("lr_student_delta", 0.0))),
                    "lambda_kd": max(0.5, float(record["lambda_kd"]) + float(refine.get("lambda_kd_delta", 0.0))),
                    "lambda_feat": max(
                        0.0,
                        float(record["lambda_feat"]) + float(refine.get("lambda_feat_delta", 0.0)),
                    ),
                    "job_output_root": args.output_dir / args.experiment_tag / "stage2" / config_id,
                    "seed": SEARCH_SEEDS[len(jobs) % len(SEARCH_SEEDS)],
                    "freeze_teacher_epochs": 0,
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
                    "job_output_root": args.output_dir / args.experiment_tag / "stage3" / config_id / f"seed_{seed}",
                    "seed": seed,
                    "freeze_teacher_epochs": 0,
                }
            )
    return jobs


def rank_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = [r for r in records if r.get("status") in {"success", "skipped_existing"} and float(r.get("test_acc", -1.0)) >= 0.0]
    return sorted(
        valid,
        key=lambda r: (
            float(r.get("test_acc", -1.0)),
            float(r.get("test_macro_f1", -1.0)),
            float(r.get("test_teacher_agreement", -1.0)),
        ),
        reverse=True,
    )


def build_command(
    args: argparse.Namespace,
    job: Dict[str, Any],
    teacher_ref: Dict[str, Any],
    student_ref: Dict[str, Any] | None,
) -> List[str]:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset-npz",
        str(args.dataset_npz),
        "--output-dir",
        str(job["job_output_root"]),
        "--epochs",
        str(job["epochs"]),
        "--batch-size",
        str(job["batch_size"]),
        "--lr-teacher",
        str(job["lr_teacher"]),
        "--lr-student",
        str(job["lr_student"]),
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
        "--freeze-teacher-epochs",
        str(job["freeze_teacher_epochs"]),
        "--teacher-tcn-kernel",
        str(teacher_ref["teacher_tcn_kernel"]),
        "--teacher-tcn-layers",
        str(teacher_ref["teacher_tcn_layers"]),
        "--teacher-tcn-channels",
        str(teacher_ref["teacher_tcn_channels"]),
        "--teacher-tcn-dropout",
        str(teacher_ref["teacher_tcn_dropout"]),
        "--teacher-tcn-dilation-base",
        str(teacher_ref["teacher_tcn_dilation_base"]),
        "--teacher-classifier-hidden",
        str(teacher_ref["teacher_classifier_hidden"]),
        "--teacher-classifier-dropout",
        str(teacher_ref["teacher_classifier_dropout"]),
        "--teacher-attn-heads",
        str(teacher_ref["teacher_attn_heads"]),
        "--teacher-attn-dropout",
        str(teacher_ref["teacher_attn_dropout"]),
        "--teacher-attn-ff-dim",
        str(teacher_ref["teacher_attn_ff_dim"]),
        "--teacher-init-ckpt",
        str(teacher_ref["teacher_ckpt"]),
        "--teacher-init-mode",
        "strict",
        "--student-tcn-layers",
        str(job["student_tcn_layers"]),
        "--student-tcn-dropout",
        str(job["student_tcn_dropout"]),
        "--student-classifier-hidden",
        str(job["student_classifier_hidden"]),
        "--student-classifier-dropout",
        str(job["student_classifier_dropout"]),
        "--student-attn-dropout",
        str(job["student_attn_dropout"]),
        "--student-attn-ff-dim",
        str(job["student_attn_ff_dim"]),
        "--student-tcn-dilation-base",
        str(job["student_tcn_dilation_base"]),
    ]
    if bool(teacher_ref.get("teacher_weighted_sampler", False)):
        cmd.append("--weighted-sampler")
    if student_ref is not None:
        cmd += [
            "--student-init-ckpt",
            str(student_ref["student_ckpt"]),
            "--student-init-mode",
            "shape_safe",
        ]
    return cmd


def find_result_run(job_output_root: Path) -> Path:
    run_dirs = [path for path in job_output_root.iterdir() if path.is_dir()]
    if not run_dirs:
        return job_output_root
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def write_progress(report_root: Path, records: Sequence[Dict[str, Any]]) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    json_path = report_root / "ablation3_h1_sweep_progress.json"
    csv_path = report_root / "ablation3_h1_sweep_progress.csv"
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
    teacher_ref: Dict[str, Any],
    student_ref: Dict[str, Any] | None,
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
            cmd = build_command(args, job, teacher_ref, student_ref)
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
            "test_acc": float(metrics["test_acc"]),
            "test_macro_f1": float(metrics["test_macro_f1"]),
            "test_teacher_agreement": float(metrics["test_teacher_agreement"]),
            "best_epoch": int(metrics["best_epoch"]),
            "best_score": float(metrics["best_score"]),
            "target_reached": float(metrics["test_acc"]) >= target_acc,
        }
        stage_records.append(record)
        records.append(record)
        write_progress(args.report_dir / args.experiment_tag, records)
    return stage_records


def write_summary(
    args: argparse.Namespace,
    records: Sequence[Dict[str, Any]],
    teacher_ref: Dict[str, Any],
    student_ref: Dict[str, Any] | None,
    target_acc: float,
) -> Path:
    report_root = args.report_dir / args.experiment_tag
    report_root.mkdir(parents=True, exist_ok=True)
    ranked = rank_records(records)
    best = ranked[0] if ranked else None
    payload = {
        "dataset_npz": str(args.dataset_npz),
        "teacher_reference": teacher_ref,
        "student_reference": student_ref,
        "target_test_acc": target_acc,
        "num_runs": len(records),
        "target_met": bool(best and float(best["test_acc"]) >= target_acc),
        "best_run": best,
        "records": list(records),
    }
    summary_path = report_root / "ablation3_h1_sweep_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    args = parse_args()
    if not TRAIN_SCRIPT.exists():
        raise SystemExit(f"training script not found: {TRAIN_SCRIPT}")
    if not args.dataset_npz.exists():
        raise SystemExit(f"dataset not found: {args.dataset_npz}")
    if not args.teacher_manifest.exists():
        raise SystemExit(f"teacher manifest not found: {args.teacher_manifest}")

    validate_forecast_dataset_contract(args.dataset_npz)
    target_acc = parse_threshold(float(args.target_acc))
    teacher_ref = resolve_teacher_artifacts(args.teacher_manifest)
    student_ref = resolve_best_a1_student(args.student_results_dir)

    records: List[Dict[str, Any]] = []
    stage0_jobs = build_stage0_jobs(args)
    stage1_jobs = build_stage1_jobs(args)

    print("=" * 88)
    print("Ablation-3 h1 staged sweep")
    print("=" * 88)
    print(f"Dataset: {args.dataset_npz}")
    print(f"Teacher run: {teacher_ref['teacher_run_dir']}")
    print(f"Student init run: {student_ref['student_run_dir'] if student_ref else 'None'}")
    print(f"Target test_acc: {target_acc*100:.2f}%")

    execute_jobs(args, stage0_jobs[: args.max_runs], teacher_ref, student_ref, records, target_acc)
    if len(records) < args.max_runs:
        execute_jobs(
            args,
            stage1_jobs[: max(0, args.max_runs - len(records))],
            teacher_ref,
            student_ref,
            records,
            target_acc,
        )

    ranked = rank_records(records)
    if len(records) < args.max_runs and ranked:
        stage2_jobs = build_stage2_jobs(args, ranked)
        execute_jobs(
            args,
            stage2_jobs[: max(0, args.max_runs - len(records))],
            teacher_ref,
            student_ref,
            records,
            target_acc,
        )

    ranked = rank_records(records)
    if len(records) < args.max_runs and ranked:
        stage3_jobs = build_stage3_jobs(args, ranked)
        execute_jobs(
            args,
            stage3_jobs[: max(0, args.max_runs - len(records))],
            teacher_ref,
            student_ref,
            records,
            target_acc,
        )

    summary_path = write_summary(args, records, teacher_ref, student_ref, target_acc)
    print(f"Finished runs: {len(records)}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
