#!/usr/bin/env python3
"""Run a staged h1 distillation scan for the teacher-student comparison reference."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "train_distill_single_tcn_student.py"
DEFAULT_EXPERIMENT_TAG = "student_h1_strict_valf1_20260522"
SUCCESS_STATUSES = {"success", "skipped_existing"}
BASE_STAGE_RUN_BUDGET = 84
FINAL_STAGE_RUN_BUDGET = 108

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training_utils import parse_channels, validate_forecast_dataset_contract  # noqa: E402

TEST_ACC_PATTERN = re.compile(
    r"---\s*Test Metrics(?: \(Student\))?\s*---.*?Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%",
    flags=re.S,
)
TEST_F1_PATTERN = re.compile(
    r"---\s*Test Metrics(?: \(Student\))?\s*---.*?Macro-F1:\s*([0-9]+(?:\.[0-9]+)?)",
    flags=re.S,
)
VAL_F1_PATTERN = re.compile(
    r"---\s*Val Metrics(?: \(Student\))?\s*---.*?Macro-F1:\s*([0-9]+(?:\.[0-9]+)?)",
    flags=re.S,
)
TEST_AGREE_PATTERN = re.compile(r"Test Teacher-Agreement:\s*([0-9]+(?:\.[0-9]+)?)%")
BEST_EPOCH_PATTERN = re.compile(r"Best epoch:\s*(-?[0-9]+)")
BEST_SCORE_PATTERN = re.compile(r"Best score:\s*([0-9]+(?:\.[0-9]+)?)")
CHECKPOINT_PATTERN = re.compile(r"Checkpoint metric:\s*([A-Za-z0-9_]+)")

LOSS_PRESETS: Sequence[Tuple[str, Dict[str, float]]] = (
    ("l1", {"temperature": 2.5, "lambda_ce": 1.0, "lambda_kd": 1.0, "lambda_feat": 0.1}),
    ("l2", {"temperature": 3.0, "lambda_ce": 0.8, "lambda_kd": 1.2, "lambda_feat": 0.2}),
    ("l3", {"temperature": 3.5, "lambda_ce": 0.7, "lambda_kd": 1.4, "lambda_feat": 0.3}),
    ("l4", {"temperature": 2.0, "lambda_ce": 0.9, "lambda_kd": 1.1, "lambda_feat": 0.0}),
)
OPT_PRESETS: Sequence[Tuple[str, Dict[str, float]]] = (
    ("o1", {"lr": 1.5e-4, "weight_decay": 1.5e-4, "student_tcn_dropout": 0.05}),
    ("o2", {"lr": 2.0e-4, "weight_decay": 1.5e-4, "student_tcn_dropout": 0.08}),
    ("o3", {"lr": 2.5e-4, "weight_decay": 2.0e-4, "student_tcn_dropout": 0.08}),
    ("o4", {"lr": 3.0e-4, "weight_decay": 1.5e-4, "student_tcn_dropout": 0.12}),
)
CAPACITY_PRESETS: Sequence[Tuple[str, Dict[str, float | int | None]]] = (
    (
        "c0",
        {
            "student_tcn_channels": None,
            "student_tcn_layers": None,
            "student_tcn_dropout": None,
            "student_classifier_hidden": 64,
            "student_classifier_dropout": 0.40,
            "student_attn_dropout": 0.10,
            "student_attn_ff_dim": None,
            "student_tcn_dilation_base": 2,
        },
    ),
    (
        "c1",
        {
            "student_tcn_channels": None,
            "student_tcn_layers": None,
            "student_tcn_dropout": 0.05,
            "student_classifier_hidden": 64,
            "student_classifier_dropout": 0.30,
            "student_attn_dropout": 0.05,
            "student_attn_ff_dim": None,
            "student_tcn_dilation_base": 2,
        },
    ),
    (
        "c2",
        {
            "student_tcn_channels": None,
            "student_tcn_layers": None,
            "student_tcn_dropout": 0.12,
            "student_classifier_hidden": 64,
            "student_classifier_dropout": 0.40,
            "student_attn_dropout": 0.10,
            "student_attn_ff_dim": None,
            "student_tcn_dilation_base": 2,
        },
    ),
    (
        "c3",
        {
            "student_tcn_channels": 48,
            "student_tcn_layers": None,
            "student_tcn_dropout": None,
            "student_classifier_hidden": 64,
            "student_classifier_dropout": 0.30,
            "student_attn_dropout": 0.05,
            "student_attn_ff_dim": None,
            "student_tcn_dilation_base": 2,
        },
    ),
    (
        "c4",
        {
            "student_tcn_channels": 96,
            "student_tcn_layers": None,
            "student_tcn_dropout": None,
            "student_classifier_hidden": 128,
            "student_classifier_dropout": 0.30,
            "student_attn_dropout": 0.05,
            "student_attn_ff_dim": None,
            "student_tcn_dilation_base": 2,
        },
    ),
    (
        "c5",
        {
            "student_tcn_channels": 64,
            "student_tcn_layers": 2,
            "student_tcn_dropout": None,
            "student_classifier_hidden": 64,
            "student_classifier_dropout": 0.30,
            "student_attn_dropout": 0.05,
            "student_attn_ff_dim": None,
            "student_tcn_dilation_base": 2,
        },
    ),
    (
        "c6",
        {
            "student_tcn_channels": 64,
            "student_tcn_layers": None,
            "student_tcn_dropout": None,
            "student_classifier_hidden": 128,
            "student_classifier_dropout": 0.25,
            "student_attn_dropout": 0.05,
            "student_attn_ff_dim": 192,
            "student_tcn_dilation_base": 2,
        },
    ),
    (
        "c7",
        {
            "student_tcn_channels": 64,
            "student_tcn_layers": None,
            "student_tcn_dropout": None,
            "student_classifier_hidden": 64,
            "student_classifier_dropout": 0.30,
            "student_attn_dropout": 0.05,
            "student_attn_ff_dim": None,
            "student_tcn_dilation_base": 1,
        },
    ),
)
LOCAL_REFINEMENTS: Sequence[Tuple[str, Dict[str, Any]]] = (
    ("r1", {"lr_delta": -5e-5}),
    ("r2", {"lr_delta": 5e-5}),
    ("r3", {"temperature_delta": 0.5}),
    ("r4", {"lambda_ce_delta": -0.1, "lambda_kd_delta": 0.2, "lambda_feat_delta": 0.0}),
    ("r5", {"lambda_ce_delta": 0.0, "lambda_kd_delta": 0.1, "lambda_feat_delta": 0.1}),
)
STAGE4_SEEDS = (7, 21, 77, 122, 183)
EXTENSION_SEEDS = (132, 230, 314, 512, 777, 1001)
OPTIONAL_OVERRIDE_FLAGS = {
    "student_tcn_dropout": "--student-tcn-dropout",
    "student_tcn_layers": "--student-tcn-layers",
    "student_tcn_channels": "--student-tcn-channels",
    "student_classifier_hidden": "--student-classifier-hidden",
    "student_classifier_dropout": "--student-classifier-dropout",
    "student_attn_dropout": "--student-attn-dropout",
    "student_attn_ff_dim": "--student-attn-ff-dim",
    "student_tcn_dilation_base": "--student-tcn-dilation-base",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged h1 distillation sweep for comparison")
    parser.add_argument(
        "--dataset-npz",
        type=Path,
        default=PROJECT_ROOT / "Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz",
    )
    parser.add_argument(
        "--teacher-manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs/teacher_h1_forecast_95/manifests/best_teacher_h1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "ablation_experiments/h1/results/comparison_18d_baselines/teacher_student_reference/student_h1_strict_valf1_20260522",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT
        / "ablation_experiments/h1/reports/comparison_18d_baselines/teacher_student_reference/student_h1_strict_valf1_20260522",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--retry-on-fail", type=int, default=0)
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--gpu-list", type=str, default="1,0")
    parser.add_argument(
        "--ranking-metric",
        type=str,
        choices=("val_macro_f1", "test_acc", "test_macro_f1", "test_teacher_agreement"),
        default="val_macro_f1",
    )
    parser.add_argument("--target-test-acc", type=float, default=0.96)
    parser.add_argument("--experiment-tag", type=str, default=DEFAULT_EXPERIMENT_TAG)
    parser.add_argument("--max-runs", type=int, default=FINAL_STAGE_RUN_BUDGET)
    parser.set_defaults(skip_existing=True)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def normalize_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, Path):
            normalized[key] = str(value)
        elif isinstance(value, (dict, list, tuple)):
            normalized[key] = json.dumps(json_safe(value), ensure_ascii=True, sort_keys=True)
        else:
            normalized[key] = value
    return normalized


def resolve_teacher_artifacts(manifest_path: Path) -> Tuple[Path, Path]:
    payload = load_json(manifest_path)
    best_run = payload.get("best_run", {})
    if not isinstance(best_run, dict):
        raise ValueError("teacher manifest best_run must be an object")

    run_args_path = Path(str(best_run["run_args_path"]))
    run_dir = Path(str(best_run["path"]))
    candidate_ckpts = [
        run_dir / "best_single_tcn.pth",
        run_dir / "best_teacher_joint_tcn_attn.pth",
        run_dir / "best_teacher_lstm.pth",
    ]
    for ckpt in candidate_ckpts:
        if ckpt.exists():
            return ckpt, run_args_path
    raise FileNotFoundError(f"could not resolve teacher checkpoint from manifest: {manifest_path}")


def resolve_teacher_tcn_channels(teacher_run_args: Path) -> int | None:
    if not teacher_run_args.exists():
        return None
    payload = load_json(teacher_run_args)
    if str(payload.get("model", "")) != "tcn_attn":
        return None
    latent_dim = int(payload.get("latent_dim", 64))
    tcn_layers = int(payload.get("tcn_layers", 3))
    channels = parse_channels(payload.get("tcn_channels"), latent_dim=latent_dim, min_layers=tcn_layers)
    return int(channels[-1]) if channels else None


def parse_run_artifacts(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "evaluation_metrics.txt"
    run_args_path = run_dir / "run_args.json"
    parsed: Dict[str, Any] = {
        "val_macro_f1": -1.0,
        "test_acc": -1.0,
        "test_macro_f1": -1.0,
        "test_teacher_agreement": -1.0,
        "best_epoch": -1,
        "best_score": -1.0,
        "checkpoint_metric": "",
        "student_config_resolved": {},
    }
    if metrics_path.exists():
        text = metrics_path.read_text(encoding="utf-8", errors="ignore")
        m_val_f1 = VAL_F1_PATTERN.search(text)
        m_acc = TEST_ACC_PATTERN.search(text)
        m_f1 = TEST_F1_PATTERN.search(text)
        m_agree = TEST_AGREE_PATTERN.search(text)
        m_epoch = BEST_EPOCH_PATTERN.search(text)
        m_score = BEST_SCORE_PATTERN.search(text)
        m_checkpoint = CHECKPOINT_PATTERN.search(text)
        parsed["val_macro_f1"] = float(m_val_f1.group(1)) if m_val_f1 else -1.0
        parsed["test_acc"] = float(m_acc.group(1)) / 100.0 if m_acc else -1.0
        parsed["test_macro_f1"] = float(m_f1.group(1)) if m_f1 else -1.0
        parsed["test_teacher_agreement"] = float(m_agree.group(1)) / 100.0 if m_agree else -1.0
        parsed["best_epoch"] = int(m_epoch.group(1)) if m_epoch else -1
        parsed["best_score"] = float(m_score.group(1)) if m_score else -1.0
        parsed["checkpoint_metric"] = m_checkpoint.group(1) if m_checkpoint else ""
    if run_args_path.exists():
        run_args = load_json(run_args_path)
        parsed["student_config_resolved"] = run_args.get("student_config_resolved", {})
        if not parsed["checkpoint_metric"]:
            parsed["checkpoint_metric"] = str(run_args.get("checkpoint_metric", ""))
    return parsed


def build_run_name(dataset_tag: str, job: Dict[str, Any]) -> str:
    return (
        f"distill_tcn_attn_{dataset_tag}_ep{job['epochs']}_lr{job['lr']}_bs{job['batch_size']}_"
        f"T{job['temperature']}_lce{job['lambda_ce']}_lkd{job['lambda_kd']}_"
        f"lf{job['lambda_feat']}_seed{job['seed']}"
    )


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
        "--drop-feature-indices",
        "3,4,5,6,7",
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
        "--class-weights",
        "auto",
        "--weighted-sampler",
        "--val-ratio",
        "0.15",
        "--early-stop-patience",
        "16",
        "--min-epochs",
        "12",
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
    ]
    for key, flag in OPTIONAL_OVERRIDE_FLAGS.items():
        value = job.get(key)
        if value is None:
            continue
        cmd.extend([flag, str(value)])
    return cmd


def write_records(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(json_safe(rows), indent=2), encoding="utf-8")
        return
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    normalized_rows = [normalize_csv_row(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(normalized_rows[0].keys()))
        writer.writeheader()
        writer.writerows(normalized_rows)


def parse_gpu_list(value: str) -> List[str]:
    gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("gpu-list must contain at least one GPU id")
    return gpu_ids


def stage_output_root(args: argparse.Namespace, stage: str, config_id: str) -> Path:
    return Path(args.output_dir) / stage / config_id


def stage_report_root(args: argparse.Namespace) -> Path:
    return Path(args.report_dir)


def create_job(
    args: argparse.Namespace,
    *,
    stage: str,
    config_id: str,
    base_config_id: str,
    checkpoint_metric: str,
    seed: int,
    temperature: float,
    lambda_ce: float,
    lambda_kd: float,
    lambda_feat: float,
    lr: float,
    weight_decay: float,
    student_tcn_dropout: float | None = None,
    student_tcn_layers: int | None = None,
    student_tcn_channels: int | None = None,
    student_classifier_hidden: int | None = None,
    student_classifier_dropout: float | None = None,
    student_attn_dropout: float | None = None,
    student_attn_ff_dim: int | None = None,
    student_tcn_dilation_base: int | None = None,
) -> Dict[str, Any]:
    return {
        "stage": stage,
        "config_id": config_id,
        "base_config_id": base_config_id,
        "checkpoint_metric": checkpoint_metric,
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "temperature": temperature,
        "lambda_ce": lambda_ce,
        "lambda_kd": lambda_kd,
        "lambda_feat": lambda_feat,
        "lr": lr,
        "weight_decay": weight_decay,
        "student_tcn_dropout": student_tcn_dropout,
        "student_tcn_layers": student_tcn_layers,
        "student_tcn_channels": student_tcn_channels,
        "student_classifier_hidden": student_classifier_hidden,
        "student_classifier_dropout": student_classifier_dropout,
        "student_attn_dropout": student_attn_dropout,
        "student_attn_ff_dim": student_attn_ff_dim,
        "student_tcn_dilation_base": student_tcn_dilation_base,
        "job_output_root": stage_output_root(args, stage, config_id),
    }


def build_stage0_jobs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for seed in (14, 42, 77, 122):
        config_id = f"s0_val_macro_f1_seed{seed}"
        jobs.append(
            create_job(
                args,
                stage="stage0",
                config_id=config_id,
                base_config_id=config_id,
                checkpoint_metric="val_macro_f1",
                seed=seed,
                temperature=3.0,
                lambda_ce=0.8,
                lambda_kd=1.2,
                lambda_feat=0.2,
                lr=2e-4,
                weight_decay=1.5e-4,
            )
        )
    return jobs


def build_stage1_jobs(args: argparse.Namespace, checkpoint_metric: str) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for loss_name, loss_cfg in LOSS_PRESETS:
        for opt_name, opt_cfg in OPT_PRESETS:
            config_id = f"s1_{loss_name}_{opt_name}"
            jobs.append(
                create_job(
                    args,
                    stage="stage1",
                    config_id=config_id,
                    base_config_id=config_id,
                    checkpoint_metric=checkpoint_metric,
                    seed=14,
                    temperature=float(loss_cfg["temperature"]),
                    lambda_ce=float(loss_cfg["lambda_ce"]),
                    lambda_kd=float(loss_cfg["lambda_kd"]),
                    lambda_feat=float(loss_cfg["lambda_feat"]),
                    lr=float(opt_cfg["lr"]),
                    weight_decay=float(opt_cfg["weight_decay"]),
                    student_tcn_dropout=float(opt_cfg["student_tcn_dropout"]),
                )
            )
    return jobs


def build_stage2_jobs(args: argparse.Namespace, top_stage1_records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for base in pick_top_configs(top_stage1_records, top_k=3, ranking_metric="val_macro_f1"):
        for capacity_name, capacity_cfg in CAPACITY_PRESETS:
            config_id = f"s2_{base['config_id']}_{capacity_name}"
            jobs.append(
                create_job(
                    args,
                    stage="stage2",
                    config_id=config_id,
                    base_config_id=config_id,
                    checkpoint_metric=str(base["checkpoint_metric"]),
                    seed=int(base["seed"]),
                    temperature=float(base["temperature"]),
                    lambda_ce=float(base["lambda_ce"]),
                    lambda_kd=float(base["lambda_kd"]),
                    lambda_feat=float(base["lambda_feat"]),
                    lr=float(base["lr"]),
                    weight_decay=float(base["weight_decay"]),
                    student_tcn_dropout=_maybe_float(capacity_cfg["student_tcn_dropout"], fallback=base.get("student_tcn_dropout")),
                    student_tcn_layers=_maybe_int(capacity_cfg["student_tcn_layers"], fallback=base.get("student_tcn_layers")),
                    student_tcn_channels=_maybe_int(capacity_cfg["student_tcn_channels"], fallback=base.get("student_tcn_channels")),
                    student_classifier_hidden=_maybe_int(
                        capacity_cfg["student_classifier_hidden"],
                        fallback=base.get("student_classifier_hidden"),
                    ),
                    student_classifier_dropout=_maybe_float(
                        capacity_cfg["student_classifier_dropout"],
                        fallback=base.get("student_classifier_dropout"),
                    ),
                    student_attn_dropout=_maybe_float(
                        capacity_cfg["student_attn_dropout"],
                        fallback=base.get("student_attn_dropout"),
                    ),
                    student_attn_ff_dim=_maybe_int(
                        capacity_cfg["student_attn_ff_dim"],
                        fallback=base.get("student_attn_ff_dim"),
                    ),
                    student_tcn_dilation_base=_maybe_int(
                        capacity_cfg["student_tcn_dilation_base"],
                        fallback=base.get("student_tcn_dilation_base"),
                    ),
                )
            )
    return jobs


def build_stage3_jobs(args: argparse.Namespace, top_stage2_records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for base in pick_top_configs(top_stage2_records, top_k=4, ranking_metric="val_macro_f1"):
        base_lr = float(base["lr"])
        base_temperature = float(base["temperature"])
        base_lambda_ce = float(base["lambda_ce"])
        base_lambda_kd = float(base["lambda_kd"])
        base_lambda_feat = float(base["lambda_feat"])
        for refinement_name, refinement in LOCAL_REFINEMENTS:
            lr = min(max(base_lr + float(refinement.get("lr_delta", 0.0)), 1.0e-4), 3.5e-4)
            temperature = min(max(base_temperature + float(refinement.get("temperature_delta", 0.0)), 2.0), 4.0)
            lambda_ce = min(max(base_lambda_ce + float(refinement.get("lambda_ce_delta", 0.0)), 0.6), 1.0)
            lambda_kd = min(max(base_lambda_kd + float(refinement.get("lambda_kd_delta", 0.0)), 0.8), 1.8)
            lambda_feat = min(max(base_lambda_feat + float(refinement.get("lambda_feat_delta", 0.0)), 0.0), 0.3)
            config_id = f"s3_{base['config_id']}_{refinement_name}"
            jobs.append(
                create_job(
                    args,
                    stage="stage3",
                    config_id=config_id,
                    base_config_id=config_id,
                    checkpoint_metric=str(base["checkpoint_metric"]),
                    seed=int(base["seed"]),
                    temperature=temperature,
                    lambda_ce=lambda_ce,
                    lambda_kd=lambda_kd,
                    lambda_feat=lambda_feat,
                    lr=lr,
                    weight_decay=float(base["weight_decay"]),
                    student_tcn_dropout=_maybe_float(base.get("student_tcn_dropout")),
                    student_tcn_layers=_maybe_int(base.get("student_tcn_layers")),
                    student_tcn_channels=_maybe_int(base.get("student_tcn_channels")),
                    student_classifier_hidden=_maybe_int(base.get("student_classifier_hidden")),
                    student_classifier_dropout=_maybe_float(base.get("student_classifier_dropout")),
                    student_attn_dropout=_maybe_float(base.get("student_attn_dropout")),
                    student_attn_ff_dim=_maybe_int(base.get("student_attn_ff_dim")),
                    student_tcn_dilation_base=_maybe_int(base.get("student_tcn_dilation_base")),
                )
            )
    return jobs


def build_seed_harvest_jobs(
    args: argparse.Namespace,
    ranked_records: Sequence[Dict[str, Any]],
    *,
    stage_name: str,
    seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for base in pick_top_configs(ranked_records, top_k=4, ranking_metric="val_macro_f1"):
        base_key = str(base.get("base_config_id") or base["config_id"])
        config_id = f"{stage_name}_{base_key}"
        for seed in seeds:
            jobs.append(
                create_job(
                    args,
                    stage=stage_name,
                    config_id=config_id,
                    base_config_id=base_key,
                    checkpoint_metric=str(base["checkpoint_metric"]),
                    seed=int(seed),
                    temperature=float(base["temperature"]),
                    lambda_ce=float(base["lambda_ce"]),
                    lambda_kd=float(base["lambda_kd"]),
                    lambda_feat=float(base["lambda_feat"]),
                    lr=float(base["lr"]),
                    weight_decay=float(base["weight_decay"]),
                    student_tcn_dropout=_maybe_float(base.get("student_tcn_dropout")),
                    student_tcn_layers=_maybe_int(base.get("student_tcn_layers")),
                    student_tcn_channels=_maybe_int(base.get("student_tcn_channels")),
                    student_classifier_hidden=_maybe_int(base.get("student_classifier_hidden")),
                    student_classifier_dropout=_maybe_float(base.get("student_classifier_dropout")),
                    student_attn_dropout=_maybe_float(base.get("student_attn_dropout")),
                    student_attn_ff_dim=_maybe_int(base.get("student_attn_ff_dim")),
                    student_tcn_dilation_base=_maybe_int(base.get("student_tcn_dilation_base")),
                )
            )
    return jobs


def _maybe_float(value: Any, fallback: Any = None) -> float | None:
    target = fallback if value is None else value
    if target is None:
        return None
    return float(target)


def _maybe_int(value: Any, fallback: Any = None) -> int | None:
    target = fallback if value is None else value
    if target is None:
        return None
    return int(target)


def metric_sort_key(row: Dict[str, Any], ranking_metric: str) -> Tuple[float, float, float]:
    val_macro_f1 = float(row.get("val_macro_f1", row.get("best_score", -1.0)))
    test_acc = float(row.get("test_acc", -1.0))
    macro_f1 = float(row.get("test_macro_f1", row.get("macro_f1", -1.0)))
    teacher_agreement = float(row.get("test_teacher_agreement", row.get("teacher_agreement", -1.0)))
    if ranking_metric == "val_macro_f1":
        return (val_macro_f1, macro_f1, test_acc)
    if ranking_metric == "test_macro_f1":
        return (macro_f1, test_acc, teacher_agreement)
    if ranking_metric == "test_teacher_agreement":
        return (teacher_agreement, test_acc, macro_f1)
    return (test_acc, macro_f1, teacher_agreement)


def rank_records(records: Sequence[Dict[str, Any]], ranking_metric: str) -> List[Dict[str, Any]]:
    filtered = [
        row
        for row in records
        if "status" not in row or str(row.get("status")) in SUCCESS_STATUSES
    ]
    return sorted(filtered, key=lambda row: metric_sort_key(row, ranking_metric), reverse=True)


def pick_top_configs(
    records: Sequence[Dict[str, Any]],
    *,
    top_k: int,
    ranking_metric: str,
) -> List[Dict[str, Any]]:
    ranked = rank_records(records, ranking_metric)
    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in ranked:
        unique_key = str(row.get("base_config_id") or row.get("config_id"))
        if unique_key in seen:
            continue
        seen.add(unique_key)
        selected.append(row)
        if len(selected) >= top_k:
            break
    return selected


def select_stage0_checkpoint_metric(records: Sequence[Dict[str, Any]]) -> str:
    ranked = rank_records(records, ranking_metric="val_macro_f1")
    if not ranked:
        raise ValueError("stage0 did not produce any successful records")
    return str(ranked[0]["checkpoint_metric"])


def write_progress(report_root: Path, records: List[Dict[str, Any]]) -> None:
    write_records(report_root / "distill_sweep_progress.json", records)
    write_records(report_root / "distill_sweep_progress.csv", records)


def execute_stage_jobs(
    args: argparse.Namespace,
    jobs: Sequence[Dict[str, Any]],
    teacher_ckpt: Path,
    teacher_run_args: Path,
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    report_root = stage_report_root(args)
    report_root.mkdir(parents=True, exist_ok=True)
    log_root = report_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    stage_records: List[Dict[str, Any]] = []
    gpu_ids = parse_gpu_list(args.gpu_list)
    dataset_tag = args.dataset_npz.stem
    teacher_tcn_channels = resolve_teacher_tcn_channels(teacher_run_args)

    for index, job in enumerate(jobs, start=1):
        run_name = build_run_name(dataset_tag, job)
        run_dir = Path(job["job_output_root"]) / run_name
        metrics_path = run_dir / "evaluation_metrics.txt"
        cmd = build_command(args, job, teacher_ckpt, teacher_run_args)
        gpu_id = gpu_ids[(len(records) + len(stage_records)) % len(gpu_ids)]
        log_path = log_root / f"{job['stage']}_{job['config_id']}_seed{job['seed']}.log"
        if args.print_command:
            print("Command:", " ".join(cmd))

        status = "failed"
        elapsed = 0.0
        attempts = 0
        if (
            teacher_tcn_channels is not None
            and job.get("student_tcn_channels") is not None
            and int(job["student_tcn_channels"]) != teacher_tcn_channels
        ):
            status = "invalid_feature_alignment"
        elif args.skip_existing and metrics_path.exists():
            status = "skipped_existing"
        else:
            while attempts <= args.retry_on_fail:
                attempts += 1
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
                start = time.time()
                with log_path.open("w", encoding="utf-8") as log_file:
                    proc = subprocess.run(
                        cmd,
                        check=False,
                        env=env,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                    )
                elapsed = time.time() - start
                status = "success" if proc.returncode == 0 else f"failed_rc_{proc.returncode}"
                if proc.returncode == 0:
                    break

        artifacts = parse_run_artifacts(run_dir)
        record = {
            **job,
            "stage_index": index,
            "status": status,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "elapsed_sec": round(elapsed, 2),
            "gpu_id": gpu_id,
            "teacher_manifest": str(args.teacher_manifest),
            "command": " ".join(cmd),
            "log_path": str(log_path),
            **artifacts,
        }
        stage_records.append(record)
        records.append(record)
        print(
            f"[{job['stage']}] {job['config_id']} seed={job['seed']} "
            f"| status={status} val_f1={float(record['val_macro_f1']):.4f} "
            f"test_acc={float(record['test_acc']) * 100:.2f}% "
            f"f1={float(record['test_macro_f1']):.4f}"
        )
        write_progress(report_root, records)
    return stage_records


def remaining_budget(args: argparse.Namespace, records: Sequence[Dict[str, Any]]) -> int:
    return max(int(args.max_runs) - len(records), 0)


def take_budget_slice(jobs: Sequence[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
    if budget <= 0:
        return []
    return list(jobs[:budget])


def build_summary(
    args: argparse.Namespace,
    teacher_ckpt: Path,
    teacher_run_args: Path,
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    ranked = rank_records(records, ranking_metric=args.ranking_metric)
    best_run = ranked[0] if ranked else None
    best_test_acc = float(best_run["test_acc"]) if best_run is not None else -1.0
    return {
        "dataset_npz": str(args.dataset_npz),
        "teacher_manifest": str(args.teacher_manifest),
        "teacher_ckpt": str(teacher_ckpt),
        "teacher_run_args": str(teacher_run_args),
        "experiment_tag": args.experiment_tag,
        "ranking_metric": args.ranking_metric,
        "target_test_acc": float(args.target_test_acc),
        "target_met": best_test_acc > float(args.target_test_acc),
        "num_runs": len(records),
        "base_stage_budget": BASE_STAGE_RUN_BUDGET,
        "max_runs": int(args.max_runs),
        "best_run": best_run,
        "top_runs": ranked[:5],
    }


def write_summary(
    args: argparse.Namespace,
    teacher_ckpt: Path,
    teacher_run_args: Path,
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    report_root = stage_report_root(args)
    report_root.mkdir(parents=True, exist_ok=True)
    ranked = rank_records(records, ranking_metric=args.ranking_metric)
    write_records(report_root / "distill_sweep_summary.csv", ranked[:10] if ranked else [])
    summary = build_summary(args, teacher_ckpt, teacher_run_args, records)
    (report_root / "distill_sweep_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    if not TRAIN_SCRIPT.exists():
        raise SystemExit(f"distill script not found: {TRAIN_SCRIPT}")

    validate_forecast_dataset_contract(args.dataset_npz)
    teacher_ckpt, teacher_run_args = resolve_teacher_artifacts(args.teacher_manifest)
    stage_report_root(args).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    stage0_jobs = take_budget_slice(build_stage0_jobs(args), remaining_budget(args, records))
    stage0_records = execute_stage_jobs(args, stage0_jobs, teacher_ckpt, teacher_run_args, records)
    if not stage0_records:
        write_summary(args, teacher_ckpt, teacher_run_args, records)
        return

    stage0_ranked = rank_records(stage0_records, ranking_metric="val_macro_f1")
    stage0_best_val_f1 = (
        float(stage0_ranked[0].get("val_macro_f1", stage0_ranked[0].get("best_score", -1.0)))
        if stage0_ranked
        else -1.0
    )
    if stage0_ranked and stage0_best_val_f1 < 0.80:
        raise SystemExit("stage0 peak val_macro_f1 stayed below 0.80; stop and inspect the h1 distill setup")

    checkpoint_metric = select_stage0_checkpoint_metric(stage0_records)

    stage1_jobs = take_budget_slice(build_stage1_jobs(args, checkpoint_metric), remaining_budget(args, records))
    stage1_records = execute_stage_jobs(args, stage1_jobs, teacher_ckpt, teacher_run_args, records)
    if not stage1_records or remaining_budget(args, records) <= 0:
        write_summary(args, teacher_ckpt, teacher_run_args, records)
        return

    stage2_jobs = take_budget_slice(build_stage2_jobs(args, stage1_records), remaining_budget(args, records))
    stage2_records = execute_stage_jobs(args, stage2_jobs, teacher_ckpt, teacher_run_args, records)
    if not stage2_records or remaining_budget(args, records) <= 0:
        write_summary(args, teacher_ckpt, teacher_run_args, records)
        return

    stage3_jobs = take_budget_slice(build_stage3_jobs(args, stage2_records), remaining_budget(args, records))
    stage3_records = execute_stage_jobs(args, stage3_jobs, teacher_ckpt, teacher_run_args, records)
    if not stage3_records or remaining_budget(args, records) <= 0:
        write_summary(args, teacher_ckpt, teacher_run_args, records)
        return

    stage4_jobs = take_budget_slice(
        build_seed_harvest_jobs(args, stage3_records, stage_name="stage4", seeds=STAGE4_SEEDS),
        remaining_budget(args, records),
    )
    stage4_records = execute_stage_jobs(args, stage4_jobs, teacher_ckpt, teacher_run_args, records)
    summary = write_summary(args, teacher_ckpt, teacher_run_args, records)
    if remaining_budget(args, records) <= 0:
        print("Distill comparison sweep complete.")
        print(f"Report dir: {stage_report_root(args)}")
        return

    extension_jobs = take_budget_slice(
        build_seed_harvest_jobs(args, stage4_records or stage3_records, stage_name="extension", seeds=EXTENSION_SEEDS),
        remaining_budget(args, records),
    )
    execute_stage_jobs(args, extension_jobs, teacher_ckpt, teacher_run_args, records)
    write_summary(args, teacher_ckpt, teacher_run_args, records)
    print("Distill comparison sweep complete.")
    print(f"Report dir: {stage_report_root(args)}")


if __name__ == "__main__":
    main()
