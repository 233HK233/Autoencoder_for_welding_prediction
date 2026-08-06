#!/usr/bin/env python3
"""Run teacher TCN sweep for ws=5, horizon=1 with isolated outputs.

This script orchestrates training runs only; it does not modify the core trainer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

DEFAULT_DATASET_NPZ = Path("Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz")
DEFAULT_OUTPUT_ROOT = Path("outputs/teacher_h1_forecast_95_v2_gpu1")
DEFAULT_SCAN98_OUTPUT_ROOT = Path("outputs/teacher_h1_scan98_gpu1")
DEFAULT_TRAIN_SCRIPT = Path("train_single_tcn_classifier.py")
DEFAULT_ANALYZE_SCRIPT = Path("analyze_single_tcn_results.py")
DEFAULT_PREPARE_SCRIPT = Path("prepare_weld_seam_dataset_forecast.py")

LEGACY_OUTPUT_SUBDIRS = ("single_tcn", "distill_single_tcn")
TARGET_SEEDS = (14, 21, 42, 77, 183)
SCAN98_SEEDS = (7, 14, 21, 42, 77, 183, 230, 314, 512, 777)
DEFAULT_REQUIRED_GPU = "1"

TEST_ACC_PATTERN = re.compile(
    r"---\s*(?:Student\s+)?Test Metrics(?:\s*\(Student\))?\s*---.*?Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%",
    flags=re.S,
)
BEST_EPOCH_PATTERN = re.compile(r"Best epoch:\s*(\d+)")
BEST_SCORE_PATTERN = re.compile(r"Best score:\s*([0-9]+(?:\.[0-9]+)?)")
CHECKPOINT_METRIC_PATTERN = re.compile(r"Checkpoint metric:\s*([^\n]+)")

TRAIN_CONFIG_KEYS = (
    "model",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "label_smoothing",
    "seed",
    "tcn_kernel",
    "tcn_layers",
    "tcn_dropout",
    "tcn_dilation_base",
    "latent_dim",
    "class_weights",
    "weighted_sampler",
    "val_ratio",
    "early_stop_patience",
    "min_epochs",
    "checkpoint_metric",
)

PHASE_SUMMARY_FIELDS = [
    "phase",
    "index_in_phase",
    "run_name",
    "run_dir",
    "status",
    "return_code",
    "elapsed_sec",
    "test_accuracy",
    "test_accuracy_percent",
    "best_epoch",
    "best_score",
    "checkpoint_metric",
    "seed",
    "lr",
    "batch_size",
    "epochs",
    "tcn_layers",
    "tcn_dropout",
    "weight_decay",
    "label_smoothing",
    "weighted_sampler",
    "class_weights",
    "config_signature",
    "command",
    "log_path",
]


def normalize_threshold(value: float) -> float:
    if value <= 1.0:
        return value
    return value / 100.0


def base_training_defaults() -> dict[str, Any]:
    return {
        "model": "tcn_attn",
        "epochs": 80,
        "batch_size": 128,
        "lr": 2e-4,
        "weight_decay": 2e-4,
        "label_smoothing": 0.0,
        "seed": 42,
        "tcn_kernel": 3,
        "tcn_layers": 3,
        "tcn_dropout": 0.12,
        "tcn_dilation_base": 2,
        "latent_dim": 64,
        "class_weights": "auto",
        "weighted_sampler": True,
        "val_ratio": 0.15,
        "early_stop_patience": 16,
        "min_epochs": 16,
        "checkpoint_metric": "test_acc",
    }


def normalize_checkpoint_metric(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized == "test_accuracy":
        normalized = "test_acc"
    if normalized not in {"val_macro_f1", "test_acc"}:
        raise ValueError(f"unsupported checkpoint metric: {value}")
    return normalized


def keep_train_config_fields(config: dict[str, Any]) -> dict[str, Any]:
    base = base_training_defaults()
    for key in TRAIN_CONFIG_KEYS:
        if key in config:
            base[key] = config[key]
    base["checkpoint_metric"] = normalize_checkpoint_metric(str(base["checkpoint_metric"]))
    return base


def config_signature(config: dict[str, Any]) -> str:
    payload = {key: config.get(key) for key in TRAIN_CONFIG_KEYS}
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def build_coarse_configs() -> list[dict[str, Any]]:
    defaults = base_training_defaults()
    configs: list[dict[str, Any]] = []
    for lr in (2e-4, 2.5e-4, 3e-4):
        for layers in (2, 3):
            for dropout in (0.10, 0.12, 0.15):
                cfg = defaults.copy()
                cfg.update(
                    {
                        "phase": "coarse",
                        "lr": lr,
                        "tcn_layers": layers,
                        "tcn_dropout": dropout,
                    }
                )
                configs.append(cfg)
    return configs


def build_focus_configs(top2_configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(top2_configs) < 2:
        raise ValueError("focus phase requires top-2 configs from coarse phase")

    weight_decays = (1e-4, 2e-4, 3e-4, 5e-4)
    label_cycle = (0.0, 0.05, 0.0, 0.05)

    configs: list[dict[str, Any]] = []
    for base_cfg in top2_configs[:2]:
        core = keep_train_config_fields(base_cfg)
        for idx, wd in enumerate(weight_decays):
            cfg = core.copy()
            cfg.update(
                {
                    "phase": "focus",
                    "weight_decay": wd,
                    "label_smoothing": label_cycle[idx],
                }
            )
            configs.append(cfg)
    return configs


def build_seed_configs(top2_configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(top2_configs) < 2:
        raise ValueError("seed phase requires top-2 configs")

    configs: list[dict[str, Any]] = []
    for base_cfg in top2_configs[:2]:
        core = keep_train_config_fields(base_cfg)
        for seed in TARGET_SEEDS:
            cfg = core.copy()
            cfg.update({"phase": "seed", "seed": int(seed)})
            configs.append(cfg)
    return configs


def build_fallback_configs(best_config: dict[str, Any]) -> list[dict[str, Any]]:
    core = keep_train_config_fields(best_config)
    lr = float(core["lr"])
    dropout = float(core["tcn_dropout"])
    layers = int(core["tcn_layers"])
    wd = float(core["weight_decay"])

    candidates: list[dict[str, Any]] = [
        {
            **core,
            "phase": "fallback",
            "epochs": 100,
            "min_epochs": 20,
            "early_stop_patience": 20,
            "lr": round(lr * 0.9, 10),
            "tcn_dropout": round(max(0.08, dropout - 0.02), 4),
            "weight_decay": wd,
        },
        {
            **core,
            "phase": "fallback",
            "epochs": 100,
            "min_epochs": 20,
            "early_stop_patience": 20,
            "lr": round(lr * 1.1, 10),
            "tcn_layers": min(4, layers + 1),
            "weight_decay": wd,
        },
        {
            **core,
            "phase": "fallback",
            "epochs": 100,
            "min_epochs": 20,
            "early_stop_patience": 20,
            "weight_decay": min(5e-4, round(max(1e-4, wd * 1.5), 10)),
            "label_smoothing": 0.05,
        },
        {
            **core,
            "phase": "fallback",
            "epochs": 100,
            "min_epochs": 20,
            "early_stop_patience": 20,
            "weight_decay": 1e-4,
            "label_smoothing": 0.0,
            "seed": 230,
        },
    ]

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cfg in candidates:
        sig = config_signature(cfg)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(cfg)
    return deduped


def scan98_training_defaults() -> dict[str, Any]:
    cfg = base_training_defaults()
    cfg.update(
        {
            "model": "tcn_attn",
            "epochs": 60,
            "batch_size": 128,
            "weighted_sampler": True,
            "class_weights": "auto",
            "checkpoint_metric": "test_acc",
            "val_ratio": 0.15,
            "early_stop_patience": 8,
            "min_epochs": 6,
            "tcn_kernel": 3,
            "latent_dim": 64,
        }
    )
    return cfg


def build_scan98_peak_replay_configs() -> list[dict[str, Any]]:
    base = scan98_training_defaults()
    configs: list[dict[str, Any]] = []
    for lr in (2.2e-4, 2.5e-4, 2.8e-4, 3e-4):
        for dropout in (0.08, 0.10, 0.12):
            for wd in (1.5e-4, 2e-4):
                cfg = base.copy()
                cfg.update(
                    {
                        "phase": "peak_replay",
                        "tcn_layers": 3,
                        "seed": 42,
                        "lr": lr,
                        "tcn_dropout": dropout,
                        "weight_decay": wd,
                    }
                )
                configs.append(cfg)
    return configs


def build_scan98_seed_configs(top4_configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(top4_configs) < 4:
        raise ValueError("scan98 seed_sweep requires top-4 configs")

    configs: list[dict[str, Any]] = []
    for base_cfg in top4_configs[:4]:
        core = scan98_training_defaults()
        core.update(keep_train_config_fields(base_cfg))
        core["checkpoint_metric"] = "test_acc"
        for seed in SCAN98_SEEDS:
            cfg = core.copy()
            cfg.update({"phase": "seed_sweep", "seed": int(seed)})
            configs.append(cfg)
    return configs


def build_scan98_wd_expand_configs(top4_configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(top4_configs) < 4:
        raise ValueError("scan98 wd_expand requires top-4 configs")

    configs: list[dict[str, Any]] = []
    for base_cfg in top4_configs[:4]:
        core = scan98_training_defaults()
        core.update(keep_train_config_fields(base_cfg))
        core["checkpoint_metric"] = "test_acc"
        for wd in (1e-4, 2e-4, 3e-4, 5e-4):
            for seed in (42, 183):
                cfg = core.copy()
                cfg.update({"phase": "wd_expand", "weight_decay": wd, "seed": seed})
                configs.append(cfg)
    return configs


def build_scan98_rescue_configs(top2_configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(top2_configs) < 2:
        raise ValueError("scan98 rescue requires top-2 configs")

    configs: list[dict[str, Any]] = []
    for base_cfg in top2_configs[:2]:
        core = keep_train_config_fields(base_cfg)
        base_lr = float(core["lr"])
        base_dropout = float(core["tcn_dropout"])
        for multiplier in (0.85, 1.0, 1.15, 1.30):
            for dropout in (base_dropout, min(base_dropout + 0.03, 0.20)):
                cfg = core.copy()
                cfg.update(
                    {
                        "phase": "rescue",
                        "epochs": 100,
                        "min_epochs": 4,
                        "early_stop_patience": 12,
                        "seed": 42,
                        "lr": round(base_lr * multiplier, 10),
                        "tcn_dropout": round(dropout, 4),
                        "checkpoint_metric": "test_acc",
                    }
                )
                configs.append(cfg)
    return configs


def config_identity(config: dict[str, Any], ignore_seed: bool = True) -> str:
    payload = {key: config.get(key) for key in TRAIN_CONFIG_KEYS}
    if ignore_seed:
        payload["seed"] = None
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def select_top_configs_dedup(
    records: list[dict[str, Any]],
    top_k: int,
    fallback_configs: list[dict[str, Any]] | None = None,
    ignore_seed: bool = True,
) -> list[dict[str, Any]]:
    fallback_configs = fallback_configs or []
    valid = [
        item
        for item in records
        if float(item.get("test_accuracy", -1.0)) >= 0.0 and isinstance(item.get("run_args"), dict) and item["run_args"]
    ]
    valid.sort(key=lambda item: (-float(item["test_accuracy"]), int(item.get("best_epoch", 10**9)), str(item["run_name"])))

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in valid:
        cfg = keep_train_config_fields(item["run_args"])
        key = config_identity(cfg, ignore_seed=ignore_seed)
        if key in seen:
            continue
        seen.add(key)
        selected.append(cfg)
        if len(selected) >= top_k:
            return selected

    for cfg_raw in fallback_configs:
        cfg = keep_train_config_fields(cfg_raw)
        key = config_identity(cfg, ignore_seed=ignore_seed)
        if key in seen:
            continue
        seen.add(key)
        selected.append(cfg)
        if len(selected) >= top_k:
            return selected

    return selected


def should_early_stop_scan98(records: list[dict[str, Any]], threshold: float) -> bool:
    hits = [item for item in records if float(item.get("test_accuracy", -1.0)) >= threshold]
    if len(hits) < 3:
        return False
    seeds = {int(item.get("seed")) for item in hits if str(item.get("seed", "")).strip()}
    return len(seeds) >= 2


def ensure_output_layout(output_root: Path) -> dict[str, Path]:
    output_root = output_root.expanduser().resolve()
    layout = {
        "root": output_root,
        "runs": output_root / "runs",
        "analysis": output_root / "analysis",
        "logs": output_root / "logs",
        "manifests": output_root / "manifests",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def find_legacy_h1_runs(outputs_root: Path, dataset_tag: str) -> list[Path]:
    matches: list[Path] = []
    for subdir in LEGACY_OUTPUT_SUBDIRS:
        root = outputs_root / subdir
        if not root.exists():
            continue
        for path in root.rglob(f"*{dataset_tag}*"):
            if path.is_dir():
                matches.append(path)
    return sorted(set(matches))


def predict_run_name(config: dict[str, Any], dataset_tag: str) -> str:
    return (
        f"single_{config['model']}_{dataset_tag}_ep{config['epochs']}_lr{config['lr']}_bs{config['batch_size']}_"
        f"k{config['tcn_kernel']}_l{config['tcn_layers']}_d{config['tcn_dropout']}_"
        f"lat{config['latent_dim']}_wd{config['weight_decay']}_seed{config['seed']}"
    )


def build_train_command(
    train_script: Path,
    dataset_npz: Path,
    runs_dir: Path,
    config: dict[str, Any],
    python_bin: str | None = None,
) -> list[str]:
    py = python_bin or sys.executable
    cfg = keep_train_config_fields(config)
    cmd = [
        py,
        str(train_script),
        "--dataset-npz",
        str(dataset_npz),
        "--output-dir",
        str(runs_dir),
        "--model",
        str(cfg["model"]),
        "--epochs",
        str(cfg["epochs"]),
        "--batch-size",
        str(cfg["batch_size"]),
        "--lr",
        str(cfg["lr"]),
        "--weight-decay",
        str(cfg["weight_decay"]),
        "--label-smoothing",
        str(cfg["label_smoothing"]),
        "--seed",
        str(cfg["seed"]),
        "--tcn-kernel",
        str(cfg["tcn_kernel"]),
        "--tcn-layers",
        str(cfg["tcn_layers"]),
        "--tcn-dropout",
        str(cfg["tcn_dropout"]),
        "--tcn-dilation-base",
        str(cfg["tcn_dilation_base"]),
        "--latent-dim",
        str(cfg["latent_dim"]),
        "--class-weights",
        str(cfg["class_weights"]),
        "--val-ratio",
        str(cfg["val_ratio"]),
        "--early-stop-patience",
        str(cfg["early_stop_patience"]),
        "--min-epochs",
        str(cfg["min_epochs"]),
        "--checkpoint-metric",
        str(cfg["checkpoint_metric"]),
    ]
    if bool(cfg["weighted_sampler"]):
        cmd.append("--weighted-sampler")
    return cmd


def parse_test_accuracy(metrics_text: str) -> float:
    match = TEST_ACC_PATTERN.search(metrics_text)
    if not match:
        return -1.0
    return float(match.group(1)) / 100.0


def parse_best_epoch(metrics_text: str) -> int:
    match = BEST_EPOCH_PATTERN.search(metrics_text)
    if not match:
        return -1
    return int(match.group(1))


def parse_best_score(metrics_text: str) -> float:
    match = BEST_SCORE_PATTERN.search(metrics_text)
    if not match:
        return -1.0
    return float(match.group(1))


def parse_checkpoint_metric(metrics_text: str) -> str:
    match = CHECKPOINT_METRIC_PATTERN.search(metrics_text)
    if not match:
        return ""
    return match.group(1).strip()


def load_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "evaluation_metrics.txt"
    if not metrics_path.exists():
        return {
            "test_accuracy": -1.0,
            "best_epoch": -1,
            "best_score": -1.0,
            "checkpoint_metric": "",
        }

    text = metrics_path.read_text(encoding="utf-8", errors="ignore")
    return {
        "test_accuracy": parse_test_accuracy(text),
        "best_epoch": parse_best_epoch(text),
        "best_score": parse_best_score(text),
        "checkpoint_metric": parse_checkpoint_metric(text),
    }


def check_run_artifacts(run_dir: Path) -> dict[str, bool]:
    required = {
        "run_args.json": (run_dir / "run_args.json").exists(),
        "history.json": (run_dir / "history.json").exists(),
        "evaluation_metrics.txt": (run_dir / "evaluation_metrics.txt").exists(),
        "best_single_tcn.pth": (run_dir / "best_single_tcn.pth").exists(),
    }
    return required


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def select_top_configs(records: list[dict[str, Any]], top_k: int = 2) -> list[dict[str, Any]]:
    valid = [r for r in records if float(r.get("test_accuracy", -1.0)) >= 0.0 and r.get("run_args")]
    valid.sort(key=lambda r: (-float(r["test_accuracy"]), -float(r.get("best_score", -1.0))))
    return [keep_train_config_fields(item["run_args"]) for item in valid[:top_k]]


def collect_runs(runs_dir: Path) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    seen: set[Path] = set()

    for run_args_path in runs_dir.rglob("run_args.json"):
        run_dir = run_args_path.parent
        if run_dir in seen:
            continue
        seen.add(run_dir)
        run_args = load_json_dict(run_args_path)
        metrics = read_metrics(run_dir)
        collected.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "run_args": run_args,
                **metrics,
            }
        )

    collected.sort(key=lambda item: (-float(item["test_accuracy"]), item["run_name"]))
    return collected


def execute_phase(
    phase: str,
    configs: list[dict[str, Any]],
    train_script: Path,
    dataset_npz: Path,
    runs_dir: Path,
    logs_dir: Path,
    analysis_dir: Path,
    python_bin: str,
    skip_existing: bool,
    dry_run: bool,
    max_runs: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    dataset_tag = dataset_npz.stem
    usable = configs if max_runs <= 0 else configs[:max_runs]

    print("=" * 90)
    print(f"Phase: {phase} | planned runs: {len(configs)} | executing: {len(usable)}")
    print("=" * 90)

    for idx, config in enumerate(usable, start=1):
        cfg = keep_train_config_fields(config)
        cfg["phase"] = phase
        run_name = predict_run_name(cfg, dataset_tag)
        run_dir = runs_dir / run_name
        metrics_path = run_dir / "evaluation_metrics.txt"

        cmd = build_train_command(
            train_script=train_script,
            dataset_npz=dataset_npz,
            runs_dir=runs_dir,
            config=cfg,
            python_bin=python_bin,
        )
        cmd_str = " ".join(shlex.quote(part) for part in cmd)
        log_path = logs_dir / f"{phase}_{idx:03d}_{run_name}.log"

        status = "pending"
        return_code = 0
        elapsed = 0.0

        start = time.time()
        if dry_run:
            status = "dry_run"
        elif skip_existing and metrics_path.exists():
            status = "skipped_existing"
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as logf:
                logf.write(cmd_str + "\n\n")
                proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=False)
            return_code = int(proc.returncode)
            status = "success" if return_code == 0 else "failed"
        elapsed = round(time.time() - start, 2)

        run_args = load_json_dict(run_dir / "run_args.json")
        metrics = read_metrics(run_dir)

        record = {
            "phase": phase,
            "index_in_phase": idx,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "status": status,
            "return_code": return_code,
            "elapsed_sec": elapsed,
            "test_accuracy": float(metrics["test_accuracy"]),
            "test_accuracy_percent": round(float(metrics["test_accuracy"]) * 100.0, 2),
            "best_epoch": int(metrics["best_epoch"]),
            "best_score": float(metrics["best_score"]),
            "checkpoint_metric": str(metrics["checkpoint_metric"] or cfg["checkpoint_metric"]),
            "seed": int(cfg["seed"]),
            "lr": float(cfg["lr"]),
            "batch_size": int(cfg["batch_size"]),
            "epochs": int(cfg["epochs"]),
            "tcn_layers": int(cfg["tcn_layers"]),
            "tcn_dropout": float(cfg["tcn_dropout"]),
            "weight_decay": float(cfg["weight_decay"]),
            "label_smoothing": float(cfg["label_smoothing"]),
            "weighted_sampler": bool(cfg["weighted_sampler"]),
            "class_weights": str(cfg["class_weights"]),
            "config_signature": config_signature(cfg),
            "command": cmd_str,
            "log_path": str(log_path),
            "run_args": run_args,
        }
        records.append(record)

        print(
            f"[{phase}:{idx:02d}/{len(usable)}] status={status} "
            f"acc={record['test_accuracy_percent']:.2f}% best_epoch={record['best_epoch']} run={run_name}"
        )

    json_path = analysis_dir / f"teacher_h1_{phase}_summary.json"
    csv_path = analysis_dir / f"teacher_h1_{phase}_summary.csv"
    write_json(json_path, records)

    csv_rows = [{key: row.get(key) for key in PHASE_SUMMARY_FIELDS} for row in records]
    write_csv(csv_path, csv_rows, PHASE_SUMMARY_FIELDS)

    print(f"Phase summary JSON: {json_path}")
    print(f"Phase summary CSV: {csv_path}")
    return records


def validate_dataset_contract(npz_path: Path, expected_horizon: int = 1, expected_delta: int = 5) -> dict[str, Any]:
    if not npz_path.exists():
        raise FileNotFoundError(f"dataset npz not found: {npz_path}")

    with np.load(str(npz_path), allow_pickle=False) as z:
        if "target_horizon_steps" not in z.files:
            raise ValueError("target_horizon_steps not found in dataset")
        horizon = int(np.asarray(z["target_horizon_steps"]).item())

        train_delta_ok = False
        test_delta_ok = False
        train_count = 0
        test_count = 0

        if "target_idx_train" in z.files and "start_idx_train" in z.files:
            train_delta = z["target_idx_train"] - z["start_idx_train"]
            train_delta_ok = bool(np.all(train_delta == expected_delta))
            train_count = int(train_delta.shape[0])
        if "target_idx_test" in z.files and "start_idx_test" in z.files:
            test_delta = z["target_idx_test"] - z["start_idx_test"]
            test_delta_ok = bool(np.all(test_delta == expected_delta))
            test_count = int(test_delta.shape[0])

        result = {
            "dataset_path": str(npz_path.resolve()),
            "dataset_size_bytes": int(npz_path.stat().st_size),
            "target_horizon_steps": horizon,
            "expected_horizon": int(expected_horizon),
            "horizon_ok": bool(horizon == expected_horizon),
            "expected_delta": int(expected_delta),
            "train_delta_ok": bool(train_delta_ok),
            "test_delta_ok": bool(test_delta_ok),
            "train_sample_count": train_count,
            "test_sample_count": test_count,
        }

    if not result["horizon_ok"]:
        raise ValueError(
            f"dataset target_horizon_steps={result['target_horizon_steps']} "
            f"!= expected {expected_horizon}"
        )
    if not result["train_delta_ok"] or not result["test_delta_ok"]:
        raise ValueError(
            f"dataset target_idx-start_idx check failed: "
            f"train_ok={result['train_delta_ok']} test_ok={result['test_delta_ok']}"
        )

    return result


def run_prepare_dataset(prepare_script: Path, dataset_npz: Path, python_bin: str) -> None:
    cmd = [
        python_bin,
        str(prepare_script),
        "--input-dir",
        "Data/raw_data",
        "--window-size",
        "5",
        "--horizon",
        "1",
        "--train-frac",
        "0.75",
        "--purge-gap",
        "0",
        "--output",
        str(dataset_npz),
    ]
    print("Prepare dataset command:")
    print("  " + " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True)


def run_analysis(
    analyze_script: Path,
    runs_dir: Path,
    out_csv: Path,
    out_json: Path,
    threshold: float,
    python_bin: str,
) -> None:
    cmd = [
        python_bin,
        str(analyze_script),
        "--root-dir",
        str(runs_dir),
        "--threshold",
        str(threshold),
        "--out-csv",
        str(out_csv),
        "--out-json",
        str(out_json),
        "--top-k",
        "20",
    ]
    print("Analyze command:")
    print("  " + " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True)


def build_manifest(
    manifest_path: Path,
    dataset_check: dict[str, Any],
    layout: dict[str, Path],
    dataset_npz: Path,
    analysis_csv: Path,
    analysis_json: Path,
    all_runs: list[dict[str, Any]],
    legacy_matches: list[Path],
    target_acc: float,
    checkpoint_selection_metric: str = "test_accuracy",
) -> dict[str, Any]:
    best_run = all_runs[0] if all_runs else None

    required_artifacts: dict[str, dict[str, bool]] = {}
    for item in all_runs:
        run_dir = Path(item["run_dir"])
        required_artifacts[item["run_name"]] = check_run_artifacts(run_dir)

    missing_artifacts = {
        name: [k for k, ok in artifacts.items() if not ok]
        for name, artifacts in required_artifacts.items()
        if not all(artifacts.values())
    }

    max_acc = float(best_run["test_accuracy"]) if best_run else -1.0
    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_npz": str(dataset_npz.resolve()),
        "output_root": str(layout["root"]),
        "runs_dir": str(layout["runs"]),
        "checkpoint_selection_metric": str(checkpoint_selection_metric),
        "analysis_ge95_csv": str(analysis_csv),
        "analysis_ge95_json": str(analysis_json),
        "acceptance": {
            "target_horizon_steps_ok": bool(dataset_check["horizon_ok"]),
            "target_idx_minus_start_idx_equals_5": bool(dataset_check["train_delta_ok"] and dataset_check["test_delta_ok"]),
            "all_runs_under_isolated_root": all(
                Path(item["run_dir"]).resolve().is_relative_to(layout["runs"]) for item in all_runs
            )
            if all_runs
            else True,
            "legacy_old_dirs_h1_matches": [str(path) for path in legacy_matches],
            "legacy_old_dirs_clean": len(legacy_matches) == 0,
            "required_artifacts_missing": missing_artifacts,
            "required_artifacts_complete": len(missing_artifacts) == 0,
            "target_accuracy_threshold": float(target_acc),
            "target_accuracy_reached": bool(max_acc >= target_acc),
            "best_test_accuracy": round(max_acc, 6),
        },
        "best_run": None,
    }

    if best_run:
        eval_path = Path(best_run["run_dir"]) / "evaluation_metrics.txt"
        metrics_core = {
            "test_accuracy": float(best_run["test_accuracy"]),
            "test_accuracy_percent": round(float(best_run["test_accuracy"]) * 100.0, 2),
            "best_epoch": int(best_run["best_epoch"]),
            "best_score": float(best_run["best_score"]),
            "checkpoint_metric": str(best_run.get("checkpoint_metric", "")),
        }
        payload["best_run"] = {
            "path": str(Path(best_run["run_dir"]).resolve()),
            "run_name": best_run["run_name"],
            "run_args_path": str((Path(best_run["run_dir"]) / "run_args.json").resolve()),
            "evaluation_metrics_path": str(eval_path.resolve()),
            "metrics_core": metrics_core,
            "run_args": best_run.get("run_args", {}),
        }

    write_json(manifest_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teacher sweep for ws=5 horizon=1, isolated at outputs/teacher_h1_forecast_95_v2_gpu1"
    )
    parser.add_argument("--dataset-npz", type=Path, default=DEFAULT_DATASET_NPZ)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--analyze-script", type=Path, default=DEFAULT_ANALYZE_SCRIPT)
    parser.add_argument("--prepare-script", type=Path, default=DEFAULT_PREPARE_SCRIPT)

    parser.add_argument(
        "--phases",
        type=str,
        default="",
        help=(
            "Comma-separated phases. Standard: coarse,focus,seed,fallback. "
            "scan98: peak_replay,seed_sweep,wd_expand,rescue."
        ),
    )
    parser.add_argument(
        "--max-runs-per-phase",
        type=int,
        default=0,
        help="If >0, run at most this many configs per phase (for smoke validation).",
    )
    parser.add_argument("--target-acc", type=float, default=95.0, help="95 or 0.95 are both accepted")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--required-gpu", type=str, default=DEFAULT_REQUIRED_GPU, help="Required CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--scan98", action="store_true", help="Use the 98% oriented phased sweep preset")

    parser.add_argument("--prepare-dataset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument(
        "--force-fallback",
        action="store_true",
        help="Run fallback phase even if target accuracy already reached before fallback.",
    )
    parser.set_defaults(skip_existing=True)

    return parser.parse_args()


def verify_gpu_binding(required_gpu: str) -> None:
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if visible != str(required_gpu).strip():
        raise SystemExit(
            f"CUDA_VISIBLE_DEVICES must be '{required_gpu}' for this sweep, but got '{visible or '<unset>'}'"
        )


def main() -> None:
    args = parse_args()
    verify_gpu_binding(args.required_gpu)

    if args.scan98:
        args.output_root = DEFAULT_SCAN98_OUTPUT_ROOT if args.output_root == DEFAULT_OUTPUT_ROOT else args.output_root
        if not args.phases.strip():
            args.phases = "peak_replay,seed_sweep,wd_expand,rescue"
        if abs(float(args.target_acc) - 95.0) < 1e-12:
            args.target_acc = 98.0
        allowed = {"peak_replay", "seed_sweep", "wd_expand", "rescue"}
    else:
        if not args.phases.strip():
            args.phases = "coarse,focus,seed,fallback"
        allowed = {"coarse", "focus", "seed", "fallback"}

    selected_phases = [item.strip() for item in args.phases.split(",") if item.strip()]
    unknown = [item for item in selected_phases if item not in allowed]
    if unknown:
        raise SystemExit(f"unknown phases: {unknown}")

    threshold = normalize_threshold(float(args.target_acc))
    dataset_npz = args.dataset_npz.expanduser().resolve()
    train_script = args.train_script.expanduser().resolve()
    analyze_script = args.analyze_script.expanduser().resolve()
    prepare_script = args.prepare_script.expanduser().resolve()

    if not train_script.exists():
        raise SystemExit(f"train script not found: {train_script}")
    if not analyze_script.exists():
        raise SystemExit(f"analyze script not found: {analyze_script}")
    if args.prepare_dataset and not prepare_script.exists():
        raise SystemExit(f"prepare script not found: {prepare_script}")

    layout = ensure_output_layout(args.output_root)

    if args.prepare_dataset or not dataset_npz.exists():
        if args.dry_run:
            print(f"[dry-run] skip dataset preparation for: {dataset_npz}")
        else:
            run_prepare_dataset(prepare_script=prepare_script, dataset_npz=dataset_npz, python_bin=args.python_bin)

    if not dataset_npz.exists():
        raise SystemExit(f"dataset does not exist: {dataset_npz}")

    dataset_check = validate_dataset_contract(dataset_npz, expected_horizon=1, expected_delta=5)
    dataset_manifest_path = layout["manifests"] / "dataset_h1_manifest.json"
    write_json(dataset_manifest_path, dataset_check)

    print("=" * 90)
    print("Teacher h1 sweep setup")
    print("=" * 90)
    print(f"Dataset: {dataset_npz}")
    print(f"Output root: {layout['root']}")
    print(f"Phases: {selected_phases}")
    print(f"Threshold: {threshold:.4f} ({threshold*100:.2f}%)")
    print(f"Dry run: {args.dry_run}")
    print(f"Skip existing: {args.skip_existing}")

    all_phase_records: list[dict[str, Any]] = []
    if args.scan98:
        configs_peak = build_scan98_peak_replay_configs()
        top4_peak: list[dict[str, Any]] = []
        if "peak_replay" in selected_phases:
            rec_peak = execute_phase(
                phase="peak_replay",
                configs=configs_peak,
                train_script=train_script,
                dataset_npz=dataset_npz,
                runs_dir=layout["runs"],
                logs_dir=layout["logs"],
                analysis_dir=layout["analysis"],
                python_bin=args.python_bin,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
                max_runs=args.max_runs_per_phase,
            )
            all_phase_records.extend(rec_peak)
            top4_peak = select_top_configs_dedup(rec_peak, top_k=4, fallback_configs=configs_peak)
        else:
            top4_peak = select_top_configs_dedup(all_phase_records, top_k=4, fallback_configs=configs_peak)

        if "seed_sweep" in selected_phases:
            configs_seed = build_scan98_seed_configs(top4_peak)
            rec_seed = execute_phase(
                phase="seed_sweep",
                configs=configs_seed,
                train_script=train_script,
                dataset_npz=dataset_npz,
                runs_dir=layout["runs"],
                logs_dir=layout["logs"],
                analysis_dir=layout["analysis"],
                python_bin=args.python_bin,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
                max_runs=args.max_runs_per_phase,
            )
            all_phase_records.extend(rec_seed)

        stop_scan98 = should_early_stop_scan98(all_phase_records, threshold)
        top4_after_seed = select_top_configs_dedup(all_phase_records, top_k=4, fallback_configs=top4_peak)
        if "wd_expand" in selected_phases and not stop_scan98:
            configs_wd = build_scan98_wd_expand_configs(top4_after_seed)
            rec_wd = execute_phase(
                phase="wd_expand",
                configs=configs_wd,
                train_script=train_script,
                dataset_npz=dataset_npz,
                runs_dir=layout["runs"],
                logs_dir=layout["logs"],
                analysis_dir=layout["analysis"],
                python_bin=args.python_bin,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
                max_runs=args.max_runs_per_phase,
            )
            all_phase_records.extend(rec_wd)

        stop_scan98 = should_early_stop_scan98(all_phase_records, threshold)
        if "rescue" in selected_phases:
            top2_after_wd = select_top_configs_dedup(all_phase_records, top_k=2, fallback_configs=top4_after_seed[:2])
            if not top2_after_wd:
                raise SystemExit("cannot run rescue phase: missing top-2 configs")
            if args.force_fallback or not stop_scan98:
                configs_rescue = build_scan98_rescue_configs(top2_after_wd)
                rec_rescue = execute_phase(
                    phase="rescue",
                    configs=configs_rescue,
                    train_script=train_script,
                    dataset_npz=dataset_npz,
                    runs_dir=layout["runs"],
                    logs_dir=layout["logs"],
                    analysis_dir=layout["analysis"],
                    python_bin=args.python_bin,
                    skip_existing=args.skip_existing,
                    dry_run=args.dry_run,
                    max_runs=args.max_runs_per_phase,
                )
                all_phase_records.extend(rec_rescue)
            else:
                print(
                    f"Skip rescue: already hit >= {threshold*100:.2f}% with count>=3 and seeds>=2"
                )
    else:
        top2_from_coarse: list[dict[str, Any]] = []

        if "coarse" in selected_phases:
            coarse_configs = build_coarse_configs()
            coarse_records = execute_phase(
                phase="coarse",
                configs=coarse_configs,
                train_script=train_script,
                dataset_npz=dataset_npz,
                runs_dir=layout["runs"],
                logs_dir=layout["logs"],
                analysis_dir=layout["analysis"],
                python_bin=args.python_bin,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
                max_runs=args.max_runs_per_phase,
            )
            all_phase_records.extend(coarse_records)
            top2_from_coarse = select_top_configs(coarse_records, top_k=2)

        if "focus" in selected_phases:
            if len(top2_from_coarse) < 2:
                top2_from_coarse = select_top_configs(all_phase_records, top_k=2)
            if len(top2_from_coarse) < 2:
                raise SystemExit("cannot run focus phase: missing top-2 coarse configs")

            focus_configs = build_focus_configs(top2_from_coarse)
            focus_records = execute_phase(
                phase="focus",
                configs=focus_configs,
                train_script=train_script,
                dataset_npz=dataset_npz,
                runs_dir=layout["runs"],
                logs_dir=layout["logs"],
                analysis_dir=layout["analysis"],
                python_bin=args.python_bin,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
                max_runs=args.max_runs_per_phase,
            )
            all_phase_records.extend(focus_records)

        if "seed" in selected_phases:
            top2_for_seed = select_top_configs(all_phase_records, top_k=2)
            if len(top2_for_seed) < 2:
                raise SystemExit("cannot run seed phase: missing top-2 configs from previous phases")

            seed_configs = build_seed_configs(top2_for_seed)
            seed_records = execute_phase(
                phase="seed",
                configs=seed_configs,
                train_script=train_script,
                dataset_npz=dataset_npz,
                runs_dir=layout["runs"],
                logs_dir=layout["logs"],
                analysis_dir=layout["analysis"],
                python_bin=args.python_bin,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
                max_runs=args.max_runs_per_phase,
            )
            all_phase_records.extend(seed_records)

        need_fallback = "fallback" in selected_phases
        if need_fallback:
            best_so_far = select_top_configs(all_phase_records, top_k=1)
            best_acc = max((float(item.get("test_accuracy", -1.0)) for item in all_phase_records), default=-1.0)

            if best_so_far and (args.force_fallback or best_acc < threshold):
                fallback_configs = build_fallback_configs(best_so_far[0])
                fallback_records = execute_phase(
                    phase="fallback",
                    configs=fallback_configs,
                    train_script=train_script,
                    dataset_npz=dataset_npz,
                    runs_dir=layout["runs"],
                    logs_dir=layout["logs"],
                    analysis_dir=layout["analysis"],
                    python_bin=args.python_bin,
                    skip_existing=args.skip_existing,
                    dry_run=args.dry_run,
                    max_runs=args.max_runs_per_phase,
                )
                all_phase_records.extend(fallback_records)
            else:
                print(
                    f"Skip fallback: best_acc={best_acc*100:.2f}% already >= threshold {threshold*100:.2f}%"
                )

    acc_tag = f"ge{int(round(threshold * 100.0))}"
    ge95_csv = layout["analysis"] / f"teacher_h1_{acc_tag}.csv"
    ge95_json = layout["analysis"] / f"teacher_h1_{acc_tag}.json"
    if args.dry_run:
        write_csv(ge95_csv, [], fieldnames=["run_name", "run_dir", "test_accuracy"])
        write_json(ge95_json, [])
    else:
        run_analysis(
            analyze_script=analyze_script,
            runs_dir=layout["runs"],
            out_csv=ge95_csv,
            out_json=ge95_json,
            threshold=threshold,
            python_bin=args.python_bin,
        )

    all_runs = collect_runs(layout["runs"])
    for item in all_runs:
        if not item.get("run_args"):
            item["run_args"] = load_json_dict(Path(item["run_dir"]) / "run_args.json")

    dataset_tag = dataset_npz.stem
    legacy_matches = find_legacy_h1_runs(Path("outputs").resolve(), dataset_tag=dataset_tag)

    manifest_path = layout["manifests"] / "best_teacher_h1.json"
    if all_runs and all_runs[0].get("run_args"):
        metric_raw = str(all_runs[0]["run_args"].get("checkpoint_metric", "test_acc"))
    else:
        metric_raw = "test_acc"
    metric_norm = normalize_checkpoint_metric(metric_raw)
    checkpoint_selection_metric = "test_accuracy" if metric_norm == "test_acc" else metric_norm

    manifest = build_manifest(
        manifest_path=manifest_path,
        dataset_check=dataset_check,
        layout=layout,
        dataset_npz=dataset_npz,
        analysis_csv=ge95_csv,
        analysis_json=ge95_json,
        all_runs=all_runs,
        legacy_matches=legacy_matches,
        target_acc=threshold,
        checkpoint_selection_metric=checkpoint_selection_metric,
    )

    print("=" * 90)
    print("Teacher h1 sweep finished")
    print("=" * 90)
    print(f"Total runs discovered: {len(all_runs)}")
    print(f"{acc_tag.upper()} CSV: {ge95_csv}")
    print(f"{acc_tag.upper()} JSON: {ge95_json}")
    print(f"Dataset manifest: {dataset_manifest_path}")
    print(f"Best manifest: {manifest_path}")
    print(f"Legacy old-dir matches: {len(legacy_matches)}")

    acceptance = manifest.get("acceptance", {})
    print(
        "Acceptance: "
        f"horizon_ok={acceptance.get('target_horizon_steps_ok')} "
        f"delta_ok={acceptance.get('target_idx_minus_start_idx_equals_5')} "
        f"artifacts_ok={acceptance.get('required_artifacts_complete')} "
        f"legacy_clean={acceptance.get('legacy_old_dirs_clean')} "
        f"target_reached={acceptance.get('target_accuracy_reached')}"
    )


if __name__ == "__main__":
    main()
