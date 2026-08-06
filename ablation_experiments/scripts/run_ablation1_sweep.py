#!/usr/bin/env python3
"""Run hyperparameter sweep for Ablation-1 future-step prediction (13D student-only)."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_ablation1_student_only.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training_utils import validate_forecast_dataset_contract  # noqa: E402

BEST_EPOCH_PATTERN = re.compile(r"Best epoch:\s*(\d+)")
BEST_SCORE_PATTERN = re.compile(r"Best score:\s*([0-9]+(?:\.[0-9]+)?)")
VAL_F1_PATTERN = re.compile(
    r"---\s*Val Metrics \(Student-only\)\s*---.*?Macro-F1:\s*([0-9]+(?:\.[0-9]+)?)",
    flags=re.S,
)
TEST_ACC_PATTERN = re.compile(
    r"---\s*Test Metrics \(Student-only\)\s*---.*?Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%",
    flags=re.S,
)
TEST_F1_PATTERN = re.compile(
    r"---\s*Test Metrics \(Student-only\)\s*---.*?Macro-F1:\s*([0-9]+(?:\.[0-9]+)?)",
    flags=re.S,
)


@dataclass
class TrialParams:
    seed: int
    lr: float
    weight_decay: float
    tcn_dropout: float
    classifier_dropout: float
    batch_size: int
    tcn_layers: int
    tcn_channels: str
    attn_dropout: float
    label_smoothing: float
    weighted_sampler: bool
    epochs: int
    tcn_kernel: int
    tcn_dilation_base: int
    classifier_hidden: int
    attn_heads: int
    attn_ff_dim: int


@dataclass
class TrialRecord:
    trial_id: int
    stage: str
    status: str
    run_dir: str
    elapsed_sec: float
    reached_target: bool
    best_epoch: int
    best_score: float
    val_macro_f1: float
    test_acc: float
    test_macro_f1: float
    seed: int
    lr: float
    weight_decay: float
    tcn_dropout: float
    classifier_dropout: float
    batch_size: int
    tcn_layers: int
    tcn_channels: str
    attn_dropout: float
    label_smoothing: float
    weighted_sampler: bool
    epochs: int
    checkpoint_metric: str
    command: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hyperparameter sweep for Ablation-1 future-step prediction"
    )
    parser.add_argument(
        "--dataset-npz",
        type=Path,
        default=PROJECT_ROOT / "Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz",
        help="Path to the prepared horizon=1 dataset (.npz)",
    )
    parser.add_argument(
        "--space-json",
        type=Path,
        default=SCRIPT_DIR / "space_ablation1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/h1/results/ablation1_sweep_strict_valf1_20260522",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/h1/reports/ablation1_sweep_strict_valf1_20260522",
    )

    parser.add_argument("--stage-a-runs", type=int, default=40)
    parser.add_argument("--stage-b-runs", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--target-acc", type=float, default=90.0, help="Use percent if >1.0 else ratio")
    parser.add_argument("--stop-on-target", dest="stop_on_target", action="store_true")
    parser.add_argument("--no-stop-on-target", dest="stop_on_target", action="store_false")
    parser.add_argument("--sweep-seed", type=int, default=20260306)

    parser.add_argument("--drop-feature-indices", type=str, default="3,4,5,6,7")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--retry-on-fail", type=int, default=0)
    parser.add_argument(
        "--checkpoint-metric",
        type=str,
        default="val_macro_f1",
        choices=("val_macro_f1", "test_acc"),
    )
    parser.add_argument("--print-command", action="store_true")
    parser.set_defaults(stop_on_target=False)
    return parser.parse_args()


def load_space(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("space json must be a JSON object")
    return payload


def parse_threshold(value: float) -> float:
    return value / 100.0 if value > 1.0 else value


def choose_one(rng: random.Random, values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError("empty candidate list in search space")
    return values[rng.randrange(len(values))]


def make_signature(params: TrialParams) -> str:
    payload = asdict(params)
    return json.dumps(payload, sort_keys=True)


def sample_stage_a_trials(space: Dict[str, Any], count: int, rng: random.Random, epochs_override: int | None) -> List[TrialParams]:
    trials: List[TrialParams] = []
    seen: set[str] = set()
    max_attempts = max(count * 50, 200)
    attempts = 0

    while len(trials) < count and attempts < max_attempts:
        attempts += 1
        params = TrialParams(
            seed=int(choose_one(rng, space["seed_candidates"])),
            lr=float(choose_one(rng, space["lr"])),
            weight_decay=float(choose_one(rng, space["weight_decay"])),
            tcn_dropout=float(choose_one(rng, space["tcn_dropout"])),
            classifier_dropout=float(choose_one(rng, space["classifier_dropout"])),
            batch_size=int(choose_one(rng, space["batch_size"])),
            tcn_layers=int(choose_one(rng, space["tcn_layers"])),
            tcn_channels=str(choose_one(rng, space["tcn_channels"])),
            attn_dropout=float(choose_one(rng, space["attn_dropout"])),
            label_smoothing=float(choose_one(rng, space["label_smoothing"])),
            weighted_sampler=bool(choose_one(rng, space["weighted_sampler"])),
            epochs=int(choose_one(rng, space["epochs"])),
            tcn_kernel=int(choose_one(rng, space["tcn_kernel"])),
            tcn_dilation_base=int(choose_one(rng, space["tcn_dilation_base"])),
            classifier_hidden=int(choose_one(rng, space["classifier_hidden"])),
            attn_heads=int(choose_one(rng, space["attn_heads"])),
            attn_ff_dim=int(choose_one(rng, space["attn_ff_dim"])),
        )
        if epochs_override is not None:
            params.epochs = int(epochs_override)
        sig = make_signature(params)
        if sig in seen:
            continue
        seen.add(sig)
        trials.append(params)

    if len(trials) < count:
        print(
            f"Warning: only sampled {len(trials)} unique stage-A trials (requested {count})",
            file=sys.stderr,
        )
    return trials


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def sample_stage_b_trials(
    top_params: List[TrialParams],
    space: Dict[str, Any],
    count: int,
    rng: random.Random,
    epochs_override: int | None,
) -> List[TrialParams]:
    if count <= 0 or not top_params:
        return []

    cfg = dict(space.get("stage_b", {}))
    lr_factors = cfg.get("lr_factors", [0.8, 1.0, 1.2])
    wd_factors = cfg.get("weight_decay_factors", [0.7, 1.0, 1.4])
    tcn_delta = cfg.get("tcn_dropout_delta", [-0.03, 0.0, 0.03])
    clf_delta = cfg.get("classifier_dropout_delta", [-0.05, 0.0, 0.05])
    attn_delta = cfg.get("attn_dropout_delta", [-0.03, 0.0, 0.03])
    ls_choices = cfg.get("label_smoothing_choices", [0.0, 0.03, 0.05, 0.08])

    seed_pool = [int(x) for x in space["seed_candidates"]]
    tcn_layer_pool = [int(x) for x in space["tcn_layers"]]
    tcn_channels_pool = [str(x) for x in space["tcn_channels"]]
    batch_pool = [int(x) for x in space["batch_size"]]

    trials: List[TrialParams] = []
    seen: set[str] = set()
    max_attempts = max(count * 60, 300)
    attempts = 0

    min_lr = float(min(space["lr"]))
    max_lr = float(max(space["lr"]))
    min_wd = float(min(space["weight_decay"]))
    max_wd = float(max(space["weight_decay"]))

    while len(trials) < count and attempts < max_attempts:
        attempts += 1
        base = top_params[rng.randrange(len(top_params))]

        params = TrialParams(
            seed=int(choose_one(rng, seed_pool)),
            lr=round(clamp(base.lr * float(choose_one(rng, lr_factors)), min_lr * 0.6, max_lr * 1.4), 8),
            weight_decay=round(
                clamp(base.weight_decay * float(choose_one(rng, wd_factors)), min_wd * 0.5, max_wd * 1.6),
                8,
            ),
            tcn_dropout=round(clamp(base.tcn_dropout + float(choose_one(rng, tcn_delta)), 0.05, 0.35), 4),
            classifier_dropout=round(
                clamp(base.classifier_dropout + float(choose_one(rng, clf_delta)), 0.15, 0.65),
                4,
            ),
            batch_size=int(choose_one(rng, [base.batch_size, *batch_pool])),
            tcn_layers=int(choose_one(rng, [base.tcn_layers, *tcn_layer_pool])),
            tcn_channels=str(choose_one(rng, [base.tcn_channels, *tcn_channels_pool])),
            attn_dropout=round(clamp(base.attn_dropout + float(choose_one(rng, attn_delta)), 0.03, 0.25), 4),
            label_smoothing=float(choose_one(rng, [base.label_smoothing, *ls_choices])),
            weighted_sampler=bool(choose_one(rng, [base.weighted_sampler, True, False])),
            epochs=base.epochs,
            tcn_kernel=base.tcn_kernel,
            tcn_dilation_base=base.tcn_dilation_base,
            classifier_hidden=base.classifier_hidden,
            attn_heads=base.attn_heads,
            attn_ff_dim=base.attn_ff_dim,
        )

        if epochs_override is not None:
            params.epochs = int(epochs_override)

        sig = make_signature(params)
        if sig in seen:
            continue
        seen.add(sig)
        trials.append(params)

    if len(trials) < count:
        print(
            f"Warning: only sampled {len(trials)} unique stage-B trials (requested {count})",
            file=sys.stderr,
        )
    return trials


def parse_metrics(metrics_path: Path) -> Dict[str, float | int]:
    if not metrics_path.exists():
        return {
            "best_epoch": -1,
            "best_score": -1.0,
            "val_macro_f1": -1.0,
            "test_acc": -1.0,
            "test_macro_f1": -1.0,
        }

    text = metrics_path.read_text(encoding="utf-8", errors="ignore")

    m_best_epoch = BEST_EPOCH_PATTERN.search(text)
    m_best_score = BEST_SCORE_PATTERN.search(text)
    m_val_f1 = VAL_F1_PATTERN.search(text)
    m_test_acc = TEST_ACC_PATTERN.search(text)
    m_test_f1 = TEST_F1_PATTERN.search(text)

    return {
        "best_epoch": int(m_best_epoch.group(1)) if m_best_epoch else -1,
        "best_score": float(m_best_score.group(1)) if m_best_score else -1.0,
        "val_macro_f1": float(m_val_f1.group(1)) if m_val_f1 else -1.0,
        "test_acc": (float(m_test_acc.group(1)) / 100.0) if m_test_acc else -1.0,
        "test_macro_f1": float(m_test_f1.group(1)) if m_test_f1 else -1.0,
    }


def write_progress(report_dir: Path, records: List[TrialRecord]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "ablation1_sweep_progress.json"
    csv_path = report_dir / "ablation1_sweep_progress.csv"

    payload = [asdict(r) for r in records]
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrialRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(payload)


def run_trial(
    trial_id: int,
    stage: str,
    params: TrialParams,
    args: argparse.Namespace,
    target: float,
) -> TrialRecord:
    trial_dir = args.output_root / "trials" / f"trial_{trial_id:04d}_{stage}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset-npz",
        str(args.dataset_npz),
        "--output-dir",
        str(trial_dir),
        "--drop-feature-indices",
        str(args.drop_feature_indices),
        "--epochs",
        str(params.epochs),
        "--batch-size",
        str(params.batch_size),
        "--lr",
        str(params.lr),
        "--weight-decay",
        str(params.weight_decay),
        "--label-smoothing",
        str(params.label_smoothing),
        "--seed",
        str(params.seed),
        "--num-workers",
        str(args.num_workers),
        "--tcn-kernel",
        str(params.tcn_kernel),
        "--tcn-layers",
        str(params.tcn_layers),
        "--tcn-channels",
        str(params.tcn_channels),
        "--tcn-dropout",
        str(params.tcn_dropout),
        "--tcn-dilation-base",
        str(params.tcn_dilation_base),
        "--classifier-hidden",
        str(params.classifier_hidden),
        "--classifier-dropout",
        str(params.classifier_dropout),
        "--attn-heads",
        str(params.attn_heads),
        "--attn-dropout",
        str(params.attn_dropout),
        "--attn-ff-dim",
        str(params.attn_ff_dim),
        "--checkpoint-metric",
        str(args.checkpoint_metric),
    ]
    if params.weighted_sampler:
        cmd.append("--weighted-sampler")

    if args.print_command:
        print("Command:", " ".join(cmd))

    status = "failed"
    proc_rc = -1
    elapsed = 0.0
    for attempt in range(args.retry_on_fail + 1):
        t0 = time.time()
        proc = subprocess.run(cmd, check=False)
        elapsed += time.time() - t0
        proc_rc = proc.returncode
        if proc.returncode == 0:
            status = "success"
            break

    run_dirs = [p for p in trial_dir.iterdir() if p.is_dir()]
    run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime) if run_dirs else trial_dir
    metrics_path = run_dir / "evaluation_metrics.txt"
    metrics = parse_metrics(metrics_path)

    if status != "success":
        status = f"failed_rc_{proc_rc}"

    reached = float(metrics["test_acc"]) >= target
    return TrialRecord(
        trial_id=trial_id,
        stage=stage,
        status=status,
        run_dir=str(run_dir),
        elapsed_sec=round(elapsed, 2),
        reached_target=reached,
        best_epoch=int(metrics["best_epoch"]),
        best_score=float(metrics["best_score"]),
        val_macro_f1=float(metrics["val_macro_f1"]),
        test_acc=float(metrics["test_acc"]),
        test_macro_f1=float(metrics["test_macro_f1"]),
        seed=params.seed,
        lr=params.lr,
        weight_decay=params.weight_decay,
        tcn_dropout=params.tcn_dropout,
        classifier_dropout=params.classifier_dropout,
        batch_size=params.batch_size,
        tcn_layers=params.tcn_layers,
        tcn_channels=params.tcn_channels,
        attn_dropout=params.attn_dropout,
        label_smoothing=params.label_smoothing,
        weighted_sampler=params.weighted_sampler,
        epochs=params.epochs,
        checkpoint_metric=str(args.checkpoint_metric),
        command=" ".join(cmd),
    )


def top_params_from_records(records: List[TrialRecord], top_k: int) -> List[TrialParams]:
    eligible = [r for r in records if r.status == "success" and r.val_macro_f1 >= 0.0]
    eligible.sort(key=lambda r: (r.val_macro_f1, r.test_macro_f1, r.test_acc), reverse=True)

    top_trials = eligible[: max(0, top_k)]
    params: List[TrialParams] = []
    for r in top_trials:
        params.append(
            TrialParams(
                seed=r.seed,
                lr=r.lr,
                weight_decay=r.weight_decay,
                tcn_dropout=r.tcn_dropout,
                classifier_dropout=r.classifier_dropout,
                batch_size=r.batch_size,
                tcn_layers=r.tcn_layers,
                tcn_channels=r.tcn_channels,
                attn_dropout=r.attn_dropout,
                label_smoothing=r.label_smoothing,
                weighted_sampler=r.weighted_sampler,
                epochs=r.epochs,
                tcn_kernel=3,
                tcn_dilation_base=2,
                classifier_hidden=128,
                attn_heads=4,
                attn_ff_dim=128,
            )
        )
    return params


def write_final_summary(report_dir: Path, records: List[TrialRecord], target: float) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "ablation1_sweep_runtime_summary.json"

    success = [r for r in records if r.status == "success"]
    reached = [r for r in success if r.reached_target]
    best = max(success, key=lambda r: (r.val_macro_f1, r.test_macro_f1, r.test_acc)) if success else None

    payload = {
        "checkpoint_metric": records[0].checkpoint_metric if records else "val_macro_f1",
        "target_test_acc": target,
        "total_trials": len(records),
        "success_trials": len(success),
        "target_reached": bool(reached),
        "target_reached_count": len(reached),
        "best_trial": asdict(best) if best else None,
        "records": [asdict(r) for r in records],
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not TRAIN_SCRIPT.exists():
        raise SystemExit(f"training script not found: {TRAIN_SCRIPT}")
    if not args.dataset_npz.exists():
        raise SystemExit(f"dataset not found: {args.dataset_npz}")
    if not args.space_json.exists():
        raise SystemExit(f"space json not found: {args.space_json}")
    validate_forecast_dataset_contract(args.dataset_npz)

    space = load_space(args.space_json)
    target = parse_threshold(args.target_acc)
    rng = random.Random(args.sweep_seed)

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    stage_a_trials = sample_stage_a_trials(space, max(0, args.stage_a_runs), rng, args.epochs_override)

    records: List[TrialRecord] = []
    trial_id = 1
    stop = False

    print("=" * 88)
    print("Ablation-1 Hyperparameter Sweep")
    print("=" * 88)
    print(f"Dataset: {args.dataset_npz}")
    print(f"Checkpoint metric: {args.checkpoint_metric}")
    print(f"Target test_acc: {target*100:.2f}%")
    print(f"Stage-A runs: {len(stage_a_trials)}")

    for params in stage_a_trials:
        rec = run_trial(trial_id, "A", params, args, target)
        records.append(rec)
        write_progress(args.report_dir, records)
        write_final_summary(args.report_dir, records, target)
        print(
            f"[A {trial_id:03d}] status={rec.status} "
            f"val_f1={rec.val_macro_f1:.4f} test_acc={rec.test_acc*100:.2f}%"
        )

        if args.stop_on_target and rec.reached_target:
            stop = True
            print(f"Stop early: reached target at trial {trial_id}")
            break
        trial_id += 1

    if not stop and args.stage_b_runs > 0:
        top_params = top_params_from_records(records, args.top_k)
        stage_b_trials = sample_stage_b_trials(
            top_params=top_params,
            space=space,
            count=max(0, args.stage_b_runs),
            rng=rng,
            epochs_override=args.epochs_override,
        )
        print(f"Stage-B runs: {len(stage_b_trials)}")

        for params in stage_b_trials:
            rec = run_trial(trial_id, "B", params, args, target)
            records.append(rec)
            write_progress(args.report_dir, records)
            write_final_summary(args.report_dir, records, target)
            print(
                f"[B {trial_id:03d}] status={rec.status} "
                f"val_f1={rec.val_macro_f1:.4f} test_acc={rec.test_acc*100:.2f}%"
            )

            if args.stop_on_target and rec.reached_target:
                print(f"Stop early: reached target at trial {trial_id}")
                break
            trial_id += 1

    write_progress(args.report_dir, records)
    write_final_summary(args.report_dir, records, target)

    print("=" * 88)
    print(f"Finished trials: {len(records)}")
    print(f"Progress CSV: {args.report_dir / 'ablation1_sweep_progress.csv'}")
    print(f"Runtime summary JSON: {args.report_dir / 'ablation1_sweep_runtime_summary.json'}")


if __name__ == "__main__":
    main()
