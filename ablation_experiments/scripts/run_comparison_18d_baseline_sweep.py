#!/usr/bin/env python3
"""Run a medium-budget sweep for a unified 18D baseline family."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_comparison_18d_baseline.py"

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
BEST_EPOCH_PATTERN = re.compile(r"Best epoch:\s*(\d+)")
BEST_SCORE_PATTERN = re.compile(r"Best score:\s*([0-9]+(?:\.[0-9]+)?)")

COMMON_SPACE: Dict[str, List[Any]] = {
    "seed": [14, 42, 77, 100, 122, 183],
    "lr": [1e-4, 2e-4, 3e-4, 5e-4],
    "weight_decay": [1e-5, 1e-4, 2e-4, 5e-4],
    "batch_size": [64, 128],
    "classifier_hidden": [64, 128],
    "classifier_dropout": [0.2, 0.35, 0.5],
}

MODEL_SPACE: Dict[str, Dict[str, List[Any]]] = {
    "lstm": {
        "rnn_hidden_size": [64, 96, 128],
        "rnn_layers": [1, 2, 3],
        "rnn_dropout": [0.1, 0.2, 0.3],
        "rnn_bidirectional": [True, False],
    },
    "gru": {
        "rnn_hidden_size": [64, 96, 128],
        "rnn_layers": [1, 2, 3],
        "rnn_dropout": [0.1, 0.2, 0.3],
        "rnn_bidirectional": [True, False],
    },
    "transformer": {
        "transformer_d_model": [64, 96, 128],
        "transformer_nhead": [4, 8],
        "transformer_layers": [2, 3, 4],
        "transformer_ff_dim": [128, 256],
        "transformer_dropout": [0.1, 0.2, 0.3],
    },
    "inception": {
        "inception_out_ch": [16, 24, 32],
        "inception_blocks": [3, 6, 9],
        "inception_bottleneck": [16, 32],
        "inception_dropout": [0.1, 0.2, 0.3],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sweep for one 18D baseline family")
    parser.add_argument(
        "--dataset-npz",
        type=Path,
        default=PROJECT_ROOT / "Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/h1/results/comparison_18d_baselines",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/h1/reports/comparison_18d_baselines",
    )
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "gru", "transformer", "inception"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--stage-a-runs", type=int, default=12)
    parser.add_argument("--stage-b-runs", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--eval-seeds", type=str, default="14,42,183")
    parser.add_argument("--retry-on-fail", type=int, default=0)
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--sweep-seed", type=int, default=20260326)
    parser.set_defaults(skip_existing=True)
    return parser.parse_args()


def parse_metrics(metrics_path: Path) -> Dict[str, float | int]:
    if not metrics_path.exists():
        return {"test_acc": -1.0, "macro_f1": -1.0, "best_epoch": -1, "best_score": -1.0}
    text = metrics_path.read_text(encoding="utf-8", errors="ignore")
    m_acc = TEST_ACC_PATTERN.search(text)
    m_f1 = TEST_F1_PATTERN.search(text)
    m_epoch = BEST_EPOCH_PATTERN.search(text)
    m_score = BEST_SCORE_PATTERN.search(text)
    return {
        "test_acc": float(m_acc.group(1)) / 100.0 if m_acc else -1.0,
        "macro_f1": float(m_f1.group(1)) if m_f1 else -1.0,
        "best_epoch": int(m_epoch.group(1)) if m_epoch else -1,
        "best_score": float(m_score.group(1)) if m_score else -1.0,
    }


def build_run_name(model: str, dataset_tag: str, params: Dict[str, Any]) -> str:
    if model in {"lstm", "gru"}:
        direction = "bi" if params["rnn_bidirectional"] else "uni"
        backbone = (
            f"h{params['rnn_hidden_size']}_l{params['rnn_layers']}_{direction}_do{params['rnn_dropout']}"
        )
    elif model == "transformer":
        backbone = (
            f"dm{params['transformer_d_model']}_nh{params['transformer_nhead']}_"
            f"l{params['transformer_layers']}_ff{params['transformer_ff_dim']}_"
            f"do{params['transformer_dropout']}"
        )
    else:
        backbone = (
            f"oc{params['inception_out_ch']}_blk{params['inception_blocks']}_"
            f"bn{params['inception_bottleneck']}_do{params['inception_dropout']}"
        )
    return (
        f"{model}_{dataset_tag}_{backbone}_ep{params['epochs']}_lr{params['lr']}_"
        f"wd{params['weight_decay']}_bs{params['batch_size']}_seed{params['seed']}"
    )


def build_command(args: argparse.Namespace, params: Dict[str, Any], model_output_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset-npz",
        str(args.dataset_npz),
        "--output-dir",
        str(model_output_dir),
        "--model",
        args.model,
        "--epochs",
        str(params["epochs"]),
        "--batch-size",
        str(params["batch_size"]),
        "--lr",
        str(params["lr"]),
        "--weight-decay",
        str(params["weight_decay"]),
        "--seed",
        str(params["seed"]),
        "--val-ratio",
        str(args.val_ratio),
        "--checkpoint-metric",
        "val_macro_f1",
        "--class-weights",
        "auto",
        "--weighted-sampler",
        "--min-epochs",
        "16",
        "--early-stop-patience",
        "16",
        "--classifier-hidden",
        str(params["classifier_hidden"]),
        "--classifier-dropout",
        str(params["classifier_dropout"]),
    ]
    if args.model in {"lstm", "gru"}:
        cmd.extend(
            [
                "--rnn-hidden-size",
                str(params["rnn_hidden_size"]),
                "--rnn-layers",
                str(params["rnn_layers"]),
                "--rnn-dropout",
                str(params["rnn_dropout"]),
            ]
        )
        cmd.append("--rnn-bidirectional" if params["rnn_bidirectional"] else "--no-rnn-bidirectional")
    elif args.model == "transformer":
        cmd.extend(
            [
                "--transformer-d-model",
                str(params["transformer_d_model"]),
                "--transformer-nhead",
                str(params["transformer_nhead"]),
                "--transformer-layers",
                str(params["transformer_layers"]),
                "--transformer-ff-dim",
                str(params["transformer_ff_dim"]),
                "--transformer-dropout",
                str(params["transformer_dropout"]),
            ]
        )
    else:
        cmd.extend(
            [
                "--inception-out-ch",
                str(params["inception_out_ch"]),
                "--inception-blocks",
                str(params["inception_blocks"]),
                "--inception-bottleneck",
                str(params["inception_bottleneck"]),
                "--inception-dropout",
                str(params["inception_dropout"]),
            ]
        )
    return cmd


def sample_stage_a(model: str, count: int, rng: random.Random, epochs: int) -> List[Dict[str, Any]]:
    params_list: List[Dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    while len(params_list) < count and attempts < count * 80:
        attempts += 1
        params: Dict[str, Any] = {k: rng.choice(v) for k, v in COMMON_SPACE.items()}
        params.update({k: rng.choice(v) for k, v in MODEL_SPACE[model].items()})
        params["epochs"] = epochs
        signature = json.dumps(params, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        params_list.append(params)
    return params_list


def mutate_param(model: str, base: Dict[str, Any], rng: random.Random, epochs: int) -> Dict[str, Any]:
    params = dict(base)
    params["epochs"] = epochs
    params["seed"] = rng.choice(COMMON_SPACE["seed"])
    params["lr"] = rng.choice(sorted({base["lr"], *COMMON_SPACE["lr"]}))
    params["weight_decay"] = rng.choice(sorted({base["weight_decay"], *COMMON_SPACE["weight_decay"]}))
    params["batch_size"] = rng.choice(sorted({base["batch_size"], *COMMON_SPACE["batch_size"]}))
    params["classifier_hidden"] = rng.choice(sorted({base["classifier_hidden"], *COMMON_SPACE["classifier_hidden"]}))
    params["classifier_dropout"] = rng.choice(
        sorted({base["classifier_dropout"], *COMMON_SPACE["classifier_dropout"]})
    )
    model_keys = MODEL_SPACE[model]
    varied_key = rng.choice(list(model_keys.keys()))
    params[varied_key] = rng.choice(model_keys[varied_key])
    return params


def write_records(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("[]\n" if path.suffix == ".json" else "", encoding="utf-8")
        return
    if path.suffix == ".json":
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not TRAIN_SCRIPT.exists():
        raise SystemExit(f"training script not found: {TRAIN_SCRIPT}")

    validate_forecast_dataset_contract(args.dataset_npz)
    rng = random.Random(args.sweep_seed)
    dataset_tag = args.dataset_npz.stem
    model_output_dir = args.output_root / args.model
    report_dir = args.report_root / args.model
    model_output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    stage_a = sample_stage_a(args.model, args.stage_a_runs, rng, args.epochs)
    stage_b: List[Dict[str, Any]] = []

    def execute(stage_name: str, params_list: List[Dict[str, Any]]) -> None:
        nonlocal records
        for index, params in enumerate(params_list, start=1):
            run_name = build_run_name(args.model, dataset_tag, params)
            run_dir = model_output_dir / run_name
            metrics_path = run_dir / "evaluation_metrics.txt"
            cmd = build_command(args, params, model_output_dir)
            if args.print_command:
                print("Command:", " ".join(cmd))

            status = "failed"
            elapsed = 0.0
            if args.skip_existing and metrics_path.exists():
                status = "skipped_existing"
            else:
                for attempt in range(args.retry_on_fail + 1):
                    start = time.time()
                    proc = subprocess.run(cmd, check=False)
                    elapsed = time.time() - start
                    status = "success" if proc.returncode == 0 else f"failed_rc_{proc.returncode}"
                    if proc.returncode == 0:
                        break
                    if attempt == args.retry_on_fail:
                        break

            metrics = parse_metrics(metrics_path)
            record = {
                "stage": stage_name,
                "stage_index": index,
                "status": status,
                "run_name": run_name,
                "run_dir": str(run_dir),
                "elapsed_sec": round(elapsed, 2),
                "test_acc": metrics["test_acc"],
                "test_macro_f1": metrics["macro_f1"],
                "best_epoch": metrics["best_epoch"],
                "best_score": metrics["best_score"],
                "epochs": params["epochs"],
                "seed": params["seed"],
                "lr": params["lr"],
                "weight_decay": params["weight_decay"],
                "batch_size": params["batch_size"],
                "classifier_hidden": params["classifier_hidden"],
                "classifier_dropout": params["classifier_dropout"],
                "command": " ".join(cmd),
            }
            for key in MODEL_SPACE[args.model]:
                record[key] = params[key]
            records.append(record)
            print(
                f"[{stage_name}] {run_name} | status={status} "
                f"test_acc={float(metrics['test_acc'])*100:.2f}% f1={float(metrics['macro_f1']):.4f}"
            )
            write_records(report_dir / "sweep_progress.json", records)
            write_records(report_dir / "sweep_progress.csv", records)

    execute("A", stage_a)
    ranked_stage_a = [row for row in records if row["stage"] == "A" and row["status"] in {"success", "skipped_existing"}]
    ranked_stage_a.sort(key=lambda row: (float(row["test_acc"]), float(row["test_macro_f1"])), reverse=True)
    top_records = ranked_stage_a[: args.top_k]

    if args.stage_b_runs > 0 and top_records:
        seen_signatures = {json.dumps({k: v for k, v in row.items() if k in COMMON_SPACE or k in MODEL_SPACE[args.model] or k == "epochs"}, sort_keys=True) for row in ranked_stage_a}
        while len(stage_b) < args.stage_b_runs and len(top_records) > 0:
            base = rng.choice(top_records)
            params = mutate_param(args.model, base, rng, args.epochs)
            signature = json.dumps(params, sort_keys=True)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            stage_b.append(params)
        execute("B", stage_b)

    eval_seeds = [int(item.strip()) for item in args.eval_seeds.split(",") if item.strip()]
    seed_eval_params: List[Dict[str, Any]] = []
    ranked_all = [row for row in records if row["status"] in {"success", "skipped_existing"}]
    ranked_all.sort(key=lambda row: (float(row["test_acc"]), float(row["test_macro_f1"])), reverse=True)
    for base in ranked_all[: args.top_k]:
        for seed in eval_seeds:
            params = {
                "epochs": args.epochs,
                "seed": seed,
                "lr": base["lr"],
                "weight_decay": base["weight_decay"],
                "batch_size": base["batch_size"],
                "classifier_hidden": base["classifier_hidden"],
                "classifier_dropout": base["classifier_dropout"],
            }
            for key in MODEL_SPACE[args.model]:
                params[key] = base[key]
            seed_eval_params.append(params)
    execute("seed_eval", seed_eval_params)

    ranked_final = [row for row in records if row["status"] in {"success", "skipped_existing"}]
    ranked_final.sort(key=lambda row: (float(row["test_acc"]), float(row["test_macro_f1"])), reverse=True)
    summary = {
        "model": args.model,
        "dataset_npz": str(args.dataset_npz),
        "num_runs": len(records),
        "best_run": ranked_final[0] if ranked_final else None,
        "top_runs": ranked_final[:5],
    }
    (report_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_records(report_dir / "sweep_summary.csv", ranked_final[:10] if ranked_final else [])
    print(f"Sweep complete for model={args.model}")
    print(f"Report dir: {report_dir}")


if __name__ == "__main__":
    main()
