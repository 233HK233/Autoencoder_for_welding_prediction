#!/usr/bin/env python3

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


TEST_ACCURACY_PATTERNS = [
    re.compile(
        r"---\s*(?:Student\s+)?Test Metrics(?:\s*\(Student\))?\s*---.*?Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%",
        flags=re.S,
    ),
]
BEST_EPOCH_PATTERN = re.compile(r"Best epoch:\s*(\d+)")
BEST_SCORE_PATTERN = re.compile(r"Best score:\s*([0-9]+(?:\.[0-9]+)?)")

CSV_FIELDS = [
    "source_root",
    "source_name",
    "run_name",
    "run_dir",
    "rel_run_dir",
    "test_accuracy",
    "test_accuracy_percent",
    "best_epoch",
    "best_score",
    "model",
    "seed",
    "lr",
    "batch_size",
    "epochs",
    "tcn_layers",
    "student_tcn_layers",
    "tcn_dropout",
    "student_tcn_dropout",
    "weight_decay",
    "temperature",
    "lambda_ce",
    "lambda_kd",
    "lambda_feat",
    "checkpoint_metric",
    "parse_status",
]


def parse_threshold(value: float) -> float:
    if value <= 1.0:
        return value
    if value <= 100.0:
        return value / 100.0
    raise ValueError("threshold must be <= 1.0 (ratio) or <= 100 (percent)")


def parse_test_accuracy(metrics_text: str) -> float | None:
    for pattern in TEST_ACCURACY_PATTERNS:
        match = pattern.search(metrics_text)
        if match:
            return float(match.group(1)) / 100.0
    return None


def parse_best_epoch(metrics_text: str) -> int | None:
    match = BEST_EPOCH_PATTERN.search(metrics_text)
    if not match:
        return None
    return int(match.group(1))


def parse_best_score(metrics_text: str) -> float | None:
    match = BEST_SCORE_PATTERN.search(metrics_text)
    if not match:
        return None
    return float(match.group(1))


def load_run_args(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def build_record(
    source_root: Path,
    run_dir: Path,
    test_accuracy: float,
    best_epoch: int | None,
    best_score: float | None,
    run_args: dict[str, Any],
    parse_status: str,
) -> dict[str, Any]:
    rel_run_dir = run_dir.relative_to(source_root)
    source_name = source_root.name
    return {
        "source_root": str(source_root),
        "source_name": source_name,
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "rel_run_dir": str(rel_run_dir),
        "test_accuracy": round(test_accuracy, 6),
        "test_accuracy_percent": round(test_accuracy * 100.0, 2),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "model": run_args.get("model"),
        "seed": run_args.get("seed"),
        "lr": run_args.get("lr"),
        "batch_size": run_args.get("batch_size"),
        "epochs": run_args.get("epochs"),
        "tcn_layers": run_args.get("tcn_layers"),
        "student_tcn_layers": run_args.get("student_tcn_layers"),
        "tcn_dropout": run_args.get("tcn_dropout"),
        "student_tcn_dropout": run_args.get("student_tcn_dropout"),
        "weight_decay": run_args.get("weight_decay"),
        "temperature": run_args.get("temperature"),
        "lambda_ce": run_args.get("lambda_ce"),
        "lambda_kd": run_args.get("lambda_kd"),
        "lambda_feat": run_args.get("lambda_feat"),
        "checkpoint_metric": run_args.get("checkpoint_metric"),
        "parse_status": parse_status,
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def contains_best_record_path(run_dir: Path, root_dir: Path) -> bool:
    rel = run_dir.relative_to(root_dir)
    return "best_record" in rel.parts


def discover_run_dirs(root_dir: Path, include_best_record: bool) -> list[Path]:
    run_dirs: set[Path] = set()

    for run_args_path in root_dir.rglob("run_args.json"):
        parent = run_args_path.parent
        if not include_best_record and contains_best_record_path(parent, root_dir):
            continue
        run_dirs.add(parent)

    for metrics_path in root_dir.rglob("evaluation_metrics.txt"):
        parent = metrics_path.parent
        if not include_best_record and contains_best_record_path(parent, root_dir):
            continue
        run_dirs.add(parent)

    return sorted(run_dirs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze experiment runs and keep only test accuracy >= threshold."
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        action="append",
        help="Experiment root directory. Can be specified multiple times.",
    )
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument(
        "--out-csv",
        type=str,
        default="outputs/analysis_test_acc_ge95.csv",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="outputs/analysis_test_acc_ge95.json",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--include-best-record", action="store_true")
    args = parser.parse_args()

    root_dirs_raw = args.root_dir or [
        "outputs/single_tcn",
        "outputs/distill_single_tcn",
    ]
    root_dirs: list[Path] = []
    seen: set[Path] = set()
    for root in root_dirs_raw:
        root_path = Path(root).resolve()
        if root_path in seen:
            continue
        seen.add(root_path)
        root_dirs.append(root_path)

    for root_dir in root_dirs:
        if not root_dir.exists() or not root_dir.is_dir():
            raise SystemExit(f"root-dir does not exist or is not a directory: {root_dir}")

    threshold = parse_threshold(args.threshold)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    selected_records: list[dict[str, Any]] = []

    total_runs = 0
    missing_eval = 0
    parse_failed = 0
    run_args_missing = 0
    parsed_ok = 0

    for root_dir in root_dirs:
        run_dirs = discover_run_dirs(root_dir, args.include_best_record)
        total_runs += len(run_dirs)

        for run_dir in run_dirs:
            eval_path = run_dir / "evaluation_metrics.txt"
            if not eval_path.exists():
                missing_eval += 1
                continue

            metrics_text = eval_path.read_text(encoding="utf-8", errors="ignore")
            test_accuracy = parse_test_accuracy(metrics_text)
            if test_accuracy is None:
                parse_failed += 1
                continue

            parsed_ok += 1
            if test_accuracy < threshold:
                continue

            run_args_path = run_dir / "run_args.json"
            run_args = load_run_args(run_args_path)
            parse_status = "ok"
            if not run_args:
                run_args_missing += 1
                parse_status = "ok_run_args_missing"

            record = build_record(
                source_root=root_dir,
                run_dir=run_dir,
                test_accuracy=test_accuracy,
                best_epoch=parse_best_epoch(metrics_text),
                best_score=parse_best_score(metrics_text),
                run_args=run_args,
                parse_status=parse_status,
            )
            selected_records.append(record)

    selected_records.sort(key=lambda x: (-x["test_accuracy"], x["run_name"]))

    write_csv(out_csv, selected_records)
    write_json(out_json, selected_records)

    top_k = max(args.top_k, 0)
    top_records = selected_records[:top_k]

    print("=" * 70)
    print("experiment analysis summary")
    print("=" * 70)
    print("Root dirs:")
    for root_dir in root_dirs:
        print(f"  - {root_dir}")
    print(f"Threshold: {threshold:.4f} ({threshold * 100:.2f}%)")
    print(f"Scanned runs: {total_runs}")
    print(f"Parsed runs: {parsed_ok}")
    print(f"Missing evaluation_metrics.txt: {missing_eval}")
    print(f"Failed to parse test accuracy: {parse_failed}")
    print(f"Selected runs (>= threshold): {len(selected_records)}")
    if run_args_missing:
        print(f"Selected runs missing/invalid run_args.json: {run_args_missing}")
    print(f"CSV output: {out_csv}")
    print(f"JSON output: {out_json}")

    if top_records:
        print("-" * 70)
        print(f"Top {len(top_records)} selected runs:")
        for idx, rec in enumerate(top_records, start=1):
            print(
                f"{idx:>2}. {rec['run_name']} | "
                f"test_acc={rec['test_accuracy_percent']:.2f}% | "
                f"seed={rec['seed']} | lr={rec['lr']} | batch_size={rec['batch_size']}"
            )
    else:
        print("-" * 70)
        print("No runs meet the threshold.")


if __name__ == "__main__":
    main()
