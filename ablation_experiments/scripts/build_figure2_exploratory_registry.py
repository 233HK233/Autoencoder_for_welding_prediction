#!/usr/bin/env python3
"""Build the exploratory Figure 2 run registry and class-metrics summary."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASS_ROW_PATTERN = re.compile(
    r"^\s*Class\s+([0-9]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+[0-9]+\s*$",
    flags=re.M,
)
TEST_REPORT_PATTERN = re.compile(
    r"=== Classification Report \(Test.*?\) ===\n(?P<body>.*?)(?:\n\s*accuracy|\Z)",
    flags=re.S,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build exploratory Figure 2 data tables"
    )
    parser.add_argument(
        "--teacher-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/teacher_h1_scan98_gpu1_v2/manifests/best_teacher_h1.json",
    )
    parser.add_argument(
        "--comparison-summary-csv",
        type=Path,
        default=PROJECT_ROOT
        / "ablation_experiments/h1/reports/comparison_18d_baselines/checkpoints/round3/comparison_summary.csv",
    )
    parser.add_argument(
        "--ablation-summary-csv",
        type=Path,
        default=PROJECT_ROOT
        / "ablation_experiments/h1/reports/run_20260327_131013/final_summary/ablation_summary.csv",
    )
    parser.add_argument(
        "--baseline-method",
        choices=("gru", "lstm"),
        default="gru",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/paper_figures/h1_results/data",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_class_metrics(metrics_path: Path) -> List[Dict[str, Any]]:
    text = metrics_path.read_text(encoding="utf-8", errors="ignore")
    test_match = TEST_REPORT_PATTERN.search(text)
    if not test_match:
        raise ValueError(f"failed to locate test classification report in {metrics_path}")
    report_body = test_match.group("body")
    rows: List[Dict[str, Any]] = []
    for match in CLASS_ROW_PATTERN.finditer(report_body):
        class_id = int(match.group(1))
        rows.extend(
            [
                {
                    "class_id": class_id,
                    "class_name": f"S{class_id}",
                    "metric": "precision",
                    "value": float(match.group(2)),
                },
                {
                    "class_id": class_id,
                    "class_name": f"S{class_id}",
                    "metric": "recall",
                    "value": float(match.group(3)),
                },
                {
                    "class_id": class_id,
                    "class_name": f"S{class_id}",
                    "metric": "f1",
                    "value": float(match.group(4)),
                },
            ]
        )
    if len(rows) != 9:
        raise ValueError(f"failed to parse 3-class metrics from {metrics_path}")
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_run_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_teacher_entry(manifest_path: Path) -> Dict[str, Any]:
    payload = load_json(manifest_path)
    best_run = payload["best_run"]
    checkpoint_metric = ""
    if isinstance(best_run.get("metrics_core"), dict):
        checkpoint_metric = str(best_run["metrics_core"].get("checkpoint_metric", ""))
    if not checkpoint_metric:
        run_args_path = Path(str(best_run["run_args_path"]))
        if run_args_path.exists():
            checkpoint_metric = str(load_json(run_args_path).get("checkpoint_metric", ""))
    return {
        "model_name": "Exploratory Teacher Best",
        "model_family": "teacher",
        "protocol": "exploratory",
        "run_dir": str(best_run["path"]),
        "teacher_source": "",
        "checkpoint_metric": checkpoint_metric,
        "selection_note": "direct reuse",
        "included_in_fig2": "yes",
        "metrics_path": str(best_run["evaluation_metrics_path"]),
    }


def build_comparison_entry(
    comparison_rows: Iterable[Dict[str, str]],
    *,
    method_name: str,
    display_name: str,
    model_family: str,
) -> Dict[str, Any]:
    for row in comparison_rows:
        if row["method_name"] != method_name:
            continue
        run_path = resolve_run_path(row["run_path"])
        return {
            "model_name": display_name,
            "model_family": model_family,
            "protocol": "exploratory",
            "run_dir": str(run_path),
            "teacher_source": "",
            "checkpoint_metric": row["checkpoint_metric"],
            "selection_note": "direct reuse",
            "included_in_fig2": "yes",
            "metrics_path": str(run_path / "evaluation_metrics.txt"),
        }
    raise FileNotFoundError(f"method {method_name} not found in comparison summary")


def build_ablation_entry(
    ablation_rows: Iterable[Dict[str, str]],
    *,
    group: str,
    display_name: str,
) -> Dict[str, Any]:
    for row in ablation_rows:
        if row["group"] != group:
            continue
        run_path = resolve_run_path(row["best_run_dir"])
        run_args_path = run_path / "run_args.json"
        checkpoint_metric = ""
        if run_args_path.exists():
            checkpoint_metric = str(load_json(run_args_path).get("checkpoint_metric", ""))
        return {
            "model_name": display_name,
            "model_family": "ablation",
            "protocol": "exploratory",
            "run_dir": str(run_path),
            "teacher_source": "",
            "checkpoint_metric": checkpoint_metric,
            "selection_note": "direct reuse",
            "included_in_fig2": "yes",
            "metrics_path": str(run_path / "evaluation_metrics.txt"),
        }
    raise FileNotFoundError(f"group {group} not found in ablation summary")


def build_class_metric_rows(registry_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in registry_rows:
        metrics_path = Path(entry["metrics_path"])
        for metric_row in parse_class_metrics(metrics_path):
            rows.append(
                {
                    "model_name": entry["model_name"],
                    "model_family": entry["model_family"],
                    "protocol": "exploratory",
                    "class_id": metric_row["class_id"],
                    "class_name": metric_row["class_name"],
                    "metric": metric_row["metric"],
                    "value": metric_row["value"],
                    "source_run_dir": entry["run_dir"],
                }
            )
    return rows


def main() -> None:
    args = parse_args()

    comparison_rows = load_csv_rows(args.comparison_summary_csv)
    ablation_rows = load_csv_rows(args.ablation_summary_csv)

    baseline_name = {
        "gru": "Best GRU baseline",
        "lstm": "Best LSTM baseline",
    }[args.baseline_method]

    registry_rows = [
        build_teacher_entry(args.teacher_manifest),
        build_comparison_entry(
            comparison_rows,
            method_name="teacher-student(student)",
            display_name="Best Student Distill",
            model_family="student",
        ),
        build_ablation_entry(
            ablation_rows,
            group="ablation1_student_only_tcn_attn",
            display_name="Best Ablation-1",
        ),
        build_ablation_entry(
            ablation_rows,
            group="ablation2_teacher_student_lstm",
            display_name="Best Ablation-2",
        ),
        build_comparison_entry(
            comparison_rows,
            method_name=args.baseline_method,
            display_name=baseline_name,
            model_family="baseline",
        ),
    ]

    public_registry = [{k: v for k, v in row.items() if k != "metrics_path"} for row in registry_rows]
    class_metric_rows = build_class_metric_rows(registry_rows)

    out_dir = args.out_dir
    write_csv(out_dir / "exploratory_run_registry.csv", public_registry)
    write_csv(out_dir / "exploratory_class_metrics_summary.csv", class_metric_rows)

    print(f"Registry: {out_dir / 'exploratory_run_registry.csv'}")
    print(f"Class metrics: {out_dir / 'exploratory_class_metrics_summary.csv'}")


if __name__ == "__main__":
    main()
