#!/usr/bin/env python3
"""Build the official strict Figure 2 run registry and class-metrics summary."""

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
    parser = argparse.ArgumentParser(description="Build official strict Figure 2 data tables")
    parser.add_argument(
        "--teacher-manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs/teacher_h1_forecast_95/manifests/best_teacher_h1.json",
    )
    parser.add_argument(
        "--student-summary-json",
        type=Path,
        default=PROJECT_ROOT
        / "ablation_experiments/h1/reports/comparison_18d_baselines/teacher_student_reference/student_h1_strict_valf1_20260522/student_h1_strict_valf1_20260522/distill_sweep_summary.json",
    )
    parser.add_argument(
        "--ablation1-summary-json",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/h1/reports/ablation1_sweep_strict_valf1_20260522/ablation1_sweep_runtime_summary.json",
    )
    parser.add_argument(
        "--ablation2-summary-json",
        type=Path,
        default=PROJECT_ROOT
        / "ablation_experiments/h1/reports/ablation2_teacher_student_lstm/ablation2_strict_valf1_20260522/ablation2_h1_scan90_20260327/ablation2_sweep_summary.json",
    )
    parser.add_argument(
        "--comparison-summary-csv",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/h1/reports/comparison_18d_baselines/comparison_summary.csv",
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
        "model_name": "Strict Teacher Best",
        "model_family": "teacher",
        "protocol": "strict",
        "run_dir": str(best_run["path"]),
        "teacher_source": "",
        "checkpoint_metric": checkpoint_metric,
        "selection_note": "direct reuse",
        "included_in_fig2": "yes",
        "metrics_path": str(best_run["evaluation_metrics_path"]),
    }


def build_summary_entry(
    summary_path: Path,
    *,
    model_name: str,
    model_family: str,
    teacher_source: str = "",
    summary_key: str = "best_run",
) -> Dict[str, Any]:
    payload = load_json(summary_path)
    best_run = payload[summary_key]
    return {
        "model_name": model_name,
        "model_family": model_family,
        "protocol": "strict",
        "run_dir": str(best_run["run_dir"]),
        "teacher_source": teacher_source,
        "checkpoint_metric": str(best_run["checkpoint_metric"]),
        "selection_note": "strict rerun",
        "included_in_fig2": "yes",
        "metrics_path": str(Path(best_run["run_dir"]) / "evaluation_metrics.txt"),
    }


def build_baseline_entry(summary_csv: Path, method_name: str, display_name: str) -> Dict[str, Any]:
    with summary_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row["method_name"] != method_name:
            continue
        return {
            "model_name": display_name,
            "model_family": "baseline",
            "protocol": "strict",
            "run_dir": row["run_path"],
            "teacher_source": "",
            "checkpoint_metric": row["checkpoint_metric"],
            "selection_note": "direct reuse",
            "included_in_fig2": "yes",
            "metrics_path": str(Path(row["run_path"]) / "evaluation_metrics.txt"),
        }
    raise FileNotFoundError(f"baseline method {method_name} not found in {summary_csv}")


def build_class_metric_rows(registry_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in registry_rows:
        metrics_path = Path(entry["metrics_path"])
        for metric_row in parse_class_metrics(metrics_path):
            rows.append(
                {
                    "model_name": entry["model_name"],
                    "model_family": entry["model_family"],
                    "protocol": "strict",
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

    registry_rows = [
        build_teacher_entry(args.teacher_manifest),
        build_summary_entry(
            args.student_summary_json,
            model_name="Best Student Distill",
            model_family="student",
            teacher_source=str(args.teacher_manifest),
        ),
        build_summary_entry(
            args.ablation1_summary_json,
            model_name="Best Ablation-1",
            model_family="ablation",
            summary_key="best_trial",
        ),
        build_summary_entry(
            args.ablation2_summary_json,
            model_name="Best Ablation-2",
            model_family="ablation",
            teacher_source="comparison_18d_baselines/lstm",
        ),
        build_baseline_entry(
            args.comparison_summary_csv,
            method_name="lstm",
            display_name="Best LSTM baseline",
        ),
    ]

    public_registry = [{k: v for k, v in row.items() if k != "metrics_path"} for row in registry_rows]
    class_metric_rows = build_class_metric_rows(registry_rows)

    out_dir = args.out_dir
    write_csv(out_dir / "official_run_registry.csv", public_registry)
    write_csv(out_dir / "class_metrics_summary.csv", class_metric_rows)

    print(f"Registry: {out_dir / 'official_run_registry.csv'}")
    print(f"Class metrics: {out_dir / 'class_metrics_summary.csv'}")


if __name__ == "__main__":
    main()
