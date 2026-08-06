#!/usr/bin/env python3
"""Build the ordered exploratory Figure 2 registry and class-metrics summary."""

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
        description="Build ordered exploratory Figure 2 data tables"
    )
    parser.add_argument(
        "--teacher-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/teacher_h1_scan98_gpu1_v2/manifests/best_teacher_h1.json",
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
        raise ValueError(
            f"failed to locate test classification report in {metrics_path}"
        )
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
        "model_name": "Exploratory Teacher Best",
        "model_family": "teacher",
        "protocol": "exploratory_ordered",
        "run_dir": str(best_run["path"]),
        "teacher_source": "",
        "checkpoint_metric": checkpoint_metric,
        "selection_note": "ordered exploratory reuse",
        "included_in_fig2": "yes",
        "metrics_path": str(best_run["evaluation_metrics_path"]),
    }


def build_run_entry(
    *,
    model_name: str,
    model_family: str,
    run_dir: Path,
    checkpoint_metric: str,
    teacher_source: str = "",
) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "model_family": model_family,
        "protocol": "exploratory_ordered",
        "run_dir": str(run_dir),
        "teacher_source": teacher_source,
        "checkpoint_metric": checkpoint_metric,
        "selection_note": "ordered exploratory reuse",
        "included_in_fig2": "yes",
        "metrics_path": str(run_dir / "evaluation_metrics.txt"),
    }


def build_class_metric_rows(
    registry_rows: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in registry_rows:
        metrics_path = Path(entry["metrics_path"])
        for metric_row in parse_class_metrics(metrics_path):
            rows.append(
                {
                    "model_name": entry["model_name"],
                    "model_family": entry["model_family"],
                    "protocol": "exploratory_ordered",
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

    teacher_entry = build_teacher_entry(args.teacher_manifest)

    student_run = (
        PROJECT_ROOT
        / "ablation_experiments/h1/results/comparison_18d_baselines/teacher_student_reference"
        / "student_h1_sweep/student_h1_scan96_20260326/stage3"
        / "s3_s2_s1_l3_o2_c2_r5"
        / "distill_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep80_lr0.0002_bs128_T3.5_lce0.7_lkd1.5_lf0.3_seed14"
    )
    ablation1_run = (
        PROJECT_ROOT
        / "ablation_experiments/h1/results/run_20260327_131013/ablation1_sweep/trials/trial_0031_A"
        / "ablation1_student_only_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep80_lr0.0001_bs64_seed144"
    )
    ablation2_run = (
        PROJECT_ROOT
        / "ablation_experiments/h1/results/run_20260327_131013/ablation2_teacher_student_lstm/students"
        / "ablation2_h1_scan90_run_20260327_131013/stage3/s3_s2_s1_l3_o2_c1_r2/seed_77"
        / "ablation2_distill_lstm_weld_seam_windows_ws5_tf75_pg0_h1_ep80_lr0.00025_bs128_T3.0_lce0.8_lkd1.3_lf0.30000000000000004_seed77"
    )

    registry_rows = [
        teacher_entry,
        build_run_entry(
            model_name="Best Student Distill",
            model_family="student",
            run_dir=student_run,
            checkpoint_metric="val_teacher_agreement",
            teacher_source=str(args.teacher_manifest),
        ),
        build_run_entry(
            model_name="Best Ablation-1",
            model_family="ablation",
            run_dir=ablation1_run,
            checkpoint_metric="test_acc",
        ),
        build_run_entry(
            model_name="Best Ablation-2",
            model_family="ablation",
            run_dir=ablation2_run,
            checkpoint_metric="test_acc",
            teacher_source="comparison_18d_baselines/lstm",
        ),
    ]

    public_registry = [
        {k: v for k, v in row.items() if k != "metrics_path"} for row in registry_rows
    ]
    class_metric_rows = build_class_metric_rows(registry_rows)

    out_dir = args.out_dir
    write_csv(out_dir / "exploratory_ordered_run_registry.csv", public_registry)
    write_csv(
        out_dir / "exploratory_ordered_class_metrics_summary.csv",
        class_metric_rows,
    )

    print(f"Registry: {out_dir / 'exploratory_ordered_run_registry.csv'}")
    print(
        "Class metrics:"
        f" {out_dir / 'exploratory_ordered_class_metrics_summary.csv'}"
    )


if __name__ == "__main__":
    main()
