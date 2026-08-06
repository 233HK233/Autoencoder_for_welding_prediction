#!/usr/bin/env python3
"""Summarize teacher-student and 18D baseline comparison results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_ACC_PATTERN = re.compile(
    r"---\s*Test Metrics.*?Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%",
    flags=re.S,
)
TEST_F1_PATTERN = re.compile(
    r"---\s*Test Metrics.*?Macro-F1:\s*([0-9]+(?:\.[0-9]+)?)",
    flags=re.S,
)
TEST_AGREE_PATTERN = re.compile(r"Test Teacher-Agreement:\s*([0-9]+(?:\.[0-9]+)?)%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize h1 comparison results")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/h1/results/comparison_18d_baselines",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/h1/reports/comparison_18d_baselines",
    )
    parser.add_argument(
        "--teacher-manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs/teacher_h1_scan98_gpu1_v2/manifests/best_teacher_h1.json",
    )
    return parser.parse_args()


def parse_metrics(metrics_path: Path) -> Optional[Dict[str, float]]:
    if not metrics_path.exists():
        return None
    text = metrics_path.read_text(encoding="utf-8", errors="ignore")
    m_acc = TEST_ACC_PATTERN.search(text)
    m_f1 = TEST_F1_PATTERN.search(text)
    if not m_acc or not m_f1:
        return None
    m_agree = TEST_AGREE_PATTERN.search(text)
    return {
        "test_acc": float(m_acc.group(1)) / 100.0,
        "macro_f1": float(m_f1.group(1)),
        "teacher_agreement": float(m_agree.group(1)) / 100.0 if m_agree else -1.0,
    }


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_run_records(root: Path, group: str) -> List[Dict[str, object]]:
    if not root.exists():
        return []

    records: List[Dict[str, object]] = []
    for metrics_path in sorted(root.rglob("evaluation_metrics.txt")):
        run_dir = metrics_path.parent
        parsed = parse_metrics(metrics_path)
        if parsed is None:
            continue
        run_args_path = run_dir / "run_args.json"
        run_args = load_json(run_args_path) if run_args_path.exists() else {}
        records.append(
            {
                "group": group,
                "run_name": run_dir.name,
                "run_path": str(run_dir),
                "seed": int(run_args.get("seed", -1)),
                "checkpoint_metric": str(run_args.get("checkpoint_metric", "")),
                "test_acc": parsed["test_acc"],
                "macro_f1": parsed["macro_f1"],
                "teacher_agreement": parsed["teacher_agreement"],
            }
        )
    return records


def pick_best(records: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not records:
        return None
    return max(records, key=lambda row: (float(row["test_acc"]), float(row["macro_f1"])))


def build_method_row(
    method_name: str,
    train_scheme: str,
    deploy_input_dim: int,
    teacher_input_dim: int | None,
    best: Dict[str, object],
    notes: str,
) -> Dict[str, object]:
    return {
        "method_name": method_name,
        "train_scheme": train_scheme,
        "deploy_input_dim": deploy_input_dim,
        "teacher_input_dim": teacher_input_dim,
        "best_test_acc": round(float(best["test_acc"]), 6),
        "best_macro_f1": round(float(best["macro_f1"]), 6),
        "teacher_agreement": round(float(best["teacher_agreement"]), 6),
        "seed": int(best["seed"]),
        "checkpoint_metric": str(best["checkpoint_metric"]),
        "run_path": str(best["run_path"]),
        "notes": notes,
    }


def build_teacher_row(manifest_path: Path) -> Dict[str, object]:
    payload = load_json(manifest_path)
    best_run = payload.get("best_run", {})
    if not isinstance(best_run, dict):
        raise ValueError("teacher manifest best_run must be an object")
    metrics_path = Path(str(best_run["evaluation_metrics_path"]))
    parsed = parse_metrics(metrics_path)
    if parsed is None:
        raise ValueError(f"teacher metrics could not be parsed: {metrics_path}")
    run_args_path = Path(str(best_run["run_args_path"]))
    run_args = load_json(run_args_path) if run_args_path.exists() else {}
    best = {
        "test_acc": parsed["test_acc"],
        "macro_f1": parsed["macro_f1"],
        "teacher_agreement": -1.0,
        "seed": int(run_args.get("seed", -1)),
        "checkpoint_metric": str(run_args.get("checkpoint_metric", "")),
        "run_path": str(best_run.get("path", metrics_path.parent)),
    }
    return build_method_row(
        method_name="teacher(18D upper bound)",
        train_scheme="teacher_only",
        deploy_input_dim=18,
        teacher_input_dim=18,
        best=best,
        notes="Best h1 teacher reference from manifest",
    )


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict[str, object]] = []
    method_rows: List[Dict[str, object]] = []

    student_runs = collect_run_records(
        args.results_root / "teacher_student_reference/student_h1_sweep",
        "teacher-student(student)",
    )
    run_rows.extend(student_runs)
    student_best = pick_best(student_runs)
    if student_best is not None:
        method_rows.append(
            build_method_row(
                method_name="teacher-student(student)",
                train_scheme="frozen_teacher_distill",
                deploy_input_dim=13,
                teacher_input_dim=18,
                best=student_best,
                notes="13D deploy student distilled from 18D teacher",
            )
        )

    baseline_specs = [
        ("lstm", "baseline_18d"),
        ("gru", "baseline_18d"),
        ("transformer", "baseline_18d"),
        ("inception", "baseline_18d"),
    ]
    for name, train_scheme in baseline_specs:
        records = collect_run_records(args.results_root / name, name)
        run_rows.extend(records)
        best = pick_best(records)
        if best is None:
            continue
        method_rows.append(
            build_method_row(
                method_name=name,
                train_scheme=train_scheme,
                deploy_input_dim=18,
                teacher_input_dim=None,
                best=best,
                notes=f"Best 18D {name} baseline run",
            )
        )

    method_rows.append(build_teacher_row(args.teacher_manifest))

    ordered_names = [
        "teacher-student(student)",
        "lstm",
        "gru",
        "transformer",
        "inception",
        "teacher(18D upper bound)",
    ]
    ordered_rows = [row for name in ordered_names for row in method_rows if row["method_name"] == name]

    write_csv(args.report_dir / "comparison_runs.csv", run_rows)
    write_csv(args.report_dir / "comparison_summary.csv", ordered_rows)

    summary_payload = {
        "results_root": str(args.results_root),
        "teacher_manifest": str(args.teacher_manifest),
        "num_runs": len(run_rows),
        "main_table": ordered_rows,
    }
    (args.report_dir / "comparison_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    md_lines = [
        "# H1 Comparison Report",
        "",
        "| Method | Train Scheme | Deploy Input | Teacher Input | Best Test Acc | Best Macro-F1 | Seed | Notes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in ordered_rows:
        teacher_dim = "" if row["teacher_input_dim"] is None else str(row["teacher_input_dim"])
        md_lines.append(
            f"| {row['method_name']} | {row['train_scheme']} | {row['deploy_input_dim']} | "
            f"{teacher_dim} | {float(row['best_test_acc'])*100:.2f}% | {float(row['best_macro_f1']):.4f} | "
            f"{row['seed']} | {row['notes']} |"
        )
    (args.report_dir / "comparison_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("Comparison summary generated.")
    print(f"Runs CSV: {args.report_dir / 'comparison_runs.csv'}")
    print(f"Summary CSV: {args.report_dir / 'comparison_summary.csv'}")
    print(f"Summary JSON: {args.report_dir / 'comparison_summary.json'}")
    print(f"Markdown report: {args.report_dir / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
