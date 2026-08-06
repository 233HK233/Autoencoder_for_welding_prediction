#!/usr/bin/env python3
"""Summarize future-step prediction ablation results into CSV/JSON/Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
SEED_PATTERN = re.compile(r"seed(\d+)")


@dataclass
class RunMetric:
    group: str
    run_name: str
    run_dir: str
    seed: int
    test_acc: float
    test_macro_f1: float
    test_teacher_agreement: float
    dataset_npz: str


@dataclass
class GroupSummary:
    group: str
    count: int
    mean_test_acc: float
    std_test_acc: float
    mean_test_macro_f1: float
    std_test_macro_f1: float
    mean_test_teacher_agreement: float
    std_test_teacher_agreement: float
    best_run_name: str
    best_run_dir: str
    best_seed: int
    best_test_acc: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize h1 ablation results")
    parser.add_argument(
        "--ablation1-dir",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/h1/results/ablation1_student_only_tcn_attn",
    )
    parser.add_argument(
        "--ablation2-dir",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/h1/results/ablation2_teacher_student_lstm/students",
    )
    parser.add_argument(
        "--ablation3-dir",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/h1/results/ablation3_joint_tcn_attn",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/distill_single_tcn",
    )
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/h1/reports",
    )
    return parser.parse_args()


def parse_metrics_file(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    m_acc = TEST_ACC_PATTERN.search(text)
    m_f1 = TEST_F1_PATTERN.search(text)
    m_agree = TEST_AGREE_PATTERN.search(text)

    if not m_acc or not m_f1:
        return None

    return {
        "test_acc": float(m_acc.group(1)) / 100.0,
        "test_macro_f1": float(m_f1.group(1)),
        "test_teacher_agreement": float(m_agree.group(1)) / 100.0 if m_agree else -1.0,
    }


def parse_seed(run_name: str) -> int:
    m = SEED_PATTERN.search(run_name)
    return int(m.group(1)) if m else -1


def parse_run_args(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def collect_group_runs(group: str, root: Path) -> List[RunMetric]:
    if not root.exists():
        return []
    records: List[RunMetric] = []
    for metrics_path in sorted(root.rglob("evaluation_metrics.txt")):
        run_dir = metrics_path.parent
        parsed = parse_metrics_file(metrics_path)
        if parsed is None:
            continue
        run_args = parse_run_args(run_dir / "run_args.json")
        records.append(
            RunMetric(
                group=group,
                run_name=run_dir.name,
                run_dir=str(run_dir),
                seed=parse_seed(run_dir.name),
                test_acc=parsed["test_acc"],
                test_macro_f1=parsed["test_macro_f1"],
                test_teacher_agreement=parsed["test_teacher_agreement"],
                dataset_npz=str(run_args.get("dataset_npz", "")),
            )
        )
    return records


def summarize_group(group: str, rows: List[RunMetric]) -> Optional[GroupSummary]:
    if not rows:
        return None

    accs = [r.test_acc for r in rows]
    f1s = [r.test_macro_f1 for r in rows]
    agrees = [r.test_teacher_agreement for r in rows if r.test_teacher_agreement >= 0.0]

    best = max(rows, key=lambda x: x.test_acc)
    return GroupSummary(
        group=group,
        count=len(rows),
        mean_test_acc=mean(accs),
        std_test_acc=pstdev(accs) if len(accs) > 1 else 0.0,
        mean_test_macro_f1=mean(f1s),
        std_test_macro_f1=pstdev(f1s) if len(f1s) > 1 else 0.0,
        mean_test_teacher_agreement=mean(agrees) if agrees else -1.0,
        std_test_teacher_agreement=pstdev(agrees) if len(agrees) > 1 else 0.0,
        best_run_name=best.run_name,
        best_run_dir=best.run_dir,
        best_seed=best.seed,
        best_test_acc=best.test_acc,
    )


def write_best_run_manifests(out_dir: Path, rows: List[RunMetric], target_acc: float = 0.90) -> List[Dict[str, object]]:
    manifests: List[Dict[str, object]] = []
    grouped: Dict[str, List[RunMetric]] = {}
    for row in rows:
        grouped.setdefault(row.group, []).append(row)

    for group, group_rows in grouped.items():
        best = max(group_rows, key=lambda item: item.test_acc)
        dataset_check: Dict[str, object] | None = None
        if best.dataset_npz and Path(best.dataset_npz).exists():
            try:
                dataset_check = validate_forecast_dataset_contract(best.dataset_npz)
            except Exception as exc:
                dataset_check = {"dataset_path": best.dataset_npz, "validation_error": str(exc)}

        acceptance = {
            "target_horizon_steps_ok": bool(dataset_check and dataset_check.get("horizon_ok", False)),
            "target_accuracy_threshold": float(target_acc),
            "target_accuracy_reached": bool(best.test_acc >= target_acc),
        }
        payload = {
            "group": group,
            "best_run_name": best.run_name,
            "best_run_dir": best.run_dir,
            "best_seed": best.seed,
            "best_test_acc": best.test_acc,
            "best_test_macro_f1": best.test_macro_f1,
            "best_test_teacher_agreement": best.test_teacher_agreement,
            "dataset_npz": best.dataset_npz,
            "dataset_check": dataset_check,
            "acceptance": acceptance,
        }
        path = out_dir / f"{group}_best_run_manifest.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        manifests.append({"group": group, "path": str(path), **payload})
    return manifests


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[RunMetric] = []
    all_rows.extend(collect_group_runs("ablation1_student_only_tcn_attn", args.ablation1_dir))
    all_rows.extend(collect_group_runs("ablation2_teacher_student_lstm", args.ablation2_dir))
    all_rows.extend(collect_group_runs("ablation3_joint_teacher_student_tcn_attn", args.ablation3_dir))
    if args.include_baseline:
        all_rows.extend(collect_group_runs("baseline_tcn_attn_distill", args.baseline_dir))

    grouped: Dict[str, List[RunMetric]] = {}
    for row in all_rows:
        grouped.setdefault(row.group, []).append(row)

    summaries: List[GroupSummary] = []
    for group_name, rows in grouped.items():
        summary = summarize_group(group_name, rows)
        if summary is not None:
            summaries.append(summary)

    runs_csv = args.out_dir / "ablation_runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(RunMetric.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in all_rows)

    summary_csv = args.out_dir / "ablation_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(GroupSummary.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in summaries)

    summary_json = args.out_dir / "ablation_summary.json"
    payload = {
        "ablation1_dir": str(args.ablation1_dir),
        "ablation2_dir": str(args.ablation2_dir),
        "ablation3_dir": str(args.ablation3_dir),
        "baseline_dir": str(args.baseline_dir),
        "include_baseline": args.include_baseline,
        "num_runs": len(all_rows),
        "group_summaries": [asdict(r) for r in summaries],
        "best_run_manifests": write_best_run_manifests(args.out_dir, all_rows),
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = ["# Ablation Comparison Report", ""]
    md_lines.append("## Group Summary")
    md_lines.append("")
    md_lines.append(
        "| Group | N | Mean Acc | Std Acc | Mean Macro-F1 | Std Macro-F1 | Mean Teacher Agreement | Best Seed | Best Acc |"
    )
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for s in sorted(summaries, key=lambda x: x.group):
        agree_text = f"{s.mean_test_teacher_agreement*100:.2f}%" if s.mean_test_teacher_agreement >= 0 else "N/A"
        md_lines.append(
            f"| {s.group} | {s.count} | {s.mean_test_acc*100:.2f}% | {s.std_test_acc*100:.2f}% | "
            f"{s.mean_test_macro_f1:.4f} | {s.std_test_macro_f1:.4f} | {agree_text} | "
            f"{s.best_seed} | {s.best_test_acc*100:.2f}% |"
        )

    md_lines.append("")
    md_lines.append("## Notes")
    md_lines.append("")
    md_lines.append("- Ablation-1: Student-only (13D, TCN+Attention) future-step prediction")
    md_lines.append("- Ablation-2: Teacher+Student distillation with LSTM backbone for future-step prediction")
    md_lines.append("- Ablation-3: Joint online teacher-student distillation with TCN+Attention for future-step prediction")
    md_lines.append("- Metrics are parsed from each run's evaluation_metrics.txt")

    md_report = args.out_dir / "ablation_comparison.md"
    md_report.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("Summary generated.")
    print(f"Runs CSV: {runs_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Markdown report: {md_report}")


if __name__ == "__main__":
    main()
