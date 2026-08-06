#!/usr/bin/env python3
"""Summarize ablation-1 sweep progress into concise report files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ablation-1 sweep results")
    parser.add_argument(
        "--progress-json",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/reports/ablation1_sweep_progress.json",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/reports/ablation1_sweep_summary.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=PROJECT_ROOT / "ablation_experiments/reports/ablation1_topk.md",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--target-acc", type=float, default=90.0, help="Use percent if >1.0 else ratio")
    return parser.parse_args()


def parse_threshold(v: float) -> float:
    return v / 100.0 if v > 1.0 else v


def load_records(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("progress json must be a list")
    return payload


def sort_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda r: (
            float(r.get("test_acc", -1.0)),
            float(r.get("val_macro_f1", -1.0)),
            -int(r.get("trial_id", 0)),
        ),
        reverse=True,
    )


def build_summary(records: List[Dict[str, Any]], target: float) -> Dict[str, Any]:
    success = [r for r in records if str(r.get("status", "")).startswith("success")]
    valid = [r for r in success if float(r.get("test_acc", -1.0)) >= 0.0 and float(r.get("val_macro_f1", -1.0)) >= 0.0]

    accs = [float(r["test_acc"]) for r in valid]
    f1s = [float(r["val_macro_f1"]) for r in valid]

    best_test = max(valid, key=lambda r: float(r["test_acc"])) if valid else None
    best_val = max(valid, key=lambda r: float(r["val_macro_f1"])) if valid else None
    reached = [r for r in valid if float(r["test_acc"]) >= target]

    return {
        "target_test_acc": target,
        "total_trials": len(records),
        "success_trials": len(success),
        "valid_trials": len(valid),
        "target_reached": len(reached) > 0,
        "target_reached_count": len(reached),
        "best_test": best_test,
        "best_val": best_val,
        "mean_test_acc": mean(accs) if accs else -1.0,
        "std_test_acc": pstdev(accs) if len(accs) > 1 else 0.0,
        "mean_val_macro_f1": mean(f1s) if f1s else -1.0,
        "std_val_macro_f1": pstdev(f1s) if len(f1s) > 1 else 0.0,
    }


def build_md(topk: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Ablation-1 Sweep Top-K")
    lines.append("")
    lines.append(f"- Target test_acc: {summary['target_test_acc']*100:.2f}%")
    lines.append(f"- Total trials: {summary['total_trials']}")
    lines.append(f"- Valid trials: {summary['valid_trials']}")
    lines.append(f"- Target reached: {summary['target_reached']} ({summary['target_reached_count']})")
    lines.append("")

    if summary["best_test"]:
        best = summary["best_test"]
        lines.append("## Best by Test Acc")
        lines.append("")
        lines.append(f"- trial_id: {best['trial_id']}")
        lines.append(f"- stage: {best['stage']}")
        lines.append(f"- test_acc: {float(best['test_acc'])*100:.2f}%")
        lines.append(f"- val_macro_f1: {float(best['val_macro_f1']):.4f}")
        lines.append(f"- run_dir: {best['run_dir']}")
        lines.append("- reproduce command:")
        lines.append("```bash")
        lines.append(str(best["command"]))
        lines.append("```")
        lines.append("")

    lines.append("## Top-K Table")
    lines.append("")
    lines.append("| Rank | Trial | Stage | Status | Test Acc | Val Macro-F1 | Seed | lr | wd | bs |")
    lines.append("|---:|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for idx, item in enumerate(topk, start=1):
        lines.append(
            f"| {idx} | {item['trial_id']} | {item['stage']} | {item['status']} | "
            f"{float(item['test_acc'])*100:.2f}% | {float(item['val_macro_f1']):.4f} | "
            f"{item['seed']} | {item['lr']} | {item['weight_decay']} | {item['batch_size']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.progress_json.exists():
        raise SystemExit(f"progress json not found: {args.progress_json}")

    target = parse_threshold(args.target_acc)
    records = load_records(args.progress_json)
    sorted_records = sort_records(records)
    summary = build_summary(sorted_records, target)

    topk = sorted_records[: max(0, args.top_k)]
    summary["topk"] = topk

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.out_md.write_text(build_md(topk, summary) + "\n", encoding="utf-8")

    print("Summary generated.")
    print(f"Summary JSON: {args.out_json}")
    print(f"Top-K Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
