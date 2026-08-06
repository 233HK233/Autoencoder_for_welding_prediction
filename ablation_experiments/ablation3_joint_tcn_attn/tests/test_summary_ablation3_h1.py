#!/usr/bin/env python3
"""Regression test for horizon=1 ablation summary defaults and report text."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUMMARY_SCRIPT = PROJECT_ROOT / "ablation_experiments/scripts/summarize_ablation_results.py"
H1_RESULTS_ROOT = PROJECT_ROOT / "ablation_experiments/h1/results"
H1_REPORTS_ROOT = PROJECT_ROOT / "ablation_experiments/h1/reports"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_metrics(run_dir: Path, acc_pct: float, macro_f1: float, agree_pct: float | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "--- Test Metrics ---",
        f"Accuracy: {acc_pct:.2f}%",
        f"Macro-F1: {macro_f1:.4f}",
    ]
    if agree_pct is not None:
        lines.append(f"Test Teacher-Agreement: {agree_pct:.2f}%")
    (run_dir / "evaluation_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


class SummaryAblation3H1DefaultsTest(unittest.TestCase):
    def test_parse_args_defaults_to_h1_roots(self) -> None:
        mod = load_module("summarize_ablation_results_h1_defaults_test", SUMMARY_SCRIPT)

        with mock.patch.object(sys, "argv", ["summarize_ablation_results.py"]):
            args = mod.parse_args()

        self.assertEqual(args.ablation3_dir, H1_RESULTS_ROOT / "ablation3_joint_tcn_attn")
        self.assertEqual(args.out_dir, H1_REPORTS_ROOT)

    def test_summary_report_mentions_future_step_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            a1_dir = tmp_path / "a1"
            a2_dir = tmp_path / "a2"
            a3_dir = tmp_path / "a3"
            out_dir = tmp_path / "reports"

            _write_metrics(
                a1_dir / "ablation1_student_only_tcn_attn_dummy_seed100",
                acc_pct=94.10,
                macro_f1=0.9410,
            )
            _write_metrics(
                a2_dir / "ablation2_distill_lstm_dummy_seed100",
                acc_pct=95.20,
                macro_f1=0.9520,
                agree_pct=98.00,
            )
            _write_metrics(
                a3_dir / "ablation3_joint_tcn_attn_dummy_seed100",
                acc_pct=95.60,
                macro_f1=0.9560,
                agree_pct=98.20,
            )

            cmd = [
                sys.executable,
                str(SUMMARY_SCRIPT),
                "--ablation1-dir",
                str(a1_dir),
                "--ablation2-dir",
                str(a2_dir),
                "--ablation3-dir",
                str(a3_dir),
                "--out-dir",
                str(out_dir),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                0,
                proc.returncode,
                msg=f"summary script failed. stdout={proc.stdout}\nstderr={proc.stderr}",
            )

            payload = json.loads((out_dir / "ablation_summary.json").read_text(encoding="utf-8"))
            groups = {row["group"] for row in payload["group_summaries"]}
            self.assertIn("ablation3_joint_teacher_student_tcn_attn", groups)

            md_text = (out_dir / "ablation_comparison.md").read_text(encoding="utf-8")
            self.assertIn("future-step prediction", md_text)


if __name__ == "__main__":
    unittest.main()
