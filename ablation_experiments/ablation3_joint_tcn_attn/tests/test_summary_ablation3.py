#!/usr/bin/env python3
"""Regression test for integrating Ablation-3 into summary script."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUMMARY_SCRIPT = PROJECT_ROOT / "ablation_experiments/scripts/summarize_ablation_results.py"


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


class SummaryAblation3Test(unittest.TestCase):
    def test_summary_accepts_ablation3_and_exports_group(self) -> None:
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
                agree_pct=None,
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


if __name__ == "__main__":
    unittest.main()
