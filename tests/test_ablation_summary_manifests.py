import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SCRIPT = PROJECT_ROOT / "ablation_experiments/scripts/summarize_ablation_results.py"
H1_DATASET = PROJECT_ROOT / "Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz"


def write_metrics(run_dir: Path, acc_pct: float, macro_f1: float, agree_pct: float | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "Best epoch: 3",
        "Checkpoint metric: test_acc",
        "Best score: 0.950000",
        "",
        "--- Test Metrics ---",
        f"Accuracy: {acc_pct:.2f}%",
        f"Macro-F1: {macro_f1:.4f}",
    ]
    if agree_pct is not None:
        lines.append(f"Test Teacher-Agreement: {agree_pct:.2f}%")
    (run_dir / "evaluation_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "run_args.json").write_text(
        json.dumps({"dataset_npz": str(H1_DATASET), "checkpoint_metric": "test_acc"}),
        encoding="utf-8",
    )


class AblationSummaryManifestTests(unittest.TestCase):
    def test_summary_recurses_into_sweep_trials_and_writes_best_run_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            a1_dir = tmp_path / "a1_trials"
            a2_dir = tmp_path / "a2_runs"
            a3_dir = tmp_path / "a3_runs"
            out_dir = tmp_path / "reports"

            write_metrics(a1_dir / "trial_0001" / "run_low", acc_pct=88.10, macro_f1=0.8610)
            write_metrics(a1_dir / "trial_0002" / "run_high", acc_pct=92.40, macro_f1=0.9120)
            write_metrics(a2_dir / "ablation2_best_seed14", acc_pct=93.10, macro_f1=0.9210, agree_pct=96.20)
            write_metrics(a3_dir / "ablation3_best_seed14", acc_pct=94.20, macro_f1=0.9350, agree_pct=97.10)

            proc = subprocess.run(
                [
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
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)

            a1_manifest = json.loads((out_dir / "ablation1_student_only_tcn_attn_best_run_manifest.json").read_text(encoding="utf-8"))
            a2_manifest = json.loads((out_dir / "ablation2_teacher_student_lstm_best_run_manifest.json").read_text(encoding="utf-8"))
            a3_manifest = json.loads((out_dir / "ablation3_joint_teacher_student_tcn_attn_best_run_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(a1_manifest["best_run_name"], "run_high")
            self.assertAlmostEqual(a1_manifest["best_test_acc"], 0.924)
            self.assertTrue(a1_manifest["acceptance"]["target_horizon_steps_ok"])
            self.assertTrue(a2_manifest["acceptance"]["target_accuracy_reached"])
            self.assertTrue(a3_manifest["acceptance"]["target_accuracy_reached"])


if __name__ == "__main__":
    unittest.main()
