import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "ablation_experiments/scripts/build_figure2_official_registry.py"


def write_metrics(
    run_dir: Path,
    *,
    checkpoint_metric: str,
    val_macro_f1: float,
    test_acc_pct: float,
    test_macro_f1: float,
    class_rows: list[tuple[str, float, float, float]],
    test_teacher_agreement_pct: float | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "Best epoch: 6",
        f"Checkpoint metric: {checkpoint_metric}",
        f"Best score: {val_macro_f1:.6f}",
        "",
        "--- Train Metrics ---",
        "Loss: 0.050000",
        "Accuracy: 99.00%",
        "Macro-F1: 0.990000",
        "",
        "=== Classification Report (Train) ===",
        "              precision    recall  f1-score   support",
        "     Class 0     1.0000   1.0000     1.0000        10",
        "     Class 1     1.0000   1.0000     1.0000        10",
        "     Class 2     1.0000   1.0000     1.0000        10",
        "",
        "--- Val Metrics ---",
        "Loss: 0.100000",
        "Accuracy: 90.00%",
        f"Macro-F1: {val_macro_f1:.6f}",
        "",
        "=== Classification Report (Val) ===",
        "              precision    recall  f1-score   support",
        "     Class 0     0.9000   0.9000     0.9000        10",
        "     Class 1     0.9000   0.9000     0.9000        10",
        "     Class 2     0.9000   0.9000     0.9000        10",
        "",
        "--- Test Metrics ---",
        "Loss: 0.200000",
        f"Accuracy: {test_acc_pct:.2f}%",
        f"Macro-F1: {test_macro_f1:.6f}",
        "",
        "=== Classification Report (Test) ===",
        "              precision    recall  f1-score   support",
    ]
    for label, precision, recall, f1 in class_rows:
        lines.append(
            f"{label:>12}     {precision:.4f}   {recall:.4f}     {f1:.4f}        10"
        )
    lines.extend(
        [
            "",
            "    accuracy                       0.9000        30",
            "   macro avg     0.0000   0.0000     0.9000        30",
        ]
    )
    if test_teacher_agreement_pct is not None:
        lines.insert(14, f"Test Teacher-Agreement: {test_teacher_agreement_pct:.2f}%")
    (run_dir / "evaluation_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


class Figure2OfficialRegistryTests(unittest.TestCase):
    def test_builds_registry_and_class_metrics_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            teacher_run = tmp_path / "teacher_run"
            write_metrics(
                teacher_run,
                checkpoint_metric="val_macro_f1",
                val_macro_f1=0.88,
                test_acc_pct=84.83,
                test_macro_f1=0.774468,
                class_rows=[
                    ("Class 0", 1.0, 0.5128, 0.6780),
                    ("Class 1", 0.6275, 0.7711, 0.6919),
                    ("Class 2", 0.9112, 1.0, 0.9535),
                ],
            )
            (teacher_run / "run_args.json").write_text(
                json.dumps({"checkpoint_metric": "val_macro_f1", "seed": 183}),
                encoding="utf-8",
            )
            teacher_manifest = tmp_path / "best_teacher_h1.json"
            teacher_manifest.write_text(
                json.dumps(
                    {
                        "best_run": {
                            "path": str(teacher_run),
                            "run_args_path": str(teacher_run / "run_args.json"),
                            "evaluation_metrics_path": str(teacher_run / "evaluation_metrics.txt"),
                        }
                    }
                ),
                encoding="utf-8",
            )

            student_run = tmp_path / "student_run"
            write_metrics(
                student_run,
                checkpoint_metric="val_macro_f1",
                val_macro_f1=0.93,
                test_acc_pct=92.10,
                test_macro_f1=0.901,
                class_rows=[
                    ("Class 0", 0.95, 0.91, 0.93),
                    ("Class 1", 0.90, 0.89, 0.895),
                    ("Class 2", 0.93, 0.96, 0.945),
                ],
            )
            (student_run / "run_args.json").write_text(
                json.dumps({"checkpoint_metric": "val_macro_f1", "seed": 14}),
                encoding="utf-8",
            )
            student_summary = tmp_path / "distill_sweep_summary.json"
            student_summary.write_text(
                json.dumps(
                    {
                        "best_run": {
                            "run_dir": str(student_run),
                            "checkpoint_metric": "val_macro_f1",
                            "val_macro_f1": 0.93,
                            "test_acc": 0.921,
                            "test_macro_f1": 0.901,
                        }
                    }
                ),
                encoding="utf-8",
            )

            ablation1_run = tmp_path / "ablation1_run"
            write_metrics(
                ablation1_run,
                checkpoint_metric="val_macro_f1",
                val_macro_f1=0.91,
                test_acc_pct=98.31,
                test_macro_f1=0.977795,
                class_rows=[
                    ("Class 0", 0.9625, 0.9872, 0.9747),
                    ("Class 1", 0.9639, 0.9639, 0.9639),
                    ("Class 2", 1.0, 0.9897, 0.9948),
                ],
            )
            (ablation1_run / "run_args.json").write_text(
                json.dumps({"checkpoint_metric": "val_macro_f1", "seed": 156}),
                encoding="utf-8",
            )
            ablation1_summary = tmp_path / "ablation1_sweep_runtime_summary.json"
            ablation1_summary.write_text(
                json.dumps(
                    {
                        "best_trial": {
                            "run_dir": str(ablation1_run),
                            "checkpoint_metric": "val_macro_f1",
                            "val_macro_f1": 0.91,
                            "test_acc": 0.9831,
                            "test_macro_f1": 0.977795,
                        }
                    }
                ),
                encoding="utf-8",
            )

            ablation2_run = tmp_path / "ablation2_run"
            write_metrics(
                ablation2_run,
                checkpoint_metric="val_macro_f1",
                val_macro_f1=0.89,
                test_acc_pct=99.44,
                test_macro_f1=0.992959,
                class_rows=[
                    ("Class 0", 0.9873, 1.0, 0.9936),
                    ("Class 1", 1.0, 0.9759, 0.9878),
                    ("Class 2", 0.9949, 1.0, 0.9974),
                ],
                test_teacher_agreement_pct=90.73,
            )
            (ablation2_run / "run_args.json").write_text(
                json.dumps({"checkpoint_metric": "val_macro_f1", "seed": 230}),
                encoding="utf-8",
            )
            ablation2_summary = tmp_path / "ablation2_sweep_summary.json"
            ablation2_summary.write_text(
                json.dumps(
                    {
                        "best_run": {
                            "run_dir": str(ablation2_run),
                            "checkpoint_metric": "val_macro_f1",
                            "val_macro_f1": 0.89,
                            "test_acc": 0.9944,
                            "test_macro_f1": 0.992959,
                            "test_teacher_agreement": 0.9073,
                        }
                    }
                ),
                encoding="utf-8",
            )

            baseline_run = tmp_path / "baseline_lstm"
            write_metrics(
                baseline_run,
                checkpoint_metric="val_macro_f1",
                val_macro_f1=0.972065,
                test_acc_pct=90.17,
                test_macro_f1=0.865073,
                class_rows=[
                    ("Class 0", 1.0, 0.6923, 0.8182),
                    ("Class 1", 0.75, 0.8675, 0.8045),
                    ("Class 2", 0.9466, 1.0, 0.9726),
                ],
            )
            (baseline_run / "run_args.json").write_text(
                json.dumps({"checkpoint_metric": "val_macro_f1", "seed": 14}),
                encoding="utf-8",
            )
            baseline_summary = tmp_path / "comparison_summary.csv"
            with baseline_summary.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "method_name",
                        "train_scheme",
                        "deploy_input_dim",
                        "teacher_input_dim",
                        "best_test_acc",
                        "best_macro_f1",
                        "teacher_agreement",
                        "seed",
                        "checkpoint_metric",
                        "run_path",
                        "notes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "method_name": "lstm",
                        "train_scheme": "baseline_18d",
                        "deploy_input_dim": 18,
                        "teacher_input_dim": "",
                        "best_test_acc": 0.9017,
                        "best_macro_f1": 0.865073,
                        "teacher_agreement": -1.0,
                        "seed": 14,
                        "checkpoint_metric": "val_macro_f1",
                        "run_path": str(baseline_run),
                        "notes": "Best 18D lstm baseline run",
                    }
                )

            out_dir = tmp_path / "out"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--teacher-manifest",
                    str(teacher_manifest),
                    "--student-summary-json",
                    str(student_summary),
                    "--ablation1-summary-json",
                    str(ablation1_summary),
                    "--ablation2-summary-json",
                    str(ablation2_summary),
                    "--comparison-summary-csv",
                    str(baseline_summary),
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)

            registry_path = out_dir / "official_run_registry.csv"
            metrics_path = out_dir / "class_metrics_summary.csv"
            self.assertTrue(registry_path.exists())
            self.assertTrue(metrics_path.exists())

            with registry_path.open(encoding="utf-8") as f:
                registry_rows = list(csv.DictReader(f))
            self.assertEqual(len(registry_rows), 5)
            self.assertTrue(all(row["protocol"] == "strict" for row in registry_rows))
            self.assertTrue(all(row["included_in_fig2"] == "yes" for row in registry_rows))
            self.assertEqual(
                {row["model_name"] for row in registry_rows},
                {"Strict Teacher Best", "Best Student Distill", "Best Ablation-1", "Best Ablation-2", "Best LSTM baseline"},
            )

            with metrics_path.open(encoding="utf-8") as f:
                metric_rows = list(csv.DictReader(f))
            self.assertEqual(len(metric_rows), 45)
            teacher_test_row = [
                row
                for row in metric_rows
                if row["model_name"] == "Strict Teacher Best"
                and row["class_name"] == "S0"
                and row["metric"] == "recall"
            ]
            self.assertEqual(len(teacher_test_row), 1)
            self.assertAlmostEqual(float(teacher_test_row[0]["value"]), 0.5128, places=6)
            student_f1 = [
                row
                for row in metric_rows
                if row["model_name"] == "Best Student Distill"
                and row["class_name"] == "S2"
                and row["metric"] == "f1"
            ]
            self.assertEqual(len(student_f1), 1)
            self.assertAlmostEqual(float(student_f1[0]["value"]), 0.945, places=6)


if __name__ == "__main__":
    unittest.main()
