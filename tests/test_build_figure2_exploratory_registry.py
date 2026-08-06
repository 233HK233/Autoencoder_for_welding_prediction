import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT / "ablation_experiments/scripts/build_figure2_exploratory_registry.py"
)


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
    (run_dir / "evaluation_metrics.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


class Figure2ExploratoryRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

        self.teacher_run = self.tmp_path / "teacher_run"
        write_metrics(
            self.teacher_run,
            checkpoint_metric="test_acc",
            val_macro_f1=0.88,
            test_acc_pct=98.31,
            test_macro_f1=0.979252,
            class_rows=[
                ("Class 0", 0.9625, 0.9872, 0.9747),
                ("Class 1", 0.9639, 0.9639, 0.9639),
                ("Class 2", 1.0, 0.9897, 0.9948),
            ],
        )
        (self.teacher_run / "run_args.json").write_text(
            json.dumps({"checkpoint_metric": "test_acc", "seed": 14}),
            encoding="utf-8",
        )
        self.teacher_manifest = self.tmp_path / "best_teacher_h1.json"
        self.teacher_manifest.write_text(
            json.dumps(
                {
                    "best_run": {
                        "path": str(self.teacher_run),
                        "run_args_path": str(self.teacher_run / "run_args.json"),
                        "evaluation_metrics_path": str(
                            self.teacher_run / "evaluation_metrics.txt"
                        ),
                        "metrics_core": {"checkpoint_metric": "test_acc"},
                    }
                }
            ),
            encoding="utf-8",
        )

        self.student_run = self.tmp_path / "student_run"
        write_metrics(
            self.student_run,
            checkpoint_metric="val_teacher_agreement",
            val_macro_f1=0.98,
            test_acc_pct=98.88,
            test_macro_f1=0.984675,
            class_rows=[
                ("Class 0", 0.98, 0.97, 0.975),
                ("Class 1", 0.98, 0.98, 0.98),
                ("Class 2", 0.994, 1.0, 0.997),
            ],
        )
        (self.student_run / "run_args.json").write_text(
            json.dumps({"checkpoint_metric": "val_teacher_agreement", "seed": 132}),
            encoding="utf-8",
        )

        self.ablation1_run = self.tmp_path / "ablation1_run"
        write_metrics(
            self.ablation1_run,
            checkpoint_metric="test_acc",
            val_macro_f1=0.91,
            test_acc_pct=98.31,
            test_macro_f1=0.977795,
            class_rows=[
                ("Class 0", 0.9625, 0.9872, 0.9747),
                ("Class 1", 0.9639, 0.9639, 0.9639),
                ("Class 2", 1.0, 0.9897, 0.9948),
            ],
        )
        (self.ablation1_run / "run_args.json").write_text(
            json.dumps({"checkpoint_metric": "test_acc", "seed": 156}),
            encoding="utf-8",
        )

        self.ablation2_run = self.tmp_path / "ablation2_run"
        write_metrics(
            self.ablation2_run,
            checkpoint_metric="test_acc",
            val_macro_f1=0.89,
            test_acc_pct=99.44,
            test_macro_f1=0.992959,
            class_rows=[
                ("Class 0", 0.9873, 1.0, 0.9936),
                ("Class 1", 1.0, 0.9759, 0.9878),
                ("Class 2", 0.9949, 1.0, 0.9974),
            ],
            test_teacher_agreement_pct=97.19,
        )
        (self.ablation2_run / "run_args.json").write_text(
            json.dumps({"checkpoint_metric": "test_acc", "seed": 230}),
            encoding="utf-8",
        )

        self.gru_run = self.tmp_path / "baseline_gru"
        write_metrics(
            self.gru_run,
            checkpoint_metric="val_macro_f1",
            val_macro_f1=0.89,
            test_acc_pct=91.57,
            test_macro_f1=0.887156,
            class_rows=[
                ("Class 0", 0.88, 0.84, 0.8595),
                ("Class 1", 0.86, 0.9, 0.8795),
                ("Class 2", 0.93, 0.98, 0.9543),
            ],
        )
        (self.gru_run / "run_args.json").write_text(
            json.dumps({"checkpoint_metric": "val_macro_f1", "seed": 14}),
            encoding="utf-8",
        )

        self.lstm_run = self.tmp_path / "baseline_lstm"
        write_metrics(
            self.lstm_run,
            checkpoint_metric="val_macro_f1",
            val_macro_f1=0.88,
            test_acc_pct=90.73,
            test_macro_f1=0.875341,
            class_rows=[
                ("Class 0", 0.87, 0.82, 0.8443),
                ("Class 1", 0.84, 0.88, 0.8595),
                ("Class 2", 0.94, 0.97, 0.9548),
            ],
        )
        (self.lstm_run / "run_args.json").write_text(
            json.dumps({"checkpoint_metric": "val_macro_f1", "seed": 42}),
            encoding="utf-8",
        )

        self.comparison_summary = self.tmp_path / "comparison_summary.csv"
        with self.comparison_summary.open("w", newline="", encoding="utf-8") as f:
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
                    "method_name": "teacher-student(student)",
                    "train_scheme": "frozen_teacher_distill",
                    "deploy_input_dim": 13,
                    "teacher_input_dim": 18,
                    "best_test_acc": 0.9888,
                    "best_macro_f1": 0.984675,
                    "teacher_agreement": 0.9719,
                    "seed": 132,
                    "checkpoint_metric": "val_teacher_agreement",
                    "run_path": str(self.student_run),
                    "notes": "13D deploy student distilled from 18D teacher",
                }
            )
            writer.writerow(
                {
                    "method_name": "gru",
                    "train_scheme": "baseline_18d",
                    "deploy_input_dim": 18,
                    "teacher_input_dim": "",
                    "best_test_acc": 0.9157,
                    "best_macro_f1": 0.887156,
                    "teacher_agreement": -1.0,
                    "seed": 14,
                    "checkpoint_metric": "val_macro_f1",
                    "run_path": str(self.gru_run),
                    "notes": "Best 18D gru baseline run",
                }
            )
            writer.writerow(
                {
                    "method_name": "lstm",
                    "train_scheme": "baseline_18d",
                    "deploy_input_dim": 18,
                    "teacher_input_dim": "",
                    "best_test_acc": 0.9073,
                    "best_macro_f1": 0.875341,
                    "teacher_agreement": -1.0,
                    "seed": 42,
                    "checkpoint_metric": "val_macro_f1",
                    "run_path": str(self.lstm_run),
                    "notes": "Best 18D lstm baseline run",
                }
            )

        self.ablation_summary = self.tmp_path / "ablation_summary.csv"
        with self.ablation_summary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "group",
                    "count",
                    "mean_test_acc",
                    "std_test_acc",
                    "mean_test_macro_f1",
                    "std_test_macro_f1",
                    "mean_test_teacher_agreement",
                    "std_test_teacher_agreement",
                    "best_run_name",
                    "best_run_dir",
                    "best_seed",
                    "best_test_acc",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "group": "ablation1_student_only_tcn_attn",
                    "count": 60,
                    "mean_test_acc": 0.89836,
                    "std_test_acc": 0.04126,
                    "mean_test_macro_f1": 0.870058,
                    "std_test_macro_f1": 0.053259,
                    "mean_test_teacher_agreement": -1.0,
                    "std_test_teacher_agreement": 0.0,
                    "best_run_name": self.ablation1_run.name,
                    "best_run_dir": str(self.ablation1_run),
                    "best_seed": 156,
                    "best_test_acc": 0.9831,
                }
            )
            writer.writerow(
                {
                    "group": "ablation2_teacher_student_lstm",
                    "count": 56,
                    "mean_test_acc": 0.901889,
                    "std_test_acc": 0.025913,
                    "mean_test_macro_f1": 0.868654,
                    "std_test_macro_f1": 0.036342,
                    "mean_test_teacher_agreement": 0.949639,
                    "std_test_teacher_agreement": 0.033351,
                    "best_run_name": self.ablation2_run.name,
                    "best_run_dir": str(self.ablation2_run),
                    "best_seed": 230,
                    "best_test_acc": 0.9944,
                }
            )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def run_script(self, *extra_args: str) -> Path:
        out_dir = self.tmp_path / "out"
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--teacher-manifest",
                str(self.teacher_manifest),
                "--comparison-summary-csv",
                str(self.comparison_summary),
                "--ablation-summary-csv",
                str(self.ablation_summary),
                "--out-dir",
                str(out_dir),
                *extra_args,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
        return out_dir

    def test_builds_registry_and_metrics_with_default_gru_baseline(self) -> None:
        out_dir = self.run_script()

        registry_path = out_dir / "exploratory_run_registry.csv"
        metrics_path = out_dir / "exploratory_class_metrics_summary.csv"
        self.assertTrue(registry_path.exists())
        self.assertTrue(metrics_path.exists())

        with registry_path.open(encoding="utf-8") as f:
            registry_rows = list(csv.DictReader(f))
        self.assertEqual(len(registry_rows), 5)
        self.assertTrue(all(row["protocol"] == "exploratory" for row in registry_rows))
        self.assertEqual(
            {row["model_name"] for row in registry_rows},
            {
                "Exploratory Teacher Best",
                "Best Student Distill",
                "Best Ablation-1",
                "Best Ablation-2",
                "Best GRU baseline",
            },
        )

        with metrics_path.open(encoding="utf-8") as f:
            metric_rows = list(csv.DictReader(f))
        self.assertEqual(len(metric_rows), 45)
        gru_f1 = [
            row
            for row in metric_rows
            if row["model_name"] == "Best GRU baseline"
            and row["class_name"] == "S2"
            and row["metric"] == "f1"
        ]
        self.assertEqual(len(gru_f1), 1)
        self.assertAlmostEqual(float(gru_f1[0]["value"]), 0.9543, places=6)

    def test_supports_lstm_baseline_override(self) -> None:
        out_dir = self.run_script("--baseline-method", "lstm")

        registry_path = out_dir / "exploratory_run_registry.csv"
        with registry_path.open(encoding="utf-8") as f:
            registry_rows = list(csv.DictReader(f))

        baseline_rows = [
            row for row in registry_rows if row["model_family"] == "baseline"
        ]
        self.assertEqual(len(baseline_rows), 1)
        self.assertEqual(baseline_rows[0]["model_name"], "Best LSTM baseline")
        self.assertEqual(baseline_rows[0]["checkpoint_metric"], "val_macro_f1")


if __name__ == "__main__":
    unittest.main()
