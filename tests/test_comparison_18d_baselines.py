import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
H1_DATASET = PROJECT_ROOT / "Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz"
COMPARISON_RESULTS_ROOT = PROJECT_ROOT / "ablation_experiments/h1/results/comparison_18d_baselines"
COMPARISON_REPORTS_ROOT = PROJECT_ROOT / "ablation_experiments/h1/reports/comparison_18d_baselines"
TRAIN_SCRIPT = PROJECT_ROOT / "ablation_experiments/scripts/train_comparison_18d_baseline.py"
SWEEP_SCRIPT = PROJECT_ROOT / "ablation_experiments/scripts/run_comparison_18d_baseline_sweep.py"
DISTILL_SWEEP_SCRIPT = PROJECT_ROOT / "ablation_experiments/scripts/run_h1_distill_comparison_sweep.py"
SUMMARY_SCRIPT = PROJECT_ROOT / "ablation_experiments/scripts/summarize_comparison_18d_results.py"
TEACHER_MANIFEST = PROJECT_ROOT / "outputs/teacher_h1_scan98_gpu1_v2/manifests/best_teacher_h1.json"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def write_demo_forecast_npz(path: Path) -> None:
    rng = np.random.default_rng(7)
    x_train = rng.normal(size=(18, 5, 18)).astype(np.float32)
    y_train = np.array(([0] * 6) + ([1] * 6) + ([2] * 6), dtype=np.int64)
    x_test = rng.normal(size=(9, 5, 18)).astype(np.float32)
    y_test = np.array(([0] * 3) + ([1] * 3) + ([2] * 3), dtype=np.int64)

    np.savez(
        path,
        X_train_full=x_train,
        y_train=y_train,
        seam_id_train=np.zeros((18,), dtype=np.int64),
        start_idx_train=np.arange(18, dtype=np.int64),
        target_idx_train=np.arange(5, 23, dtype=np.int64),
        X_test_full=x_test,
        y_test=y_test,
        seam_id_test=np.ones((9,), dtype=np.int64),
        start_idx_test=np.arange(100, 109, dtype=np.int64),
        target_idx_test=np.arange(105, 114, dtype=np.int64),
        target_horizon_steps=np.array(1, dtype=np.int64),
        seam_name_order=np.array(["demo"], dtype="<U4"),
    )


def write_eval_metrics(
    path: Path,
    test_acc: float,
    macro_f1: float,
    agreement: float | None = None,
) -> None:
    lines = [
        "Best epoch: 3",
        "Checkpoint metric: val_macro_f1",
        "Best score: 0.900000",
        "",
        "--- Test Metrics ---",
        f"Accuracy: {test_acc * 100:.2f}%",
        f"Macro-F1: {macro_f1:.6f}",
    ]
    if agreement is not None:
        lines.append(f"Test Teacher-Agreement: {agreement * 100:.2f}%")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class Comparison18DBaselineDefaultTests(unittest.TestCase):
    def test_train_script_defaults_to_h1_dataset_and_result_root(self) -> None:
        mod = load_module("train_comparison_18d_baseline_test", TRAIN_SCRIPT)

        with mock.patch.object(sys, "argv", ["train_comparison_18d_baseline.py"]):
            args = mod.parse_args()

        self.assertEqual(args.dataset_npz, H1_DATASET)
        self.assertEqual(args.output_dir, COMPARISON_RESULTS_ROOT / "lstm")
        self.assertEqual(args.model, "lstm")

    def test_sweep_script_defaults_to_h1_roots(self) -> None:
        mod = load_module("run_comparison_18d_baseline_sweep_test", SWEEP_SCRIPT)

        with mock.patch.object(sys, "argv", ["run_comparison_18d_baseline_sweep.py"]):
            args = mod.parse_args()

        self.assertEqual(args.dataset_npz, H1_DATASET)
        self.assertEqual(args.output_root, COMPARISON_RESULTS_ROOT)
        self.assertEqual(args.report_root, COMPARISON_REPORTS_ROOT)

    def test_distill_sweep_defaults_to_h1_teacher_manifest_and_roots(self) -> None:
        mod = load_module("run_h1_distill_comparison_sweep_test", DISTILL_SWEEP_SCRIPT)

        with mock.patch.object(sys, "argv", ["run_h1_distill_comparison_sweep.py"]):
            args = mod.parse_args()

        self.assertEqual(args.dataset_npz, H1_DATASET)
        self.assertEqual(args.teacher_manifest, TEACHER_MANIFEST)
        self.assertEqual(
            args.output_dir,
            COMPARISON_RESULTS_ROOT / "teacher_student_reference/student_h1_sweep",
        )
        self.assertEqual(
            args.report_dir,
            COMPARISON_REPORTS_ROOT / "teacher_student_reference/student_h1_sweep",
        )

    def test_summary_defaults_to_comparison_roots(self) -> None:
        mod = load_module("summarize_comparison_18d_results_test", SUMMARY_SCRIPT)

        with mock.patch.object(sys, "argv", ["summarize_comparison_18d_results.py"]):
            args = mod.parse_args()

        self.assertEqual(args.results_root, COMPARISON_RESULTS_ROOT)
        self.assertEqual(args.report_dir, COMPARISON_REPORTS_ROOT)
        self.assertEqual(args.teacher_manifest, TEACHER_MANIFEST)


class Comparison18DBaselineSmokeTests(unittest.TestCase):
    def test_trainer_runs_for_all_four_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = tmp_path / "demo_h1.npz"
            write_demo_forecast_npz(dataset_path)

            for model in ("lstm", "gru", "transformer", "inception"):
                with self.subTest(model=model):
                    output_dir = tmp_path / "runs" / model
                    cmd = [
                        sys.executable,
                        str(TRAIN_SCRIPT),
                        "--dataset-npz",
                        str(dataset_path),
                        "--output-dir",
                        str(output_dir),
                        "--model",
                        model,
                        "--epochs",
                        "1",
                        "--batch-size",
                        "4",
                        "--val-ratio",
                        "0.2",
                        "--min-epochs",
                        "1",
                        "--early-stop-patience",
                        "1",
                        "--seed",
                        "11",
                    ]

                    result = subprocess.run(
                        cmd,
                        cwd=PROJECT_ROOT,
                        capture_output=True,
                        text=True,
                        check=False,
                    )

                    self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
                    run_dirs = [path for path in output_dir.iterdir() if path.is_dir()]
                    self.assertEqual(len(run_dirs), 1)
                    run_dir = run_dirs[0]
                    expected = {
                        "run_args.json",
                        "history.json",
                        "evaluation_metrics.txt",
                        "best_model.pth",
                    }
                    self.assertTrue(expected.issubset({path.name for path in run_dir.iterdir()}))


class ComparisonSummarySmokeTests(unittest.TestCase):
    def test_summary_generates_main_report_with_required_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results_root = tmp_path / "results"
            report_dir = tmp_path / "reports"

            teacher_run_dir = tmp_path / "teacher_run"
            teacher_run_dir.mkdir(parents=True)
            write_eval_metrics(teacher_run_dir / "evaluation_metrics.txt", test_acc=0.9831, macro_f1=0.979252)

            teacher_manifest = tmp_path / "best_teacher_h1.json"
            teacher_manifest.write_text(
                json.dumps(
                    {
                        "best_run": {
                            "path": str(teacher_run_dir),
                            "run_name": "teacher_best_seed14",
                            "run_args_path": str(teacher_run_dir / "run_args.json"),
                            "evaluation_metrics_path": str(teacher_run_dir / "evaluation_metrics.txt"),
                        }
                    }
                ),
                encoding="utf-8",
            )
            (teacher_run_dir / "run_args.json").write_text(
                json.dumps({"model": "tcn_attn", "seed": 14}),
                encoding="utf-8",
            )

            student_dir = results_root / "teacher_student_reference/student_h1_sweep" / "distill_seed42"
            student_dir.mkdir(parents=True)
            (student_dir / "run_args.json").write_text(
                json.dumps({"seed": 42, "checkpoint_metric": "val_teacher_agreement"}),
                encoding="utf-8",
            )
            write_eval_metrics(
                student_dir / "evaluation_metrics.txt",
                test_acc=0.9540,
                macro_f1=0.942000,
                agreement=0.9680,
            )

            baseline_metrics = {
                "lstm": (0.9010, 0.8801),
                "gru": (0.9170, 0.9012),
                "transformer": (0.9340, 0.9205),
                "inception": (0.9410, 0.9322),
            }
            for model, (acc, macro_f1) in baseline_metrics.items():
                run_dir = results_root / model / f"{model}_seed14"
                run_dir.mkdir(parents=True)
                (run_dir / "run_args.json").write_text(
                    json.dumps({"model": model, "seed": 14, "checkpoint_metric": "val_macro_f1"}),
                    encoding="utf-8",
                )
                write_eval_metrics(run_dir / "evaluation_metrics.txt", test_acc=acc, macro_f1=macro_f1)

            cmd = [
                sys.executable,
                str(SUMMARY_SCRIPT),
                "--results-root",
                str(results_root),
                "--report-dir",
                str(report_dir),
                "--teacher-manifest",
                str(teacher_manifest),
            ]
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            self.assertTrue((report_dir / "comparison_runs.csv").exists())
            self.assertTrue((report_dir / "comparison_summary.csv").exists())
            self.assertTrue((report_dir / "comparison_summary.json").exists())
            self.assertTrue((report_dir / "comparison_report.md").exists())

            payload = json.loads((report_dir / "comparison_summary.json").read_text(encoding="utf-8"))
            names = [row["method_name"] for row in payload["main_table"]]
            self.assertEqual(
                names,
                [
                    "teacher-student(student)",
                    "lstm",
                    "gru",
                    "transformer",
                    "inception",
                    "teacher(18D upper bound)",
                ],
            )
            student_row = payload["main_table"][0]
            self.assertEqual(student_row["deploy_input_dim"], 13)
            self.assertEqual(student_row["teacher_input_dim"], 18)


if __name__ == "__main__":
    unittest.main()
