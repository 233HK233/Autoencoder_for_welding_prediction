import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
H1_DATASET = PROJECT_ROOT / "Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz"
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


class AblationH1DefaultsTests(unittest.TestCase):
    def test_ablation2_teacher_defaults_to_h1_dataset_and_output(self) -> None:
        mod = load_module(
            "train_ablation2_teacher_lstm_test",
            PROJECT_ROOT / "ablation_experiments/scripts/train_ablation2_teacher_lstm.py",
        )

        with mock.patch.object(sys, "argv", ["train_ablation2_teacher_lstm.py"]):
            args = mod.parse_args()

        self.assertEqual(args.dataset_npz, H1_DATASET)
        self.assertEqual(
            args.output_dir,
            H1_RESULTS_ROOT / "ablation2_teacher_student_lstm/teachers",
        )

    def test_ablation2_student_defaults_to_h1_dataset_and_output(self) -> None:
        mod = load_module(
            "train_ablation2_distill_lstm_student_test",
            PROJECT_ROOT / "ablation_experiments/scripts/train_ablation2_distill_lstm_student.py",
        )

        with mock.patch.object(
            sys,
            "argv",
            [
                "train_ablation2_distill_lstm_student.py",
                "--teacher-ckpt",
                "teacher.pth",
                "--teacher-run-args",
                "teacher.json",
            ],
        ):
            args = mod.parse_args()

        self.assertEqual(args.dataset_npz, H1_DATASET)
        self.assertEqual(
            args.output_dir,
            H1_RESULTS_ROOT / "ablation2_teacher_student_lstm/students",
        )

    def test_ablation3_train_defaults_to_h1_dataset_and_output(self) -> None:
        mod = load_module(
            "train_ablation3_joint_tcn_attn_test",
            PROJECT_ROOT
            / "ablation_experiments/ablation3_joint_tcn_attn/scripts/train_ablation3_joint_tcn_attn.py",
        )

        with mock.patch.object(sys, "argv", ["train_ablation3_joint_tcn_attn.py"]):
            args = mod.parse_args()

        self.assertEqual(args.dataset_npz, H1_DATASET)
        self.assertEqual(
            args.output_dir,
            H1_RESULTS_ROOT / "ablation3_joint_tcn_attn",
        )

    def test_ablation3_sweep_defaults_to_h1_dataset_output_and_report(self) -> None:
        mod = load_module(
            "run_ablation3_seed_sweep_test",
            PROJECT_ROOT
            / "ablation_experiments/ablation3_joint_tcn_attn/scripts/run_ablation3_seed_sweep.py",
        )

        with mock.patch.object(sys, "argv", ["run_ablation3_seed_sweep.py"]):
            args = mod.parse_args()

        self.assertEqual(args.dataset_npz, H1_DATASET)
        self.assertEqual(args.output_dir, H1_RESULTS_ROOT / "ablation3_joint_tcn_attn")
        self.assertEqual(args.report_dir, H1_REPORTS_ROOT / "ablation3_joint_tcn_attn")

    def test_ablation_suite_defaults_to_h1_dataset_and_reports(self) -> None:
        mod = load_module(
            "run_ablation_suite_test",
            PROJECT_ROOT / "ablation_experiments/scripts/run_ablation_suite.py",
        )

        with mock.patch.object(sys, "argv", ["run_ablation_suite.py"]):
            args = mod.parse_args()

        self.assertEqual(args.dataset_npz, H1_DATASET)
        self.assertEqual(args.output_root, H1_RESULTS_ROOT)
        self.assertEqual(args.report_dir, H1_REPORTS_ROOT)

    def test_summary_defaults_to_h1_result_and_report_roots(self) -> None:
        mod = load_module(
            "summarize_ablation_results_test",
            PROJECT_ROOT / "ablation_experiments/scripts/summarize_ablation_results.py",
        )

        with mock.patch.object(sys, "argv", ["summarize_ablation_results.py"]):
            args = mod.parse_args()

        self.assertEqual(
            args.ablation1_dir,
            H1_RESULTS_ROOT / "ablation1_student_only_tcn_attn",
        )
        self.assertEqual(
            args.ablation2_dir,
            H1_RESULTS_ROOT / "ablation2_teacher_student_lstm/students",
        )
        self.assertEqual(
            args.ablation3_dir,
            H1_RESULTS_ROOT / "ablation3_joint_tcn_attn",
        )
        self.assertEqual(args.out_dir, H1_REPORTS_ROOT)


class ForecastDatasetContractTests(unittest.TestCase):
    def test_validate_forecast_dataset_contract_accepts_valid_h1_dataset(self) -> None:
        import training_utils as mod

        result = mod.validate_forecast_dataset_contract(H1_DATASET)

        self.assertEqual(result["target_horizon_steps"], 1)
        self.assertTrue(result["horizon_ok"])
        self.assertTrue(result["train_delta_ok"])
        self.assertTrue(result["test_delta_ok"])

    def test_validate_forecast_dataset_contract_rejects_wrong_horizon(self) -> None:
        import training_utils as mod

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid_h1.npz"
            np.savez(
                path,
                target_horizon_steps=np.array(0, dtype=np.int64),
                start_idx_train=np.array([0, 1], dtype=np.int64),
                target_idx_train=np.array([5, 6], dtype=np.int64),
                start_idx_test=np.array([10], dtype=np.int64),
                target_idx_test=np.array([15], dtype=np.int64),
            )

            with self.assertRaisesRegex(ValueError, "target_horizon_steps=0"):
                mod.validate_forecast_dataset_contract(path)

    def test_validate_forecast_dataset_contract_rejects_wrong_target_delta(self) -> None:
        import training_utils as mod

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid_delta.npz"
            np.savez(
                path,
                target_horizon_steps=np.array(1, dtype=np.int64),
                start_idx_train=np.array([0, 1], dtype=np.int64),
                target_idx_train=np.array([4, 6], dtype=np.int64),
                start_idx_test=np.array([10], dtype=np.int64),
                target_idx_test=np.array([15], dtype=np.int64),
            )

            with self.assertRaisesRegex(ValueError, "target_idx-start_idx"):
                mod.validate_forecast_dataset_contract(path)


if __name__ == "__main__":
    unittest.main()
