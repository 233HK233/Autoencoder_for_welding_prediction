import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "ablation_experiments/scripts/train_ablation1_student_only.py"
SWEEP_SCRIPT = PROJECT_ROOT / "ablation_experiments/scripts/run_ablation1_sweep.py"
H1_DATASET = PROJECT_ROOT / "Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz"
H1_RESULTS_ROOT = PROJECT_ROOT / "ablation_experiments/h1/results"
H1_REPORTS_ROOT = PROJECT_ROOT / "ablation_experiments/h1/reports"
H1_SWEEP_DIR_NAME = "ablation1_sweep_strict_valf1_20260522"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class Ablation1H1DefaultTests(unittest.TestCase):
    def test_train_script_defaults_to_h1_dataset(self) -> None:
        mod = load_module("train_ablation1_student_only_test", TRAIN_SCRIPT)

        with mock.patch.object(sys, "argv", ["train_ablation1_student_only.py"]):
            args = mod.parse_args()

        self.assertEqual(args.dataset_npz, H1_DATASET)
        self.assertEqual(args.output_dir, H1_RESULTS_ROOT / "ablation1_student_only_tcn_attn")
        self.assertEqual(args.student_latent_dim, 80)

    def test_train_script_accepts_student_latent_dim_override(self) -> None:
        mod = load_module("train_ablation1_student_only_latent_dim_test", TRAIN_SCRIPT)

        with mock.patch.object(sys, "argv", ["train_ablation1_student_only.py", "--student-latent-dim", "64"]):
            args = mod.parse_args()

        self.assertEqual(args.student_latent_dim, 64)

    def test_sweep_script_defaults_to_h1_dataset(self) -> None:
        mod = load_module("run_ablation1_sweep_test", SWEEP_SCRIPT)

        with mock.patch.object(sys, "argv", ["run_ablation1_sweep.py"]):
            args = mod.parse_args()

        self.assertEqual(args.dataset_npz, H1_DATASET)
        self.assertEqual(args.output_root, H1_RESULTS_ROOT / H1_SWEEP_DIR_NAME)
        self.assertEqual(args.report_dir, H1_REPORTS_ROOT / H1_SWEEP_DIR_NAME)

    def test_h1_dataset_exists(self) -> None:
        self.assertTrue(H1_DATASET.exists(), f"Missing dataset: {H1_DATASET}")


if __name__ == "__main__":
    unittest.main()
