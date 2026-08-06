import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "ablation_experiments/scripts/export_figure3_roc_predictions.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class ExportFigure3RocPredictionHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_module("export_figure3_roc_predictions_test", SCRIPT_PATH)

    def test_resolve_run_path_supports_absolute_and_relative_paths(self) -> None:
        root = Path("/tmp/demo-root")

        relative = self.mod.resolve_run_path(root, "ablation_experiments/h1/results/demo_run")
        absolute = self.mod.resolve_run_path(root, "/tmp/external/demo_run")

        self.assertEqual(relative, root / "ablation_experiments/h1/results/demo_run")
        self.assertEqual(absolute, Path("/tmp/external/demo_run"))

    def test_checkpoint_filename_selection_is_method_specific(self) -> None:
        expected = {
            "teacher-student(student)": "best_student_distill.pth",
            "teacher(18D upper bound)": "best_single_tcn.pth",
            "lstm": "best_model.pth",
            "gru": "best_model.pth",
            "transformer": "best_model.pth",
            "inception": "best_model.pth",
        }

        for method_name, filename in expected.items():
            with self.subTest(method_name=method_name):
                self.assertEqual(self.mod.checkpoint_filename_for_method(method_name), filename)

    def test_keep_feature_indices_fall_back_to_drop_feature_indices(self) -> None:
        keep = self.mod.resolve_keep_feature_indices(
            {"drop_feature_indices": "3,4,5,6,7"},
            input_dim=18,
        )

        self.assertEqual(keep, [0, 1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])

    def test_prediction_validation_rejects_non_normalized_probabilities(self) -> None:
        y_true = np.zeros((356,), dtype=np.int64)
        logits = np.zeros((356, 3), dtype=np.float32)
        probs = np.full((356, 3), 0.5, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            self.mod.validate_prediction_outputs(y_true=y_true, logits=logits, probabilities=probs)

    def test_prediction_validation_rejects_wrong_sample_count(self) -> None:
        y_true = np.zeros((10,), dtype=np.int64)
        logits = np.zeros((10, 3), dtype=np.float32)
        probs = np.full((10, 3), 1.0 / 3.0, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "356"):
            self.mod.validate_prediction_outputs(y_true=y_true, logits=logits, probabilities=probs)


if __name__ == "__main__":
    unittest.main()
