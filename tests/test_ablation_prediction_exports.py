import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABLATION1_SCRIPT = PROJECT_ROOT / "ablation_experiments/scripts/train_ablation1_student_only.py"
ABLATION2_SCRIPT = PROJECT_ROOT / "ablation_experiments/scripts/train_ablation2_distill_lstm_student.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class AblationPredictionExportTests(unittest.TestCase):
    def test_ablation1_reuses_required_test_prediction_schema(self) -> None:
        mod = load_module("train_ablation1_prediction_export", ABLATION1_SCRIPT)
        rows = mod.build_test_prediction_rows(
            ds={
                "y_test": np.array([1], dtype=np.int64),
                "seam_id_test": np.array([0], dtype=np.int64),
                "start_idx_test": np.array([20], dtype=np.int64),
                "target_idx_test": np.array([25], dtype=np.int64),
                "seam_name_order": np.array(["a01"]),
            },
            y_pred=np.array([1], dtype=np.int64),
            probabilities=np.array([[0.1, 0.8, 0.1]], dtype=np.float32),
        )

        self.assertEqual(rows[0]["seam_name"], "a01")
        self.assertEqual(rows[0]["prob_class_1"], 0.800000011920929)

    def test_ablation2_reuses_required_test_prediction_schema(self) -> None:
        mod = load_module("train_ablation2_prediction_export", ABLATION2_SCRIPT)
        rows = mod.build_test_prediction_rows(
            ds={
                "y_test": np.array([2], dtype=np.int64),
                "seam_id_test": np.array([2], dtype=np.int64),
                "start_idx_test": np.array([30], dtype=np.int64),
                "target_idx_test": np.array([35], dtype=np.int64),
                "seam_name_order": np.array(["a01", "b01", "c01"]),
            },
            y_pred=np.array([0], dtype=np.int64),
            probabilities=np.array([[0.7, 0.2, 0.1]], dtype=np.float32),
        )

        self.assertEqual(rows[0]["seam_name"], "c01")
        self.assertEqual(rows[0]["y_true"], 2)
        self.assertEqual(rows[0]["y_pred"], 0)


if __name__ == "__main__":
    unittest.main()
