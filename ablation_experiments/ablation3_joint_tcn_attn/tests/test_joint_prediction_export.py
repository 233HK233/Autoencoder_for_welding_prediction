import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "ablation_experiments/ablation3_joint_tcn_attn/scripts/train_ablation3_joint_tcn_attn.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class JointPredictionExportTests(unittest.TestCase):
    def test_joint_trainer_uses_required_test_prediction_schema(self) -> None:
        mod = load_module("train_ablation3_joint_prediction_export", SCRIPT_PATH)

        ds = {
            "y_test": np.array([0], dtype=np.int64),
            "seam_id_test": np.array([1], dtype=np.int64),
            "start_idx_test": np.array([11], dtype=np.int64),
            "target_idx_test": np.array([16], dtype=np.int64),
            "seam_name_order": np.array(["a01", "b01"]),
        }
        rows = mod.build_test_prediction_rows(
            ds=ds,
            y_pred=np.array([2], dtype=np.int64),
            probabilities=np.array([[0.2, 0.3, 0.5]], dtype=np.float32),
        )

        self.assertEqual(rows[0]["seam_name"], "b01")
        self.assertEqual(rows[0]["target_idx"], 16)
        self.assertAlmostEqual(rows[0]["prob_class_2"], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
