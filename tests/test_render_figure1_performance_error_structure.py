import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "render_figure1_performance_error_structure.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class Figure1PerformanceErrorStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_module("render_figure1_performance_error_structure_test", SCRIPT_PATH)

    def test_parse_test_metrics_returns_balanced_accuracy_and_s1_recall(self) -> None:
        metrics_text = """
--- Test Metrics (demo) ---
Loss: 0.123
Accuracy: 95.51%
Macro-F1: 0.931234

=== Classification Report (Test) ===
              precision    recall  f1-score   support
     Class 0     1.0000   0.9000     0.9474        10
     Class 1     0.8000   0.7500     0.7742         8
     Class 2     0.9700   1.0000     0.9848        20

    accuracy                       0.9551        38
   macro avg     0.0000   0.0000     0.9312        38
"""

        parsed = self.mod.parse_test_metrics(metrics_text)

        self.assertAlmostEqual(parsed["accuracy"], 0.9551, places=6)
        self.assertAlmostEqual(parsed["macro_f1"], 0.931234, places=6)
        self.assertAlmostEqual(parsed["balanced_accuracy"], (0.9 + 0.75 + 1.0) / 3.0, places=6)
        self.assertAlmostEqual(parsed["s1_recall"], 0.75, places=6)
        self.assertEqual(len(parsed["class_rows"]), 3)

    def test_row_normalize_confusion_handles_empty_rows(self) -> None:
        y_true = np.array([0, 0, 2, 2], dtype=np.int64)
        y_pred = np.array([0, 2, 0, 2], dtype=np.int64)

        counts, normalized = self.mod.compute_confusion_matrices(y_true, y_pred, num_classes=3)

        np.testing.assert_array_equal(
            counts,
            np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.int64),
        )
        np.testing.assert_allclose(normalized[0], np.array([0.5, 0.0, 0.5]))
        np.testing.assert_allclose(normalized[1], np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(normalized[2], np.array([0.5, 0.0, 0.5]))

    def test_select_fixed_config_group_prefers_seed_count_then_mean_macro_f1(self) -> None:
        df = pd.DataFrame(
            {
                "config_id": ["a", "a", "b", "b", "b", "c"],
                "seed": [1, 2, 1, 2, 3, 1],
                "macro_f1": [0.99, 0.99, 0.80, 0.82, 0.84, 1.0],
            }
        )

        selected, summary = self.mod.select_representative_config_group(df, "config_id")

        self.assertEqual(selected["config_id"].unique().tolist(), ["b"])
        self.assertEqual(len(selected), 3)
        self.assertEqual(summary["selected_config_id"], "b")
        self.assertEqual(summary["n_seeds"], 3)

    def test_build_paired_delta_uses_only_overlapping_seeds(self) -> None:
        metrics = pd.DataFrame(
            {
                "method": [
                    "SEAL-Weld Student-13D",
                    "SEAL-Weld Student-13D",
                    "Student-only-13D",
                    "Student-only-13D",
                ],
                "seed": [1, 2, 2, 3],
                "macro_f1": [0.91, 0.92, 0.88, 0.87],
            }
        )

        deltas = self.mod.build_paired_student_delta(metrics)

        self.assertEqual(deltas["seed"].tolist(), [2])
        self.assertAlmostEqual(float(deltas["macro_f1_delta"].iloc[0]), 0.04, places=6)

    def test_teacher_retention_pct_uses_mean_macro_f1(self) -> None:
        metrics = pd.DataFrame(
            {
                "method": ["Teacher-18D", "Teacher-18D", "SEAL-Weld Student-13D"],
                "macro_f1": [0.98, 1.00, 0.94],
            }
        )

        retention = self.mod.compute_teacher_retention_pct(metrics)

        self.assertAlmostEqual(retention, 0.94 / 0.99 * 100.0, places=6)


if __name__ == "__main__":
    unittest.main()
