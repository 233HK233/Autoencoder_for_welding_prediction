import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "build_figure4_temporal_case_results.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class Figure4TemporalCaseResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_module("build_figure4_temporal_case_results_test", SCRIPT_PATH)

    def test_build_full_seam_windows_uses_h1_target_delta(self) -> None:
        data = np.arange(16, dtype=np.float32).reshape(8, 2)
        labels = np.array([0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
        mean = np.array([1.0, 2.0], dtype=np.float32)
        scale = np.array([1.0, 2.0], dtype=np.float32)

        windows, starts, targets, y_true = self.mod.build_full_seam_windows(
            data=data,
            labels=labels,
            scaler_mean=mean,
            scaler_scale=scale,
            window_size=3,
            horizon=1,
        )

        self.assertEqual(windows.shape, (5, 3, 2))
        self.assertEqual(starts.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(targets.tolist(), [3, 4, 5, 6, 7])
        self.assertEqual(y_true.tolist(), [1, 1, 2, 2, 2])
        np.testing.assert_allclose(windows[0, 0], np.array([-1.0, -0.5], dtype=np.float32))

    def test_merge_method_predictions_rejects_misaligned_identity_columns(self) -> None:
        base = pd.DataFrame(
            {
                "sample_index": [0, 1],
                "seam_id": [0, 0],
                "seam_name": ["a01", "a01"],
                "start_idx": [10, 11],
                "target_idx": [15, 16],
                "y_true": [0, 1],
                "y_pred": [0, 1],
                "prob_class_0": [0.8, 0.1],
                "prob_class_1": [0.1, 0.8],
                "prob_class_2": [0.1, 0.1],
            }
        )
        shifted = base.copy()
        shifted.loc[1, "target_idx"] = 99

        with self.assertRaisesRegex(ValueError, "identity columns"):
            self.mod.merge_method_predictions(
                {
                    "Teacher-18D": base,
                    "SEAL-Weld Student-13D": shifted,
                }
            )

    def test_select_panel_rows_prefers_farthest_correct_core_and_nearest_successful_boundary(self) -> None:
        rows = [
            self._row(0, "a01", "S0-core", 0, 0, 0, 20, 0.50, False),
            self._row(1, "a01", "S0-core", 0, 0, 0, 35, 0.40, False),
            self._row(2, "a01", "S0-core", 0, 1, 1, 50, 0.95, False),
            self._row(3, "a01", "0->1-boundary", 0, 0, 0, 4, 0.55, False),
            self._row(4, "a01", "0->1-boundary", 0, 0, 0, 1, 0.45, False),
            self._row(5, "a01", "0->1-boundary", 0, 1, 1, 0, 0.99, False),
        ]
        selected = self.mod.select_panel_rows(
            pd.DataFrame(rows),
            seam_order=("a01",),
            case_order=("S0-core", "0->1-boundary"),
        )

        by_case = selected.set_index("case_type")
        self.assertEqual(int(by_case.loc["S0-core", "sample_index"]), 1)
        self.assertEqual(by_case.loc["S0-core", "selection_rule"], "core_farthest_correct")
        self.assertEqual(int(by_case.loc["0->1-boundary", "sample_index"]), 4)
        self.assertEqual(by_case.loc["0->1-boundary", "selection_rule"], "successful_boundary_nearest")

    def test_select_highlight_cases_returns_transfer_gain_and_failure_rows(self) -> None:
        rows = [
            self._row(0, "a01", "S1-core", 1, 1, 0, 7, 0.60, True),
            self._row(1, "b01", "1->2-boundary", 1, 2, 1, 1, 0.40, False),
            self._row(2, "c02", "0->1-boundary", 0, 1, 0, 3, 0.30, False),
        ]
        highlights = self.mod.select_highlight_cases(pd.DataFrame(rows))

        by_role = highlights.set_index("highlight_role")
        self.assertEqual(int(by_role.loc["transfer_gain", "sample_index"]), 0)
        self.assertEqual(int(by_role.loc["seal_weld_failure", "sample_index"]), 1)

    def test_reduce_attention_weights_returns_normalized_source_ribbon(self) -> None:
        weights = np.zeros((2, 3, 4, 4), dtype=np.float32)
        weights[:, :, :, 0] = 0.1
        weights[:, :, :, 1] = 0.2
        weights[:, :, :, 2] = 0.3
        weights[:, :, :, 3] = 0.4

        ribbon = self.mod.reduce_attention_weights(weights)

        self.assertEqual(ribbon.shape, (2, 4))
        np.testing.assert_allclose(ribbon.sum(axis=1), np.ones(2), atol=1e-6)
        np.testing.assert_allclose(ribbon[0], np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), atol=1e-6)

    def _row(
        self,
        sample_index: int,
        seam_name: str,
        case_type: str,
        y_true: int,
        seal_pred: int,
        student_only_pred: int,
        nearest_boundary_dist: int,
        seal_confidence: float,
        transfer_gain: bool,
    ) -> dict[str, object]:
        return {
            "sample_index": sample_index,
            "seam_name": seam_name,
            "case_type": case_type,
            "target_idx": 100 + sample_index,
            "start_idx": 95 + sample_index,
            "y_true": y_true,
            "teacher_pred": y_true,
            "seal_student_pred": seal_pred,
            "student_only_pred": student_only_pred,
            "teacher_correct": True,
            "seal_student_correct": seal_pred == y_true,
            "teacher_student_both_correct": seal_pred == y_true,
            "student_only_correct": student_only_pred == y_true,
            "transfer_gain": transfer_gain,
            "seal_weld_failure": seal_pred != y_true,
            "nearest_boundary_dist": nearest_boundary_dist,
            "boundary_distance_bin": "Near" if nearest_boundary_dist <= 5 else "Far",
            "seal_student_confidence": seal_confidence,
        }


if __name__ == "__main__":
    unittest.main()
