import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class SelectFigure1PanelsTests(unittest.TestCase):
    def test_annotate_samples_adds_boundary_distance_and_case_type(self) -> None:
        import select_figure1_panels as mod

        label_sequence = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
        df = pd.DataFrame(
            [
                {
                    "sample_index": 0,
                    "seam_id": 0,
                    "seam_name": "a01",
                    "start_idx": 0,
                    "target_idx": 0,
                    "y_true": 0,
                    "raw_pred": 0,
                    "final_pred": 0,
                    "decode_mode": "raw",
                    "prob_class_0": 0.90,
                    "prob_class_1": 0.05,
                    "prob_class_2": 0.05,
                },
                {
                    "sample_index": 1,
                    "seam_id": 0,
                    "seam_name": "a01",
                    "start_idx": 0,
                    "target_idx": 2,
                    "y_true": 0,
                    "raw_pred": 0,
                    "final_pred": 0,
                    "decode_mode": "raw",
                    "prob_class_0": 0.88,
                    "prob_class_1": 0.08,
                    "prob_class_2": 0.04,
                },
                {
                    "sample_index": 2,
                    "seam_id": 0,
                    "seam_name": "a01",
                    "start_idx": 0,
                    "target_idx": 3,
                    "y_true": 1,
                    "raw_pred": 1,
                    "final_pred": 1,
                    "decode_mode": "raw",
                    "prob_class_0": 0.10,
                    "prob_class_1": 0.80,
                    "prob_class_2": 0.10,
                },
                {
                    "sample_index": 3,
                    "seam_id": 0,
                    "seam_name": "a01",
                    "start_idx": 0,
                    "target_idx": 6,
                    "y_true": 2,
                    "raw_pred": 1,
                    "final_pred": 1,
                    "decode_mode": "raw",
                    "prob_class_0": 0.05,
                    "prob_class_1": 0.70,
                    "prob_class_2": 0.25,
                },
            ]
        )

        annotated = mod.annotate_samples(df, {"a01": label_sequence})

        self.assertIn("nearest_boundary_dist", annotated.columns)
        self.assertIn("boundary_distance_bin", annotated.columns)
        self.assertIn("case_type", annotated.columns)
        self.assertIn("error_type", annotated.columns)
        self.assertIn("confidence", annotated.columns)

        row0 = annotated.loc[annotated["sample_index"] == 0].iloc[0]
        row1 = annotated.loc[annotated["sample_index"] == 1].iloc[0]
        row2 = annotated.loc[annotated["sample_index"] == 2].iloc[0]
        row3 = annotated.loc[annotated["sample_index"] == 3].iloc[0]

        self.assertEqual(int(row0["nearest_boundary_dist"]), 3)
        self.assertEqual(row0["boundary_distance_bin"], "Near")
        self.assertEqual(row0["case_type"], "0->1-boundary")
        self.assertEqual(row0["error_type"], "correct")

        self.assertEqual(int(row1["nearest_boundary_dist"]), 1)
        self.assertEqual(row1["case_type"], "0->1-boundary")

        self.assertEqual(int(row2["nearest_boundary_dist"]), 0)
        self.assertEqual(row2["case_type"], "S1-core")

        self.assertEqual(int(row3["nearest_boundary_dist"]), 0)
        self.assertEqual(row3["case_type"], "1->2-boundary")
        self.assertEqual(row3["error_type"], "adjacent_error")
        self.assertAlmostEqual(float(row3["confidence"]), 0.70, places=6)

    def test_select_panel_candidates_returns_one_row_per_seam_case_type(self) -> None:
        import select_figure1_panels as mod

        rows = []
        sample_index = 0
        for seam_name in ("a01", "b01", "c01", "c02"):
            for case_type, y_true in (
                ("S0-core", 0),
                ("0->1-boundary", 0),
                ("S1-core", 1),
                ("1->2-boundary", 2),
            ):
                rows.append(
                    {
                        "sample_index": sample_index,
                        "seam_id": 0,
                        "seam_name": seam_name,
                        "start_idx": 100 + sample_index,
                        "target_idx": 200 + sample_index,
                        "y_true": y_true,
                        "raw_pred": y_true,
                        "final_pred": y_true,
                        "decode_mode": "raw",
                        "prob_class_0": 0.9 if y_true == 0 else 0.05,
                        "prob_class_1": 0.9 if y_true == 1 else 0.05,
                        "prob_class_2": 0.9 if y_true == 2 else 0.05,
                        "nearest_boundary_dist": 20 if case_type in {"S0-core", "S1-core"} else 0,
                        "boundary_distance_bin": "Far" if case_type in {"S0-core", "S1-core"} else "Near",
                        "case_type": case_type,
                        "error_type": "correct",
                        "confidence": 0.9,
                    }
                )
                sample_index += 1

        df = pd.DataFrame(rows)
        selected = mod.select_panel_candidates(df)

        self.assertEqual(len(selected), 16)
        self.assertEqual(
            set(selected["panel_key"]),
            {
                f"{seam}|{case_type}"
                for seam in ("a01", "b01", "c01", "c02")
                for case_type in ("S0-core", "0->1-boundary", "S1-core", "1->2-boundary")
            },
        )
        self.assertTrue((selected["panel_status"] == "selected").all())

    def test_main_writes_enriched_and_panel_csv(self) -> None:
        import select_figure1_panels as mod

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pred_csv = tmp_path / "pred.csv"
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "sample_index": 0,
                        "seam_id": 0,
                        "seam_name": "a01",
                        "start_idx": 0,
                        "target_idx": 0,
                        "y_true": 0,
                        "raw_pred": 0,
                        "final_pred": 0,
                        "decode_mode": "raw",
                        "prob_class_0": 0.90,
                        "prob_class_1": 0.05,
                        "prob_class_2": 0.05,
                    }
                ]
            ).to_csv(pred_csv, index=False)

            seam_df = pd.DataFrame(
                [
                    [0.0] * 18 + ["quasistable"],
                    [0.0] * 18 + ["quasistable"],
                    [0.0] * 18 + ["quasistable"],
                    [0.0] * 18 + ["nonstationary"],
                    [0.0] * 18 + ["nonstationary"],
                    [0.0] * 18 + ["nonstationary"],
                    [0.0] * 18 + ["instability"],
                    [0.0] * 18 + ["instability"],
                    [0.0] * 18 + ["instability"],
                ]
            )
            seam_df.to_csv(raw_dir / "a01.csv", header=False, index=False)

            enriched_csv = tmp_path / "enriched.csv"
            panel_csv = tmp_path / "panels.csv"

            mod.main(
                [
                    "--predictions-csv",
                    str(pred_csv),
                    "--raw-data-dir",
                    str(raw_dir),
                    "--output-enriched-csv",
                    str(enriched_csv),
                    "--output-panel-csv",
                    str(panel_csv),
                ]
            )

            self.assertTrue(enriched_csv.exists())
            self.assertTrue(panel_csv.exists())
            panel_df = pd.read_csv(panel_csv)
            self.assertFalse((panel_df["panel_status"] == "missing").any())

    def test_cross_seam_backfill_fills_missing_slot(self) -> None:
        import select_figure1_panels as mod

        df = pd.DataFrame(
            [
                {
                    "sample_index": 1,
                    "seam_id": 0,
                    "seam_name": "a01",
                    "start_idx": 10,
                    "target_idx": 10,
                    "y_true": 0,
                    "raw_pred": 0,
                    "final_pred": 0,
                    "decode_mode": "raw",
                    "prob_class_0": 0.9,
                    "prob_class_1": 0.05,
                    "prob_class_2": 0.05,
                    "nearest_boundary_dist": 20,
                    "boundary_distance_bin": "Far",
                    "case_type": "S0-core",
                    "error_type": "correct",
                    "confidence": 0.9,
                    "panel_key": "a01|S0-core",
                    "panel_status": "selected",
                },
                {
                    "sample_index": None,
                    "seam_id": None,
                    "seam_name": "c01",
                    "start_idx": None,
                    "target_idx": None,
                    "y_true": None,
                    "raw_pred": None,
                    "final_pred": None,
                    "decode_mode": None,
                    "prob_class_0": None,
                    "prob_class_1": None,
                    "prob_class_2": None,
                    "nearest_boundary_dist": None,
                    "boundary_distance_bin": None,
                    "case_type": "S0-core",
                    "error_type": None,
                    "confidence": None,
                    "panel_key": "c01|S0-core",
                    "panel_status": "missing",
                },
                {
                    "sample_index": 2,
                    "seam_id": 1,
                    "seam_name": "b01",
                    "start_idx": 20,
                    "target_idx": 20,
                    "y_true": 0,
                    "raw_pred": 0,
                    "final_pred": 0,
                    "decode_mode": "raw",
                    "prob_class_0": 0.8,
                    "prob_class_1": 0.1,
                    "prob_class_2": 0.1,
                    "nearest_boundary_dist": 18,
                    "boundary_distance_bin": "Far",
                    "case_type": "S0-core",
                    "error_type": "correct",
                    "confidence": 0.8,
                    "panel_key": "b01|S0-core",
                    "panel_status": "selected",
                },
            ]
        )

        filled = mod.apply_cross_seam_backfill(df)
        row = filled.loc[filled["panel_key"] == "c01|S0-core"].iloc[0]

        self.assertEqual(row["panel_status"], "backfilled")
        self.assertEqual(row["source_seam_name"], "a01")
        self.assertEqual(int(row["sample_index"]), 1)


if __name__ == "__main__":
    unittest.main()
