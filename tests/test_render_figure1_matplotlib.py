import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class RenderFigure1MatplotlibTests(unittest.TestCase):
    def test_build_relative_time_axis_uses_target_idx_as_zero(self) -> None:
        import render_figure1_matplotlib as mod

        xs = np.array([95, 96, 97, 98, 99, 100, 101], dtype=np.int64)
        axis = mod.build_relative_time_axis(xs, target_idx=100, sample_period_s=0.01)

        self.assertTrue(np.allclose(axis, np.array([-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01])))

    def test_channel_labels_are_not_generic(self) -> None:
        import render_figure1_matplotlib as mod

        labels = mod.make_channel_labels([3, 7, 11])

        self.assertEqual(
            labels,
            [
                "1st Principal High-Var Channel",
                "2nd Principal High-Var Channel",
                "3rd Principal High-Var Channel",
            ],
        )

    def test_build_background_segments_returns_relative_time_bounds(self) -> None:
        import render_figure1_matplotlib as mod

        labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        segments = mod.build_background_segments(labels, x0=0, x1=6, target_idx=3, sample_period_s=0.01)

        self.assertEqual(len(segments), 3)
        self.assertTrue(np.allclose(segments[0][1:], (-0.03, -0.01)))
        self.assertTrue(np.allclose(segments[1][1:], (-0.01, 0.01)))
        self.assertTrue(np.allclose(segments[2][1:], (0.01, 0.02)))

    def test_build_local_state_trajectory_returns_gt_and_pred_steps(self) -> None:
        import render_figure1_matplotlib as mod

        local_df = pd.DataFrame(
            [
                {"target_idx": 98, "y_true": 0, "final_pred": 0},
                {"target_idx": 99, "y_true": 0, "final_pred": 1},
                {"target_idx": 100, "y_true": 1, "final_pred": 1},
                {"target_idx": 101, "y_true": 1, "final_pred": 1},
            ]
        )

        xs, gt, pred = mod.build_local_state_trajectory(local_df, target_idx=100, sample_period_s=0.01)

        self.assertTrue(np.allclose(xs, np.array([-0.02, -0.01, 0.0, 0.01])))
        self.assertTrue(np.array_equal(gt, np.array([0, 0, 1, 1])))
        self.assertTrue(np.array_equal(pred, np.array([0, 1, 1, 1])))

    def test_main_writes_pdf_and_png(self) -> None:
        import render_figure1_matplotlib as mod

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            rows = []
            labels = (
                ["quasistable"] * 20
                + ["nonstationary"] * 20
                + ["instability"] * 20
            )
            for idx, label in enumerate(labels):
                rows.append([float(idx + j) for j in range(18)] + [label])

            for seam_name in ("a01", "b01", "c01", "c02"):
                pd.DataFrame(rows).to_csv(raw_dir / f"{seam_name}.csv", header=False, index=False)

            panel_rows = []
            sample_index = 0
            for seam_name in ("a01", "b01", "c01", "c02"):
                for case_type, target_idx, y_true in (
                    ("S0-core", 10, 0),
                    ("0->1-boundary", 19, 0),
                    ("S1-core", 30, 1),
                    ("1->2-boundary", 39, 1),
                ):
                    panel_rows.append(
                        {
                            "sample_index": sample_index,
                            "seam_id": 0,
                            "seam_name": seam_name,
                            "start_idx": max(0, target_idx - 5),
                            "target_idx": target_idx,
                            "y_true": y_true,
                            "raw_pred": y_true,
                            "final_pred": y_true,
                            "decode_mode": "raw",
                            "prob_class_0": 0.8 if y_true == 0 else 0.1,
                            "prob_class_1": 0.8 if y_true == 1 else 0.1,
                            "prob_class_2": 0.8 if y_true == 2 else 0.1,
                            "nearest_boundary_dist": 20 if "core" in case_type else 1,
                            "boundary_distance_bin": "Far" if "core" in case_type else "Near",
                            "case_type": case_type,
                            "error_type": "correct",
                            "confidence": 0.8,
                            "panel_key": f"{seam_name}|{case_type}",
                            "panel_status": "selected",
                            "source_seam_name": seam_name,
                        }
                    )
                    sample_index += 1

            panel_csv = tmp_path / "panels.csv"
            pd.DataFrame(panel_rows).to_csv(panel_csv, index=False)
            pred_csv = tmp_path / "predictions.csv"
            pd.DataFrame(
                [
                    {
                        "seam_name": row["seam_name"],
                        "target_idx": row["target_idx"],
                        "y_true": row["y_true"],
                        "final_pred": row["final_pred"],
                    }
                    for row in panel_rows
                ]
            ).to_csv(pred_csv, index=False)

            output_root = tmp_path / "paper_figures"
            mod.main(
                [
                    "--panel-csv",
                    str(panel_csv),
                    "--predictions-csv",
                    str(pred_csv),
                    "--raw-data-dir",
                    str(raw_dir),
                    "--output-root",
                    str(output_root),
                    "--output-name",
                    "figure1_h1_case_gallery_matplotlib",
                ]
            )

            out_dir = output_root / "figure1_h1_case_gallery_matplotlib_v01"
            out_pdf = out_dir / "figure1_h1_case_gallery_matplotlib.pdf"
            out_png = out_dir / "figure1_h1_case_gallery_matplotlib.png"
            out_svg = out_dir / "figure1_h1_case_gallery_matplotlib.svg"

            self.assertTrue(out_dir.exists())
            self.assertTrue(out_pdf.exists())
            self.assertTrue(out_png.exists())
            self.assertTrue(out_svg.exists())
            self.assertGreater(out_pdf.stat().st_size, 0)
            self.assertGreater(out_png.stat().st_size, 0)
            self.assertGreater(out_svg.stat().st_size, 0)

    def test_prepare_output_dir_creates_isolated_versioned_folder(self) -> None:
        import render_figure1_matplotlib as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "paper_figures"
            root.mkdir(parents=True, exist_ok=True)

            out_dir = mod.prepare_output_dir(root, "figure1_h1_case_gallery_matplotlib")

            self.assertTrue(out_dir.exists())
            self.assertEqual(out_dir.parent, root)
            self.assertIn("figure1_h1_case_gallery_matplotlib", out_dir.name)


if __name__ == "__main__":
    unittest.main()
