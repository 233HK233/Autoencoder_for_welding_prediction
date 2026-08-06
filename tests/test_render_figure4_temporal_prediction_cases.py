import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import pandas as pd


SEAM_ORDER = ("a01", "b01", "c01", "c02")
CASE_ORDER = ("S0-core", "0->1-boundary", "S1-core", "1->2-boundary")
METHOD_ORDER = ("Teacher-18D", "SEAL-Weld Student-13D", "Student-only-13D")


def build_result_fixture(root: Path) -> Path:
    result_dir = root / "figure4_results"
    result_dir.mkdir(parents=True, exist_ok=True)

    selection_rows: list[dict[str, object]] = []
    timeseries_rows: list[dict[str, object]] = []
    attention_rows: list[dict[str, object]] = []
    sample_index = 0

    for seam_idx, seam_name in enumerate(SEAM_ORDER):
        for case_idx, case_type in enumerate(CASE_ORDER):
            panel_key = f"{seam_name}|{case_type}"
            y_true = (seam_idx + case_idx) % 3
            source_seam_name = "b01" if panel_key == "c01|S0-core" else seam_name
            panel_status = "backfilled" if panel_key == "c01|S0-core" else "selected"
            selection_rows.append(
                {
                    "panel_key": panel_key,
                    "seam_name": seam_name,
                    "source_seam_name": source_seam_name,
                    "case_type": case_type,
                    "panel_status": panel_status,
                    "selection_rule": "cross_seam_backfill" if panel_status == "backfilled" else "fixture_rule",
                    "sample_index": sample_index,
                    "start_idx": 100 + sample_index,
                    "target_idx": 105 + sample_index,
                    "y_true": y_true,
                    "teacher_pred": y_true,
                    "seal_student_pred": y_true,
                    "student_only_pred": (y_true + (1 if case_idx == 1 else 0)) % 3,
                    "nearest_boundary_dist": 1 if "boundary" in case_type else 20,
                    "boundary_distance_bin": "Near" if "boundary" in case_type else "Far",
                    "teacher_student_both_correct": True,
                    "transfer_gain": False,
                    "seal_weld_failure": False,
                }
            )

            for relative_step in range(-3, 4):
                row = {
                    "panel_key": panel_key,
                    "panel_seam_name": seam_name,
                    "source_seam_name": source_seam_name,
                    "case_type": case_type,
                    "time_idx": 1000 + sample_index * 10 + relative_step,
                    "relative_step": relative_step,
                    "relative_time_s": relative_step * 0.01,
                    "state_label": y_true if relative_step < 0 else min(2, y_true + (1 if "boundary" in case_type else 0)),
                    "economic_mean_z": relative_step / 5.0,
                    "economic_pc1": relative_step / 4.0,
                    "teacher_pred": y_true,
                    "seal_student_pred": y_true,
                    "student_only_pred": (y_true + (1 if case_idx == 1 else 0)) % 3,
                }
                for prefix, base in (
                    ("teacher", 0.70),
                    ("seal_student", 0.76),
                    ("student_only", 0.58),
                ):
                    probs = [0.12, 0.16, 0.18]
                    probs[y_true] = base - abs(relative_step) * 0.015
                    total = sum(probs)
                    for class_idx, value in enumerate(probs):
                        row[f"{prefix}_prob_s{class_idx}"] = value / total
                timeseries_rows.append(row)

            for method_idx, method in enumerate(METHOD_ORDER):
                for input_step in range(5):
                    attention_rows.append(
                        {
                            "panel_key": panel_key,
                            "panel_seam_name": seam_name,
                            "source_seam_name": source_seam_name,
                            "case_type": case_type,
                            "method": method,
                            "method_slug": method.lower().split("-")[0],
                            "input_step": input_step,
                            "raw_idx": 100 + sample_index + input_step,
                            "relative_step": input_step - 5,
                            "relative_time_s": (input_step - 5) * 0.01,
                            "attention_weight": [0.10, 0.15, 0.20, 0.25, 0.30][input_step],
                        }
                    )

            sample_index += 1

    pd.DataFrame(selection_rows).to_csv(result_dir / "figure4_panel_selection.csv", index=False)
    pd.DataFrame(timeseries_rows).to_csv(result_dir / "figure4_panel_timeseries.csv", index=False)
    pd.DataFrame(attention_rows).to_csv(result_dir / "figure4_panel_attention.csv", index=False)
    pd.DataFrame(
        [
            {
                "sample_index": 2,
                "seam_name": "a01",
                "case_type": "S1-core",
                "target_idx": 107,
                "highlight_role": "transfer_gain",
                "selection_rule": "seal_student_correct_and_student_only_wrong_closest_boundary",
                "panel_status": "selected",
            },
            {
                "sample_index": 15,
                "seam_name": "c02",
                "case_type": "1->2-boundary",
                "target_idx": 120,
                "highlight_role": "seal_weld_failure",
                "selection_rule": "seal_student_wrong_closest_boundary",
                "panel_status": "selected",
            },
        ]
    ).to_csv(result_dir / "figure4_highlight_cases.csv", index=False)
    pd.DataFrame(
        [
            {
                "method": method,
                "matched_rows": 16,
                "max_probability_abs_delta": 0.00001,
                "prediction_argmax_match": True,
            }
            for method in METHOD_ORDER
        ]
    ).to_csv(result_dir / "figure4_validation_summary.csv", index=False)
    return result_dir


class RenderFigure4TemporalPredictionCasesTests(unittest.TestCase):
    def test_load_figure4_data_preserves_guided_layout_and_target_probabilities(self) -> None:
        import render_figure4_temporal_prediction_cases as mod

        with tempfile.TemporaryDirectory() as tmp:
            result_dir = build_result_fixture(Path(tmp))
            data = mod.load_figure4_data(result_dir)

            expected_keys = [f"{seam}|{case}" for seam in mod.SEAM_ORDER for case in mod.CASE_ORDER]
            self.assertEqual(data.panel_summary_df["panel_key"].tolist(), expected_keys)
            self.assertEqual(data.panel_summary_df["row_index"].tolist(), [idx for idx in range(4) for _ in range(4)])
            self.assertEqual(data.panel_summary_df["col_index"].tolist(), list(range(4)) * 4)

            first = data.panel_summary_df.iloc[0]
            self.assertEqual(first["teacher_target_prob_col"], f"teacher_prob_s{int(first['y_true'])}")
            self.assertEqual(first["seal_student_target_prob_col"], f"seal_student_prob_s{int(first['y_true'])}")
            self.assertEqual(first["student_only_target_prob_col"], f"student_only_prob_s{int(first['y_true'])}")

            backfilled = data.panel_summary_df.loc[data.panel_summary_df["panel_status"] == "backfilled"]
            self.assertEqual(backfilled["panel_key"].tolist(), ["c01|S0-core"])
            self.assertNotEqual(backfilled["seam_name"].iloc[0], backfilled["source_seam_name"].iloc[0])

    def test_render_uses_four_by_four_layout_and_exports_all_required_formats(self) -> None:
        import render_figure4_temporal_prediction_cases as mod

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data = mod.load_figure4_data(build_result_fixture(tmp_path))
            output_base = tmp_path / "figure4_temporal_prediction_cases"

            with mock.patch.object(mod.plt, "subplots", wraps=mod.plt.subplots) as subplots_mock:
                mod.render_figure(data=data, output_base=output_base, title="")

            subplots_mock.assert_called_once_with(
                4,
                4,
                figsize=mod.FIGURE_SIZE,
                sharex=False,
                sharey=False,
            )
            for suffix in (".svg", ".pdf", ".png", ".tiff"):
                self.assertTrue(output_base.with_suffix(suffix).exists())

    def test_main_writes_versioned_bundle_with_panel_summary_and_qa_notes(self) -> None:
        import render_figure4_temporal_prediction_cases as mod

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result_dir = build_result_fixture(tmp_path)
            output_root = tmp_path / "paper_figures"

            mod.main(["--result-dir", str(result_dir), "--output-root", str(output_root)])

            output_dir = output_root / "figure4_temporal_prediction_cases_v01"
            self.assertTrue((output_dir / "figure4_temporal_prediction_cases.svg").exists())
            self.assertTrue((output_dir / "figure4_temporal_prediction_cases.pdf").exists())
            self.assertTrue((output_dir / "figure4_temporal_prediction_cases.png").exists())
            self.assertTrue((output_dir / "figure4_temporal_prediction_cases.tiff").exists())
            self.assertTrue((output_dir / "figure4_temporal_prediction_cases_panel_summary.csv").exists())

            qa_notes = (output_dir / "figure4_qa_notes.txt").read_text(encoding="utf-8")
            self.assertIn("4 x 4 guided layout: PASS", qa_notes)
            self.assertIn("Backfilled panels: c01|S0-core", qa_notes)
            self.assertIn("Compact probability tracks: target-class only", qa_notes)


if __name__ == "__main__":
    unittest.main()
