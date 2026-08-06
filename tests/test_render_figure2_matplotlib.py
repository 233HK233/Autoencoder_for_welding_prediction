import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt


def write_metrics(
    run_dir: Path,
    *,
    test_heading: str,
    test_acc_pct: float,
    test_macro_f1: float,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "Best epoch: 4",
        "Checkpoint metric: test_acc",
        "Best score: 0.920000",
        "",
        "--- Train Metrics ---",
        "Loss: 0.100000",
        "Accuracy: 90.00%",
        "Macro-F1: 0.900000",
        "",
        "--- Val Metrics ---",
        "Loss: 0.110000",
        "Accuracy: 91.00%",
        "Macro-F1: 0.910000",
        "",
        test_heading,
        "Loss: 0.120000",
        f"Accuracy: {test_acc_pct:.2f}%",
        f"Macro-F1: {test_macro_f1:.6f}",
        "",
        "=== Classification Report (Test) ===",
        "              precision    recall  f1-score   support",
        "     Class 0     1.0000   0.9500     0.9744        10",
        "     Class 1     0.9000   0.9000     0.9000        10",
        "     Class 2     0.9800   1.0000     0.9899        10",
        "",
        "    accuracy                       0.9000        30",
        "   macro avg     0.0000   0.0000     0.9000        30",
    ]
    (run_dir / "evaluation_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_registry(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_name", "run_dir", "included_in_fig2"])
        writer.writeheader()
        writer.writerows(rows)


class RenderFigure2MatplotlibTests(unittest.TestCase):
    def _build_summary_df(self, tmp_path: Path):
        import render_figure2_matplotlib as mod

        registry_csv = tmp_path / "registry.csv"
        rows = []
        for idx, (model_name, test_heading, acc, macro_f1) in enumerate(
            [
                ("Exploratory Teacher Best", "--- Test Metrics ---", 98.31, 0.979252),
                ("Best Student Distill", "--- Test Metrics (Student) ---", 98.03, 0.973757),
                ("Best Ablation-1", "--- Test Metrics (Student-only) ---", 94.66, 0.936082),
                (
                    "Best Ablation-2",
                    "--- Test Metrics (Student LSTM Distill, future-step prediction) ---",
                    92.70,
                    0.894788,
                ),
            ]
        ):
            run_dir = tmp_path / f"run_{idx}"
            write_metrics(run_dir, test_heading=test_heading, test_acc_pct=acc, test_macro_f1=macro_f1)
            rows.append(
                {
                    "model_name": model_name,
                    "run_dir": str(run_dir),
                    "included_in_fig2": "yes",
                }
            )
        write_registry(registry_csv, rows)
        return mod.build_summary_dataframe(mod.prepare_registry_dataframe(registry_csv))

    def test_parse_args_uses_human_readable_dataset_label_by_default(self) -> None:
        import render_figure2_matplotlib as mod

        args = mod.parse_args([])

        self.assertEqual(args.dataset_label, "Weld seam feature windows")

    def test_build_summary_dataframe_reads_metrics_and_teacher_deltas(self) -> None:
        import render_figure2_matplotlib as mod

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            teacher_dir = tmp_path / "teacher"
            student_dir = tmp_path / "student"
            ablation1_dir = tmp_path / "ablation1"
            ablation2_dir = tmp_path / "ablation2"

            write_metrics(
                teacher_dir,
                test_heading="--- Test Metrics ---",
                test_acc_pct=98.31,
                test_macro_f1=0.979252,
            )
            write_metrics(
                student_dir,
                test_heading="--- Test Metrics (Student) ---",
                test_acc_pct=98.03,
                test_macro_f1=0.973757,
            )
            write_metrics(
                ablation1_dir,
                test_heading="--- Test Metrics (Student-only) ---",
                test_acc_pct=94.66,
                test_macro_f1=0.936082,
            )
            write_metrics(
                ablation2_dir,
                test_heading="--- Test Metrics (Student LSTM Distill, future-step prediction) ---",
                test_acc_pct=92.70,
                test_macro_f1=0.894788,
            )

            registry_csv = tmp_path / "registry.csv"
            write_registry(
                registry_csv,
                [
                    {
                        "model_name": "Exploratory Teacher Best",
                        "run_dir": str(teacher_dir),
                        "included_in_fig2": "yes",
                    },
                    {
                        "model_name": "Best Student Distill",
                        "run_dir": str(student_dir),
                        "included_in_fig2": "yes",
                    },
                    {
                        "model_name": "Best Ablation-1",
                        "run_dir": str(ablation1_dir),
                        "included_in_fig2": "yes",
                    },
                    {
                        "model_name": "Best Ablation-2",
                        "run_dir": str(ablation2_dir),
                        "included_in_fig2": "yes",
                    },
                ],
            )

            registry_df = mod.prepare_registry_dataframe(registry_csv)
            summary_df = mod.build_summary_dataframe(registry_df)

            self.assertEqual(
                summary_df["display_name"].astype(str).tolist(),
                ["Teacher", "Student", "Ablation-1", "Ablation-2"],
            )
            self.assertAlmostEqual(float(summary_df.iloc[0]["test_acc_pct"]), 98.31, places=2)
            self.assertAlmostEqual(float(summary_df.iloc[1]["delta_acc_pp"]), -0.28, places=2)
            self.assertAlmostEqual(float(summary_df.iloc[1]["delta_macro_f1_pp"]), -0.5495, places=4)
            self.assertAlmostEqual(float(summary_df.iloc[3]["delta_acc_pp"]), -5.61, places=2)
            self.assertAlmostEqual(float(summary_df.iloc[3]["delta_macro_f1_pp"]), -8.4464, places=4)

    def test_text_only_panel_helpers_are_removed(self) -> None:
        import render_figure2_matplotlib as mod

        self.assertFalse(hasattr(mod, "plot_method_cards"))
        self.assertFalse(hasattr(mod, "plot_note_panel"))

    def test_panel_titles_match_six_panel_layout(self) -> None:
        import render_figure2_matplotlib as mod

        self.assertEqual(set(mod.PANEL_TITLES.keys()), set("abcdef"))
        self.assertEqual(mod.PANEL_TITLES["c"], "Accuracy distribution")
        self.assertEqual(mod.PANEL_TITLES["d"], "Accuracy-Macro-F1 balance")

    def test_plot_ranking_strip_uses_accuracy_number_line(self) -> None:
        import render_figure2_matplotlib as mod

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary_df = self._build_summary_df(tmp_path)
            fig, ax = plt.subplots()
            try:
                mod.plot_ranking_strip(ax, summary_df)
                texts = [text.get_text() for text in ax.texts]
                self.assertFalse(any(">" in text for text in texts))
                xmin, xmax = ax.get_xlim()
                self.assertAlmostEqual(xmin, 90.0, places=3)
                self.assertAlmostEqual(xmax, 100.0, places=3)
                self.assertFalse(ax.get_yaxis().get_visible())
                self.assertGreater(len(ax.collections), 0)
                self.assertEqual(len(ax.patches), 0)
                self.assertEqual(ax.spines["bottom"].get_position(), ("data", 0))
                self.assertTrue(all(collection.get_alpha() == 0.85 for collection in ax.collections))
            finally:
                plt.close(fig)

    def test_plot_ranking_strip_offsets_close_labels(self) -> None:
        import render_figure2_matplotlib as mod

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary_df = self._build_summary_df(tmp_path)
            fig, ax = plt.subplots()
            try:
                mod.plot_ranking_strip(ax, summary_df)
                labels = {text.get_text(): text for text in ax.texts if text.get_text() in mod.DISPLAY_ORDER}
                self.assertEqual(labels["Teacher"].get_va(), "bottom")
                self.assertEqual(labels["Student"].get_va(), "top")
            finally:
                plt.close(fig)

    def test_plot_balance_panel_uses_scatter_axes_for_two_metrics(self) -> None:
        import render_figure2_matplotlib as mod

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary_df = self._build_summary_df(tmp_path)
            fig, ax = plt.subplots()
            try:
                mod.plot_balance_panel(ax, summary_df)
                self.assertEqual(ax.get_xlabel(), "Test Accuracy (%)")
                self.assertEqual(ax.get_ylabel(), "Macro-F1 (%)")
                self.assertEqual(len(ax.lines), 0)
                self.assertGreater(len(ax.collections), 0)
            finally:
                plt.close(fig)

    def test_plot_balance_panel_uses_full_grid_and_marker_alpha(self) -> None:
        import render_figure2_matplotlib as mod

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary_df = self._build_summary_df(tmp_path)
            fig, ax = plt.subplots()
            try:
                mod.plot_balance_panel(ax, summary_df)
                self.assertGreater(len(ax.get_xgridlines()), 0)
                self.assertGreater(len(ax.get_ygridlines()), 0)
                self.assertTrue(all(grid.get_visible() for grid in ax.get_xgridlines()))
                self.assertTrue(all(grid.get_visible() for grid in ax.get_ygridlines()))
                self.assertTrue(all(collection.get_alpha() == 0.85 for collection in ax.collections))
            finally:
                plt.close(fig)

    def test_plot_balance_panel_offsets_teacher_and_student_labels(self) -> None:
        import render_figure2_matplotlib as mod

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary_df = self._build_summary_df(tmp_path)
            fig, ax = plt.subplots()
            try:
                mod.plot_balance_panel(ax, summary_df)
                labels = {text.get_text(): text for text in ax.texts}
                self.assertEqual(labels["Teacher"].get_va(), "bottom")
                self.assertEqual(labels["Student"].get_va(), "top")
            finally:
                plt.close(fig)

    def test_plot_delta_bars_tightens_upper_limit_and_emphasizes_zero_line(self) -> None:
        import render_figure2_matplotlib as mod

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary_df = self._build_summary_df(tmp_path)
            fig, ax = plt.subplots()
            try:
                mod.plot_delta_bars(
                    ax,
                    summary_df,
                    "delta_acc_pp",
                    "Δ Accuracy vs Teacher (pp)",
                    lower_limit=-6.4,
                    y_ticks=[-6, -4, -2, 0],
                )
                _, upper = ax.get_ylim()
                self.assertAlmostEqual(upper, 0.0, places=3)
                zero_lines = [
                    line
                    for line in ax.lines
                    if len(line.get_ydata()) >= 2 and all(value == 0.0 for value in line.get_ydata())
                ]
                self.assertEqual(len(zero_lines), 1)
                self.assertAlmostEqual(zero_lines[0].get_linewidth(), 1.2, places=3)
            finally:
                plt.close(fig)

    def test_render_uses_two_by_three_layout(self) -> None:
        import render_figure2_matplotlib as mod

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary_df = self._build_summary_df(tmp_path)
            output_pdf = tmp_path / "out.pdf"
            output_png = tmp_path / "out.png"
            output_svg = tmp_path / "out.svg"

            with mock.patch.object(mod.plt, "subplots", wraps=mod.plt.subplots) as subplots_mock:
                mod.render(
                    summary_df=summary_df,
                    output_pdf=output_pdf,
                    output_png=output_png,
                    output_svg=output_svg,
                    dataset_label="Weld seam feature windows",
                    title="",
                )

            subplots_mock.assert_called_once_with(2, 3, figsize=(12.8, 6.8))

    def test_plot_lollipop_metric_keeps_print_visible_stems(self) -> None:
        import render_figure2_matplotlib as mod

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            summary_df = self._build_summary_df(tmp_path)
            fig, ax = plt.subplots()
            try:
                mod.plot_lollipop_metric(ax, summary_df, "test_acc_pct", "Test Accuracy (%)")
                visible_alpha_values = [
                    collection.get_alpha()
                    for collection in ax.collections
                    if collection.get_alpha() is not None
                ]
                self.assertIn(0.5, visible_alpha_values)
            finally:
                plt.close(fig)

    def test_main_writes_pdf_png_and_svg(self) -> None:
        import render_figure2_matplotlib as mod

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            teacher_dir = tmp_path / "teacher"
            student_dir = tmp_path / "student"
            ablation1_dir = tmp_path / "ablation1"
            ablation2_dir = tmp_path / "ablation2"

            write_metrics(
                teacher_dir,
                test_heading="--- Test Metrics ---",
                test_acc_pct=98.31,
                test_macro_f1=0.979252,
            )
            write_metrics(
                student_dir,
                test_heading="--- Test Metrics (Student) ---",
                test_acc_pct=98.03,
                test_macro_f1=0.973757,
            )
            write_metrics(
                ablation1_dir,
                test_heading="--- Test Metrics (Student-only) ---",
                test_acc_pct=94.66,
                test_macro_f1=0.936082,
            )
            write_metrics(
                ablation2_dir,
                test_heading="--- Test Metrics (Student LSTM Distill, future-step prediction) ---",
                test_acc_pct=92.70,
                test_macro_f1=0.894788,
            )

            registry_csv = tmp_path / "registry.csv"
            write_registry(
                registry_csv,
                [
                    {
                        "model_name": "Exploratory Teacher Best",
                        "run_dir": str(teacher_dir),
                        "included_in_fig2": "yes",
                    },
                    {
                        "model_name": "Best Student Distill",
                        "run_dir": str(student_dir),
                        "included_in_fig2": "yes",
                    },
                    {
                        "model_name": "Best Ablation-1",
                        "run_dir": str(ablation1_dir),
                        "included_in_fig2": "yes",
                    },
                    {
                        "model_name": "Best Ablation-2",
                        "run_dir": str(ablation2_dir),
                        "included_in_fig2": "yes",
                    },
                ],
            )

            output_root = tmp_path / "paper_figures"
            mod.main(
                [
                    "--registry-csv",
                    str(registry_csv),
                    "--output-root",
                    str(output_root),
                    "--output-name",
                    "fig2_h1_composite_performance",
                ]
            )

            out_dir = output_root / "fig2_h1_composite_performance_v01"
            self.assertTrue(out_dir.exists())
            self.assertTrue((out_dir / "fig2_h1_composite_performance.pdf").exists())
            self.assertTrue((out_dir / "fig2_h1_composite_performance.png").exists())
            self.assertTrue((out_dir / "fig2_h1_composite_performance.svg").exists())


if __name__ == "__main__":
    unittest.main()
