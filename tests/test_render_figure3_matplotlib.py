import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import pandas as pd


def write_metrics(
    run_dir: Path,
    *,
    test_heading: str,
    test_acc_pct: float,
    test_macro_f1: float,
    class_rows: list[tuple[int, float, float, float, int]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "Best epoch: 4",
        "Checkpoint metric: val_macro_f1",
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
    ]
    for class_id, precision, recall, f1, support in class_rows:
        lines.append(
            f"     Class {class_id}     {precision:.4f}   {recall:.4f}     {f1:.4f}       {support}"
        )
    lines.extend(
        [
            "",
            "    accuracy                       0.9000        30",
            "   macro avg     0.0000   0.0000     0.9000        30",
        ]
    )
    (run_dir / "evaluation_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_registry(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "model_name",
        "model_family",
        "protocol",
        "run_dir",
        "teacher_source",
        "checkpoint_metric",
        "selection_note",
        "included_in_fig2",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_class_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "model_name",
        "model_family",
        "protocol",
        "class_id",
        "class_name",
        "metric",
        "value",
        "source_run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_baseline_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "method_name",
        "train_scheme",
        "deploy_input_dim",
        "teacher_input_dim",
        "best_test_acc",
        "best_macro_f1",
        "teacher_agreement",
        "seed",
        "checkpoint_metric",
        "run_path",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class RenderFigure3MatplotlibTests(unittest.TestCase):
    def _build_fixture(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        teacher_dir = tmp_path / "teacher"
        student_dir = tmp_path / "student"
        gru_dir = tmp_path / "gru"
        lstm_dir = tmp_path / "lstm"
        inception_dir = tmp_path / "inception"
        transformer_dir = tmp_path / "transformer"

        write_metrics(
            teacher_dir,
            test_heading="--- Test Metrics ---",
            test_acc_pct=98.31,
            test_macro_f1=0.979252,
            class_rows=[
                (0, 1.0000, 0.9615, 0.9804, 78),
                (1, 0.9326, 1.0000, 0.9651, 83),
                (2, 1.0000, 0.9846, 0.9922, 195),
            ],
        )
        write_metrics(
            student_dir,
            test_heading="--- Test Metrics (Student) ---",
            test_acc_pct=98.03,
            test_macro_f1=0.973757,
            class_rows=[
                (0, 1.0000, 0.9359, 0.9669, 78),
                (1, 0.9222, 1.0000, 0.9595, 83),
                (2, 1.0000, 0.9897, 0.9948, 195),
            ],
        )
        write_metrics(
            gru_dir,
            test_heading="--- Test Metrics (gru) ---",
            test_acc_pct=91.57,
            test_macro_f1=0.887156,
            class_rows=[
                (0, 1.0000, 0.7564, 0.8613, 78),
                (1, 0.7912, 0.8675, 0.8276, 83),
                (2, 0.9466, 1.0000, 0.9726, 195),
            ],
        )
        write_metrics(
            lstm_dir,
            test_heading="--- Test Metrics (lstm) ---",
            test_acc_pct=90.73,
            test_macro_f1=0.875341,
            class_rows=[
                (0, 1.0000, 0.7308, 0.8444, 78),
                (1, 0.7717, 0.8554, 0.8114, 83),
                (2, 0.9420, 1.0000, 0.9701, 195),
            ],
        )
        write_metrics(
            inception_dir,
            test_heading="--- Test Metrics (inception) ---",
            test_acc_pct=90.17,
            test_macro_f1=0.873180,
            class_rows=[
                (0, 1.0000, 0.7308, 0.8444, 78),
                (1, 0.7353, 0.9036, 0.8108, 83),
                (2, 0.9594, 0.9692, 0.9643, 195),
            ],
        )
        write_metrics(
            transformer_dir,
            test_heading="--- Test Metrics (transformer) ---",
            test_acc_pct=88.76,
            test_macro_f1=0.862035,
            class_rows=[
                (0, 1.0000, 0.8718, 0.9315, 78),
                (1, 0.8413, 0.6386, 0.7260, 83),
                (2, 0.8667, 1.0000, 0.9286, 195),
            ],
        )

        registry_csv = tmp_path / "registry.csv"
        write_registry(
            registry_csv,
            [
                {
                    "model_name": "Exploratory Teacher Best",
                    "model_family": "teacher",
                    "protocol": "exploratory_ordered",
                    "run_dir": str(teacher_dir),
                    "teacher_source": "",
                    "checkpoint_metric": "test_acc",
                    "selection_note": "ordered exploratory reuse",
                    "included_in_fig2": "yes",
                },
                {
                    "model_name": "Best Student Distill",
                    "model_family": "student",
                    "protocol": "exploratory_ordered",
                    "run_dir": str(student_dir),
                    "teacher_source": str(teacher_dir),
                    "checkpoint_metric": "val_teacher_agreement",
                    "selection_note": "ordered exploratory reuse",
                    "included_in_fig2": "yes",
                },
            ],
        )

        class_rows: list[dict[str, object]] = []
        teacher_student_class_metrics = {
            "Exploratory Teacher Best": {
                "S0": {"precision": 1.0000, "recall": 0.9615, "f1": 0.9804},
                "S1": {"precision": 0.9326, "recall": 1.0000, "f1": 0.9651},
                "S2": {"precision": 1.0000, "recall": 0.9846, "f1": 0.9922},
            },
            "Best Student Distill": {
                "S0": {"precision": 1.0000, "recall": 0.9359, "f1": 0.9669},
                "S1": {"precision": 0.9222, "recall": 1.0000, "f1": 0.9595},
                "S2": {"precision": 1.0000, "recall": 0.9897, "f1": 0.9948},
            },
        }
        for model_name, per_class in teacher_student_class_metrics.items():
            for class_name, metrics in per_class.items():
                class_id = int(class_name[1:])
                for metric_name, value in metrics.items():
                    class_rows.append(
                        {
                            "model_name": model_name,
                            "model_family": "teacher" if "Teacher" in model_name else "student",
                            "protocol": "exploratory_ordered",
                            "class_id": class_id,
                            "class_name": class_name,
                            "metric": metric_name,
                            "value": value,
                            "source_run_dir": str(teacher_dir if "Teacher" in model_name else student_dir),
                        }
                    )
        class_metrics_csv = tmp_path / "class_metrics.csv"
        write_class_metrics(class_metrics_csv, class_rows)

        baseline_summary_csv = tmp_path / "baseline_summary.csv"
        write_baseline_summary(
            baseline_summary_csv,
            [
                {
                    "method_name": "gru",
                    "train_scheme": "baseline_18d",
                    "deploy_input_dim": 18,
                    "teacher_input_dim": "",
                    "best_test_acc": 0.9157,
                    "best_macro_f1": 0.887156,
                    "teacher_agreement": -1.0,
                    "seed": 14,
                    "checkpoint_metric": "val_macro_f1",
                    "run_path": str(gru_dir),
                    "notes": "Best 18D gru baseline run",
                },
                {
                    "method_name": "lstm",
                    "train_scheme": "baseline_18d",
                    "deploy_input_dim": 18,
                    "teacher_input_dim": "",
                    "best_test_acc": 0.9073,
                    "best_macro_f1": 0.875341,
                    "teacher_agreement": -1.0,
                    "seed": 42,
                    "checkpoint_metric": "val_macro_f1",
                    "run_path": str(lstm_dir),
                    "notes": "Best 18D lstm baseline run",
                },
                {
                    "method_name": "inception",
                    "train_scheme": "baseline_18d",
                    "deploy_input_dim": 18,
                    "teacher_input_dim": "",
                    "best_test_acc": 0.9017,
                    "best_macro_f1": 0.873180,
                    "teacher_agreement": -1.0,
                    "seed": 14,
                    "checkpoint_metric": "val_macro_f1",
                    "run_path": str(inception_dir),
                    "notes": "Best 18D inception baseline run",
                },
                {
                    "method_name": "transformer",
                    "train_scheme": "baseline_18d",
                    "deploy_input_dim": 18,
                    "teacher_input_dim": "",
                    "best_test_acc": 0.8876,
                    "best_macro_f1": 0.862035,
                    "teacher_agreement": -1.0,
                    "seed": 14,
                    "checkpoint_metric": "val_macro_f1",
                    "run_path": str(transformer_dir),
                    "notes": "Best 18D transformer baseline run",
                },
            ],
        )
        return registry_csv, class_metrics_csv, baseline_summary_csv

    def test_build_panel_values_preserves_frozen_order_and_gru_deltas(self) -> None:
        import render_figure3_matplotlib as mod

        with tempfile.TemporaryDirectory() as tmp:
            registry_csv, class_metrics_csv, baseline_summary_csv = self._build_fixture(Path(tmp))
            data = mod.build_figure3_data(
                registry_csv=registry_csv,
                class_metrics_csv=class_metrics_csv,
                baseline_summary_csv=baseline_summary_csv,
            )

            self.assertEqual(sorted(data.panel_values_df["panel"].unique().tolist()), ["A", "B", "C", "D"])

            panel_a = data.panel_values_df.loc[data.panel_values_df["panel"] == "A"]
            acc_rows = panel_a.loc[panel_a["metric"] == "accuracy"]
            macro_rows = panel_a.loc[panel_a["metric"] == "macro_f1"]
            self.assertEqual(acc_rows["method"].tolist(), list(mod.METHOD_ORDER))
            self.assertEqual(macro_rows["method"].tolist(), list(mod.METHOD_ORDER))
            self.assertAlmostEqual(
                float(acc_rows.loc[acc_rows["method"] == "Teacher", "value_pct"].iloc[0]),
                98.31,
                places=2,
            )

            panel_c = data.panel_values_df.loc[
                (data.panel_values_df["panel"] == "C")
                & (data.panel_values_df["method"] == "Student")
                & (data.panel_values_df["class_name"] == "S1")
            ]
            self.assertAlmostEqual(float(panel_c["value_pct"].iloc[0]), 95.95, places=2)

            panel_d = data.panel_values_df.loc[data.panel_values_df["panel"] == "D"]
            self.assertEqual(panel_d["method"].tolist(), list(mod.METHOD_ORDER) * 2)
            teacher_recall = panel_d.loc[
                (panel_d["method"] == "Teacher") & (panel_d["metric"] == "recall"), "value_pct"
            ].iloc[0]
            transformer_precision = panel_d.loc[
                (panel_d["method"] == "Transformer") & (panel_d["metric"] == "precision"), "value_pct"
            ].iloc[0]
            self.assertAlmostEqual(float(teacher_recall), 100.0, places=2)
            self.assertAlmostEqual(float(transformer_precision), 84.13, places=2)

    def test_build_class_metrics_dataframe_requires_baseline_metric_files(self) -> None:
        import render_figure3_matplotlib as mod

        with tempfile.TemporaryDirectory() as tmp:
            registry_csv, class_metrics_csv, baseline_summary_csv = self._build_fixture(Path(tmp))
            baseline_df = pd.read_csv(baseline_summary_csv)
            missing_dir = Path(baseline_df.loc[baseline_df["method_name"] == "gru", "run_path"].iloc[0])
            (missing_dir / "evaluation_metrics.txt").unlink()

            with self.assertRaises(FileNotFoundError):
                mod.build_class_metrics_dataframe(class_metrics_csv, baseline_summary_csv)

    def test_render_uses_two_by_three_layout(self) -> None:
        import render_figure3_matplotlib as mod

        with tempfile.TemporaryDirectory() as tmp:
            registry_csv, class_metrics_csv, baseline_summary_csv = self._build_fixture(Path(tmp))
            data = mod.build_figure3_data(
                registry_csv=registry_csv,
                class_metrics_csv=class_metrics_csv,
                baseline_summary_csv=baseline_summary_csv,
            )
            tmp_path = Path(tmp)
            output_pdf = tmp_path / "out.pdf"
            output_png = tmp_path / "out.png"
            output_svg = tmp_path / "out.svg"

            with mock.patch.object(mod.plt, "subplots", wraps=mod.plt.subplots) as subplots_mock:
                mod.render(
                    data=data,
                    output_pdf=output_pdf,
                    output_png=output_png,
                    output_svg=output_svg,
                    title="",
                )

            subplots_mock.assert_called_once_with(2, 3, figsize=(14.2, 7.8))

    def test_overall_metrics_panel_annotates_bar_values(self) -> None:
        import render_figure3_matplotlib as mod

        with tempfile.TemporaryDirectory() as tmp:
            registry_csv, class_metrics_csv, baseline_summary_csv = self._build_fixture(Path(tmp))
            data = mod.build_figure3_data(
                registry_csv=registry_csv,
                class_metrics_csv=class_metrics_csv,
                baseline_summary_csv=baseline_summary_csv,
            )
            fig, ax = plt.subplots()
            try:
                mod._plot_overall_metrics_panel(ax, data.overall_df)
                labels = {text.get_text() for text in ax.texts}
                self.assertIn("98.3", labels)
                self.assertIn("88.7", labels)
            finally:
                plt.close(fig)

    def test_heatmap_keeps_row_labels_high_contrast(self) -> None:
        import render_figure3_matplotlib as mod

        fig, ax = plt.subplots()
        try:
            mod._plot_heatmap(
                ax,
                matrix=pd.DataFrame(
                    [
                        [98.0, 96.5, 99.2],
                        [96.7, 96.0, 99.5],
                    ]
                ).to_numpy(),
                row_labels=("Teacher", "Student"),
                col_labels=("S0", "S1", "S2"),
                cmap="YlGnBu",
                vmin=70.0,
                vmax=100.0,
                value_fmt=".1f",
            )
            for tick in ax.get_yticklabels():
                self.assertEqual(tick.get_color(), "#334155")
        finally:
            plt.close(fig)

    def test_s1_pr_scatter_uses_white_marker_borders(self) -> None:
        import render_figure3_matplotlib as mod

        with tempfile.TemporaryDirectory() as tmp:
            registry_csv, class_metrics_csv, baseline_summary_csv = self._build_fixture(Path(tmp))
            data = mod.build_figure3_data(
                registry_csv=registry_csv,
                class_metrics_csv=class_metrics_csv,
                baseline_summary_csv=baseline_summary_csv,
            )
            fig, ax = plt.subplots()
            try:
                mod._plot_s1_pr_scatter(ax, data.class_metrics_df)
                self.assertGreater(len(ax.collections), 0)
                self.assertTrue(all(collection.get_linewidths()[0] >= 1.0 for collection in ax.collections))
            finally:
                plt.close(fig)

    def test_main_writes_pdf_png_svg_and_panel_values_csv(self) -> None:
        import render_figure3_matplotlib as mod

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            registry_csv, class_metrics_csv, baseline_summary_csv = self._build_fixture(tmp_path)
            output_root = tmp_path / "paper_figures"
            mod.main(
                [
                    "--registry-csv",
                    str(registry_csv),
                    "--class-metrics-csv",
                    str(class_metrics_csv),
                    "--baseline-summary-csv",
                    str(baseline_summary_csv),
                    "--output-root",
                    str(output_root),
                ]
            )

            out_dir = output_root / "fig3_h1_comparison_layout_v01"
            panel_values_csv = out_dir / "figure3_h1_comparison_panel_values.csv"
            self.assertTrue(out_dir.exists())
            self.assertTrue((out_dir / "figure3_h1_comparison.pdf").exists())
            self.assertTrue((out_dir / "figure3_h1_comparison.png").exists())
            self.assertTrue((out_dir / "figure3_h1_comparison.svg").exists())
            self.assertTrue(panel_values_csv.exists())

            exported = pd.read_csv(panel_values_csv)
            self.assertEqual(sorted(exported["panel"].unique().tolist()), ["A", "B", "C", "D"])


if __name__ == "__main__":
    unittest.main()
