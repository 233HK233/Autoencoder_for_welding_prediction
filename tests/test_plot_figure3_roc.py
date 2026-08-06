import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "ablation_experiments/scripts/plot_figure3_roc.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_prediction_table(mod, method_name: str, method_slug: str, probs: np.ndarray, y_true: np.ndarray):
    n_samples = y_true.shape[0]
    return mod.PredictionTable(
        method_name=method_name,
        method_slug=method_slug,
        sample_index=np.arange(n_samples, dtype=np.int64),
        seam_id=np.zeros((n_samples,), dtype=np.int64),
        seam_name=np.array(["a01"] * n_samples, dtype=object),
        start_idx=np.arange(100, 100 + n_samples, dtype=np.int64),
        target_idx=np.arange(105, 105 + n_samples, dtype=np.int64),
        y_true=y_true,
        y_pred=np.argmax(probs, axis=1).astype(np.int64),
        probabilities=probs,
    )


class PlotFigure3RocHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_module("plot_figure3_roc_test", SCRIPT_PATH)

    def test_identity_validation_catches_mismatched_y_true_order(self) -> None:
        y_true = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        probs = np.array(
            [
                [0.9, 0.05, 0.05],
                [0.1, 0.8, 0.1],
                [0.05, 0.1, 0.85],
                [0.7, 0.2, 0.1],
                [0.1, 0.75, 0.15],
                [0.1, 0.1, 0.8],
            ],
            dtype=np.float64,
        )
        table_a = build_prediction_table(self.mod, "lstm", "lstm", probs, y_true)
        table_b = build_prediction_table(self.mod, "gru", "gru", probs, y_true[::-1])

        with self.assertRaisesRegex(ValueError, "y_true"):
            self.mod.validate_identity_columns([table_a, table_b])

    def test_roc_table_generation_returns_thirty_auc_rows(self) -> None:
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
        base_probs = np.array(
            [
                [0.92, 0.05, 0.03],
                [0.09, 0.82, 0.09],
                [0.04, 0.11, 0.85],
                [0.73, 0.17, 0.10],
                [0.16, 0.69, 0.15],
                [0.13, 0.12, 0.75],
                [0.68, 0.22, 0.10],
                [0.18, 0.63, 0.19],
                [0.08, 0.13, 0.79],
            ],
            dtype=np.float64,
        )

        tables = []
        for idx, method_name in enumerate(self.mod.FIGURE3_METHOD_ORDER):
            probs = base_probs.copy()
            probs[:, idx % 3] = np.clip(probs[:, idx % 3] + 0.01, 0.0, 0.98)
            probs = probs / probs.sum(axis=1, keepdims=True)
            tables.append(
                build_prediction_table(
                    self.mod,
                    method_name=method_name,
                    method_slug=self.mod.METHOD_SLUGS[method_name],
                    probs=probs,
                    y_true=y_true,
                )
            )

        auc_rows, curve_rows = self.mod.build_roc_tables(tables)

        self.assertEqual(len(auc_rows), 30)
        self.assertGreater(len(curve_rows), 30)

    def test_macro_and_micro_rows_exist_for_every_method(self) -> None:
        y_true = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        probs = np.array(
            [
                [0.88, 0.07, 0.05],
                [0.10, 0.82, 0.08],
                [0.05, 0.12, 0.83],
                [0.76, 0.15, 0.09],
                [0.14, 0.70, 0.16],
                [0.11, 0.14, 0.75],
            ],
            dtype=np.float64,
        )
        tables = [
            build_prediction_table(
                self.mod,
                method_name=method_name,
                method_slug=self.mod.METHOD_SLUGS[method_name],
                probs=probs,
                y_true=y_true,
            )
            for method_name in self.mod.FIGURE3_METHOD_ORDER
        ]

        auc_rows, _ = self.mod.build_roc_tables(tables)

        expected_pairs = {
            (method_name, "macro-average") for method_name in self.mod.FIGURE3_METHOD_ORDER
        } | {
            (method_name, "micro-average") for method_name in self.mod.FIGURE3_METHOD_ORDER
        }
        actual_pairs = {(row["method_name"], row["roc_view"]) for row in auc_rows}

        self.assertTrue(expected_pairs.issubset(actual_pairs))

    def test_partial_auc_is_normalized_over_requested_fpr_window(self) -> None:
        fpr = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        tpr = np.array([0.0, 1.0, 1.0], dtype=np.float64)

        pauc = self.mod.compute_normalized_partial_auc(fpr=fpr, tpr=tpr, max_fpr=0.10)

        self.assertAlmostEqual(pauc, 1.0, places=6)

    def test_pauc_summary_rows_exist_for_all_methods(self) -> None:
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
        base_probs = np.array(
            [
                [0.92, 0.05, 0.03],
                [0.09, 0.82, 0.09],
                [0.04, 0.11, 0.85],
                [0.73, 0.17, 0.10],
                [0.16, 0.69, 0.15],
                [0.13, 0.12, 0.75],
                [0.68, 0.22, 0.10],
                [0.18, 0.63, 0.19],
                [0.08, 0.13, 0.79],
            ],
            dtype=np.float64,
        )
        probs = base_probs / base_probs.sum(axis=1, keepdims=True)
        tables = [
            build_prediction_table(
                self.mod,
                method_name=method_name,
                method_slug=self.mod.METHOD_SLUGS[method_name],
                probs=probs,
                y_true=y_true,
            )
            for method_name in self.mod.FIGURE3_METHOD_ORDER
        ]

        pauc_rows = self.mod.build_pauc_summary_rows(tables, max_fpr=0.10)

        self.assertEqual(len(pauc_rows), 6)
        self.assertEqual({row["method_name"] for row in pauc_rows}, set(self.mod.FIGURE3_METHOD_ORDER))
        self.assertTrue(all(0.0 <= float(row["pauc_at_fpr_0p10"]) <= 1.0 for row in pauc_rows))

    def test_output_filenames_are_deterministic(self) -> None:
        paths = self.mod.build_output_paths(Path("/tmp/roc_figure3"))

        self.assertEqual(paths["roc_auc_summary"].name, "roc_auc_summary.csv")
        self.assertEqual(paths["roc_curve_points"].name, "roc_curve_points.csv")
        self.assertEqual(paths["macro_svg"].name, "figure3_roc_macro_average.svg")
        self.assertEqual(paths["macro_pdf"].name, "figure3_roc_macro_average.pdf")
        self.assertEqual(paths["macro_png"].name, "figure3_roc_macro_average.png")
        self.assertEqual(paths["per_class_svg"].name, "figure3_roc_per_class.svg")
        self.assertEqual(paths["per_class_pdf"].name, "figure3_roc_per_class.pdf")
        self.assertEqual(paths["per_class_png"].name, "figure3_roc_per_class.png")

    def test_v1_output_filenames_do_not_overwrite_existing_roc_files(self) -> None:
        paths = self.mod.build_output_paths(Path("/tmp/roc_figure3"), tag="v1")

        self.assertEqual(paths["macro_svg"].name, "figure3_roc_macro_average_v1.svg")
        self.assertEqual(paths["macro_pdf"].name, "figure3_roc_macro_average_v1.pdf")
        self.assertEqual(paths["macro_png"].name, "figure3_roc_macro_average_v1.png")
        self.assertEqual(paths["pauc_summary"].name, "roc_pauc_summary_at_fpr_0p10_v1.csv")

    def test_publication_legend_labels_use_requested_display_names(self) -> None:
        expected = {
            "teacher-student(student)": "Proposed, 13D (0.9991)",
            "teacher(18D upper bound)": "Teacher, 18D (0.9997)",
            "transformer": "Transformer, 18D (0.9866)",
            "gru": "GRU, 18D (0.9798)",
            "lstm": "LSTM, 18D (0.9791)",
            "inception": "InceptionTime, 18D (0.9691)",
        }

        for method_name, label in expected.items():
            with self.subTest(method_name=method_name):
                auc_value = float(label.rsplit("(", 1)[1].rstrip(")"))
                self.assertEqual(self.mod.format_method_label(method_name, auc_value), label)

    def test_legend_config_makes_macro_auc_semantics_explicit(self) -> None:
        legend_cfg = self.mod.build_macro_legend_config()

        self.assertEqual(legend_cfg["title"], "Method (macro-AUC)")
        self.assertEqual(legend_cfg["handlelength"], 1.8)
        self.assertEqual(legend_cfg["handletextpad"], 0.6)
        self.assertEqual(legend_cfg["labelspacing"], 0.48)

    def test_right_side_info_panels_share_aligned_left_boundary(self) -> None:
        info_cfg = self.mod.build_right_info_panel_config()

        self.assertEqual(info_cfg["legend_bounds"][0], info_cfg["table_bounds"][0])
        self.assertGreater(info_cfg["legend_bounds"][1], info_cfg["table_bounds"][1])

    def test_pauc_table_config_uses_lighter_title_and_borders(self) -> None:
        table_cfg = self.mod.build_pauc_table_config()

        self.assertEqual(table_cfg["title"], "pAUC at FPR \u2264 0.10")
        self.assertLess(table_cfg["font_size"], 6.4)
        self.assertLessEqual(table_cfg["header_linewidth"], 0.25)
        self.assertLess(table_cfg["title_fontsize"], 7.0)

    def test_plot_style_contract_uses_requested_weight_hierarchy(self) -> None:
        styles = self.mod.build_method_style_map()

        self.assertEqual(styles["teacher-student(student)"]["linestyle"], "-")
        self.assertEqual(styles["teacher(18D upper bound)"]["linestyle"], (0, (5, 3)))
        self.assertEqual(styles["transformer"]["linestyle"], (0, (4, 2, 1, 2)))
        self.assertEqual(styles["gru"]["linestyle"], (0, (4, 2)))
        self.assertEqual(styles["lstm"]["linestyle"], (0, (6, 3)))
        self.assertEqual(styles["inception"]["linestyle"], (0, (1, 2)))
        self.assertEqual(styles["teacher-student(student)"]["linewidth"], 1.8)
        self.assertEqual(styles["teacher(18D upper bound)"]["linewidth"], 1.4)
        self.assertEqual(styles["transformer"]["linewidth"], 1.2)
        self.assertEqual(styles["gru"]["linewidth"], 1.2)
        self.assertEqual(styles["lstm"]["linewidth"], 1.2)
        self.assertEqual(styles["inception"]["linewidth"], 1.2)
        self.assertEqual(styles["teacher-student(student)"]["alpha"], 1.0)
        self.assertEqual(styles["teacher(18D upper bound)"]["alpha"], 0.85)
        self.assertEqual(styles["transformer"]["alpha"], 0.78)
        self.assertEqual(styles["gru"]["alpha"], 0.78)
        self.assertEqual(styles["lstm"]["alpha"], 0.78)
        self.assertEqual(styles["inception"]["alpha"], 0.78)
        self.assertEqual(styles["teacher-student(student)"]["color"], "#0B4F9C")
        self.assertEqual(styles["teacher(18D upper bound)"]["color"], "#3A3A3A")

    def test_macro_inset_contract_matches_requested_zoom_window(self) -> None:
        inset_cfg = self.mod.build_macro_inset_config()

        self.assertEqual(inset_cfg["xlim"], (0.0, 0.10))
        self.assertEqual(inset_cfg["ylim"], (0.88, 1.01))
        self.assertEqual(inset_cfg["bounds"], (0.48, 0.24, 0.34, 0.34))
        self.assertFalse(inset_cfg["draw_connectors"])
        self.assertIn("teacher(18D upper bound)", inset_cfg["method_names"])
        self.assertIn("teacher-student(student)", inset_cfg["method_names"])
        self.assertEqual(inset_cfg["title"], "FPR <= 0.10")

    def test_chance_line_is_deemphasized(self) -> None:
        chance = self.mod.build_chance_line_style()

        self.assertEqual(chance["color"], "0.70")
        self.assertEqual(chance["linewidth"], 0.7)
        self.assertEqual(chance["linestyle"], "--")
        self.assertEqual(chance["alpha"], 0.40)
        self.assertEqual(chance["zorder"], 0)

    def test_macro_panel_contract_uses_caption_only_title_and_panel_label(self) -> None:
        panel_cfg = self.mod.build_macro_panel_config()

        self.assertEqual(panel_cfg["panel_label"], "a")
        self.assertIsNone(panel_cfg["title"])
        self.assertEqual(panel_cfg["x_label"], "False positive rate")
        self.assertEqual(panel_cfg["y_label"], "True positive rate")
        self.assertEqual(panel_cfg["xlim"], (0.0, 1.0))
        self.assertEqual(panel_cfg["ylim"], (0.0, 1.02))
        self.assertLess(panel_cfg["inset_tick_fontsize"], panel_cfg["main_tick_fontsize"])


if __name__ == "__main__":
    unittest.main()
