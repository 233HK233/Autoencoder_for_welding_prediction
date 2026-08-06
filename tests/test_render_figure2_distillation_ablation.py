import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "render_figure2_distillation_ablation.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def write_metrics(path: Path, *, macro_f1: float, s1_recall: float, agreement_pct: float = 95.0) -> None:
    path.write_text(
        "\n".join(
            [
                "Best epoch: 4",
                "Checkpoint metric: val_teacher_agreement",
                "Best score: 0.900000",
                "",
                "--- Test Metrics (Student) ---",
                "Accuracy: 98.03%",
                f"Macro-F1: {macro_f1:.6f}",
                f"Test Teacher-Agreement: {agreement_pct:.2f}%",
                "",
                "=== Classification Report (Test) ===",
                "              precision    recall  f1-score   support",
                "     Class 0     1.0000   0.9500     0.9744        10",
                f"     Class 1     0.9000   {s1_recall:.4f}     0.9000        10",
                "     Class 2     0.9800   1.0000     0.9899        10",
                "",
                "    accuracy                       0.9000        30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_predictions(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", "y_true", "y_pred"])
        writer.writeheader()
        writer.writerows(
            [
                {"sample_index": 0, "y_true": 0, "y_pred": 2},
                {"sample_index": 1, "y_true": 1, "y_pred": 1},
                {"sample_index": 2, "y_true": 2, "y_pred": 2},
            ]
        )


class RenderFigure2DistillationAblationTests(unittest.TestCase):
    def test_parse_test_metrics_reads_macro_f1_s1_and_teacher_agreement(self) -> None:
        mod = load_module("render_fig2_distill_parse", SCRIPT_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "evaluation_metrics.txt"
            write_metrics(metrics_path, macro_f1=0.973757, s1_recall=1.0, agreement_pct=96.35)

            parsed = mod.parse_test_metrics(metrics_path)

        self.assertAlmostEqual(parsed["macro_f1"], 0.973757)
        self.assertAlmostEqual(parsed["s1_recall"], 1.0)
        self.assertAlmostEqual(parsed["teacher_agreement"], 0.9635)

    def test_severe_cross_stage_error_uses_ordered_class_distance(self) -> None:
        mod = load_module("render_fig2_distill_severe", SCRIPT_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            prediction_path = Path(tmp) / "test_predictions.csv"
            write_predictions(prediction_path)

            severe = mod.severe_cross_stage_error(prediction_path)

        self.assertAlmostEqual(severe, 1.0 / 3.0)

    def test_build_component_table_uses_run_specs_and_prediction_files(self) -> None:
        mod = load_module("render_fig2_distill_component", SCRIPT_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            write_metrics(run_dir / "evaluation_metrics.txt", macro_f1=0.9, s1_recall=0.8)
            write_predictions(run_dir / "test_predictions.csv")
            specs = [
                mod.MethodSpec(
                    method="Full SEAL-Weld",
                    role="full",
                    run_dir=run_dir,
                    protocol_note="matched",
                )
            ]

            table, notes = mod.build_component_ablation_metrics(specs)

        self.assertEqual(table.loc[0, "method"], "Full SEAL-Weld")
        self.assertAlmostEqual(float(table.loc[0, "macro_f1_pct"]), 90.0)
        self.assertAlmostEqual(float(table.loc[0, "s1_recall_pct"]), 80.0)
        self.assertAlmostEqual(float(table.loc[0, "severe_cross_stage_error_pct"]), 100.0 / 3.0)
        self.assertEqual(notes, [])

    def test_build_heatmap_source_excludes_failed_rows(self) -> None:
        mod = load_module("render_fig2_distill_heatmap", SCRIPT_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            good_run = tmp_path / "good"
            good_run.mkdir()
            write_metrics(good_run / "evaluation_metrics.txt", macro_f1=0.92, s1_recall=0.75)
            sweep = pd.DataFrame(
                [
                    {
                        "temperature": 3.0,
                        "lambda_kd": 1.2,
                        "lambda_feat": 0.2,
                        "test_macro_f1": 0.92,
                        "test_teacher_agreement": 0.95,
                        "run_dir": str(good_run),
                    },
                    {
                        "temperature": 3.5,
                        "lambda_kd": 1.5,
                        "lambda_feat": 0.3,
                        "test_macro_f1": -1.0,
                        "test_teacher_agreement": -1.0,
                        "run_dir": str(tmp_path / "failed"),
                    },
                ]
            )

            table = mod.build_heatmap_source(
                sweep,
                x_col="temperature",
                y_col="lambda_kd",
                value_col="test_macro_f1",
                value_name="macro_f1",
            )

        self.assertEqual(len(table), 1)
        self.assertAlmostEqual(float(table.loc[0, "macro_f1_pct"]), 92.0)
        self.assertEqual(int(table.loc[0, "n_runs"]), 1)

    def test_render_uses_three_by_three_layout(self) -> None:
        mod = load_module("render_fig2_distill_layout", SCRIPT_PATH)
        component = pd.DataFrame(
            {
                "method": ["Full SEAL-Weld"],
                "macro_f1_pct": [97.0],
                "s1_recall_pct": [100.0],
                "severe_cross_stage_error_pct": [0.0],
                "available": [True],
            }
        )
        heatmap = pd.DataFrame({"x": [1.0], "y": [2.0], "value_pct": [90.0], "n_runs": [1]})
        dynamics = pd.DataFrame(
            {
                "method": ["Full SEAL-Weld"],
                "epoch": [1],
                "ce_loss": [1.0],
                "kd_loss": [0.2],
                "latent_mse": [0.3],
                "val_macro_f1_pct": [90.0],
                "teacher_agreement_pct": [95.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with mock.patch.object(mod.plt, "subplots", wraps=mod.plt.subplots) as subplots_mock:
                mod.render(
                    component_df=component,
                    heatmap_macro=heatmap,
                    heatmap_s1=heatmap,
                    heatmap_agree=heatmap,
                    dynamics_df=dynamics,
                    output_base=out / "figure2_distillation_ablation",
                )

        subplots_mock.assert_called_once_with(3, 3, figsize=(13.5, 10.5), constrained_layout=True)
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
