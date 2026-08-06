import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _write_demo_csv(path: Path, rows: int = 72) -> None:
    labels = []
    for idx in range(rows):
        if idx < 24:
            labels.append("quasistable")
        elif idx < 48:
            labels.append("nonstationary")
        else:
            labels.append("instability")

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for idx, label in enumerate(labels):
            feats = [round(idx + j * 0.1, 4) for j in range(18)]
            writer.writerow(feats + [label])


class BaselineLSTMExperimentSmokeTests(unittest.TestCase):
    def test_cli_runs_and_writes_analysis_artifacts(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "raw"
            output_dir = tmp_path / "outputs"
            input_dir.mkdir()
            _write_demo_csv(input_dir / "demo.csv")

            cmd = [
                sys.executable,
                str(repo_root / "train_baseline_lstm_classifier.py"),
                "--input-dir",
                str(input_dir),
                "--seams",
                "demo.csv",
                "--output-dir",
                str(output_dir),
                "--epochs",
                "2",
                "--batch-size",
                "4",
                "--window-size",
                "4",
                "--target-offset",
                "2",
                "--train-frac",
                "0.75",
                "--seed",
                "7",
            ]

            result = subprocess.run(
                cmd,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            self.assertNotIn("Traceback", result.stderr)

            run_dirs = [path for path in output_dir.iterdir() if path.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]

            expected_files = {
                "artifact_status.json",
                "run_args.json",
                "history.json",
                "evaluation_metrics.txt",
                "dataset_split_summary.json",
                "test_predictions.csv",
                "confusion_matrix_test.csv",
                "loss_curve.png",
                "metrics_curve.png",
                "best_baseline_lstm.pth",
            }
            self.assertTrue(expected_files.issubset({path.name for path in run_dir.iterdir()}))

            summary = json.loads((run_dir / "dataset_split_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["train_frac"], 0.75)
            self.assertEqual(summary["seams"], ["demo"])
            self.assertEqual(summary["split_strategy"], "label_segment_time_split")
            self.assertEqual(summary["validation_strategy"], "none")
            self.assertGreater(summary["train_samples"], 0)
            self.assertGreater(summary["test_samples"], 0)
            self.assertEqual(summary["test_class_counts"], {"0": 1, "1": 1, "2": 1})
            self.assertFalse(summary["warnings"])

            artifact_status = json.loads((run_dir / "artifact_status.json").read_text(encoding="utf-8"))
            self.assertIn(artifact_status["loss_curve"], {"matplotlib", "placeholder_png"})
            self.assertIn(artifact_status["metrics_curve"], {"matplotlib", "placeholder_png"})


class BaselineLSTMWindowingTests(unittest.TestCase):
    def test_create_future_offset_sequences_tracks_target_indices(self) -> None:
        import numpy as np
        import train_baseline_lstm_classifier as mod

        data = np.arange(12 * 2, dtype=np.float32).reshape(12, 2)
        labels = np.arange(12, dtype=np.int64) % 3

        x, y, start_idx, target_idx = mod.create_future_offset_sequences(
            data,
            labels,
            window_size=4,
            target_offset=2,
            start_offset=10,
        )

        self.assertEqual(x.shape, (7, 4, 2))
        self.assertEqual(start_idx.tolist(), [10, 11, 12, 13, 14, 15, 16])
        self.assertEqual(target_idx.tolist(), [15, 16, 17, 18, 19, 20, 21])
        self.assertEqual(y.tolist(), labels[5:12].tolist())

    def test_split_and_window_single_seam_keeps_train_test_windows_separate(self) -> None:
        import numpy as np
        import train_baseline_lstm_classifier as mod

        data = np.arange(72 * 3, dtype=np.float32).reshape(72, 3)
        labels = np.array(([0] * 24) + ([1] * 24) + ([2] * 24), dtype=np.int64)

        split = mod.split_and_window_single_seam_by_label_segments(
            data,
            labels,
            train_frac=0.75,
            window_size=4,
            target_offset=2,
        )

        self.assertEqual(split["segment_lengths"], [24, 24, 24])
        self.assertEqual(split["x_train"].shape[0], 39)
        self.assertEqual(split["x_test"].shape[0], 3)
        self.assertEqual(split["start_train"][0], 0)
        self.assertEqual(split["target_train"][0], 5)
        self.assertEqual(split["start_test"].tolist(), [18, 42, 66])
        self.assertEqual(split["target_test"].tolist(), [23, 47, 71])
        self.assertEqual(split["y_train"].tolist().count(0), 13)
        self.assertEqual(split["y_train"].tolist().count(1), 13)
        self.assertEqual(split["y_train"].tolist().count(2), 13)
        self.assertEqual(split["y_test"].tolist(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
