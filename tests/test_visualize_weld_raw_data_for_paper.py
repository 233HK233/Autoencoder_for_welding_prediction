import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _write_demo_csv(path: Path, rows_per_label: int = 10) -> None:
    labels = (
        ["quasistable"] * rows_per_label
        + ["nonstationary"] * rows_per_label
        + ["instability"] * rows_per_label
    )

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for idx, label in enumerate(labels):
            row = []
            for feat_idx in range(18):
                value = idx * (feat_idx + 1) + feat_idx * 0.25
                row.append(round(value, 6))
            writer.writerow(row + [label])


class RawDataFigureUnitTests(unittest.TestCase):
    def test_load_raw_seam_csv_preserves_first_row(self) -> None:
        import numpy as np
        import visualize_weld_raw_data_for_paper as mod

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "demo.csv"
            _write_demo_csv(csv_path, rows_per_label=2)

            data, labels, raw_labels = mod.load_raw_seam_csv(csv_path)

            self.assertEqual(data.shape, (6, 18))
            self.assertEqual(labels.tolist(), [0, 0, 1, 1, 2, 2])
            self.assertEqual(raw_labels.tolist(), ["quasistable", "quasistable", "nonstationary", "nonstationary", "instability", "instability"])
            expected_first_row = np.array([feat_idx * 0.25 for feat_idx in range(18)], dtype=np.float32)
            np.testing.assert_allclose(data[0], expected_first_row)

    def test_build_feature_groups_uses_default_expensive_indices(self) -> None:
        import visualize_weld_raw_data_for_paper as mod

        groups = mod.build_feature_groups(18, [3, 4, 5, 6, 7])

        self.assertEqual(groups["expensive"], [3, 4, 5, 6, 7])
        self.assertEqual(groups["economic"], [0, 1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])

    def test_select_representative_channels_prefers_high_label_correlation(self) -> None:
        import numpy as np
        import visualize_weld_raw_data_for_paper as mod

        labels = np.array(([0] * 20) + ([1] * 20) + ([2] * 20), dtype=np.int64)
        base = labels.astype(np.float32)
        data = np.zeros((labels.shape[0], 18), dtype=np.float32)

        alternating = np.where(np.arange(labels.shape[0]) % 2 == 0, 1.0, -1.0).astype(np.float32)
        staircase = np.concatenate(
            [
                np.zeros(20, dtype=np.float32),
                np.ones(20, dtype=np.float32),
                np.ones(20, dtype=np.float32),
            ]
        )
        data[:, 0] = alternating
        data[:, 1] = np.roll(alternating, 2)
        data[:, 2] = 1.0
        data[:, 3] = alternating
        data[:, 4] = base * 1.4
        data[:, 5] = -base * 1.2
        data[:, 6] = 0.5
        data[:, 7] = np.roll(alternating, 1)
        data[:, 8] = np.roll(alternating, 3)
        data[:, 9] = staircase
        data[:, 10] = base * 1.8
        data[:, 11] = 0.2
        data[:, 12] = np.concatenate(
            [
                np.zeros(20, dtype=np.float32),
                np.full(20, 1.8, dtype=np.float32),
                np.full(20, 0.2, dtype=np.float32),
            ]
        )
        data[:, 13] = 0.1
        data[:, 14] = np.concatenate(
            [
                np.zeros(20, dtype=np.float32),
                np.full(20, 0.4, dtype=np.float32),
                np.full(20, 2.0, dtype=np.float32),
            ]
        )
        data[:, 15] = 0.3
        data[:, 16] = np.concatenate(
            [
                np.full(20, 0.2, dtype=np.float32),
                np.full(20, 1.1, dtype=np.float32),
                np.full(20, 1.3, dtype=np.float32),
            ]
        )
        data[:, 17] = 0.4

        groups = mod.build_feature_groups(18, [3, 4, 5, 6, 7])
        selected = mod.select_representative_channels(data, labels, groups, top_k=2)

        self.assertEqual(selected["expensive"], [4, 5])
        self.assertEqual(selected["economic"], [10, 14])

    def test_main_figure_group_labels_leave_gap_from_heatmap_row_labels(self) -> None:
        import visualize_weld_raw_data_for_paper as mod

        layout = mod.main_figure_layout()

        eco_group_left, eco_group_right = mod.multiline_text_bounds(
            layout["economic_group_label_x"],
            "Economic\nfeatures",
            size=10.0,
            anchor="middle",
        )
        eco_row_left_edges = [
            mod.single_line_text_bounds(
                layout["row_label_x"],
                f"Eco-{idx}",
                size=8.5,
                anchor="end",
            )[0]
            for idx in range(1, 14)
        ]

        exp_group_left, exp_group_right = mod.multiline_text_bounds(
            layout["expensive_group_label_x"],
            "Expensive\nfeatures",
            size=10.0,
            anchor="middle",
        )
        exp_row_left_edges = [
            mod.single_line_text_bounds(
                layout["row_label_x"],
                f"Exp-{idx}",
                size=8.5,
                anchor="end",
            )[0]
            for idx in range(1, 6)
        ]

        self.assertLessEqual(eco_group_right, min(eco_row_left_edges) - 6.0)
        self.assertLessEqual(exp_group_right, min(exp_row_left_edges) - 6.0)


class RawDataFigureCliSmokeTests(unittest.TestCase):
    def test_cli_generates_expected_figure_files(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "raw"
            output_dir = tmp_path / "figures"
            input_dir.mkdir()
            _write_demo_csv(input_dir / "demo.csv", rows_per_label=12)

            cmd = [
                sys.executable,
                str(repo_root / "visualize_weld_raw_data_for_paper.py"),
                "--input-dir",
                str(input_dir),
                "--seam",
                "demo.csv",
                "--output-dir",
                str(output_dir),
                "--time-step",
                "0.01",
            ]

            result = subprocess.run(
                cmd,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            expected = {
                "demo_raw_group_main.pdf",
                "demo_raw_group_main.svg",
                "demo_raw_group_main.png",
                "demo_raw_representative_channels.pdf",
                "demo_raw_representative_channels.svg",
                "demo_raw_representative_channels.png",
            }
            actual = {path.name for path in output_dir.iterdir()}
            self.assertTrue(expected.issubset(actual))
            for name in expected:
                self.assertGreater((output_dir / name).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
