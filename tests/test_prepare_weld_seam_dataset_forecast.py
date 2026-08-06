import unittest


class ForecastWindowingTests(unittest.TestCase):
    def test_time_split_with_horizon_one(self) -> None:
        from prepare_weld_seam_dataset_forecast import (
            time_split_window_starts_forecast,
        )

        train_starts, test_starts, stats = time_split_window_starts_forecast(
            seg_len=10,
            window_size=4,
            stride=1,
            train_frac=0.5,
            purge_gap=0,
            horizon=1,
        )

        self.assertEqual(train_starts, [0])
        self.assertEqual(test_starts, [5])
        self.assertEqual(stats["total"], 6)
        self.assertEqual(stats["train"], 1)
        self.assertEqual(stats["test"], 1)
        self.assertEqual(stats["purged"], 4)

    def test_target_index_alignment(self) -> None:
        import numpy as np
        from prepare_weld_seam_dataset_forecast import (
            windows_from_segments_time_split_forecast,
        )

        seg = np.arange(10 * 2, dtype=np.float32).reshape(10, 2)
        segments = [(1, seg, 100)]
        (
            train_windows,
            test_windows,
            train_start_idx,
            test_start_idx,
            train_target_idx,
            test_target_idx,
            _stats,
        ) = windows_from_segments_time_split_forecast(
            segments=segments,
            window_size=4,
            stride=1,
            train_frac=0.5,
            purge_gap=0,
            horizon=1,
        )

        self.assertEqual(len(train_windows[1]), 1)
        self.assertEqual(len(test_windows[1]), 1)
        self.assertEqual(train_start_idx[1], [100])
        self.assertEqual(test_start_idx[1], [105])
        self.assertEqual(train_target_idx[1], [104])
        self.assertEqual(test_target_idx[1], [109])
        self.assertEqual(train_windows[1][0].shape, (4, 2))
        self.assertEqual(test_windows[1][0].shape, (4, 2))


if __name__ == "__main__":
    unittest.main()
