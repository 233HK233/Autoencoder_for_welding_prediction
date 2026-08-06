import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class InferWithMonotonicPostprocessTests(unittest.TestCase):
    def test_parse_args_accepts_raw_decode_alias(self) -> None:
        import infer_with_monotonic_postprocess as mod

        with patch.object(sys, "argv", ["prog", "--run-dir", "demo", "--decode", "raw"]):
            args = mod.parse_args()

        self.assertEqual(args.decode, "raw")

    def test_load_run_config_from_run_dir_resolves_checkpoint_and_model_args(self) -> None:
        import infer_with_monotonic_postprocess as mod

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "demo_run"
            run_dir.mkdir(parents=True, exist_ok=True)

            payload = {
                "dataset_npz": "Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz",
                "tcn_layers": 3,
                "tcn_kernel": 3,
                "tcn_dropout": 0.08,
                "tcn_dilation_base": 2,
                "latent_dim": 64,
                "attn_heads": 4,
                "attn_dropout": 0.1,
                "attn_ff_dim": 128,
                "classifier_hidden": 64,
                "classifier_dropout": 0.4,
            }
            (run_dir / "run_args.json").write_text(json.dumps(payload), encoding="utf-8")
            (run_dir / "best_single_tcn.pth").write_bytes(b"fake")

            resolved = mod.load_run_config_from_run_dir(run_dir)

            self.assertEqual(resolved["dataset_npz"], payload["dataset_npz"])
            self.assertEqual(resolved["checkpoints"], [str(run_dir / "best_single_tcn.pth")])
            self.assertEqual(resolved["tcn_layers"], 3)
            self.assertEqual(resolved["tcn_kernel"], 3)
            self.assertEqual(resolved["tcn_dropout"], 0.08)
            self.assertEqual(resolved["tcn_dilation_base"], 2)
            self.assertEqual(resolved["channels"], 64)
            self.assertEqual(resolved["attn_heads"], 4)
            self.assertEqual(resolved["attn_dropout"], 0.1)
            self.assertEqual(resolved["attn_ff_dim"], 128)
            self.assertEqual(resolved["classifier_hidden"], 64)
            self.assertEqual(resolved["classifier_dropout"], 0.4)

    def test_save_prediction_details_writes_expected_columns(self) -> None:
        import infer_with_monotonic_postprocess as mod

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "predictions.csv"
            seam_name_order = np.array(["a01", "b01"], dtype=object)
            seam_ids = np.array([0, 1], dtype=np.int64)
            start_idx = np.array([10, 20], dtype=np.int64)
            target_idx = np.array([14, 24], dtype=np.int64)
            y_true = np.array([1, 2], dtype=np.int64)
            raw_pred = np.array([1, 1], dtype=np.int64)
            final_pred = np.array([1, 2], dtype=np.int64)
            probabilities = np.array(
                [
                    [0.1, 0.8, 0.1],
                    [0.2, 0.6, 0.2],
                ],
                dtype=np.float64,
            )

            mod.save_prediction_details_csv(
                path=out_path,
                seam_name_order=seam_name_order,
                seam_ids=seam_ids,
                start_idx=start_idx,
                target_idx=target_idx,
                y_true=y_true,
                raw_pred=raw_pred,
                final_pred=final_pred,
                probabilities=probabilities,
                decode_mode="raw",
            )

            with out_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 2)
            self.assertEqual(
                rows[0].keys(),
                {
                    "sample_index",
                    "seam_id",
                    "seam_name",
                    "start_idx",
                    "target_idx",
                    "y_true",
                    "raw_pred",
                    "final_pred",
                    "decode_mode",
                    "prob_class_0",
                    "prob_class_1",
                    "prob_class_2",
                },
            )
            self.assertEqual(rows[0]["seam_name"], "a01")
            self.assertEqual(rows[0]["decode_mode"], "raw")
            self.assertEqual(rows[1]["target_idx"], "24")
            self.assertEqual(rows[1]["final_pred"], "2")
            self.assertEqual(rows[1]["prob_class_1"], "0.600000")


if __name__ == "__main__":
    unittest.main()
