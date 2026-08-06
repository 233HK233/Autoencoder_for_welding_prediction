import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "train_distill_single_tcn_student.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TrainDistillSingleTCNStudentTests(unittest.TestCase):
    def test_parse_args_defaults_to_attention_student(self) -> None:
        mod = load_module("train_distill_student_args_default", TRAIN_SCRIPT)

        with mock.patch.object(
            sys,
            "argv",
            [
                "train_distill_single_tcn_student.py",
                "--dataset-npz",
                "dataset.npz",
                "--teacher-ckpt",
                "teacher.pth",
                "--teacher-run-args",
                "run_args.json",
            ],
        ):
            args = mod.parse_args()

        self.assertEqual(args.student_architecture, "tcn_attn")

    def test_build_student_model_can_remove_attention_while_preserving_latent_shape(self) -> None:
        mod = load_module("train_distill_student_architecture", TRAIN_SCRIPT)

        cfg = {
            "tcn_channels": 16,
            "tcn_layers": 2,
            "tcn_kernel": 3,
            "tcn_dropout": 0.05,
            "tcn_dilation_base": 2,
            "attn_heads": 4,
            "attn_dropout": 0.1,
            "attn_ff_dim": 32,
            "classifier_hidden": 24,
            "classifier_dropout": 0.2,
        }

        student = mod.build_student_model(
            input_dim=13,
            num_classes=3,
            cfg=cfg,
            architecture="tcn_no_attn",
        )
        logits, z = student(torch.randn(5, 7, 13))

        self.assertFalse(hasattr(student, "attn"))
        self.assertEqual(tuple(logits.shape), (5, 3))
        self.assertEqual(tuple(z.shape), (5, 32))

    def test_build_test_prediction_rows_exports_required_columns_and_seam_names(self) -> None:
        mod = load_module("train_distill_student_prediction_rows", TRAIN_SCRIPT)

        ds = {
            "y_test": np.array([2, 1], dtype=np.int64),
            "seam_id_test": np.array([3, 1], dtype=np.int64),
            "start_idx_test": np.array([208, 496], dtype=np.int64),
            "target_idx_test": np.array([213, 501], dtype=np.int64),
            "seam_name_order": np.array(["a01", "b01", "c01", "c02"]),
        }
        rows = mod.build_test_prediction_rows(
            ds=ds,
            y_pred=np.array([2, 0], dtype=np.int64),
            probabilities=np.array(
                [
                    [0.01, 0.04, 0.95],
                    [0.51, 0.44, 0.05],
                ],
                dtype=np.float32,
            ),
        )

        self.assertEqual(
            list(rows[0].keys()),
            [
                "sample_index",
                "seam_id",
                "seam_name",
                "start_idx",
                "target_idx",
                "y_true",
                "y_pred",
                "prob_class_0",
                "prob_class_1",
                "prob_class_2",
            ],
        )
        self.assertEqual(rows[0]["sample_index"], 0)
        self.assertEqual(rows[0]["seam_id"], 3)
        self.assertEqual(rows[0]["seam_name"], "c02")
        self.assertEqual(rows[1]["y_true"], 1)
        self.assertEqual(rows[1]["y_pred"], 0)
        self.assertAlmostEqual(rows[1]["prob_class_0"], 0.51, places=6)


if __name__ == "__main__":
    unittest.main()
