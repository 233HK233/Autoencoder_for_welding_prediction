import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = PROJECT_ROOT / "export_figure3_latent_attention.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class Figure3LatentAttentionExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.export_mod = load_module("export_figure3_latent_attention_test", EXPORT_SCRIPT)

    def test_attention_classifier_default_return_contract_stays_two_tuple(self) -> None:
        from models_tcn import AttentionTCNClassifier

        torch.manual_seed(123)
        model = AttentionTCNClassifier(
            input_dim=3,
            num_classes=3,
            channels=8,
            tcn_layers=1,
            attn_heads=2,
            classifier_hidden=8,
        )
        model.eval()

        result = model(torch.randn(2, 5, 3))

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        logits, z = result
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(z.shape), (2, 16))

    def test_attention_classifier_return_attention_contract_returns_per_head_weights(self) -> None:
        from models_tcn import AttentionTCNClassifier

        torch.manual_seed(123)
        model = AttentionTCNClassifier(
            input_dim=3,
            num_classes=3,
            channels=8,
            tcn_layers=1,
            attn_heads=2,
            classifier_hidden=8,
        )
        model.eval()

        logits, z, attention_weights = model(torch.randn(2, 5, 3), return_attention=True)

        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(z.shape), (2, 16))
        self.assertEqual(tuple(attention_weights.shape), (2, 2, 5, 5))

    def test_attention_weights_reduce_to_normalized_five_step_ribbon(self) -> None:
        weights = np.zeros((2, 3, 5, 5), dtype=np.float32)
        weights[:, :, :, 0] = 0.05
        weights[:, :, :, 1] = 0.10
        weights[:, :, :, 2] = 0.20
        weights[:, :, :, 3] = 0.25
        weights[:, :, :, 4] = 0.40

        ribbon = self.export_mod.reduce_attention_weights(weights)

        self.assertEqual(ribbon.shape, (2, 5))
        np.testing.assert_allclose(ribbon.sum(axis=1), np.ones(2), atol=1e-6)
        np.testing.assert_allclose(
            ribbon[0],
            np.array([0.05, 0.10, 0.20, 0.25, 0.40], dtype=np.float32),
            atol=1e-6,
        )

    def test_unified_projection_rejects_mismatched_dims_in_strict_raw_latent(self) -> None:
        latents = {
            "Teacher": np.zeros((4, 128), dtype=np.float32),
            "Student-only": np.zeros((4, 160), dtype=np.float32),
            "SEAL-Weld Student": np.zeros((4, 128), dtype=np.float32),
        }

        with self.assertRaisesRegex(ValueError, "strict_raw_latent"):
            self.export_mod.fit_unified_projection(
                latents,
                projection_method="umap",
                projection_space="strict_raw_latent",
                random_state=42,
            )

    def test_linear_cka_returns_one_for_identical_matrices(self) -> None:
        matrix = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 5.0],
                [4.0, 7.0, 11.0],
                [8.0, 13.0, 21.0],
            ],
            dtype=np.float64,
        )

        self.assertAlmostEqual(self.export_mod.linear_cka(matrix, matrix), 1.0, places=7)

    def test_boundary_distance_binning_uses_required_thresholds(self) -> None:
        self.assertEqual(self.export_mod.classify_boundary_distance_bin(5), "Near")
        self.assertEqual(self.export_mod.classify_boundary_distance_bin(6), "Mid")
        self.assertEqual(self.export_mod.classify_boundary_distance_bin(15), "Mid")
        self.assertEqual(self.export_mod.classify_boundary_distance_bin(16), "Far")


if __name__ == "__main__":
    unittest.main()
