import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ablation_experiments.scripts.models_ablation import LSTMClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "ablation_experiments/scripts/run_h1_representation_analysis.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class H1RepresentationAnalysisHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = load_module("run_h1_representation_analysis_test", SCRIPT_PATH)

    def test_projection_perplexity_follows_spec(self) -> None:
        self.assertEqual(self.mod.resolve_projection_perplexity(356), 30)
        self.assertEqual(self.mod.resolve_projection_perplexity(20), 6)
        self.assertEqual(self.mod.resolve_projection_perplexity(5), 1)

    def test_representation_metrics_reward_separated_clusters(self) -> None:
        z = np.array(
            [
                [-5.0, -5.0],
                [-5.2, -4.8],
                [0.0, 0.0],
                [0.2, 0.1],
                [5.0, 5.0],
                [5.2, 4.9],
            ],
            dtype=np.float32,
        )
        y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

        metrics = self.mod.compute_representation_metrics(z, y_true)

        self.assertGreater(metrics["silhouette_score"], 0.9)
        self.assertLess(metrics["davies_bouldin_index"], 0.2)
        self.assertGreater(metrics["calinski_harabasz_score"], 100.0)
        self.assertLess(metrics["mean_intra_class_distance"], 0.3)
        self.assertGreater(metrics["mean_inter_class_centroid_distance"], 6.0)
        self.assertGreater(metrics["inter_intra_distance_ratio"], 20.0)

    def test_write_embedding_file_persists_required_fields(self) -> None:
        payload = {
            "Z_test": np.ones((4, 3), dtype=np.float32),
            "logits": np.zeros((4, 3), dtype=np.float32),
            "prob": np.full((4, 3), 1.0 / 3.0, dtype=np.float32),
            "y_true": np.array([0, 1, 2, 1], dtype=np.int64),
            "y_pred": np.array([0, 1, 2, 0], dtype=np.int64),
            "sample_index": np.arange(4, dtype=np.int64),
            "seam_id": np.array([0, 0, 1, 1], dtype=np.int64),
            "seam_name": np.array(["a01", "a01", "b01", "b01"]),
            "start_idx": np.array([10, 20, 30, 40], dtype=np.int64),
            "target_idx": np.array([15, 25, 35, 45], dtype=np.int64),
            "model_name": np.array("Teacher"),
            "run_path": np.array("/tmp/demo_run"),
            "checkpoint_path": np.array("/tmp/demo_run/best_single_tcn.pth"),
            "dataset_npz": np.array("/tmp/demo_h1.npz"),
            "feature_layer": np.array("attention_tcn.z"),
        }

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "teacher_embeddings_test.npz"
            self.mod.write_embedding_file(out_path, payload)

            with np.load(out_path, allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), set(payload.keys()))
                self.assertEqual(saved["Z_test"].shape, (4, 3))
                self.assertEqual(saved["logits"].shape, (4, 3))
                self.assertEqual(saved["prob"].shape, (4, 3))
                self.assertEqual(saved["model_name"].item(), "Teacher")
                self.assertEqual(saved["feature_layer"].item(), "attention_tcn.z")


class LSTMFeatureExtractionTests(unittest.TestCase):
    def test_forward_features_returns_post_fc1_representation(self) -> None:
        model = LSTMClassifier(
            input_dim=18,
            num_classes=3,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            bidirectional=True,
            classifier_hidden=5,
            classifier_dropout=0.0,
        )
        x = torch.randn(2, 5, 18)

        logits, features = model.forward_features(x)

        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(features.shape), (2, 5))
        self.assertTrue(torch.all(features >= 0.0))


if __name__ == "__main__":
    unittest.main()
