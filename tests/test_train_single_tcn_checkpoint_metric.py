import sys
import unittest
from unittest.mock import patch


class TrainSingleTCNCheckpointMetricTests(unittest.TestCase):
    def test_default_checkpoint_metric_is_test_acc(self) -> None:
        import train_single_tcn_classifier as mod

        with patch.object(sys, "argv", ["prog", "--dataset-npz", "dummy.npz"]):
            args = mod.parse_args()

        self.assertEqual(args.checkpoint_metric, "test_acc")

    def test_alias_test_accuracy_is_normalized(self) -> None:
        import train_single_tcn_classifier as mod

        with patch.object(
            sys,
            "argv",
            ["prog", "--dataset-npz", "dummy.npz", "--checkpoint-metric", "test_accuracy"],
        ):
            args = mod.parse_args()

        self.assertEqual(args.checkpoint_metric, "test_acc")


if __name__ == "__main__":
    unittest.main()
