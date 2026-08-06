import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from ablation_experiments.scripts.models_ablation import LSTMClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
A1_SWEEP_SCRIPT = PROJECT_ROOT / "ablation_experiments/scripts/run_ablation1_sweep.py"
A2_TRAIN_SCRIPT = PROJECT_ROOT / "ablation_experiments/scripts/train_ablation2_distill_lstm_student.py"
A2_SWEEP_SCRIPT = PROJECT_ROOT / "ablation_experiments/scripts/run_ablation2_sweep.py"
A3_TRAIN_SCRIPT = (
    PROJECT_ROOT / "ablation_experiments/ablation3_joint_tcn_attn/scripts/train_ablation3_joint_tcn_attn.py"
)
A3_SWEEP_SCRIPT = (
    PROJECT_ROOT / "ablation_experiments/ablation3_joint_tcn_attn/scripts/run_ablation3_h1_sweep.py"
)
ORCHESTRATOR_SCRIPT = PROJECT_ROOT / "ablation_experiments/scripts/run_h1_ablation_target_sweep.py"
H1_DATASET = PROJECT_ROOT / "Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz"
COMPARISON_LSTM_ROOT = PROJECT_ROOT / "ablation_experiments/h1/results/comparison_18d_baselines/lstm"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class Ablation1RankingTests(unittest.TestCase):
    def test_top_params_prefers_val_macro_f1_before_test_accuracy(self) -> None:
        mod = load_module("run_ablation1_sweep_ranking_test", A1_SWEEP_SCRIPT)

        records = [
            mod.TrialRecord(
                trial_id=1,
                stage="A",
                status="success",
                run_dir="/tmp/run1",
                elapsed_sec=1.0,
                reached_target=False,
                best_epoch=4,
                best_score=0.90,
                val_macro_f1=0.95,
                test_acc=0.88,
                test_macro_f1=0.86,
                seed=14,
                lr=2e-4,
                weight_decay=2e-4,
                tcn_dropout=0.12,
                classifier_dropout=0.35,
                batch_size=128,
                tcn_layers=3,
                tcn_channels="80,80,80",
                attn_dropout=0.10,
                label_smoothing=0.0,
                weighted_sampler=True,
                epochs=80,
                checkpoint_metric="val_macro_f1",
                command="python train.py",
            ),
            mod.TrialRecord(
                trial_id=2,
                stage="A",
                status="success",
                run_dir="/tmp/run2",
                elapsed_sec=1.0,
                reached_target=True,
                best_epoch=5,
                best_score=0.91,
                val_macro_f1=0.94,
                test_acc=0.91,
                test_macro_f1=0.88,
                seed=42,
                lr=3e-4,
                weight_decay=1e-4,
                tcn_dropout=0.08,
                classifier_dropout=0.30,
                batch_size=128,
                tcn_layers=4,
                tcn_channels="96,96,96",
                attn_dropout=0.15,
                label_smoothing=0.0,
                weighted_sampler=True,
                epochs=80,
                checkpoint_metric="val_macro_f1",
                command="python train.py",
            ),
        ]

        top = mod.top_params_from_records(records, top_k=1)
        self.assertEqual(len(top), 1)
        self.assertEqual(top[0].seed, 14)
        self.assertEqual(top[0].tcn_layers, 3)


class Ablation2WarmStartTests(unittest.TestCase):
    def test_resolve_teacher_config_accepts_comparison_baseline_rnn_keys(self) -> None:
        mod = load_module("train_ablation2_distill_lstm_student_config_test", A2_TRAIN_SCRIPT)

        cfg = mod.resolve_teacher_config(
            {
                "rnn_hidden_size": 64,
                "rnn_layers": 2,
                "rnn_dropout": 0.3,
                "rnn_bidirectional": True,
                "classifier_hidden": 64,
                "classifier_dropout": 0.35,
            }
        )

        self.assertEqual(cfg["lstm_hidden"], 64)
        self.assertEqual(cfg["lstm_layers"], 2)
        self.assertEqual(cfg["lstm_dropout"], 0.3)
        self.assertTrue(cfg["lstm_bidirectional"])
        self.assertEqual(cfg["classifier_hidden"], 64)
        self.assertEqual(cfg["classifier_dropout"], 0.35)

    def test_shape_safe_student_init_loads_matching_weights_only(self) -> None:
        mod = load_module("train_ablation2_distill_lstm_student_init_test", A2_TRAIN_SCRIPT)

        source = LSTMClassifier(
            input_dim=18,
            num_classes=3,
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            classifier_hidden=64,
            classifier_dropout=0.35,
        )
        target = LSTMClassifier(
            input_dim=13,
            num_classes=3,
            hidden_size=64,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            classifier_hidden=64,
            classifier_dropout=0.35,
        )

        for _, param in source.named_parameters():
            torch.nn.init.constant_(param, 0.25)

        result = mod.load_model_init_weights(
            target,
            source.state_dict(),
            init_mode="shape_safe",
        )

        self.assertGreater(result["loaded_keys"], 0)
        self.assertIn("lstm.weight_ih_l0", result["skipped_keys"])
        self.assertTrue(torch.allclose(target.classifier[1].weight, source.classifier[1].weight))
        self.assertFalse(torch.allclose(target.lstm.weight_ih_l0, source.lstm.weight_ih_l0[:, :13]))

    def test_a2_sweep_defaults_to_h1_comparison_lstm_and_strict_outputs(self) -> None:
        mod = load_module("run_ablation2_sweep_defaults_test", A2_SWEEP_SCRIPT)

        with mock.patch.object(sys, "argv", ["run_ablation2_sweep.py"]):
            args = mod.parse_args()

        self.assertEqual(args.dataset_npz, H1_DATASET)
        self.assertEqual(args.teacher_results_dir, COMPARISON_LSTM_ROOT)
        self.assertEqual(
            args.output_dir,
            PROJECT_ROOT
            / "ablation_experiments/h1/results/ablation2_teacher_student_lstm/students/ablation2_strict_valf1_20260522",
        )
        self.assertEqual(
            args.report_dir,
            PROJECT_ROOT
            / "ablation_experiments/h1/reports/ablation2_teacher_student_lstm/ablation2_strict_valf1_20260522",
        )
        self.assertEqual(args.student_init_source, "teacher")
        self.assertEqual(args.checkpoint_metric, "val_macro_f1")
        self.assertEqual(args.max_runs, 56)

    def test_a2_stage_builders_match_expected_budgets(self) -> None:
        mod = load_module("run_ablation2_sweep_stage_builder_test", A2_SWEEP_SCRIPT)

        with mock.patch.object(sys, "argv", ["run_ablation2_sweep.py"]):
            args = mod.parse_args()

        stage0 = mod.build_stage0_jobs(args)
        self.assertEqual(len(stage0), 8)

        stage1 = mod.build_stage1_jobs(args)
        self.assertEqual(len(stage1), 32)

        ranked = [
            {
                **job,
                "val_macro_f1": 0.95 - idx * 0.01,
                "test_acc": 0.9 - idx * 0.01,
                "test_macro_f1": 0.88 - idx * 0.01,
                "test_teacher_agreement": 0.96 - idx * 0.01,
            }
            for idx, job in enumerate(stage1[:4])
        ]
        stage2 = mod.build_stage2_jobs(args, ranked)
        self.assertEqual(len(stage2), 8)
        self.assertTrue(
            all(
                str(job["job_output_root"]).startswith(str(args.output_dir / "stage2"))
                for job in stage2
            )
        )

        stage3 = mod.build_stage3_jobs(args, ranked)
        self.assertEqual(len(stage3), 8)
        self.assertTrue(
            all(
                str(job["job_output_root"]).startswith(str(args.output_dir / "stage3"))
                for job in stage3
            )
        )

    def test_a2_rank_records_prefers_val_macro_f1(self) -> None:
        mod = load_module("run_ablation2_sweep_ranking_test", A2_SWEEP_SCRIPT)

        ranked = mod.rank_records(
            [
                {
                    "config_id": "a",
                    "status": "success",
                    "val_macro_f1": 0.94,
                    "test_acc": 0.98,
                    "test_macro_f1": 0.95,
                    "test_teacher_agreement": 0.90,
                },
                {
                    "config_id": "b",
                    "status": "success",
                    "val_macro_f1": 0.96,
                    "test_acc": 0.91,
                    "test_macro_f1": 0.92,
                    "test_teacher_agreement": 0.94,
                },
            ]
        )

        self.assertEqual([row["config_id"] for row in ranked], ["b", "a"])

    def test_a2_build_command_wires_teacher_ckpt_as_student_init_by_default(self) -> None:
        mod = load_module("run_ablation2_sweep_command_init_test", A2_SWEEP_SCRIPT)

        with mock.patch.object(
            sys,
            "argv",
            [
                "run_ablation2_sweep.py",
                "--output-dir",
                str(PROJECT_ROOT / "tmp_results"),
                "--report-dir",
                str(PROJECT_ROOT / "tmp_reports"),
            ],
        ):
            args = mod.parse_args()

        job = mod.make_job(
            args,
            stage="stage1",
            config_id="cfg",
            base_config_id=None,
            loss_cfg={"temperature": 2.5, "lambda_ce": 0.9, "lambda_kd": 1.1, "lambda_feat": 0.0},
            opt_cfg={"lr": 1e-4, "weight_decay": 1e-5, "weighted_sampler": True},
            capacity_cfg={
                "student_lstm_layers": 2,
                "student_lstm_dropout": 0.3,
                "student_classifier_hidden": 64,
                "student_classifier_dropout": 0.35,
            },
            seed=42,
        )

        cmd = mod.build_command(
            args,
            job,
            teacher_ckpt=Path("/tmp/teacher.pth"),
            teacher_run_args=Path("/tmp/teacher_run_args.json"),
        )
        joined = " ".join(cmd)
        self.assertIn("--student-init-ckpt /tmp/teacher.pth", joined)
        self.assertIn("--student-init-mode shape_safe", joined)


class Ablation3WarmStartTests(unittest.TestCase):
    def test_train_script_exposes_warm_start_and_freeze_controls(self) -> None:
        mod = load_module("train_ablation3_joint_warm_start_args_test", A3_TRAIN_SCRIPT)

        with mock.patch.object(sys, "argv", ["train_ablation3_joint_tcn_attn.py"]):
            args = mod.parse_args()

        self.assertEqual(args.dataset_npz, H1_DATASET)
        self.assertEqual(args.checkpoint_metric, "test_acc")
        self.assertEqual(args.freeze_teacher_epochs, 0)
        self.assertEqual(args.teacher_init_mode, "strict")
        self.assertEqual(args.student_init_mode, "shape_safe")

    def test_train_script_preserves_teacher_width_from_explicit_channel_string(self) -> None:
        mod = load_module("train_ablation3_joint_width_preserve_test", A3_TRAIN_SCRIPT)

        with mock.patch.object(
            sys,
            "argv",
            [
                "train_ablation3_joint_tcn_attn.py",
                "--teacher-tcn-layers",
                "3",
                "--teacher-tcn-channels",
                "64,64,64",
            ],
        ):
            args = mod.parse_args()

        teacher_cfg = mod.build_teacher_cfg(args)
        student_cfg = mod.build_student_cfg(args, teacher_cfg)

        self.assertEqual(teacher_cfg["tcn_channels"], 64)
        self.assertEqual(student_cfg["tcn_channels"], 64)

    def test_a3_h1_sweep_defaults_to_h1_roots(self) -> None:
        mod = load_module("run_ablation3_h1_sweep_defaults_test", A3_SWEEP_SCRIPT)

        with mock.patch.object(sys, "argv", ["run_ablation3_h1_sweep.py"]):
            args = mod.parse_args()

        self.assertEqual(args.dataset_npz, H1_DATASET)
        self.assertEqual(
            args.output_dir,
            PROJECT_ROOT / "ablation_experiments/h1/results/ablation3_joint_tcn_attn",
        )
        self.assertEqual(
            args.report_dir,
            PROJECT_ROOT / "ablation_experiments/h1/reports/ablation3_joint_tcn_attn",
        )
        self.assertEqual(
            args.student_results_dir,
            PROJECT_ROOT / "ablation_experiments/h1/results/ablation1_sweep/trials",
        )
        self.assertEqual(args.max_runs, 40)

    def test_a3_stage_builders_match_expected_budgets(self) -> None:
        mod = load_module("run_ablation3_h1_sweep_stage_builder_test", A3_SWEEP_SCRIPT)

        with mock.patch.object(sys, "argv", ["run_ablation3_h1_sweep.py"]):
            args = mod.parse_args()

        stage0 = mod.build_stage0_jobs(args)
        self.assertEqual(len(stage0), 6)

        stage1 = mod.build_stage1_jobs(args)
        self.assertEqual(len(stage1), 18)

        ranked = [{**job, "test_acc": 0.95 - idx * 0.01} for idx, job in enumerate(stage1[:4])]
        stage2 = mod.build_stage2_jobs(args, ranked)
        self.assertEqual(len(stage2), 8)

        stage3 = mod.build_stage3_jobs(args, ranked)
        self.assertEqual(len(stage3), 8)

    def test_resolve_best_a1_student_recurses_into_trial_run_dirs(self) -> None:
        mod = load_module("run_ablation3_h1_sweep_recursive_a1_test", A3_SWEEP_SCRIPT)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weak_run = root / "trial_0001_A" / "run_weak"
            weak_run.mkdir(parents=True)
            (weak_run / "best_student_only.pth").write_bytes(b"stub")
            (weak_run / "run_args.json").write_text("{}", encoding="utf-8")
            (weak_run / "evaluation_metrics.txt").write_text(
                "--- Test Metrics ---\nAccuracy: 88.00%\nMacro-F1: 0.8800\n",
                encoding="utf-8",
            )

            best_run = root / "trial_0002_A" / "nested" / "run_best"
            best_run.mkdir(parents=True)
            (best_run / "best_student_only.pth").write_bytes(b"stub")
            (best_run / "run_args.json").write_text("{}", encoding="utf-8")
            (best_run / "evaluation_metrics.txt").write_text(
                "--- Test Metrics ---\nAccuracy: 93.00%\nMacro-F1: 0.9300\n",
                encoding="utf-8",
            )

            result = mod.resolve_best_a1_student(root)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(Path(result["student_run_dir"]).name, "run_best")
        self.assertAlmostEqual(float(result["test_acc"]), 0.93)


class AblationTargetOrchestratorTests(unittest.TestCase):
    def test_orchestrator_defaults_to_h1_inputs_and_reports(self) -> None:
        mod = load_module("run_h1_ablation_target_sweep_defaults_test", ORCHESTRATOR_SCRIPT)

        with mock.patch.object(sys, "argv", ["run_h1_ablation_target_sweep.py"]):
            args = mod.parse_args()

        self.assertEqual(args.dataset_npz, H1_DATASET)
        self.assertEqual(args.target_acc, 90.0)
        self.assertEqual(
            args.report_dir,
            PROJECT_ROOT / "ablation_experiments/h1/reports",
        )

    def test_orchestrator_passes_a1_trial_root_into_a3_sweep(self) -> None:
        mod = load_module("run_h1_ablation_target_sweep_wiring_test", ORCHESTRATOR_SCRIPT)

        captured: list[list[str]] = []

        def _capture(cmd: list[str], print_command: bool) -> None:
            del print_command
            captured.append(cmd)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with mock.patch.object(
                sys,
                "argv",
                [
                    "run_h1_ablation_target_sweep.py",
                    "--results-root",
                    str(tmp_path / "results"),
                    "--report-dir",
                    str(tmp_path / "reports"),
                ],
            ):
                args = mod.parse_args()
            with mock.patch.object(mod, "run_cmd", side_effect=_capture):
                with mock.patch.object(mod, "parse_args", return_value=args):
                    mod.main()

        self.assertEqual(len(captured), 4)
        a3_cmd = next(cmd for cmd in captured if str(A3_SWEEP_SCRIPT) in cmd)
        self.assertIn("--student-results-dir", a3_cmd)
        self.assertIn(str((tmp_path / "results" / "ablation1_sweep" / "trials")), a3_cmd)


if __name__ == "__main__":
    unittest.main()
