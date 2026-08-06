import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "ablation_experiments/scripts/run_h1_distill_comparison_sweep.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class H1DistillSweepPlannerTests(unittest.TestCase):
    def test_parse_args_defaults_to_strict_protocol_controls(self) -> None:
        mod = load_module("run_h1_distill_comparison_sweep_plan_defaults", SCRIPT_PATH)

        with mock.patch.object(sys, "argv", ["run_h1_distill_comparison_sweep.py"]):
            args = mod.parse_args()

        self.assertEqual(args.experiment_tag, "student_h1_strict_valf1_20260522")
        self.assertEqual(
            args.teacher_manifest,
            PROJECT_ROOT / "outputs/teacher_h1_forecast_95/manifests/best_teacher_h1.json",
        )
        self.assertEqual(
            args.output_dir,
            PROJECT_ROOT
            / "ablation_experiments/h1/results/comparison_18d_baselines/teacher_student_reference/student_h1_strict_valf1_20260522",
        )
        self.assertEqual(
            args.report_dir,
            PROJECT_ROOT
            / "ablation_experiments/h1/reports/comparison_18d_baselines/teacher_student_reference/student_h1_strict_valf1_20260522",
        )
        self.assertEqual(args.gpu_list, "1,0")
        self.assertEqual(args.ranking_metric, "val_macro_f1")
        self.assertAlmostEqual(args.target_test_acc, 0.96)
        self.assertEqual(args.max_runs, 108)

    def test_stage_builders_match_expected_budgets_and_are_unique(self) -> None:
        mod = load_module("run_h1_distill_comparison_sweep_stage_builders", SCRIPT_PATH)

        with mock.patch.object(sys, "argv", ["run_h1_distill_comparison_sweep.py"]):
            args = mod.parse_args()

        stage0 = mod.build_stage0_jobs(args)
        self.assertEqual(len(stage0), 4)
        self.assertTrue(all(job["checkpoint_metric"] == "val_macro_f1" for job in stage0))

        checkpoint_metric = mod.select_stage0_checkpoint_metric(stage0)
        self.assertEqual(checkpoint_metric, "val_macro_f1")

        stage1 = mod.build_stage1_jobs(args, checkpoint_metric)
        self.assertEqual(len(stage1), 16)

        ranked_stage1 = [
            {
                **job,
                "val_macro_f1": 0.97 - idx * 0.01,
                "test_acc": 0.9 - idx * 0.01,
                "test_macro_f1": 0.88 - idx * 0.01,
                "test_teacher_agreement": 0.95,
            }
            for idx, job in enumerate(stage1[:3])
        ]
        stage2 = mod.build_stage2_jobs(args, ranked_stage1)
        self.assertEqual(len(stage2), 24)

        ranked_stage2 = [
            {
                **job,
                "val_macro_f1": 0.97 - idx * 0.01,
                "test_acc": 0.9 - idx * 0.01,
                "test_macro_f1": 0.88 - idx * 0.01,
                "test_teacher_agreement": 0.95,
            }
            for idx, job in enumerate(stage2[:4])
        ]
        stage3 = mod.build_stage3_jobs(args, ranked_stage2)
        self.assertEqual(len(stage3), 20)

        ranked_stage3 = [
            {
                **job,
                "val_macro_f1": 0.97 - idx * 0.005,
                "test_acc": 0.9 - idx * 0.01,
                "test_macro_f1": 0.88 - idx * 0.01,
                "test_teacher_agreement": 0.95,
            }
            for idx, job in enumerate(stage3[:4])
        ]
        stage4 = mod.build_seed_harvest_jobs(args, ranked_stage3, stage_name="stage4", seeds=[7, 21, 77, 122, 183])
        self.assertEqual(len(stage4), 20)

        extension = mod.build_seed_harvest_jobs(
            args,
            ranked_stage3,
            stage_name="extension",
            seeds=[132, 230, 314, 512, 777, 1001],
        )
        self.assertEqual(len(extension), 24)

        all_jobs = stage0 + stage1 + stage2 + stage3 + stage4 + extension
        unique_locations = {
            (job["stage"], job["config_id"], job["seed"], str(job["job_output_root"])) for job in all_jobs
        }
        self.assertEqual(len(unique_locations), len(all_jobs))
        self.assertEqual(len(stage0 + stage1 + stage2 + stage3 + stage4), 84)
        self.assertEqual(len(all_jobs), 108)

    def test_rank_records_prefers_val_macro_f1_then_test_f1_then_accuracy(self) -> None:
        mod = load_module("run_h1_distill_comparison_sweep_ranking", SCRIPT_PATH)

        records = [
            {
                "config_id": "a",
                "val_macro_f1": 0.95,
                "test_acc": 0.94,
                "test_macro_f1": 0.91,
                "test_teacher_agreement": 0.94,
            },
            {
                "config_id": "b",
                "val_macro_f1": 0.95,
                "test_acc": 0.93,
                "test_macro_f1": 0.92,
                "test_teacher_agreement": 0.90,
            },
            {
                "config_id": "c",
                "val_macro_f1": 0.94,
                "test_acc": 0.99,
                "test_macro_f1": 0.97,
                "test_teacher_agreement": 0.96,
            },
        ]

        ranked = mod.rank_records(records, ranking_metric="val_macro_f1")
        self.assertEqual([row["config_id"] for row in ranked], ["b", "a", "c"])

    def test_build_command_includes_checkpoint_metric_and_student_overrides(self) -> None:
        mod = load_module("run_h1_distill_comparison_sweep_command", SCRIPT_PATH)

        with mock.patch.object(
            sys,
            "argv",
            [
                "run_h1_distill_comparison_sweep.py",
                "--output-dir",
                str(PROJECT_ROOT / "tmp_results"),
                "--report-dir",
                str(PROJECT_ROOT / "tmp_reports"),
            ],
        ):
            args = mod.parse_args()

        job = {
            "stage": "stage2",
            "config_id": "s2_cfg_01",
            "base_config_id": "s1_cfg_01",
            "seed": 14,
            "epochs": 80,
            "batch_size": 128,
            "temperature": 3.0,
            "lambda_ce": 0.8,
            "lambda_kd": 1.2,
            "lambda_feat": 0.2,
            "lr": 2.5e-4,
            "weight_decay": 1.5e-4,
            "checkpoint_metric": "val_macro_f1",
            "job_output_root": Path(args.output_dir) / args.experiment_tag / "stage2" / "s2_cfg_01",
            "student_tcn_dropout": 0.05,
            "student_tcn_layers": 2,
            "student_tcn_channels": 48,
            "student_classifier_hidden": 128,
            "student_classifier_dropout": 0.25,
            "student_attn_dropout": 0.05,
            "student_attn_ff_dim": 192,
            "student_tcn_dilation_base": 1,
        }

        cmd = mod.build_command(
            args=args,
            job=job,
            teacher_ckpt=Path("/tmp/teacher.pth"),
            teacher_run_args=Path("/tmp/run_args.json"),
        )

        joined = " ".join(cmd)
        self.assertIn("--checkpoint-metric val_macro_f1", joined)
        self.assertIn("--weight-decay 0.00015", joined)
        self.assertIn("--student-tcn-layers 2", joined)
        self.assertIn("--student-tcn-channels 48", joined)
        self.assertIn("--student-classifier-hidden 128", joined)
        self.assertIn("--student-classifier-dropout 0.25", joined)
        self.assertIn("--student-attn-dropout 0.05", joined)
        self.assertIn("--student-attn-ff-dim 192", joined)
        self.assertIn("--student-tcn-dilation-base 1", joined)
        self.assertIn(str(job["job_output_root"]), joined)

    def test_execute_stage_skips_invalid_feature_alignment_channels(self) -> None:
        mod = load_module("run_h1_distill_comparison_sweep_invalid_channels", SCRIPT_PATH)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            teacher_run_args = tmp_path / "teacher_run_args.json"
            teacher_run_args.write_text(
                json.dumps(
                    {
                        "model": "tcn_attn",
                        "latent_dim": 64,
                        "tcn_layers": 3,
                        "tcn_channels": None,
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(
                sys,
                "argv",
                [
                    "run_h1_distill_comparison_sweep.py",
                    "--output-dir",
                    str(tmp_path / "results"),
                    "--report-dir",
                    str(tmp_path / "reports"),
                ],
            ):
                args = mod.parse_args()

            job = mod.create_job(
                args,
                stage="stage2",
                config_id="invalid_width",
                base_config_id="invalid_width",
                checkpoint_metric="val_macro_f1",
                seed=14,
                temperature=3.0,
                lambda_ce=0.8,
                lambda_kd=1.2,
                lambda_feat=0.2,
                lr=2e-4,
                weight_decay=1.5e-4,
                student_tcn_channels=48,
            )

            records = []
            with mock.patch.object(mod.subprocess, "run", side_effect=AssertionError("should not launch")):
                stage_records = mod.execute_stage_jobs(
                    args,
                    [job],
                    teacher_ckpt=tmp_path / "teacher.pth",
                    teacher_run_args=teacher_run_args,
                    records=records,
                )

            self.assertEqual(len(stage_records), 1)
            self.assertEqual(stage_records[0]["status"], "invalid_feature_alignment")
            self.assertEqual(stage_records[0]["test_acc"], -1.0)


class H1DistillSweepMainSmokeTests(unittest.TestCase):
    def test_main_writes_summary_with_target_fields(self) -> None:
        mod = load_module("run_h1_distill_comparison_sweep_main_smoke", SCRIPT_PATH)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            teacher_dir = tmp_path / "teacher"
            teacher_dir.mkdir(parents=True)
            teacher_ckpt = teacher_dir / "best_single_tcn.pth"
            teacher_ckpt.write_bytes(b"fake")
            teacher_run_args = teacher_dir / "run_args.json"
            teacher_run_args.write_text(json.dumps({"model": "tcn_attn"}), encoding="utf-8")
            teacher_manifest = tmp_path / "teacher_manifest.json"
            teacher_manifest.write_text(
                json.dumps(
                    {
                        "best_run": {
                            "path": str(teacher_dir),
                            "run_args_path": str(teacher_run_args),
                        }
                    }
                ),
                encoding="utf-8",
            )

            output_root = tmp_path / "results"
            report_root = tmp_path / "reports"

            def fake_execute_stage(args, jobs, teacher_ckpt, teacher_run_args, records):
                generated = []
                for idx, job in enumerate(jobs, start=1):
                    generated.append(
                        {
                            **job,
                            "status": "success",
                            "stage_index": idx,
                            "run_name": f"{job['config_id']}_seed{job['seed']}",
                            "run_dir": str(job["job_output_root"] / f"{job['config_id']}_seed{job['seed']}"),
                            "elapsed_sec": 0.1,
                            "gpu_id": "0",
                            "val_macro_f1": 0.965 if idx == 1 else 0.91,
                            "test_acc": 0.965 if idx == 1 else 0.91,
                            "test_macro_f1": 0.94 if idx == 1 else 0.88,
                            "test_teacher_agreement": 0.97 if idx == 1 else 0.9,
                            "best_epoch": 2,
                            "best_score": 0.965 if idx == 1 else 0.91,
                            "student_config_resolved": {"tcn_layers": 3},
                            "teacher_manifest": str(args.teacher_manifest),
                            "command": "python fake_train.py",
                            "log_path": str(report_root / args.experiment_tag / "logs" / f"{job['config_id']}.log"),
                        }
                    )
                records.extend(generated)
                return generated

            argv = [
                "run_h1_distill_comparison_sweep.py",
                "--teacher-manifest",
                str(teacher_manifest),
                "--output-dir",
                str(output_root),
                "--report-dir",
                str(report_root),
                "--max-runs",
                "4",
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(mod, "validate_forecast_dataset_contract"):
                    with mock.patch.object(mod, "execute_stage_jobs", side_effect=fake_execute_stage):
                        mod.main()

            summary_path = report_root / "distill_sweep_summary.json"
            self.assertTrue(summary_path.exists())
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["num_runs"], 4)
            self.assertTrue(payload["target_met"])
            self.assertEqual(payload["ranking_metric"], "val_macro_f1")
            self.assertAlmostEqual(payload["target_test_acc"], 0.96)
            self.assertEqual(payload["best_run"]["test_acc"], 0.965)


if __name__ == "__main__":
    unittest.main()
