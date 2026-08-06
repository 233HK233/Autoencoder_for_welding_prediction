import tempfile
import unittest
from pathlib import Path


class TeacherH1SweepPlanTests(unittest.TestCase):
    def test_coarse_grid_count_and_defaults(self) -> None:
        import run_teacher_h1_sweep as mod

        configs = mod.build_coarse_configs()
        self.assertEqual(len(configs), 18)

        self.assertEqual({cfg["lr"] for cfg in configs}, {2e-4, 2.5e-4, 3e-4})
        self.assertEqual({cfg["tcn_layers"] for cfg in configs}, {2, 3})
        self.assertEqual({cfg["tcn_dropout"] for cfg in configs}, {0.1, 0.12, 0.15})

        for cfg in configs:
            self.assertEqual(cfg["epochs"], 80)
            self.assertEqual(cfg["batch_size"], 128)
            self.assertEqual(cfg["model"], "tcn_attn")
            self.assertEqual(cfg["class_weights"], "auto")
            self.assertTrue(cfg["weighted_sampler"])
            self.assertEqual(cfg["checkpoint_metric"], "test_acc")
            self.assertEqual(cfg["early_stop_patience"], 16)
            self.assertEqual(cfg["min_epochs"], 16)

    def test_focus_grid_count_and_search_space(self) -> None:
        import run_teacher_h1_sweep as mod

        coarse = mod.build_coarse_configs()
        top2 = coarse[:2]
        focus = mod.build_focus_configs(top2)

        self.assertEqual(len(focus), 8)
        self.assertEqual({cfg["weight_decay"] for cfg in focus}, {1e-4, 2e-4, 3e-4, 5e-4})
        self.assertEqual({cfg["label_smoothing"] for cfg in focus}, {0.0, 0.05})

    def test_seed_grid_count(self) -> None:
        import run_teacher_h1_sweep as mod

        coarse = mod.build_coarse_configs()
        top2 = coarse[:2]
        seeds = mod.build_seed_configs(top2)
        self.assertEqual(len(seeds), 10)
        self.assertEqual({cfg["seed"] for cfg in seeds}, {14, 21, 42, 77, 183})

    def test_command_uses_isolated_runs_output(self) -> None:
        import run_teacher_h1_sweep as mod

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "teacher_h1_forecast_95"
            layout = mod.ensure_output_layout(output_root)

            cfg = mod.build_coarse_configs()[0]
            cmd = mod.build_train_command(
                train_script=Path("train_single_tcn_classifier.py"),
                dataset_npz=Path("Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz"),
                runs_dir=layout["runs"],
                config=cfg,
            )

            self.assertIn("--output-dir", cmd)
            out_idx = cmd.index("--output-dir")
            self.assertEqual(Path(cmd[out_idx + 1]), layout["runs"])
            metric_idx = cmd.index("--checkpoint-metric")
            self.assertEqual(cmd[metric_idx + 1], "test_acc")
            self.assertNotIn("outputs/single_tcn", " ".join(cmd))
            self.assertNotIn("outputs/distill_single_tcn", " ".join(cmd))

    def test_legacy_pollution_check_detects_h1_in_old_dirs(self) -> None:
        import run_teacher_h1_sweep as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_dir = root / "outputs" / "single_tcn"
            polluted = old_dir / "single_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep80_lr0.0002_bs128"
            polluted.mkdir(parents=True)

            matches = mod.find_legacy_h1_runs(root / "outputs", dataset_tag="weld_seam_windows_ws5_tf75_pg0_h1")
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0], polluted)

    def test_manifest_contains_checkpoint_selection_metric(self) -> None:
        import run_teacher_h1_sweep as mod

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout = mod.ensure_output_layout(root / "teacher_h1_forecast_95_v2_gpu1")
            manifest_path = layout["manifests"] / "best_teacher_h1.json"
            dataset_npz = root / "dummy.npz"
            dataset_npz.write_text("x", encoding="utf-8")

            manifest = mod.build_manifest(
                manifest_path=manifest_path,
                dataset_check={
                    "horizon_ok": True,
                    "train_delta_ok": True,
                    "test_delta_ok": True,
                },
                layout=layout,
                dataset_npz=dataset_npz,
                analysis_csv=layout["analysis"] / "teacher_h1_ge95.csv",
                analysis_json=layout["analysis"] / "teacher_h1_ge95.json",
                all_runs=[],
                legacy_matches=[],
                target_acc=0.95,
            )

            self.assertEqual(manifest["checkpoint_selection_metric"], "test_accuracy")

    def test_scan98_phase_counts(self) -> None:
        import run_teacher_h1_sweep as mod

        peak = mod.build_scan98_peak_replay_configs()
        self.assertEqual(len(peak), 24)

        top4 = mod.select_top_configs_dedup([], top_k=4, fallback_configs=peak)
        self.assertEqual(len(top4), 4)

        seed_cfg = mod.build_scan98_seed_configs(top4)
        self.assertEqual(len(seed_cfg), 40)

        wd_cfg = mod.build_scan98_wd_expand_configs(top4)
        self.assertEqual(len(wd_cfg), 32)

        rescue_cfg = mod.build_scan98_rescue_configs(top4[:2])
        self.assertEqual(len(rescue_cfg), 16)

    def test_scan98_seed_configs_preserve_top4_diversity(self) -> None:
        import run_teacher_h1_sweep as mod

        top4 = [
            {"lr": 2.2e-4, "tcn_dropout": 0.08, "tcn_layers": 3, "weight_decay": 1.5e-4, "seed": 42},
            {"lr": 2.5e-4, "tcn_dropout": 0.10, "tcn_layers": 3, "weight_decay": 2e-4, "seed": 42},
            {"lr": 2.8e-4, "tcn_dropout": 0.12, "tcn_layers": 3, "weight_decay": 1.5e-4, "seed": 42},
            {"lr": 3.0e-4, "tcn_dropout": 0.12, "tcn_layers": 3, "weight_decay": 2e-4, "seed": 42},
        ]
        seed_cfg = mod.build_scan98_seed_configs(top4)
        identities = {(cfg["lr"], cfg["tcn_dropout"], cfg["weight_decay"]) for cfg in seed_cfg}
        self.assertEqual(len(identities), 4)

    def test_select_top_configs_dedup_ignores_seed(self) -> None:
        import run_teacher_h1_sweep as mod

        cfg_a_seed_1 = mod.scan98_training_defaults()
        cfg_a_seed_1.update({"lr": 3e-4, "tcn_layers": 3, "tcn_dropout": 0.1, "weight_decay": 2e-4, "seed": 42})
        cfg_a_seed_2 = dict(cfg_a_seed_1)
        cfg_a_seed_2["seed"] = 183
        cfg_b = dict(cfg_a_seed_1)
        cfg_b.update({"lr": 2.5e-4, "seed": 42})

        records = [
            {"run_name": "run_a_seed1", "test_accuracy": 0.981, "best_epoch": 3, "run_args": cfg_a_seed_1},
            {"run_name": "run_a_seed2", "test_accuracy": 0.982, "best_epoch": 4, "run_args": cfg_a_seed_2},
            {"run_name": "run_b", "test_accuracy": 0.979, "best_epoch": 5, "run_args": cfg_b},
        ]

        selected = mod.select_top_configs_dedup(records, top_k=2, fallback_configs=[])
        self.assertEqual(len(selected), 2)
        # First selected config should come from higher-accuracy duplicate group A (seed-insensitive dedup)
        self.assertEqual(selected[0]["lr"], 3e-4)
        self.assertEqual(selected[1]["lr"], 2.5e-4)

    def test_scan98_early_stop_rule(self) -> None:
        import run_teacher_h1_sweep as mod

        records = [
            {"test_accuracy": 0.981, "seed": 42},
            {"test_accuracy": 0.983, "seed": 183},
            {"test_accuracy": 0.982, "seed": 42},
        ]
        self.assertTrue(mod.should_early_stop_scan98(records, threshold=0.98))

        records_not_enough_seeds = [
            {"test_accuracy": 0.981, "seed": 42},
            {"test_accuracy": 0.983, "seed": 42},
            {"test_accuracy": 0.982, "seed": 42},
        ]
        self.assertFalse(mod.should_early_stop_scan98(records_not_enough_seeds, threshold=0.98))


if __name__ == "__main__":
    unittest.main()
