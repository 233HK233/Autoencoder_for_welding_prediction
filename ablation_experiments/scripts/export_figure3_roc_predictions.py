#!/usr/bin/env python3
"""Export Figure 3 test-set probabilities for ROC plotting."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models_tcn import AttentionTCNClassifier  # noqa: E402
from training_utils import parse_channels, validate_forecast_dataset_contract  # noqa: E402

from ablation_experiments.scripts.train_comparison_18d_baseline import build_model as build_baseline_model  # noqa: E402


FIGURE3_METHOD_ORDER = [
    "teacher-student(student)",
    "teacher(18D upper bound)",
    "lstm",
    "gru",
    "transformer",
    "inception",
]

METHOD_SLUGS = {
    "teacher-student(student)": "teacher_student_student",
    "teacher(18D upper bound)": "teacher_18d_upper_bound",
    "lstm": "lstm",
    "gru": "gru",
    "transformer": "transformer",
    "inception": "inception",
}

EXPECTED_TEST_SAMPLES = 356
EXPECTED_NUM_CLASSES = 3
IDENTITY_COLUMNS = ["sample_index", "seam_id", "seam_name", "start_idx", "target_idx", "y_true"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Figure 3 test predictions for ROC plotting")
    parser.add_argument(
        "--registry",
        type=Path,
        required=True,
        help="Path to outputs/figure 3/source_tables/comparison_summary.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_run_path(repo_root: Path, run_path: str) -> Path:
    candidate = Path(run_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root / candidate).resolve()


def checkpoint_filename_for_method(method_name: str) -> str:
    mapping = {
        "teacher-student(student)": "best_student_distill.pth",
        "teacher(18D upper bound)": "best_single_tcn.pth",
        "lstm": "best_model.pth",
        "gru": "best_model.pth",
        "transformer": "best_model.pth",
        "inception": "best_model.pth",
    }
    try:
        return mapping[method_name]
    except KeyError as exc:
        raise ValueError(f"unsupported Figure 3 method: {method_name}") from exc


def parse_drop_feature_indices(value: str, input_dim: int) -> list[int]:
    if not str(value).strip():
        return []
    drop = sorted({int(part.strip()) for part in str(value).split(",") if part.strip()})
    for idx in drop:
        if idx < 0 or idx >= input_dim:
            raise ValueError(f"drop feature index {idx} out of range for input_dim={input_dim}")
    return drop


def resolve_keep_feature_indices(run_args: Mapping[str, Any], input_dim: int) -> list[int]:
    keep = run_args.get("keep_feature_indices")
    if keep is not None:
        values = [int(idx) for idx in keep]
        if not values:
            raise ValueError("keep_feature_indices must not be empty")
        return values
    drop = set(parse_drop_feature_indices(str(run_args.get("drop_feature_indices", "")), input_dim=input_dim))
    values = [idx for idx in range(input_dim) if idx not in drop]
    if not values:
        raise ValueError("all features were dropped; keep at least one feature")
    return values


def validate_prediction_outputs(
    y_true: np.ndarray,
    logits: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    if y_true.shape[0] != EXPECTED_TEST_SAMPLES:
        raise ValueError(f"test sample count must be {EXPECTED_TEST_SAMPLES}, got {y_true.shape[0]}")
    if logits.shape != (EXPECTED_TEST_SAMPLES, EXPECTED_NUM_CLASSES):
        raise ValueError(
            f"logits shape must be ({EXPECTED_TEST_SAMPLES}, {EXPECTED_NUM_CLASSES}), got {tuple(logits.shape)}"
        )
    if probabilities.shape != (EXPECTED_TEST_SAMPLES, EXPECTED_NUM_CLASSES):
        raise ValueError(
            "probability shape must be "
            f"({EXPECTED_TEST_SAMPLES}, {EXPECTED_NUM_CLASSES}), got {tuple(probabilities.shape)}"
        )
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("probability rows must sum to 1.0 within tolerance 1e-5")


def load_registry_rows(registry_path: Path) -> list[dict[str, str]]:
    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    methods = {row["method_name"] for row in rows}
    missing = [method for method in FIGURE3_METHOD_ORDER if method not in methods]
    if missing:
        raise ValueError(f"registry missing Figure 3 methods: {missing}")
    by_method = {row["method_name"]: row for row in rows}
    return [by_method[method] for method in FIGURE3_METHOD_ORDER]


def load_run_args(run_dir: Path) -> dict[str, Any]:
    run_args_path = run_dir / "run_args.json"
    if not run_args_path.exists():
        raise FileNotFoundError(f"run_args.json missing under {run_dir}")
    payload = json.loads(run_args_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"run_args.json under {run_dir} must decode to a JSON object")
    return payload


def load_dataset(npz_path: Path) -> dict[str, np.ndarray]:
    validate_forecast_dataset_contract(npz_path, expected_horizon=1, expected_delta=5)
    with np.load(npz_path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def build_teacher_upper_bound_model(run_args: Mapping[str, Any], input_dim: int, num_classes: int) -> torch.nn.Module:
    latent_dim = int(run_args.get("latent_dim", 64))
    tcn_layers = int(run_args.get("tcn_layers", 3))
    channels = parse_channels(run_args.get("tcn_channels"), latent_dim=latent_dim, min_layers=tcn_layers)
    return AttentionTCNClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        channels=int(channels[-1]),
        tcn_layers=tcn_layers,
        tcn_kernel=int(run_args.get("tcn_kernel", 3)),
        tcn_dropout=float(run_args.get("tcn_dropout", 0.15)),
        dilation_base=int(run_args.get("tcn_dilation_base", 2)),
        attn_heads=int(run_args.get("attn_heads", 4)),
        attn_dropout=float(run_args.get("attn_dropout", 0.1)),
        ff_dim=int(run_args.get("attn_ff_dim", 128)),
        classifier_hidden=int(run_args.get("classifier_hidden", 128)),
        classifier_dropout=float(run_args.get("classifier_dropout", 0.35)),
    )


def build_student_model(run_args: Mapping[str, Any], input_dim: int, num_classes: int) -> torch.nn.Module:
    cfg = run_args.get("student_config_resolved")
    if not isinstance(cfg, dict):
        raise ValueError("student_config_resolved missing from student run_args.json")
    return AttentionTCNClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        channels=int(cfg["tcn_channels"]),
        tcn_layers=int(cfg["tcn_layers"]),
        tcn_kernel=int(cfg["tcn_kernel"]),
        tcn_dropout=float(cfg["tcn_dropout"]),
        dilation_base=int(cfg["tcn_dilation_base"]),
        attn_heads=int(cfg["attn_heads"]),
        attn_dropout=float(cfg["attn_dropout"]),
        ff_dim=int(cfg["attn_ff_dim"]),
        classifier_hidden=int(cfg["classifier_hidden"]),
        classifier_dropout=float(cfg["classifier_dropout"]),
    )


def build_model_for_method(method_name: str, run_args: Mapping[str, Any], input_dim: int, num_classes: int) -> torch.nn.Module:
    if method_name == "teacher(18D upper bound)":
        return build_teacher_upper_bound_model(run_args, input_dim=input_dim, num_classes=num_classes)
    if method_name == "teacher-student(student)":
        return build_student_model(run_args, input_dim=input_dim, num_classes=num_classes)
    namespace = SimpleNamespace(**dict(run_args))
    return build_baseline_model(namespace, input_dim=input_dim, num_classes=num_classes)


def load_strict_state_dict(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, Mapping):
        model.load_state_dict(payload, strict=True)
        return
    raise ValueError(f"checkpoint payload at {checkpoint_path} is not a supported state_dict mapping")


def output_directories(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "predictions": output_dir / "predictions",
        "tables": output_dir / "tables",
        "figure_exports": output_dir / "figure_exports",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def build_identity_arrays(dataset: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    seam_names = np.asarray(dataset["seam_name_order"])
    seam_ids = np.asarray(dataset["seam_id_test"], dtype=np.int64)
    return {
        "sample_index": np.arange(seam_ids.shape[0], dtype=np.int64),
        "seam_id": seam_ids,
        "seam_name": seam_names[seam_ids].astype(str),
        "start_idx": np.asarray(dataset["start_idx_test"], dtype=np.int64),
        "target_idx": np.asarray(dataset["target_idx_test"], dtype=np.int64),
        "y_true": np.asarray(dataset["y_test"], dtype=np.int64),
    }


def write_prediction_csv(
    csv_path: Path,
    identities: Mapping[str, np.ndarray],
    y_pred: np.ndarray,
    logits: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    headers = IDENTITY_COLUMNS + [
        "y_pred",
        "logit_class_0",
        "logit_class_1",
        "logit_class_2",
        "prob_class_0",
        "prob_class_1",
        "prob_class_2",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for idx in range(y_pred.shape[0]):
            row = [
                int(identities["sample_index"][idx]),
                int(identities["seam_id"][idx]),
                str(identities["seam_name"][idx]),
                int(identities["start_idx"][idx]),
                int(identities["target_idx"][idx]),
                int(identities["y_true"][idx]),
                int(y_pred[idx]),
            ]
            row.extend(f"{float(value):.8f}" for value in logits[idx])
            row.extend(f"{float(value):.8f}" for value in probabilities[idx])
            writer.writerow(row)


def run_inference(model: torch.nn.Module, features: np.ndarray, labels: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    logits_batches: list[np.ndarray] = []
    probs_batches: list[np.ndarray] = []
    pred_batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            logits, _ = model(x_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            logits_batches.append(logits.cpu().numpy())
            probs_batches.append(probs.cpu().numpy())
            pred_batches.append(preds.cpu().numpy())
    logits_np = np.concatenate(logits_batches, axis=0)
    probs_np = np.concatenate(probs_batches, axis=0)
    preds_np = np.concatenate(pred_batches, axis=0)
    return preds_np, logits_np, probs_np


def export_method_predictions(
    method_row: Mapping[str, str],
    dataset: Mapping[str, np.ndarray],
    output_dir: Path,
    device: torch.device,
) -> dict[str, str | float | int]:
    method_name = method_row["method_name"]
    run_dir = resolve_run_path(PROJECT_ROOT, method_row["run_path"])
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory missing for {method_name}: {run_dir}")
    run_args = load_run_args(run_dir)
    checkpoint_path = run_dir / checkpoint_filename_for_method(method_name)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint missing for {method_name}: {checkpoint_path}")

    x_test = np.asarray(dataset["X_test_full"], dtype=np.float32)
    y_test = np.asarray(dataset["y_test"], dtype=np.int64)
    if method_name == "teacher-student(student)":
        keep_indices = resolve_keep_feature_indices(run_args, input_dim=x_test.shape[2])
        x_eval = x_test[:, :, keep_indices]
    else:
        keep_indices = list(range(x_test.shape[2]))
        x_eval = x_test

    num_classes = int(np.max(y_test) + 1)
    if num_classes != EXPECTED_NUM_CLASSES:
        raise ValueError(f"class count must be {EXPECTED_NUM_CLASSES}, got {num_classes}")

    model = build_model_for_method(method_name, run_args, input_dim=x_eval.shape[2], num_classes=num_classes).to(device)
    load_strict_state_dict(model, checkpoint_path, device=device)
    y_pred, logits, probabilities = run_inference(model, x_eval, y_test, device=device)
    validate_prediction_outputs(y_true=y_test, logits=logits, probabilities=probabilities)

    accuracy = float(np.mean(y_pred == y_test))
    registry_acc = float(method_row["best_test_acc"])
    if abs(accuracy - registry_acc) > 0.0015:
        raise ValueError(
            f"exported accuracy for {method_name} differs from registry by more than 0.0015: "
            f"exported={accuracy:.6f} registry={registry_acc:.6f}"
        )

    identities = build_identity_arrays(dataset)
    prediction_path = output_dir / "predictions" / f"{METHOD_SLUGS[method_name]}_test_predictions.csv"
    write_prediction_csv(prediction_path, identities=identities, y_pred=y_pred, logits=logits, probabilities=probabilities)
    return {
        "method_name": method_name,
        "method_slug": METHOD_SLUGS[method_name],
        "run_path": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "prediction_csv": str(prediction_path),
        "test_samples": int(y_test.shape[0]),
        "input_dim": int(x_eval.shape[2]),
        "accuracy": accuracy,
        "registry_best_test_acc": registry_acc,
        "accuracy_delta": accuracy - registry_acc,
        "kept_feature_indices": json.dumps(keep_indices, ensure_ascii=True),
    }


def write_summary_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("summary rows must not be empty")
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dirs = output_directories(args.output_dir)
    registry_rows = load_registry_rows(args.registry)
    dataset = load_dataset(PROJECT_ROOT / "Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz")
    summaries = [
        export_method_predictions(method_row=row, dataset=dataset, output_dir=args.output_dir, device=device)
        for row in registry_rows
    ]
    write_summary_csv(dirs["tables"] / "prediction_export_summary.csv", summaries)


if __name__ == "__main__":
    main()
