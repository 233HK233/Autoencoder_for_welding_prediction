#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ablation_experiments.scripts.export_figure3_roc_predictions import (  # noqa: E402
    build_student_model,
    build_teacher_upper_bound_model,
    load_run_args,
    resolve_keep_feature_indices,
)
from data_utils import load_seam_csv  # noqa: E402
from models_tcn import AttentionTCNClassifier  # noqa: E402
from training_utils import extract_state_dict, parse_channels, validate_forecast_dataset_contract  # noqa: E402


TEACHER_LABEL = "Teacher-18D"
SEAL_LABEL = "SEAL-Weld Student-13D"
STUDENT_ONLY_LABEL = "Student-only-13D"
METHOD_LABELS = (TEACHER_LABEL, SEAL_LABEL, STUDENT_ONLY_LABEL)
METHOD_SLUGS = {
    TEACHER_LABEL: "teacher",
    SEAL_LABEL: "seal_student",
    STUDENT_ONLY_LABEL: "student_only",
}
SUMMARY_METHOD_NAMES = {
    TEACHER_LABEL: "teacher(18D upper bound)",
    SEAL_LABEL: "teacher-student(student)",
}
SEAM_ORDER = ("a01", "b01", "c01", "c02")
CASE_ORDER = ("S0-core", "0->1-boundary", "S1-core", "1->2-boundary")
CORE_CASE_TYPES = {"S0-core", "S1-core"}
BOUNDARY_CASE_TYPES = {"0->1-boundary", "1->2-boundary"}
IDENTITY_COLUMNS = ["sample_index", "seam_id", "seam_name", "start_idx", "target_idx", "y_true"]
PROBABILITY_COLUMNS = ["prob_class_0", "prob_class_1", "prob_class_2"]
EXPENSIVE_FEATURE_INDICES = (3, 4, 5, 6, 7)
WINDOW_SIZE = 5
HORIZON = 1
PANEL_RADIUS = 25
SAMPLE_PERIOD_S = 0.01


@dataclass(frozen=True)
class MethodSpec:
    label: str
    slug: str
    kind: str
    run_dir: Path
    checkpoint_path: Path
    keep_feature_indices: tuple[int, ...]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build all source results needed for Figure 4 temporal cases.")
    parser.add_argument("--dataset-npz", type=Path, default=Path("Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz"))
    parser.add_argument("--raw-data-dir", type=Path, default=Path("Data/raw_data"))
    parser.add_argument("--prediction-export-summary", type=Path, default=Path("outputs/roc_figure3/tables/prediction_export_summary.csv"))
    parser.add_argument("--teacher-predictions", type=Path, default=Path("outputs/roc_figure3/predictions/teacher_18d_upper_bound_test_predictions.csv"))
    parser.add_argument("--seal-predictions", type=Path, default=Path("outputs/roc_figure3/predictions/teacher_student_student_test_predictions.csv"))
    parser.add_argument(
        "--student-only-predictions",
        type=Path,
        default=Path("outputs/paper_figures/figure1_performance_error_structure_v02/source_predictions/student_only_13d_test_predictions.csv"),
    )
    parser.add_argument(
        "--student-only-run-dir",
        type=Path,
        default=Path(
            "ablation_experiments/h1/results/run_20260327_131013/ablation1_sweep/trials/"
            "trial_0004_A/ablation1_student_only_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep80_lr0.00015_bs96_seed156"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/paper_figures"))
    parser.add_argument("--output-name", type=str, default="figure4_temporal_prediction_cases_results")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args(argv)


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def prepare_output_dir(output_root: Path, output_name: str) -> Path:
    output_root = resolve_repo_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = output_root / f"{output_name}_v{index:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        index += 1


def build_full_seam_windows(
    *,
    data: np.ndarray,
    labels: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    window_size: int = WINDOW_SIZE,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got {data.shape}")
    if labels.ndim != 1 or labels.shape[0] != data.shape[0]:
        raise ValueError("labels must be 1D and match data length")

    delta = int(window_size) - 1 + int(horizon)
    n_windows = int(data.shape[0] - delta)
    if n_windows <= 0:
        raise ValueError(f"not enough time points for window_size={window_size}, horizon={horizon}")

    scale = np.where(np.asarray(scaler_scale, dtype=np.float32) < 1e-8, 1.0, scaler_scale).astype(np.float32)
    scaled = (data.astype(np.float32) - np.asarray(scaler_mean, dtype=np.float32)) / scale

    starts = np.arange(n_windows, dtype=np.int64)
    targets = starts + delta
    windows = np.stack([scaled[start : start + window_size] for start in starts], axis=0).astype(np.float32)
    y_true = labels[targets].astype(np.int64)
    return windows, starts, targets.astype(np.int64), y_true


def reduce_attention_weights(attention_weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(attention_weights, dtype=np.float32)
    if weights.ndim == 4:
        ribbon = weights.mean(axis=(1, 2))
    elif weights.ndim == 3:
        ribbon = weights.mean(axis=1)
    else:
        raise ValueError(f"attention weights must be 3D or 4D, got {weights.shape}")
    row_sums = ribbon.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-8, 1.0, row_sums)
    return (ribbon / row_sums).astype(np.float32)


def compute_transition_boundaries(labels: np.ndarray) -> tuple[int, int]:
    changes = np.where(labels[:-1] != labels[1:])[0]
    if len(changes) != 2:
        raise ValueError(f"Expected exactly 2 transitions, got {len(changes)}")
    return int(changes[0] + 1), int(changes[1] + 1)


def classify_boundary_distance_bin(distance: int) -> str:
    if int(distance) <= 5:
        return "Near"
    if int(distance) <= 15:
        return "Mid"
    return "Far"


def classify_case_type(y_true: int, target_idx: int, boundary_01: int, boundary_12: int) -> str:
    dist_01 = abs(int(target_idx) - int(boundary_01))
    dist_12 = abs(int(target_idx) - int(boundary_12))
    nearest = min(dist_01, dist_12)
    if int(y_true) == 0:
        return "0->1-boundary" if nearest <= 5 else "S0-core"
    if int(y_true) == 1:
        return "1->2-boundary" if dist_12 <= 5 and dist_12 <= dist_01 else "S1-core"
    if int(y_true) == 2:
        return "1->2-boundary" if nearest <= 5 else "S1-core"
    raise ValueError(f"Unsupported class label: {y_true}")


def load_prediction_tables(paths: Mapping[str, Path]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for label, path in paths.items():
        df = pd.read_csv(resolve_repo_path(path))
        missing = set(IDENTITY_COLUMNS + ["y_pred"] + PROBABILITY_COLUMNS).difference(df.columns)
        if missing:
            raise ValueError(f"{label} prediction CSV missing columns: {sorted(missing)}")
        tables[label] = df.copy()
    return tables


def merge_method_predictions(method_tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    labels = list(method_tables.keys())
    if not labels:
        raise ValueError("method_tables must not be empty")

    first = method_tables[labels[0]][IDENTITY_COLUMNS].reset_index(drop=True)
    for label in labels[1:]:
        candidate = method_tables[label][IDENTITY_COLUMNS].reset_index(drop=True)
        if not first.equals(candidate):
            raise ValueError(f"{label} prediction CSV is not aligned on identity columns")

    merged = first.copy()
    for label in labels:
        slug = METHOD_SLUGS.get(label, label.lower().replace(" ", "_"))
        df = method_tables[label].reset_index(drop=True)
        merged[f"{slug}_pred"] = df["y_pred"].astype(int)
        for class_id in range(3):
            merged[f"{slug}_prob_s{class_id}"] = df[f"prob_class_{class_id}"].astype(float)
    return merged


def annotate_merged_predictions(merged: pd.DataFrame, raw_data_dir: Path) -> pd.DataFrame:
    raw_data_dir = resolve_repo_path(raw_data_dir)
    seam_boundaries: dict[str, tuple[int, int]] = {}
    for seam_name in sorted(merged["seam_name"].astype(str).unique()):
        _data, labels = load_seam_csv(raw_data_dir / f"{seam_name}.csv")
        seam_boundaries[seam_name] = compute_transition_boundaries(labels)

    result = merged.copy()
    nearest: list[int] = []
    bins: list[str] = []
    cases: list[str] = []
    for row in result.itertuples(index=False):
        boundary_01, boundary_12 = seam_boundaries[str(row.seam_name)]
        dist = min(abs(int(row.target_idx) - boundary_01), abs(int(row.target_idx) - boundary_12))
        nearest.append(int(dist))
        bins.append(classify_boundary_distance_bin(dist))
        cases.append(classify_case_type(int(row.y_true), int(row.target_idx), boundary_01, boundary_12))

    result["nearest_boundary_dist"] = nearest
    result["boundary_distance_bin"] = bins
    result["case_type"] = cases

    for label in METHOD_LABELS:
        slug = METHOD_SLUGS[label]
        pred_col = f"{slug}_pred"
        result[f"{slug}_correct"] = result[pred_col].astype(int) == result["y_true"].astype(int)
        result[f"{slug}_confidence"] = [
            float(row[f"{slug}_prob_s{int(row[pred_col])}"]) for _, row in result.iterrows()
        ]

    result["teacher_student_both_correct"] = result["teacher_correct"] & result["seal_student_correct"]
    result["transfer_gain"] = result["seal_student_correct"] & ~result["student_only_correct"]
    result["seal_weld_failure"] = ~result["seal_student_correct"]
    return result


def _empty_panel_row(seam_name: str, case_type: str) -> dict[str, Any]:
    return {
        "seam_name": seam_name,
        "source_seam_name": "",
        "case_type": case_type,
        "panel_key": f"{seam_name}|{case_type}",
        "panel_status": "missing",
        "selection_rule": "no_same_seam_candidate",
    }


def _row_with_panel_metadata(row: pd.Series, seam_name: str, case_type: str, rule: str, status: str = "selected") -> dict[str, Any]:
    payload = row.to_dict()
    payload["seam_name"] = seam_name
    payload["source_seam_name"] = str(row["seam_name"])
    payload["case_type"] = case_type
    payload["panel_key"] = f"{seam_name}|{case_type}"
    payload["panel_status"] = status
    payload["selection_rule"] = rule
    return payload


def select_panel_rows(
    annotated: pd.DataFrame,
    *,
    seam_order: Iterable[str] = SEAM_ORDER,
    case_order: Iterable[str] = CASE_ORDER,
) -> pd.DataFrame:
    selected: list[dict[str, Any]] = []
    for seam_name in seam_order:
        seam_df = annotated.loc[annotated["seam_name"].astype(str).eq(str(seam_name))].copy()
        for case_type in case_order:
            subset = seam_df.loc[seam_df["case_type"].astype(str).eq(str(case_type))].copy()
            if subset.empty:
                selected.append(_empty_panel_row(str(seam_name), str(case_type)))
                continue

            if case_type in CORE_CASE_TYPES:
                candidates = subset.loc[subset["teacher_student_both_correct"]].copy()
                rule = "core_farthest_correct"
                if candidates.empty:
                    candidates = subset.loc[subset["seal_student_correct"]].copy()
                    rule = "core_farthest_student_correct_fallback"
                if candidates.empty:
                    candidates = subset.copy()
                    rule = "core_farthest_available_fallback"
                candidates = candidates.sort_values(
                    by=["nearest_boundary_dist", "seal_student_confidence", "sample_index"],
                    ascending=[False, False, True],
                    kind="mergesort",
                )
            elif case_type in BOUNDARY_CASE_TYPES:
                candidates = subset.loc[subset["teacher_student_both_correct"]].copy()
                rule = "successful_boundary_nearest"
                if candidates.empty:
                    candidates = subset.copy()
                    rule = "boundary_nearest_available_fallback"
                candidates = candidates.sort_values(
                    by=["nearest_boundary_dist", "seal_student_confidence", "sample_index"],
                    ascending=[True, False, True],
                    kind="mergesort",
                )
            else:
                raise ValueError(f"Unsupported case_type: {case_type}")

            selected.append(_row_with_panel_metadata(candidates.iloc[0], str(seam_name), str(case_type), rule))

    result = pd.DataFrame(selected)
    result = apply_cross_seam_backfill(result)
    order = {case: idx for idx, case in enumerate(case_order)}
    result["_case_order"] = result["case_type"].map(order)
    result = result.sort_values(["seam_name", "_case_order"], kind="mergesort").drop(columns=["_case_order"])
    return result.reset_index(drop=True)


def apply_cross_seam_backfill(panel_df: pd.DataFrame) -> pd.DataFrame:
    filled = panel_df.copy()
    selected = filled.loc[filled["panel_status"].eq("selected")].copy()
    for idx, row in filled.loc[filled["panel_status"].eq("missing")].iterrows():
        case_type = str(row["case_type"])
        candidates = selected.loc[selected["case_type"].astype(str).eq(case_type)].copy()
        if candidates.empty:
            continue
        candidates = candidates.sort_values(
            by=["seal_student_confidence", "nearest_boundary_dist", "sample_index"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        donor = candidates.iloc[0]
        for col in filled.columns:
            if col in {"panel_key", "seam_name", "case_type"}:
                continue
            filled.at[idx, col] = donor.get(col)
        filled.at[idx, "source_seam_name"] = donor.get("source_seam_name", donor.get("seam_name"))
        filled.at[idx, "panel_status"] = "backfilled"
        filled.at[idx, "selection_rule"] = f"cross_seam_backfill:{donor.get('source_seam_name', donor.get('seam_name'))}"
    return filled


def select_highlight_cases(annotated: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specs = [
        (
            "transfer_gain",
            annotated.loc[annotated["transfer_gain"]].copy(),
            ["nearest_boundary_dist", "seal_student_confidence", "sample_index"],
            [True, False, True],
            "seal_student_correct_and_student_only_wrong_closest_boundary",
        ),
        (
            "seal_weld_failure",
            annotated.loc[annotated["seal_weld_failure"]].copy(),
            ["nearest_boundary_dist", "seal_student_confidence", "sample_index"],
            [True, True, True],
            "seal_student_wrong_closest_boundary",
        ),
    ]
    for role, candidates, by, ascending, rule in specs:
        if candidates.empty:
            rows.append({"highlight_role": role, "selection_rule": "missing", "panel_status": "missing"})
            continue
        winner = candidates.sort_values(by=by, ascending=ascending, kind="mergesort").iloc[0].to_dict()
        winner["highlight_role"] = role
        winner["selection_rule"] = rule
        winner["panel_status"] = "selected"
        rows.append(winner)
    return pd.DataFrame(rows)


def load_dataset(npz_path: Path) -> dict[str, np.ndarray]:
    validate_forecast_dataset_contract(resolve_repo_path(npz_path), expected_horizon=1, expected_delta=5)
    with np.load(resolve_repo_path(npz_path), allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def parse_json_int_tuple(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        loaded = json.loads(value)
    else:
        loaded = list(value)
    return tuple(int(v) for v in loaded)


def build_method_specs(args: argparse.Namespace) -> list[MethodSpec]:
    summary = pd.read_csv(resolve_repo_path(args.prediction_export_summary))
    by_name = {str(row["method_name"]): row for _, row in summary.iterrows()}
    specs: list[MethodSpec] = []

    for label in (TEACHER_LABEL, SEAL_LABEL):
        method_name = SUMMARY_METHOD_NAMES[label]
        if method_name not in by_name:
            raise ValueError(f"prediction export summary missing {method_name}")
        row = by_name[method_name]
        run_dir = resolve_repo_path(str(row["run_path"]))
        checkpoint = resolve_repo_path(str(row["checkpoint_path"]))
        keep = parse_json_int_tuple(str(row["kept_feature_indices"]))
        specs.append(
            MethodSpec(
                label=label,
                slug=METHOD_SLUGS[label],
                kind="teacher" if label == TEACHER_LABEL else "seal_student",
                run_dir=run_dir,
                checkpoint_path=checkpoint,
                keep_feature_indices=keep,
            )
        )

    student_run_dir = resolve_repo_path(args.student_only_run_dir)
    student_args = load_run_args(student_run_dir)
    keep = tuple(resolve_keep_feature_indices(student_args, input_dim=18))
    specs.append(
        MethodSpec(
            label=STUDENT_ONLY_LABEL,
            slug=METHOD_SLUGS[STUDENT_ONLY_LABEL],
            kind="student_only",
            run_dir=student_run_dir,
            checkpoint_path=student_run_dir / "best_student_only.pth",
            keep_feature_indices=keep,
        )
    )
    for spec in specs:
        if not spec.checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint missing for {spec.label}: {spec.checkpoint_path}")
    return specs


def build_student_only_model(run_args: Mapping[str, Any], input_dim: int, num_classes: int) -> AttentionTCNClassifier:
    tcn_layers = int(run_args.get("tcn_layers", 3))
    latent_dim = int(run_args.get("student_latent_dim", run_args.get("latent_dim", 80)))
    channels = parse_channels(run_args.get("tcn_channels"), latent_dim=latent_dim, min_layers=tcn_layers)
    return AttentionTCNClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        channels=int(channels[-1]),
        tcn_layers=tcn_layers,
        tcn_kernel=int(run_args.get("tcn_kernel", 3)),
        tcn_dropout=float(run_args.get("tcn_dropout", 0.12)),
        dilation_base=int(run_args.get("tcn_dilation_base", 2)),
        attn_heads=int(run_args.get("attn_heads", 4)),
        attn_dropout=float(run_args.get("attn_dropout", 0.1)),
        ff_dim=int(run_args.get("attn_ff_dim", 128)),
        classifier_hidden=int(run_args.get("classifier_hidden", 128)),
        classifier_dropout=float(run_args.get("classifier_dropout", 0.35)),
    )


def build_model_for_spec(spec: MethodSpec, input_dim: int, num_classes: int) -> torch.nn.Module:
    run_args = load_run_args(spec.run_dir)
    if spec.kind == "teacher":
        return build_teacher_upper_bound_model(run_args, input_dim=input_dim, num_classes=num_classes)
    if spec.kind == "seal_student":
        return build_student_model(run_args, input_dim=input_dim, num_classes=num_classes)
    if spec.kind == "student_only":
        return build_student_only_model(run_args, input_dim=input_dim, num_classes=num_classes)
    raise ValueError(f"Unsupported method kind: {spec.kind}")


def load_model_state(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(extract_state_dict(payload), strict=True)


def run_attention_inference(
    model: torch.nn.Module,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(model, AttentionTCNClassifier):
        raise TypeError("Figure 4 attention export expects AttentionTCNClassifier models")
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    logits_batches: list[np.ndarray] = []
    prob_batches: list[np.ndarray] = []
    pred_batches: list[np.ndarray] = []
    ribbon_batches: list[np.ndarray] = []
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            logits, _z, attn_weights = model(x_batch, return_attention=True)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            logits_batches.append(logits.cpu().numpy())
            prob_batches.append(probs.cpu().numpy())
            pred_batches.append(preds.cpu().numpy())
            ribbon_batches.append(reduce_attention_weights(attn_weights.cpu().numpy()))
    logits_np = np.concatenate(logits_batches, axis=0)
    probs_np = np.concatenate(prob_batches, axis=0)
    preds_np = np.concatenate(pred_batches, axis=0)
    ribbons_np = np.concatenate(ribbon_batches, axis=0)
    return preds_np, logits_np, probs_np, ribbons_np


def build_continuous_identity_and_windows(
    dataset: Mapping[str, np.ndarray],
    raw_data_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    raw_data_dir = resolve_repo_path(raw_data_dir)
    seam_names = np.asarray(dataset["seam_name_order"]).astype(str).tolist()
    identity_rows: list[pd.DataFrame] = []
    window_batches: list[np.ndarray] = []
    raw_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    continuous_index_offset = 0
    for seam_id, seam_name in enumerate(seam_names):
        data, labels = load_seam_csv(raw_data_dir / f"{seam_name}.csv")
        mean = np.asarray(dataset[f"scaler_mean_{seam_name}"], dtype=np.float32)
        scale = np.asarray(dataset[f"scaler_scale_{seam_name}"], dtype=np.float32)
        windows, starts, targets, y_true = build_full_seam_windows(
            data=data,
            labels=labels,
            scaler_mean=mean,
            scaler_scale=scale,
            window_size=WINDOW_SIZE,
            horizon=HORIZON,
        )
        indices = np.arange(continuous_index_offset, continuous_index_offset + windows.shape[0], dtype=np.int64)
        continuous_index_offset += windows.shape[0]
        identity_rows.append(
            pd.DataFrame(
                {
                    "continuous_index": indices,
                    "seam_id": int(seam_id),
                    "seam_name": seam_name,
                    "start_idx": starts,
                    "target_idx": targets,
                    "y_true": y_true,
                }
            )
        )
        window_batches.append(windows)
        scale_safe = np.where(scale < 1e-8, 1.0, scale)
        scaled_full = ((data.astype(np.float32) - mean) / scale_safe).astype(np.float32)
        raw_cache[seam_name] = (data.astype(np.float32), labels.astype(np.int64), scaled_full)
    return pd.concat(identity_rows, ignore_index=True), np.concatenate(window_batches, axis=0), raw_cache


def run_continuous_inference(
    specs: Iterable[MethodSpec],
    identity_df: pd.DataFrame,
    windows_full: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    num_classes = int(identity_df["y_true"].max() + 1)
    for spec in specs:
        x_eval = windows_full[:, :, list(spec.keep_feature_indices)]
        model = build_model_for_spec(spec, input_dim=x_eval.shape[2], num_classes=num_classes).to(device)
        load_model_state(model, spec.checkpoint_path, device)
        y_pred, logits, probs, ribbons = run_attention_inference(model, x_eval, device=device, batch_size=batch_size)
        method_df = identity_df.copy()
        method_df["method"] = spec.label
        method_df["method_slug"] = spec.slug
        method_df["y_pred"] = y_pred.astype(np.int64)
        for class_id in range(3):
            method_df[f"logit_class_{class_id}"] = logits[:, class_id].astype(float)
            method_df[f"prob_class_{class_id}"] = probs[:, class_id].astype(float)
        for step in range(ribbons.shape[1]):
            method_df[f"attn_step_{step}"] = ribbons[:, step].astype(float)
        rows.append(method_df)
    return pd.concat(rows, ignore_index=True)


def build_panel_timeseries(
    panel_df: pd.DataFrame,
    continuous_df: pd.DataFrame,
    raw_cache: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    method_slugs = {label: slug for label, slug in METHOD_SLUGS.items()}
    for panel in panel_df.itertuples(index=False):
        if str(panel.panel_status) == "missing" or pd.isna(getattr(panel, "target_idx", np.nan)):
            continue
        source_seam = str(panel.source_seam_name)
        target_idx = int(panel.target_idx)
        _raw, labels, scaled = raw_cache[source_seam]
        x0 = max(0, target_idx - PANEL_RADIUS)
        x1 = min(labels.shape[0], target_idx + PANEL_RADIUS + 1)
        eco_idx = [idx for idx in range(scaled.shape[1]) if idx not in set(EXPENSIVE_FEATURE_INDICES)]
        eco = scaled[:, eco_idx]
        pc1_scores = first_principal_component_scores(eco)
        local_predictions = continuous_df.loc[
            continuous_df["seam_name"].astype(str).eq(source_seam)
            & continuous_df["target_idx"].between(x0, x1 - 1)
        ].copy()
        by_time_method = {
            (int(row.target_idx), str(row.method)): row
            for row in local_predictions.itertuples(index=False)
        }
        for time_idx in range(x0, x1):
            row: dict[str, Any] = {
                "panel_key": str(panel.panel_key),
                "panel_seam_name": str(panel.seam_name),
                "source_seam_name": source_seam,
                "case_type": str(panel.case_type),
                "time_idx": int(time_idx),
                "relative_step": int(time_idx - target_idx),
                "relative_time_s": float((time_idx - target_idx) * SAMPLE_PERIOD_S),
                "state_label": int(labels[time_idx]),
                "economic_mean_z": float(np.mean(eco[time_idx])),
                "economic_pc1": float(pc1_scores[time_idx]),
            }
            for label, slug in method_slugs.items():
                prediction = by_time_method.get((time_idx, label))
                if prediction is None:
                    for class_id in range(3):
                        row[f"{slug}_prob_s{class_id}"] = np.nan
                    row[f"{slug}_pred"] = np.nan
                    continue
                row[f"{slug}_pred"] = int(prediction.y_pred)
                for class_id in range(3):
                    row[f"{slug}_prob_s{class_id}"] = float(getattr(prediction, f"prob_class_{class_id}"))
            rows.append(row)
    return pd.DataFrame(rows)


def first_principal_component_scores(values: np.ndarray) -> np.ndarray:
    centered = values.astype(np.float64) - values.astype(np.float64).mean(axis=0, keepdims=True)
    if centered.shape[0] == 0:
        return np.array([], dtype=np.float32)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ vt[0]
    std = float(scores.std(ddof=0))
    if std < 1e-8:
        return np.zeros(scores.shape, dtype=np.float32)
    return ((scores - float(scores.mean())) / std).astype(np.float32)


def build_panel_attention(panel_df: pd.DataFrame, continuous_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    attn_cols = [f"attn_step_{idx}" for idx in range(WINDOW_SIZE)]
    for panel in panel_df.itertuples(index=False):
        if str(panel.panel_status) == "missing" or pd.isna(getattr(panel, "start_idx", np.nan)):
            continue
        source_seam = str(panel.source_seam_name)
        start_idx = int(panel.start_idx)
        target_idx = int(panel.target_idx)
        matching = continuous_df.loc[
            continuous_df["seam_name"].astype(str).eq(source_seam)
            & continuous_df["start_idx"].astype(int).eq(start_idx)
            & continuous_df["target_idx"].astype(int).eq(target_idx)
        ].copy()
        for prediction in matching.itertuples(index=False):
            for step, col in enumerate(attn_cols):
                raw_idx = start_idx + step
                rows.append(
                    {
                        "panel_key": str(panel.panel_key),
                        "panel_seam_name": str(panel.seam_name),
                        "source_seam_name": source_seam,
                        "case_type": str(panel.case_type),
                        "method": str(prediction.method),
                        "method_slug": str(prediction.method_slug),
                        "input_step": int(step),
                        "raw_idx": int(raw_idx),
                        "relative_step": int(raw_idx - target_idx),
                        "relative_time_s": float((raw_idx - target_idx) * SAMPLE_PERIOD_S),
                        "attention_weight": float(getattr(prediction, col)),
                    }
                )
    return pd.DataFrame(rows)


def validate_continuous_against_test(continuous_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in METHOD_LABELS:
        slug = METHOD_SLUGS[label]
        cont = continuous_df.loc[continuous_df["method"].eq(label)].copy()
        joined = test_df.merge(
            cont,
            on=["seam_name", "start_idx", "target_idx", "y_true"],
            suffixes=("_test", "_continuous"),
        )
        max_delta = 0.0
        for class_id in range(3):
            delta = np.abs(joined[f"{slug}_prob_s{class_id}"] - joined[f"prob_class_{class_id}"])
            if len(delta):
                max_delta = max(max_delta, float(delta.max()))
        pred_match = bool((joined[f"{slug}_pred"].astype(int) == joined["y_pred"].astype(int)).all())
        rows.append(
            {
                "method": label,
                "matched_rows": int(len(joined)),
                "max_probability_abs_delta": max_delta,
                "prediction_argmax_match": pred_match,
            }
        )
        if len(joined) != len(test_df) or max_delta > 1e-4 or not pred_match:
            raise ValueError(
                f"continuous inference validation failed for {label}: "
                f"matched={len(joined)} max_delta={max_delta:.6g} pred_match={pred_match}"
            )
    return pd.DataFrame(rows)


def write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    specs: Iterable[MethodSpec],
    output_files: Mapping[str, Path],
    test_df: pd.DataFrame,
    continuous_df: pd.DataFrame,
    panel_df: pd.DataFrame,
) -> None:
    specs_payload = [
        {
            "label": spec.label,
            "slug": spec.slug,
            "kind": spec.kind,
            "run_dir": str(spec.run_dir),
            "checkpoint_path": str(spec.checkpoint_path),
            "keep_feature_indices": list(spec.keep_feature_indices),
        }
        for spec in specs
    ]
    payload = {
        "figure": "Figure 4 temporal prediction cases",
        "dataset_npz": str(resolve_repo_path(args.dataset_npz)),
        "raw_data_dir": str(resolve_repo_path(args.raw_data_dir)),
        "sample_period_s": SAMPLE_PERIOD_S,
        "window_size": WINDOW_SIZE,
        "horizon": HORIZON,
        "methods": specs_payload,
        "selection_rules": {
            "core": "Within each seam/context, choose the farthest Teacher+SEAL-Weld-correct test window from the nearest state boundary; ties use higher SEAL-Weld confidence then lower sample_index.",
            "successful_boundary": "Within each seam/boundary context, choose the nearest Teacher+SEAL-Weld-correct test window to the boundary; ties use higher SEAL-Weld confidence then lower sample_index.",
            "transfer_gain": "Global highlight: nearest boundary window where SEAL-Weld is correct and Student-only is wrong.",
            "seal_weld_failure": "Global highlight: nearest boundary window where SEAL-Weld remains wrong.",
            "backfill": "If a seam/context has no test candidate, fill the panel from the strongest selected same-context donor and mark panel_status=backfilled.",
        },
        "counts": {
            "test_windows": int(len(test_df)),
            "continuous_rows": int(len(continuous_df)),
            "continuous_windows_per_method": int(len(continuous_df) / len(METHOD_LABELS)),
            "panel_rows": int(len(panel_df)),
            "backfilled_panels": int(panel_df["panel_status"].eq("backfilled").sum()),
        },
        "output_files": {key: str(value) for key, value in output_files.items()},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = prepare_output_dir(args.output_root, args.output_name)
    device = resolve_device(args.device)

    prediction_tables = load_prediction_tables(
        {
            TEACHER_LABEL: args.teacher_predictions,
            SEAL_LABEL: args.seal_predictions,
            STUDENT_ONLY_LABEL: args.student_only_predictions,
        }
    )
    merged = merge_method_predictions(prediction_tables)
    annotated = annotate_merged_predictions(merged, args.raw_data_dir)
    panel_df = select_panel_rows(annotated)
    highlight_df = select_highlight_cases(annotated)

    dataset = load_dataset(args.dataset_npz)
    identity_df, windows_full, raw_cache = build_continuous_identity_and_windows(dataset, args.raw_data_dir)
    specs = build_method_specs(args)
    continuous_df = run_continuous_inference(
        specs,
        identity_df,
        windows_full,
        device=device,
        batch_size=int(args.batch_size),
    )
    validation_df = validate_continuous_against_test(continuous_df, annotated)
    panel_timeseries_df = build_panel_timeseries(panel_df, continuous_df, raw_cache)
    panel_attention_df = build_panel_attention(panel_df, continuous_df)

    files = {
        "test_window_predictions": output_dir / "figure4_test_window_predictions.csv",
        "panel_selection": output_dir / "figure4_panel_selection.csv",
        "highlight_cases": output_dir / "figure4_highlight_cases.csv",
        "continuous_predictions": output_dir / "figure4_continuous_predictions.csv",
        "panel_timeseries": output_dir / "figure4_panel_timeseries.csv",
        "panel_attention": output_dir / "figure4_panel_attention.csv",
        "validation_summary": output_dir / "figure4_validation_summary.csv",
        "metadata": output_dir / "figure4_result_metadata.json",
    }
    annotated.to_csv(files["test_window_predictions"], index=False)
    panel_df.to_csv(files["panel_selection"], index=False)
    highlight_df.to_csv(files["highlight_cases"], index=False)
    continuous_df.to_csv(files["continuous_predictions"], index=False)
    panel_timeseries_df.to_csv(files["panel_timeseries"], index=False)
    panel_attention_df.to_csv(files["panel_attention"], index=False)
    validation_df.to_csv(files["validation_summary"], index=False)
    write_metadata(
        files["metadata"],
        args=args,
        specs=specs,
        output_files=files,
        test_df=annotated,
        continuous_df=continuous_df,
        panel_df=panel_df,
    )

    print(f"Saved Figure 4 result package to: {output_dir}")
    for key, path in files.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
