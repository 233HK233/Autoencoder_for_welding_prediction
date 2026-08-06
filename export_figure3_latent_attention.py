#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
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
)
from data_utils import load_seam_csv  # noqa: E402
from models_tcn import AttentionTCNClassifier  # noqa: E402
from training_utils import extract_state_dict, parse_channels, validate_forecast_dataset_contract  # noqa: E402

try:
    from sklearn.metrics import f1_score
except Exception:  # pragma: no cover - fallback for minimal environments.
    f1_score = None


EXPECTED_DATASET = Path("Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz")
RAW_DATA_DIR = Path("Data/raw_data")
EXPECTED_TEST_SAMPLES = 356
EXPECTED_NUM_CLASSES = 3
WINDOW_SIZE = 5
HORIZON = 1
SAMPLE_PERIOD_S = 0.01
PROJECTION_SPACE = "strict_raw_latent"

TEACHER_NAME = "Teacher"
STUDENT_ONLY_NAME = "Student-only"
SEAL_NAME = "SEAL-Weld Student"
MODEL_ORDER = (TEACHER_NAME, STUDENT_ONLY_NAME, SEAL_NAME)
METHOD_SLUGS = {
    TEACHER_NAME: "teacher",
    STUDENT_ONLY_NAME: "student_only",
    SEAL_NAME: "seal_weld_student",
}
EMBEDDING_FILENAMES = {
    TEACHER_NAME: "teacher_embeddings_test.npz",
    STUDENT_ONLY_NAME: "student_only_embeddings_test.npz",
    SEAL_NAME: "seal_weld_student_embeddings_test.npz",
}
CLASS_LABELS = {0: "S0", 1: "S1", 2: "S2"}
BOUNDARY_BIN_ORDER = ("Far", "Mid", "Near")

DEFAULT_TEACHER_RUN_DIR = Path(
    "outputs/teacher_h1_scan98_gpu1_v2/runs/"
    "single_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep60_lr0.0003_bs128_k3_l3_d0.08_lat64_wd0.00015_seed14"
)
DEFAULT_SEAL_RUN_DIR = Path(
    "ablation_experiments/h1/results/figure2_matched_distillation_ablation/seed14/full_seal_weld/"
    "distill_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep80_lr0.0002_bs128_T3.5_lce0.7_lkd1.5_lf0.3_seed14"
)
DEFAULT_STUDENT_ONLY_RUN_DIR = Path(
    "ablation_experiments/h1/results/figure2_matched_distillation_ablation/seed14/student_only_anchor/"
    "ablation1_student_only_tcn_attn_weld_seam_windows_ws5_tf75_pg0_h1_ep80_lr0.0001_bs64_seed144"
)

SOURCE_TABLES = (
    "selected_model_runs.csv",
    "unified_projection_coordinates.csv",
    "class_centroid_cosine_similarity.csv",
    "linear_cka_by_class.csv",
    "similarity_by_boundary_bin.csv",
    "selected_boundary_windows.csv",
    "boundary_attention_heatmap.csv",
    "reproduced_classification_metrics.csv",
)


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    method_slug: str
    role: str
    run_dir: Path
    checkpoint_path: Path
    keep_feature_indices: tuple[int, ...]


@dataclass(frozen=True)
class InferenceResult:
    spec: ModelSpec
    z_test: np.ndarray
    logits: np.ndarray
    prob: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    sample_index: np.ndarray
    seam_id: np.ndarray
    seam_name: np.ndarray
    start_idx: np.ndarray
    target_idx: np.ndarray
    attention_weights: np.ndarray
    attention_ribbon: np.ndarray


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export source data for Figure 3 latent transfer and attention interpretability."
    )
    parser.add_argument("--dataset-npz", type=Path, default=EXPECTED_DATASET)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/paper_figures"))
    parser.add_argument("--projection-method", type=str, default="umap", choices=["umap", "pacmap", "auto", "pca"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--student-only-run-dir", type=Path, default=DEFAULT_STUDENT_ONLY_RUN_DIR)
    parser.add_argument("--seal-weld-run-dir", type=Path, default=DEFAULT_SEAL_RUN_DIR)
    parser.add_argument("--teacher-run-dir", type=Path, default=DEFAULT_TEACHER_RUN_DIR)
    parser.add_argument(
        "--allow-pca-draft",
        action="store_true",
        help="Allow projection_method=pca for draft-only runs. Final strict runs must use UMAP or PaCMAP.",
    )
    return parser.parse_args(argv)


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def prepare_versioned_output_dir(output_root: Path, output_name: str = "figure3_latent_attention") -> Path:
    root = resolve_repo_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = root / f"{output_name}_v{index:02d}"
        if not candidate.exists():
            (candidate / "embeddings").mkdir(parents=True, exist_ok=False)
            (candidate / "source_tables").mkdir(parents=True, exist_ok=False)
            return candidate
        index += 1


def load_run_args(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_args.json"
    if not path.exists():
        raise FileNotFoundError(f"run_args.json missing under {run_dir}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must decode to a JSON object")
    return payload


def parse_drop_feature_indices(value: str, input_dim: int) -> list[int]:
    if not str(value).strip():
        return []
    drop = sorted({int(part.strip()) for part in str(value).split(",") if part.strip()})
    for idx in drop:
        if idx < 0 or idx >= input_dim:
            raise ValueError(f"drop feature index {idx} out of range for input_dim={input_dim}")
    return drop


def resolve_keep_feature_indices(run_args: Mapping[str, Any], input_dim: int) -> tuple[int, ...]:
    keep = run_args.get("keep_feature_indices")
    if keep is not None:
        values = tuple(int(idx) for idx in keep)
        if not values:
            raise ValueError("keep_feature_indices must not be empty")
        return values
    drop = set(parse_drop_feature_indices(str(run_args.get("drop_feature_indices", "")), input_dim=input_dim))
    values = tuple(idx for idx in range(input_dim) if idx not in drop)
    if not values:
        raise ValueError("all features were dropped; keep at least one feature")
    return values


def load_dataset(npz_path: Path) -> dict[str, np.ndarray]:
    validate_forecast_dataset_contract(npz_path, expected_horizon=1, expected_delta=5)
    with np.load(npz_path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def build_identity_arrays(dataset: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    seam_names = np.asarray(dataset["seam_name_order"]).astype(str)
    seam_ids = np.asarray(dataset["seam_id_test"], dtype=np.int64)
    return {
        "sample_index": np.arange(seam_ids.shape[0], dtype=np.int64),
        "seam_id": seam_ids,
        "seam_name": seam_names[seam_ids].astype(str),
        "start_idx": np.asarray(dataset["start_idx_test"], dtype=np.int64),
        "target_idx": np.asarray(dataset["target_idx_test"], dtype=np.int64),
        "y_true": np.asarray(dataset["y_test"], dtype=np.int64),
    }


def build_model_specs(args: argparse.Namespace, dataset: Mapping[str, np.ndarray]) -> list[ModelSpec]:
    input_dim = int(np.asarray(dataset["X_test_full"]).shape[2])
    teacher_run_dir = resolve_repo_path(args.teacher_run_dir)
    student_run_dir = resolve_repo_path(args.student_only_run_dir)
    seal_run_dir = resolve_repo_path(args.seal_weld_run_dir)

    teacher_args = load_run_args(teacher_run_dir)
    student_args = load_run_args(student_run_dir)
    seal_args = load_run_args(seal_run_dir)

    specs = [
        ModelSpec(
            model_name=TEACHER_NAME,
            method_slug=METHOD_SLUGS[TEACHER_NAME],
            role="18D upper-bound teacher",
            run_dir=teacher_run_dir,
            checkpoint_path=teacher_run_dir / "best_single_tcn.pth",
            keep_feature_indices=tuple(range(input_dim)),
        ),
        ModelSpec(
            model_name=STUDENT_ONLY_NAME,
            method_slug=METHOD_SLUGS[STUDENT_ONLY_NAME],
            role="13D supervised student",
            run_dir=student_run_dir,
            checkpoint_path=student_run_dir / "best_student_only.pth",
            keep_feature_indices=resolve_keep_feature_indices(student_args, input_dim=input_dim),
        ),
        ModelSpec(
            model_name=SEAL_NAME,
            method_slug=METHOD_SLUGS[SEAL_NAME],
            role="13D distilled student",
            run_dir=seal_run_dir,
            checkpoint_path=seal_run_dir / "best_student_distill.pth",
            keep_feature_indices=resolve_keep_feature_indices(seal_args, input_dim=input_dim),
        ),
    ]
    for spec in specs:
        if not spec.checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint missing for {spec.model_name}: {spec.checkpoint_path}")
    return specs


def build_student_only_model(run_args: Mapping[str, Any], input_dim: int, num_classes: int) -> AttentionTCNClassifier:
    tcn_layers = int(run_args.get("tcn_layers", 3))
    latent_dim = int(run_args.get("student_latent_dim", run_args.get("latent_dim", 80)))
    channels = parse_channels(str(run_args.get("tcn_channels", "")), latent_dim=latent_dim, min_layers=tcn_layers)
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


def build_model_for_spec(spec: ModelSpec, input_dim: int, num_classes: int) -> torch.nn.Module:
    run_args = load_run_args(spec.run_dir)
    if spec.model_name == TEACHER_NAME:
        return build_teacher_upper_bound_model(run_args, input_dim=input_dim, num_classes=num_classes)
    if spec.model_name == SEAL_NAME:
        return build_student_model(run_args, input_dim=input_dim, num_classes=num_classes)
    if spec.model_name == STUDENT_ONLY_NAME:
        return build_student_only_model(run_args, input_dim=input_dim, num_classes=num_classes)
    raise ValueError(f"unsupported model spec: {spec.model_name}")


def load_model_state(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(extract_state_dict(payload), strict=True)


def reduce_attention_weights(attention_weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(attention_weights, dtype=np.float32)
    if weights.ndim != 4:
        raise ValueError(f"attention weights must have shape [batch, heads, target_step, source_step], got {weights.shape}")
    ribbon = weights.mean(axis=(1, 2))
    row_sums = ribbon.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-8, 1.0, row_sums)
    return (ribbon / row_sums).astype(np.float32)


def run_model_inference(
    model: torch.nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(model, AttentionTCNClassifier):
        raise TypeError("Figure 3 attention export expects AttentionTCNClassifier models")
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logits_batches: list[np.ndarray] = []
    prob_batches: list[np.ndarray] = []
    pred_batches: list[np.ndarray] = []
    z_batches: list[np.ndarray] = []
    attention_batches: list[np.ndarray] = []
    ribbon_batches: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            logits, z, attention_weights = model(x_batch, return_attention=True)
            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)
            attn_np = attention_weights.detach().cpu().numpy().astype(np.float32)
            logits_batches.append(logits.detach().cpu().numpy().astype(np.float32))
            prob_batches.append(prob.detach().cpu().numpy().astype(np.float32))
            pred_batches.append(pred.detach().cpu().numpy().astype(np.int64))
            z_batches.append(z.detach().cpu().numpy().astype(np.float32))
            attention_batches.append(attn_np)
            ribbon_batches.append(reduce_attention_weights(attn_np))

    return (
        np.concatenate(z_batches, axis=0),
        np.concatenate(logits_batches, axis=0),
        np.concatenate(prob_batches, axis=0),
        np.concatenate(pred_batches, axis=0),
        np.concatenate(attention_batches, axis=0),
        np.concatenate(ribbon_batches, axis=0),
    )


def export_embeddings(
    specs: Iterable[ModelSpec],
    dataset: Mapping[str, np.ndarray],
    output_dir: Path,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, InferenceResult]:
    x_test_full = np.asarray(dataset["X_test_full"], dtype=np.float32)
    y_test = np.asarray(dataset["y_test"], dtype=np.int64)
    num_classes = int(np.max(y_test) + 1)
    if num_classes != EXPECTED_NUM_CLASSES:
        raise ValueError(f"expected {EXPECTED_NUM_CLASSES} classes, got {num_classes}")
    identities = build_identity_arrays(dataset)
    results: dict[str, InferenceResult] = {}
    embeddings_dir = output_dir / "embeddings"

    for spec in specs:
        x_eval = x_test_full[:, :, list(spec.keep_feature_indices)]
        model = build_model_for_spec(spec, input_dim=x_eval.shape[2], num_classes=num_classes).to(device)
        load_model_state(model, spec.checkpoint_path, device=device)
        z_test, logits, prob, y_pred, attention_weights, attention_ribbon = run_model_inference(
            model,
            x_eval,
            y_test,
            device=device,
            batch_size=batch_size,
        )
        path = embeddings_dir / EMBEDDING_FILENAMES[spec.model_name]
        np.savez(
            path,
            Z_test=z_test,
            logits=logits,
            prob=prob,
            y_true=y_test,
            y_pred=y_pred,
            sample_index=identities["sample_index"],
            seam_id=identities["seam_id"],
            seam_name=identities["seam_name"],
            start_idx=identities["start_idx"],
            target_idx=identities["target_idx"],
            attention_weights=attention_weights,
            attention_ribbon=attention_ribbon,
        )
        results[spec.model_name] = InferenceResult(
            spec=spec,
            z_test=z_test,
            logits=logits,
            prob=prob,
            y_true=y_test,
            y_pred=y_pred,
            sample_index=identities["sample_index"],
            seam_id=identities["seam_id"],
            seam_name=identities["seam_name"],
            start_idx=identities["start_idx"],
            target_idx=identities["target_idx"],
            attention_weights=attention_weights,
            attention_ribbon=attention_ribbon,
        )
    return results


def _standardize_matrix(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, ddof=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return ((x - mean) / std).astype(np.float32)


def _resolve_projection_method(method: str, allow_pca_draft: bool = False) -> str:
    normalized = method.lower()
    if normalized == "auto":
        try:
            import umap  # noqa: F401

            return "umap"
        except Exception:
            try:
                import pacmap  # noqa: F401

                return "pacmap"
            except Exception as exc:
                raise ImportError("Neither umap-learn nor pacmap is installed; final Figure 3 requires one.") from exc
    if normalized == "pca" and not allow_pca_draft:
        raise ValueError("PCA projection is draft-only. Pass --allow-pca-draft to use it explicitly.")
    return normalized


def fit_unified_projection(
    latents: Mapping[str, np.ndarray],
    *,
    projection_method: str,
    projection_space: str,
    random_state: int,
    allow_pca_draft: bool = False,
) -> dict[str, np.ndarray]:
    if tuple(latents.keys()) != MODEL_ORDER:
        latents = {name: latents[name] for name in MODEL_ORDER}
    dims = {name: int(np.asarray(z).shape[1]) for name, z in latents.items()}
    if projection_space == "strict_raw_latent" and len(set(dims.values())) != 1:
        raise ValueError(f"strict_raw_latent requires matching latent dimensions, got {dims}")

    standardized = [_standardize_matrix(np.asarray(latents[name], dtype=np.float32)) for name in MODEL_ORDER]
    counts = [matrix.shape[0] for matrix in standardized]
    concatenated = np.concatenate(standardized, axis=0)
    method = _resolve_projection_method(projection_method, allow_pca_draft=allow_pca_draft)

    if method == "umap":
        try:
            import umap
        except Exception as exc:
            raise ImportError("umap-learn is required for projection_method=umap.") from exc
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(concatenated)
    elif method == "pacmap":
        try:
            import pacmap
        except Exception as exc:
            raise ImportError("pacmap is required for projection_method=pacmap.") from exc
        reducer = pacmap.PaCMAP(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(concatenated, init="pca")
    elif method == "pca":
        centered = concatenated - concatenated.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vt[:2].T
    else:
        raise ValueError(f"unsupported projection method: {projection_method}")

    split: dict[str, np.ndarray] = {}
    offset = 0
    for name, count in zip(MODEL_ORDER, counts):
        split[name] = np.asarray(coords[offset : offset + count], dtype=np.float32)
        offset += count
    return split


def build_projection_table(
    results: Mapping[str, InferenceResult],
    identities: Mapping[str, np.ndarray],
    *,
    projection_method: str,
    projection_space: str,
    random_state: int,
    allow_pca_draft: bool,
) -> tuple[pd.DataFrame, str]:
    latents = {name: results[name].z_test for name in MODEL_ORDER}
    resolved_method = _resolve_projection_method(projection_method, allow_pca_draft=allow_pca_draft)
    coords_by_model = fit_unified_projection(
        latents,
        projection_method=resolved_method,
        projection_space=projection_space,
        random_state=random_state,
        allow_pca_draft=allow_pca_draft,
    )
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        result = results[model_name]
        coords = coords_by_model[model_name]
        for idx in range(coords.shape[0]):
            y_true = int(result.y_true[idx])
            y_pred = int(result.y_pred[idx])
            rows.append(
                {
                    "model_name": model_name,
                    "sample_index": int(identities["sample_index"][idx]),
                    "x_2d": float(coords[idx, 0]),
                    "y_2d": float(coords[idx, 1]),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "correct": bool(y_pred == y_true),
                    "seam_name": str(identities["seam_name"][idx]),
                    "start_idx": int(identities["start_idx"][idx]),
                    "target_idx": int(identities["target_idx"][idx]),
                    "projection_method": resolved_method,
                    "projection_space": projection_space,
                    "projection_random_state": int(random_state),
                }
            )
    return pd.DataFrame(rows), resolved_method


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] != y.shape[0]:
        raise ValueError("linear CKA requires the same number of rows")
    if x.shape[0] == 0:
        return float("nan")
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    numerator = float(np.linalg.norm(xc.T @ yc, ord="fro") ** 2)
    denom = float(np.linalg.norm(xc.T @ xc, ord="fro") * np.linalg.norm(yc.T @ yc, ord="fro"))
    if denom < 1e-12:
        return float("nan")
    return numerator / denom


def cosine_between(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def paired_cosine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    numerator = np.sum(x * y, axis=1)
    denom = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    denom = np.where(denom < 1e-12, np.nan, denom)
    return numerator / denom


def compute_alignment_tables(
    results: Mapping[str, InferenceResult],
    annotated_identity: pd.DataFrame,
    *,
    projection_space: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    standardized = {name: _standardize_matrix(results[name].z_test) for name in MODEL_ORDER}
    y_true = results[TEACHER_NAME].y_true
    centroid_rows: list[dict[str, Any]] = []
    cka_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []

    teacher_z = standardized[TEACHER_NAME]
    for student_name in (STUDENT_ONLY_NAME, SEAL_NAME):
        student_z = standardized[student_name]
        dims_match = teacher_z.shape[1] == student_z.shape[1]
        for class_id in sorted(CLASS_LABELS):
            mask = y_true == class_id
            teacher_class = teacher_z[mask]
            student_class = student_z[mask]
            cka_rows.append(
                {
                    "student_model": student_name,
                    "class_id": int(class_id),
                    "class_label": CLASS_LABELS[class_id],
                    "n_samples": int(mask.sum()),
                    "linear_cka": float(linear_cka(teacher_class, student_class)),
                    "teacher_latent_dim": int(teacher_z.shape[1]),
                    "student_latent_dim": int(student_z.shape[1]),
                }
            )
            if not dims_match:
                raise ValueError(
                    f"{projection_space} centroid cosine requires matching dims for {student_name}: "
                    f"teacher={teacher_z.shape[1]} student={student_z.shape[1]}"
                )
            centroid_rows.append(
                {
                    "student_model": student_name,
                    "class_id": int(class_id),
                    "class_label": CLASS_LABELS[class_id],
                    "n_samples": int(mask.sum()),
                    "cosine_similarity": float(cosine_between(teacher_class.mean(axis=0), student_class.mean(axis=0))),
                    "teacher_latent_dim": int(teacher_z.shape[1]),
                    "student_latent_dim": int(student_z.shape[1]),
                    "projection_space": projection_space,
                }
            )

        for bin_name in BOUNDARY_BIN_ORDER:
            mask = annotated_identity["boundary_distance_bin"].astype(str).eq(bin_name).to_numpy()
            teacher_bin = teacher_z[mask]
            student_bin = student_z[mask]
            if dims_match and int(mask.sum()) > 0:
                mean_cos = float(np.nanmean(paired_cosine(teacher_bin, student_bin)))
                mse = float(np.mean((teacher_bin - student_bin) ** 2))
            else:
                mean_cos = float("nan")
                mse = float("nan")
            boundary_rows.append(
                {
                    "student_model": student_name,
                    "boundary_distance_bin": bin_name,
                    "n_samples": int(mask.sum()),
                    "linear_cka": float(linear_cka(teacher_bin, student_bin)),
                    "mean_paired_cosine": mean_cos,
                    "mean_teacher_student_mse": mse,
                    "projection_space": projection_space,
                }
            )

    return pd.DataFrame(centroid_rows), pd.DataFrame(cka_rows), pd.DataFrame(boundary_rows)


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


def build_annotated_identity(
    identities: Mapping[str, np.ndarray],
    raw_data_dir: Path,
    results: Mapping[str, InferenceResult],
) -> pd.DataFrame:
    raw_data_dir = resolve_repo_path(raw_data_dir)
    boundary_by_seam: dict[str, tuple[int, int]] = {}
    for seam_name in sorted(set(np.asarray(identities["seam_name"]).astype(str).tolist())):
        _data, labels = load_seam_csv(raw_data_dir / f"{seam_name}.csv")
        boundary_by_seam[seam_name] = compute_transition_boundaries(labels)

    rows: list[dict[str, Any]] = []
    n = int(np.asarray(identities["sample_index"]).shape[0])
    for idx in range(n):
        seam_name = str(identities["seam_name"][idx])
        target_idx = int(identities["target_idx"][idx])
        boundary_01, boundary_12 = boundary_by_seam[seam_name]
        dist_01 = abs(target_idx - boundary_01)
        dist_12 = abs(target_idx - boundary_12)
        if dist_01 <= dist_12:
            nearest_idx = boundary_01
            boundary_type = "0->1"
            distance = dist_01
        else:
            nearest_idx = boundary_12
            boundary_type = "1->2"
            distance = dist_12
        row = {
            "sample_index": int(identities["sample_index"][idx]),
            "seam_id": int(identities["seam_id"][idx]),
            "seam_name": seam_name,
            "start_idx": int(identities["start_idx"][idx]),
            "target_idx": target_idx,
            "y_true": int(identities["y_true"][idx]),
            "boundary_type": boundary_type,
            "nearest_transition_idx": int(nearest_idx),
            "nearest_boundary_distance": int(distance),
            "boundary_distance_bin": classify_boundary_distance_bin(distance),
        }
        for model_name in MODEL_ORDER:
            slug = METHOD_SLUGS[model_name]
            pred = int(results[model_name].y_pred[idx])
            row[f"{slug}_pred"] = pred
            row[f"{slug}_correct"] = bool(pred == row["y_true"])
        rows.append(row)
    return pd.DataFrame(rows)


def select_boundary_windows(annotated: pd.DataFrame, max_per_boundary_type: int = 16) -> pd.DataFrame:
    selected_frames: list[pd.DataFrame] = []
    for boundary_type in ("0->1", "1->2"):
        candidates = annotated.loc[annotated["boundary_type"].eq(boundary_type)].copy()
        if candidates.empty:
            continue
        candidates["preferred"] = (
            candidates["boundary_distance_bin"].eq("Near")
            & candidates["teacher_correct"].astype(bool)
            & candidates["seal_weld_student_correct"].astype(bool)
        )
        candidates = candidates.sort_values(
            by=["preferred", "nearest_boundary_distance", "sample_index"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        selected_frames.append(candidates.head(max_per_boundary_type))
    if not selected_frames:
        raise ValueError("No boundary windows available for Figure 3 attention panels")
    selected = pd.concat(selected_frames, ignore_index=True)
    selected = selected.sort_values(
        by=["boundary_type", "nearest_boundary_distance", "sample_index"],
        ascending=[True, True, True],
        kind="mergesort",
    ).drop(columns=["preferred"], errors="ignore")
    selected["display_mode"] = "sample_heatmap" if len(selected) <= 32 else "mean_attention"
    return selected.reset_index(drop=True)


def build_boundary_attention_heatmap(
    selected: pd.DataFrame,
    results: Mapping[str, InferenceResult],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_indices = selected["sample_index"].astype(int).tolist()
    selected_by_index = {int(row.sample_index): row for row in selected.itertuples(index=False)}
    for model_name in MODEL_ORDER:
        result = results[model_name]
        slug = METHOD_SLUGS[model_name]
        for sample_index in selected_indices:
            row = selected_by_index[sample_index]
            pos = int(sample_index)
            start_idx = int(row.start_idx)
            target_idx = int(row.target_idx)
            for input_step in range(WINDOW_SIZE):
                raw_idx = start_idx + input_step
                rows.append(
                    {
                        "model_name": model_name,
                        "method_slug": slug,
                        "sample_index": sample_index,
                        "seam_name": str(row.seam_name),
                        "boundary_type": str(row.boundary_type),
                        "boundary_distance_bin": str(row.boundary_distance_bin),
                        "start_idx": start_idx,
                        "target_idx": target_idx,
                        "input_step": int(input_step),
                        "relative_step": int(raw_idx - target_idx),
                        "relative_time_s": float((raw_idx - target_idx) * SAMPLE_PERIOD_S),
                        "attention_weight": float(result.attention_ribbon[pos, input_step]),
                        "y_true": int(result.y_true[pos]),
                        "y_pred": int(result.y_pred[pos]),
                        "correct": bool(result.y_pred[pos] == result.y_true[pos]),
                    }
                )
    return pd.DataFrame(rows)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if f1_score is not None:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    values = []
    for class_id in sorted(CLASS_LABELS):
        tp = float(np.sum((y_true == class_id) & (y_pred == class_id)))
        fp = float(np.sum((y_true != class_id) & (y_pred == class_id)))
        fn = float(np.sum((y_true == class_id) & (y_pred != class_id)))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    return float(np.mean(values))


def parse_evaluation_metrics(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    test_match = re.search(r"--- Test Metrics.*?(?=(?:\n---|\Z))", text, flags=re.S)
    section = test_match.group(0) if test_match else text
    accuracy_match = re.search(r"Accuracy:\s*([0-9.]+)%", section)
    macro_match = re.search(r"Macro-F1:\s*([0-9.]+)", section)
    if not accuracy_match or not macro_match:
        raise ValueError(f"Could not parse test Accuracy and Macro-F1 from {path}")
    return {
        "source_accuracy": float(accuracy_match.group(1)) / 100.0,
        "source_macro_f1": float(macro_match.group(1)),
    }


def build_reproduced_metrics_table(results: Mapping[str, InferenceResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        result = results[model_name]
        accuracy = float(np.mean(result.y_pred == result.y_true))
        macro_f1 = _macro_f1(result.y_true, result.y_pred)
        source = parse_evaluation_metrics(result.spec.run_dir / "evaluation_metrics.txt")
        rows.append(
            {
                "model_name": model_name,
                "method_slug": result.spec.method_slug,
                "n_samples": int(result.y_true.shape[0]),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "source_accuracy": source["source_accuracy"],
                "source_macro_f1": source["source_macro_f1"],
                "accuracy_abs_delta_pct_points": abs(accuracy - source["source_accuracy"]) * 100.0,
                "macro_f1_abs_delta_pct_points": abs(macro_f1 - source["source_macro_f1"]) * 100.0,
            }
        )
    return pd.DataFrame(rows)


def write_selected_model_runs(specs: Iterable[ModelSpec], results: Mapping[str, InferenceResult], path: Path) -> None:
    rows = []
    for spec in specs:
        rows.append(
            {
                "model_name": spec.model_name,
                "method_slug": spec.method_slug,
                "role": spec.role,
                "run_dir": str(spec.run_dir),
                "checkpoint_path": str(spec.checkpoint_path),
                "checkpoint_file": spec.checkpoint_path.name,
                "input_dim": int(len(spec.keep_feature_indices)),
                "keep_feature_indices": json.dumps(list(spec.keep_feature_indices)),
                "latent_dim": int(results[spec.model_name].z_test.shape[1]),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def validate_export(
    *,
    dataset_path: Path,
    dataset: Mapping[str, np.ndarray],
    identities: Mapping[str, np.ndarray],
    results: Mapping[str, InferenceResult],
    projection_df: pd.DataFrame,
    projection_space: str,
    metrics_df: pd.DataFrame,
) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append((name, bool(passed), detail))

    add(
        "dataset_path_is_h1",
        dataset_path.resolve() == (PROJECT_ROOT / EXPECTED_DATASET).resolve(),
        str(dataset_path.resolve()),
    )
    y_test = np.asarray(dataset["y_test"], dtype=np.int64)
    identity_cols = ("sample_index", "seam_id", "start_idx", "target_idx")
    first = None
    for model_name in MODEL_ORDER:
        result = results[model_name]
        add(f"{model_name}_embedding_rows", result.z_test.shape[0] == EXPECTED_TEST_SAMPLES, str(result.z_test.shape))
        add(f"{model_name}_y_true_matches_dataset", np.array_equal(result.y_true, y_test))
        if first is None:
            first = result
        else:
            add(f"{model_name}_sample_order_matches_teacher", np.array_equal(result.y_true, first.y_true))
        add(f"{model_name}_prob_rows_sum_to_one", bool(np.allclose(result.prob.sum(axis=1), 1.0, atol=1e-5)))
        add(f"{model_name}_z_finite", bool(np.isfinite(result.z_test).all()))
        add(f"{model_name}_attention_finite", bool(np.isfinite(result.attention_weights).all()))
        add(f"{model_name}_attention_ribbon_sums_to_one", bool(np.allclose(result.attention_ribbon.sum(axis=1), 1.0, atol=1e-5)))

    for col in identity_cols:
        expected = np.asarray(identities[col])
        for model_name in MODEL_ORDER:
            actual = np.asarray(getattr(results[model_name], col))
            add(f"{model_name}_identity_{col}_matches_dataset", np.array_equal(actual, expected))

    per_model_counts = projection_df.groupby("model_name").size().to_dict()
    add(
        "projection_fitted_once_on_concatenated_latents",
        all(per_model_counts.get(name) == EXPECTED_TEST_SAMPLES for name in MODEL_ORDER)
        and projection_df["projection_random_state"].nunique() == 1,
        str(per_model_counts),
    )
    add(
        "projection_space_strict_raw_latent",
        projection_space == "strict_raw_latent"
        and set(projection_df["projection_space"].astype(str)) == {"strict_raw_latent"},
    )
    add(
        "reproduced_metrics_within_0p1_pct_points",
        bool(
            (metrics_df["accuracy_abs_delta_pct_points"] <= 0.1).all()
            and (metrics_df["macro_f1_abs_delta_pct_points"] <= 0.1).all()
        ),
        metrics_df[
            ["model_name", "accuracy_abs_delta_pct_points", "macro_f1_abs_delta_pct_points"]
        ].to_dict(orient="records"),
    )
    return checks


def write_qa_notes(path: Path, checks: Iterable[tuple[str, bool, str]], error: Exception | None = None) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for name, passed, detail in checks:
            status = "PASS" if passed else "FAIL"
            handle.write(f"{status}\t{name}\t{detail}\n")
        if error is not None:
            handle.write(f"FAIL\texception\t{type(error).__name__}: {error}\n")


def write_manifest(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "value"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"key": row["key"], "value": row["value"]})


def caption_text(projection_method: str, projection_space: str) -> str:
    aligned_note = (
        " Panels (a) through (c) use a post-hoc aligned latent space."
        if projection_space == "aligned_latent"
        else ""
    )
    return (
        "Latent distillation moves the deployable SEAL-Weld Student toward the teacher representation "
        "and attention pattern near state boundaries. Panels (a) through (c) show one "
        f"{projection_method.upper()} projection fitted on the concatenated latent matrix from all three models."
        f"{aligned_note} Panels (d) through (f) quantify teacher-student alignment in the declared "
        f"{projection_space} space and class- or boundary-conditioned subsets. Panels (g) through (i) "
        "use identical selected boundary windows for every model. Data are from the H1 weld seam "
        "classification task with S0, S1, and S2 labels."
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = prepare_versioned_output_dir(args.output_root)
    qa_path = output_dir / "qa_notes.txt"
    checks: list[tuple[str, bool, str]] = []
    try:
        dataset_path = resolve_repo_path(args.dataset_npz)
        dataset = load_dataset(dataset_path)
        identities = build_identity_arrays(dataset)
        device = resolve_device(args.device)
        specs = build_model_specs(args, dataset)
        results = export_embeddings(specs, dataset, output_dir, device=device, batch_size=args.batch_size)
        annotated_identity = build_annotated_identity(identities, RAW_DATA_DIR, results)
        projection_df, resolved_projection_method = build_projection_table(
            results,
            identities,
            projection_method=args.projection_method,
            projection_space=PROJECTION_SPACE,
            random_state=args.random_state,
            allow_pca_draft=args.allow_pca_draft,
        )
        centroid_df, cka_df, boundary_similarity_df = compute_alignment_tables(
            results,
            annotated_identity,
            projection_space=PROJECTION_SPACE,
        )
        selected_windows_df = select_boundary_windows(annotated_identity)
        attention_df = build_boundary_attention_heatmap(selected_windows_df, results)
        metrics_df = build_reproduced_metrics_table(results)

        source_dir = output_dir / "source_tables"
        write_selected_model_runs(specs, results, source_dir / "selected_model_runs.csv")
        projection_df.to_csv(source_dir / "unified_projection_coordinates.csv", index=False)
        centroid_df.to_csv(source_dir / "class_centroid_cosine_similarity.csv", index=False)
        cka_df.to_csv(source_dir / "linear_cka_by_class.csv", index=False)
        boundary_similarity_df.to_csv(source_dir / "similarity_by_boundary_bin.csv", index=False)
        selected_windows_df.to_csv(source_dir / "selected_boundary_windows.csv", index=False)
        attention_df.to_csv(source_dir / "boundary_attention_heatmap.csv", index=False)
        metrics_df.to_csv(source_dir / "reproduced_classification_metrics.csv", index=False)

        checks = validate_export(
            dataset_path=dataset_path,
            dataset=dataset,
            identities=identities,
            results=results,
            projection_df=projection_df,
            projection_space=PROJECTION_SPACE,
            metrics_df=metrics_df,
        )
        failed = [name for name, passed, _detail in checks if not passed]
        if failed:
            raise RuntimeError(f"Figure 3 export validation failed: {failed}")

        manifest_rows = [
            {"key": "dataset_npz", "value": str(dataset_path)},
            {"key": "output_dir", "value": str(output_dir)},
            {"key": "projection_method", "value": resolved_projection_method},
            {"key": "projection_space", "value": PROJECTION_SPACE},
            {"key": "projection_random_state", "value": int(args.random_state)},
            {"key": "importance_method", "value": "attention"},
            {"key": "attention_display_mode", "value": str(selected_windows_df["display_mode"].iloc[0])},
        ]
        for spec in specs:
            manifest_rows.extend(
                [
                    {"key": f"{spec.method_slug}_run_dir", "value": str(spec.run_dir)},
                    {"key": f"{spec.method_slug}_checkpoint_path", "value": str(spec.checkpoint_path)},
                ]
            )
        write_manifest(output_dir / "manifest.csv", manifest_rows)
        (output_dir / "figure3_caption.txt").write_text(
            caption_text(resolved_projection_method, PROJECTION_SPACE) + "\n",
            encoding="utf-8",
        )
        write_qa_notes(qa_path, checks)
        print(output_dir)
    except Exception as exc:
        write_qa_notes(qa_path, checks, error=exc)
        raise


if __name__ == "__main__":
    main()
