#!/usr/bin/env python3
"""Run horizon=1 representation analysis for the Figure 3 comparison models."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.manifold import TSNE
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, f1_score, silhouette_score
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover - sklearn is required for this script.
    raise RuntimeError("scikit-learn is required for representation analysis") from exc

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover - optional dependency.
    umap = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training_utils import validate_forecast_dataset_contract  # noqa: E402

from ablation_experiments.scripts.export_figure3_roc_predictions import (  # noqa: E402
    build_identity_arrays,
    build_model_for_method,
    checkpoint_filename_for_method,
    load_run_args,
    load_strict_state_dict,
    resolve_keep_feature_indices,
    resolve_run_path,
)


METHOD_SPECS = (
    {
        "method_name": "teacher(18D upper bound)",
        "display_name": "Teacher",
        "slug": "teacher",
        "feature_layer": "attention_tcn.z",
    },
    {
        "method_name": "teacher-student(student)",
        "display_name": "Student",
        "slug": "student",
        "feature_layer": "attention_tcn.z",
    },
    {
        "method_name": "gru",
        "display_name": "GRU",
        "slug": "gru",
        "feature_layer": "gru.pooled_hidden",
    },
    {
        "method_name": "lstm",
        "display_name": "LSTM",
        "slug": "lstm",
        "feature_layer": "lstm.classifier_fc1_relu",
    },
    {
        "method_name": "inception",
        "display_name": "Inception",
        "slug": "inception",
        "feature_layer": "inception.pooled_z",
    },
    {
        "method_name": "transformer",
        "display_name": "Transformer",
        "slug": "transformer",
        "feature_layer": "transformer.pooled_encoder",
    },
)

CLASS_LABELS = {0: "S0", 1: "S1", 2: "S2"}
CLASS_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
MAX_METRIC_DELTA_PERCENT_POINTS = 0.1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H1 point-cloud representation analysis")
    parser.add_argument(
        "--registry",
        type=Path,
        default=PROJECT_ROOT / "outputs/figure 3/source_tables/comparison_summary.csv",
        help="Figure 3 comparison registry CSV.",
    )
    parser.add_argument(
        "--dataset-npz",
        type=Path,
        default=PROJECT_ROOT / "Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz",
        help="H1 dataset used for every model.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/representation_analysis/h1",
        help="Directory for embeddings, metrics, projections, figures, and QA outputs.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args(argv)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def prepare_output_dirs(output_root: Path) -> dict[str, Path]:
    paths = {
        "root": output_root,
        "embeddings": output_root / "embeddings",
        "projections": output_root / "projections",
        "metrics": output_root / "metrics",
        "figures": output_root / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"cannot write empty csv: {path}")
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_embedding_file(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    np.savez(path, **payload)


def load_dataset(npz_path: Path) -> dict[str, np.ndarray]:
    validate_forecast_dataset_contract(npz_path, expected_horizon=1, expected_delta=5)
    with np.load(npz_path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def load_registry_rows(registry_path: Path) -> list[dict[str, str]]:
    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"method_name", "run_path", "best_test_acc", "best_macro_f1"}
    missing_columns = required.difference(rows[0].keys() if rows else set())
    if missing_columns:
        raise ValueError(f"registry missing required columns: {sorted(missing_columns)}")
    rows_by_name = {row["method_name"]: row for row in rows}
    missing_methods = [spec["method_name"] for spec in METHOD_SPECS if spec["method_name"] not in rows_by_name]
    if missing_methods:
        raise ValueError(f"registry missing required methods: {missing_methods}")
    return [rows_by_name[spec["method_name"]] for spec in METHOD_SPECS]


def resolve_projection_perplexity(n_test: int) -> int:
    if n_test < 2:
        raise ValueError("t-SNE requires at least two samples")
    if n_test < 31:
        return max(1, min(30, (n_test - 1) // 3))
    return 30


def standardize_latents(z_test: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(np.asarray(z_test, dtype=np.float64)).astype(np.float32)


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def compute_representation_metrics(z_std: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=np.int64)
    features = np.asarray(z_std, dtype=np.float64)
    unique_labels = sorted(np.unique(labels).tolist())
    centroids = {
        label: features[labels == label].mean(axis=0)
        for label in unique_labels
    }

    intra_distances: list[float] = []
    for label in unique_labels:
        class_features = features[labels == label]
        centroid = centroids[label]
        intra_distances.extend(np.linalg.norm(class_features - centroid, axis=1).tolist())

    inter_centroid_distances: list[float] = []
    for i, left in enumerate(unique_labels):
        for right in unique_labels[i + 1 :]:
            dist = float(np.linalg.norm(centroids[left] - centroids[right]))
            inter_centroid_distances.append(dist)

    mean_intra = float(np.mean(intra_distances))
    mean_inter = float(np.mean(inter_centroid_distances))
    ratio = mean_inter / mean_intra if mean_intra > 0.0 else math.inf
    return {
        "silhouette_score": float(silhouette_score(features, labels)),
        "davies_bouldin_index": float(davies_bouldin_score(features, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(features, labels)),
        "mean_intra_class_distance": mean_intra,
        "mean_inter_class_centroid_distance": mean_inter,
        "inter_intra_distance_ratio": float(ratio),
    }


def run_inference_with_representations(
    model: torch.nn.Module,
    method_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logits_batches: list[np.ndarray] = []
    prob_batches: list[np.ndarray] = []
    pred_batches: list[np.ndarray] = []
    z_batches: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            if method_name == "lstm" and hasattr(model, "forward_features"):
                logits, z = model.forward_features(x_batch)
            else:
                logits, z = model(x_batch)
            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)
            logits_batches.append(logits.cpu().numpy())
            prob_batches.append(prob.cpu().numpy())
            pred_batches.append(pred.cpu().numpy())
            z_batches.append(z.cpu().numpy())

    logits_np = np.concatenate(logits_batches, axis=0)
    prob_np = np.concatenate(prob_batches, axis=0)
    pred_np = np.concatenate(pred_batches, axis=0)
    z_np = np.concatenate(z_batches, axis=0)
    return pred_np, logits_np, prob_np, z_np


def validate_embedding_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    logits: np.ndarray,
    probabilities: np.ndarray,
    z_test: np.ndarray,
    expected_n_test: int,
) -> None:
    if y_true.shape != (expected_n_test,):
        raise ValueError(f"y_true shape mismatch: expected {(expected_n_test,)}, got {tuple(y_true.shape)}")
    if y_pred.shape != (expected_n_test,):
        raise ValueError(f"y_pred shape mismatch: expected {(expected_n_test,)}, got {tuple(y_pred.shape)}")
    if logits.shape != (expected_n_test, 3):
        raise ValueError(f"logits shape mismatch: expected {(expected_n_test, 3)}, got {tuple(logits.shape)}")
    if probabilities.shape != (expected_n_test, 3):
        raise ValueError(
            f"probability shape mismatch: expected {(expected_n_test, 3)}, got {tuple(probabilities.shape)}"
        )
    if z_test.shape[0] != expected_n_test:
        raise ValueError(f"Z_test row count mismatch: expected {expected_n_test}, got {z_test.shape[0]}")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("probability rows must sum to 1.0 within tolerance 1e-5")
    if not np.all(np.isfinite(z_test)):
        raise ValueError("Z_test contains NaN or infinite values")


def build_embedding_payload(
    *,
    z_test: np.ndarray,
    logits: np.ndarray,
    prob: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    identities: Mapping[str, np.ndarray],
    display_name: str,
    run_path: Path,
    checkpoint_path: Path,
    dataset_npz: Path,
    feature_layer: str,
) -> dict[str, np.ndarray]:
    payload = {
        "Z_test": np.asarray(z_test, dtype=np.float32),
        "logits": np.asarray(logits, dtype=np.float32),
        "prob": np.asarray(prob, dtype=np.float32),
        "y_true": np.asarray(y_true, dtype=np.int64),
        "y_pred": np.asarray(y_pred, dtype=np.int64),
        "sample_index": np.asarray(identities["sample_index"], dtype=np.int64),
        "model_name": np.array(display_name),
        "run_path": np.array(str(run_path)),
        "checkpoint_path": np.array(str(checkpoint_path)),
        "dataset_npz": np.array(str(dataset_npz)),
        "feature_layer": np.array(feature_layer),
    }
    for key in ("seam_id", "seam_name", "start_idx", "target_idx"):
        if key in identities:
            payload[key] = np.asarray(identities[key])
    return payload


def project_tsne(z_std: np.ndarray, random_state: int) -> np.ndarray:
    perplexity = resolve_projection_perplexity(z_std.shape[0])
    kwargs: dict[str, Any] = {
        "n_components": 2,
        "random_state": random_state,
        "perplexity": perplexity,
        "init": "pca",
    }
    try:
        tsne = TSNE(learning_rate="auto", **kwargs)
    except TypeError:
        tsne = TSNE(**kwargs)
    return np.asarray(tsne.fit_transform(z_std), dtype=np.float32)


def project_umap_if_available(z_std: np.ndarray, random_state: int) -> np.ndarray | None:
    if umap is None:
        return None
    reducer = umap.UMAP(n_components=2, random_state=random_state)
    return np.asarray(reducer.fit_transform(z_std), dtype=np.float32)


def render_tsne_figure(tsne_df: pd.DataFrame, output_prefix: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4))
    axes_flat = axes.flatten()

    for ax, spec in zip(axes_flat, METHOD_SPECS):
        display_name = spec["display_name"]
        subset = tsne_df.loc[tsne_df["model_name"] == display_name]
        for class_id in sorted(CLASS_LABELS):
            class_rows = subset.loc[subset["y_true"] == class_id]
            ax.scatter(
                class_rows["x_2d"],
                class_rows["y_2d"],
                s=10,
                alpha=0.8,
                color=CLASS_COLORS[class_id],
                label=CLASS_LABELS[class_id],
                linewidths=0.0,
            )
        ax.set_title(display_name)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.grid(alpha=0.15, linewidth=0.4)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=5,
            markerfacecolor=CLASS_COLORS[class_id],
            markeredgecolor=CLASS_COLORS[class_id],
            label=CLASS_LABELS[class_id],
        )
        for class_id in sorted(CLASS_LABELS)
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.91, 0.5), frameon=False)
    fig.subplots_adjust(right=0.87, wspace=0.28, hspace=0.34)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_prefix.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_qa_report(path: Path, failures: list[str], exc: Exception | None = None) -> None:
    lines = ["# H1 representation analysis QA report", ""]
    if failures:
        lines.append("## Validation failures")
        lines.extend([f"- {failure}" for failure in failures])
        lines.append("")
    if exc is not None:
        lines.append("## Exception")
        lines.append("")
        lines.append("```text")
        lines.append("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip())
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    output_dirs = prepare_output_dirs(args.output_root)
    qa_report_path = output_dirs["root"] / "qa_report.md"

    try:
        registry_rows = load_registry_rows(args.registry)
        dataset_path = args.dataset_npz.resolve()
        dataset = load_dataset(dataset_path)
        identities = build_identity_arrays(dataset)
        y_expected = np.asarray(dataset["y_test"], dtype=np.int64)
        n_test = int(y_expected.shape[0])
        num_classes = int(np.max(y_expected) + 1)
        if num_classes != 3:
            raise ValueError(f"expected 3 classes for H1 analysis, got {num_classes}")

        manifest_rows: list[dict[str, Any]] = []
        classification_rows: list[dict[str, Any]] = []
        representation_rows: list[dict[str, Any]] = []
        tsne_rows: list[dict[str, Any]] = []
        umap_rows: list[dict[str, Any]] = []
        qa_failures: list[str] = []
        embedding_sample_counts: set[int] = set()
        umap_note = "" if umap is not None else "umap skipped: umap-learn not installed"

        for spec, method_row in zip(METHOD_SPECS, registry_rows):
            method_name = spec["method_name"]
            display_name = spec["display_name"]
            feature_layer = spec["feature_layer"]
            run_dir = resolve_run_path(PROJECT_ROOT, method_row["run_path"])
            checkpoint_path = run_dir / checkpoint_filename_for_method(method_name)
            run_args = load_run_args(run_dir)
            run_dataset = Path(str(run_args.get("dataset_npz", dataset_path))).expanduser().resolve()
            if run_dataset != dataset_path:
                raise ValueError(
                    f"{display_name} run dataset mismatch: run_args={run_dataset} expected={dataset_path}"
                )

            x_test = np.asarray(dataset["X_test_full"], dtype=np.float32)
            if method_name == "teacher-student(student)":
                keep_indices = resolve_keep_feature_indices(run_args, input_dim=x_test.shape[2])
                x_eval = x_test[:, :, keep_indices]
            else:
                keep_indices = list(range(x_test.shape[2]))
                x_eval = x_test

            model = build_model_for_method(
                method_name,
                run_args,
                input_dim=int(x_eval.shape[2]),
                num_classes=num_classes,
            ).to(device)
            load_strict_state_dict(model, checkpoint_path, device=device)

            y_pred, logits, prob, z_test = run_inference_with_representations(
                model=model,
                method_name=method_name,
                features=x_eval,
                labels=y_expected,
                device=device,
                batch_size=args.batch_size,
            )
            validate_embedding_outputs(
                y_true=y_expected,
                y_pred=y_pred,
                logits=logits,
                probabilities=prob,
                z_test=z_test,
                expected_n_test=n_test,
            )
            if not np.array_equal(y_expected, np.asarray(identities["y_true"], dtype=np.int64)):
                raise ValueError(f"{display_name} y_true does not match dataset y_test")

            embedding_payload = build_embedding_payload(
                z_test=z_test,
                logits=logits,
                prob=prob,
                y_true=y_expected,
                y_pred=y_pred,
                identities=identities,
                display_name=display_name,
                run_path=run_dir,
                checkpoint_path=checkpoint_path,
                dataset_npz=dataset_path,
                feature_layer=feature_layer,
            )
            embedding_path = output_dirs["embeddings"] / f"{spec['slug']}_embeddings_test.npz"
            write_embedding_file(embedding_path, embedding_payload)
            embedding_sample_counts.add(int(z_test.shape[0]))

            accuracy = float(np.mean(y_pred == y_expected))
            macro_f1 = compute_macro_f1(y_expected, y_pred)
            registry_acc = float(method_row["best_test_acc"])
            registry_f1 = float(method_row["best_macro_f1"])
            accuracy_delta_pp = (accuracy - registry_acc) * 100.0
            macro_f1_delta_pp = (macro_f1 - registry_f1) * 100.0
            within_tolerance = (
                abs(accuracy_delta_pp) <= MAX_METRIC_DELTA_PERCENT_POINTS
                and abs(macro_f1_delta_pp) <= MAX_METRIC_DELTA_PERCENT_POINTS
            )
            if not within_tolerance:
                qa_failures.append(
                    f"{display_name} classification metrics differ from Figure 3 by more than "
                    f"{MAX_METRIC_DELTA_PERCENT_POINTS:.1f} percentage points"
                )

            z_std = standardize_latents(z_test)
            repr_metrics = compute_representation_metrics(z_std, y_expected)
            latent_dim = int(z_test.shape[1])
            tsne_coords = project_tsne(z_std, random_state=args.random_state)
            perplexity = resolve_projection_perplexity(n_test)
            for idx in range(n_test):
                tsne_rows.append(
                    {
                        "model_name": display_name,
                        "sample_index": int(identities["sample_index"][idx]),
                        "x_2d": float(tsne_coords[idx, 0]),
                        "y_2d": float(tsne_coords[idx, 1]),
                        "y_true": int(y_expected[idx]),
                        "y_pred": int(y_pred[idx]),
                        "correct": bool(y_pred[idx] == y_expected[idx]),
                        "seam_name": str(identities["seam_name"][idx]),
                        "start_idx": int(identities["start_idx"][idx]),
                        "target_idx": int(identities["target_idx"][idx]),
                        "projection_method": "tsne",
                        "projection_random_state": int(args.random_state),
                    }
                )

            umap_coords = project_umap_if_available(z_std, random_state=args.random_state)
            if umap_coords is not None:
                for idx in range(n_test):
                    umap_rows.append(
                        {
                            "model_name": display_name,
                            "sample_index": int(identities["sample_index"][idx]),
                            "x_2d": float(umap_coords[idx, 0]),
                            "y_2d": float(umap_coords[idx, 1]),
                            "y_true": int(y_expected[idx]),
                            "y_pred": int(y_pred[idx]),
                            "correct": bool(y_pred[idx] == y_expected[idx]),
                            "seam_name": str(identities["seam_name"][idx]),
                            "start_idx": int(identities["start_idx"][idx]),
                            "target_idx": int(identities["target_idx"][idx]),
                            "projection_method": "umap",
                            "projection_random_state": int(args.random_state),
                        }
                    )

            classification_rows.append(
                {
                    "model_name": display_name,
                    "method_name": method_name,
                    "n_test": n_test,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "figure3_accuracy": registry_acc,
                    "figure3_macro_f1": registry_f1,
                    "accuracy_delta_pct_points": accuracy_delta_pp,
                    "macro_f1_delta_pct_points": macro_f1_delta_pp,
                    "within_0_1_pp_tolerance": bool(within_tolerance),
                    "run_path": str(run_dir),
                    "checkpoint_path": str(checkpoint_path),
                }
            )
            representation_rows.append(
                {
                    "model_name": display_name,
                    "method_name": method_name,
                    "n_test": n_test,
                    "latent_dim": latent_dim,
                    "feature_layer": feature_layer,
                    **repr_metrics,
                }
            )
            manifest_rows.append(
                {
                    "model_name": display_name,
                    "method_name": method_name,
                    "run_path": str(run_dir),
                    "checkpoint_path": str(checkpoint_path),
                    "dataset_npz": str(dataset_path),
                    "feature_layer": feature_layer,
                    "projection_method": "tsne",
                    "projection_random_state": int(args.random_state),
                    "projection_perplexity": int(perplexity),
                    "umap_available": bool(umap is not None),
                    "embedding_path": str(embedding_path),
                    "accuracy_delta_pct_points": accuracy_delta_pp,
                    "macro_f1_delta_pct_points": macro_f1_delta_pp,
                    "notes": umap_note,
                }
            )

        tsne_path = output_dirs["projections"] / "tsne_coordinates.csv"
        write_csv(tsne_path, tsne_rows)
        if umap_rows:
            write_csv(output_dirs["projections"] / "umap_coordinates.csv", umap_rows)

        classification_path = output_dirs["metrics"] / "reproduced_classification_metrics.csv"
        representation_path = output_dirs["metrics"] / "representation_metrics.csv"
        manifest_path = output_dirs["root"] / "manifest.csv"
        write_csv(classification_path, classification_rows)
        write_csv(representation_path, representation_rows)
        write_csv(manifest_path, manifest_rows)

        tsne_df = pd.DataFrame(tsne_rows)
        figure_prefix = output_dirs["figures"] / "h1_representation_tsne"
        render_tsne_figure(tsne_df, figure_prefix)

        if len(embedding_sample_counts) != 1:
            qa_failures.append("embedding files do not share the same n_test")
        for ext in ("png", "pdf", "svg"):
            if not figure_prefix.with_suffix(f".{ext}").exists():
                qa_failures.append(f"missing figure export: {figure_prefix.with_suffix(f'.{ext}')}")
        if not manifest_rows:
            qa_failures.append("manifest.csv did not record any runs")
        if qa_failures:
            write_qa_report(qa_report_path, qa_failures)
            raise RuntimeError("representation analysis validation failed")
        if qa_report_path.exists():
            qa_report_path.unlink()
    except Exception as exc:
        if not qa_report_path.exists():
            write_qa_report(qa_report_path, failures=[], exc=exc)
        raise


if __name__ == "__main__":
    main()
