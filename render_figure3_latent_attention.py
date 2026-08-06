#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Mapping

import matplotlib

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
FIGURE_BASENAME = "figure3_latent_attention"
MODEL_ORDER = ("Teacher", "Student-only", "SEAL-Weld Student")
STUDENT_ORDER = ("Student-only", "SEAL-Weld Student")
CLASS_ORDER = (0, 1, 2)
CLASS_LABELS = {0: "S0", 1: "S1", 2: "S2"}
CLASS_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
STUDENT_COLORS = {"Student-only": "#8C6D31", "SEAL-Weld Student": "#0B5CAD"}
BOUNDARY_ORDER = ("Far", "Mid", "Near")
SOURCE_TABLES = {
    "selected_model_runs": "selected_model_runs.csv",
    "projection": "unified_projection_coordinates.csv",
    "centroid": "class_centroid_cosine_similarity.csv",
    "cka": "linear_cka_by_class.csv",
    "boundary_similarity": "similarity_by_boundary_bin.csv",
    "selected_windows": "selected_boundary_windows.csv",
    "attention": "boundary_attention_heatmap.csv",
    "metrics": "reproduced_classification_metrics.csv",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Figure 3 from exported source tables.")
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser.parse_args(argv)


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.6,
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_source_tables(input_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = resolve_repo_path(input_dir)
    source_dir = root / "source_tables"
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in SOURCE_TABLES.items():
        path = source_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required source table: {path}")
        tables[key] = pd.read_csv(path)
    return tables


def require_columns(df: pd.DataFrame, required: set[str], *, table_name: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{table_name} missing required columns: {sorted(missing)}")


def panel_title(label: str, title: str) -> str:
    return f"{label} {title}"


def draw_projection_row(axes: np.ndarray, projection_df: pd.DataFrame) -> None:
    require_columns(
        projection_df,
        {"model_name", "x_2d", "y_2d", "y_true", "projection_method", "projection_space"},
        table_name="unified_projection_coordinates.csv",
    )
    x = projection_df["x_2d"].to_numpy(dtype=float)
    y = projection_df["y_2d"].to_numpy(dtype=float)
    x_margin = max((float(np.nanmax(x)) - float(np.nanmin(x))) * 0.04, 1e-3)
    y_margin = max((float(np.nanmax(y)) - float(np.nanmin(y))) * 0.04, 1e-3)
    x_limits = (float(np.nanmin(x)) - x_margin, float(np.nanmax(x)) + x_margin)
    y_limits = (float(np.nanmin(y)) - y_margin, float(np.nanmax(y)) + y_margin)
    labels = ("(a)", "(b)", "(c)")
    titles = ("Teacher latent", "Student-only latent", "SEAL-Weld Student latent")
    for ax, model_name, label, title in zip(axes, MODEL_ORDER, labels, titles):
        subset = projection_df.loc[projection_df["model_name"].astype(str).eq(model_name)].copy()
        for class_id in CLASS_ORDER:
            class_subset = subset.loc[subset["y_true"].astype(int).eq(class_id)]
            ax.scatter(
                class_subset["x_2d"],
                class_subset["y_2d"],
                s=8,
                alpha=0.78,
                linewidths=0,
                color=CLASS_COLORS[class_id],
                label=CLASS_LABELS[class_id],
            )
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_title(panel_title(label, title), loc="left", fontweight="bold")
        ax.set_xlabel("Projection 1")
        ax.set_ylabel("Projection 2")
        ax.grid(True, color="#E2E8F0", linewidth=0.45)
    handles, labels_out = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels_out, loc="best", frameon=False, ncol=3, handletextpad=0.3, columnspacing=0.8)


def _grouped_bar(
    ax: plt.Axes,
    pivot: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] = (0.0, 1.05),
) -> None:
    labels = list(pivot.index)
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    for offset, student in zip((-width / 2, width / 2), STUDENT_ORDER):
        values = pivot[student].to_numpy(dtype=float) if student in pivot.columns else np.full(len(labels), np.nan)
        ax.bar(
            x + offset,
            values,
            width=width,
            color=STUDENT_COLORS[student],
            label=student,
            edgecolor="#1F2937",
            linewidth=0.35,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.axhline(0.0, color="#475569", linewidth=0.55)
    ax.grid(True, axis="y", color="#E2E8F0", linewidth=0.45)


def _bounded_metric_ylim(pivot: pd.DataFrame, *, floor: float = -1.0, ceiling: float = 1.0) -> tuple[float, float]:
    values = pivot.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.05)
    vmin = float(values.min())
    vmax = float(values.max())
    span = max(vmax - vmin, 0.08)
    lower = min(0.0, vmin - 0.12 * span)
    upper = max(0.0, vmax + 0.12 * span)
    if upper - lower < 0.2:
        pad = (0.2 - (upper - lower)) / 2.0
        lower -= pad
        upper += pad
    return (max(floor, lower), min(ceiling, upper))


def draw_metric_row(axes: np.ndarray, tables: Mapping[str, pd.DataFrame]) -> None:
    centroid = tables["centroid"].copy()
    cka = tables["cka"].copy()
    boundary = tables["boundary_similarity"].copy()
    require_columns(
        centroid,
        {"student_model", "class_id", "class_label", "cosine_similarity"},
        table_name="class_centroid_cosine_similarity.csv",
    )
    require_columns(
        cka,
        {"student_model", "class_id", "class_label", "linear_cka"},
        table_name="linear_cka_by_class.csv",
    )
    require_columns(
        boundary,
        {"student_model", "boundary_distance_bin", "linear_cka", "mean_paired_cosine"},
        table_name="similarity_by_boundary_bin.csv",
    )
    centroid["class_label"] = pd.Categorical(
        centroid["class_id"].astype(int).map(CLASS_LABELS),
        categories=[CLASS_LABELS[idx] for idx in CLASS_ORDER],
        ordered=True,
    )
    cka["class_label"] = pd.Categorical(
        cka["class_id"].astype(int).map(CLASS_LABELS),
        categories=[CLASS_LABELS[idx] for idx in CLASS_ORDER],
        ordered=True,
    )
    boundary["boundary_distance_bin"] = pd.Categorical(
        boundary["boundary_distance_bin"].astype(str),
        categories=list(BOUNDARY_ORDER),
        ordered=True,
    )

    centroid_pivot = centroid.pivot_table(
        index="class_label",
        columns="student_model",
        values="cosine_similarity",
        observed=False,
    ).sort_index()
    cka_pivot = cka.pivot_table(
        index="class_label",
        columns="student_model",
        values="linear_cka",
        observed=False,
    ).sort_index()
    value_col = "mean_paired_cosine"
    if boundary[value_col].isna().all():
        value_col = "linear_cka"
    boundary_pivot = boundary.pivot_table(
        index="boundary_distance_bin",
        columns="student_model",
        values=value_col,
        observed=False,
    ).sort_index()

    _grouped_bar(
        axes[0],
        centroid_pivot,
        title=panel_title("(d)", "Class-centroid cosine"),
        ylabel="Cosine",
        ylim=_bounded_metric_ylim(centroid_pivot),
    )
    _grouped_bar(
        axes[1],
        cka_pivot,
        title=panel_title("(e)", "Class-wise linear CKA"),
        ylabel="Linear CKA",
    )
    _grouped_bar(
        axes[2],
        boundary_pivot,
        title=panel_title("(f)", "Boundary-distance similarity"),
        ylabel="Mean paired cosine" if value_col == "mean_paired_cosine" else "Linear CKA",
        ylim=_bounded_metric_ylim(boundary_pivot) if value_col == "mean_paired_cosine" else (0.0, 1.05),
    )
    axes[1].legend(loc="best", frameon=False)


def _attention_matrix(attention_df: pd.DataFrame, selected_windows: pd.DataFrame, model_name: str) -> tuple[np.ndarray, list[str], list[int]]:
    subset = attention_df.loc[attention_df["model_name"].astype(str).eq(model_name)].copy()
    sample_count = selected_windows["sample_index"].nunique()
    if sample_count <= 32:
        sample_order = selected_windows["sample_index"].astype(int).tolist()
        subset["sample_index"] = pd.Categorical(subset["sample_index"].astype(int), categories=sample_order, ordered=True)
        pivot = subset.pivot_table(
            index="sample_index",
            columns="relative_step",
            values="attention_weight",
            observed=False,
        ).sort_index()
        selected_lookup = selected_windows.set_index("sample_index")
        labels = [
            f"{selected_lookup.loc[int(sample), 'boundary_type']} #{int(sample)}"
            for sample in pivot.index.astype(int).tolist()
        ]
    else:
        pivot = subset.pivot_table(
            index="boundary_type",
            columns="relative_step",
            values="attention_weight",
            aggfunc="mean",
            observed=False,
        ).sort_index()
        labels = [str(value) for value in pivot.index.tolist()]
    columns = [int(col) for col in pivot.columns.astype(int).tolist()]
    return pivot.to_numpy(dtype=float), labels, columns


def draw_attention_row(axes: np.ndarray, attention_df: pd.DataFrame, selected_windows: pd.DataFrame) -> None:
    require_columns(
        attention_df,
        {
            "model_name",
            "sample_index",
            "boundary_type",
            "relative_step",
            "attention_weight",
            "y_true",
            "y_pred",
            "correct",
        },
        table_name="boundary_attention_heatmap.csv",
    )
    require_columns(
        selected_windows,
        {"sample_index", "boundary_type", "boundary_distance_bin"},
        table_name="selected_boundary_windows.csv",
    )
    matrices = []
    labels_by_model = []
    cols_by_model = []
    for model_name in MODEL_ORDER:
        matrix, labels, cols = _attention_matrix(attention_df, selected_windows, model_name)
        matrices.append(matrix)
        labels_by_model.append(labels)
        cols_by_model.append(cols)
    finite_values = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    vmin = float(np.nanmin(finite_values))
    vmax = float(np.nanmax(finite_values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    titles = ("Teacher attention", "Student-only attention", "SEAL-Weld Student attention")
    labels = ("(g)", "(h)", "(i)")
    images = []
    for ax, model_name, title, panel_label, matrix, ylabels, xcols in zip(
        axes,
        MODEL_ORDER,
        titles,
        labels,
        matrices,
        labels_by_model,
        cols_by_model,
    ):
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis", vmin=vmin, vmax=vmax)
        images.append(image)
        ax.set_title(panel_title(panel_label, title), loc="left", fontweight="bold")
        ax.set_xticks(np.arange(len(xcols)))
        ax.set_xticklabels([str(col) for col in xcols])
        ax.set_xlabel("Relative step")
        ax.set_yticks(np.arange(len(ylabels)))
        if len(ylabels) <= 16:
            ax.set_yticklabels(ylabels)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("Selected windows")
    plt.colorbar(images[-1], ax=list(axes), fraction=0.018, pad=0.012, label="Attention weight")


def render_figure(input_dir: str | Path) -> list[Path]:
    root = resolve_repo_path(input_dir)
    tables = load_source_tables(root)
    configure_style()
    fig, axes = plt.subplots(3, 3, figsize=(14.6, 9.4), constrained_layout=True)
    draw_projection_row(axes[0], tables["projection"])
    draw_metric_row(axes[1], tables)
    draw_attention_row(axes[2], tables["attention"], tables["selected_windows"])
    outputs = []
    for suffix in ("png", "pdf", "svg"):
        path = root / f"{FIGURE_BASENAME}.{suffix}"
        fig.savefig(path, dpi=450 if suffix == "png" else None)
        outputs.append(path)
    plt.close(fig)
    append_render_qa(root, outputs)
    return outputs


def append_render_qa(root: Path, outputs: list[Path]) -> None:
    qa_path = root / "qa_notes.txt"
    existing_lines: list[str] = []
    if qa_path.exists():
        existing_lines = [
            line
            for line in qa_path.read_text(encoding="utf-8").splitlines()
            if "\trenderer_export_" not in line
        ]
    with qa_path.open("w", encoding="utf-8") as handle:
        for line in existing_lines:
            handle.write(f"{line}\n")
        for path in outputs:
            status = "PASS" if path.exists() and path.stat().st_size > 0 else "FAIL"
            handle.write(f"{status}\trenderer_export_{path.suffix.lstrip('.')}\t{path}\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = render_figure(args.input_dir)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
