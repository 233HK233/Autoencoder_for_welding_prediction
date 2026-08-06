#!/usr/bin/env python3
"""Plot Figure 3 ROC curves from exported test predictions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

FIGURE3_METHOD_ORDER = [
    "teacher-student(student)",
    "teacher(18D upper bound)",
    "lstm",
    "gru",
    "transformer",
    "inception",
]

PLOT_METHOD_ORDER = [
    "teacher-student(student)",
    "teacher(18D upper bound)",
    "transformer",
    "gru",
    "lstm",
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

ROC_VIEWS = ["S0 vs rest", "S1 vs rest", "S2 vs rest", "macro-average", "micro-average"]
CLASS_VIEW_NAMES = {0: "S0 vs rest", 1: "S1 vs rest", 2: "S2 vs rest"}

DISPLAY_METHOD_NAMES = {
    "teacher-student(student)": ("Proposed", "13D"),
    "teacher(18D upper bound)": ("Teacher", "18D"),
    "transformer": ("Transformer", "18D"),
    "gru": ("GRU", "18D"),
    "lstm": ("LSTM", "18D"),
    "inception": ("InceptionTime", "18D"),
}

METHOD_STYLE_MAP = {
    "teacher-student(student)": {
        "color": "#0B4F9C",
        "linestyle": "-",
        "linewidth": 1.8,
        "alpha": 1.0,
        "zorder": 6,
    },
    "teacher(18D upper bound)": {
        "color": "#3A3A3A",
        "linestyle": (0, (5, 3)),
        "linewidth": 1.4,
        "alpha": 0.85,
        "zorder": 5,
    },
    "transformer": {
        "color": "#D98C32",
        "linestyle": (0, (4, 2, 1, 2)),
        "linewidth": 1.2,
        "alpha": 0.78,
        "zorder": 4,
    },
    "gru": {
        "color": "#7E57A8",
        "linestyle": (0, (4, 2)),
        "linewidth": 1.2,
        "alpha": 0.78,
        "zorder": 3,
    },
    "lstm": {
        "color": "#56A878",
        "linestyle": (0, (6, 3)),
        "linewidth": 1.2,
        "alpha": 0.78,
        "zorder": 2,
    },
    "inception": {
        "color": "#6BB7E8",
        "linestyle": (0, (1, 2)),
        "linewidth": 1.2,
        "alpha": 0.78,
        "zorder": 1,
    },
}

MACRO_PANEL_CONFIG = {
    "panel_label": "a",
    "title": None,
    "x_label": "False positive rate",
    "y_label": "True positive rate",
    "figure_size": (10.0, 6.6),
    "xlim": (0.0, 1.0),
    "ylim": (0.0, 1.02),
    "main_tick_fontsize": 8,
    "inset_tick_fontsize": 6,
}

MACRO_INSET_CONFIG = {
    "xlim": (0.0, 0.10),
    "ylim": (0.88, 1.01),
    "bounds": (0.48, 0.24, 0.34, 0.34),
    "draw_connectors": False,
    "title": "FPR <= 0.10",
    "method_names": [
        "teacher-student(student)",
        "teacher(18D upper bound)",
        "transformer",
        "gru",
        "lstm",
        "inception",
    ],
    "line_widths": {
        "teacher-student(student)": 1.5,
        "teacher(18D upper bound)": 1.2,
        "transformer": 1.0,
        "gru": 1.0,
        "lstm": 1.0,
        "inception": 1.0,
    },
}

MACRO_LEGEND_CONFIG = {
    "title": "Method (macro-AUC)",
    "loc": "upper left",
    "bbox_to_anchor": (0.0, 1.0),
    "frameon": False,
    "fontsize": 6.6,
    "title_fontsize": 6.6,
    "handlelength": 1.8,
    "handletextpad": 0.6,
    "labelspacing": 0.48,
    "borderaxespad": 0.0,
    "ncol": 1,
}

CHANCE_LINE_STYLE = {
    "color": "0.70",
    "linewidth": 0.7,
    "linestyle": "--",
    "alpha": 0.40,
    "zorder": 0,
}

RIGHT_INFO_PANEL_CONFIG = {
    "legend_bounds": (0.72, 0.42, 0.25, 0.23),
    "table_bounds": (0.72, 0.12, 0.25, 0.16),
}

PAUC_TABLE_CONFIG = {
    "title": "pAUC at FPR \u2264 0.10",
    "title_fontsize": 6.7,
    "font_size": 5.8,
    "scale_x": 0.96,
    "scale_y": 0.96,
    "header_linewidth": 0.20,
}

PER_CLASS_PANEL_LABELS = ["a", "b", "c"]


@dataclass
class PredictionTable:
    method_name: str
    method_slug: str
    sample_index: np.ndarray
    seam_id: np.ndarray
    seam_name: np.ndarray
    start_idx: np.ndarray
    target_idx: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 3 ROC curves")
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Figure 3 ROC comparison")
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def format_method_label(method_name: str, auc_value: float) -> str:
    display_name, input_dim = DISPLAY_METHOD_NAMES[method_name]
    return f"{display_name}, {input_dim} ({auc_value:.4f})"


def build_method_style_map() -> dict[str, dict[str, object]]:
    return METHOD_STYLE_MAP.copy()


def build_macro_inset_config() -> dict[str, object]:
    return dict(MACRO_INSET_CONFIG)


def build_macro_panel_config() -> dict[str, object]:
    return dict(MACRO_PANEL_CONFIG)


def build_macro_legend_config() -> dict[str, object]:
    return dict(MACRO_LEGEND_CONFIG)


def build_chance_line_style() -> dict[str, object]:
    return dict(CHANCE_LINE_STYLE)


def build_right_info_panel_config() -> dict[str, object]:
    return dict(RIGHT_INFO_PANEL_CONFIG)


def build_pauc_table_config() -> dict[str, object]:
    return dict(PAUC_TABLE_CONFIG)


def build_output_paths(output_dir: Path, tag: str = "") -> dict[str, Path]:
    figure_exports = output_dir / "figure_exports"
    tables_dir = output_dir / "tables"
    suffix = f"_{tag}" if tag else ""
    return {
        "roc_auc_summary": tables_dir / f"roc_auc_summary{suffix}.csv",
        "roc_curve_points": tables_dir / f"roc_curve_points{suffix}.csv",
        "pauc_summary": tables_dir / f"roc_pauc_summary_at_fpr_0p10{suffix}.csv",
        "macro_svg": figure_exports / f"figure3_roc_macro_average{suffix}.svg",
        "macro_pdf": figure_exports / f"figure3_roc_macro_average{suffix}.pdf",
        "macro_png": figure_exports / f"figure3_roc_macro_average{suffix}.png",
        "per_class_svg": figure_exports / f"figure3_roc_per_class{suffix}.svg",
        "per_class_pdf": figure_exports / f"figure3_roc_per_class{suffix}.pdf",
        "per_class_png": figure_exports / f"figure3_roc_per_class{suffix}.png",
        "caption_notes": figure_exports / f"figure3_roc_caption_notes{suffix}.txt",
    }


def ensure_output_dirs(output_dir: Path, tag: str = "") -> dict[str, Path]:
    paths = build_output_paths(output_dir, tag=tag)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figure_exports").mkdir(parents=True, exist_ok=True)
    return paths


def load_prediction_table(csv_path: Path, method_name: str) -> PredictionTable:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"prediction CSV is empty: {csv_path}")
    probabilities = np.array(
        [[float(row[f"prob_class_{idx}"]) for idx in range(3)] for row in rows],
        dtype=np.float64,
    )
    return PredictionTable(
        method_name=method_name,
        method_slug=METHOD_SLUGS[method_name],
        sample_index=np.array([int(row["sample_index"]) for row in rows], dtype=np.int64),
        seam_id=np.array([int(row["seam_id"]) for row in rows], dtype=np.int64),
        seam_name=np.array([row["seam_name"] for row in rows], dtype=object),
        start_idx=np.array([int(row["start_idx"]) for row in rows], dtype=np.int64),
        target_idx=np.array([int(row["target_idx"]) for row in rows], dtype=np.int64),
        y_true=np.array([int(row["y_true"]) for row in rows], dtype=np.int64),
        y_pred=np.array([int(row["y_pred"]) for row in rows], dtype=np.int64),
        probabilities=probabilities,
    )


def load_prediction_tables(predictions_dir: Path) -> list[PredictionTable]:
    tables = []
    for method_name in FIGURE3_METHOD_ORDER:
        csv_path = predictions_dir / f"{METHOD_SLUGS[method_name]}_test_predictions.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"missing prediction CSV for {method_name}: {csv_path}")
        tables.append(load_prediction_table(csv_path, method_name=method_name))
    return tables


def validate_identity_columns(tables: Iterable[PredictionTable]) -> None:
    tables = list(tables)
    reference = tables[0]
    for table in tables[1:]:
        for field in ("sample_index", "seam_id", "start_idx", "target_idx", "y_true"):
            ref_values = getattr(reference, field)
            cur_values = getattr(table, field)
            if not np.array_equal(ref_values, cur_values):
                raise ValueError(f"prediction identity column mismatch for {field} between methods")


def require_sklearn_metrics():
    try:
        from sklearn.metrics import auc, roc_curve
    except Exception as exc:  # pragma: no cover - exercised in runtime environment
        raise RuntimeError("scikit-learn is required for ROC plotting; install sklearn before running this script") from exc
    return roc_curve, auc


def compute_method_roc(method_name: str, probabilities: np.ndarray, y_true: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    roc_curve_fn, auc_fn = require_sklearn_metrics()
    auc_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    y_onehot = np.eye(3, dtype=np.int64)[y_true]
    class_curves: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}

    for class_idx in range(3):
        fpr, tpr, _ = roc_curve_fn(y_onehot[:, class_idx], probabilities[:, class_idx])
        roc_auc = float(auc_fn(fpr, tpr))
        class_curves[class_idx] = (fpr, tpr, roc_auc)
        auc_rows.append(
            {
                "method_name": method_name,
                "method_slug": METHOD_SLUGS[method_name],
                "roc_view": CLASS_VIEW_NAMES[class_idx],
                "auc": f"{roc_auc:.8f}",
            }
        )
        for x_value, y_value in zip(fpr, tpr, strict=True):
            curve_rows.append(
                {
                    "method_name": method_name,
                    "method_slug": METHOD_SLUGS[method_name],
                    "roc_view": CLASS_VIEW_NAMES[class_idx],
                    "fpr": f"{float(x_value):.8f}",
                    "tpr": f"{float(y_value):.8f}",
                }
            )

    micro_fpr, micro_tpr, _ = roc_curve_fn(y_onehot.ravel(), probabilities.ravel())
    micro_auc = float(auc_fn(micro_fpr, micro_tpr))
    auc_rows.append(
        {
            "method_name": method_name,
            "method_slug": METHOD_SLUGS[method_name],
            "roc_view": "micro-average",
            "auc": f"{micro_auc:.8f}",
        }
    )
    for x_value, y_value in zip(micro_fpr, micro_tpr, strict=True):
        curve_rows.append(
            {
                "method_name": method_name,
                "method_slug": METHOD_SLUGS[method_name],
                "roc_view": "micro-average",
                "fpr": f"{float(x_value):.8f}",
                "tpr": f"{float(y_value):.8f}",
            }
        )

    all_fpr = np.unique(np.concatenate([class_curves[idx][0] for idx in range(3)]))
    mean_tpr = np.zeros_like(all_fpr)
    for class_idx in range(3):
        mean_tpr += np.interp(all_fpr, class_curves[class_idx][0], class_curves[class_idx][1])
    mean_tpr /= 3.0
    macro_auc = float(auc_fn(all_fpr, mean_tpr))
    auc_rows.append(
        {
            "method_name": method_name,
            "method_slug": METHOD_SLUGS[method_name],
            "roc_view": "macro-average",
            "auc": f"{macro_auc:.8f}",
        }
    )
    for x_value, y_value in zip(all_fpr, mean_tpr, strict=True):
        curve_rows.append(
            {
                "method_name": method_name,
                "method_slug": METHOD_SLUGS[method_name],
                "roc_view": "macro-average",
                "fpr": f"{float(x_value):.8f}",
                "tpr": f"{float(y_value):.8f}",
            }
        )
    return auc_rows, curve_rows


def build_roc_tables(tables: Iterable[PredictionTable]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    auc_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for table in tables:
        method_auc_rows, method_curve_rows = compute_method_roc(
            method_name=table.method_name,
            probabilities=table.probabilities,
            y_true=table.y_true,
        )
        auc_rows.extend(method_auc_rows)
        curve_rows.extend(method_curve_rows)
    return auc_rows, curve_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def get_curve_points(curve_rows: Iterable[dict[str, object]], method_name: str, roc_view: str) -> tuple[list[float], list[float]]:
    points = [row for row in curve_rows if row["method_name"] == method_name and row["roc_view"] == roc_view]
    return [float(row["fpr"]) for row in points], [float(row["tpr"]) for row in points]


def compute_normalized_partial_auc(fpr: np.ndarray, tpr: np.ndarray, max_fpr: float) -> float:
    if max_fpr <= 0.0 or max_fpr > 1.0:
        raise ValueError(f"max_fpr must be in (0, 1], got {max_fpr}")
    if fpr.ndim != 1 or tpr.ndim != 1 or fpr.shape != tpr.shape:
        raise ValueError("fpr and tpr must be 1D arrays with identical shape")
    clipped_fpr = np.asarray(fpr, dtype=np.float64)
    clipped_tpr = np.asarray(tpr, dtype=np.float64)
    if max_fpr not in clipped_fpr:
        interp_tpr = float(np.interp(max_fpr, clipped_fpr, clipped_tpr))
        insert_at = int(np.searchsorted(clipped_fpr, max_fpr))
        clipped_fpr = np.insert(clipped_fpr, insert_at, max_fpr)
        clipped_tpr = np.insert(clipped_tpr, insert_at, interp_tpr)
    mask = clipped_fpr <= max_fpr
    clipped_fpr = clipped_fpr[mask]
    clipped_tpr = clipped_tpr[mask]
    area = float(np.trapezoid(clipped_tpr, clipped_fpr))
    return area / max_fpr


def build_pauc_summary_rows(tables: Iterable[PredictionTable], max_fpr: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for table in tables:
        method_auc_rows, method_curve_rows = compute_method_roc(
            method_name=table.method_name,
            probabilities=table.probabilities,
            y_true=table.y_true,
        )
        macro_auc = next(float(row["auc"]) for row in method_auc_rows if row["roc_view"] == "macro-average")
        fpr_values, tpr_values = get_curve_points(method_curve_rows, method_name=table.method_name, roc_view="macro-average")
        pauc = compute_normalized_partial_auc(
            fpr=np.asarray(fpr_values, dtype=np.float64),
            tpr=np.asarray(tpr_values, dtype=np.float64),
            max_fpr=max_fpr,
        )
        rows.append(
            {
                "method_name": table.method_name,
                "method_slug": table.method_slug,
                "macro_auc": f"{macro_auc:.8f}",
                "pauc_at_fpr_0p10": f"{pauc:.8f}",
            }
        )
    return rows


def build_caption_notes() -> str:
    return (
        "The main panel shows full macro-average one-vs-rest ROC curves on the fixed H1 test set.\n\n"
        "The inset magnifies the low-false-positive region (FPR <= 0.10), which is most relevant for conservative deployment.\n\n"
        "Values in parentheses denote macro-AUC.\n\n"
        "Teacher denotes the 18D upper bound reference model.\n\n"
        "The same Figure 3 selected runs were used; models were not reselected by ROC-AUC.\n"
    )


def add_pauc_table(fig, pauc_rows: list[dict[str, object]]) -> None:
    info_cfg = build_right_info_panel_config()
    table_cfg = build_pauc_table_config()

    table_ax = fig.add_axes(list(info_cfg["table_bounds"]))
    table_ax.axis("off")
    ordered_rows = sorted(pauc_rows, key=lambda row: PLOT_METHOD_ORDER.index(row["method_name"]))
    cell_text = [
        [DISPLAY_METHOD_NAMES[row["method_name"]][0], f"{float(row['pauc_at_fpr_0p10']):.4f}"]
        for row in ordered_rows
    ]
    table = table_ax.table(
        cellText=cell_text,
        colLabels=["Method", "pAUC@0.10"],
        colLoc="left",
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(table_cfg["font_size"])
    table.scale(table_cfg["scale_x"], table_cfg["scale_y"])
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_linewidth(0.0 if row_idx else table_cfg["header_linewidth"])
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#F4F4F4")

    table_ax.text(
        0.0,
        1.06,
        table_cfg["title"],
        transform=table_ax.transAxes,
        fontsize=table_cfg["title_fontsize"],
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def add_macro_legend(fig, ax) -> None:
    info_cfg = build_right_info_panel_config()
    legend_cfg = build_macro_legend_config()
    legend_ax = fig.add_axes(list(info_cfg["legend_bounds"]))
    legend_ax.axis("off")
    handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(handles, labels, **legend_cfg)


def apply_publication_rcparams() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.9,
            "legend.frameon": False,
        }
    )


def add_panel_label(ax, panel_label: str) -> None:
    ax.text(
        -0.10,
        1.02,
        panel_label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def plot_macro_average(
    auc_rows: list[dict[str, object]],
    curve_rows: list[dict[str, object]],
    pauc_rows: list[dict[str, object]],
    title: str,
    output_paths: dict[str, Path],
) -> None:
    apply_publication_rcparams()
    import matplotlib.pyplot as plt

    macro_auc = {
        row["method_name"]: float(row["auc"]) for row in auc_rows if row["roc_view"] == "macro-average"
    }
    style_map = build_method_style_map()
    panel_cfg = build_macro_panel_config()
    inset_cfg = build_macro_inset_config()
    chance_style = build_chance_line_style()

    fig, ax = plt.subplots(figsize=panel_cfg["figure_size"])
    for method_name in PLOT_METHOD_ORDER:
        x_values, y_values = get_curve_points(curve_rows, method_name=method_name, roc_view="macro-average")
        style = style_map[method_name]
        ax.plot(
            x_values,
            y_values,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            zorder=style["zorder"],
            label=format_method_label(method_name, macro_auc[method_name]),
        )
    ax.plot([0.0, 1.0], [0.0, 1.0], **chance_style)
    ax.set_xlim(*panel_cfg["xlim"])
    ax.set_ylim(*panel_cfg["ylim"])
    ax.set_xlabel(str(panel_cfg["x_label"]))
    ax.set_ylabel(str(panel_cfg["y_label"]))
    ax.set_title("")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18, linewidth=0.6)
    ax.tick_params(labelsize=panel_cfg["main_tick_fontsize"])
    add_panel_label(ax, str(panel_cfg["panel_label"]))

    axins = ax.inset_axes(list(inset_cfg["bounds"]))
    for method_name in inset_cfg["method_names"]:
        x_values, y_values = get_curve_points(curve_rows, method_name=method_name, roc_view="macro-average")
        style = style_map[method_name]
        axins.plot(
            x_values,
            y_values,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=inset_cfg["line_widths"][method_name],
            alpha=style["alpha"],
            zorder=style["zorder"],
        )
    axins.set_xlim(*inset_cfg["xlim"])
    axins.set_ylim(*inset_cfg["ylim"])
    axins.grid(True, alpha=0.18, linewidth=0.5)
    axins.tick_params(labelsize=panel_cfg["inset_tick_fontsize"])
    axins.set_title(str(inset_cfg["title"]), fontsize=6.2, pad=2.0)

    add_macro_legend(fig, ax)
    add_pauc_table(fig, pauc_rows)
    fig.subplots_adjust(left=0.10, right=0.68, bottom=0.12, top=0.96)
    fig.savefig(output_paths["macro_svg"], bbox_inches="tight")
    fig.savefig(output_paths["macro_pdf"], bbox_inches="tight")
    fig.savefig(output_paths["macro_png"], dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_class(
    auc_rows: list[dict[str, object]],
    curve_rows: list[dict[str, object]],
    title: str,
    output_paths: dict[str, Path],
) -> None:
    apply_publication_rcparams()
    import matplotlib.pyplot as plt

    auc_lookup = {(row["method_name"], row["roc_view"]): float(row["auc"]) for row in auc_rows}
    style_map = build_method_style_map()
    chance_style = build_chance_line_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.0), sharex=True, sharey=True)
    for class_idx, ax in enumerate(axes):
        view_name = CLASS_VIEW_NAMES[class_idx]
        for method_name in PLOT_METHOD_ORDER:
            x_values, y_values = get_curve_points(curve_rows, method_name=method_name, roc_view=view_name)
            style = style_map[method_name]
            ax.plot(
                x_values,
                y_values,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                alpha=style["alpha"],
                zorder=style["zorder"],
                label=format_method_label(method_name, auc_lookup[(method_name, view_name)]),
            )
        ax.plot([0.0, 1.0], [0.0, 1.0], **chance_style)
        ax.set_title(view_name)
        ax.set_xlabel("False positive rate")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18, linewidth=0.6)
        add_panel_label(ax, PER_CLASS_PANEL_LABELS[class_idx])
    axes[0].set_ylabel("True positive rate")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        fontsize=6.4,
        handlelength=2.8,
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    fig.savefig(output_paths["per_class_svg"], bbox_inches="tight")
    fig.savefig(output_paths["per_class_pdf"], bbox_inches="tight")
    fig.savefig(output_paths["per_class_png"], dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_paths = ensure_output_dirs(args.output_dir, tag=args.tag)
    tables = load_prediction_tables(args.predictions_dir)
    validate_identity_columns(tables)
    auc_rows, curve_rows = build_roc_tables(tables)
    pauc_rows = build_pauc_summary_rows(tables, max_fpr=0.10)
    write_csv(output_paths["roc_auc_summary"], auc_rows)
    write_csv(output_paths["roc_curve_points"], curve_rows)
    write_csv(output_paths["pauc_summary"], pauc_rows)
    write_text(output_paths["caption_notes"], build_caption_notes())
    plot_macro_average(
        auc_rows=auc_rows,
        curve_rows=curve_rows,
        pauc_rows=pauc_rows,
        title=args.title,
        output_paths=output_paths,
    )
    plot_per_class(auc_rows=auc_rows, curve_rows=curve_rows, title=args.title, output_paths=output_paths)


if __name__ == "__main__":
    main()
