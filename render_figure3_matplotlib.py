#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

METHOD_ORDER = ("Teacher", "Student", "GRU", "LSTM", "Inception", "Transformer")
BASELINE_ORDER = ("GRU", "LSTM", "Inception", "Transformer")
CLASS_ORDER = ("S0", "S1", "S2")
MODEL_NAME_TO_DISPLAY = {
    "Exploratory Teacher Best": "Teacher",
    "Best Student Distill": "Student",
}
BASELINE_NAME_TO_DISPLAY = {
    "gru": "GRU",
    "lstm": "LSTM",
    "inception": "Inception",
    "transformer": "Transformer",
}
METHOD_COLORS = {
    "Teacher": "#1F3A5F",
    "Student": "#4C8D9B",
    "GRU": "#D98C3F",
    "LSTM": "#B55D5C",
    "Inception": "#7A8F3C",
    "Transformer": "#6B7280",
}
METHOD_MARKERS = {
    "Teacher": "o",
    "Student": "s",
    "GRU": "D",
    "LSTM": "^",
    "Inception": "v",
    "Transformer": "P",
}
PANEL_TITLES = {
    "A": "Overall metrics",
    "B": "Per-class F1 profile",
    "C": "Per-class F1 heatmap",
    "D": "S1 precision-recall signature",
}
TEST_METRICS_PATTERN = re.compile(
    r"--- Test Metrics.*?Accuracy:\s*([0-9.]+)%.*?Macro-F1:\s*([0-9.]+)",
    flags=re.DOTALL,
)
CLASS_REPORT_PATTERN = re.compile(
    r"=== Classification Report \(Test\) ===(?P<block>.*?)(?:\n\s*accuracy\s+|\Z)",
    flags=re.DOTALL,
)
CLASS_ROW_PATTERN = re.compile(
    r"^\s*Class\s+(?P<class_id>\d+)\s+"
    r"(?P<precision>[0-9.]+)\s+"
    r"(?P<recall>[0-9.]+)\s+"
    r"(?P<f1>[0-9.]+)\s+"
    r"(?P<support>\d+)\s*$",
    flags=re.MULTILINE,
)


@dataclass(frozen=True)
class Figure3Data:
    overall_df: pd.DataFrame
    class_metrics_df: pd.DataFrame
    panel_values_df: pd.DataFrame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the paper-facing Figure 3 comparison composite with matplotlib."
    )
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=Path("outputs/paper_figures/h1_results/data/exploratory_ordered_run_registry.csv"),
    )
    parser.add_argument(
        "--class-metrics-csv",
        type=Path,
        default=Path("outputs/paper_figures/h1_results/data/exploratory_ordered_class_metrics_summary.csv"),
    )
    parser.add_argument(
        "--baseline-summary-csv",
        type=Path,
        default=Path(
            "ablation_experiments/h1/reports/comparison_18d_baselines/checkpoints/round3/comparison_summary.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/paper_figures"),
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="figure3_h1_comparison",
    )
    parser.add_argument(
        "--output-dir-name",
        type=str,
        default="fig3_h1_comparison_layout",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional figure-level title. Leave empty for the paper-facing export.",
    )
    return parser.parse_args(argv)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def prepare_output_dir(output_root: Path, output_name: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = output_root / f"{output_name}_v{index:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        index += 1


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#111827",
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")
    ax.tick_params(colors="#334155")


def extract_test_metrics(metrics_text: str) -> tuple[float, float]:
    match = TEST_METRICS_PATTERN.search(metrics_text)
    if match is None:
        raise ValueError("Could not parse test Accuracy/Macro-F1 from evaluation_metrics.txt")
    return float(match.group(1)), float(match.group(2)) * 100.0


def extract_class_metrics(metrics_text: str) -> list[dict[str, float | int | str]]:
    report_match = CLASS_REPORT_PATTERN.search(metrics_text)
    if report_match is None:
        raise ValueError("Could not parse Test classification report block from evaluation_metrics.txt")

    rows: list[dict[str, float | int | str]] = []
    for row_match in CLASS_ROW_PATTERN.finditer(report_match.group("block")):
        class_id = int(row_match.group("class_id"))
        rows.append(
            {
                "class_id": class_id,
                "class_name": f"S{class_id}",
                "precision": float(row_match.group("precision")),
                "recall": float(row_match.group("recall")),
                "f1": float(row_match.group("f1")),
            }
        )
    if len(rows) != 3:
        raise ValueError("Expected exactly three test classes in classification report")
    return rows


def prepare_registry_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"model_name", "run_dir"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    df["display_name"] = df["model_name"].map(MODEL_NAME_TO_DISPLAY)
    df = df.loc[df["display_name"].isin(METHOD_ORDER[:2])].copy()

    missing_roles = [name for name in METHOD_ORDER[:2] if name not in df["display_name"].tolist()]
    if missing_roles:
        raise ValueError(f"Missing teacher/student roles in {path}: {missing_roles}")

    df["display_name"] = pd.Categorical(df["display_name"], categories=METHOD_ORDER[:2], ordered=True)
    return df.sort_values("display_name").reset_index(drop=True)


def prepare_baseline_summary_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"method_name", "best_test_acc", "best_macro_f1", "run_path"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    df["display_name"] = df["method_name"].map(BASELINE_NAME_TO_DISPLAY)
    df = df.loc[df["display_name"].isin(BASELINE_ORDER)].copy()

    missing_methods = [name for name in BASELINE_ORDER if name not in df["display_name"].tolist()]
    if missing_methods:
        raise ValueError(f"Missing baseline roles in {path}: {missing_methods}")

    df["display_name"] = pd.Categorical(df["display_name"], categories=BASELINE_ORDER, ordered=True)
    return df.sort_values("display_name").reset_index(drop=True)


def build_overall_metrics_dataframe(registry_csv: Path, baseline_summary_csv: Path) -> pd.DataFrame:
    registry_df = prepare_registry_dataframe(registry_csv)
    baseline_df = prepare_baseline_summary_dataframe(baseline_summary_csv)

    rows: list[dict[str, object]] = []
    for row in registry_df.itertuples(index=False):
        metrics_path = Path(str(row.run_dir)) / "evaluation_metrics.txt"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing evaluation metrics file: {metrics_path}")
        metrics_text = metrics_path.read_text(encoding="utf-8")
        accuracy_pct, macro_f1_pct = extract_test_metrics(metrics_text)
        rows.append(
            {
                "method": str(row.display_name),
                "test_acc_pct": accuracy_pct,
                "macro_f1_pct": macro_f1_pct,
                "run_path": str(row.run_dir),
            }
        )

    for row in baseline_df.itertuples(index=False):
        rows.append(
            {
                "method": str(row.display_name),
                "test_acc_pct": float(row.best_test_acc) * 100.0,
                "macro_f1_pct": float(row.best_macro_f1) * 100.0,
                "run_path": str(row.run_path),
            }
        )

    overall_df = pd.DataFrame(rows)
    overall_df["method"] = pd.Categorical(overall_df["method"], categories=METHOD_ORDER, ordered=True)
    overall_df = overall_df.sort_values("method").reset_index(drop=True)

    gru_row = overall_df.loc[overall_df["method"] == "GRU"].iloc[0]
    overall_df["delta_acc_vs_gru_pp"] = overall_df["test_acc_pct"].astype(float) - float(gru_row["test_acc_pct"])
    overall_df["delta_macro_f1_vs_gru_pp"] = (
        overall_df["macro_f1_pct"].astype(float) - float(gru_row["macro_f1_pct"])
    )
    return overall_df


def build_class_metrics_dataframe(
    class_metrics_csv: Path,
    baseline_summary_csv: Path,
) -> pd.DataFrame:
    teacher_student_df = pd.read_csv(class_metrics_csv)
    required = {"model_name", "class_name", "metric", "value"}
    missing = required.difference(teacher_student_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {class_metrics_csv}: {sorted(missing)}")

    teacher_student_df["method"] = teacher_student_df["model_name"].map(MODEL_NAME_TO_DISPLAY)
    teacher_student_df = teacher_student_df.loc[teacher_student_df["method"].isin(METHOD_ORDER[:2])].copy()

    teacher_student_wide = (
        teacher_student_df.pivot_table(
            index=["method", "class_name"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    baseline_summary_df = prepare_baseline_summary_dataframe(baseline_summary_csv)
    baseline_rows: list[dict[str, object]] = []
    for row in baseline_summary_df.itertuples(index=False):
        run_dir = Path(str(row.run_path))
        metrics_path = run_dir / "evaluation_metrics.txt"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing evaluation metrics file: {metrics_path}")
        metrics_text = metrics_path.read_text(encoding="utf-8")
        for class_row in extract_class_metrics(metrics_text):
            baseline_rows.append(
                {
                    "method": str(row.display_name),
                    "class_name": class_row["class_name"],
                    "precision": class_row["precision"],
                    "recall": class_row["recall"],
                    "f1": class_row["f1"],
                }
            )

    baseline_df = pd.DataFrame(baseline_rows)
    class_metrics_df = pd.concat([teacher_student_wide, baseline_df], ignore_index=True)
    class_metrics_df["method"] = pd.Categorical(
        class_metrics_df["method"], categories=METHOD_ORDER, ordered=True
    )
    class_metrics_df["class_name"] = pd.Categorical(
        class_metrics_df["class_name"], categories=CLASS_ORDER, ordered=True
    )
    class_metrics_df = class_metrics_df.sort_values(["method", "class_name"]).reset_index(drop=True)
    return class_metrics_df


def build_panel_values_dataframe(overall_df: pd.DataFrame, class_metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        row = overall_df.loc[overall_df["method"] == method].iloc[0]
        rows.append(
            {
                "panel": "A",
                "method": method,
                "class_name": "",
                "metric": "accuracy",
                "value_pct": float(row["test_acc_pct"]),
            }
        )
        rows.append(
            {
                "panel": "A",
                "method": method,
                "class_name": "",
                "metric": "macro_f1",
                "value_pct": float(row["macro_f1_pct"]),
            }
        )

    for method in METHOD_ORDER:
        for class_name in CLASS_ORDER:
            row = class_metrics_df.loc[
                (class_metrics_df["method"] == method) & (class_metrics_df["class_name"] == class_name)
            ].iloc[0]
            rows.append(
                {
                    "panel": "B",
                    "method": method,
                    "class_name": class_name,
                    "metric": "f1",
                    "value_pct": float(row["f1"]) * 100.0,
                }
            )
        for class_name in CLASS_ORDER:
            row = class_metrics_df.loc[
                (class_metrics_df["method"] == method) & (class_metrics_df["class_name"] == class_name)
            ].iloc[0]
            rows.append(
                {
                    "panel": "C",
                    "method": method,
                    "class_name": class_name,
                    "metric": "f1",
                    "value_pct": float(row["f1"]) * 100.0,
                }
            )

    for metric_name in ("precision", "recall"):
        for method in METHOD_ORDER:
            row = class_metrics_df.loc[
                (class_metrics_df["method"] == method) & (class_metrics_df["class_name"] == "S1")
            ].iloc[0]
            rows.append(
                {
                    "panel": "D",
                    "method": method,
                    "class_name": "S1",
                    "metric": metric_name,
                    "value_pct": float(row[metric_name]) * 100.0,
                }
            )

    panel_values_df = pd.DataFrame(rows)
    return panel_values_df


def build_figure3_data(
    *,
    registry_csv: Path,
    class_metrics_csv: Path,
    baseline_summary_csv: Path,
) -> Figure3Data:
    overall_df = build_overall_metrics_dataframe(registry_csv, baseline_summary_csv)
    class_metrics_df = build_class_metrics_dataframe(class_metrics_csv, baseline_summary_csv)
    panel_values_df = build_panel_values_dataframe(overall_df, class_metrics_df)
    return Figure3Data(
        overall_df=overall_df,
        class_metrics_df=class_metrics_df,
        panel_values_df=panel_values_df,
    )


def _plot_overall_metrics_panel(ax: plt.Axes, overall_df: pd.DataFrame) -> None:
    group_centers = np.array([0.0, 1.55])
    bar_width = 0.12
    offsets = (np.arange(len(METHOD_ORDER)) - (len(METHOD_ORDER) - 1) / 2.0) * bar_width

    for idx, method in enumerate(METHOD_ORDER):
        row = overall_df.loc[overall_df["method"] == method].iloc[0]
        values = [float(row["test_acc_pct"]), float(row["macro_f1_pct"])]
        ax.bar(
            group_centers + offsets[idx],
            values,
            width=bar_width * 0.92,
            color=METHOD_COLORS[method],
            edgecolor="#334155",
            linewidth=0.6 if method in {"Teacher", "Student"} else 0.45,
            zorder=3,
        )
        for x_pos, value in zip(group_centers + offsets[idx], values):
            ax.text(
                x_pos,
                value + 0.35,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.2,
                color="#111827",
            )

    ax.set_xlim(-0.55, 2.10)
    ax.set_ylim(70.0, 100.0)
    ax.set_yticks([70, 80, 90, 100])
    ax.set_xticks(group_centers)
    ax.set_xticklabels(["Test accuracy", "Macro-F1"])
    ax.set_ylabel("Score (%)")
    ax.grid(axis="y", color="#cbd5e1", alpha=0.55, linewidth=0.6)
    ax.set_axisbelow(True)
    style_axis(ax)


def _plot_class_profile(ax: plt.Axes, class_metrics_df: pd.DataFrame) -> None:
    x = np.arange(len(CLASS_ORDER))
    ax.axvspan(0.60, 1.40, color="#E2E8F0", alpha=0.55, zorder=0)
    for method in METHOD_ORDER:
        subset = class_metrics_df.loc[class_metrics_df["method"] == method].set_index("class_name")
        values = [float(subset.loc[class_name, "f1"]) * 100.0 for class_name in CLASS_ORDER]
        ax.plot(
            x,
            values,
            marker=METHOD_MARKERS[method],
            markersize=4.8,
            linewidth=1.7 if method in {"Teacher", "Student"} else 1.3,
            color=METHOD_COLORS[method],
            alpha=0.95,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_ylabel("F1 (%)")
    ax.set_ylim(70.0, 100.0)
    ax.text(1.0, 99.1, "Transition class", ha="center", va="top", fontsize=6.5, color="#334155")
    ax.grid(axis="y", color="#cbd5e1", alpha=0.55, linewidth=0.6)
    ax.set_axisbelow(True)
    style_axis(ax)


def _plot_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    row_labels: tuple[str, ...],
    col_labels: tuple[str, ...],
    cmap: str,
    vmin: float,
    vmax: float,
    value_fmt: str,
) -> None:
    image = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            text_color = "white" if Normalize(vmin=vmin, vmax=vmax)(value) > 0.55 else "#111827"
            ax.text(
                col_idx,
                row_idx,
                format(value, value_fmt),
                ha="center",
                va="center",
                fontsize=6.4,
                color=text_color,
            )
    style_axis(ax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return image


def _plot_s1_pr_scatter(ax: plt.Axes, class_metrics_df: pd.DataFrame) -> None:
    subset = class_metrics_df.loc[class_metrics_df["class_name"] == "S1"]
    for row in subset.itertuples(index=False):
        method = str(row.method)
        recall = float(row.recall) * 100.0
        precision = float(row.precision) * 100.0
        ax.scatter(
            recall,
            precision,
            s=72 if method in {"Teacher", "Student"} else 58,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            edgecolor="white",
            linewidth=1.2,
            alpha=0.90,
            zorder=3,
        )
    ax.set_xlim(60.0, 101.0)
    ax.set_ylim(70.0, 96.0)
    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.grid(True, which="major", axis="both", color="#cbd5e1", alpha=0.55, linewidth=0.6)
    ax.set_axisbelow(True)
    style_axis(ax)


def render(
    *,
    data: Figure3Data,
    output_pdf: Path,
    output_png: Path,
    output_svg: Path,
    title: str,
) -> None:
    configure_style()
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.8))
    plt.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.11, wspace=0.34, hspace=0.42)

    if title:
        fig.text(0.065, 0.975, title, ha="left", va="top", fontsize=10.5, fontweight="bold", color="#111827")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[name],
            marker=METHOD_MARKERS[name],
            linewidth=1.6,
            markersize=5.0,
            label=name,
        )
        for name in METHOD_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.965 if title else 0.975),
        ncol=6,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.3,
    )

    overall_df = data.overall_df.copy()
    class_df = data.class_metrics_df.copy()

    gs = axes[0, 0].get_gridspec()
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.remove()

    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = axes[0, 2]
    ax_c = fig.add_subplot(gs[1, 0:2])
    ax_d = axes[1, 2]

    _plot_overall_metrics_panel(ax_a, overall_df)
    _plot_class_profile(ax_b, class_df)

    heatmap_matrix = (
        class_df.pivot_table(
            index="method",
            columns="class_name",
            values="f1",
            aggfunc="first",
            observed=False,
        )
        .loc[list(METHOD_ORDER), list(CLASS_ORDER)]
        .astype(float)
        .to_numpy()
        * 100.0
    )
    heatmap = _plot_heatmap(
        ax_c,
        heatmap_matrix,
        METHOD_ORDER,
        CLASS_ORDER,
        cmap="YlGnBu",
        vmin=70.0,
        vmax=100.0,
        value_fmt=".1f",
    )

    _plot_s1_pr_scatter(ax_d, class_df)

    for label, ax in zip(PANEL_TITLES.keys(), (ax_a, ax_b, ax_c, ax_d)):
        add_panel_label(ax, label)
        ax.set_title(PANEL_TITLES[label], pad=7, color="#111827")

    cbar = fig.colorbar(heatmap, ax=ax_c, fraction=0.046, pad=0.04)
    cbar.set_label("F1 (%)", color="#334155")
    cbar.ax.tick_params(labelsize=7, colors="#334155")

    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mpl_config_dir = Path(".mplconfig")
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir.resolve()))

    data = build_figure3_data(
        registry_csv=args.registry_csv,
        class_metrics_csv=args.class_metrics_csv,
        baseline_summary_csv=args.baseline_summary_csv,
    )

    output_dir = prepare_output_dir(args.output_root, args.output_dir_name)
    output_pdf = output_dir / f"{args.output_name}.pdf"
    output_png = output_dir / f"{args.output_name}.png"
    output_svg = output_dir / f"{args.output_name}.svg"
    panel_values_csv = output_dir / f"{args.output_name}_panel_values.csv"

    data.panel_values_df.to_csv(panel_values_csv, index=False)
    render(
        data=data,
        output_pdf=output_pdf,
        output_png=output_png,
        output_svg=output_svg,
        title=args.title,
    )

    print(f"Saved Figure 3 outputs to: {output_dir}")
    print(f"PDF: {output_pdf}")
    print(f"PNG: {output_png}")
    print(f"SVG: {output_svg}")
    print(f"Panel values: {panel_values_csv}")


if __name__ == "__main__":
    main()
