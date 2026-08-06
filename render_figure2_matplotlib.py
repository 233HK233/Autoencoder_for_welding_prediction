#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

DISPLAY_ORDER = ("Teacher", "Student", "Ablation-1", "Ablation-2")
MODEL_NAME_TO_DISPLAY = {
    "Exploratory Teacher Best": "Teacher",
    "Strict Teacher Best": "Teacher",
    "Teacher": "Teacher",
    "Best Student Distill": "Student",
    "Student": "Student",
    "Best Ablation-1": "Ablation-1",
    "Ablation-1": "Ablation-1",
    "Best Ablation-2": "Ablation-2",
    "Ablation-2": "Ablation-2",
}
METHOD_COLORS = {
    "Teacher": "#1F3A5F",
    "Student": "#4C8D9B",
    "Ablation-1": "#D98C3F",
    "Ablation-2": "#B55D5C",
}
METHOD_DESCRIPTIONS = {
    "Teacher": "TCN-Attention with\nfull-feature input",
    "Student": "Distilled TCN-Attention\nstudent",
    "Ablation-1": "TCN-Attention student\nwithout distillation",
    "Ablation-2": "LSTM-based\ndistillation variant",
}
TEST_METRICS_PATTERN = re.compile(
    r"--- Test Metrics.*?Accuracy:\s*([0-9.]+)%.*?Macro-F1:\s*([0-9.]+)",
    flags=re.DOTALL,
)
PANEL_TITLES = {
    "a": "Test accuracy",
    "b": "Macro-F1",
    "c": "Accuracy distribution",
    "d": "Accuracy-Macro-F1 balance",
    "e": "Accuracy drop vs Teacher",
    "f": "Macro-F1 drop vs Teacher",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the paper-facing Figure 2 composite with matplotlib."
    )
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=Path("outputs/paper_figures/h1_results/data/exploratory_ordered_run_registry.csv"),
    )
    parser.add_argument(
        "--class-metrics-csv",
        type=Path,
        default=None,
        help="Deprecated compatibility argument. The composite Figure 2 renderer uses the registry CSV only.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/paper_figures"),
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="fig2_h1_composite_performance",
    )
    parser.add_argument(
        "--dataset-label",
        type=str,
        default="Weld seam feature windows",
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


def extract_test_metrics(metrics_text: str) -> tuple[float, float]:
    match = TEST_METRICS_PATTERN.search(metrics_text)
    if match is None:
        raise ValueError("Could not parse test Accuracy/Macro-F1 from evaluation_metrics.txt")
    return float(match.group(1)), float(match.group(2)) * 100.0


def prepare_registry_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"model_name", "run_dir"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    if "included_in_fig2" in df.columns:
        include_mask = df["included_in_fig2"].astype(str).str.lower().eq("yes")
        df = df.loc[include_mask].copy()

    df["display_name"] = df["model_name"].map(MODEL_NAME_TO_DISPLAY)
    df = df.loc[df["display_name"].isin(DISPLAY_ORDER)].copy()

    duplicates = df["display_name"].duplicated(keep=False)
    if duplicates.any():
        dup_names = sorted(df.loc[duplicates, "display_name"].unique().tolist())
        raise ValueError(f"Duplicate Figure 2 display roles found in {path}: {dup_names}")

    missing_roles = [name for name in DISPLAY_ORDER if name not in df["display_name"].tolist()]
    if missing_roles:
        raise ValueError(f"Missing Figure 2 model roles in {path}: {missing_roles}")

    df["display_name"] = pd.Categorical(df["display_name"], categories=DISPLAY_ORDER, ordered=True)
    return df.sort_values("display_name").reset_index(drop=True)


def build_summary_dataframe(registry_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in registry_df.itertuples(index=False):
        run_dir = Path(str(row.run_dir))
        metrics_path = run_dir / "evaluation_metrics.txt"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing evaluation metrics file: {metrics_path}")
        metrics_text = metrics_path.read_text(encoding="utf-8")
        test_acc_pct, macro_f1_pct = extract_test_metrics(metrics_text)
        rows.append(
            {
                "model_name": row.model_name,
                "display_name": row.display_name,
                "run_dir": str(run_dir),
                "test_acc_pct": test_acc_pct,
                "macro_f1_pct": macro_f1_pct,
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df["display_name"] = pd.Categorical(
        summary_df["display_name"],
        categories=DISPLAY_ORDER,
        ordered=True,
    )
    summary_df = summary_df.sort_values("display_name").reset_index(drop=True)

    teacher_row = summary_df.loc[summary_df["display_name"] == "Teacher"].iloc[0]
    teacher_acc = float(teacher_row["test_acc_pct"])
    teacher_macro_f1 = float(teacher_row["macro_f1_pct"])

    summary_df["delta_acc_pp"] = summary_df["test_acc_pct"].astype(float) - teacher_acc
    summary_df["delta_macro_f1_pp"] = (
        summary_df["macro_f1_pct"].astype(float) - teacher_macro_f1
    )
    return summary_df


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        f"({label})",
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


def plot_lollipop_metric(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    y_min: float = 88.0,
    y_max: float = 100.0,
) -> None:
    x = np.arange(len(DISPLAY_ORDER), dtype=float)
    values = summary_df[value_col].astype(float).to_numpy()
    tick_labels = summary_df["display_name"].astype(str).tolist()

    ax.set_ylim(y_min, y_max)
    ax.set_yticks([88, 92, 96, 100])
    ax.grid(axis="y", color="#cbd5e1", alpha=0.55, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.axhline(y_min, color="#94a3b8", linewidth=0.7, linestyle="--", alpha=0.7)

    for idx, (display_name, value) in enumerate(zip(tick_labels, values)):
        color = METHOD_COLORS[display_name]
        ax.vlines(idx, y_min, value, color=color, linewidth=4.5, alpha=0.50, zorder=1)
        ax.scatter(
            idx,
            value,
            s=65,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.text(
            idx,
            value + 0.32,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#111827",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    style_axis(ax)


def plot_delta_bars(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    lower_limit: float,
    y_ticks: list[float],
) -> None:
    x = np.arange(len(DISPLAY_ORDER), dtype=float)
    values = summary_df[value_col].astype(float).to_numpy()
    tick_labels = summary_df["display_name"].astype(str).tolist()
    colors = [METHOD_COLORS[name] for name in tick_labels]

    bars = ax.bar(x, values, color=colors, edgecolor="#334155", linewidth=0.4, width=0.62)
    ax.axhline(0.0, color="#475569", linewidth=1.2, zorder=3)
    ax.set_ylim(lower_limit, 0.0)
    ax.set_yticks(y_ticks)
    ax.grid(axis="y", color="#cbd5e1", alpha=0.55, linewidth=0.6)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        if value < 0:
            y = value - 0.22
            va = "top"
        else:
            y = value - 0.10
            va = "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            f"{value:+.2f}".replace("+", ""),
            ha="center",
            va=va,
            fontsize=6.8,
            color="#111827",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    style_axis(ax)


def plot_ranking_strip(ax: plt.Axes, summary_df: pd.DataFrame) -> None:
    ranking_df = summary_df.sort_values("test_acc_pct", ascending=False).reset_index(drop=True)
    ax.set_xlim(90.0, 100.0)
    ax.set_ylim(-0.55, 0.55)
    ax.get_yaxis().set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["bottom"].set_color("#475569")
    ax.spines["bottom"].set_linewidth(1.0)
    ax.set_yticks([])
    ax.set_xticks([90, 92, 94, 96, 98, 100])
    ax.grid(axis="x", color="#cbd5e1", alpha=0.45, linewidth=0.6)
    ax.set_axisbelow(True)

    label_offsets = {
        "Teacher": (0.0, 0.16, "bottom"),
        "Student": (0.0, -0.16, "top"),
        "Ablation-1": (0.0, 0.16, "bottom"),
        "Ablation-2": (0.0, -0.16, "top"),
    }

    for row in ranking_df.itertuples(index=False):
        x = float(row.test_acc_pct)
        display_name = str(row.display_name)
        color = METHOD_COLORS[display_name]
        _, y_offset, va = label_offsets[display_name]
        ax.scatter(
            x,
            0.0,
            s=85,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            alpha=0.85,
            zorder=3,
        )
        ax.text(
            x,
            y_offset,
            display_name,
            ha="center",
            va=va,
            fontsize=7.2,
            color="#111827",
        )

def plot_balance_panel(ax: plt.Axes, summary_df: pd.DataFrame) -> None:
    for row in summary_df.itertuples(index=False):
        color = METHOD_COLORS[row.display_name]
        x = float(row.test_acc_pct)
        y = float(row.macro_f1_pct)
        ax.scatter(
            x,
            y,
            s=70,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            alpha=0.85,
            zorder=3,
        )
        x_offset = 0.10
        y_offset = 0.14
        va = "bottom"
        if row.display_name == "Student":
            y_offset = -0.18
            va = "top"
        elif row.display_name == "Teacher":
            y_offset = 0.20
            va = "bottom"
        elif row.display_name == "Ablation-1":
            y_offset = -0.14
            va = "top"

        ax.text(
            x + x_offset,
            y + y_offset,
            row.display_name,
            ha="left",
            va=va,
            fontsize=6.5,
            color="#111827",
        )

    ax.set_xlim(88.0, 100.0)
    ax.set_ylim(88.0, 100.0)
    ax.set_xticks([88, 92, 96, 100])
    ax.set_yticks([88, 92, 96, 100])
    ax.set_xlabel("Test Accuracy (%)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.grid(True, which="major", axis="both", color="#cbd5e1", alpha=0.55, linewidth=0.6)
    ax.set_axisbelow(True)
    style_axis(ax)


def render(
    summary_df: pd.DataFrame,
    output_pdf: Path,
    output_png: Path,
    output_svg: Path,
    *,
    dataset_label: str,
    title: str,
) -> None:
    configure_style()
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 6.8))
    plt.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.11, wspace=0.33, hspace=0.40)

    if title:
        fig.text(
            0.07,
            0.965,
            title,
            ha="left",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            color="#111827",
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[name],
            marker="o",
            linewidth=1.8,
            markersize=5.2,
            label=name,
        )
        for name in DISPLAY_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.965 if title else 0.975),
        ncol=4,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.5,
    )

    plot_lollipop_metric(axes[0, 0], summary_df, "test_acc_pct", "Test Accuracy (%)")
    plot_lollipop_metric(axes[0, 1], summary_df, "macro_f1_pct", "Macro-F1 (%)")
    plot_ranking_strip(axes[0, 2], summary_df)

    plot_balance_panel(axes[1, 0], summary_df)
    plot_delta_bars(
        axes[1, 1],
        summary_df,
        "delta_acc_pp",
        "Δ Accuracy vs Teacher (pp)",
        lower_limit=-6.4,
        y_ticks=[-6, -4, -2, 0],
    )
    plot_delta_bars(
        axes[1, 2],
        summary_df,
        "delta_macro_f1_pp",
        "Δ Macro-F1 vs Teacher (pp)",
        lower_limit=-9.3,
        y_ticks=[-9, -6, -3, 0],
    )

    for label, ax in zip("abcdef", axes.flat):
        add_panel_label(ax, label)
        ax.set_title(PANEL_TITLES[label], pad=8, color="#111827")

    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mpl_config_dir = Path(".mplconfig")
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir.resolve()))

    registry_df = prepare_registry_dataframe(args.registry_csv)
    summary_df = build_summary_dataframe(registry_df)

    output_dir = prepare_output_dir(args.output_root, args.output_name)
    output_pdf = output_dir / f"{args.output_name}.pdf"
    output_png = output_dir / f"{args.output_name}.png"
    output_svg = output_dir / f"{args.output_name}.svg"

    render(
        summary_df=summary_df,
        output_pdf=output_pdf,
        output_png=output_png,
        output_svg=output_svg,
        dataset_label=args.dataset_label,
        title=args.title,
    )

    print(f"Saved Figure 2 outputs to: {output_dir}")
    print(f"PDF: {output_pdf}")
    print(f"PNG: {output_png}")
    print(f"SVG: {output_svg}")


if __name__ == "__main__":
    main()
