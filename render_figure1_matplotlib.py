#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from data_utils import load_seam_csv

CASE_ORDER = ("S0-core", "0->1-boundary", "S1-core", "1->2-boundary")
SEAM_ORDER = ("a01", "b01", "c01", "c02")
STATE_NAMES = {0: "Quasistable", 1: "Nonstationary", 2: "Instability"}
STATE_SHORT = {0: "S0", 1: "S1", 2: "S2"}
STATE_COLORS = {
    0: "#b7d4ea",
    1: "#f4d06f",
    2: "#e07a5f",
}
LINE_COLORS = ["#1d3557", "#457b9d", "#2a9d8f"]
WINDOW_RADIUS = 25
SAMPLE_PERIOD_S = 0.01


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Figure 1 with matplotlib for paper submission.")
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--raw-data-dir", type=Path, default=Path("Data/raw_data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/paper_figures"))
    parser.add_argument("--output-name", type=str, default="figure1_h1_case_gallery_matplotlib")
    return parser.parse_args(argv)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
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


def load_raw_cache(raw_data_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for seam in SEAM_ORDER:
        data, labels = load_seam_csv(raw_data_dir / f"{seam}.csv")
        cache[seam] = (data, labels)
    return cache


def window_range(target_idx: int, total_len: int) -> tuple[int, int]:
    start = max(0, target_idx - WINDOW_RADIUS)
    end = min(total_len, target_idx + WINDOW_RADIUS + 1)
    if end - start < 20:
        end = min(total_len, start + 20)
    return start, end


def channel_subset(data: np.ndarray, count: int = 3) -> list[int]:
    variances = np.var(data, axis=0)
    order = np.argsort(-variances)
    return [int(v) for v in order[:count]]


def make_channel_labels(channel_indices: list[int]) -> list[str]:
    return [
        "1st Principal High-Var Channel",
        "2nd Principal High-Var Channel",
        "3rd Principal High-Var Channel",
    ][: len(channel_indices)]


def build_relative_time_axis(xs: np.ndarray, target_idx: int, sample_period_s: float = SAMPLE_PERIOD_S) -> np.ndarray:
    return (xs.astype(np.float64) - float(target_idx)) * float(sample_period_s)


def build_background_segments(
    labels: np.ndarray,
    x0: int,
    x1: int,
    target_idx: int,
    sample_period_s: float = SAMPLE_PERIOD_S,
) -> list[tuple[int, float, float]]:
    segments: list[tuple[int, float, float]] = []
    start = x0
    current = int(labels[x0])
    for idx in range(x0 + 1, x1):
        if int(labels[idx]) != current:
            left = float((start - target_idx) * sample_period_s)
            right = float((idx - target_idx) * sample_period_s)
            segments.append((current, left, right))
            start = idx
            current = int(labels[idx])
    left = float((start - target_idx) * sample_period_s)
    right = float(((x1 - 1) - target_idx) * sample_period_s)
    segments.append((current, left, right))
    return segments


def draw_background_states(ax: plt.Axes, labels: np.ndarray, x0: int, x1: int, target_idx: int) -> None:
    segments = build_background_segments(labels, x0=x0, x1=x1, target_idx=target_idx, sample_period_s=SAMPLE_PERIOD_S)
    for seg_idx, (label, left, right) in enumerate(segments):
        ax.axvspan(left, right, color=STATE_COLORS[label], alpha=0.20, lw=0)
        if seg_idx < len(segments) - 1:
            ax.axvline(right, color="#888888", lw=0.6, ls="--", alpha=0.7)


def draw_prediction_target(ax: plt.Axes, target_time_s: float) -> None:
    ax.axvline(target_time_s, color="#b22222", lw=1.4, ls=(0, (4, 2)))
    ymin, ymax = ax.get_ylim()
    arrow_y = ymax - 0.08 * (ymax - ymin)
    text_y = ymax - 0.03 * (ymax - ymin)
    ax.annotate(
        "",
        xy=(target_time_s + 0.04, arrow_y),
        xytext=(target_time_s, arrow_y),
        arrowprops=dict(arrowstyle="->", color="#b22222", lw=1.0),
    )
    ax.text(
        target_time_s + 0.045,
        text_y,
        "Horizon=1",
        color="#b22222",
        fontsize=7,
        ha="left",
        va="top",
    )


def build_local_state_trajectory(local_df: pd.DataFrame, target_idx: int, sample_period_s: float = SAMPLE_PERIOD_S) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_df = local_df.sort_values("target_idx").copy()
    xs = build_relative_time_axis(local_df["target_idx"].to_numpy(dtype=np.int64), target_idx=target_idx, sample_period_s=sample_period_s)
    gt = local_df["y_true"].to_numpy(dtype=np.int64)
    pred = local_df["final_pred"].to_numpy(dtype=np.int64)
    return xs, gt, pred


def add_state_trajectory_inset(ax: plt.Axes, local_df: pd.DataFrame, target_idx: int) -> None:
    inset = ax.inset_axes([0.58, 0.60, 0.36, 0.27])
    inset.patch.set_facecolor("white")
    inset.patch.set_alpha(0.85)
    xs, gt, pred = build_local_state_trajectory(local_df, target_idx=target_idx, sample_period_s=SAMPLE_PERIOD_S)
    inset.step(xs, gt, where="post", color="#222222", lw=1.0, label="GT")
    inset.step(xs, pred, where="post", color="#c0392b", lw=1.0, ls="--", label="Pred")
    inset.set_xlim(float(xs[0]), float(xs[-1]))
    inset.set_ylim(-0.2, 2.2)
    inset.set_yticks([0, 1, 2])
    inset.set_yticklabels(["S0", "S1", "S2"])
    inset.set_xticks([float(xs[0]), 0.0, float(xs[-1])])
    inset.tick_params(axis="both", labelsize=5, pad=1)
    inset.grid(True, axis="y", alpha=0.18, lw=0.4)
    inset.axvline(0.0, color="#b22222", lw=0.8, ls=(0, (3, 2)))
    for spine in inset.spines.values():
        spine.set_linewidth(0.6)


def add_prediction_badge(ax: plt.Axes, row: pd.Series) -> None:
    gt = int(row["y_true"])
    pred = int(row["final_pred"])
    conf = float(row["confidence"])
    if pred == gt:
        text = "Accurate"
        bbox = dict(boxstyle="round,pad=0.22", fc="white", ec="#2a9d55", lw=0.9)
        color = "#2a9d55"
    else:
        text = f"Misclassified as {STATE_SHORT.get(pred, pred)}\nConf: {conf*100:.0f}%"
        bbox = dict(boxstyle="round,pad=0.22", fc="white", ec="#c0392b", lw=0.9)
        color = "#c0392b"
    ax.text(
        0.03,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=color,
        bbox=bbox,
        fontsize=7,
    )


def render(panel_df: pd.DataFrame, pred_df: pd.DataFrame, raw_cache: dict[str, tuple[np.ndarray, np.ndarray]], output_pdf: Path, output_png: Path, output_svg: Path | None) -> None:
    configure_style()
    fig, axes = plt.subplots(4, 4, figsize=(12.5, 10.0), sharex=False, sharey=False)
    plt.subplots_adjust(left=0.075, right=0.89, top=0.86, bottom=0.08, wspace=0.22, hspace=0.28)

    source_channels: dict[str, list[int]] = {}
    for seam, (data, _labels) in raw_cache.items():
        source_channels[seam] = channel_subset(data, count=3)
    line_labels = make_channel_labels(source_channels[SEAM_ORDER[0]])

    for row_idx, seam_name in enumerate(SEAM_ORDER):
        for col_idx, case_type in enumerate(CASE_ORDER):
            ax = axes[row_idx, col_idx]
            panel_key = f"{seam_name}|{case_type}"
            row_df = panel_df.loc[panel_df["panel_key"] == panel_key]
            if row_df.empty:
                raise ValueError(f"panel_df is missing required panel {panel_key}")
            row = row_df.iloc[0]
            source_seam = str(row.get("source_seam_name", seam_name))
            data, labels = raw_cache[source_seam]
            target_idx = int(row["target_idx"])
            x0, x1 = window_range(target_idx, len(labels))
            xs = np.arange(x0, x1)
            rel_time = build_relative_time_axis(xs, target_idx=target_idx, sample_period_s=SAMPLE_PERIOD_S)
            channels = source_channels[source_seam]
            window = data[x0:x1][:, channels]

            draw_background_states(ax, labels, x0, x1, target_idx=target_idx)

            for line_idx, channel_idx in enumerate(channels):
                ax.plot(
                    rel_time,
                    data[x0:x1, channel_idx],
                    color=LINE_COLORS[line_idx],
                    lw=1.1,
                )

            source_seam_df = pred_df.loc[pred_df["seam_name"] == source_seam].copy()
            local_pred_df = source_seam_df.loc[
                (source_seam_df["target_idx"] >= x0) & (source_seam_df["target_idx"] < x1)
            ].copy()
            if not local_pred_df.empty:
                add_state_trajectory_inset(ax, local_pred_df, target_idx=target_idx)
            add_prediction_badge(ax, row)
            ax.set_title(f"{seam_name} | {case_type}", pad=3)
            ax.grid(True, axis="y", alpha=0.22, lw=0.5)

            if col_idx == 0:
                ax.set_ylabel("Normalized amplitude")
            else:
                ax.set_ylabel("")

            if row_idx == len(SEAM_ORDER) - 1:
                ax.set_xlabel("Relative Time (s)")
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            ymin = float(np.min(window))
            ymax = float(np.max(window))
            if abs(ymax - ymin) < 1e-8:
                ymin -= 1.0
                ymax += 1.0
            margin = 0.10 * (ymax - ymin)
            ax.set_ylim(ymin - margin, ymax + margin)
            ax.set_xlim(float(rel_time[0]), float(rel_time[-1]))
            draw_prediction_target(ax, target_time_s=0.0)

    line_handles = [Line2D([0], [0], color=LINE_COLORS[idx], lw=1.4, label=line_labels[idx]) for idx in range(3)]
    state_handles = [Patch(facecolor=STATE_COLORS[idx], edgecolor="none", alpha=0.35, label=f"{STATE_SHORT[idx]} ({STATE_NAMES[idx]})") for idx in range(3)]
    fig.legend(
        handles=line_handles + state_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.2,
    )

    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    if output_svg is not None:
        fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mpl_config_dir = Path(".mplconfig")
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir.resolve()))
    panel_df = pd.read_csv(args.panel_csv)
    pred_csv = args.predictions_csv
    if pred_csv is None:
        pred_csv = args.panel_csv.parent.parent / "test_predictions_enriched.csv"
    pred_df = pd.read_csv(pred_csv)
    raw_cache = load_raw_cache(args.raw_data_dir)
    output_dir = prepare_output_dir(args.output_root, args.output_name)
    output_pdf = output_dir / f"{args.output_name}.pdf"
    output_png = output_dir / f"{args.output_name}.png"
    output_svg = output_dir / f"{args.output_name}.svg"
    render(panel_df, pred_df, raw_cache, output_pdf, output_png, output_svg)
    print(f"Saved Figure 1 outputs to: {output_dir}")
    print(f"PDF: {output_pdf}")
    print(f"PNG: {output_png}")
    print(f"SVG: {output_svg}")


if __name__ == "__main__":
    main()
