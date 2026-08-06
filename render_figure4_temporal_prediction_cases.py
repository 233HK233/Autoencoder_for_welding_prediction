#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = Path("outputs/paper_figures/figure4_temporal_prediction_cases_results_v01")
SEAM_ORDER = ("a01", "b01", "c01", "c02")
CASE_ORDER = ("S0-core", "0->1-boundary", "S1-core", "1->2-boundary")
METHOD_ORDER = ("Teacher-18D", "SEAL-Weld Student-13D", "Student-only-13D")
METHOD_PREFIX = {
    "Teacher-18D": "teacher",
    "SEAL-Weld Student-13D": "seal_student",
    "Student-only-13D": "student_only",
}
METHOD_COLORS = {
    "Teacher-18D": "#30323D",
    "SEAL-Weld Student-13D": "#0B5CAD",
    "Student-only-13D": "#D6811F",
}
METHOD_LINESTYLES = {
    "Teacher-18D": "-",
    "SEAL-Weld Student-13D": "-",
    "Student-only-13D": (0, (4, 2)),
}
METHOD_LINEWIDTHS = {
    "Teacher-18D": 1.05,
    "SEAL-Weld Student-13D": 1.45,
    "Student-only-13D": 1.15,
}
STATE_COLORS = {
    0: "#9ECAE1",
    1: "#E8BD4B",
    2: "#D95F3A",
}
STATE_NAMES = {
    0: "S0",
    1: "S1",
    2: "S2",
}
FIGURE_SIZE = (14.8, 10.6)
OUTPUT_FORMATS = ("svg", "pdf", "png", "tiff")


@dataclass(frozen=True)
class Figure4Data:
    result_dir: Path
    selection_df: pd.DataFrame
    timeseries_df: pd.DataFrame
    attention_df: pd.DataFrame
    highlight_df: pd.DataFrame
    validation_df: pd.DataFrame
    panel_summary_df: pd.DataFrame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Figure 4: temporal prediction cases across weld seams."
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/paper_figures"))
    parser.add_argument("--output-name", type=str, default="figure4_temporal_prediction_cases")
    parser.add_argument("--output-dir-name", type=str, default="figure4_temporal_prediction_cases")
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
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "font.size": 6.6,
            "axes.titlesize": 7.1,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "legend.fontsize": 6.5,
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def prepare_output_dir(output_root: Path, output_name: str) -> Path:
    root = resolve_repo_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = root / f"{output_name}_v{index:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        index += 1


def require_columns(df: pd.DataFrame, required: set[str], *, table_name: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{table_name} missing required columns: {sorted(missing)}")


def _target_probability_column(prefix: str, y_true: int) -> str:
    if y_true not in STATE_NAMES:
        raise ValueError(f"y_true must be one of 0, 1, 2, got {y_true}")
    return f"{prefix}_prob_s{y_true}"


def build_panel_summary(selection_df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "panel_key",
        "seam_name",
        "source_seam_name",
        "case_type",
        "panel_status",
        "selection_rule",
        "sample_index",
        "start_idx",
        "target_idx",
        "y_true",
        "teacher_pred",
        "seal_student_pred",
        "student_only_pred",
        "nearest_boundary_dist",
        "boundary_distance_bin",
    }
    require_columns(selection_df, required, table_name="figure4_panel_selection.csv")

    expected_keys = [f"{seam}|{case}" for seam in SEAM_ORDER for case in CASE_ORDER]
    duplicate_keys = selection_df.loc[selection_df["panel_key"].duplicated(), "panel_key"].tolist()
    if duplicate_keys:
        raise ValueError(f"Duplicate Figure 4 panel_key values: {duplicate_keys}")

    missing_keys = [key for key in expected_keys if key not in set(selection_df["panel_key"].astype(str))]
    unexpected_keys = sorted(set(selection_df["panel_key"].astype(str)).difference(expected_keys))
    if missing_keys or unexpected_keys:
        raise ValueError(f"Figure 4 panels do not match the fixed 4 x 4 layout. Missing={missing_keys}; unexpected={unexpected_keys}")

    panel_df = selection_df.copy()
    panel_df["panel_key"] = pd.Categorical(panel_df["panel_key"], categories=expected_keys, ordered=True)
    panel_df = panel_df.sort_values("panel_key").reset_index(drop=True)
    panel_df["panel_key"] = panel_df["panel_key"].astype(str)
    panel_df["y_true"] = panel_df["y_true"].astype(int)
    panel_df["row_index"] = panel_df["seam_name"].map({seam: idx for idx, seam in enumerate(SEAM_ORDER)}).astype(int)
    panel_df["col_index"] = panel_df["case_type"].map({case: idx for idx, case in enumerate(CASE_ORDER)}).astype(int)
    panel_df["target_state"] = panel_df["y_true"].map(STATE_NAMES)
    for method, prefix in METHOD_PREFIX.items():
        slug = prefix
        panel_df[f"{slug}_target_prob_col"] = panel_df["y_true"].map(
            lambda class_id, method_prefix=prefix: _target_probability_column(method_prefix, int(class_id))
        )
        panel_df[f"{slug}_display_name"] = method
    return panel_df


def load_figure4_data(result_dir: Path) -> Figure4Data:
    resolved = resolve_repo_path(result_dir)
    paths = {
        "selection": resolved / "figure4_panel_selection.csv",
        "timeseries": resolved / "figure4_panel_timeseries.csv",
        "attention": resolved / "figure4_panel_attention.csv",
        "highlights": resolved / "figure4_highlight_cases.csv",
        "validation": resolved / "figure4_validation_summary.csv",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing Figure 4 {name} file: {path}")

    selection_df = pd.read_csv(paths["selection"])
    timeseries_df = pd.read_csv(paths["timeseries"])
    attention_df = pd.read_csv(paths["attention"])
    highlight_df = pd.read_csv(paths["highlights"])
    validation_df = pd.read_csv(paths["validation"])

    panel_summary_df = build_panel_summary(selection_df)
    expected_keys = panel_summary_df["panel_key"].tolist()

    require_columns(
        timeseries_df,
        {
            "panel_key",
            "relative_time_s",
            "state_label",
            "economic_mean_z",
            "economic_pc1",
            "teacher_prob_s0",
            "teacher_prob_s1",
            "teacher_prob_s2",
            "seal_student_prob_s0",
            "seal_student_prob_s1",
            "seal_student_prob_s2",
            "student_only_prob_s0",
            "student_only_prob_s1",
            "student_only_prob_s2",
        },
        table_name="figure4_panel_timeseries.csv",
    )
    require_columns(
        attention_df,
        {"panel_key", "method", "input_step", "relative_time_s", "attention_weight"},
        table_name="figure4_panel_attention.csv",
    )
    require_columns(
        highlight_df,
        {"seam_name", "case_type", "highlight_role", "selection_rule"},
        table_name="figure4_highlight_cases.csv",
    )
    require_columns(
        validation_df,
        {"method", "matched_rows", "max_probability_abs_delta", "prediction_argmax_match"},
        table_name="figure4_validation_summary.csv",
    )

    for table_name, df in (("timeseries", timeseries_df), ("attention", attention_df)):
        missing = [key for key in expected_keys if key not in set(df["panel_key"].astype(str))]
        if missing:
            raise ValueError(f"Figure 4 {table_name} table missing panel_key values: {missing}")

    attention_methods = set(attention_df["method"].astype(str))
    missing_methods = [method for method in METHOD_ORDER if method not in attention_methods]
    if missing_methods:
        raise ValueError(f"Figure 4 attention table missing methods: {missing_methods}")
    expected_attention_rows = len(expected_keys) * len(METHOD_ORDER) * 5
    if len(attention_df) != expected_attention_rows:
        raise ValueError(f"Expected {expected_attention_rows} attention rows, got {len(attention_df)}")

    return Figure4Data(
        result_dir=resolved,
        selection_df=selection_df,
        timeseries_df=timeseries_df,
        attention_df=attention_df,
        highlight_df=highlight_df,
        validation_df=validation_df,
        panel_summary_df=panel_summary_df,
    )


def _scale_to_band(values: pd.Series, *, bottom: float, top: float) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.full_like(arr, (bottom + top) / 2.0)
    lo, hi = np.nanpercentile(finite, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-9:
        return np.full_like(arr, (bottom + top) / 2.0)
    clipped = np.clip(arr, lo, hi)
    return bottom + (clipped - lo) / (hi - lo) * (top - bottom)


def _draw_state_strip(ax: plt.Axes, panel_ts: pd.DataFrame) -> None:
    sorted_ts = panel_ts.sort_values("relative_time_s")
    x = sorted_ts["relative_time_s"].to_numpy(dtype=float)
    if x.size < 2:
        return
    step = float(np.nanmedian(np.diff(x)))
    if not np.isfinite(step) or step <= 0:
        step = 0.01
    for row in sorted_ts.itertuples(index=False):
        state = int(getattr(row, "state_label"))
        x_center = float(getattr(row, "relative_time_s"))
        ax.add_patch(
            Rectangle(
                (x_center - step / 2.0, 0.330),
                step,
                0.055,
                facecolor=STATE_COLORS[state],
                edgecolor="none",
                alpha=0.92,
                zorder=1,
            )
        )


def _draw_attention_ribbon(ax: plt.Axes, panel_attention: pd.DataFrame) -> None:
    band_height = 0.044
    band_gap = 0.010
    top = 0.965
    for method_idx, method in enumerate(METHOD_ORDER):
        subset = panel_attention.loc[panel_attention["method"] == method].sort_values("input_step")
        if subset.empty:
            continue
        weights = subset["attention_weight"].to_numpy(dtype=float)
        max_weight = float(np.nanmax(weights)) if weights.size else 1.0
        max_weight = max(max_weight, 1e-12)
        y0 = top - (method_idx + 1) * band_height - method_idx * band_gap
        x = subset["relative_time_s"].to_numpy(dtype=float)
        step = float(np.nanmedian(np.diff(np.sort(x)))) if x.size > 1 else 0.01
        if not np.isfinite(step) or step <= 0:
            step = 0.01
        for row in subset.itertuples(index=False):
            weight = float(getattr(row, "attention_weight"))
            x_center = float(getattr(row, "relative_time_s"))
            rgba = to_rgba(METHOD_COLORS[method], alpha=0.12 + 0.58 * weight / max_weight)
            ax.add_patch(
                Rectangle(
                    (x_center - step / 2.0, y0),
                    step,
                    band_height,
                    facecolor=rgba,
                    edgecolor="white",
                    linewidth=0.20,
                    zorder=2,
                )
            )


def _draw_boundary_markers(ax: plt.Axes, panel_ts: pd.DataFrame) -> None:
    sorted_ts = panel_ts.sort_values("relative_time_s")
    states = sorted_ts["state_label"].astype(int).to_numpy()
    times = sorted_ts["relative_time_s"].to_numpy(dtype=float)
    for idx in range(1, len(states)):
        if states[idx] != states[idx - 1]:
            x_boundary = (times[idx] + times[idx - 1]) / 2.0
            ax.axvline(
                x_boundary,
                color="#64748B",
                linewidth=0.65,
                linestyle=(0, (2, 2)),
                alpha=0.82,
                zorder=0,
            )


def _draw_layer_labels(ax: plt.Axes) -> None:
    labels = [
        (0.885, "Attention"),
        (0.555, "Economic"),
        (0.357, "State"),
        (0.165, "p(target)"),
    ]
    for y, text in labels:
        ax.text(
            -0.045,
            y,
            text,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=5.4,
            color="#475569",
            clip_on=False,
        )


def _draw_highlight_labels(ax: plt.Axes, highlight_rows: pd.DataFrame) -> None:
    role_labels = {
        "transfer_gain": ("transfer gain", "#137333"),
        "seal_weld_failure": ("SEAL failure", "#B42318"),
    }
    y = 0.720
    for row in highlight_rows.itertuples(index=False):
        role = str(getattr(row, "highlight_role"))
        label, color = role_labels.get(role, (role.replace("_", " "), "#334155"))
        ax.text(
            0.982,
            y,
            label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.6,
            color=color,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": color, "linewidth": 0.35, "boxstyle": "round,pad=0.16"},
        )
        y -= 0.075


def _plot_panel(
    ax: plt.Axes,
    *,
    panel_row: pd.Series,
    panel_ts: pd.DataFrame,
    panel_attention: pd.DataFrame,
    highlight_rows: pd.DataFrame,
    show_layer_labels: bool,
    show_x_label: bool,
) -> None:
    sorted_ts = panel_ts.sort_values("relative_time_s")
    x = sorted_ts["relative_time_s"].to_numpy(dtype=float)
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    x_pad = max(0.012, (x_max - x_min) * 0.025)

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_facecolor("white")
    for y in (0.315, 0.400, 0.745):
        ax.axhline(y, color="#E2E8F0", linewidth=0.45, zorder=0)
    ax.grid(axis="x", color="#E2E8F0", alpha=0.58, linewidth=0.38)
    ax.axvline(0.0, color="#111827", linewidth=0.75, alpha=0.88, zorder=3)
    _draw_boundary_markers(ax, sorted_ts)

    _draw_attention_ribbon(ax, panel_attention)

    economic_y = _scale_to_band(sorted_ts["economic_pc1"], bottom=0.450, top=0.690)
    ax.plot(x, economic_y, color="#52616B", linewidth=0.95, alpha=0.95, zorder=4)

    _draw_state_strip(ax, sorted_ts)

    y_true = int(panel_row["y_true"])
    for method in METHOD_ORDER:
        prefix = METHOD_PREFIX[method]
        col = _target_probability_column(prefix, y_true)
        prob = sorted_ts[col].to_numpy(dtype=float)
        prob_y = 0.055 + np.clip(prob, 0.0, 1.0) * 0.215
        ax.plot(
            x,
            prob_y,
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=METHOD_LINEWIDTHS[method],
            alpha=0.96,
            zorder=5,
        )

    ax.text(
        0.020,
        0.720,
        f"target {STATE_NAMES[y_true]}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.6,
        color="#0F172A",
    )
    if str(panel_row["panel_status"]) == "backfilled":
        ax.text(
            0.020,
            0.060,
            f"backfill: {panel_row['source_seam_name']}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.4,
            color="#B45309",
            bbox={"facecolor": "#FFF7ED", "edgecolor": "#F59E0B", "linewidth": 0.35, "boxstyle": "round,pad=0.16"},
        )
    _draw_highlight_labels(ax, highlight_rows)

    if show_layer_labels:
        _draw_layer_labels(ax)
    if show_x_label:
        ax.set_xlabel("Relative time (s)", labelpad=1.5)
    else:
        ax.set_xticklabels([])
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.55)
    ax.tick_params(axis="x", colors="#475569", length=2.0, width=0.45, pad=1.5)


def render_figure(*, data: Figure4Data, output_base: Path, title: str) -> None:
    configure_style()
    fig, axes = plt.subplots(
        4,
        4,
        figsize=FIGURE_SIZE,
        sharex=False,
        sharey=False,
    )
    plt.subplots_adjust(left=0.070, right=0.985, top=0.895 if title else 0.920, bottom=0.072, wspace=0.145, hspace=0.245)

    if title:
        fig.text(0.070, 0.982, title, ha="left", va="top", fontsize=9.5, fontweight="bold", color="#111827")

    method_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=METHOD_LINEWIDTHS[method] + 0.25,
            label=method,
        )
        for method in METHOD_ORDER
    ]
    state_handles = [
        Patch(facecolor=STATE_COLORS[class_id], edgecolor="none", label=f"{STATE_NAMES[class_id]} true state")
        for class_id in (0, 1, 2)
    ]
    fig.legend(
        handles=method_handles + state_handles,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.982 if not title else 0.948),
        ncol=6,
        frameon=False,
        handlelength=1.95,
        columnspacing=1.10,
        handletextpad=0.36,
    )

    for row_idx, seam_name in enumerate(SEAM_ORDER):
        axes[row_idx, 0].text(
            -0.230,
            0.505,
            seam_name,
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=7.4,
            fontweight="bold",
            color="#111827",
            clip_on=False,
        )
        for col_idx, case_type in enumerate(CASE_ORDER):
            ax = axes[row_idx, col_idx]
            panel_key = f"{seam_name}|{case_type}"
            panel_row = data.panel_summary_df.loc[data.panel_summary_df["panel_key"] == panel_key].iloc[0]
            panel_ts = data.timeseries_df.loc[data.timeseries_df["panel_key"] == panel_key].copy()
            panel_attention = data.attention_df.loc[data.attention_df["panel_key"] == panel_key].copy()
            highlight_rows = data.highlight_df.loc[
                data.highlight_df["seam_name"].astype(str).eq(seam_name)
                & data.highlight_df["case_type"].astype(str).eq(case_type)
            ]
            _plot_panel(
                ax,
                panel_row=panel_row,
                panel_ts=panel_ts,
                panel_attention=panel_attention,
                highlight_rows=highlight_rows,
                show_layer_labels=col_idx == 0,
                show_x_label=row_idx == len(SEAM_ORDER) - 1,
            )
            if row_idx == 0:
                ax.set_title(case_type, pad=4.0, color="#111827")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_qa_notes(path: Path, *, data: Figure4Data) -> None:
    backfilled = data.panel_summary_df.loc[data.panel_summary_df["panel_status"].astype(str).eq("backfilled")]
    backfilled_keys = ", ".join(backfilled["panel_key"].astype(str).tolist()) if not backfilled.empty else "none"
    max_prob_delta = float(data.validation_df["max_probability_abs_delta"].astype(float).max())
    argmax_match = bool(data.validation_df["prediction_argmax_match"].astype(bool).all())
    highlight_roles = ", ".join(data.highlight_df["highlight_role"].astype(str).tolist())

    lines = [
        "Figure 4 QA notes",
        "",
        "Core conclusion: SEAL-Weld preserves teacher-like temporal predictions across weld seams while exposing where transfer gain and residual failure occur near state transitions.",
        "Figure archetype: quantitative grid.",
        "Backend: Python/Matplotlib only.",
        f"Final size: {FIGURE_SIZE[0]:.1f} x {FIGURE_SIZE[1]:.1f} inches before manuscript scaling.",
        "Export formats: SVG, PDF, PNG, and TIFF.",
        "",
        "Guide compliance:",
        f"- 4 x 4 guided layout: {'PASS' if len(data.panel_summary_df) == 16 else 'FAIL'}",
        f"- Fixed seam order: {', '.join(SEAM_ORDER)}",
        f"- Fixed case order: {', '.join(CASE_ORDER)}",
        "- Four-layer panel grammar: attention ribbon, economic_pc1, true-state strip, target-class probability tracks.",
        "- Compact probability tracks: target-class only",
        f"- Backfilled panels: {backfilled_keys}",
        f"- Highlight roles: {highlight_roles}",
        "",
        "Source-data checks:",
        f"- Panel selection rows: {len(data.selection_df)}",
        f"- Panel timeseries rows: {len(data.timeseries_df)}",
        f"- Panel attention rows: {len(data.attention_df)}",
        f"- Continuous/test argmax consistency: {'PASS' if argmax_match else 'FAIL'}",
        f"- Max probability absolute delta: {max_prob_delta:.2e}",
        "",
        "Reviewer-risk notes:",
        "- c01|S0-core is cross-seam backfilled and must not be described as a native c01 S0-core sample.",
        "- The main figure uses target-class probabilities for readability; full S0/S1/S2 probability traces belong in supplementary material if needed.",
        "- Boundary markers are shown only when a state change falls inside the local plotted window.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mpl_config_dir = Path(".mplconfig")
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir.resolve()))

    data = load_figure4_data(args.result_dir)
    output_dir = prepare_output_dir(args.output_root, args.output_dir_name)
    output_base = output_dir / args.output_name
    panel_summary_csv = output_dir / f"{args.output_name}_panel_summary.csv"
    qa_notes = output_dir / "figure4_qa_notes.txt"

    data.panel_summary_df.to_csv(panel_summary_csv, index=False)
    render_figure(data=data, output_base=output_base, title=args.title)
    write_qa_notes(qa_notes, data=data)

    print(f"Saved Figure 4 outputs to: {output_dir}")
    for fmt in OUTPUT_FORMATS:
        print(f"{fmt.upper()}: {output_base.with_suffix('.' + fmt)}")
    print(f"Panel summary: {panel_summary_csv}")
    print(f"QA notes: {qa_notes}")


if __name__ == "__main__":
    main()
