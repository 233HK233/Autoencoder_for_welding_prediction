#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from visualize_weld_raw_data_for_paper import (
    PngCanvas,
    Rect,
    compute_label_segments,
    draw_plot_border,
    draw_state_backgrounds,
    draw_transition_lines,
    load_raw_seam_csv,
    map_x,
    map_y,
)

LABEL_NAMES = {0: "S0", 1: "S1", 2: "S2"}
CASE_ORDER = ("S0-core", "0->1-boundary", "S1-core", "1->2-boundary")
CANVAS_W = 1600.0
CANVAS_H = 1400.0
PANEL_COLS = 4
PANEL_ROWS = 4
MARGIN_X = 90.0
MARGIN_Y = 70.0
GAP_X = 28.0
GAP_Y = 34.0
WINDOW_RADIUS = 25
CHANNELS_TO_DRAW = (0, 1, 2)
LINE_COLORS = ("#1f5aa6", "#b0433b", "#2f7d32")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a paper-ready Figure 1 PNG gallery.")
    parser.add_argument("--panel-csv", type=Path, required=True, help="Panel candidate CSV from select_figure1_panels.py")
    parser.add_argument("--raw-data-dir", type=Path, default=Path("Data/raw_data"), help="Directory containing seam CSV files.")
    parser.add_argument("--output-png", type=Path, required=True, help="Output PNG path.")
    return parser.parse_args(argv)


def _panel_rect(row_idx: int, col_idx: int) -> Rect:
    panel_w = (CANVAS_W - 2 * MARGIN_X - (PANEL_COLS - 1) * GAP_X) / PANEL_COLS
    panel_h = (CANVAS_H - 2 * MARGIN_Y - (PANEL_ROWS - 1) * GAP_Y) / PANEL_ROWS
    x = MARGIN_X + col_idx * (panel_w + GAP_X)
    y = MARGIN_Y + row_idx * (panel_h + GAP_Y)
    return Rect(x, y, panel_w, panel_h)


def _subrects(panel: Rect) -> tuple[Rect, Rect, Rect]:
    title_h = 22.0
    state_h = 24.0
    footer_h = 18.0
    gap = 8.0
    plot_h = panel.h - title_h - state_h - footer_h - 3 * gap
    plot_rect = Rect(panel.x + 8.0, panel.y + title_h + gap, panel.w - 16.0, plot_h)
    state_rect = Rect(panel.x + 8.0, plot_rect.bottom + gap, panel.w - 16.0, state_h)
    footer_rect = Rect(panel.x + 8.0, state_rect.bottom + gap, panel.w - 16.0, footer_h)
    return plot_rect, state_rect, footer_rect


def _window_range(target_idx: int, total_len: int) -> tuple[int, int]:
    start = max(0, target_idx - WINDOW_RADIUS)
    end = min(total_len, target_idx + WINDOW_RADIUS + 1)
    if end - start < 20:
        end = min(total_len, start + 20)
    return start, end


def _draw_series(canvas: PngCanvas, rect: Rect, data: np.ndarray, x0: int, x1: int) -> None:
    window = data[x0:x1, list(CHANNELS_TO_DRAW)]
    y_min = float(window.min())
    y_max = float(window.max())
    if abs(y_max - y_min) < 1e-8:
        y_min -= 1.0
        y_max += 1.0

    xs = np.arange(x0, x1, dtype=np.float64)
    for ch_idx, color in zip(range(window.shape[1]), LINE_COLORS):
        points = [
            (map_x(float(x), float(x0), float(x1 - 1), rect), map_y(float(v), y_min, y_max, rect))
            for x, v in zip(xs, window[:, ch_idx])
        ]
        canvas.polyline(points, stroke=color, stroke_width=1.2)


def _draw_state_bar(canvas: PngCanvas, rect: Rect, labels: np.ndarray, x0: int, x1: int) -> None:
    segments = compute_label_segments(labels)
    local_segments = []
    for label, start, end in segments:
        if end <= x0 or start >= x1:
            continue
        local_segments.append((label, max(start, x0), min(end, x1)))
    draw_state_backgrounds(canvas, rect, local_segments, float(x0), float(x1 - 1))
    draw_transition_lines(canvas, rect, local_segments, float(x0), float(x1 - 1))
    draw_plot_border(canvas, rect)


def _draw_footer(canvas: PngCanvas, rect: Rect, row: pd.Series) -> None:
    gt = LABEL_NAMES.get(int(row["y_true"])) if pd.notna(row["y_true"]) else "NA"
    pred = LABEL_NAMES.get(int(row["final_pred"])) if pd.notna(row["final_pred"]) else "NA"
    conf = float(row["confidence"]) if pd.notna(row["confidence"]) else float("nan")
    footer = f"GT={gt}  Pred={pred}  Conf={conf:.3f}"
    if str(row.get("panel_status", "selected")) == "backfilled":
        footer += f"  donor={row.get('source_seam_name', 'NA')}"
    if str(row.get("panel_status", "selected")) == "missing":
        footer = "N/A"
    canvas.text(rect.x, rect.y, footer, size=8.5, fill="#222222")


def render_figure(panel_df: pd.DataFrame, raw_data_dir: Path, output_png: Path) -> None:
    canvas = PngCanvas(CANVAS_W, CANVAS_H, scale=3.0)
    canvas.text(CANVAS_W / 2.0, 24.0, "Figure 1. Qualitative prediction gallery for horizon=1 state forecasting", size=16.0, anchor="middle", bold=True)

    seams = ["a01", "b01", "c01", "c02"]
    raw_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for seam in seams:
        data, labels, _raw_labels = load_raw_seam_csv(raw_data_dir / f"{seam}.csv")
        raw_cache[seam] = (data, labels)

    for row_idx, seam_name in enumerate(seams):
        seam_rows = panel_df.loc[panel_df["seam_name"] == seam_name].copy()
        for col_idx, case_type in enumerate(CASE_ORDER):
            panel = _panel_rect(row_idx, col_idx)
            plot_rect, state_rect, footer_rect = _subrects(panel)
            draw_plot_border(canvas, panel)
            canvas.text(panel.x + panel.w / 2.0, panel.y + 4.0, f"{seam_name} | {case_type}", size=10.5, anchor="middle", bold=True)

            panel_match = seam_rows.loc[seam_rows["case_type"] == case_type]
            if panel_match.empty:
                canvas.text(panel.x + panel.w / 2.0, panel.y + panel.h / 2.0, "N/A", size=14.0, anchor="middle", valign="middle", bold=True)
                continue

            row = panel_match.iloc[0]
            source_seam = str(row.get("source_seam_name", seam_name)) if pd.notna(row.get("source_seam_name", seam_name)) else seam_name
            data, labels = raw_cache[source_seam]
            if pd.isna(row["target_idx"]):
                canvas.text(panel.x + panel.w / 2.0, panel.y + panel.h / 2.0, "N/A", size=14.0, anchor="middle", valign="middle", bold=True)
                continue

            target_idx = int(row["target_idx"])
            x0, x1 = _window_range(target_idx, len(labels))
            _draw_series(canvas, plot_rect, data, x0, x1)
            _draw_state_bar(canvas, state_rect, labels, x0, x1)
            marker_x = map_x(float(target_idx), float(x0), float(x1 - 1), plot_rect)
            canvas.line(marker_x, plot_rect.y, marker_x, state_rect.bottom, stroke="#111111", stroke_width=0.8, dash=(2.0, 2.0))
            _draw_footer(canvas, footer_rect, row)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    panel_df = pd.read_csv(args.panel_csv)
    render_figure(panel_df, args.raw_data_dir, args.output_png)
    print(f"Saved Figure 1 PNG to: {args.output_png}")


if __name__ == "__main__":
    main()
