#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import Color
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas as pdf_canvas

from data_utils import LABEL_MAP

LABEL_DISPLAY_NAMES = {
    0: "Quasi-stable",
    1: "Non-stationary",
    2: "Instability",
}

STATE_COLORS = {
    0: "#d8ead3",
    1: "#fff2cc",
    2: "#f4cccc",
}

GROUP_STYLE = {
    "economic": {"line": "#1f5aa6", "fill": "#b8d0f0"},
    "expensive": {"line": "#b0433b", "fill": "#efc0bb"},
}

MAIN_SIZE = (828.0, 662.0)
SUPP_SIZE = (828.0, 562.0)
PNG_SCALE = 4.0
HEATMAP_ROW_LABEL_OFFSET = 8.0
HEATMAP_GROUP_LABEL_OFFSET = 66.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready raw-data visualizations for a weld seam."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Data/raw_data"),
        help="Directory containing raw seam CSV files.",
    )
    parser.add_argument(
        "--seam",
        type=str,
        default="b01.csv",
        help="CSV filename of the representative seam to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_figures"),
        help="Directory for figure outputs.",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=0.01,
        help="Sampling interval in seconds.",
    )
    parser.add_argument(
        "--expensive-indices",
        type=str,
        default="3,4,5,6,7",
        help="Comma-separated 0-based feature indices treated as expensive.",
    )
    return parser.parse_args()


def parse_feature_indices(text: str, total_features: int) -> list[int]:
    indices: list[int] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        idx = int(raw)
        if idx < 0 or idx >= total_features:
            raise ValueError(f"feature index {idx} is out of range for {total_features} features")
        indices.append(idx)

    unique = sorted(set(indices))
    if not unique:
        raise ValueError("expensive feature indices cannot be empty")
    return unique


def load_raw_seam_csv(
    file_path: str | Path,
    n_features: int = 18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(file_path, header=None)
    if df.shape[1] < n_features + 1:
        raise ValueError(
            f"Expected at least {n_features + 1} columns in {file_path}, got {df.shape[1]}"
        )

    data = df.iloc[:, :n_features].to_numpy(dtype=np.float32)
    raw_labels = df.iloc[:, n_features].astype(str).str.strip()
    mapped = raw_labels.map(LABEL_MAP)
    if mapped.isna().any():
        bad = sorted(raw_labels[mapped.isna()].unique().tolist())
        raise ValueError(f"Unmapped labels in {file_path}: {bad}")
    labels = mapped.to_numpy(dtype=np.int64)
    return data, labels, raw_labels.to_numpy(dtype=object)


def build_feature_groups(total_features: int, expensive_indices: Iterable[int]) -> dict[str, list[int]]:
    expensive = sorted(set(int(idx) for idx in expensive_indices))
    if any(idx < 0 or idx >= total_features for idx in expensive):
        raise ValueError("expensive feature indices are out of range")
    economic = [idx for idx in range(total_features) if idx not in expensive]
    if not economic:
        raise ValueError("economic feature group cannot be empty")
    return {"economic": economic, "expensive": expensive}


def select_representative_channels(
    data: np.ndarray,
    labels: np.ndarray,
    feature_groups: dict[str, list[int]],
    top_k: int = 2,
) -> dict[str, list[int]]:
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    if data.shape[0] != labels.shape[0]:
        raise ValueError("data and labels must have matching lengths")

    def _score_column(column: np.ndarray) -> float:
        if np.allclose(column, column[0]):
            return 0.0
        corr = np.corrcoef(column.astype(np.float64), labels.astype(np.float64))[0, 1]
        if np.isnan(corr):
            return 0.0
        return float(abs(corr))

    selected: dict[str, list[int]] = {}
    for group_name, indices in feature_groups.items():
        ranking = sorted(indices, key=lambda idx: (-_score_column(data[:, idx]), idx))
        selected[group_name] = ranking[: min(top_k, len(ranking))]
    return selected


def zscore_per_feature(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, ddof=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (data - mean) / std


def compute_label_segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    if labels.size == 0:
        return []

    segments: list[tuple[int, int, int]] = []
    start = 0
    current = int(labels[0])
    for idx in range(1, labels.shape[0]):
        if int(labels[idx]) != current:
            segments.append((current, start, idx))
            start = idx
            current = int(labels[idx])
    segments.append((current, start, labels.shape[0]))
    return segments


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[idx : idx + 2], 16) for idx in (0, 2, 4))


def hex_with_alpha(color: str, alpha: float) -> tuple[int, int, int, int]:
    r, g, b = hex_to_rgb(color)
    return (r, g, b, max(0, min(255, int(round(alpha * 255)))))


def color_to_reportlab(color: str, alpha: float = 1.0) -> Color:
    r, g, b = hex_to_rgb(color)
    return Color(r / 255.0, g / 255.0, b / 255.0, alpha=alpha)


def blend_channel(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * t))


def blend_hex(color_a: str, color_b: str, t: float) -> str:
    rgb_a = hex_to_rgb(color_a)
    rgb_b = hex_to_rgb(color_b)
    blended = tuple(blend_channel(a, b, t) for a, b in zip(rgb_a, rgb_b))
    return "#{:02x}{:02x}{:02x}".format(*blended)


def diverging_color(value: float, vmin: float = -3.0, vmax: float = 3.0) -> str:
    low = "#3b4cc0"
    mid = "#f7f7f7"
    high = "#b40426"

    value = min(max(value, vmin), vmax)
    if vmax <= vmin:
        return mid

    pivot = 0.0
    if value <= pivot:
        span = max(pivot - vmin, 1e-8)
        t = (value - vmin) / span
        return blend_hex(low, mid, t)

    span = max(vmax - pivot, 1e-8)
    t = (value - pivot) / span
    return blend_hex(mid, high, t)


class DrawingCanvas:
    def rectangle(
        self,
        rect: Rect,
        *,
        fill: str | None = None,
        stroke: str | None = None,
        stroke_width: float = 1.0,
        fill_alpha: float = 1.0,
        stroke_alpha: float = 1.0,
    ) -> None:
        raise NotImplementedError

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str,
        stroke_width: float = 1.0,
        stroke_alpha: float = 1.0,
        dash: tuple[float, float] | None = None,
    ) -> None:
        raise NotImplementedError

    def polyline(
        self,
        points: list[tuple[float, float]],
        *,
        stroke: str,
        stroke_width: float = 1.0,
    ) -> None:
        raise NotImplementedError

    def polygon(
        self,
        points: list[tuple[float, float]],
        *,
        fill: str,
        stroke: str | None = None,
        stroke_width: float = 1.0,
        fill_alpha: float = 1.0,
        stroke_alpha: float = 1.0,
    ) -> None:
        raise NotImplementedError

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: float = 11.0,
        fill: str = "#111111",
        anchor: str = "start",
        valign: str = "top",
        bold: bool = False,
    ) -> None:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError


class SvgCanvas(DrawingCanvas):
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
        self.elements = [
            f'<rect x="0" y="0" width="{width:.2f}" height="{height:.2f}" fill="#ffffff" />'
        ]

    def rectangle(
        self,
        rect: Rect,
        *,
        fill: str | None = None,
        stroke: str | None = None,
        stroke_width: float = 1.0,
        fill_alpha: float = 1.0,
        stroke_alpha: float = 1.0,
    ) -> None:
        fill_attr = fill if fill is not None else "none"
        stroke_attr = stroke if stroke is not None else "none"
        extra = []
        if fill is not None:
            extra.append(f'fill-opacity="{fill_alpha:.3f}"')
        if stroke is not None:
            extra.append(f'stroke-opacity="{stroke_alpha:.3f}"')
            extra.append(f'stroke-width="{stroke_width:.2f}"')
        self.elements.append(
            f'<rect x="{rect.x:.2f}" y="{rect.y:.2f}" width="{rect.w:.2f}" height="{rect.h:.2f}" '
            f'fill="{fill_attr}" stroke="{stroke_attr}" {" ".join(extra)} />'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str,
        stroke_width: float = 1.0,
        stroke_alpha: float = 1.0,
        dash: tuple[float, float] | None = None,
    ) -> None:
        dash_attr = ""
        if dash is not None:
            dash_attr = f' stroke-dasharray="{dash[0]:.2f},{dash[1]:.2f}"'
        self.elements.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.2f}" stroke-opacity="{stroke_alpha:.3f}"{dash_attr} />'
        )

    def polyline(
        self,
        points: list[tuple[float, float]],
        *,
        stroke: str,
        stroke_width: float = 1.0,
    ) -> None:
        encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.elements.append(
            f'<polyline points="{encoded}" fill="none" stroke="{stroke}" stroke-width="{stroke_width:.2f}" />'
        )

    def polygon(
        self,
        points: list[tuple[float, float]],
        *,
        fill: str,
        stroke: str | None = None,
        stroke_width: float = 1.0,
        fill_alpha: float = 1.0,
        stroke_alpha: float = 1.0,
    ) -> None:
        encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        stroke_attr = stroke if stroke is not None else "none"
        self.elements.append(
            f'<polygon points="{encoded}" fill="{fill}" fill-opacity="{fill_alpha:.3f}" '
            f'stroke="{stroke_attr}" stroke-width="{stroke_width:.2f}" stroke-opacity="{stroke_alpha:.3f}" />'
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: float = 11.0,
        fill: str = "#111111",
        anchor: str = "start",
        valign: str = "top",
        bold: bool = False,
    ) -> None:
        anchor_map = {"start": "start", "middle": "middle", "end": "end"}
        baseline_map = {"top": "hanging", "middle": "middle", "bottom": "baseline"}
        weight = "700" if bold else "400"
        self.elements.append(
            f'<text x="{x:.2f}" y="{y:.2f}" fill="{fill}" font-size="{size:.2f}" '
            f'font-family="Times New Roman, DejaVu Serif, serif" font-weight="{weight}" '
            f'text-anchor="{anchor_map[anchor]}" dominant-baseline="{baseline_map[valign]}">{value}</text>'
        )

    def save(self, path: Path) -> None:
        body = "\n".join(self.elements)
        path.write_text(
            "\n".join(
                [
                    '<?xml version="1.0" encoding="UTF-8"?>',
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width:.2f}pt" height="{self.height:.2f}pt" '
                    f'viewBox="0 0 {self.width:.2f} {self.height:.2f}">',
                    body,
                    "</svg>",
                ]
            ),
            encoding="utf-8",
        )


class PdfCanvas(DrawingCanvas):
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
        self._path: Path | None = None
        self.canvas = None

    def bind(self, path: Path) -> None:
        self._path = path
        self.canvas = pdf_canvas.Canvas(str(path), pagesize=(self.width, self.height))
        self.canvas.setTitle(path.stem)
        self.canvas.setAuthor("OpenAI Codex")

    def _require_canvas(self):
        if self.canvas is None:
            raise RuntimeError("PDF canvas is not bound to an output path")
        return self.canvas

    def _with_alpha(self, *, fill_alpha: float | None = None, stroke_alpha: float | None = None) -> None:
        canvas = self._require_canvas()
        try:
            if fill_alpha is not None:
                canvas.setFillAlpha(fill_alpha)
            if stroke_alpha is not None:
                canvas.setStrokeAlpha(stroke_alpha)
        except AttributeError:
            return

    def rectangle(
        self,
        rect: Rect,
        *,
        fill: str | None = None,
        stroke: str | None = None,
        stroke_width: float = 1.0,
        fill_alpha: float = 1.0,
        stroke_alpha: float = 1.0,
    ) -> None:
        canvas = self._require_canvas()
        canvas.saveState()
        pdf_y = self.height - rect.y - rect.h
        if fill is not None:
            canvas.setFillColor(color_to_reportlab(fill, fill_alpha))
            self._with_alpha(fill_alpha=fill_alpha)
        else:
            canvas.setFillColor(Color(1, 1, 1, alpha=0))
        if stroke is not None:
            canvas.setStrokeColor(color_to_reportlab(stroke, stroke_alpha))
            canvas.setLineWidth(stroke_width)
            self._with_alpha(stroke_alpha=stroke_alpha)
        else:
            canvas.setStrokeColor(Color(1, 1, 1, alpha=0))
        canvas.rect(rect.x, pdf_y, rect.w, rect.h, fill=int(fill is not None), stroke=int(stroke is not None))
        canvas.restoreState()

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str,
        stroke_width: float = 1.0,
        stroke_alpha: float = 1.0,
        dash: tuple[float, float] | None = None,
    ) -> None:
        canvas = self._require_canvas()
        canvas.saveState()
        canvas.setStrokeColor(color_to_reportlab(stroke, stroke_alpha))
        canvas.setLineWidth(stroke_width)
        self._with_alpha(stroke_alpha=stroke_alpha)
        if dash is not None:
            canvas.setDash(dash[0], dash[1])
        canvas.line(x1, self.height - y1, x2, self.height - y2)
        canvas.restoreState()

    def polyline(
        self,
        points: list[tuple[float, float]],
        *,
        stroke: str,
        stroke_width: float = 1.0,
    ) -> None:
        if not points:
            return
        canvas = self._require_canvas()
        canvas.saveState()
        canvas.setStrokeColor(color_to_reportlab(stroke))
        canvas.setLineWidth(stroke_width)
        path = canvas.beginPath()
        first_x, first_y = points[0]
        path.moveTo(first_x, self.height - first_y)
        for x, y in points[1:]:
            path.lineTo(x, self.height - y)
        canvas.drawPath(path, fill=0, stroke=1)
        canvas.restoreState()

    def polygon(
        self,
        points: list[tuple[float, float]],
        *,
        fill: str,
        stroke: str | None = None,
        stroke_width: float = 1.0,
        fill_alpha: float = 1.0,
        stroke_alpha: float = 1.0,
    ) -> None:
        if not points:
            return
        canvas = self._require_canvas()
        canvas.saveState()
        canvas.setFillColor(color_to_reportlab(fill, fill_alpha))
        self._with_alpha(fill_alpha=fill_alpha)
        if stroke is not None:
            canvas.setStrokeColor(color_to_reportlab(stroke, stroke_alpha))
            canvas.setLineWidth(stroke_width)
            self._with_alpha(stroke_alpha=stroke_alpha)
        else:
            canvas.setStrokeColor(Color(1, 1, 1, alpha=0))
        path = canvas.beginPath()
        first_x, first_y = points[0]
        path.moveTo(first_x, self.height - first_y)
        for x, y in points[1:]:
            path.lineTo(x, self.height - y)
        path.close()
        canvas.drawPath(path, fill=1, stroke=int(stroke is not None))
        canvas.restoreState()

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: float = 11.0,
        fill: str = "#111111",
        anchor: str = "start",
        valign: str = "top",
        bold: bool = False,
    ) -> None:
        canvas = self._require_canvas()
        font_name = "Times-Bold" if bold else "Times-Roman"
        width = stringWidth(value, font_name, size)
        if anchor == "middle":
            x = x - width / 2.0
        elif anchor == "end":
            x = x - width

        baseline_y = self.height - y
        if valign == "top":
            baseline_y = baseline_y - size
        elif valign == "middle":
            baseline_y = baseline_y - size * 0.35

        canvas.saveState()
        canvas.setFont(font_name, size)
        canvas.setFillColor(color_to_reportlab(fill))
        canvas.drawString(x, baseline_y, value)
        canvas.restoreState()

    def save(self, path: Path) -> None:
        canvas = self._require_canvas()
        canvas.showPage()
        canvas.save()


class PngCanvas(DrawingCanvas):
    def __init__(self, width: float, height: float, scale: float) -> None:
        self.width = width
        self.height = height
        self.scale = scale
        pixel_size = (int(round(width * scale)), int(round(height * scale)))
        self.image = Image.new("RGBA", pixel_size, (255, 255, 255, 255))
        self.draw = ImageDraw.Draw(self.image, "RGBA")
        self._font_cache: dict[tuple[int, bool], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

    def _sx(self, value: float) -> float:
        return value * self.scale

    def _font(self, size: float, bold: bool) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        key = (int(round(size * self.scale)), bold)
        if key in self._font_cache:
            return self._font_cache[key]

        search_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
        ]
        for path in search_paths:
            if Path(path).exists():
                font = ImageFont.truetype(path, key[0])
                self._font_cache[key] = font
                return font

        font = ImageFont.load_default()
        self._font_cache[key] = font
        return font

    def rectangle(
        self,
        rect: Rect,
        *,
        fill: str | None = None,
        stroke: str | None = None,
        stroke_width: float = 1.0,
        fill_alpha: float = 1.0,
        stroke_alpha: float = 1.0,
    ) -> None:
        xy = [
            self._sx(rect.x),
            self._sx(rect.y),
            self._sx(rect.x + rect.w),
            self._sx(rect.y + rect.h),
        ]
        fill_rgba = hex_with_alpha(fill, fill_alpha) if fill is not None else None
        outline_rgba = hex_with_alpha(stroke, stroke_alpha) if stroke is not None else None
        self.draw.rectangle(
            xy,
            fill=fill_rgba,
            outline=outline_rgba,
            width=max(1, int(round(stroke_width * self.scale))),
        )

    def _draw_dashed_line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: tuple[int, int, int, int],
        stroke_width: int,
        dash: tuple[float, float],
    ) -> None:
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length == 0:
            return
        dash_len = max(1.0, dash[0] * self.scale)
        gap_len = max(1.0, dash[1] * self.scale)
        step = dash_len + gap_len
        ux = dx / length
        uy = dy / length
        cursor = 0.0
        while cursor < length:
            start_x = x1 + ux * cursor
            start_y = y1 + uy * cursor
            end_cursor = min(length, cursor + dash_len)
            end_x = x1 + ux * end_cursor
            end_y = y1 + uy * end_cursor
            self.draw.line((start_x, start_y, end_x, end_y), fill=stroke, width=stroke_width)
            cursor += step

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str,
        stroke_width: float = 1.0,
        stroke_alpha: float = 1.0,
        dash: tuple[float, float] | None = None,
    ) -> None:
        rgba = hex_with_alpha(stroke, stroke_alpha)
        width = max(1, int(round(stroke_width * self.scale)))
        sx1, sy1, sx2, sy2 = self._sx(x1), self._sx(y1), self._sx(x2), self._sx(y2)
        if dash is None:
            self.draw.line((sx1, sy1, sx2, sy2), fill=rgba, width=width)
        else:
            self._draw_dashed_line(sx1, sy1, sx2, sy2, stroke=rgba, stroke_width=width, dash=dash)

    def polyline(
        self,
        points: list[tuple[float, float]],
        *,
        stroke: str,
        stroke_width: float = 1.0,
    ) -> None:
        if not points:
            return
        encoded = [(self._sx(x), self._sx(y)) for x, y in points]
        self.draw.line(
            encoded,
            fill=hex_with_alpha(stroke, 1.0),
            width=max(1, int(round(stroke_width * self.scale))),
            joint="curve",
        )

    def polygon(
        self,
        points: list[tuple[float, float]],
        *,
        fill: str,
        stroke: str | None = None,
        stroke_width: float = 1.0,
        fill_alpha: float = 1.0,
        stroke_alpha: float = 1.0,
    ) -> None:
        if not points:
            return
        encoded = [(self._sx(x), self._sx(y)) for x, y in points]
        self.draw.polygon(encoded, fill=hex_with_alpha(fill, fill_alpha))
        if stroke is not None:
            self.draw.line(
                encoded + [encoded[0]],
                fill=hex_with_alpha(stroke, stroke_alpha),
                width=max(1, int(round(stroke_width * self.scale))),
            )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: float = 11.0,
        fill: str = "#111111",
        anchor: str = "start",
        valign: str = "top",
        bold: bool = False,
    ) -> None:
        font = self._font(size, bold)
        bbox = self.draw.textbbox((0, 0), value, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        sx = self._sx(x)
        sy = self._sx(y)
        if anchor == "middle":
            sx -= width / 2.0
        elif anchor == "end":
            sx -= width

        if valign == "middle":
            sy -= height / 2.0
        elif valign == "bottom":
            sy -= height

        self.draw.text((sx, sy), value, fill=hex_with_alpha(fill, 1.0), font=font)

    def save(self, path: Path) -> None:
        self.image.save(path, dpi=(600, 600))


def create_canvas(path: Path, width: float, height: float) -> DrawingCanvas:
    suffix = path.suffix.lower()
    if suffix == ".svg":
        return SvgCanvas(width, height)
    if suffix == ".pdf":
        canvas = PdfCanvas(width, height)
        canvas.bind(path)
        return canvas
    if suffix == ".png":
        return PngCanvas(width, height, PNG_SCALE)
    raise ValueError(f"Unsupported output suffix: {suffix}")


def map_x(value: float, x_min: float, x_max: float, rect: Rect) -> float:
    if x_max <= x_min:
        return rect.x + rect.w / 2.0
    return rect.x + (value - x_min) / (x_max - x_min) * rect.w


def map_y(value: float, y_min: float, y_max: float, rect: Rect) -> float:
    if y_max <= y_min:
        return rect.y + rect.h / 2.0
    frac = (value - y_min) / (y_max - y_min)
    return rect.y + rect.h - frac * rect.h


def nice_time_ticks(n_samples: int, time_step: float, count: int = 5) -> list[tuple[float, str]]:
    if n_samples <= 1:
        return [(0.0, "0.00")]
    tick_positions = np.linspace(0, n_samples - 1, num=min(count, n_samples))
    ticks: list[tuple[float, str]] = []
    for pos in tick_positions:
        label = f"{pos * time_step:.2f}"
        ticks.append((float(pos), label))
    return ticks


def y_ticks_from_range(y_min: float, y_max: float, count: int = 5) -> list[float]:
    if y_max <= y_min:
        return [y_min]
    return [float(v) for v in np.linspace(y_min, y_max, num=count)]


def single_line_text_bounds(
    anchor_x: float,
    value: str,
    *,
    size: float,
    anchor: str = "start",
    bold: bool = False,
) -> tuple[float, float]:
    font_name = "Times-Bold" if bold else "Times-Roman"
    width = float(stringWidth(value, font_name, size))
    if anchor == "middle":
        return anchor_x - width / 2.0, anchor_x + width / 2.0
    if anchor == "end":
        return anchor_x - width, anchor_x
    return anchor_x, anchor_x + width


def multiline_text_bounds(
    anchor_x: float,
    value: str,
    *,
    size: float,
    anchor: str = "start",
    bold: bool = False,
) -> tuple[float, float]:
    bounds = [
        single_line_text_bounds(anchor_x, line, size=size, anchor=anchor, bold=bold)
        for line in value.split("\n")
    ]
    return min(left for left, _right in bounds), max(right for _left, right in bounds)


def main_figure_layout() -> dict[str, Rect | float]:
    left = 102.0
    right = 760.0
    top = 60.0
    gap = 20.0
    state_h = 48.0
    eco_h = 180.0
    exp_h = 120.0
    summary_h = 150.0
    colorbar_w = 14.0
    plot_w = right - left - colorbar_w - 28.0
    colorbar_x = left + plot_w + 16.0

    state_rect = Rect(left, top, plot_w, state_h)
    eco_rect = Rect(left, state_rect.bottom + gap, plot_w, eco_h)
    eco_cbar = Rect(colorbar_x, eco_rect.y, colorbar_w, eco_rect.h)
    exp_rect = Rect(left, eco_rect.bottom + gap, plot_w, exp_h)
    exp_cbar = Rect(colorbar_x, exp_rect.y, colorbar_w, exp_rect.h)
    summary_rect = Rect(left, exp_rect.bottom + gap, plot_w, summary_h)

    return {
        "state_rect": state_rect,
        "eco_rect": eco_rect,
        "eco_cbar": eco_cbar,
        "exp_rect": exp_rect,
        "exp_cbar": exp_cbar,
        "summary_rect": summary_rect,
        "row_label_x": eco_rect.x - HEATMAP_ROW_LABEL_OFFSET,
        "economic_group_label_x": eco_rect.x - HEATMAP_GROUP_LABEL_OFFSET,
        "expensive_group_label_x": exp_rect.x - HEATMAP_GROUP_LABEL_OFFSET,
    }


def draw_panel_label(canvas: DrawingCanvas, rect: Rect, label: str) -> None:
    canvas.text(rect.x - 26.0, rect.y - 12.0, label, size=14.0, bold=True)


def draw_plot_border(canvas: DrawingCanvas, rect: Rect) -> None:
    canvas.rectangle(rect, fill=None, stroke="#4a4a4a", stroke_width=0.8)


def draw_state_backgrounds(
    canvas: DrawingCanvas,
    rect: Rect,
    segments: list[tuple[int, int, int]],
    x_min: float,
    x_max: float,
) -> None:
    for label, start, end in segments:
        left = map_x(start - 0.5, x_min, x_max, rect)
        right = map_x(end - 0.5, x_min, x_max, rect)
        canvas.rectangle(
            Rect(left, rect.y, right - left, rect.h),
            fill=STATE_COLORS[label],
            stroke=None,
            fill_alpha=0.45,
        )


def draw_transition_lines(
    canvas: DrawingCanvas,
    rect: Rect,
    segments: list[tuple[int, int, int]],
    x_min: float,
    x_max: float,
) -> None:
    for _label, _start, end in segments[:-1]:
        boundary = map_x(end - 0.5, x_min, x_max, rect)
        canvas.line(
            boundary,
            rect.y,
            boundary,
            rect.bottom,
            stroke="#666666",
            stroke_width=0.9,
            stroke_alpha=0.9,
            dash=(4.0, 3.0),
        )


def draw_x_axis(
    canvas: DrawingCanvas,
    rect: Rect,
    x_min: float,
    x_max: float,
    ticks: list[tuple[float, str]],
    label: str | None = None,
) -> None:
    canvas.line(rect.x, rect.bottom, rect.right, rect.bottom, stroke="#333333", stroke_width=0.8)
    for position, text_value in ticks:
        x = map_x(position, x_min, x_max, rect)
        canvas.line(x, rect.bottom, x, rect.bottom + 4.0, stroke="#333333", stroke_width=0.8)
        canvas.text(x, rect.bottom + 8.0, text_value, size=9.0, anchor="middle")
    if label is not None:
        canvas.text(rect.x + rect.w / 2.0, rect.bottom + 26.0, label, size=10.5, anchor="middle")


def draw_y_axis(
    canvas: DrawingCanvas,
    rect: Rect,
    y_min: float,
    y_max: float,
    ticks: list[float],
    label: str | None = None,
    formatter=lambda value: f"{value:.1f}",
) -> None:
    canvas.line(rect.x, rect.y, rect.x, rect.bottom, stroke="#333333", stroke_width=0.8)
    for tick in ticks:
        y = map_y(tick, y_min, y_max, rect)
        canvas.line(rect.x - 4.0, y, rect.x, y, stroke="#333333", stroke_width=0.8)
        canvas.text(rect.x - 8.0, y, formatter(tick), size=8.5, anchor="end", valign="middle")
    if label is not None:
        lines = label.split("\n")
        base_y = rect.y + rect.h / 2.0 - (len(lines) - 1) * 8.0
        for idx, line in enumerate(lines):
            canvas.text(rect.x - 45.0, base_y + idx * 16.0, line, size=10.0, anchor="middle", valign="middle")


def draw_grid(
    canvas: DrawingCanvas,
    rect: Rect,
    y_min: float,
    y_max: float,
    ticks: list[float],
) -> None:
    for tick in ticks:
        y = map_y(tick, y_min, y_max, rect)
        canvas.line(rect.x, y, rect.right, y, stroke="#cccccc", stroke_width=0.6, stroke_alpha=0.65)


def draw_heatmap(
    canvas: DrawingCanvas,
    rect: Rect,
    values: np.ndarray,
    y_labels: list[str],
    *,
    row_label_x: float | None = None,
) -> None:
    n_rows, n_cols = values.shape
    cell_w = rect.w / max(n_cols, 1)
    cell_h = rect.h / max(n_rows, 1)
    label_x = rect.x - HEATMAP_ROW_LABEL_OFFSET if row_label_x is None else row_label_x
    for row in range(n_rows):
        for col in range(n_cols):
            fill = diverging_color(float(values[row, col]))
            x = rect.x + col * cell_w
            y = rect.y + row * cell_h
            canvas.rectangle(Rect(x, y, cell_w + 0.25, cell_h + 0.25), fill=fill, stroke=None)
    for row, label in enumerate(y_labels):
        y = rect.y + (row + 0.5) * cell_h
        canvas.text(label_x, y, label, size=8.5, anchor="end", valign="middle")
    draw_plot_border(canvas, rect)


def draw_colorbar(
    canvas: DrawingCanvas,
    rect: Rect,
    *,
    vmin: float = -3.0,
    vmax: float = 3.0,
) -> None:
    bins = 60
    step = rect.h / bins
    for idx in range(bins):
        value = vmax - idx / max(bins - 1, 1) * (vmax - vmin)
        fill = diverging_color(value, vmin=vmin, vmax=vmax)
        y = rect.y + idx * step
        canvas.rectangle(Rect(rect.x, y, rect.w, step + 0.3), fill=fill, stroke=None)
    draw_plot_border(canvas, rect)
    for value in (vmax, 0.0, vmin):
        y = map_y(value, vmin, vmax, rect)
        canvas.line(rect.right, y, rect.right + 4.0, y, stroke="#333333", stroke_width=0.7)
        canvas.text(rect.right + 7.0, y, f"{value:.0f}", size=8.0, valign="middle")
    canvas.text(rect.x + rect.w / 2.0, rect.bottom + 10.0, "z-score", size=8.5, anchor="middle")


def summary_envelopes(
    standardized: np.ndarray,
    feature_groups: dict[str, list[int]],
) -> dict[str, dict[str, np.ndarray]]:
    summary: dict[str, dict[str, np.ndarray]] = {}
    for group_name, indices in feature_groups.items():
        group_data = standardized[:, indices]
        summary[group_name] = {
            "median": np.median(group_data, axis=1),
            "lower": np.percentile(group_data, 25, axis=1),
            "upper": np.percentile(group_data, 75, axis=1),
        }
    return summary


def padded_range(values: np.ndarray, pad_ratio: float = 0.08) -> tuple[float, float]:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        delta = 1.0 if np.isclose(vmin, 0.0) else abs(vmin) * 0.1
        return vmin - delta, vmax + delta
    pad = (vmax - vmin) * pad_ratio
    return vmin - pad, vmax + pad


def draw_line_series(
    canvas: DrawingCanvas,
    rect: Rect,
    series: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    stroke: str,
    stroke_width: float = 1.7,
) -> None:
    points = [
        (map_x(float(idx), x_min, x_max, rect), map_y(float(value), y_min, y_max, rect))
        for idx, value in enumerate(series.tolist())
    ]
    canvas.polyline(points, stroke=stroke, stroke_width=stroke_width)


def draw_main_figure(
    canvas: DrawingCanvas,
    data: np.ndarray,
    labels: np.ndarray,
    feature_groups: dict[str, list[int]],
    seam_name: str,
    time_step: float,
) -> None:
    width, height = MAIN_SIZE
    standardized = zscore_per_feature(data)
    segments = compute_label_segments(labels)
    x_min = -0.5
    x_max = data.shape[0] - 0.5

    layout = main_figure_layout()
    state_rect = layout["state_rect"]
    eco_rect = layout["eco_rect"]
    eco_cbar = layout["eco_cbar"]
    exp_rect = layout["exp_rect"]
    exp_cbar = layout["exp_cbar"]
    summary_rect = layout["summary_rect"]
    row_label_x = float(layout["row_label_x"])
    economic_group_label_x = float(layout["economic_group_label_x"])
    expensive_group_label_x = float(layout["expensive_group_label_x"])

    canvas.text(width / 2.0, 24.0, f"Raw-signal evolution for seam {seam_name}", size=14.0, anchor="middle", bold=True)

    draw_panel_label(canvas, state_rect, "A")
    draw_state_backgrounds(canvas, state_rect, segments, x_min, x_max)
    draw_transition_lines(canvas, state_rect, segments, x_min, x_max)
    draw_plot_border(canvas, state_rect)
    canvas.text(state_rect.x - 45.0, state_rect.y + state_rect.h / 2.0, "State", size=10.0, anchor="middle", valign="middle")
    for label, start, end in segments:
        center_x = map_x((start + end - 1) / 2.0, x_min, x_max, state_rect)
        canvas.text(center_x, state_rect.y + state_rect.h / 2.0, LABEL_DISPLAY_NAMES[label], size=10.0, anchor="middle", valign="middle", bold=True)

    draw_panel_label(canvas, eco_rect, "B")
    draw_state_backgrounds(canvas, eco_rect, segments, x_min, x_max)
    draw_transition_lines(canvas, eco_rect, segments, x_min, x_max)
    eco_data = standardized[:, feature_groups["economic"]].T
    draw_heatmap(
        canvas,
        eco_rect,
        eco_data,
        [f"Eco-{idx + 1}" for idx in range(eco_data.shape[0])],
        row_label_x=row_label_x,
    )
    draw_colorbar(canvas, eco_cbar)
    canvas.text(economic_group_label_x, eco_rect.y + eco_rect.h / 2.0, "Economic\nfeatures", size=10.0, anchor="middle", valign="middle")

    draw_panel_label(canvas, exp_rect, "C")
    draw_state_backgrounds(canvas, exp_rect, segments, x_min, x_max)
    draw_transition_lines(canvas, exp_rect, segments, x_min, x_max)
    exp_data = standardized[:, feature_groups["expensive"]].T
    draw_heatmap(
        canvas,
        exp_rect,
        exp_data,
        [f"Exp-{idx + 1}" for idx in range(exp_data.shape[0])],
        row_label_x=row_label_x,
    )
    draw_colorbar(canvas, exp_cbar)
    canvas.text(expensive_group_label_x, exp_rect.y + exp_rect.h / 2.0, "Expensive\nfeatures", size=10.0, anchor="middle", valign="middle")

    draw_panel_label(canvas, summary_rect, "D")
    draw_state_backgrounds(canvas, summary_rect, segments, x_min, x_max)
    draw_transition_lines(canvas, summary_rect, segments, x_min, x_max)

    envelopes = summary_envelopes(standardized, feature_groups)
    stacked = np.concatenate(
        [
            envelopes["economic"]["lower"],
            envelopes["economic"]["upper"],
            envelopes["expensive"]["lower"],
            envelopes["expensive"]["upper"],
        ]
    )
    y_min, y_max = padded_range(stacked, pad_ratio=0.1)
    ticks = y_ticks_from_range(y_min, y_max)
    draw_grid(canvas, summary_rect, y_min, y_max, ticks)
    draw_y_axis(canvas, summary_rect, y_min, y_max, ticks, label="Grouped\nz-score")

    for group_name in ("economic", "expensive"):
        lower = envelopes[group_name]["lower"]
        upper = envelopes[group_name]["upper"]
        median = envelopes[group_name]["median"]
        top_points = [
            (map_x(float(idx), x_min, x_max, summary_rect), map_y(float(value), y_min, y_max, summary_rect))
            for idx, value in enumerate(upper.tolist())
        ]
        bottom_points = [
            (map_x(float(idx), x_min, x_max, summary_rect), map_y(float(value), y_min, y_max, summary_rect))
            for idx, value in reversed(list(enumerate(lower.tolist())))
        ]
        canvas.polygon(
            top_points + bottom_points,
            fill=GROUP_STYLE[group_name]["fill"],
            fill_alpha=0.6,
        )
        draw_line_series(
            canvas,
            summary_rect,
            median,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            stroke=GROUP_STYLE[group_name]["line"],
            stroke_width=2.0,
        )

    draw_plot_border(canvas, summary_rect)
    draw_x_axis(canvas, summary_rect, x_min, x_max, nice_time_ticks(data.shape[0], time_step), label="Time (s)")

    legend_x = summary_rect.x + 10.0
    legend_y = summary_rect.y + 10.0
    for idx, group_name in enumerate(("economic", "expensive")):
        y = legend_y + idx * 17.0
        canvas.rectangle(
            Rect(legend_x, y + 3.0, 12.0, 8.0),
            fill=GROUP_STYLE[group_name]["fill"],
            stroke="#666666",
            stroke_width=0.6,
            fill_alpha=0.8,
        )
        canvas.line(legend_x, y + 7.0, legend_x + 12.0, y + 7.0, stroke=GROUP_STYLE[group_name]["line"], stroke_width=1.8)
        canvas.text(legend_x + 18.0, y, f"{group_name.title()} median/IQR", size=9.0)


def draw_representative_figure(
    canvas: DrawingCanvas,
    data: np.ndarray,
    labels: np.ndarray,
    representative_channels: dict[str, list[int]],
    seam_name: str,
    time_step: float,
) -> None:
    width, _height = SUPP_SIZE
    segments = compute_label_segments(labels)
    x_min = -0.5
    x_max = data.shape[0] - 0.5
    ticks = nice_time_ticks(data.shape[0], time_step)

    left = 82.0
    top = 64.0
    plot_w = 290.0
    plot_h = 170.0
    gap_x = 85.0
    gap_y = 70.0

    canvas.text(width / 2.0, 24.0, f"Representative raw channels for seam {seam_name}", size=14.0, anchor="middle", bold=True)

    panels = [
        ("economic", representative_channels["economic"][0], Rect(left, top, plot_w, plot_h), "A"),
        ("economic", representative_channels["economic"][1], Rect(left + plot_w + gap_x, top, plot_w, plot_h), "B"),
        ("expensive", representative_channels["expensive"][0], Rect(left, top + plot_h + gap_y, plot_w, plot_h), "C"),
        ("expensive", representative_channels["expensive"][1], Rect(left + plot_w + gap_x, top + plot_h + gap_y, plot_w, plot_h), "D"),
    ]

    for group_name, column_idx, rect, label in panels:
        draw_panel_label(canvas, rect, label)
        draw_state_backgrounds(canvas, rect, segments, x_min, x_max)
        draw_transition_lines(canvas, rect, segments, x_min, x_max)

        y_min, y_max = padded_range(data[:, column_idx], pad_ratio=0.1)
        yticks = y_ticks_from_range(y_min, y_max, count=4)
        draw_grid(canvas, rect, y_min, y_max, yticks)
        draw_y_axis(canvas, rect, y_min, y_max, yticks, label="Raw value", formatter=lambda value: f"{value:.0f}" if abs(value) >= 10 else f"{value:.2f}")
        draw_line_series(
            canvas,
            rect,
            data[:, column_idx],
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            stroke=GROUP_STYLE[group_name]["line"],
            stroke_width=1.8,
        )
        draw_plot_border(canvas, rect)
        draw_x_axis(canvas, rect, x_min, x_max, ticks, label="Time (s)")
        canvas.text(rect.x + rect.w / 2.0, rect.y - 20.0, f"{group_name.title()} (col {column_idx})", size=11.0, anchor="middle", bold=True)


def render_all_formats(
    output_base: Path,
    *,
    logical_size: tuple[float, float],
    drawer,
) -> list[Path]:
    outputs: list[Path] = []
    for suffix in (".pdf", ".svg", ".png"):
        path = output_base.with_suffix(suffix)
        canvas = create_canvas(path, logical_size[0], logical_size[1])
        drawer(canvas)
        canvas.save(path)
        outputs.append(path)
    return outputs


def generate_paper_figures(
    input_dir: Path,
    seam_filename: str,
    output_dir: Path,
    time_step: float,
    expensive_indices: list[int],
) -> list[Path]:
    seam_path = input_dir / seam_filename
    if not seam_path.exists():
        raise FileNotFoundError(f"Seam file not found: {seam_path}")

    data, labels, _raw_labels = load_raw_seam_csv(seam_path)
    feature_groups = build_feature_groups(data.shape[1], expensive_indices)
    representative_channels = select_representative_channels(data, labels, feature_groups, top_k=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    seam_stem = seam_path.stem

    outputs: list[Path] = []
    outputs.extend(
        render_all_formats(
            output_dir / f"{seam_stem}_raw_group_main",
            logical_size=MAIN_SIZE,
            drawer=lambda canvas: draw_main_figure(
                canvas=canvas,
                data=data,
                labels=labels,
                feature_groups=feature_groups,
                seam_name=seam_stem,
                time_step=time_step,
            ),
        )
    )
    outputs.extend(
        render_all_formats(
            output_dir / f"{seam_stem}_raw_representative_channels",
            logical_size=SUPP_SIZE,
            drawer=lambda canvas: draw_representative_figure(
                canvas=canvas,
                data=data,
                labels=labels,
                representative_channels=representative_channels,
                seam_name=seam_stem,
                time_step=time_step,
            ),
        )
    )
    return outputs


def main() -> None:
    args = parse_args()
    expensive_indices = parse_feature_indices(args.expensive_indices, total_features=18)
    generated = generate_paper_figures(
        input_dir=args.input_dir,
        seam_filename=args.seam,
        output_dir=args.output_dir,
        time_step=args.time_step,
        expensive_indices=expensive_indices,
    )
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
