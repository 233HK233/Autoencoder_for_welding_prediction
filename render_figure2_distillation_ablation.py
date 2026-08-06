#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
CLASS_IDS = (0, 1, 2)
CLASS_NAMES = ("S0", "S1", "S2")

TEST_METRICS_PATTERN = re.compile(
    r"---\s*Test Metrics.*?Accuracy:\s*(?P<accuracy>[0-9.]+)%.*?Macro-F1:\s*(?P<macro_f1>[0-9.]+)",
    flags=re.DOTALL,
)
TEST_AGREEMENT_PATTERN = re.compile(r"Test Teacher-Agreement:\s*(?P<agreement>[0-9.]+)%")
CLASS_REPORT_PATTERN = re.compile(
    r"=== Classification Report \((?:Student )?Test.*?\) ===(?P<body>.*?)(?:\n\s*accuracy|\Z)",
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

METHOD_ORDER = (
    "Full SEAL-Weld",
    "w/o attention",
    "w/o logit KD",
    "w/o latent alignment",
    "Latent alignment only",
    "Student-only",
    "LSTM KD",
    "Joint teacher-student",
)
METHOD_COLORS = {
    "Full SEAL-Weld": "#0072B2",
    "w/o attention": "#CC79A7",
    "w/o logit KD": "#E69F00",
    "w/o latent alignment": "#56B4E9",
    "Latent alignment only": "#F0E442",
    "Student-only": "#D55E00",
    "LSTM KD": "#009E73",
    "Joint teacher-student": "#8A63B8",
}
LINE_STYLES = {
    "Full SEAL-Weld": "-",
    "Student-only": "--",
    "Joint teacher-student": "-.",
}


@dataclass(frozen=True)
class MethodSpec:
    method: str
    role: str
    run_dir: Path | None
    protocol_note: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Figure 2: distillation components and training strategy ablation."
    )
    parser.add_argument(
        "--matched-root",
        type=Path,
        default=Path("ablation_experiments/h1/results/figure2_matched_distillation_ablation/seed14"),
    )
    parser.add_argument(
        "--student-sweep-csv",
        type=Path,
        default=Path(
            "ablation_experiments/h1/reports/comparison_18d_baselines/"
            "teacher_student_reference/student_h1_sweep/student_h1_scan96_20260326/"
            "distill_sweep_progress.csv"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/paper_figures"))
    parser.add_argument("--output-name", type=str, default="figure2_distillation_ablation")
    return parser.parse_args(argv)


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.1,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.75,
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


def _unit_fraction(value: float) -> float:
    return value / 100.0 if value > 1.0 else value


def parse_test_metrics(metrics_path: Path) -> dict[str, float]:
    text = metrics_path.read_text(encoding="utf-8", errors="ignore")
    metrics_match = TEST_METRICS_PATTERN.search(text)
    if metrics_match is None:
        raise ValueError(f"Could not parse test Accuracy/Macro-F1 from {metrics_path}")

    report_match = CLASS_REPORT_PATTERN.search(text)
    if report_match is None:
        raise ValueError(f"Could not parse test classification report from {metrics_path}")

    class_recalls: dict[int, float] = {}
    for row_match in CLASS_ROW_PATTERN.finditer(report_match.group("body")):
        class_recalls[int(row_match.group("class_id"))] = float(row_match.group("recall"))
    missing = set(CLASS_IDS).difference(class_recalls)
    if missing:
        raise ValueError(f"Missing class rows {sorted(missing)} in {metrics_path}")

    agreement_match = TEST_AGREEMENT_PATTERN.search(text)
    agreement = math.nan
    if agreement_match is not None:
        agreement = float(agreement_match.group("agreement")) / 100.0

    return {
        "accuracy": float(metrics_match.group("accuracy")) / 100.0,
        "macro_f1": _unit_fraction(float(metrics_match.group("macro_f1"))),
        "s1_recall": _unit_fraction(class_recalls[1]),
        "teacher_agreement": agreement,
    }


def severe_cross_stage_error(prediction_path: Path) -> float:
    with prediction_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction table is empty: {prediction_path}")
    severe = sum(1 for row in rows if abs(int(row["y_true"]) - int(row["y_pred"])) >= 2)
    return severe / len(rows)


def build_component_ablation_metrics(specs: Iterable[MethodSpec]) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for spec in specs:
        run_dir = resolve_repo_path(spec.run_dir) if spec.run_dir is not None else None
        metrics_path = run_dir / "evaluation_metrics.txt" if run_dir is not None else None
        prediction_path = run_dir / "test_predictions.csv" if run_dir is not None else None
        if run_dir is None or metrics_path is None or not metrics_path.exists():
            notes.append(f"{spec.method}: N/A because evaluation_metrics.txt is missing.")
            rows.append(
                {
                    "method": spec.method,
                    "role": spec.role,
                    "run_dir": "" if run_dir is None else str(run_dir),
                    "protocol_note": spec.protocol_note,
                    "macro_f1": math.nan,
                    "macro_f1_pct": math.nan,
                    "s1_recall": math.nan,
                    "s1_recall_pct": math.nan,
                    "severe_cross_stage_error": math.nan,
                    "severe_cross_stage_error_pct": math.nan,
                    "teacher_agreement": math.nan,
                    "available": False,
                }
            )
            continue

        parsed = parse_test_metrics(metrics_path)
        severe = math.nan
        if prediction_path is not None and prediction_path.exists():
            severe = severe_cross_stage_error(prediction_path)
        else:
            notes.append(f"{spec.method}: severe cross-stage error N/A because test_predictions.csv is missing.")
        rows.append(
            {
                "method": spec.method,
                "role": spec.role,
                "run_dir": str(run_dir),
                "protocol_note": spec.protocol_note,
                "macro_f1": parsed["macro_f1"],
                "macro_f1_pct": parsed["macro_f1"] * 100.0,
                "s1_recall": parsed["s1_recall"],
                "s1_recall_pct": parsed["s1_recall"] * 100.0,
                "severe_cross_stage_error": severe,
                "severe_cross_stage_error_pct": severe * 100.0 if not math.isnan(severe) else math.nan,
                "teacher_agreement": parsed["teacher_agreement"],
                "available": True,
            }
        )

    df = pd.DataFrame(rows)
    if "method" in df.columns:
        df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
        df = df.sort_values("method").reset_index(drop=True)
        df["method"] = df["method"].astype(str)
    return df, notes


def _value_to_pct(value: float) -> float:
    return _unit_fraction(float(value)) * 100.0


def _grid_value(value: Any) -> float:
    return round(float(value), 6)


def build_heatmap_source(
    sweep_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    value_col: str,
    value_name: str,
) -> pd.DataFrame:
    required = {x_col, y_col, value_col}
    missing = required.difference(sweep_df.columns)
    if missing:
        raise ValueError(f"Missing columns for heatmap source: {sorted(missing)}")

    rows: list[dict[str, float]] = []
    for row in sweep_df.to_dict(orient="records"):
        raw_value = row.get(value_col)
        if raw_value is None or pd.isna(raw_value):
            continue
        raw_value = float(raw_value)
        if raw_value < 0:
            continue
        rows.append({"x": _grid_value(row[x_col]), "y": _grid_value(row[y_col]), "value_pct": _value_to_pct(raw_value)})

    if not rows:
        return pd.DataFrame(columns=["x", "y", "value_pct", f"{value_name}_pct", "n_runs"])

    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["x", "y"], dropna=False)
        .agg(value_pct=("value_pct", "mean"), n_runs=("value_pct", "size"))
        .reset_index()
        .sort_values(["y", "x"])
        .reset_index(drop=True)
    )
    grouped[f"{value_name}_pct"] = grouped["value_pct"]
    return grouped


def build_s1_recall_heatmap_source(sweep_df: pd.DataFrame, *, x_col: str, y_col: str) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for row in sweep_df.to_dict(orient="records"):
        if float(row.get("test_macro_f1", -1.0)) < 0:
            continue
        run_dir_value = row.get("run_dir")
        if not run_dir_value or pd.isna(run_dir_value):
            continue
        metrics_path = resolve_repo_path(str(run_dir_value)) / "evaluation_metrics.txt"
        if not metrics_path.exists():
            continue
        parsed = parse_test_metrics(metrics_path)
        rows.append({"x": _grid_value(row[x_col]), "y": _grid_value(row[y_col]), "value_pct": parsed["s1_recall"] * 100.0})
    if not rows:
        return pd.DataFrame(columns=["x", "y", "value_pct", "s1_recall_pct", "n_runs"])
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["x", "y"], dropna=False)
        .agg(value_pct=("value_pct", "mean"), n_runs=("value_pct", "size"))
        .reset_index()
        .sort_values(["y", "x"])
        .reset_index(drop=True)
    )
    grouped["s1_recall_pct"] = grouped["value_pct"]
    return grouped


def build_training_dynamics(specs: Iterable[MethodSpec]) -> pd.DataFrame:
    keep = {"Full SEAL-Weld", "Student-only", "Joint teacher-student"}
    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.method not in keep or spec.run_dir is None:
            continue
        history_path = resolve_repo_path(spec.run_dir) / "history.json"
        if not history_path.exists():
            continue
        history = json.loads(history_path.read_text(encoding="utf-8"))
        for item in history:
            val_f1 = item.get("val_macro_f1", item.get("val_student_macro_f1", math.nan))
            teacher_agreement = item.get("val_teacher_agreement", math.nan)
            rows.append(
                {
                    "method": spec.method,
                    "epoch": int(item["epoch"]),
                    "ce_loss": float(
                        item.get("train_ce_loss", item.get("train_loss", item.get("train_student_ce_loss", math.nan)))
                    ),
                    "kd_loss": float(item.get("train_kd_loss", math.nan)),
                    "latent_mse": float(item.get("train_feat_loss", math.nan)),
                    "val_macro_f1_pct": _value_to_pct(float(val_f1)) if not pd.isna(val_f1) else math.nan,
                    "teacher_agreement_pct": _value_to_pct(float(teacher_agreement))
                    if not pd.isna(teacher_agreement)
                    else math.nan,
                }
            )
    return pd.DataFrame(rows)


def find_run_dir(matched_root: Path, variant: str) -> Path | None:
    variant_dir = resolve_repo_path(matched_root) / variant
    if not variant_dir.exists():
        return None
    candidates = sorted([path for path in variant_dir.iterdir() if path.is_dir()])
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def default_method_specs(matched_root: Path) -> list[MethodSpec]:
    return [
        MethodSpec("Full SEAL-Weld", "full", find_run_dir(matched_root, "full_seal_weld"), "matched seed14"),
        MethodSpec("w/o attention", "component", find_run_dir(matched_root, "wo_attention"), "matched seed14"),
        MethodSpec("w/o logit KD", "component", find_run_dir(matched_root, "wo_logit_kd"), "matched seed14"),
        MethodSpec(
            "w/o latent alignment",
            "component",
            find_run_dir(matched_root, "wo_latent_alignment"),
            "matched seed14",
        ),
        MethodSpec(
            "Latent alignment only",
            "component",
            find_run_dir(matched_root, "latent_alignment_only"),
            "matched seed14; same CE+latent protocol as w/o logit KD",
        ),
        MethodSpec("Student-only", "baseline", find_run_dir(matched_root, "student_only_anchor"), "anchor rerun"),
        MethodSpec("LSTM KD", "baseline", find_run_dir(matched_root, "lstm_kd_anchor"), "anchor rerun"),
        MethodSpec(
            "Joint teacher-student",
            "strategy",
            find_run_dir(matched_root, "joint_teacher_student"),
            "aligned joint-training seed14",
        ),
    ]


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#111827",
    )


def _wrapped_labels(labels: Iterable[str]) -> list[str]:
    return ["\n".join(textwrap.wrap(str(label), width=13)) for label in labels]


def plot_component_metric(ax: plt.Axes, component_df: pd.DataFrame, value_col: str, title: str, ylabel: str) -> None:
    labels = component_df["method"].astype(str).tolist()
    values = component_df[value_col].astype(float).to_numpy()
    x = np.arange(len(labels))
    colors = [METHOD_COLORS.get(label, "#9CA3AF") for label in labels]
    bars = ax.bar(x, np.nan_to_num(values, nan=0.0), color=colors, edgecolor="white", linewidth=0.7)
    finite_values = values[np.isfinite(values)]
    if value_col == "severe_cross_stage_error_pct":
        upper = max(1.0, float(np.nanmax(finite_values)) + 0.25) if len(finite_values) else 1.0
        label_offset = upper * 0.035
    else:
        upper = 104.0
        label_offset = 1.0
    for bar, label, value in zip(bars, labels, values):
        if math.isnan(value):
            bar.set_color("#D1D5DB")
            ax.text(bar.get_x() + bar.get_width() / 2, label_offset, "N/A", ha="center", va="bottom", fontsize=6.2)
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + label_offset,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=5.9,
                color="#111827",
            )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(_wrapped_labels(labels), rotation=45, ha="right")
    ax.set_ylim(0, upper)
    ax.grid(axis="y", color="#CBD5E1", alpha=0.6, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_heatmap(ax: plt.Axes, table: pd.DataFrame, title: str, x_label: str, y_label: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if table.empty:
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center", color="#6B7280")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    xs = sorted(table["x"].dropna().unique().tolist())
    ys = sorted(table["y"].dropna().unique().tolist())
    matrix = np.full((len(ys), len(xs)), np.nan)
    for row in table.itertuples(index=False):
        matrix[ys.index(float(row.y)), xs.index(float(row.x))] = float(row.value_pct)
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis", vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    ax.set_xticks(np.arange(len(xs)))
    ax.set_yticks(np.arange(len(ys)))
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_yticklabels([f"{y:g}" for y in ys])
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            value = matrix[yi, xi]
            if not np.isnan(value):
                ax.text(xi, yi, f"{value:.1f}", ha="center", va="center", fontsize=6.0, color="white")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.set_ylabel("%", rotation=270, labelpad=9)


def plot_ce_loss(ax: plt.Axes, dynamics_df: pd.DataFrame) -> None:
    ax.set_title("Training CE loss")
    for method in ("Full SEAL-Weld", "Student-only", "Joint teacher-student"):
        sub = dynamics_df.loc[dynamics_df["method"] == method].sort_values("epoch")
        if sub.empty:
            continue
        ax.plot(
            sub["epoch"],
            sub["ce_loss"],
            label=method,
            color=METHOD_COLORS[method],
            linestyle=LINE_STYLES[method],
            linewidth=1.4,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CE-equivalent loss")
    ax.grid(color="#CBD5E1", alpha=0.55, linewidth=0.5)
    ax.legend(loc="best", frameon=False)


def plot_kd_latent(ax: plt.Axes, dynamics_df: pd.DataFrame) -> None:
    ax.set_title("KD and latent-MSE loss")
    for method in ("Full SEAL-Weld", "Joint teacher-student"):
        sub = dynamics_df.loc[dynamics_df["method"] == method].sort_values("epoch")
        if sub.empty:
            continue
        ax.plot(sub["epoch"], sub["kd_loss"], color=METHOD_COLORS[method], linestyle="-", linewidth=1.3, label=f"{method} KD")
        ax.plot(
            sub["epoch"],
            sub["latent_mse"],
            color=METHOD_COLORS[method],
            linestyle=":",
            linewidth=1.5,
            label=f"{method} latent",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(color="#CBD5E1", alpha=0.55, linewidth=0.5)
    ax.legend(loc="best", frameon=False, ncol=1)


def plot_validation(ax: plt.Axes, dynamics_df: pd.DataFrame) -> None:
    ax.set_title("Validation Macro-F1 and agreement")
    for method in ("Full SEAL-Weld", "Student-only", "Joint teacher-student"):
        sub = dynamics_df.loc[dynamics_df["method"] == method].sort_values("epoch")
        if sub.empty:
            continue
        ax.plot(
            sub["epoch"],
            sub["val_macro_f1_pct"],
            color=METHOD_COLORS[method],
            linestyle=LINE_STYLES[method],
            linewidth=1.4,
            label=f"{method} F1",
        )
        if sub["teacher_agreement_pct"].notna().any():
            ax.plot(
                sub["epoch"],
                sub["teacher_agreement_pct"],
                color=METHOD_COLORS[method],
                linestyle=":",
                linewidth=1.2,
                label=f"{method} agree",
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("%")
    ax.set_ylim(0, 104)
    ax.grid(color="#CBD5E1", alpha=0.55, linewidth=0.5)
    ax.legend(loc="lower right", frameon=False, ncol=1)


def render(
    *,
    component_df: pd.DataFrame,
    heatmap_macro: pd.DataFrame,
    heatmap_s1: pd.DataFrame,
    heatmap_agree: pd.DataFrame,
    dynamics_df: pd.DataFrame,
    output_base: Path,
) -> None:
    configure_style()
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10.5), constrained_layout=True)

    plot_component_metric(axes[0, 0], component_df, "macro_f1_pct", "Test Macro-F1", "Macro-F1 (%)")
    plot_component_metric(axes[0, 1], component_df, "s1_recall_pct", "Transition-state S1 recall", "S1 recall (%)")
    plot_component_metric(
        axes[0, 2],
        component_df,
        "severe_cross_stage_error_pct",
        "Severe cross-stage error",
        "S0 <-> S2 error (%)",
    )

    plot_heatmap(axes[1, 0], heatmap_macro, "Temperature x lambda_kd", "Temperature", "lambda_kd")
    plot_heatmap(axes[1, 1], heatmap_s1, "lambda_kd x lambda_feat", "lambda_kd", "lambda_feat")
    plot_heatmap(axes[1, 2], heatmap_agree, "Temperature x lambda_feat", "Temperature", "lambda_feat")

    plot_ce_loss(axes[2, 0], dynamics_df)
    plot_kd_latent(axes[2, 1], dynamics_df)
    plot_validation(axes[2, 2], dynamics_df)

    for label, ax in zip("abcdefghi", axes.ravel()):
        add_panel_label(ax, label)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_source_tables(
    *,
    out_dir: Path,
    component_df: pd.DataFrame,
    heatmap_macro: pd.DataFrame,
    heatmap_s1: pd.DataFrame,
    heatmap_agree: pd.DataFrame,
    dynamics_df: pd.DataFrame,
) -> None:
    source_dir = out_dir / "source_tables"
    source_dir.mkdir(parents=True, exist_ok=True)
    component_df.to_csv(source_dir / "component_ablation_metrics.csv", index=False)
    heatmap_macro.to_csv(source_dir / "hyperparameter_response_macro_f1.csv", index=False)
    heatmap_s1.to_csv(source_dir / "hyperparameter_response_s1_recall.csv", index=False)
    heatmap_agree.to_csv(source_dir / "hyperparameter_response_teacher_agreement.csv", index=False)
    dynamics_df.to_csv(source_dir / "training_dynamics.csv", index=False)


def write_qa_notes(out_dir: Path, notes: list[str], specs: Iterable[MethodSpec]) -> None:
    lines = [
        "Figure 2 distillation ablation QA notes",
        "",
        "Figure contract:",
        "- Core conclusion: full frozen-teacher SEAL-Weld is compared against matched component removals, baselines, and joint training.",
        "- Evidence chain: row 1 final test metrics, row 2 hyperparameter response surfaces, row 3 training dynamics.",
        "- Archetype: quantitative grid.",
        "- Backend/export: Python Matplotlib, PDF/SVG vector plus PNG preview.",
        "",
        "Run sources:",
    ]
    for spec in specs:
        lines.append(f"- {spec.method}: {spec.run_dir if spec.run_dir is not None else 'N/A'} ({spec.protocol_note})")
    lines.append("")
    lines.append("N/A and caveats:")
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- No row-1 N/A items: all plotted methods have evaluation_metrics.txt and test_predictions.csv.")
    lines.append("- Latent alignment only and w/o logit KD use the same CE + latent-alignment protocol by definition.")
    (out_dir / "qa_notes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    specs = default_method_specs(args.matched_root)
    component_df, notes = build_component_ablation_metrics(specs)

    sweep_df = pd.read_csv(resolve_repo_path(args.student_sweep_csv))
    heatmap_macro = build_heatmap_source(
        sweep_df,
        x_col="temperature",
        y_col="lambda_kd",
        value_col="test_macro_f1",
        value_name="macro_f1",
    )
    heatmap_s1 = build_s1_recall_heatmap_source(sweep_df, x_col="lambda_kd", y_col="lambda_feat")
    heatmap_agree = build_heatmap_source(
        sweep_df,
        x_col="temperature",
        y_col="lambda_feat",
        value_col="test_teacher_agreement",
        value_name="teacher_agreement",
    )
    dynamics_df = build_training_dynamics(specs)

    out_dir = prepare_output_dir(resolve_repo_path(args.output_root), args.output_name)
    write_source_tables(
        out_dir=out_dir,
        component_df=component_df,
        heatmap_macro=heatmap_macro,
        heatmap_s1=heatmap_s1,
        heatmap_agree=heatmap_agree,
        dynamics_df=dynamics_df,
    )
    write_qa_notes(out_dir, notes, specs)
    render(
        component_df=component_df,
        heatmap_macro=heatmap_macro,
        heatmap_s1=heatmap_s1,
        heatmap_agree=heatmap_agree,
        dynamics_df=dynamics_df,
        output_base=out_dir / args.output_name,
    )
    print(f"Figure 2 written to: {out_dir}")


if __name__ == "__main__":
    main()
