#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from data_utils import load_seam_csv

CASE_TYPE_ORDER = ("S0-core", "0->1-boundary", "S1-core", "1->2-boundary")
CORE_CASE_TYPES = {"S0-core", "S1-core"}
BOUNDARY_CASE_TYPES = {"0->1-boundary", "1->2-boundary"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select Figure 1 panel candidates from detailed predictions.")
    parser.add_argument("--predictions-csv", type=Path, required=True, help="Detailed predictions CSV.")
    parser.add_argument("--raw-data-dir", type=Path, default=Path("Data/raw_data"), help="Directory containing seam CSV files.")
    parser.add_argument("--output-enriched-csv", type=Path, required=True, help="Output CSV with derived columns.")
    parser.add_argument("--output-panel-csv", type=Path, required=True, help="Output CSV with selected panel candidates.")
    return parser.parse_args(argv)


def load_seam_label_sequences(raw_data_dir: Path, seam_names: Iterable[str]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for seam_name in sorted(set(seam_names)):
        _, labels = load_seam_csv(raw_data_dir / f"{seam_name}.csv")
        result[seam_name] = labels.astype(np.int64)
    return result


def compute_transition_boundaries(labels: np.ndarray) -> tuple[int, int]:
    changes = np.where(labels[:-1] != labels[1:])[0]
    if len(changes) != 2:
        raise ValueError(f"Expected exactly 2 transitions in seam labels, got {len(changes)}")
    return int(changes[0] + 1), int(changes[1] + 1)


def classify_case_type(y_true: int, target_idx: int, boundary_01: int, boundary_12: int) -> str:
    dist_01 = abs(target_idx - boundary_01)
    dist_12 = abs(target_idx - boundary_12)
    nearest = min(dist_01, dist_12)
    if y_true == 0:
        return "0->1-boundary" if nearest <= 5 else "S0-core"
    if y_true == 1:
        return "1->2-boundary" if dist_12 <= 5 and dist_12 <= dist_01 else "S1-core"
    if y_true == 2:
        return "1->2-boundary" if nearest <= 5 else "S1-core"
    raise ValueError(f"Unsupported class label: {y_true}")


def classify_error_type(y_true: int, final_pred: int) -> str:
    if y_true == final_pred:
        return "correct"
    if abs(int(y_true) - int(final_pred)) == 1:
        return "adjacent_error"
    return "cross_stage_error"


def classify_boundary_distance_bin(distance: int) -> str:
    if distance <= 5:
        return "Near"
    if distance <= 15:
        return "Mid"
    return "Far"


def annotate_samples(df: pd.DataFrame, seam_labels: dict[str, np.ndarray]) -> pd.DataFrame:
    annotated = df.copy()
    nearest_dist: list[int] = []
    distance_bin: list[str] = []
    case_type: list[str] = []
    error_type: list[str] = []
    confidence: list[float] = []

    for _, row in annotated.iterrows():
        seam_name = str(row["seam_name"])
        labels = seam_labels[seam_name]
        boundary_01, boundary_12 = compute_transition_boundaries(labels)
        target_idx = int(row["target_idx"])
        true_cls = int(row["y_true"])
        pred_cls = int(row["final_pred"])

        dist = min(abs(target_idx - boundary_01), abs(target_idx - boundary_12))
        nearest_dist.append(dist)
        distance_bin.append(classify_boundary_distance_bin(dist))
        case_type.append(classify_case_type(true_cls, target_idx, boundary_01, boundary_12))
        error_type.append(classify_error_type(true_cls, pred_cls))

        confidence.append(float(row[f"prob_class_{pred_cls}"]))

    annotated["nearest_boundary_dist"] = nearest_dist
    annotated["boundary_distance_bin"] = distance_bin
    annotated["case_type"] = case_type
    annotated["error_type"] = error_type
    annotated["confidence"] = confidence
    return annotated


def _priority_score(row: pd.Series) -> tuple[int, float, int]:
    correct_priority = 0 if row["error_type"] == "correct" else 1
    return (correct_priority, -float(row["confidence"]), int(row["sample_index"]))


def select_panel_candidates(annotated: pd.DataFrame) -> pd.DataFrame:
    selected_rows: list[dict[str, object]] = []

    for seam_name in sorted(annotated["seam_name"].unique().tolist()):
        seam_df = annotated.loc[annotated["seam_name"] == seam_name].copy()
        for case_type in CASE_TYPE_ORDER:
            subset = seam_df.loc[seam_df["case_type"] == case_type].copy()
            if subset.empty:
                selected_rows.append(
                    {
                        "seam_name": seam_name,
                        "case_type": case_type,
                        "panel_key": f"{seam_name}|{case_type}",
                        "panel_status": "missing",
                    }
                )
                continue

            if case_type in CORE_CASE_TYPES:
                subset = subset.sort_values(
                    by=["boundary_distance_bin", "nearest_boundary_dist", "confidence", "sample_index"],
                    ascending=[False, False, False, True],
                )
            else:
                subset = subset.sort_values(
                    by=["error_type", "confidence", "sample_index"],
                    key=lambda col: col.map({"correct": 1, "adjacent_error": 0, "cross_stage_error": 0})
                    if col.name == "error_type"
                    else col,
                    ascending=[True, False, True],
                )

            best_row = min(subset.to_dict("records"), key=lambda rec: _priority_score(pd.Series(rec)))
            best_row["panel_key"] = f"{seam_name}|{case_type}"
            best_row["panel_status"] = "selected"
            selected_rows.append(best_row)

    if not selected_rows:
        return annotated.iloc[0:0].copy()

    result = pd.DataFrame(selected_rows).copy()
    result = result.sort_values(by=["seam_name", "case_type", "sample_index"]).reset_index(drop=True)
    return result


def apply_cross_seam_backfill(panel_df: pd.DataFrame) -> pd.DataFrame:
    filled = panel_df.copy()
    if "source_seam_name" not in filled.columns:
        filled["source_seam_name"] = filled["seam_name"]

    global_candidates = filled.loc[filled["panel_status"] == "selected"].copy()

    for idx, row in filled.loc[filled["panel_status"] == "missing"].iterrows():
        case_type = str(row["case_type"])
        candidates = filled.loc[
            (filled["case_type"] == case_type) & (filled["panel_status"] == "selected")
        ].copy()
        if candidates.empty:
            candidates = global_candidates.copy()
        if candidates.empty:
            continue
        candidates = candidates.sort_values(
            by=["confidence", "nearest_boundary_dist", "sample_index"],
            ascending=[False, False, True],
        )
        donor = candidates.iloc[0]
        for col in filled.columns:
            if col in {"panel_key", "seam_name", "case_type"}:
                continue
            filled.at[idx, col] = donor.get(col)
        filled.at[idx, "source_seam_name"] = donor.get("seam_name")
        filled.at[idx, "panel_status"] = "backfilled"

    return filled


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pred_df = pd.read_csv(args.predictions_csv)
    seam_labels = load_seam_label_sequences(args.raw_data_dir, pred_df["seam_name"].astype(str).tolist())
    annotated = annotate_samples(pred_df, seam_labels)
    selected = apply_cross_seam_backfill(select_panel_candidates(annotated))

    args.output_enriched_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_panel_csv.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(args.output_enriched_csv, index=False)
    selected.to_csv(args.output_panel_csv, index=False)

    print(f"Saved enriched predictions to: {args.output_enriched_csv}")
    print(f"Saved Figure 1 panel candidates to: {args.output_panel_csv}")
    print(f"Selected panels: {len(selected)}")


if __name__ == "__main__":
    main()
