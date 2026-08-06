#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METHOD_ORDER = (
    "Teacher-18D",
    "SEAL-Weld Student-13D",
    "Student-only-13D",
    "TCN-13D",
    "LSTM",
    "GRU",
    "Transformer",
    "Inception",
)
CLASS_NAMES = ("S0", "S1", "S2")
CLASS_IDS = (0, 1, 2)
SEAL_STUDENT_LABEL = "SEAL-Weld Student-13D"
STUDENT_ONLY_LABEL = "Student-only-13D"
TEACHER_LABEL = "Teacher-18D"
MISSING_TCN_LABEL = "TCN-13D"

COMPARISON_GROUP_TO_METHOD = {
    "teacher-student(student)": SEAL_STUDENT_LABEL,
    "lstm": "LSTM",
    "gru": "GRU",
    "transformer": "Transformer",
    "inception": "Inception",
}
PREDICTION_FILES = {
    TEACHER_LABEL: "teacher_18d_upper_bound_test_predictions.csv",
    SEAL_STUDENT_LABEL: "teacher_student_student_test_predictions.csv",
    "LSTM": "lstm_test_predictions.csv",
    "GRU": "gru_test_predictions.csv",
    "Transformer": "transformer_test_predictions.csv",
    "Inception": "inception_test_predictions.csv",
}
METHOD_COLORS = {
    TEACHER_LABEL: "#30323D",
    SEAL_STUDENT_LABEL: "#0B5CAD",
    STUDENT_ONLY_LABEL: "#D6811F",
    MISSING_TCN_LABEL: "#B8C0CC",
    "LSTM": "#5CA370",
    "GRU": "#8A63B8",
    "Transformer": "#C36B4F",
    "Inception": "#5499C7",
}
METHOD_MARKERS = {
    TEACHER_LABEL: "o",
    SEAL_STUDENT_LABEL: "s",
    STUDENT_ONLY_LABEL: "^",
    MISSING_TCN_LABEL: "x",
    "LSTM": "D",
    "GRU": "P",
    "Transformer": "v",
    "Inception": "X",
}
DISPLAY_LABELS = {
    TEACHER_LABEL: "Teacher\n18D",
    SEAL_STUDENT_LABEL: "SEAL-Weld\nStudent 13D",
    STUDENT_ONLY_LABEL: "Student-only\n13D",
    MISSING_TCN_LABEL: "TCN\n13D",
    "LSTM": "LSTM",
    "GRU": "GRU",
    "Transformer": "Transformer",
    "Inception": "Inception",
}

TEST_METRICS_PATTERN = re.compile(
    r"--- Test Metrics.*?Accuracy:\s*(?P<accuracy>[0-9.]+)%.*?Macro-F1:\s*(?P<macro_f1>[0-9.]+)",
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
SEED_SUFFIX_PATTERN = re.compile(r"_seed\d+$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Figure 1: overall prediction performance and error structure."
    )
    parser.add_argument(
        "--teacher-summary-csv",
        type=Path,
        default=Path("outputs/teacher_h1_scan98_gpu1_v2/analysis/teacher_h1_seed_sweep_summary.csv"),
    )
    parser.add_argument(
        "--comparison-runs-csv",
        type=Path,
        default=Path("ablation_experiments/h1/reports/comparison_18d_baselines/checkpoints/round3/comparison_runs.csv"),
    )
    parser.add_argument(
        "--student-only-sweep-csv",
        type=Path,
        default=Path("ablation_experiments/h1/reports/run_20260327_131013/ablation1_sweep/ablation1_sweep_progress.csv"),
        help=(
            "Student-only 13D sweep table. The script keeps one best run per seed because "
            "no fixed-config seed sweep exists in the current repository."
        ),
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=Path("outputs/roc_figure3/predictions"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/paper_figures"))
    parser.add_argument("--output-name", type=str, default="figure1_performance_error_structure")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    parser.add_argument("--title", type=str, default="")
    return parser.parse_args(argv)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.8,
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


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_test_metrics(metrics_text: str) -> dict[str, Any]:
    metrics_match = TEST_METRICS_PATTERN.search(metrics_text)
    if metrics_match is None:
        raise ValueError("Could not parse test Accuracy/Macro-F1 from evaluation_metrics.txt")

    report_match = CLASS_REPORT_PATTERN.search(metrics_text)
    if report_match is None:
        raise ValueError("Could not parse Test classification report block from evaluation_metrics.txt")

    class_rows: list[dict[str, float | int | str]] = []
    for row_match in CLASS_ROW_PATTERN.finditer(report_match.group("block")):
        class_id = int(row_match.group("class_id"))
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": f"S{class_id}",
                "precision": float(row_match.group("precision")),
                "recall": float(row_match.group("recall")),
                "f1": float(row_match.group("f1")),
                "support": int(row_match.group("support")),
            }
        )

    if sorted(int(row["class_id"]) for row in class_rows) != list(CLASS_IDS):
        raise ValueError("Expected test classification report rows for Class 0, Class 1, and Class 2")

    class_by_id = {int(row["class_id"]): row for row in class_rows}
    balanced_accuracy = float(np.mean([float(class_by_id[class_id]["recall"]) for class_id in CLASS_IDS]))
    return {
        "accuracy": float(metrics_match.group("accuracy")) / 100.0,
        "macro_f1": float(metrics_match.group("macro_f1")),
        "balanced_accuracy": balanced_accuracy,
        "s1_recall": float(class_by_id[1]["recall"]),
        "class_rows": class_rows,
    }


def read_run_metrics(run_path: str | Path) -> dict[str, Any]:
    metrics_path = resolve_repo_path(run_path) / "evaluation_metrics.txt"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing evaluation metrics file: {metrics_path}")
    return parse_test_metrics(metrics_path.read_text(encoding="utf-8"))


def config_signature_without_seed(value: str) -> str:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("config_signature must decode to a JSON object")
    payload.pop("seed", None)
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def run_name_without_seed(value: str) -> str:
    return SEED_SUFFIX_PATTERN.sub("_seed*", str(value))


def select_representative_config_group(df: pd.DataFrame, config_col: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if df.empty:
        raise ValueError("Cannot select a representative config from an empty dataframe")
    required = {config_col, "seed", "macro_f1"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns for config selection: {sorted(missing)}")

    summary = (
        df.groupby(config_col, dropna=False)
        .agg(
            n_seeds=("seed", lambda values: int(pd.Series(values).nunique())),
            n_runs=("seed", "size"),
            mean_macro_f1=("macro_f1", "mean"),
            max_macro_f1=("macro_f1", "max"),
        )
        .reset_index()
        .sort_values(
            ["n_seeds", "mean_macro_f1", "max_macro_f1", config_col],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
    )
    selected_config = summary.iloc[0][config_col]
    selected_df = df.loc[df[config_col] == selected_config].copy()
    selected_df = selected_df.sort_values("seed").reset_index(drop=True)
    selected_summary = {
        "selected_config_id": selected_config,
        "n_seeds": int(summary.iloc[0]["n_seeds"]),
        "n_runs": int(summary.iloc[0]["n_runs"]),
        "mean_macro_f1": float(summary.iloc[0]["mean_macro_f1"]),
        "max_macro_f1": float(summary.iloc[0]["max_macro_f1"]),
    }
    return selected_df, selected_summary


def build_teacher_seed_metrics(summary_csv: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = pd.read_csv(resolve_repo_path(summary_csv))
    rows: list[dict[str, Any]] = []
    for source_row in source.itertuples(index=False):
        if str(getattr(source_row, "status", "success")) != "success":
            continue
        metrics = read_run_metrics(getattr(source_row, "run_dir"))
        rows.append(
            {
                "method": TEACHER_LABEL,
                "seed": int(getattr(source_row, "seed")),
                "run_path": str(getattr(source_row, "run_dir")),
                "config_id": config_signature_without_seed(getattr(source_row, "config_signature")),
                "input_dim": 18,
                "train_scheme": "teacher_only",
                "provenance": "fixed_config_seed_sweep",
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "s1_recall": metrics["s1_recall"],
            }
        )
    metrics_df = pd.DataFrame(rows)
    selected, summary = select_representative_config_group(metrics_df, "config_id")
    summary.update(
        {
            "method": TEACHER_LABEL,
            "source": str(summary_csv),
            "status": "available",
            "selection_rule": "fixed config with most seeds, then highest mean Macro-F1",
        }
    )
    return selected, summary


def build_comparison_group_metrics(
    runs_csv: Path,
    *,
    group: str,
    method: str,
    input_dim: int,
    train_scheme: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = pd.read_csv(resolve_repo_path(runs_csv))
    source = source.loc[source["group"] == group].copy()
    if source.empty:
        raise ValueError(f"No comparison rows found for group={group}")
    source["config_id"] = source["run_name"].map(run_name_without_seed)

    rows: list[dict[str, Any]] = []
    for source_row in source.itertuples(index=False):
        metrics = read_run_metrics(getattr(source_row, "run_path"))
        rows.append(
            {
                "method": method,
                "seed": int(getattr(source_row, "seed")),
                "run_path": str(getattr(source_row, "run_path")),
                "config_id": str(getattr(source_row, "config_id")),
                "input_dim": input_dim,
                "train_scheme": train_scheme,
                "provenance": "fixed_config_seed_group",
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "s1_recall": metrics["s1_recall"],
                "teacher_agreement": float(getattr(source_row, "teacher_agreement")),
            }
        )
    metrics_df = pd.DataFrame(rows)
    selected, summary = select_representative_config_group(metrics_df, "config_id")
    summary.update(
        {
            "method": method,
            "source": str(runs_csv),
            "status": "available",
            "selection_rule": "fixed config with most seeds, then highest mean Macro-F1",
        }
    )
    return selected, summary


def build_student_only_sweep_metrics(sweep_csv: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    source = pd.read_csv(resolve_repo_path(sweep_csv))
    source = source.loc[source["status"].astype(str).eq("success")].copy()
    rows: list[dict[str, Any]] = []
    for source_row in source.itertuples(index=False):
        run_path = str(getattr(source_row, "run_dir"))
        metrics = read_run_metrics(run_path)
        rows.append(
            {
                "method": STUDENT_ONLY_LABEL,
                "seed": int(getattr(source_row, "seed")),
                "run_path": run_path,
                "config_id": run_name_without_seed(Path(run_path).name),
                "input_dim": 13,
                "train_scheme": "student_only",
                "provenance": "sweep_best_per_seed_not_fixed_config",
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "s1_recall": metrics["s1_recall"],
                "trial_id": int(getattr(source_row, "trial_id")),
                "stage": str(getattr(source_row, "stage")),
            }
        )

    all_runs = pd.DataFrame(rows)
    if all_runs.empty:
        raise ValueError(f"No successful Student-only rows found in {sweep_csv}")
    per_seed = (
        all_runs.sort_values(["macro_f1", "accuracy", "trial_id"], ascending=[False, False, True], kind="mergesort")
        .drop_duplicates("seed", keep="first")
        .sort_values("seed")
        .reset_index(drop=True)
    )
    reference = per_seed.sort_values(["macro_f1", "accuracy"], ascending=[False, False], kind="mergesort").iloc[0]
    summary = {
        "method": STUDENT_ONLY_LABEL,
        "source": str(sweep_csv),
        "status": "available_with_caveat",
        "selection_rule": "one best Macro-F1 run per seed from hyperparameter sweep; not a fixed-config seed sweep",
        "selected_config_id": "sweep_best_per_seed",
        "n_seeds": int(per_seed["seed"].nunique()),
        "n_runs": int(len(per_seed)),
        "mean_macro_f1": float(per_seed["macro_f1"].mean()),
        "max_macro_f1": float(per_seed["macro_f1"].max()),
    }
    return per_seed, summary, reference.to_dict()


def build_multiseed_metrics(
    teacher_summary_csv: Path,
    comparison_runs_csv: Path,
    student_only_sweep_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metric_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []

    teacher_df, teacher_summary = build_teacher_seed_metrics(teacher_summary_csv)
    metric_frames.append(teacher_df)
    selection_rows.append(teacher_summary)

    student_df, student_summary = build_comparison_group_metrics(
        comparison_runs_csv,
        group="teacher-student(student)",
        method=SEAL_STUDENT_LABEL,
        input_dim=13,
        train_scheme="frozen_teacher_distill",
    )
    metric_frames.append(student_df)
    selection_rows.append(student_summary)

    student_only_df, student_only_summary, student_only_reference = build_student_only_sweep_metrics(
        student_only_sweep_csv
    )
    metric_frames.append(student_only_df)
    selection_rows.append(student_only_summary)

    selection_rows.append(
        {
            "method": MISSING_TCN_LABEL,
            "source": "",
            "status": "missing",
            "selection_rule": "independent fixed-config TCN-13D baseline not found",
            "selected_config_id": "",
            "n_seeds": 0,
            "n_runs": 0,
            "mean_macro_f1": math.nan,
            "max_macro_f1": math.nan,
        }
    )

    for group, method in (
        ("lstm", "LSTM"),
        ("gru", "GRU"),
        ("transformer", "Transformer"),
        ("inception", "Inception"),
    ):
        baseline_df, baseline_summary = build_comparison_group_metrics(
            comparison_runs_csv,
            group=group,
            method=method,
            input_dim=18,
            train_scheme="baseline_18d",
        )
        baseline_summary["status"] = "available_with_caveat"
        baseline_summary["selection_rule"] += "; current table labels this baseline as 18D"
        metric_frames.append(baseline_df)
        selection_rows.append(baseline_summary)

    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics["method"] = pd.Categorical(metrics["method"], categories=METHOD_ORDER, ordered=True)
    metrics = metrics.sort_values(["method", "seed"]).reset_index(drop=True)
    selection_df = pd.DataFrame(selection_rows)
    selection_df["method"] = pd.Categorical(selection_df["method"], categories=METHOD_ORDER, ordered=True)
    selection_df = selection_df.sort_values("method").reset_index(drop=True)
    return metrics, selection_df, student_only_reference


def compute_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true.astype(np.int64), y_pred.astype(np.int64)):
        if 0 <= truth < num_classes and 0 <= pred < num_classes:
            counts[int(truth), int(pred)] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts, dtype=np.float64),
        where=row_sums != 0,
    )
    return counts, normalized


def compute_class_metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, Any]]:
    counts, _ = compute_confusion_matrices(y_true, y_pred, num_classes=3)
    rows: list[dict[str, Any]] = []
    for class_id in CLASS_IDS:
        tp = float(counts[class_id, class_id])
        predicted = float(counts[:, class_id].sum())
        support = float(counts[class_id, :].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append(
            {
                "class_id": class_id,
                "class_name": f"S{class_id}",
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support),
            }
        )
    return rows


def load_prediction_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(resolve_repo_path(csv_path))
    required = {"sample_index", "y_true", "y_pred"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Prediction CSV missing required columns {sorted(missing)}: {csv_path}")
    return df


def write_prediction_csv(
    csv_path: Path,
    identities: Mapping[str, np.ndarray],
    y_pred: np.ndarray,
    logits: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "sample_index",
        "seam_id",
        "seam_name",
        "start_idx",
        "target_idx",
        "y_true",
        "y_pred",
        "logit_class_0",
        "logit_class_1",
        "logit_class_2",
        "prob_class_0",
        "prob_class_1",
        "prob_class_2",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for idx in range(y_pred.shape[0]):
            row = [
                int(identities["sample_index"][idx]),
                int(identities["seam_id"][idx]),
                str(identities["seam_name"][idx]),
                int(identities["start_idx"][idx]),
                int(identities["target_idx"][idx]),
                int(identities["y_true"][idx]),
                int(y_pred[idx]),
            ]
            row.extend(f"{float(value):.8f}" for value in logits[idx])
            row.extend(f"{float(value):.8f}" for value in probabilities[idx])
            writer.writerow(row)


def resolve_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def export_student_only_predictions(
    run_path: str | Path,
    output_csv: Path,
    *,
    device_arg: str,
) -> Path:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from models_tcn import AttentionTCNClassifier
    from training_utils import parse_channels, validate_forecast_dataset_contract

    run_dir = resolve_repo_path(run_path)
    run_args_path = run_dir / "run_args.json"
    if not run_args_path.exists():
        raise FileNotFoundError(f"Missing Student-only run_args.json: {run_args_path}")
    run_args = json.loads(run_args_path.read_text(encoding="utf-8"))

    dataset_npz = resolve_repo_path(run_args.get("dataset_npz", "Data/processed_data/weld_seam_windows_ws5_tf75_pg0_h1.npz"))
    validate_forecast_dataset_contract(dataset_npz, expected_horizon=1, expected_delta=5)
    with np.load(dataset_npz, allow_pickle=False) as payload:
        dataset = {key: payload[key] for key in payload.files}

    x_test = np.asarray(dataset["X_test_full"], dtype=np.float32)
    y_test = np.asarray(dataset["y_test"], dtype=np.int64)
    keep_indices = run_args.get("keep_feature_indices")
    if keep_indices is None:
        drop_indices = parse_drop_feature_indices(str(run_args.get("drop_feature_indices", "3,4,5,6,7")), x_test.shape[2])
        keep_indices = [idx for idx in range(x_test.shape[2]) if idx not in set(drop_indices)]
    keep_indices = [int(idx) for idx in keep_indices]
    x_test = x_test[:, :, keep_indices]
    if x_test.shape[2] != 13:
        raise ValueError(f"Student-only prediction export expected 13D input, got {x_test.shape[2]}D")

    tcn_layers = int(run_args.get("tcn_layers", 3))
    channels = parse_channels(run_args.get("tcn_channels"), latent_dim=80, min_layers=tcn_layers)
    num_classes = int(np.max(y_test) + 1)
    model = AttentionTCNClassifier(
        input_dim=int(x_test.shape[2]),
        num_classes=num_classes,
        channels=int(channels[-1]),
        tcn_layers=tcn_layers,
        tcn_kernel=int(run_args.get("tcn_kernel", 3)),
        tcn_dropout=float(run_args.get("tcn_dropout", 0.12)),
        dilation_base=int(run_args.get("tcn_dilation_base", 2)),
        attn_heads=int(run_args.get("attn_heads", 4)),
        attn_dropout=float(run_args.get("attn_dropout", 0.1)),
        ff_dim=int(run_args.get("attn_ff_dim", 128)),
        classifier_hidden=int(run_args.get("classifier_hidden", 128)),
        classifier_dropout=float(run_args.get("classifier_dropout", 0.35)),
    )

    checkpoint_path = run_dir / "best_student_only.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing Student-only checkpoint: {checkpoint_path}")
    device = resolve_device(device_arg)
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload.get("model_state_dict") if isinstance(payload, Mapping) and "model_state_dict" in payload else payload
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Unsupported checkpoint payload: {checkpoint_path}")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    loader = DataLoader(
        TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )
    logits_batches: list[np.ndarray] = []
    prob_batches: list[np.ndarray] = []
    pred_batches: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch, _ in loader:
            logits, _ = model(x_batch.to(device))
            probabilities = torch.softmax(logits, dim=1)
            logits_batches.append(logits.cpu().numpy())
            prob_batches.append(probabilities.cpu().numpy())
            pred_batches.append(torch.argmax(logits, dim=1).cpu().numpy())

    logits_np = np.concatenate(logits_batches, axis=0)
    probabilities_np = np.concatenate(prob_batches, axis=0)
    y_pred_np = np.concatenate(pred_batches, axis=0)
    seam_names = np.asarray(dataset["seam_name_order"])
    seam_ids = np.asarray(dataset["seam_id_test"], dtype=np.int64)
    identities = {
        "sample_index": np.arange(y_test.shape[0], dtype=np.int64),
        "seam_id": seam_ids,
        "seam_name": seam_names[seam_ids].astype(str),
        "start_idx": np.asarray(dataset["start_idx_test"], dtype=np.int64),
        "target_idx": np.asarray(dataset["target_idx_test"], dtype=np.int64),
        "y_true": y_test,
    }
    write_prediction_csv(output_csv, identities, y_pred_np, logits_np, probabilities_np)
    return output_csv


def parse_drop_feature_indices(value: str, input_dim: int) -> list[int]:
    if not value.strip():
        return []
    drop = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    for idx in drop:
        if idx < 0 or idx >= input_dim:
            raise ValueError(f"drop feature index {idx} out of range for input_dim={input_dim}")
    return drop


def build_confusion_and_profile_sources(
    prediction_dir: Path,
    output_dir: Path,
    student_only_reference: Mapping[str, Any],
    *,
    device_arg: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], str]:
    prediction_tables: dict[str, pd.DataFrame] = {}
    source_rows: list[dict[str, Any]] = []

    for method, filename in PREDICTION_FILES.items():
        csv_path = resolve_repo_path(prediction_dir) / filename
        if csv_path.exists():
            prediction_tables[method] = load_prediction_table(csv_path)
            source_rows.append({"method": method, "prediction_csv": str(csv_path), "status": "loaded"})

    student_only_csv = output_dir / "source_predictions" / "student_only_13d_test_predictions.csv"
    export_student_only_predictions(student_only_reference["run_path"], student_only_csv, device_arg=device_arg)
    prediction_tables[STUDENT_ONLY_LABEL] = load_prediction_table(student_only_csv)
    source_rows.append(
        {
            "method": STUDENT_ONLY_LABEL,
            "prediction_csv": str(student_only_csv),
            "status": "generated_from_best_sweep_checkpoint",
        }
    )

    profile_rows: list[dict[str, Any]] = []
    for method, pred_df in prediction_tables.items():
        y_true = pred_df["y_true"].to_numpy(dtype=np.int64)
        y_pred = pred_df["y_pred"].to_numpy(dtype=np.int64)
        macro_f1 = float(np.mean([row["f1"] for row in compute_class_metrics_from_predictions(y_true, y_pred)]))
        for row in compute_class_metrics_from_predictions(y_true, y_pred):
            profile_rows.append(
                {
                    "method": method,
                    "class_id": row["class_id"],
                    "class_name": row["class_name"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                    "support": row["support"],
                    "prediction_macro_f1": macro_f1,
                }
            )

    profile_df = pd.DataFrame(profile_rows)
    strongest_available_baseline = (
        profile_df.loc[profile_df["method"].isin(["LSTM", "GRU", "Transformer", "Inception"])]
        .groupby("method")["prediction_macro_f1"]
        .first()
        .sort_values(ascending=False)
        .index[0]
    )
    prediction_source_df = pd.DataFrame(source_rows)
    return profile_df, prediction_source_df, prediction_tables, strongest_available_baseline


def build_confusion_source_rows(
    prediction_tables: Mapping[str, pd.DataFrame],
    methods: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in methods:
        table = prediction_tables[method]
        counts, normalized = compute_confusion_matrices(
            table["y_true"].to_numpy(dtype=np.int64),
            table["y_pred"].to_numpy(dtype=np.int64),
            num_classes=3,
        )
        for row_idx, true_name in enumerate(CLASS_NAMES):
            for col_idx, pred_name in enumerate(CLASS_NAMES):
                rows.append(
                    {
                        "method": method,
                        "true_class": true_name,
                        "pred_class": pred_name,
                        "count": int(counts[row_idx, col_idx]),
                        "row_normalized": float(normalized[row_idx, col_idx]),
                    }
                )
    return pd.DataFrame(rows)


def build_paired_student_delta(metrics: pd.DataFrame) -> pd.DataFrame:
    subset = metrics.loc[metrics["method"].isin([SEAL_STUDENT_LABEL, STUDENT_ONLY_LABEL]), ["method", "seed", "macro_f1"]]
    pivot = subset.pivot_table(index="seed", columns="method", values="macro_f1", aggfunc="first", observed=False)
    required = [SEAL_STUDENT_LABEL, STUDENT_ONLY_LABEL]
    for col in required:
        if col not in pivot.columns:
            return pd.DataFrame(columns=["seed", "student_macro_f1", "student_only_macro_f1", "macro_f1_delta"])
    paired = pivot.dropna(subset=required).copy()
    paired["macro_f1_delta"] = paired[SEAL_STUDENT_LABEL] - paired[STUDENT_ONLY_LABEL]
    paired = paired.reset_index().rename(
        columns={
            SEAL_STUDENT_LABEL: "student_macro_f1",
            STUDENT_ONLY_LABEL: "student_only_macro_f1",
        }
    )
    return paired[["seed", "student_macro_f1", "student_only_macro_f1", "macro_f1_delta"]].sort_values("seed")


def compute_teacher_retention_pct(metrics: pd.DataFrame) -> float:
    teacher_mean = float(metrics.loc[metrics["method"] == TEACHER_LABEL, "macro_f1"].mean())
    student_mean = float(metrics.loc[metrics["method"] == SEAL_STUDENT_LABEL, "macro_f1"].mean())
    if teacher_mean == 0.0 or math.isnan(teacher_mean) or math.isnan(student_mean):
        return math.nan
    return student_mean / teacher_mean * 100.0


def build_teacher_agreement_scatter(comparison_runs_csv: Path, selected_student_config: str) -> pd.DataFrame:
    source = pd.read_csv(resolve_repo_path(comparison_runs_csv))
    source = source.loc[source["group"] == "teacher-student(student)"].copy()
    source = source.loc[source["teacher_agreement"].astype(float) >= 0.0].copy()
    source["config_id"] = source["run_name"].map(run_name_without_seed)
    source["selected_fixed_config"] = source["config_id"].astype(str).eq(str(selected_student_config))
    return source[
        [
            "run_name",
            "run_path",
            "seed",
            "macro_f1",
            "teacher_agreement",
            "config_id",
            "selected_fixed_config",
        ]
    ].copy()


def metric_ylim(values: pd.Series, lower_floor: float = 60.0) -> tuple[float, float]:
    clean = values.dropna().astype(float) * 100.0
    if clean.empty:
        return lower_floor, 101.0
    y_min = max(lower_floor, math.floor((float(clean.min()) - 2.0) / 5.0) * 5.0)
    y_max = min(101.0, math.ceil((float(clean.max()) + 1.0) / 5.0) * 5.0)
    if y_max - y_min < 10.0:
        y_min = max(lower_floor, y_max - 10.0)
    return y_min, y_max


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")
    ax.tick_params(colors="#334155", length=2.5, width=0.7)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.06,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color="#111827",
    )


def plot_distribution_panel(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    *,
    metric_col: str,
    ylabel: str,
    panel_label: str,
) -> None:
    rng = np.random.default_rng(20260620)
    positions = np.arange(1, len(METHOD_ORDER) + 1, dtype=float)
    for idx, method in enumerate(METHOD_ORDER):
        values = metrics.loc[metrics["method"] == method, metric_col].dropna().astype(float).to_numpy() * 100.0
        x_pos = positions[idx]
        color = METHOD_COLORS[method]
        if values.size >= 2:
            violin = ax.violinplot([values], positions=[x_pos], widths=0.72, showmeans=False, showmedians=False, showextrema=False)
            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor("none")
                body.set_alpha(0.18 if method != SEAL_STUDENT_LABEL else 0.28)
            q1, med, q3 = np.percentile(values, [25, 50, 75])
            ax.vlines(x_pos, q1, q3, color=color, linewidth=3.2, alpha=0.88, zorder=3)
            ax.scatter([x_pos], [med], s=18, color="white", edgecolor=color, linewidth=0.9, zorder=4)
        elif values.size == 0:
            ax.text(x_pos, 0.5, "no\ndata", transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=5.7, color="#64748B")

        if values.size:
            jitter = rng.uniform(-0.13, 0.13, size=values.size)
            ax.scatter(
                np.full(values.size, x_pos) + jitter,
                values,
                s=16 if method in {TEACHER_LABEL, SEAL_STUDENT_LABEL} else 13,
                marker=METHOD_MARKERS[method],
                color=color,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.78,
                zorder=5,
            )

    y_min, y_max = metric_ylim(metrics[metric_col])
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.4, len(METHOD_ORDER) + 0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([DISPLAY_LABELS[name] for name in METHOD_ORDER], rotation=35, ha="right")
    ax.grid(axis="y", color="#CBD5E1", alpha=0.55, linewidth=0.55)
    ax.set_axisbelow(True)
    style_axis(ax)
    add_panel_label(ax, panel_label)


def plot_confusion_panel(
    ax: plt.Axes,
    counts: np.ndarray,
    normalized: np.ndarray,
    *,
    title: str,
    panel_label: str,
    show_ylabel: bool = False,
    baseline_normalized: np.ndarray | None = None,
) -> Any:
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(CLASS_NAMES if show_ylabel else [])
    ax.set_xlabel("Predicted")
    if show_ylabel:
        ax.set_ylabel("True")
    ax.set_title(title, pad=5)
    serious_cells = [(0, 2), (2, 0)]
    s1_miss_cell = (1, 0)
    for row in range(3):
        for col in range(3):
            value = normalized[row, col]
            text_color = "white" if Normalize(0.0, 1.0)(value) > 0.55 else "#0F172A"
            ax.text(col, row - 0.08, f"{value * 100:.1f}", ha="center", va="center", fontsize=6.4, color=text_color)
            ax.text(col, row + 0.20, f"n={counts[row, col]}", ha="center", va="center", fontsize=5.2, color=text_color, alpha=0.88)
            if baseline_normalized is not None and row != col:
                delta = float(baseline_normalized[row, col] - normalized[row, col])
                if delta > 0.004:
                    ax.text(col, row + 0.40, f"-{delta * 100:.1f}pp", ha="center", va="center", fontsize=5.0, color="#137333")
    for row, col in serious_cells:
        ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="#B42318", linewidth=1.25))
    row, col = s1_miss_cell
    ax.add_patch(Rectangle((col - 0.44, row - 0.44), 0.88, 0.88, fill=False, edgecolor="#D6811F", linewidth=1.15, linestyle="--"))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    add_panel_label(ax, panel_label)
    return image


def plot_class_profile_panel(
    ax: plt.Axes,
    profile_df: pd.DataFrame,
    *,
    methods: list[str],
    panel_label: str,
) -> None:
    x_labels: list[str] = []
    x_positions: list[int] = []
    x_idx = 0
    for class_name in CLASS_NAMES:
        for metric in ("P", "R", "F1"):
            x_labels.append(f"{class_name}\n{metric}")
            x_positions.append(x_idx)
            x_idx += 1

    metric_lookup = {"P": "precision", "R": "recall", "F1": "f1"}
    for method in methods:
        subset = profile_df.loc[profile_df["method"] == method].set_index("class_name")
        if subset.empty:
            continue
        values: list[float] = []
        for class_name in CLASS_NAMES:
            for metric in ("P", "R", "F1"):
                values.append(float(subset.loc[class_name, metric_lookup[metric]]) * 100.0)
        ax.plot(
            x_positions,
            values,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            linewidth=1.75 if method in {TEACHER_LABEL, SEAL_STUDENT_LABEL} else 1.15,
            markersize=3.5,
            alpha=0.95,
            label=method,
        )
    for boundary in (2.5, 5.5):
        ax.axvline(boundary, color="#CBD5E1", linewidth=0.7, linestyle="--", zorder=0)
    ax.axvspan(3 - 0.5, 6 - 0.5, color="#F8FAFC", zorder=0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(65.0, 101.0)
    ax.grid(axis="y", color="#CBD5E1", alpha=0.55, linewidth=0.55)
    ax.legend(loc="lower left", frameon=False, ncol=1, fontsize=5.9, handlelength=1.4)
    ax.set_axisbelow(True)
    style_axis(ax)
    add_panel_label(ax, panel_label)


def plot_paired_delta_panel(ax: plt.Axes, paired_df: pd.DataFrame, *, panel_label: str) -> None:
    ax.axhline(0.0, color="#64748B", linewidth=0.8, linestyle="--", zorder=0)
    if paired_df.empty:
        ax.text(
            0.5,
            0.52,
            "No overlapping seeds\nbetween Student and Student-only",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.0,
            color="#475569",
        )
        ax.set_xticks([])
        ax.set_ylabel("Macro-F1 gain (pp)")
        ax.set_ylim(-5.0, 5.0)
    else:
        x = np.arange(len(paired_df), dtype=float)
        y = paired_df["macro_f1_delta"].astype(float).to_numpy() * 100.0
        colors = np.where(y >= 0.0, "#137333", "#B42318")
        ax.vlines(x, 0.0, y, color=colors, linewidth=2.0, alpha=0.72)
        ax.scatter(x, y, s=35, color=colors, edgecolor="white", linewidth=0.7, zorder=3)
        for xi, yi in zip(x, y):
            ax.text(xi, yi + (0.35 if yi >= 0 else -0.35), f"{yi:+.1f}", ha="center", va="bottom" if yi >= 0 else "top", fontsize=6.0)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(seed)) for seed in paired_df["seed"]], rotation=0)
        ax.set_xlabel("Overlapping seed")
        ax.set_ylabel("Macro-F1 gain (pp)")
        y_abs = max(5.0, float(np.max(np.abs(y))) + 2.0)
        ax.set_ylim(-y_abs, y_abs)
        if len(paired_df) < 3:
            ax.text(
                0.02,
                0.95,
                f"n={len(paired_df)} true pair(s)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.2,
                color="#B45309",
            )
    ax.grid(axis="y", color="#CBD5E1", alpha=0.55, linewidth=0.55)
    ax.set_axisbelow(True)
    style_axis(ax)
    add_panel_label(ax, panel_label)


def plot_teacher_agreement_panel(ax: plt.Axes, agreement_df: pd.DataFrame, *, panel_label: str) -> None:
    x = agreement_df["teacher_agreement"].astype(float).to_numpy() * 100.0
    y = agreement_df["macro_f1"].astype(float).to_numpy() * 100.0
    selected = agreement_df["selected_fixed_config"].astype(bool).to_numpy()
    ax.scatter(
        x[~selected],
        y[~selected],
        s=17,
        color="#94A3B8",
        edgecolor="white",
        linewidth=0.25,
        alpha=0.65,
        label="Other distill runs",
    )
    ax.scatter(
        x[selected],
        y[selected],
        s=32,
        color=METHOD_COLORS[SEAL_STUDENT_LABEL],
        marker=METHOD_MARKERS[SEAL_STUDENT_LABEL],
        edgecolor="white",
        linewidth=0.55,
        alpha=0.90,
        label="Selected fixed config",
    )
    if len(x) >= 3:
        coeff = np.polyfit(x, y, deg=1)
        xs = np.linspace(max(75.0, float(np.min(x)) - 1.0), min(100.0, float(np.max(x)) + 1.0), 100)
        ys = coeff[0] * xs + coeff[1]
        ax.plot(xs, ys, color="#334155", linewidth=1.0, linestyle=(0, (4, 2)), alpha=0.85)
        corr = float(np.corrcoef(x, y)[0, 1])
        ax.text(0.04, 0.94, f"r={corr:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=6.4, color="#334155")
    ax.set_xlabel("Teacher agreement (%)")
    ax.set_ylabel("True Macro-F1 (%)")
    ax.set_xlim(max(75.0, float(np.min(x)) - 2.0), 100.3)
    ax.set_ylim(max(65.0, float(np.min(y)) - 4.0), 101.0)
    ax.grid(True, color="#CBD5E1", alpha=0.55, linewidth=0.55)
    ax.legend(loc="lower right", frameon=False, fontsize=5.9, handlelength=1.2)
    ax.set_axisbelow(True)
    style_axis(ax)
    add_panel_label(ax, panel_label)


def render_figure(
    *,
    metrics: pd.DataFrame,
    profile_df: pd.DataFrame,
    prediction_tables: Mapping[str, pd.DataFrame],
    confusion_methods: list[str],
    strongest_available_baseline: str,
    paired_delta_df: pd.DataFrame,
    teacher_agreement_df: pd.DataFrame,
    teacher_retention_pct: float,
    output_base: Path,
    title: str,
) -> None:
    configure_style()
    fig, axes = plt.subplots(3, 3, figsize=(12.4, 10.0))
    plt.subplots_adjust(left=0.063, right=0.982, top=0.91, bottom=0.08, wspace=0.34, hspace=0.58)

    if title:
        fig.text(0.063, 0.975, title, ha="left", va="top", fontsize=10.0, fontweight="bold", color="#111827")
    fig.text(
        0.063,
        0.944,
        f"Teacher retention = mean F1(Student) / mean F1(Teacher) x 100% = {teacher_retention_pct:.1f}%",
        ha="left",
        va="top",
        fontsize=7.6,
        color="#0F172A",
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            label=method,
            linewidth=1.5 if method in {TEACHER_LABEL, SEAL_STUDENT_LABEL} else 1.0,
            markersize=4.5,
        )
        for method in METHOD_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.982, 0.972),
        ncol=4,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.0,
        handletextpad=0.35,
        fontsize=6.2,
    )

    plot_distribution_panel(axes[0, 0], metrics, metric_col="macro_f1", ylabel="Macro-F1 (%)", panel_label="a")
    plot_distribution_panel(axes[0, 1], metrics, metric_col="balanced_accuracy", ylabel="Balanced accuracy (%)", panel_label="b")
    plot_distribution_panel(axes[0, 2], metrics, metric_col="s1_recall", ylabel="S1 recall (%)", panel_label="c")

    confusion_norms: dict[str, np.ndarray] = {}
    confusion_counts: dict[str, np.ndarray] = {}
    for method in confusion_methods:
        table = prediction_tables[method]
        counts, normalized = compute_confusion_matrices(table["y_true"].to_numpy(dtype=np.int64), table["y_pred"].to_numpy(dtype=np.int64))
        confusion_counts[method] = counts
        confusion_norms[method] = normalized

    image = plot_confusion_panel(
        axes[1, 0],
        confusion_counts[confusion_methods[0]],
        confusion_norms[confusion_methods[0]],
        title="Teacher-18D",
        panel_label="d",
        show_ylabel=True,
    )
    plot_confusion_panel(
        axes[1, 1],
        confusion_counts[confusion_methods[1]],
        confusion_norms[confusion_methods[1]],
        title="SEAL-Weld Student-13D",
        panel_label="e",
        baseline_normalized=confusion_norms[confusion_methods[2]],
    )
    baseline_title = "Student-only-13D\n(strongest available 13D)"
    if confusion_methods[2] != STUDENT_ONLY_LABEL:
        baseline_title = f"{confusion_methods[2]}\n(strongest available baseline)"
    plot_confusion_panel(
        axes[1, 2],
        confusion_counts[confusion_methods[2]],
        confusion_norms[confusion_methods[2]],
        title=baseline_title,
        panel_label="f",
    )
    cbar = fig.colorbar(image, ax=axes[1, :], fraction=0.025, pad=0.012)
    cbar.set_label("Row-normalized proportion", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)

    profile_methods = [TEACHER_LABEL, SEAL_STUDENT_LABEL, STUDENT_ONLY_LABEL, strongest_available_baseline]
    profile_methods = list(dict.fromkeys(profile_methods))
    plot_class_profile_panel(axes[2, 0], profile_df, methods=profile_methods, panel_label="g")
    plot_paired_delta_panel(axes[2, 1], paired_delta_df, panel_label="h")
    plot_teacher_agreement_panel(axes[2, 2], teacher_agreement_df, panel_label="i")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_qa_notes(
    path: Path,
    *,
    selection_df: pd.DataFrame,
    student_only_reference: Mapping[str, Any],
    strongest_available_baseline: str,
    paired_delta_df: pd.DataFrame,
    teacher_retention_pct: float,
) -> None:
    lines = [
        "Figure 1 QA notes",
        "",
        f"Teacher retention uses mean selected-seed Macro-F1: {teacher_retention_pct:.2f}%.",
        f"Student-only confusion/profile reference run: seed={int(student_only_reference['seed'])}, "
        f"Macro-F1={float(student_only_reference['macro_f1']):.6f}, source=sweep best per seed.",
        f"Strongest available 18D neural baseline from prediction exports: {strongest_available_baseline}.",
        f"Paired Student vs Student-only delta uses true overlapping seeds only: n={len(paired_delta_df)}.",
        "",
        "Data caveats:",
        "- Independent fixed-config TCN-13D result was not found; its method slot is shown as missing.",
        "- LSTM, GRU, Transformer, and Inception entries come from comparison_18d_baselines and are labelled baseline_18d in the repository.",
        "- Student-only-13D points come from a hyperparameter sweep, reduced to one best run per seed; this is not a fixed-config multi-seed repeat.",
        "- Confusion matrices use single representative test prediction exports/checkpoints, while the top-row panels use multi-run distributions.",
        "",
        "Method selection summary:",
        selection_df.to_string(index=False),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_source_tables(
    output_dir: Path,
    *,
    metrics: pd.DataFrame,
    selection_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    prediction_source_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    paired_delta_df: pd.DataFrame,
    teacher_agreement_df: pd.DataFrame,
) -> None:
    source_dir = output_dir / "source_tables"
    source_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(source_dir / "source_multiseed_metrics.csv", index=False)
    selection_df.to_csv(source_dir / "source_method_selection.csv", index=False)
    profile_df.to_csv(source_dir / "source_class_profile.csv", index=False)
    prediction_source_df.to_csv(source_dir / "source_prediction_inputs.csv", index=False)
    confusion_df.to_csv(source_dir / "source_confusion_matrices.csv", index=False)
    paired_delta_df.to_csv(source_dir / "source_paired_student_delta.csv", index=False)
    teacher_agreement_df.to_csv(source_dir / "source_teacher_agreement.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = prepare_output_dir(resolve_repo_path(args.output_root), args.output_name)
    metrics, selection_df, student_only_reference = build_multiseed_metrics(
        args.teacher_summary_csv,
        args.comparison_runs_csv,
        args.student_only_sweep_csv,
    )
    selected_student_config = str(
        selection_df.loc[selection_df["method"] == SEAL_STUDENT_LABEL, "selected_config_id"].iloc[0]
    )
    teacher_agreement_df = build_teacher_agreement_scatter(args.comparison_runs_csv, selected_student_config)
    profile_df, prediction_source_df, prediction_tables, strongest_available_baseline = build_confusion_and_profile_sources(
        args.prediction_dir,
        output_dir,
        student_only_reference,
        device_arg=args.device,
    )
    confusion_methods = [TEACHER_LABEL, SEAL_STUDENT_LABEL, STUDENT_ONLY_LABEL]
    confusion_df = build_confusion_source_rows(prediction_tables, confusion_methods)
    paired_delta_df = build_paired_student_delta(metrics)
    teacher_retention_pct = compute_teacher_retention_pct(metrics)

    output_base = output_dir / args.output_name
    render_figure(
        metrics=metrics,
        profile_df=profile_df,
        prediction_tables=prediction_tables,
        confusion_methods=confusion_methods,
        strongest_available_baseline=strongest_available_baseline,
        paired_delta_df=paired_delta_df,
        teacher_agreement_df=teacher_agreement_df,
        teacher_retention_pct=teacher_retention_pct,
        output_base=output_base,
        title=args.title,
    )
    write_source_tables(
        output_dir,
        metrics=metrics,
        selection_df=selection_df,
        profile_df=profile_df,
        prediction_source_df=prediction_source_df,
        confusion_df=confusion_df,
        paired_delta_df=paired_delta_df,
        teacher_agreement_df=teacher_agreement_df,
    )
    write_qa_notes(
        output_dir / "figure1_qa_notes.txt",
        selection_df=selection_df,
        student_only_reference=student_only_reference,
        strongest_available_baseline=strongest_available_baseline,
        paired_delta_df=paired_delta_df,
        teacher_retention_pct=teacher_retention_pct,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
