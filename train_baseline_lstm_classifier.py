#!/usr/bin/env python3
"""
Train the baseline LSTM classifier on raw weld seam CSVs with label-segment time split.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .data_utils import StandardScaler, load_seam_csv
    from .training_utils import EvalResult, evaluate
except ImportError:
    from data_utils import StandardScaler, load_seam_csv
    from training_utils import EvalResult, evaluate


SEAM_DEFAULTS = ["a01.csv", "b01.csv", "c01.csv", "c02.csv"]
_MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\xdac\xf8\xff"
    b"\xff?\x00\x05\xfe\x02\xfeA\xdd\x94\xf5\x00\x00\x00\x00IEND\xaeB`\x82"
)


class BaselineLSTMClassifier(nn.Module):
    """Baseline LSTM + MLP head matching baseline_LSTM.py semantics."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        lstm_layers: int,
        output_size: int,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        features = self.relu(self.fc1(last_hidden))
        logits = self.fc2(features)
        return logits, features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline LSTM classifier on raw seam CSVs")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Data/raw_data"),
        help="Directory containing raw seam CSV files",
    )
    parser.add_argument(
        "--seams",
        type=str,
        nargs="*",
        default=SEAM_DEFAULTS,
        help="Relative CSV names under --input-dir",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/baseline_lstm"),
        help="Root directory for experiment runs",
    )
    parser.add_argument("--train-frac", type=float, default=0.75)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--target-offset", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_future_offset_sequences(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    target_offset: int,
    start_offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if target_offset <= 0:
        raise ValueError("target_offset must be > 0")
    if data.shape[0] != labels.shape[0]:
        raise ValueError("data and labels must have the same length")

    count = data.shape[0] - window_size - target_offset + 1
    feat_dim = int(data.shape[1]) if data.ndim == 2 else 0
    if count <= 0:
        empty_x = np.empty((0, window_size, feat_dim), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.int64)
        empty_idx = np.empty((0,), dtype=np.int64)
        return empty_x, empty_y, empty_idx, empty_idx.copy()

    sequences: List[np.ndarray] = []
    sequence_labels: List[int] = []
    start_indices: List[int] = []
    target_indices: List[int] = []

    for i in range(count):
        target_idx = start_offset + i + window_size + target_offset - 1
        sequences.append(data[i : i + window_size].astype(np.float32, copy=False))
        sequence_labels.append(int(labels[i + window_size + target_offset - 1]))
        start_indices.append(start_offset + i)
        target_indices.append(target_idx)

    return (
        np.stack(sequences, axis=0).astype(np.float32),
        np.array(sequence_labels, dtype=np.int64),
        np.array(start_indices, dtype=np.int64),
        np.array(target_indices, dtype=np.int64),
    )


def split_seam_into_3_segments_by_label(
    data: np.ndarray,
    labels: np.ndarray,
) -> List[Tuple[int, np.ndarray, np.ndarray, int]]:
    if data.shape[0] != labels.shape[0]:
        raise ValueError(f"data and labels length mismatch: {data.shape[0]} vs {labels.shape[0]}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")

    changes = np.where(labels[:-1] != labels[1:])[0] + 1
    if len(changes) != 2:
        raise ValueError(f"expected exactly 2 label transitions (0->1->2), got {len(changes)}")

    seq = [int(labels[0]), int(labels[changes[0]]), int(labels[changes[1]])]
    if seq != [0, 1, 2]:
        raise ValueError(f"expected label order [0, 1, 2], got {seq}")

    ranges = [(0, 0, int(changes[0])), (1, int(changes[0]), int(changes[1])), (2, int(changes[1]), int(data.shape[0]))]
    segments: List[Tuple[int, np.ndarray, np.ndarray, int]] = []
    for label, start, end in ranges:
        seg_labels = labels[start:end]
        if not np.all(seg_labels == label):
            raise ValueError("non-contiguous labels detected inside segment")
        segments.append((label, data[start:end], seg_labels, start))
    return segments


def choose_segment_train_cut(seg_len: int, train_frac: float, required_len: int) -> Tuple[int, str | None]:
    requested = int(np.floor(float(train_frac) * seg_len))
    requested = min(max(requested, 0), seg_len)

    if seg_len < required_len:
        return requested, "segment shorter than one usable window"

    min_train = required_len
    min_test = required_len
    if seg_len >= (min_train + min_test):
        cut = min(max(requested, min_train), seg_len - min_test)
        if cut != requested:
            return cut, "segment split adjusted to keep both train and test windows"
        return cut, None

    if requested >= required_len:
        return requested, "segment too short for both splits; using train-only windows"
    if (seg_len - requested) >= required_len:
        return requested, "segment too short for both splits; using test-only windows"
    return requested, "segment too short to create train or test windows"


def _concat_or_empty(arrays: List[np.ndarray], shape_tail: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if arrays:
        return np.concatenate(arrays, axis=0).astype(dtype, copy=False)
    return np.empty((0,) + shape_tail, dtype=dtype)


def split_and_window_single_seam_by_label_segments(
    data: np.ndarray,
    labels: np.ndarray,
    train_frac: float,
    window_size: int,
    target_offset: int,
) -> Dict[str, object]:
    if not (0.0 < float(train_frac) < 1.0):
        raise ValueError("train_frac must be in (0,1)")

    required_len = int(window_size + target_offset)
    segments = split_seam_into_3_segments_by_label(data, labels)

    x_train_parts: List[np.ndarray] = []
    y_train_parts: List[np.ndarray] = []
    x_test_parts: List[np.ndarray] = []
    y_test_parts: List[np.ndarray] = []
    start_train_parts: List[np.ndarray] = []
    start_test_parts: List[np.ndarray] = []
    target_train_parts: List[np.ndarray] = []
    target_test_parts: List[np.ndarray] = []
    segment_lengths: List[int] = []
    segment_summaries: List[Dict[str, object]] = []
    warnings: List[str] = []

    for label, seg_data, seg_labels, seg_start in segments:
        seg_len = int(seg_data.shape[0])
        segment_lengths.append(seg_len)
        cut, warning = choose_segment_train_cut(seg_len, train_frac, required_len)
        train_data = seg_data[:cut]
        train_labels = seg_labels[:cut]
        test_data = seg_data[cut:]
        test_labels = seg_labels[cut:]

        x_train, y_train, start_train, target_train = create_future_offset_sequences(
            train_data,
            train_labels,
            window_size=window_size,
            target_offset=target_offset,
            start_offset=seg_start,
        )
        x_test, y_test, start_test, target_test = create_future_offset_sequences(
            test_data,
            test_labels,
            window_size=window_size,
            target_offset=target_offset,
            start_offset=seg_start + cut,
        )

        x_train_parts.append(x_train)
        y_train_parts.append(y_train)
        x_test_parts.append(x_test)
        y_test_parts.append(y_test)
        start_train_parts.append(start_train)
        start_test_parts.append(start_test)
        target_train_parts.append(target_train)
        target_test_parts.append(target_test)

        segment_summary = {
            "label": int(label),
            "segment_len": seg_len,
            "raw_train_rows": int(cut),
            "raw_test_rows": int(seg_len - cut),
            "train_samples": int(x_train.shape[0]),
            "test_samples": int(x_test.shape[0]),
        }
        if warning:
            message = f"label {label}: {warning}"
            warnings.append(message)
            segment_summary["warning"] = warning
        segment_summaries.append(segment_summary)

    feat_dim = int(data.shape[1]) if data.ndim == 2 else 0
    return {
        "x_train": _concat_or_empty(x_train_parts, (window_size, feat_dim), np.float32),
        "y_train": _concat_or_empty(y_train_parts, tuple(), np.int64),
        "x_test": _concat_or_empty(x_test_parts, (window_size, feat_dim), np.float32),
        "y_test": _concat_or_empty(y_test_parts, tuple(), np.int64),
        "start_train": _concat_or_empty(start_train_parts, tuple(), np.int64),
        "start_test": _concat_or_empty(start_test_parts, tuple(), np.int64),
        "target_train": _concat_or_empty(target_train_parts, tuple(), np.int64),
        "target_test": _concat_or_empty(target_test_parts, tuple(), np.int64),
        "segment_lengths": segment_lengths,
        "segment_summaries": segment_summaries,
        "warnings": warnings,
    }


def _bincount_dict(values: np.ndarray, num_classes: int) -> Dict[str, int]:
    counts = np.bincount(values, minlength=num_classes) if len(values) else np.zeros(num_classes, dtype=np.int64)
    return {str(idx): int(counts[idx]) for idx in range(num_classes)}


def prepare_raw_dataset(
    input_dir: Path,
    seams: Sequence[str],
    train_frac: float,
    window_size: int,
    target_offset: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    x_train_parts: List[np.ndarray] = []
    y_train_parts: List[np.ndarray] = []
    x_test_parts: List[np.ndarray] = []
    y_test_parts: List[np.ndarray] = []
    seam_id_train_parts: List[np.ndarray] = []
    seam_id_test_parts: List[np.ndarray] = []
    start_idx_train_parts: List[np.ndarray] = []
    start_idx_test_parts: List[np.ndarray] = []
    target_idx_train_parts: List[np.ndarray] = []
    target_idx_test_parts: List[np.ndarray] = []
    seam_name_order: List[str] = []
    seam_summaries: List[Dict[str, object]] = []
    warnings: List[str] = []

    for seam_idx, seam_name in enumerate(seams):
        path = input_dir / seam_name
        if not path.exists():
            raise FileNotFoundError(f"Missing seam file: {path}")

        raw_x, raw_y = load_seam_csv(path)
        scaler = StandardScaler()
        scaled_x = scaler.fit_transform(raw_x.astype(np.float32)).astype(np.float32)
        split = split_and_window_single_seam_by_label_segments(
            scaled_x,
            raw_y,
            train_frac=train_frac,
            window_size=window_size,
            target_offset=target_offset,
        )

        seam_id = path.stem
        seam_name_order.append(seam_id)
        x_train_parts.append(split["x_train"])
        y_train_parts.append(split["y_train"])
        x_test_parts.append(split["x_test"])
        y_test_parts.append(split["y_test"])
        seam_id_train_parts.append(np.full(split["x_train"].shape[0], seam_idx, dtype=np.int64))
        seam_id_test_parts.append(np.full(split["x_test"].shape[0], seam_idx, dtype=np.int64))
        start_idx_train_parts.append(split["start_train"])
        start_idx_test_parts.append(split["start_test"])
        target_idx_train_parts.append(split["target_train"])
        target_idx_test_parts.append(split["target_test"])

        num_classes = int(max(raw_y.max(), 0) + 1)
        seam_summaries.append(
            {
                "seam_id": seam_id,
                "rows_total": int(raw_x.shape[0]),
                "segment_lengths": split["segment_lengths"],
                "train_samples": int(split["x_train"].shape[0]),
                "test_samples": int(split["x_test"].shape[0]),
                "train_class_counts": _bincount_dict(split["y_train"], num_classes),
                "test_class_counts": _bincount_dict(split["y_test"], num_classes),
                "segments": split["segment_summaries"],
            }
        )
        warnings.extend([f"{seam_id}: {msg}" for msg in split["warnings"]])

    if not x_train_parts or not x_test_parts:
        raise ValueError("No seam data was loaded")

    feat_dim = x_train_parts[0].shape[2] if x_train_parts and x_train_parts[0].ndim == 3 else int(window_size)
    x_train = _concat_or_empty(x_train_parts, (window_size, feat_dim), np.float32)
    y_train = _concat_or_empty(y_train_parts, tuple(), np.int64)
    x_test = _concat_or_empty(x_test_parts, (window_size, feat_dim), np.float32)
    y_test = _concat_or_empty(y_test_parts, tuple(), np.int64)
    seam_id_train = _concat_or_empty(seam_id_train_parts, tuple(), np.int64)
    seam_id_test = _concat_or_empty(seam_id_test_parts, tuple(), np.int64)
    start_idx_train = _concat_or_empty(start_idx_train_parts, tuple(), np.int64)
    start_idx_test = _concat_or_empty(start_idx_test_parts, tuple(), np.int64)
    target_idx_train = _concat_or_empty(target_idx_train_parts, tuple(), np.int64)
    target_idx_test = _concat_or_empty(target_idx_test_parts, tuple(), np.int64)

    if x_train.shape[0] == 0:
        raise ValueError("No train windows created; increase data size or reduce window_size/target_offset")
    if x_test.shape[0] == 0:
        raise ValueError("No test windows created; increase data size or reduce window_size/target_offset")

    num_classes = int(max(y_train.max(), y_test.max()) + 1)
    summary = {
        "input_dir": str(input_dir),
        "seams": seam_name_order,
        "split_strategy": "label_segment_time_split",
        "validation_strategy": "none",
        "train_frac": float(train_frac),
        "window_size": int(window_size),
        "target_offset": int(target_offset),
        "raw_rows_total": int(sum(item["rows_total"] for item in seam_summaries)),
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "train_class_counts": _bincount_dict(y_train, num_classes),
        "test_class_counts": _bincount_dict(y_test, num_classes),
        "warnings": warnings,
        "per_seam": seam_summaries,
    }

    dataset = {
        "X_train_full": x_train,
        "y_train": y_train,
        "X_test_full": x_test,
        "y_test": y_test,
        "seam_id_train": seam_id_train,
        "seam_id_test": seam_id_test,
        "start_idx_train": start_idx_train,
        "start_idx_test": start_idx_test,
        "target_idx_train": target_idx_train,
        "target_idx_test": target_idx_test,
        "seam_name_order": np.array(seam_name_order),
    }
    return dataset, summary


def _metrics_from_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    class_names: List[str],
) -> EvalResult:
    return evaluate(model, loader, device, loss_fn, class_names)


def _build_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true.tolist(), y_pred.tolist()):
        mat[int(true_label), int(pred_label)] += 1
    return mat


def _write_confusion_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    header = ["true/pred"] + [f"class_{idx}" for idx in range(matrix.shape[1])]
    lines = [",".join(header)]
    for row_idx in range(matrix.shape[0]):
        row = [f"class_{row_idx}"] + [str(int(v)) for v in matrix[row_idx]]
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_test_predictions(
    path: Path,
    seam_name_order: Sequence[str],
    seam_ids: np.ndarray,
    start_idx: np.ndarray,
    target_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    headers = [
        "sample_index",
        "seam_id",
        "seam_name",
        "start_idx",
        "target_idx",
        "y_true",
        "y_pred",
    ] + [f"prob_class_{idx}" for idx in range(probabilities.shape[1])]

    lines = [",".join(headers)]
    for idx in range(len(y_true)):
        row = [
            str(idx),
            str(int(seam_ids[idx])),
            str(seam_name_order[int(seam_ids[idx])]),
            str(int(start_idx[idx])),
            str(int(target_idx[idx])),
            str(int(y_true[idx])),
            str(int(y_pred[idx])),
        ] + [f"{float(p):.6f}" for p in probabilities[idx]]
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_load_pyplot():
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
    except Exception:
        return None
    return plt


def _plot_loss_curve(history: List[Dict[str, float]], path: Path) -> str:
    plt = _safe_load_pyplot()
    if plt is None:
        path.write_bytes(_MINIMAL_PNG)
        return "placeholder_png"

    epochs = [item["epoch"] for item in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [item["train_loss"] for item in history], label="train_loss")
    plt.plot(epochs, [item["test_loss"] for item in history], label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Baseline LSTM Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return "matplotlib"


def _plot_metrics_curve(history: List[Dict[str, float]], path: Path) -> str:
    plt = _safe_load_pyplot()
    if plt is None:
        path.write_bytes(_MINIMAL_PNG)
        return "placeholder_png"

    epochs = [item["epoch"] for item in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [item["train_acc"] for item in history], label="train_acc")
    plt.plot(epochs, [item["test_acc"] for item in history], label="test_acc")
    plt.plot(epochs, [item["test_macro_f1"] for item in history], label="test_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Baseline LSTM Metric Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return "matplotlib"


def _predict_probabilities(
    model: nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(TensorDataset(torch.tensor(x)), batch_size=batch_size, shuffle=False)
    logits_parts: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits, _ = model(xb)
            logits_parts.append(logits.cpu().numpy())

    logits_arr = np.concatenate(logits_parts, axis=0)
    logits_t = torch.tensor(logits_arr)
    probs = torch.softmax(logits_t, dim=1).numpy()
    preds = probs.argmax(axis=1).astype(np.int64)
    return preds, probs


def main() -> None:
    args = parse_args()
    if args.window_size <= 0:
        raise ValueError("window-size must be > 0")
    if args.target_offset <= 0:
        raise ValueError("target-offset must be > 0")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")
    if args.epochs <= 0:
        raise ValueError("epochs must be > 0")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, summary = prepare_raw_dataset(
        input_dir=args.input_dir,
        seams=args.seams,
        train_frac=args.train_frac,
        window_size=args.window_size,
        target_offset=args.target_offset,
    )

    x_train = dataset["X_train_full"].astype(np.float32)
    y_train = dataset["y_train"].astype(np.int64)
    x_test = dataset["X_test_full"].astype(np.float32)
    y_test = dataset["y_test"].astype(np.int64)

    num_classes = int(max(y_train.max(), y_test.max()) + 1)
    class_names = [f"Class {idx}" for idx in range(num_classes)]

    model = BaselineLSTMClassifier(
        input_size=int(x_train.shape[2]),
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        output_size=num_classes,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(x_test), torch.tensor(y_test, dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_tag = args.input_dir.name
    run_name = (
        f"baseline_lstm_segment_{dataset_tag}_tf{int(round(args.train_frac * 100))}_"
        f"ws{args.window_size}_to{args.target_offset}_ep{args.epochs}_"
        f"lr{args.lr}_bs{args.batch_size}_seed{args.seed}"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    run_args_payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    run_args_payload["resolved_output_dir"] = str(run_dir)
    run_args_payload["split_strategy"] = "label_segment_time_split"
    run_args_payload["validation_strategy"] = "none"
    with (run_dir / "run_args.json").open("w", encoding="utf-8") as f:
        json.dump(run_args_payload, f, indent=2, ensure_ascii=True)

    with (run_dir / "dataset_split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print("=" * 80)
    print("Baseline LSTM Classifier")
    print("=" * 80)
    print(f"Input dir: {args.input_dir}")
    print(f"Run dir: {run_dir}")
    print(f"Device: {device}")
    print(f"Train/Test: {len(x_train)} / {len(x_test)}")
    print(f"Input shape: T={x_train.shape[1]}, C={x_train.shape[2]}")
    if summary["warnings"]:
        print("Warnings:")
        for item in summary["warnings"]:
            print(f"  - {item}")

    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total_count += int(yb.size(0))
            n_batches += 1

        train_loss = total_loss / n_batches if n_batches else 0.0
        train_acc = float(total_correct / total_count) if total_count else 0.0
        test_res = _metrics_from_loader(model, test_loader, device, loss_fn, class_names)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_res.loss,
                "test_acc": test_res.accuracy,
                "test_macro_f1": test_res.macro_f1,
                "lr": float(args.lr),
            }
        )
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"test_acc={test_res.accuracy*100:.2f}% test_f1={test_res.macro_f1:.4f}"
        )

    train_eval = _metrics_from_loader(model, train_loader, device, loss_fn, class_names)
    test_eval = _metrics_from_loader(model, test_loader, device, loss_fn, class_names)
    torch.save(model.state_dict(), run_dir / "best_baseline_lstm.pth")

    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=True)

    test_pred, test_prob = _predict_probabilities(model, x_test, args.batch_size, device)
    _save_test_predictions(
        run_dir / "test_predictions.csv",
        seam_name_order=dataset["seam_name_order"].tolist(),
        seam_ids=dataset["seam_id_test"],
        start_idx=dataset["start_idx_test"],
        target_idx=dataset["target_idx_test"],
        y_true=y_test,
        y_pred=test_pred,
        probabilities=test_prob,
    )

    conf_mat = _build_confusion_matrix(y_test, test_pred, num_classes)
    _write_confusion_matrix_csv(run_dir / "confusion_matrix_test.csv", conf_mat)
    loss_curve_status = _plot_loss_curve(history, run_dir / "loss_curve.png")
    metrics_curve_status = _plot_metrics_curve(history, run_dir / "metrics_curve.png")
    with (run_dir / "artifact_status.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "loss_curve": loss_curve_status,
                "metrics_curve": metrics_curve_status,
                "model_checkpoint": "final_epoch",
            },
            f,
            indent=2,
            ensure_ascii=True,
        )

    with (run_dir / "evaluation_metrics.txt").open("w", encoding="utf-8") as f:
        f.write(f"Training epochs: {args.epochs}\n")
        f.write("Model selection: final_epoch_no_validation\n\n")
        f.write("--- Train Metrics ---\n")
        f.write(f"Loss: {train_eval.loss:.6f}\n")
        f.write(f"Accuracy: {train_eval.accuracy*100:.2f}%\n")
        f.write(f"Macro-F1: {train_eval.macro_f1:.6f}\n")
        f.write("\n=== Classification Report (Train) ===\n")
        f.write(train_eval.report + "\n\n")
        f.write("--- Test Metrics ---\n")
        f.write(f"Loss: {test_eval.loss:.6f}\n")
        f.write(f"Accuracy: {test_eval.accuracy*100:.2f}%\n")
        f.write(f"Macro-F1: {test_eval.macro_f1:.6f}\n")
        f.write("\n=== Classification Report (Test) ===\n")
        f.write(test_eval.report + "\n")

    print("=" * 80)
    print(f"Training epochs: {args.epochs}")
    print(f"Final test accuracy: {test_eval.accuracy*100:.2f}%")
    print(f"Final test macro-F1: {test_eval.macro_f1:.4f}")
    print(f"Saved to: {run_dir}")


if __name__ == "__main__":
    main()
