"""
Shared training utilities for single-model and distillation scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import classification_report as sk_classification_report
except Exception:
    sk_classification_report = None


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    macro_f1: float
    report: str


def extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        if all(isinstance(k, str) and torch.is_tensor(v) for k, v in payload.items()):
            return payload

        for key in ("state_dict", "model_state_dict", "model", "weights"):
            value = payload.get(key)
            if isinstance(value, Mapping) and all(
                isinstance(k, str) and torch.is_tensor(v) for k, v in value.items()
            ):
                return value

    raise ValueError("checkpoint payload does not contain a supported state dict mapping")


def load_model_init_weights(
    model: nn.Module,
    state_payload: Any,
    init_mode: str = "strict",
) -> dict[str, Any]:
    mode = str(init_mode).strip().lower()
    if mode not in {"strict", "shape_safe"}:
        raise ValueError(f"unsupported init_mode: {init_mode}")

    state_dict = extract_state_dict(state_payload)

    if mode == "strict":
        model.load_state_dict(state_dict, strict=True)
        return {
            "init_mode": mode,
            "loaded_keys": len(state_dict),
            "skipped_keys": [],
            "missing_model_keys": [],
        }

    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    skipped = sorted(key for key in state_dict.keys() if key not in compatible)
    missing = sorted(key for key in model_state.keys() if key not in compatible)
    model.load_state_dict(compatible, strict=False)
    return {
        "init_mode": mode,
        "loaded_keys": len(compatible),
        "skipped_keys": skipped,
        "missing_model_keys": missing,
    }


def parse_channels(value: str | None, latent_dim: int, min_layers: int) -> List[int]:
    if not value:
        channels = [latent_dim] * max(min_layers, 1)
    else:
        parts = [p.strip() for p in value.split(",") if p.strip()]
        channels = [int(p) for p in parts]
        if len(channels) < max(min_layers, 1):
            channels = channels + [channels[-1]] * (max(min_layers, 1) - len(channels))
    if channels[-1] != latent_dim:
        channels.append(latent_dim)
    return channels


def split_train_val_stratified(y: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0,1)")

    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []

    for cls in sorted(np.unique(y).tolist()):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * val_ratio)))
        n_val = min(n_val, len(cls_idx) - 1) if len(cls_idx) > 1 else 1
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())

    if not train_idx:
        raise ValueError("No train samples after split; decrease val_ratio")
    if not val_idx:
        raise ValueError("No val samples after split; increase val_ratio")

    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


def compute_class_weights(y: np.ndarray, num_classes: int, mode: str) -> torch.Tensor | None:
    mode = mode.strip().lower()
    if mode == "none":
        return None
    if mode == "auto":
        counts = np.bincount(y, minlength=num_classes)
        counts = np.maximum(counts, 1)
        inv_freq = counts.sum() / (num_classes * counts)
        return torch.tensor(inv_freq, dtype=torch.float32)

    vals = [float(v.strip()) for v in mode.split(",") if v.strip()]
    if len(vals) != num_classes:
        raise ValueError(f"class-weights must have {num_classes} values, got {len(vals)}")
    return torch.tensor(vals, dtype=torch.float32)


def validate_forecast_dataset_contract(
    npz_path: str | Path,
    expected_horizon: int = 1,
    expected_delta: int = 5,
) -> dict[str, int | bool | str]:
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"dataset npz not found: {path}")

    with np.load(str(path), allow_pickle=False) as z:
        if "target_horizon_steps" not in z.files:
            raise ValueError("target_horizon_steps not found in dataset")
        horizon = int(np.asarray(z["target_horizon_steps"]).item())

        train_delta_ok = False
        test_delta_ok = False
        train_count = 0
        test_count = 0

        if "target_idx_train" in z.files and "start_idx_train" in z.files:
            train_delta = z["target_idx_train"] - z["start_idx_train"]
            train_delta_ok = bool(np.all(train_delta == expected_delta))
            train_count = int(train_delta.shape[0])
        if "target_idx_test" in z.files and "start_idx_test" in z.files:
            test_delta = z["target_idx_test"] - z["start_idx_test"]
            test_delta_ok = bool(np.all(test_delta == expected_delta))
            test_count = int(test_delta.shape[0])

    result = {
        "dataset_path": str(path.resolve()),
        "dataset_size_bytes": int(path.stat().st_size),
        "target_horizon_steps": horizon,
        "expected_horizon": int(expected_horizon),
        "horizon_ok": bool(horizon == expected_horizon),
        "expected_delta": int(expected_delta),
        "train_delta_ok": bool(train_delta_ok),
        "test_delta_ok": bool(test_delta_ok),
        "train_sample_count": train_count,
        "test_sample_count": test_count,
    }

    if not result["horizon_ok"]:
        raise ValueError(
            f"dataset target_horizon_steps={result['target_horizon_steps']} "
            f"!= expected {expected_horizon}"
        )
    if not result["train_delta_ok"] or not result["test_delta_ok"]:
        raise ValueError(
            "dataset target_idx-start_idx check failed: "
            f"train_ok={result['train_delta_ok']} test_ok={result['test_delta_ok']}"
        )

    return result


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def build_classification_report(
    y_true: List[int], y_pred: List[int], class_names: List[str]
) -> Tuple[str, float]:
    if sk_classification_report is not None:
        try:
            report_dict = sk_classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                output_dict=True,
                digits=4,
                zero_division=0,
            )
            text = sk_classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                digits=4,
                zero_division=0,
            )
            macro_f1 = float(report_dict.get("macro avg", {}).get("f1-score", 0.0))
            return text, macro_f1
        except Exception:
            pass

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    lines = ["              precision    recall  f1-score   support"]
    f1_list: List[float] = []

    for cls, name in enumerate(class_names):
        tp = int(np.sum((y_pred_np == cls) & (y_true_np == cls)))
        fp = int(np.sum((y_pred_np == cls) & (y_true_np != cls)))
        fn = int(np.sum((y_pred_np != cls) & (y_true_np == cls)))
        support = int(np.sum(y_true_np == cls))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        f1_list.append(f1)
        lines.append(f"{name:>12} {precision:10.4f} {recall:8.4f} {f1:10.4f} {support:9d}")

    acc = float(np.mean(y_true_np == y_pred_np)) if len(y_true_np) else 0.0
    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    lines.append(f"\n    accuracy {acc:28.4f} {len(y_true):9d}")
    lines.append(f"   macro avg {0.0:10.4f} {0.0:8.4f} {macro_f1:10.4f} {len(y_true):9d}")
    return "\n".join(lines), macro_f1


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    class_names: List[str],
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits, _ = model(x_batch)
            loss = loss_fn(logits, y_batch)

            total_loss += float(loss.item())
            n_batches += 1
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(y_batch.cpu().tolist())

    avg_loss = total_loss / n_batches if n_batches else 0.0
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels))) if all_labels else 0.0
    report, macro_f1 = build_classification_report(all_labels, all_preds, class_names)
    return EvalResult(loss=avg_loss, accuracy=accuracy, macro_f1=macro_f1, report=report)
