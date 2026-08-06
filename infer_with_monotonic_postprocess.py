#!/usr/bin/env python3
"""Inference with optional ensemble averaging and monotonic temporal decoding.

Usage example:
python autoencoder_benchmark/infer_with_monotonic_postprocess.py \
  --dataset-npz autoencoder_benchmark/Data/processed_data/weld_seam_windows_ws5_tf75_pg0.npz \
  --checkpoints ckpt1.pth ckpt2.pth
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

try:
    from .models_tcn import AttentionTCNClassifier
except ImportError:
    from models_tcn import AttentionTCNClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensemble inference + temporal post-processing")
    p.add_argument("--dataset-npz", type=str, default=None)
    p.add_argument("--checkpoints", type=str, nargs="+", default=None)
    p.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Existing run directory containing run_args.json and best_single_tcn.pth",
    )
    p.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to export detailed per-sample predictions",
    )

    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # model config (must match checkpoint architecture)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--tcn-layers", type=int, default=3)
    p.add_argument("--tcn-kernel", type=int, default=3)
    p.add_argument("--tcn-dropout", type=float, default=0.15)
    p.add_argument("--tcn-dilation-base", type=int, default=2)
    p.add_argument("--attn-heads", type=int, default=4)
    p.add_argument("--attn-dropout", type=float, default=0.1)
    p.add_argument("--attn-ff-dim", type=int, default=128)
    p.add_argument("--classifier-hidden", type=int, default=128)
    p.add_argument("--classifier-dropout", type=float, default=0.35)

    p.add_argument(
        "--decode",
        type=str,
        default="both",
        choices=["raw", "none", "monotonic", "three_segment", "both"],
    )
    return p.parse_args()


def build_model(args: argparse.Namespace, input_dim: int, num_classes: int, device: torch.device) -> AttentionTCNClassifier:
    m = AttentionTCNClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        channels=args.channels,
        tcn_layers=args.tcn_layers,
        tcn_kernel=args.tcn_kernel,
        tcn_dropout=args.tcn_dropout,
        dilation_base=args.tcn_dilation_base,
        attn_heads=args.attn_heads,
        attn_dropout=args.attn_dropout,
        ff_dim=args.attn_ff_dim,
        classifier_hidden=args.classifier_hidden,
        classifier_dropout=args.classifier_dropout,
    )
    return m.to(device)


def load_run_config_from_run_dir(run_dir: str | Path) -> Dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    args_path = run_path / "run_args.json"
    ckpt_path = run_path / "best_single_tcn.pth"
    if not args_path.exists():
        raise FileNotFoundError(f"run_args.json not found under {run_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"best_single_tcn.pth not found under {run_path}")

    payload = json.loads(args_path.read_text(encoding="utf-8"))
    dataset_npz = payload.get("dataset_npz")
    if not dataset_npz:
        raise ValueError(f"dataset_npz missing in {args_path}")

    return {
        "dataset_npz": str(dataset_npz),
        "checkpoints": [str(ckpt_path)],
        "channels": int(payload.get("latent_dim", 64)),
        "tcn_layers": int(payload.get("tcn_layers", 3)),
        "tcn_kernel": int(payload.get("tcn_kernel", 3)),
        "tcn_dropout": float(payload.get("tcn_dropout", 0.15)),
        "tcn_dilation_base": int(payload.get("tcn_dilation_base", 2)),
        "attn_heads": int(payload.get("attn_heads", 4)),
        "attn_dropout": float(payload.get("attn_dropout", 0.1)),
        "attn_ff_dim": int(payload.get("attn_ff_dim", 128)),
        "classifier_hidden": int(payload.get("classifier_hidden", 128)),
        "classifier_dropout": float(payload.get("classifier_dropout", 0.35)),
    }


def save_prediction_details_csv(
    path: str | Path,
    seam_name_order: np.ndarray,
    seam_ids: np.ndarray,
    start_idx: np.ndarray,
    target_idx: np.ndarray,
    y_true: np.ndarray,
    raw_pred: np.ndarray,
    final_pred: np.ndarray,
    probabilities: np.ndarray,
    decode_mode: str,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "sample_index",
        "seam_id",
        "seam_name",
        "start_idx",
        "target_idx",
        "y_true",
        "raw_pred",
        "final_pred",
        "decode_mode",
    ] + [f"prob_class_{idx}" for idx in range(probabilities.shape[1])]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for idx in range(len(y_true)):
            row = [
                idx,
                int(seam_ids[idx]),
                str(seam_name_order[int(seam_ids[idx])]),
                int(start_idx[idx]),
                int(target_idx[idx]),
                int(y_true[idx]),
                int(raw_pred[idx]),
                int(final_pred[idx]),
                decode_mode,
            ] + [f"{float(p):.6f}" for p in probabilities[idx]]
            writer.writerow(row)


def resolve_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.decode == "none":
        args.decode = "raw"
    if args.run_dir is None:
        if not args.dataset_npz:
            raise ValueError("--dataset-npz is required when --run-dir is not provided")
        if not args.checkpoints:
            raise ValueError("--checkpoints is required when --run-dir is not provided")
        return args

    resolved = load_run_config_from_run_dir(args.run_dir)
    if args.dataset_npz is None:
        args.dataset_npz = resolved["dataset_npz"]
    if not args.checkpoints:
        args.checkpoints = resolved["checkpoints"]
    for field in (
        "channels",
        "tcn_layers",
        "tcn_kernel",
        "tcn_dropout",
        "tcn_dilation_base",
        "attn_heads",
        "attn_dropout",
        "attn_ff_dim",
        "classifier_hidden",
        "classifier_dropout",
    ):
        setattr(args, field, resolved[field])
    if args.output_csv is None:
        args.output_csv = str(Path(args.run_dir) / "test_predictions_detailed.csv")
    return args


def monotonic_decode(probs: np.ndarray, seam: np.ndarray, start_idx: np.ndarray) -> np.ndarray:
    pred = np.empty(probs.shape[0], dtype=np.int64)
    for sid in np.unique(seam):
        idx = np.where(seam == sid)[0]
        ord_idx = idx[np.argsort(start_idx[idx])]

        lp = np.log(np.clip(probs[ord_idx], 1e-12, 1.0))
        t, c = lp.shape
        dp = np.full((t, c), -1e18, dtype=np.float64)
        prev = np.full((t, c), -1, dtype=np.int64)
        dp[0] = lp[0]

        for i in range(1, t):
            for cls in range(c):
                k = int(np.argmax(dp[i - 1, : cls + 1]))
                dp[i, cls] = lp[i, cls] + dp[i - 1, k]
                prev[i, cls] = k

        seq = np.zeros(t, dtype=np.int64)
        seq[-1] = int(np.argmax(dp[-1]))
        for i in range(t - 1, 0, -1):
            seq[i - 1] = prev[i, seq[i]]
        pred[ord_idx] = seq

    return pred


def three_segment_decode(probs: np.ndarray, seam: np.ndarray, start_idx: np.ndarray) -> np.ndarray:
    pred = np.empty(probs.shape[0], dtype=np.int64)
    for sid in np.unique(seam):
        idx = np.where(seam == sid)[0]
        ord_idx = idx[np.argsort(start_idx[idx])]
        lp = np.log(np.clip(probs[ord_idx], 1e-12, 1.0))
        n = lp.shape[0]

        cs = np.cumsum(lp, axis=0)

        def seg_sum(cls: int, l: int, r: int) -> float:
            if r <= l:
                return -1e18
            return float(cs[r - 1, cls] - (cs[l - 1, cls] if l > 0 else 0.0))

        best = -1e18
        b1, b2 = 1, 2
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                score = seg_sum(0, 0, i) + seg_sum(1, i, j) + seg_sum(2, j, n)
                if score > best:
                    best = score
                    b1, b2 = i, j

        seq = np.empty(n, dtype=np.int64)
        seq[:b1] = 0
        seq[b1:b2] = 1
        seq[b2:] = 2
        pred[ord_idx] = seq

    return pred


def print_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    acc = float((y_true == y_pred).mean())
    print(f"{name} accuracy: {acc * 100:.4f}%")
    for cls in sorted(np.unique(y_true).tolist()):
        m = y_true == cls
        rec = float((y_pred[m] == cls).mean()) if m.any() else 0.0
        print(f"  class {cls} recall: {rec:.4f} (support={int(m.sum())})")


def main() -> None:
    args = resolve_runtime_args(parse_args())
    device = torch.device(args.device)

    z = np.load(args.dataset_npz)
    x_test = z["X_test_full"].astype(np.float32)
    y_test = z["y_test"].astype(np.int64)
    seam = z["seam_id_test"].astype(np.int64)
    start_idx = z["start_idx_test"].astype(np.int64)
    target_idx = z["target_idx_test"].astype(np.int64)
    seam_name_order = z["seam_name_order"]

    num_classes = int(y_test.max() + 1)
    input_dim = int(x_test.shape[2])

    probs_sum = np.zeros((x_test.shape[0], num_classes), dtype=np.float64)

    for ckpt in args.checkpoints:
        ckpt_path = Path(ckpt)
        model = build_model(args, input_dim=input_dim, num_classes=num_classes, device=device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        all_probs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, x_test.shape[0], args.batch_size):
                xb = torch.tensor(x_test[i : i + args.batch_size], dtype=torch.float32, device=device)
                logits, _ = model(xb)
                p = torch.softmax(logits, dim=1).detach().cpu().numpy()
                all_probs.append(p)
        probs = np.concatenate(all_probs, axis=0)
        probs_sum += probs

    probs_avg = probs_sum / float(len(args.checkpoints))
    raw_pred = probs_avg.argmax(axis=1)
    print_metrics("raw", y_test, raw_pred)
    final_pred = raw_pred
    final_mode = "raw"

    if args.decode in ("monotonic", "both"):
        mono_pred = monotonic_decode(probs_avg, seam=seam, start_idx=start_idx)
        print_metrics("monotonic", y_test, mono_pred)
        if args.decode == "monotonic":
            final_pred = mono_pred
            final_mode = "monotonic"

    if args.decode in ("three_segment", "both"):
        seg_pred = three_segment_decode(probs_avg, seam=seam, start_idx=start_idx)
        print_metrics("three_segment", y_test, seg_pred)
        if args.decode == "three_segment":
            final_pred = seg_pred
            final_mode = "three_segment"

    if args.decode == "both":
        final_pred = raw_pred
        final_mode = "raw"

    if args.output_csv:
        save_prediction_details_csv(
            path=args.output_csv,
            seam_name_order=seam_name_order,
            seam_ids=seam,
            start_idx=start_idx,
            target_idx=target_idx,
            y_true=y_test,
            raw_pred=raw_pred,
            final_pred=final_pred,
            probabilities=probs_avg,
            decode_mode=final_mode,
        )
        print(f"Saved detailed predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
