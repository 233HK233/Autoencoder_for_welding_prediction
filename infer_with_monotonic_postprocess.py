#!/usr/bin/env python3
"""Inference with optional ensemble averaging and monotonic temporal decoding.

Usage example:
python autoencoder_benchmark/infer_with_monotonic_postprocess.py \
  --dataset-npz autoencoder_benchmark/Data/processed_data/weld_seam_windows_ws5_tf75_pg0.npz \
  --checkpoints ckpt1.pth ckpt2.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch

try:
    from .models_tcn import AttentionTCNClassifier
except ImportError:
    from models_tcn import AttentionTCNClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensemble inference + temporal post-processing")
    p.add_argument("--dataset-npz", type=str, required=True)
    p.add_argument("--checkpoints", type=str, nargs="+", required=True)

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

    p.add_argument("--decode", type=str, default="both", choices=["none", "monotonic", "three_segment", "both"])
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
    args = parse_args()
    device = torch.device(args.device)

    z = np.load(args.dataset_npz)
    x_test = z["X_test_full"].astype(np.float32)
    y_test = z["y_test"].astype(np.int64)
    seam = z["seam_id_test"].astype(np.int64)
    start_idx = z["start_idx_test"].astype(np.int64)

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

    if args.decode in ("monotonic", "both"):
        mono_pred = monotonic_decode(probs_avg, seam=seam, start_idx=start_idx)
        print_metrics("monotonic", y_test, mono_pred)

    if args.decode in ("three_segment", "both"):
        seg_pred = three_segment_decode(probs_avg, seam=seam, start_idx=start_idx)
        print_metrics("three_segment", y_test, seg_pred)


if __name__ == "__main__":
    main()
