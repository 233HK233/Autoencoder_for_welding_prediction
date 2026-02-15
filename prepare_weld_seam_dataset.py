#!/usr/bin/env python3
"""
焊缝数据集预处理脚本 - 带标签安全窗口化和焊缝覆盖约束

============================================================================
功能概述:
============================================================================
本脚本用于将原始焊缝时序数据转换为适合自编码器训练的滑动窗口数据集。
主要解决以下问题:
1. 数据标准化: 消除不同焊缝之间的尺度差异
2. 标签安全: 确保滑动窗口不跨越不同标签的边界
3. 覆盖约束: 确保训练集和测试集都包含所有焊缝的所有标签类型

============================================================================
处理流程 (Pipeline):
============================================================================
1) 按焊缝独立标准化: 对每个焊缝的完整数据拟合StandardScaler，然后变换
   - 这样做的好处是保持每个焊缝内部的相对关系
   - 同时消除不同焊缝之间的绝对尺度差异

2) 按标签分段: 将每个焊缝分成3个连续的标签段 (0->1->2，无重复)
   - 假设数据是按时间顺序排列的
   - 标签从0逐渐过渡到1，再到2（代表不同的健康状态）

3) 标签段内时间切分 + Purge Gap:
   - 在每个(焊缝,标签段)内按时间顺序切分 train/test 时间块
   - 在切分边界处留出 purge gap，确保 train/test 之间不共享原始时间点

4) 标签安全滑动窗口: 分别在 train/test 时间块内进行窗口化
   - 窗口既不跨标签边界，也不跨 train/test 边界

5) 覆盖约束采样: 按(焊缝,标签)组聚合
   - 覆盖约束: 对于每个标签(0/1/2)，训练集和测试集都必须包含所有焊缝的样本
   - 实现方式: 每个(焊缝,标签)组至少保留 train_min=1 和 test_min=1 个样本
   - 配额机制: 仅作为训练侧 cap/下采样上限

============================================================================
输出文件内容 (.npz格式):
============================================================================
- X_train_full: 训练集特征数组，形状为 (n_train, window_size, n_features)
- y_train: 训练集标签数组，形状为 (n_train,)
- X_test_full: 测试集特征数组，形状为 (n_test, window_size, n_features)
- y_test: 测试集标签数组，形状为 (n_test,)
- seam_id_train: 训练样本所属焊缝的ID，形状为 (n_train,)
- seam_id_test: 测试样本所属焊缝的ID，形状为 (n_test,)
- start_idx_train: 训练样本窗口起点在原始焊缝时间轴上的索引，形状为 (n_train,)
- start_idx_test: 测试样本窗口起点在原始焊缝时间轴上的索引，形状为 (n_test,)
- seam_name_order: 焊缝名称到ID的映射顺序
- scaler_mean_<seam>: 各焊缝标准化器的均值
- scaler_scale_<seam>: 各焊缝标准化器的标度

============================================================================
使用示例:
============================================================================
基本用法 (使用默认参数):
  python autoencoder_benchmark/prepare_weld_seam_dataset.py \
    --input-dir autoencoder_benchmark/Data/raw_data \
    --output autoencoder_benchmark/Data/processed_data/weld_seam_windows.npz \
    --window-size 20 --stride 1 --seed 42 \
    --train-frac 0.6

使用自定义配额 (精确控制每个组的训练样本数):
  python ... --quota-json '{"a01": {"0": 200, "1": 300, "2": 300}, "b01": {"0": 200, "1": 300, "2": 300}, "c01": {"0": 80, "1": 80, "2": 80}, "c02": {"0": 80, "1": 80, "2": 80}}'
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from .data_utils import load_seam_csv, standardize_per_seam_full_fit
except ImportError:
    from data_utils import load_seam_csv, standardize_per_seam_full_fit

SEAM_DEFAULTS = ["a01.csv", "b01.csv", "c01.csv", "c02.csv"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="准备焊缝数据集，支持覆盖约束和配额采样"
    )

    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("autoencoder_benchmark/Data/raw_data"),
        help="包含焊缝CSV文件的目录路径",
    )

    p.add_argument(
        "--seams",
        type=str,
        nargs="*",
        default=SEAM_DEFAULTS,
        help="相对于--input-dir的焊缝CSV文件名 (默认: a01.csv b01.csv c01.csv c02.csv)",
    )

    p.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="滑动窗口大小，即每个样本的时间步数 (默认: 20)",
    )

    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="滑动窗口步长 (默认: 1)",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于可复现的数据划分 (默认: 42)",
    )

    p.add_argument(
        "--quota-json",
        type=str,
        default=None,
        help=(
            "可选的JSON配置，指定每个(焊缝,标签)组的训练样本配额。"
            "格式: '{\"a01\": {\"0\": 200, \"1\": 300, \"2\": 300}, ...}'。"
            "如果不提供，则默认不对训练窗口做下采样。"
        ),
    )

    p.add_argument(
        "--train-frac",
        type=float,
        default=0.6,
        help="训练时间块比例，用于时间切分 (默认: 0.6)",
    )

    p.add_argument(
        "--purge-gap",
        type=int,
        default=None,
        help="训练/测试边界的隔离带长度（时间步），默认 window_size-1",
    )

    p.add_argument(
        "--output",
        type=Path,
        default=Path("autoencoder_benchmark/Data/processed_data/weld_seam_windows.npz"),
        help="输出.npz文件的路径",
    )

    return p.parse_args()


def split_seam_into_3_segments_by_label(
    data: np.ndarray,
    labels: np.ndarray,
) -> List[Tuple[int, np.ndarray, int]]:
    if data.shape[0] != labels.shape[0]:
        raise ValueError(f"数据和标签长度不匹配: {data.shape[0]} vs {labels.shape[0]}")

    if labels.ndim != 1:
        raise ValueError(f"标签必须是1维数组，当前形状: {labels.shape}")

    changes = np.where(labels[:-1] != labels[1:])[0] + 1
    if len(changes) != 2:
        raise ValueError(f"期望恰好2个标签转换点(0->1->2)，实际得到 {len(changes)} 个")

    seq = [int(labels[0]), int(labels[changes[0]]), int(labels[changes[1]])]
    if seq != [0, 1, 2]:
        raise ValueError(f"期望标签顺序为 [0, 1, 2]，实际得到 {seq}")

    s0 = data[: changes[0]]
    s1 = data[changes[0] : changes[1]]
    s2 = data[changes[1] :]
    print(f"  分段长度: label0={s0.shape[0]}, label1={s1.shape[0]}, label2={s2.shape[0]}")

    segment_0_pure = np.all(labels[: changes[0]] == 0)
    segment_1_pure = np.all(labels[changes[0] : changes[1]] == 1)
    segment_2_pure = np.all(labels[changes[1] :] == 2)

    if not (segment_0_pure and segment_1_pure and segment_2_pure):
        raise ValueError("检测到非连续标签；期望每个段内只有单一标签")

    return [(0, s0, 0), (1, s1, int(changes[0])), (2, s2, int(changes[1]))]


def time_split_window_starts(
    seg_len: int,
    window_size: int,
    stride: int,
    train_frac: float,
    purge_gap: int,
) -> Tuple[List[int], List[int], Dict[str, int]]:
    if seg_len < window_size:
        return [], [], {"seg_len": seg_len, "total": 0, "train": 0, "test": 0, "purged": 0}

    total_starts = list(range(0, seg_len - window_size + 1, stride))
    cut_time = int(np.floor(train_frac * seg_len))
    purge = max(int(purge_gap), 0)

    train_time_end = max(0, cut_time - purge)
    test_time_start = min(seg_len, cut_time + purge)

    train_starts = [i for i in total_starts if i + window_size <= train_time_end]
    test_starts = [i for i in total_starts if i >= test_time_start]
    purged = len(total_starts) - len(train_starts) - len(test_starts)

    stats = {
        "seg_len": seg_len,
        "total": len(total_starts),
        "train": len(train_starts),
        "test": len(test_starts),
        "purged": purged,
    }
    return train_starts, test_starts, stats


def windows_from_segments_time_split(
    segments: List[Tuple[int, np.ndarray, int]],
    window_size: int,
    stride: int,
    train_frac: float,
    purge_gap: int,
) -> Tuple[
    Dict[int, List[np.ndarray]],
    Dict[int, List[np.ndarray]],
    Dict[int, List[int]],
    Dict[int, List[int]],
    Dict[int, Dict[str, int]],
]:
    train_windows: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
    test_windows: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
    train_start_idx: Dict[int, List[int]] = {0: [], 1: [], 2: []}
    test_start_idx: Dict[int, List[int]] = {0: [], 1: [], 2: []}
    stats: Dict[int, Dict[str, int]] = {}

    for label, seg, seg_start in segments:
        seg_len = seg.shape[0]
        train_starts, test_starts, stat = time_split_window_starts(
            seg_len=seg_len,
            window_size=window_size,
            stride=stride,
            train_frac=train_frac,
            purge_gap=purge_gap,
        )
        stats[label] = stat

        for i in train_starts:
            window = seg[i : i + window_size].astype(np.float32, copy=False)
            train_windows[label].append(window)
            train_start_idx[label].append(int(seg_start + i))

        for i in test_starts:
            window = seg[i : i + window_size].astype(np.float32, copy=False)
            test_windows[label].append(window)
            test_start_idx[label].append(int(seg_start + i))

    return train_windows, test_windows, train_start_idx, test_start_idx, stats


def _normalize_quota_dict(quota: Dict[str, Dict[str, int]]) -> Dict[str, Dict[int, int]]:
    fixed: Dict[str, Dict[int, int]] = {}
    for seam_id, m in quota.items():
        fixed[seam_id] = {}
        for k, v in m.items():
            fixed[seam_id][int(k)] = int(v)
    return fixed


def build_quota_from_available_train(
    available_train: Dict[str, Dict[int, int]],
) -> Dict[str, Dict[int, int]]:
    q: Dict[str, Dict[int, int]] = {}
    for seam_id, per_label in available_train.items():
        q[seam_id] = {label: int(n) for label, n in per_label.items()}
    return q


def sample_train_test_with_coverage_and_quota(
    windows_train: Dict[str, Dict[int, List[np.ndarray]]],
    windows_test: Dict[str, Dict[int, List[np.ndarray]]],
    start_idx_train: Dict[str, Dict[int, List[int]]],
    start_idx_test: Dict[str, Dict[int, List[int]]],
    quota: Dict[str, Dict[int, int]],
    seed: int,
    train_min: int = 1,
    test_min: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    X_train: List[np.ndarray] = []
    y_train: List[int] = []
    seam_train: List[int] = []
    start_train: List[int] = []

    X_test: List[np.ndarray] = []
    y_test: List[int] = []
    seam_test: List[int] = []
    start_test: List[int] = []

    seam_ids = sorted(windows_train.keys())
    seam_to_int = {sid: i for i, sid in enumerate(seam_ids)}

    for seam_id in seam_ids:
        for label in (0, 1, 2):
            train_group = windows_train[seam_id][label]
            test_group = windows_test[seam_id][label]
            train_starts = start_idx_train[seam_id][label]
            test_starts = start_idx_test[seam_id][label]

            n_train = len(train_group)
            n_test = len(test_group)

            if n_train < train_min or n_test < test_min:
                raise ValueError(
                    f"覆盖约束无法满足: seam={seam_id}, label={label}, "
                    f"train={n_train}, test={n_test}。"
                    "尝试减小window_size/stride，或减小purge-gap，或调整train-frac。"
                )

            q = int(quota.get(seam_id, {}).get(label, n_train))
            if q < train_min:
                q = train_min

            if n_train > q:
                sel = rng.choice(n_train, size=q, replace=False)
            else:
                sel = np.arange(n_train)

            for i in sel:
                X_train.append(train_group[int(i)])
                y_train.append(label)
                seam_train.append(seam_to_int[seam_id])
                start_train.append(int(train_starts[int(i)]))

            for i in range(n_test):
                X_test.append(test_group[i])
                y_test.append(label)
                seam_test.append(seam_to_int[seam_id])
                start_test.append(int(test_starts[i]))

    def _stack(
        X: List[np.ndarray],
        y: List[int],
        seam: List[int],
        start_idx: List[int],
        seed_offset: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_arr = np.stack(X, axis=0).astype(np.float32)
        y_arr = np.array(y, dtype=np.int64)
        seam_arr = np.array(seam, dtype=np.int64)
        start_arr = np.array(start_idx, dtype=np.int64)

        perm = np.random.default_rng(seed + seed_offset).permutation(len(X_arr))
        return X_arr[perm], y_arr[perm], seam_arr[perm], start_arr[perm]

    X_tr, y_tr, seam_tr, start_tr = _stack(X_train, y_train, seam_train, start_train, 1)
    X_te, y_te, seam_te, start_te = _stack(X_test, y_test, seam_test, start_test, 2)

    return X_tr, y_tr, seam_tr, start_tr, X_te, y_te, seam_te, start_te


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir
    seam_files = [input_dir / s for s in args.seams]

    for p in seam_files:
        if not p.exists():
            raise FileNotFoundError(f"找不到焊缝文件: {p}")

    seam_data: Dict[str, np.ndarray] = {}
    seam_labels: Dict[str, np.ndarray] = {}

    for p in seam_files:
        seam_id = p.stem
        X, y = load_seam_csv(p)
        seam_data[seam_id] = X
        seam_labels[seam_id] = y

    seam_scaled, scalers = standardize_per_seam_full_fit(seam_data)

    windows_train: Dict[str, Dict[int, List[np.ndarray]]] = {}
    windows_test: Dict[str, Dict[int, List[np.ndarray]]] = {}
    start_idx_train: Dict[str, Dict[int, List[int]]] = {}
    start_idx_test: Dict[str, Dict[int, List[int]]] = {}
    available_train_counts: Dict[str, Dict[int, int]] = {}
    available_test_counts: Dict[str, Dict[int, int]] = {}

    purge_gap = args.window_size - 1 if args.purge_gap is None else int(args.purge_gap)
    print(f"\n时间切分参数: train_frac={args.train_frac}, purge_gap={purge_gap}")

    for seam_id in sorted(seam_scaled.keys()):
        segments = split_seam_into_3_segments_by_label(
            seam_scaled[seam_id],
            seam_labels[seam_id],
        )

        w_train, w_test, s_train, s_test, stats = windows_from_segments_time_split(
            segments,
            window_size=args.window_size,
            stride=args.stride,
            train_frac=float(args.train_frac),
            purge_gap=purge_gap,
        )

        windows_train[seam_id] = w_train
        windows_test[seam_id] = w_test
        start_idx_train[seam_id] = s_train
        start_idx_test[seam_id] = s_test
        available_train_counts[seam_id] = {k: len(v) for k, v in w_train.items()}
        available_test_counts[seam_id] = {k: len(v) for k, v in w_test.items()}

        print(f"\n[{seam_id}] 时间切分统计:")
        for label in (0, 1, 2):
            st = stats.get(label, {"seg_len": 0, "total": 0, "train": 0, "test": 0, "purged": 0})
            print(
                f"  label{label}: seg_len={st['seg_len']} total={st['total']} "
                f"train={st['train']} test={st['test']} purged={st['purged']}"
            )

    print("\n每个(焊缝,标签)组的训练窗口数:")
    for seam_id in sorted(available_train_counts.keys()):
        c = available_train_counts[seam_id]
        print(f"  {seam_id}: label0={c.get(0,0)} label1={c.get(1,0)} label2={c.get(2,0)}")

    print("\n每个(焊缝,标签)组的测试窗口数:")
    for seam_id in sorted(available_test_counts.keys()):
        c = available_test_counts[seam_id]
        print(f"  {seam_id}: label0={c.get(0,0)} label1={c.get(1,0)} label2={c.get(2,0)}")

    if args.quota_json:
        quota_raw = json.loads(args.quota_json)
        quota = _normalize_quota_dict(quota_raw)
    else:
        quota = build_quota_from_available_train(available_train_counts)

    print("\n每个(焊缝,标签)组的训练配额:")
    for seam_id in sorted(quota.keys()):
        q = quota[seam_id]
        print(f"  {seam_id}: label0={q.get(0)} label1={q.get(1)} label2={q.get(2)}")

    (
        X_train,
        y_train,
        seam_train,
        start_train,
        X_test,
        y_test,
        seam_test,
        start_test,
    ) = sample_train_test_with_coverage_and_quota(
        windows_train=windows_train,
        windows_test=windows_test,
        start_idx_train=start_idx_train,
        start_idx_test=start_idx_test,
        quota=quota,
        seed=int(args.seed),
        train_min=1,
        test_min=1,
    )

    print("\n最终数据集形状:")
    print(f"  X_train_full: {X_train.shape}, y_train: {y_train.shape}, start_idx: {start_train.shape}")
    print(f"  X_test_full:  {X_test.shape},  y_test:  {y_test.shape},  start_idx: {start_test.shape}")

    def _coverage(seam_arr: np.ndarray, y_arr: np.ndarray, split: str) -> None:
        seam_ids = sorted(windows_train.keys())
        for label in (0, 1, 2):
            seams_present = set(seam_arr[y_arr == label].tolist())
            ok = len(seams_present) == len(seam_ids)
            print(f"  [{split}] label={label} 包含的焊缝={sorted(seams_present)} 覆盖完整={ok}")

    print(f"\n覆盖约束检查 (焊缝ID为0..{len(windows_train.keys()) - 1}，按文件名排序):")
    _coverage(seam_train, y_train, "train")
    _coverage(seam_test, y_test, "test")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = {
        "X_train_full": X_train.astype(np.float32),
        "y_train": y_train.astype(np.int64),
        "seam_id_train": seam_train.astype(np.int64),
        "start_idx_train": start_train.astype(np.int64),
        "X_test_full": X_test.astype(np.float32),
        "y_test": y_test.astype(np.int64),
        "seam_id_test": seam_test.astype(np.int64),
        "start_idx_test": start_test.astype(np.int64),
        "seam_name_order": np.array(sorted(windows_train.keys())),
    }

    for seam_id, scaler in scalers.items():
        save_kwargs[f"scaler_mean_{seam_id}"] = scaler.mean_.astype(np.float32)
        save_kwargs[f"scaler_scale_{seam_id}"] = scaler.scale_.astype(np.float32)

    np.savez(args.output, **save_kwargs)
    print(f"\n数据已保存至: {args.output}")


if __name__ == "__main__":
    main()
