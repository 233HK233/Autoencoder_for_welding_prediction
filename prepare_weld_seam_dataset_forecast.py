#!/usr/bin/env python3
"""
焊缝数据集预处理脚本（未来步预测版）

目标:
- 保留当前主流程的数据组织方式（按焊缝标准化、按标签段切分、覆盖约束采样）
- 将标签语义改为“窗口末端后 horizon 步”的目标标签
- 采用同段内标注策略：仅保留未来目标仍在当前标签段内的窗口

说明:
- 当 horizon=1 且采样周期为 0.01s 时，目标对应 +0.01s
- 当 horizon=0 时，行为退化为当前时刻/窗口末端标签分类
"""

from __future__ import annotations

import argparse
import json
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
        description="准备焊缝未来步预测数据集（同段内标注 + 覆盖约束）"
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
        default=5,
        help="滑动窗口大小，即每个样本的时间步数",
    )

    p.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="未来预测步数（horizon=1 对应 +1 个采样步）",
    )

    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="滑动窗口步长",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于可复现的数据划分",
    )

    p.add_argument(
        "--quota-json",
        type=str,
        default=None,
        help=(
            "可选JSON配置，指定每个(焊缝,标签)组的训练样本配额。"
            "格式: '{\"a01\": {\"0\": 200, \"1\": 300, \"2\": 300}, ...}'。"
        ),
    )

    p.add_argument(
        "--train-frac",
        type=float,
        default=0.75,
        help="训练时间块比例",
    )

    p.add_argument(
        "--purge-gap",
        type=int,
        default=0,
        help="训练/测试边界隔离带长度（时间步）",
    )

    p.add_argument(
        "--output",
        type=Path,
        default=Path("autoencoder_benchmark/Data/processed_data/weld_seam_windows_forecast_h1.npz"),
        help="输出.npz文件路径",
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


def time_split_window_starts_forecast(
    seg_len: int,
    window_size: int,
    stride: int,
    train_frac: float,
    purge_gap: int,
    horizon: int,
) -> Tuple[List[int], List[int], Dict[str, int]]:
    """返回可用窗口起点。

    同段内标注约束下，窗口起点 i 必须满足:
    i + window_size - 1 + horizon < seg_len
    """
    max_start = seg_len - window_size - horizon
    if max_start < 0:
        return [], [], {
            "seg_len": seg_len,
            "total": 0,
            "train": 0,
            "test": 0,
            "purged": 0,
        }

    total_starts = list(range(0, max_start + 1, stride))
    cut_time = int(np.floor(train_frac * seg_len))
    purge = max(int(purge_gap), 0)

    train_time_end = max(0, cut_time - purge)
    test_time_start = min(seg_len, cut_time + purge)

    # train 样本的未来目标也必须落在训练时间块内。
    train_starts = [
        i for i in total_starts if i + window_size + horizon <= train_time_end
    ]
    # test 样本按窗口起点落在测试时间块定义。
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


def windows_from_segments_time_split_forecast(
    segments: List[Tuple[int, np.ndarray, int]],
    window_size: int,
    stride: int,
    train_frac: float,
    purge_gap: int,
    horizon: int,
) -> Tuple[
    Dict[int, List[np.ndarray]],
    Dict[int, List[np.ndarray]],
    Dict[int, List[int]],
    Dict[int, List[int]],
    Dict[int, List[int]],
    Dict[int, List[int]],
    Dict[int, Dict[str, int]],
]:
    train_windows: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
    test_windows: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}
    train_start_idx: Dict[int, List[int]] = {0: [], 1: [], 2: []}
    test_start_idx: Dict[int, List[int]] = {0: [], 1: [], 2: []}
    train_target_idx: Dict[int, List[int]] = {0: [], 1: [], 2: []}
    test_target_idx: Dict[int, List[int]] = {0: [], 1: [], 2: []}
    stats: Dict[int, Dict[str, int]] = {}

    for label, seg, seg_start in segments:
        seg_len = seg.shape[0]
        train_starts, test_starts, stat = time_split_window_starts_forecast(
            seg_len=seg_len,
            window_size=window_size,
            stride=stride,
            train_frac=train_frac,
            purge_gap=purge_gap,
            horizon=horizon,
        )
        stats[label] = stat

        for i in train_starts:
            window = seg[i : i + window_size].astype(np.float32, copy=False)
            global_start = int(seg_start + i)
            global_target = int(seg_start + i + window_size - 1 + horizon)
            train_windows[label].append(window)
            train_start_idx[label].append(global_start)
            train_target_idx[label].append(global_target)

        for i in test_starts:
            window = seg[i : i + window_size].astype(np.float32, copy=False)
            global_start = int(seg_start + i)
            global_target = int(seg_start + i + window_size - 1 + horizon)
            test_windows[label].append(window)
            test_start_idx[label].append(global_start)
            test_target_idx[label].append(global_target)

    return (
        train_windows,
        test_windows,
        train_start_idx,
        test_start_idx,
        train_target_idx,
        test_target_idx,
        stats,
    )


def _normalize_quota_dict(quota: Dict[str, Dict[str, int]]) -> Dict[str, Dict[int, int]]:
    fixed: Dict[str, Dict[int, int]] = {}
    for seam_id, mapping in quota.items():
        fixed[seam_id] = {}
        for key, value in mapping.items():
            fixed[seam_id][int(key)] = int(value)
    return fixed


def build_quota_from_available_train(
    available_train: Dict[str, Dict[int, int]],
) -> Dict[str, Dict[int, int]]:
    quota: Dict[str, Dict[int, int]] = {}
    for seam_id, per_label in available_train.items():
        quota[seam_id] = {label: int(n) for label, n in per_label.items()}
    return quota


def sample_train_test_with_coverage_and_quota_forecast(
    windows_train: Dict[str, Dict[int, List[np.ndarray]]],
    windows_test: Dict[str, Dict[int, List[np.ndarray]]],
    start_idx_train: Dict[str, Dict[int, List[int]]],
    start_idx_test: Dict[str, Dict[int, List[int]]],
    target_idx_train: Dict[str, Dict[int, List[int]]],
    target_idx_test: Dict[str, Dict[int, List[int]]],
    quota: Dict[str, Dict[int, int]],
    seed: int,
    train_min: int = 1,
    test_min: int = 1,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(seed)

    x_train: List[np.ndarray] = []
    y_train: List[int] = []
    seam_train: List[int] = []
    start_train: List[int] = []
    target_train: List[int] = []

    x_test: List[np.ndarray] = []
    y_test: List[int] = []
    seam_test: List[int] = []
    start_test: List[int] = []
    target_test: List[int] = []

    seam_ids = sorted(windows_train.keys())
    seam_to_int = {sid: idx for idx, sid in enumerate(seam_ids)}

    for seam_id in seam_ids:
        for label in (0, 1, 2):
            train_group = windows_train[seam_id][label]
            test_group = windows_test[seam_id][label]
            train_starts = start_idx_train[seam_id][label]
            test_starts = start_idx_test[seam_id][label]
            train_targets = target_idx_train[seam_id][label]
            test_targets = target_idx_test[seam_id][label]

            n_train = len(train_group)
            n_test = len(test_group)

            if n_train < train_min or n_test < test_min:
                raise ValueError(
                    f"覆盖约束无法满足: seam={seam_id}, label={label}, "
                    f"train={n_train}, test={n_test}。"
                    "尝试减小window_size/stride，或减小horizon/purge-gap，或调整train-frac。"
                )

            cap = int(quota.get(seam_id, {}).get(label, n_train))
            if cap < train_min:
                cap = train_min

            if n_train > cap:
                selected = rng.choice(n_train, size=cap, replace=False)
            else:
                selected = np.arange(n_train)

            for i in selected:
                ii = int(i)
                x_train.append(train_group[ii])
                y_train.append(label)
                seam_train.append(seam_to_int[seam_id])
                start_train.append(int(train_starts[ii]))
                target_train.append(int(train_targets[ii]))

            for i in range(n_test):
                x_test.append(test_group[i])
                y_test.append(label)
                seam_test.append(seam_to_int[seam_id])
                start_test.append(int(test_starts[i]))
                target_test.append(int(test_targets[i]))

    def _stack(
        x_list: List[np.ndarray],
        y_list: List[int],
        seam_list: List[int],
        start_list: List[int],
        target_list: List[int],
        seed_offset: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_arr = np.stack(x_list, axis=0).astype(np.float32)
        y_arr = np.array(y_list, dtype=np.int64)
        seam_arr = np.array(seam_list, dtype=np.int64)
        start_arr = np.array(start_list, dtype=np.int64)
        target_arr = np.array(target_list, dtype=np.int64)

        perm = np.random.default_rng(seed + seed_offset).permutation(len(x_arr))
        return (
            x_arr[perm],
            y_arr[perm],
            seam_arr[perm],
            start_arr[perm],
            target_arr[perm],
        )

    x_tr, y_tr, seam_tr, start_tr, target_tr = _stack(
        x_train, y_train, seam_train, start_train, target_train, 1
    )
    x_te, y_te, seam_te, start_te, target_te = _stack(
        x_test, y_test, seam_test, start_test, target_test, 2
    )

    return x_tr, y_tr, seam_tr, start_tr, target_tr, x_te, y_te, seam_te, start_te, target_te


def main() -> None:
    args = parse_args()

    if args.window_size <= 0:
        raise ValueError("window-size 必须 > 0")
    if args.stride <= 0:
        raise ValueError("stride 必须 > 0")
    if not (0.0 < float(args.train_frac) < 1.0):
        raise ValueError("train-frac 必须在 (0,1) 内")
    if args.horizon < 0:
        raise ValueError("horizon 必须 >= 0")

    input_dir = args.input_dir
    seam_files = [input_dir / seam for seam in args.seams]

    for path in seam_files:
        if not path.exists():
            raise FileNotFoundError(f"找不到焊缝文件: {path}")

    seam_data: Dict[str, np.ndarray] = {}
    seam_labels: Dict[str, np.ndarray] = {}

    for path in seam_files:
        seam_id = path.stem
        x, y = load_seam_csv(path)
        seam_data[seam_id] = x
        seam_labels[seam_id] = y

    seam_scaled, scalers = standardize_per_seam_full_fit(seam_data)

    windows_train: Dict[str, Dict[int, List[np.ndarray]]] = {}
    windows_test: Dict[str, Dict[int, List[np.ndarray]]] = {}
    start_idx_train: Dict[str, Dict[int, List[int]]] = {}
    start_idx_test: Dict[str, Dict[int, List[int]]] = {}
    target_idx_train: Dict[str, Dict[int, List[int]]] = {}
    target_idx_test: Dict[str, Dict[int, List[int]]] = {}
    available_train_counts: Dict[str, Dict[int, int]] = {}
    available_test_counts: Dict[str, Dict[int, int]] = {}

    purge_gap = int(args.purge_gap)
    print(
        "\n时间切分参数: "
        f"train_frac={args.train_frac}, purge_gap={purge_gap}, "
        f"horizon={args.horizon}"
    )

    for seam_id in sorted(seam_scaled.keys()):
        segments = split_seam_into_3_segments_by_label(
            seam_scaled[seam_id],
            seam_labels[seam_id],
        )

        (
            w_train,
            w_test,
            s_train,
            s_test,
            t_train,
            t_test,
            stats,
        ) = windows_from_segments_time_split_forecast(
            segments=segments,
            window_size=int(args.window_size),
            stride=int(args.stride),
            train_frac=float(args.train_frac),
            purge_gap=purge_gap,
            horizon=int(args.horizon),
        )

        windows_train[seam_id] = w_train
        windows_test[seam_id] = w_test
        start_idx_train[seam_id] = s_train
        start_idx_test[seam_id] = s_test
        target_idx_train[seam_id] = t_train
        target_idx_test[seam_id] = t_test
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
        x_train,
        y_train,
        seam_train,
        start_train,
        target_train,
        x_test,
        y_test,
        seam_test,
        start_test,
        target_test,
    ) = sample_train_test_with_coverage_and_quota_forecast(
        windows_train=windows_train,
        windows_test=windows_test,
        start_idx_train=start_idx_train,
        start_idx_test=start_idx_test,
        target_idx_train=target_idx_train,
        target_idx_test=target_idx_test,
        quota=quota,
        seed=int(args.seed),
        train_min=1,
        test_min=1,
    )

    print("\n最终数据集形状:")
    print(
        f"  X_train_full: {x_train.shape}, y_train: {y_train.shape}, "
        f"start_idx: {start_train.shape}, target_idx: {target_train.shape}"
    )
    print(
        f"  X_test_full:  {x_test.shape},  y_test:  {y_test.shape},  "
        f"start_idx: {start_test.shape}, target_idx: {target_test.shape}"
    )

    expected_delta = int(args.window_size - 1 + args.horizon)
    train_delta_ok = bool(np.all((target_train - start_train) == expected_delta))
    test_delta_ok = bool(np.all((target_test - start_test) == expected_delta))
    print(
        "\n索引关系检查: "
        f"target_idx - start_idx == window_size-1+horizon ({expected_delta}) "
        f"train={train_delta_ok} test={test_delta_ok}"
    )

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
        "X_train_full": x_train.astype(np.float32),
        "y_train": y_train.astype(np.int64),
        "seam_id_train": seam_train.astype(np.int64),
        "start_idx_train": start_train.astype(np.int64),
        "target_idx_train": target_train.astype(np.int64),
        "X_test_full": x_test.astype(np.float32),
        "y_test": y_test.astype(np.int64),
        "seam_id_test": seam_test.astype(np.int64),
        "start_idx_test": start_test.astype(np.int64),
        "target_idx_test": target_test.astype(np.int64),
        "target_horizon_steps": np.array(int(args.horizon), dtype=np.int64),
        "seam_name_order": np.array(sorted(windows_train.keys())),
    }

    for seam_id, scaler in scalers.items():
        save_kwargs[f"scaler_mean_{seam_id}"] = scaler.mean_.astype(np.float32)
        save_kwargs[f"scaler_scale_{seam_id}"] = scaler.scale_.astype(np.float32)

    np.savez(args.output, **save_kwargs)
    print(f"\n数据已保存至: {args.output}")


if __name__ == "__main__":
    main()
