#!/usr/bin/env python3
"""焊缝数据集预处理脚本（两条焊缝，按焊缝固定划分 Train/Test）

============================================================================
功能概述
============================================================================
本脚本用于将两条焊缝（两个CSV文件）的时序数据转换为“标签安全（label-safe）”的滑动窗口数据集，
并且**按焊缝身份固定划分训练集/测试集**：
- 训练集：来自 `--train-seam` 的所有窗口
- 测试集：来自 `--test-seam` 的所有窗口

适用场景：你希望做“焊缝域泛化/跨焊缝测试”（train on a01, test on b01），
而不是在每条焊缝内部随机划分 train/test。

============================================================================
处理流程（Pipeline）
============================================================================
1) 每条焊缝独立标准化（per-seam full-fit StandardScaler）
   - 对每条焊缝分别用其全量数据 `fit` scaler，然后 `transform`
   - 注意：这会产生“对测试焊缝也用它自身全量 fit”的信息泄漏（这是你之前允许的策略）

2) 按 label 连续三段切分
   - 假设每条焊缝的标签严格为 0→1→2 且各自连续
   - 即：恰好只有 2 次标签变化点，且顺序必须是 [0,1,2]

3) 段内滑窗（label-safe windowing）
   - 只在每个标签段内部滑窗，不跨 label 边界
   - 因此：每个窗口只对应**一个**标签（0/1/2），而不是 20 个标签

4) 按焊缝固定划分 Train/Test
   - `train-seam` 产生的所有窗口 -> train
   - `test-seam` 产生的所有窗口  -> test
   - 不做 quota/coverage 采样，不做“每组至少 1 个 train/test”的约束

============================================================================
输出（.npz）包含字段
============================================================================
- X_train_full: (N_train, window_size, n_features)
- y_train:      (N_train,)
- seam_id_train:(N_train,)  训练样本所属焊缝ID（train=0）
- X_test_full:  (N_test, window_size, n_features)
- y_test:       (N_test,)
- seam_id_test: (N_test,)   测试样本所属焊缝ID（test=1）
- seam_name_order: [train_seam_id, test_seam_id]
- scaler_mean_<seam>, scaler_scale_<seam>: 保存 scaler 参数用于复现

============================================================================
使用示例
============================================================================
python -m autoencoder_benchmark.prepare_weld_seam_dataset_seam_split \
  --input-dir autoencoder_benchmark/Data/raw_data \
  --train-seam a01.csv --test-seam b01.csv \
  --window-size 20 --stride 1 --seed 42 \
  --output autoencoder_benchmark/Data/processed_data/a01_train__b01_test_ws20_s1.npz

然后训练：
python -m autoencoder_benchmark.train_benchmark_classifier \
  --dataset-npz autoencoder_benchmark/Data/processed_data/a01_train__b01_test_ws20_s1.npz \
  --output-dir outputs/a01_train_b01_test
"""

from __future__ import annotations

# ==========================================================================
# 标准库
# ==========================================================================
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# ==========================================================================
# 第三方库
# ==========================================================================
import numpy as np

# ==========================================================================
# 项目内工具函数
# ==========================================================================
# 支持两种运行方式：
# 1) 作为模块运行：python -m autoencoder_benchmark.prepare_weld_seam_dataset_seam_split
# 2) 直接运行脚本：python autoencoder_benchmark/prepare_weld_seam_dataset_seam_split.py
try:
    from .data_utils import load_seam_csv, standardize_per_seam_full_fit
except ImportError:
    from data_utils import load_seam_csv, standardize_per_seam_full_fit


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    p = argparse.ArgumentParser(
        description="Prepare weld-seam dataset with label-safe windowing + seam-holdout split."
    )

    # 输入数据目录（包含 a01.csv、b01.csv 等）
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("autoencoder_benchmark/Data/raw_data"),
        help="包含焊缝CSV文件的目录。",
    )

    # 训练焊缝：该焊缝产生的所有窗口都进入训练集
    p.add_argument(
        "--train-seam",
        type=str,
        default="a01.csv",
        help="训练焊缝CSV文件名（相对于 --input-dir）。例如 a01.csv",
    )

    # 测试焊缝：该焊缝产生的所有窗口都进入测试集
    p.add_argument(
        "--test-seam",
        type=str,
        default="b01.csv",
        help="测试焊缝CSV文件名（相对于 --input-dir）。例如 b01.csv",
    )

    # 滑动窗口大小（时间步数）
    p.add_argument("--window-size", type=int, default=20)

    # 滑动步长（stride=1 表示窗口最大重叠）
    p.add_argument("--stride", type=int, default=1)

    # 随机种子：这里只用于最终把训练/测试样本顺序 shuffle（不影响成员归属）
    p.add_argument("--seed", type=int, default=42)

    # 输出路径
    p.add_argument(
        "--output",
        type=Path,
        default=Path("autoencoder_benchmark/Data/processed_data/a01_train__b01_test_ws20_s1.npz"),
        help="输出 .npz 文件路径。",
    )

    return p.parse_args()


def split_seam_into_3_segments_by_label(
    data: np.ndarray,
    labels: np.ndarray,
) -> List[Tuple[int, np.ndarray]]:
    """把单条焊缝按标签切成 3 个连续段：0段、1段、2段。

    关键假设：
    - 标签严格为 0→1→2
    - 恰好只有 2 次变化点（0->1 和 1->2）

    返回：[(0, seg0), (1, seg1), (2, seg2)]
    """

    # 1) 基本输入检查
    if data.shape[0] != labels.shape[0]:
        raise ValueError(f"data/labels length mismatch: {data.shape[0]} vs {labels.shape[0]}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got {labels.shape}")

    # 2) 寻找标签变化点（labels[i] != labels[i+1] 的位置）
    # changes 是变化发生后段的起点索引
    changes = np.where(labels[:-1] != labels[1:])[0] + 1

    # 必须恰好有 2 个变化点
    if len(changes) != 2:
        raise ValueError(f"Expected exactly 2 label transitions (0->1->2), got {len(changes)}")

    # 3) 检查变化后的 label 顺序必须是 [0,1,2]
    seq = [int(labels[0]), int(labels[changes[0]]), int(labels[changes[1]])]
    if seq != [0, 1, 2]:
        raise ValueError(f"Expected label order [0,1,2], got {seq}")

    # 4) 切段
    s0 = data[: changes[0]]
    s1 = data[changes[0] : changes[1]]
    s2 = data[changes[1] :]

    # 5) 保证每段内部标签纯净（单一标签）
    if not (
        np.all(labels[: changes[0]] == 0)
        and np.all(labels[changes[0] : changes[1]] == 1)
        and np.all(labels[changes[1] :] == 2)
    ):
        raise ValueError("Non-contiguous labels detected; expected each segment to be pure")

    return [(0, s0), (1, s1), (2, s2)]


def windows_from_segments(
    segments: List[Tuple[int, np.ndarray]],
    window_size: int,
    stride: int,
) -> Dict[int, List[np.ndarray]]:
    """在每个标签段内部提取滑动窗口（不跨 label）。

    返回字典：
    - out[0] = label0段产生的窗口列表
    - out[1] = label1段产生的窗口列表
    - out[2] = label2段产生的窗口列表

    每个窗口形状为 (window_size, n_features)。
    """

    out: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: []}

    for label, seg in segments:
        seg_len = seg.shape[0]

        # 如果当前段长度不足 window_size，则该段无法产生任何窗口
        if seg_len < window_size:
            continue

        # 只在段内滑动：i 的最大值保证窗口不会越界
        for i in range(0, seg_len - window_size + 1, stride):
            out[label].append(seg[i : i + window_size].astype(np.float32, copy=False))

    return out


def _stack(
    windows: Dict[int, List[np.ndarray]],
    seam_int: int,
) -> Tuple[List[np.ndarray], List[int], List[int]]:
    """把某条焊缝的窗口按 label 拼成 (X,y,seam_id) 三个列表。

    - X: 窗口序列
    - y: 每个窗口的标签（0/1/2），因为窗口只在段内滑动，所以每个窗口只有一个 label
    - seam_id: 每个窗口所属焊缝ID（train=0, test=1）
    """

    X: List[np.ndarray] = []
    y: List[int] = []
    seam: List[int] = []

    for label in (0, 1, 2):
        for w in windows[label]:
            X.append(w)
            y.append(label)
            seam.append(seam_int)

    return X, y, seam


def main() -> None:
    args = parse_args()

    # ----------------------------------------------------------------------
    # 1) 定位输入文件
    # ----------------------------------------------------------------------
    train_path = args.input_dir / args.train_seam
    test_path = args.input_dir / args.test_seam

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train seam file: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test seam file: {test_path}")

    train_id = train_path.stem  # a01
    test_id = test_path.stem    # b01
    if train_id == test_id:
        raise ValueError("train seam and test seam must be different")

    # ----------------------------------------------------------------------
    # 2) 读取两条焊缝原始数据
    # ----------------------------------------------------------------------
    seam_data: Dict[str, np.ndarray] = {}
    seam_labels: Dict[str, np.ndarray] = {}

    for p in (train_path, test_path):
        seam_id = p.stem
        X, y = load_seam_csv(p)  # X: (T,18), y: (T,) label in {0,1,2}
        seam_data[seam_id] = X
        seam_labels[seam_id] = y

    # ----------------------------------------------------------------------
    # 3) 每条焊缝独立标准化（全量 fit 再 transform）
    # ----------------------------------------------------------------------
    seam_scaled, scalers = standardize_per_seam_full_fit(seam_data)

    # ----------------------------------------------------------------------
    # 4) 标签安全窗口化：三段切分 -> 段内滑窗
    # ----------------------------------------------------------------------
    windows_by_seam: Dict[str, Dict[int, List[np.ndarray]]] = {}
    available_counts: Dict[str, Dict[int, int]] = {}

    for seam_id in (train_id, test_id):
        segments = split_seam_into_3_segments_by_label(seam_scaled[seam_id], seam_labels[seam_id])
        w = windows_from_segments(segments, window_size=args.window_size, stride=args.stride)
        windows_by_seam[seam_id] = w
        available_counts[seam_id] = {k: len(v) for k, v in w.items()}

    print("\nAvailable windows per (seam,label):")
    for seam_id in (train_id, test_id):
        c = available_counts[seam_id]
        print(f"  {seam_id}: label0={c.get(0,0)} label1={c.get(1,0)} label2={c.get(2,0)}")

    # seam_name_order 约定：train 为 0，test 为 1
    seam_name_order = np.array([train_id, test_id])
    seam_to_int = {train_id: 0, test_id: 1}

    # ----------------------------------------------------------------------
    # 5) 按焊缝固定划分 Train/Test（核心差异点）
    # ----------------------------------------------------------------------
    # train：仅来自 train_id（例如 a01）
    X_train_list, y_train_list, seam_train_list = _stack(
        windows_by_seam[train_id],
        seam_to_int[train_id],
    )

    # test：仅来自 test_id（例如 b01）
    X_test_list, y_test_list, seam_test_list = _stack(
        windows_by_seam[test_id],
        seam_to_int[test_id],
    )

    if not X_train_list:
        raise ValueError("No training windows produced; try decreasing window_size or stride")
    if not X_test_list:
        raise ValueError("No test windows produced; try decreasing window_size or stride")

    # ----------------------------------------------------------------------
    # 6) 为训练方便，对样本顺序做一次 shuffle（不改变训练/测试成员归属）
    # ----------------------------------------------------------------------
    def _shuffle(X: List[np.ndarray], y: List[int], seam: List[int], seed_offset: int):
        X_arr = np.stack(X, axis=0).astype(np.float32)
        y_arr = np.array(y, dtype=np.int64)
        seam_arr = np.array(seam, dtype=np.int64)
        perm = np.random.default_rng(int(args.seed) + seed_offset).permutation(len(X_arr))
        return X_arr[perm], y_arr[perm], seam_arr[perm]

    X_train, y_train, seam_train = _shuffle(X_train_list, y_train_list, seam_train_list, 1)
    X_test, y_test, seam_test = _shuffle(X_test_list, y_test_list, seam_test_list, 2)

    print("\nFinal dataset shapes:")
    print(f"  X_train_full: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test_full:  {X_test.shape},  y_test:  {y_test.shape}")

    # 额外打印：每个标签在 train/test 中出现的焊缝ID
    # 在 seam-holdout 场景下：train 只能看到 [0]，test 只能看到 [1]
    for split, y_arr, seam_arr in [("train", y_train, seam_train), ("test", y_test, seam_test)]:
        print(f"\nSeam presence by label [{split}]:")
        for label in (0, 1, 2):
            seams_present = sorted(set(seam_arr[y_arr == label].tolist()))
            print(f"  label={label}: seams_present={seams_present}")

    # ----------------------------------------------------------------------
    # 7) 保存 .npz
    # ----------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = {
        "X_train_full": X_train.astype(np.float32),
        "y_train": y_train.astype(np.int64),
        "seam_id_train": seam_train.astype(np.int64),
        "X_test_full": X_test.astype(np.float32),
        "y_test": y_test.astype(np.int64),
        "seam_id_test": seam_test.astype(np.int64),
        "seam_name_order": seam_name_order,
    }

    # 保存每条焊缝的 scaler 参数，便于复现
    for seam_id, scaler in scalers.items():
        save_kwargs[f"scaler_mean_{seam_id}"] = scaler.mean_.astype(np.float32)
        save_kwargs[f"scaler_scale_{seam_id}"] = scaler.scale_.astype(np.float32)

    np.savez(args.output, **save_kwargs)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
