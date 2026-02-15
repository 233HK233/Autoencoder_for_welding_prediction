"""
焊接实验数据预处理脚本
处理 c01.csv 和 c02.csv 文件，进行独立标准化、按标签分段和数据融合
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path


# 配置
DATA_DIR = Path(__file__).parent / "Data" / "raw_data" / "train_data"
OUTPUT_FILE = Path(__file__).parent / "processed_c_data.csv"
FILE_LIST = ["c01.csv", "c02.csv"]
FEATURE_COLS = list(range(18))  # 第0-17列为特征
LABEL_COL = 18  # 第18列为标签
LABEL_CATEGORIES = ["quasistable", "nonstationary", "instability"]


def load_and_standardize(file_path: Path) -> pd.DataFrame:
    """
    读取单个CSV文件并进行独立标准化

    【关键说明 - 为什么要独立标准化】
    每个CSV文件来自不同的实验工况，数值范围和分布可能存在显著差异。
    如果将所有文件合并后再标准化，会导致：
    1. 分布偏移(Distribution Shift)：某些工况的数据可能被其他工况"淹没"
    2. 特征尺度失真：不同工况的特征均值和方差混合，无法准确反映各自的特性
    3. 模型泛化能力下降：模型可能学习到工况间的差异而非真正的物理特征

    因此，对每个文件单独进行 Z-Score 标准化 (x - mean) / std，
    使每个工况的数据都转换为均值为0、标准差为1的分布，消除工况间的分布偏移。
    """
    # 读取CSV，无表头
    df = pd.read_csv(file_path, header=None)

    # 分离特征和标签
    features = df.iloc[:, FEATURE_COLS].values
    labels = df.iloc[:, LABEL_COL].values

    # 对特征进行独立标准化 (Z-Score)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 重新组合为DataFrame
    df_scaled = pd.DataFrame(features_scaled, columns=[f"feature_{i}" for i in range(18)])
    df_scaled["label"] = labels

    print(f"已加载并标准化: {file_path.name}, 样本数: {len(df_scaled)}")
    return df_scaled


def segment_by_label(df: pd.DataFrame) -> dict:
    """
    按标签分段，将数据拆分为3个子集
    """
    segments = {}
    for label in LABEL_CATEGORIES:
        segment = df[df["label"] == label].copy()
        segments[label] = segment
        print(f"  - {label}: {len(segment)} 样本")
    return segments


def fuse_datasets(all_segments: list) -> pd.DataFrame:
    """
    数据融合：将所有子集合并为一个DataFrame
    """
    all_dfs = []
    for segments in all_segments:
        for label, df in segments.items():
            all_dfs.append(df)

    # 合并并重置索引
    c_dataset = pd.concat(all_dfs, ignore_index=True)
    return c_dataset


def main():
    """主函数：执行完整的预处理流程"""
    print("=" * 60)
    print("焊接实验数据预处理")
    print("=" * 60)

    # Step 1: 独立标准化
    print("\n[Step 1] 独立标准化...")
    all_segments = []

    for file_name in FILE_LIST:
        file_path = DATA_DIR / file_name
        df_scaled = load_and_standardize(file_path)

        # Step 2: 按标签分段
        print(f"  分段结果:")
        segments = segment_by_label(df_scaled)
        all_segments.append(segments)

    # Step 3: 数据融合
    print("\n[Step 2] 按标签分段... (已在上一步完成)")
    print("\n[Step 3] 数据融合...")
    c_dataset = fuse_datasets(all_segments)

    # Step 4: 验证与输出
    print("\n[Step 4] 验证与输出...")
    print(f"  c_dataset 形状: {c_dataset.shape}")
    print(f"\n  各标签类别样本数量:")
    print(c_dataset["label"].value_counts())

    # 保存处理后的数据
    c_dataset.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  已保存至: {OUTPUT_FILE}")
    print("=" * 60)

    return c_dataset


if __name__ == "__main__":
    final_dataset = main()
