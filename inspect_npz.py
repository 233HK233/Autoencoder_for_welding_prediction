import numpy as np
import pandas as pd
from pathlib import Path

# === 配置部分 ===
# 你的 .npz 文件路径
NPZ_PATH = "/home/huang_kai/wind_turbine_phm/PHOENIX-main/autoencoder_benchmark/Data/processed_data/weld_seam_windows.npz"
# 输出的检查用 CSV 路径
OUTPUT_CSV = "autoencoder_benchmark/Data/processed_data/data_check_preview.csv"
# 想要查看多少行样本（前 N 个）
PREVIEW_ROWS = 100


def inspect_data():
    if not Path(NPZ_PATH).exists():
        print(f"错误：找不到文件 {NPZ_PATH}")
        return

    # 1. 加载 .npz
    data = np.load(NPZ_PATH)
    print(f"成功加载文件，包含的数组键值: {list(data.files)}\n")

    # 2. 提取主要数据
    X_train = data["X_train_full"]
    y_train = data["y_train"]
    seam_train = data["seam_id_train"]
    start_train = data["start_idx_train"] if "start_idx_train" in data.files else None

    X_test = data["X_test_full"]
    y_test = data["y_test"]
    seam_test = data["seam_id_test"]
    start_test = data["start_idx_test"] if "start_idx_test" in data.files else None

    seam_names = data["seam_name_order"]

    # 3. 打印统计报告 (最核心的检查)
    print("=" * 40)
    print(" [1] 数据集形状检查")
    print("=" * 40)
    print(f"训练集 X: {X_train.shape} (样本数, 窗口长, 特征数)")
    print(f"测试集 X: {X_test.shape}")
    if start_train is not None and start_test is not None:
        print(f"start_idx_train: {start_train.shape}, start_idx_test: {start_test.shape}")
    print("-" * 40)

    print("\n" + "=" * 40)
    print(" [2] 覆盖率与分布检查 (验证配额逻辑)")
    print("=" * 40)

    def print_dist(name, y_arr, s_arr):
        df = pd.DataFrame({"seam_id": s_arr, "label": y_arr})
        # 把数字 ID 映射回焊缝名字 (a01, b01...)
        df["seam_name"] = df["seam_id"].apply(lambda x: seam_names[x])

        # 交叉表统计
        print(f"\n>>> {name} 分布表 (行=焊缝, 列=Label):")
        print(pd.crosstab(df["seam_name"], df["label"]))

    print_dist("训练集 (Train)", y_train, seam_train)
    print_dist("测试集 (Test)", y_test, seam_test)

    # 4. 导出 CSV 供 Excel 查看
    print("\n" + "=" * 40)
    print(f" [3] 生成可视化预览 (前 {PREVIEW_ROWS} 条)")
    print("=" * 40)

    # 取前 N 个训练样本
    sample_X = X_train[:PREVIEW_ROWS]
    print(f"样本 X 形状: {sample_X.shape} (样本数, 窗口长, 特征数)")
    print("样本 X :", sample_X[:100])
    sample_y = y_train[:PREVIEW_ROWS]
    print(f"样本 y 形状: {sample_y.shape} (样本数,)")
    print("样本 y :", sample_y[:100])
    sample_s = seam_train[:PREVIEW_ROWS]

    # 为了方便查看，我们计算每个窗口内，每个特征的“均值”
    # shape 变从 (N, window_size, 18) -> (N, 18)
    X_flattened = sample_X.mean(axis=1)

    columns = [f"Feat_{i}_Mean" for i in range(X_flattened.shape[1])]
    df_view = pd.DataFrame(X_flattened, columns=columns)

    df_view.insert(0, "Label", sample_y)
    df_view.insert(0, "Seam_Name", [seam_names[i] for i in sample_s])

    df_view.to_csv(OUTPUT_CSV, index=False)
    print(f"已保存预览文件到: {OUTPUT_CSV}")
    window_size = X_train.shape[1]
    print(f"注意：为了方便查看，CSV中显示的是该窗口内 {window_size} 个时间步的【平均值】。")

    # 5. 可选：打印 start_idx 预览
    if start_train is not None:
        print("\nstart_idx_train 预览:", start_train[:PREVIEW_ROWS])


if __name__ == "__main__":
    inspect_data()
