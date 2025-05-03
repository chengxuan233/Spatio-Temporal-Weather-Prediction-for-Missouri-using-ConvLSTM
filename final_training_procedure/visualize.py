import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from final_train import DeeperConvLSTMNet, load_split_pt_chunks, normalize, denormalize, init_normalization

# === 模型与数据路径 ===
MODEL_PATH = "training_logs_dynamic/final_model_20250501_071627_L1.pt"
HYPERPARAM_PATH = "best_hparams.json"
SPLIT_NAME = "test_sorted_pt_chunks"
CSV_PATH = "G:/stream_split_dataset_small/test_sorted.csv"

# === 加载测试集张量和原始 CSV ===
tensor = load_split_pt_chunks(SPLIT_NAME)
init_normalization(tensor, output_channels=2)

csv_df = pd.read_csv(CSV_PATH)
H, W = tensor.shape[-2:]

# === 设置预测帧 ===
sample_index = 137  # 可以自行设置
sequence_length = 12
output_channels = 2

x_sample = tensor[sample_index - sequence_length:sample_index].unsqueeze(0)
y_true = tensor[sample_index, 0:output_channels]  # [C, H, W]

# === 模型构建 ===
import json
with open(HYPERPARAM_PATH) as f:
    h = json.load(f)

C = tensor.shape[1]
model = DeeperConvLSTMNet(C, hidden_channels=h["hidden_size"], output_channels=output_channels,
                          height=H, width=W, kernel_size=h["kernel_size"])
model.eval()
with torch.no_grad():
    _ = model(tensor[:sequence_length].unsqueeze(0))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

# === 获取真实日期信息 + 网格构建 ===
row_idx = sample_index * H * W
csv_slice = csv_df.iloc[row_idx: row_idx + H * W]
date_row = csv_slice.iloc[0]
plot_time = f"Date: {int(date_row.year)}-{int(date_row.month):02d}-{int(date_row.day):02d}"

# 构建 lat/lon 网格
lat_grid = csv_slice['lat'].values.reshape(H, W)
lon_grid = csv_slice['lon'].values.reshape(H, W)

# === 预测 ===
with torch.no_grad():
    pred = model(x_sample)[0]  # [C, H, W]

# === 从完整 CSV 中构建真实值图像 ===
channel_matrices = []
for col in ["prcp", "tavg"]:
    mask = (
        (csv_df['year'] == date_row.year)
        & (csv_df['month'] == date_row.month)
        & (csv_df['day'] == date_row.day)
    )
    full_slice = csv_df[mask]
    pivot = full_slice.pivot(index="lat", columns="lon", values=col)
    pivot = pivot.sort_index(ascending=False)
    grid = pivot.values
    channel_matrices.append(grid)

lat_grid = pivot.index.values[:, None].repeat(grid.shape[1], axis=1)
lon_grid = pivot.columns.values[None, :].repeat(grid.shape[0], axis=0)

y_true_csv = np.stack(channel_matrices)  # shape: [2, H, W]
y_true_denorm = torch.tensor(y_true_csv)
pred_denorm = denormalize(pred.unsqueeze(0))[0]
# 对 prcp 通道做非负裁剪
pred_denorm[0] = pred_denorm[0].clamp(min=0)


# === 经纬度坐标轴准备 ===
x_ticks = np.linspace(0, W - 1, 5, dtype=int)
y_ticks = np.linspace(0, H - 1, 5, dtype=int)
x_labels = [f"{lon_grid[0, i]:.1f}" for i in x_ticks]
y_labels = [f"{lat_grid[j, 0]:.1f}" for j in y_ticks]

# === 可视化 ===
SAVE_DIR = "visual_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for i in range(output_channels):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title(f"True Channel {i} | {plot_time}")
    im1 = plt.imshow(np.flipud(y_true_denorm[i].cpu()), cmap="viridis")
    plt.xticks(x_ticks, x_labels)
    plt.yticks(y_ticks, y_labels)
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title(f"Predicted Channel {i} | {plot_time}")
    im2 = plt.imshow(pred_denorm[i].cpu(), cmap="viridis")
    plt.xticks(x_ticks, x_labels)
    plt.yticks(y_ticks, y_labels)
    plt.colorbar()

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f"compare_channel_{i}_step{sample_index}_{timestamp}.png")
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Saved: {save_path}")
