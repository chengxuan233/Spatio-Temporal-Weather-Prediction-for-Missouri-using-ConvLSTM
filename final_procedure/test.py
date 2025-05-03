import torch
import numpy as np
import json
import pandas as pd
from final_train import DeeperConvLSTMNet, load_split_pt_chunks, normalize, denormalize, init_normalization, compute_metrics
from tqdm import tqdm  # Progress bar for loop tracking
from skimage.metrics import structural_similarity as ssim

# === Path Configuration ===
MODEL_PATH = "training_logs_dynamic/final_model_20250501_071627_L1.pt"
HYPERPARAM_PATH = "best_hparams.json"
SPLIT_NAME = "val_sorted_pt_chunks"

# === Load and normalize input tensor ===
tensor = load_split_pt_chunks(SPLIT_NAME)
init_normalization(tensor, output_channels=2)

sequence_length = 12
forecast_horizon = 1
output_channels = 2
C, H, W = tensor.shape[1:]

# === Load model and apply saved hyperparameters ===
with open(HYPERPARAM_PATH) as f:
    h = json.load(f)

model = DeeperConvLSTMNet(
    C,
    hidden_channels=h["hidden_size"],
    output_channels=output_channels,
    height=H,
    width=W,
    kernel_size=h["kernel_size"]
)
model.eval()

# Dummy forward to initialize LSTM layer (lazy initialization)
with torch.no_grad():
    _ = model(tensor[:sequence_length].unsqueeze(0))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

# === Evaluation Setup ===
thresh = 1e-3  # Precipitation threshold in mm for binary classification

# Metrics storage
channel_maes, channel_rmses = [], []
ssim_list = [[] for _ in range(output_channels)]
pearson_list = [[] for _ in range(output_channels)]
binary_acc_list, precision_list, recall_list, f1_list = [], [], [], []

# === Evaluate model on each sliding window of sequence ===
with torch.no_grad():
    for t in tqdm(range(sequence_length, tensor.shape[0] - forecast_horizon), desc="Full Evaluation"):
        x = tensor[t - sequence_length:t].unsqueeze(0)
        y_true = tensor[t:t + forecast_horizon, 0:output_channels][0]
        pred = model(x)[0]

        # De-normalize outputs
        pred_denorm = denormalize(pred.unsqueeze(0))[0]
        y_true_denorm = denormalize(y_true.unsqueeze(0))[0]
        pred_denorm[0] = pred_denorm[0].clamp(min=0)  # Clamp precipitation (channel 0) to non-negative

        # === Per-channel MAE, RMSE, SSIM, and Pearson Correlation ===
        mae_per_channel = []
        rmse_per_channel = []
        for i in range(output_channels):
            true_i = y_true_denorm[i].cpu().numpy()
            pred_i = pred_denorm[i].cpu().numpy()
            mae = np.mean(np.abs(true_i - pred_i))
            rmse = np.sqrt(np.mean((true_i - pred_i) ** 2))
            mae_per_channel.append(mae)
            rmse_per_channel.append(rmse)
            try:
                ssim_val = ssim(true_i, pred_i, data_range=pred_i.max() - pred_i.min())
            except:
                ssim_val = np.nan
            corr = np.corrcoef(true_i.flatten(), pred_i.flatten())[0, 1]
            ssim_list[i].append(ssim_val)
            pearson_list[i].append(corr)

        channel_maes.append(mae_per_channel)
        channel_rmses.append(rmse_per_channel)

        # === Binary classification metrics for channel 0 (precipitation) ===
        true_mask = (y_true_denorm[0].cpu().numpy() >= thresh).astype(np.uint8)
        pred_mask = (pred_denorm[0].cpu().numpy() >= thresh).astype(np.uint8)
        acc = (true_mask == pred_mask).sum() / true_mask.size
        binary_acc_list.append(acc)

        TP = ((pred_mask == 1) & (true_mask == 1)).sum()
        FP = ((pred_mask == 1) & (true_mask == 0)).sum()
        FN = ((pred_mask == 0) & (true_mask == 1)).sum()

        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

# === Summary Statistics ===
channel_maes = np.array(channel_maes)
channel_rmses = np.array(channel_rmses)

for i in range(output_channels):
    print(f"Channel {i} MAE:  {channel_maes[:, i].mean():.4f}")
    print(f"Channel {i} RMSE: {channel_rmses[:, i].mean():.4f}")
    print(f"Channel {i} SSIM: {np.nanmean(ssim_list[i]):.4f}")
    print(f"Channel {i} Corr: {np.nanmean(pearson_list[i]):.4f}")

print(f"Binary Mask Accuracy (Channel 0, thresh={thresh}): {np.mean(binary_acc_list):.4f}")
print(f"Precision: {np.mean(precision_list):.4f}")
print(f"Recall:    {np.mean(recall_list):.4f}")
print(f"F1 Score:  {np.mean(f1_list):.4f}")

# === Save Per-Channel Metrics as CSV ===
per_channel_df = pd.DataFrame(channel_maes, columns=[f"MAE_channel_{i}" for i in range(output_channels)])
for i in range(output_channels):
    per_channel_df[f"RMSE_channel_{i}"] = channel_rmses[:, i]
    per_channel_df[f"SSIM_channel_{i}"] = ssim_list[i]
    per_channel_df[f"Corr_channel_{i}"] = pearson_list[i]

per_channel_df["BinaryAcc_channel_0"] = binary_acc_list
per_channel_df["Precision_channel_0"] = precision_list
per_channel_df["Recall_channel_0"] = recall_list
per_channel_df["F1_channel_0"] = f1_list
per_channel_df.insert(0, "Index", list(range(sequence_length, sequence_length + len(channel_maes))))
per_channel_df.to_csv("eval_metrics_per_channel.csv", index=False)
print("\ud83d\udcc4 Saved per-channel metrics to eval_metrics_per_channel.csv")
