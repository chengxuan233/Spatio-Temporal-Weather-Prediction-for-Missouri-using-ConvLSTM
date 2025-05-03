# 🔍 test_eval_convlstm.py
# Evaluate the trained model performance (after denormalization) and save outputs

import torch
# Import core libraries for deep learning (PyTorch), file handling, data loading, and visualization
import glob
import os
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from training_all_GA_attention import (
# Import the trained ConvLSTM model, target channel configuration, and normalization function from training module
    ConvLSTMNet, TARGET_CHANNEL_INDEXES, compute_mean_std
)

# ✅ Load configuration
MODEL_PATH = "training_outputs_GA/best_model_11var.pt"
# Set path to the best-trained model, test dataset directory, and file for saving prediction output
TEST_DATA_DIR = "G:/stream_split_dataset/test_sorted_pt_chunks"
BATCH_SIZE = 4
OUTPUT_SAVE_PATH = "training_outputs_GA/prediction_output.npy"

# ✅ Initialize mean and standard deviation
# Initialize as placeholder
INPUT_MEAN = None
# Initialize mean and standard deviation tensors for normalization — required for inverse scaling
INPUT_STD = None
OUTPUT_MEAN = None
OUTPUT_STD = None

if INPUT_MEAN is None or OUTPUT_MEAN is None:
# Compute normalization statistics from test data
# Principle: z = (x - μ) / σ during normalization; prediction must be reversed via x = z * σ + μ
    pt_files = glob.glob(os.path.join(TEST_DATA_DIR, '*.pt'))
    INPUT_MEAN, INPUT_STD = compute_mean_std(pt_files, channels=1, mode='input')
    OUTPUT_MEAN, OUTPUT_STD = compute_mean_std(pt_files, channels=11, mode='target')
    INPUT_MEAN = INPUT_MEAN.view(-1, 1, 1)
    INPUT_STD = INPUT_STD.view(-1, 1, 1)
    OUTPUT_MEAN = OUTPUT_MEAN.view(-1, 1, 1)
    OUTPUT_STD = OUTPUT_STD.view(-1, 1, 1)

# ✅ Redefine Dataset (avoid using global variables)
class ClimatePTChunkDataset(torch.utils.data.Dataset):
# Define a custom PyTorch Dataset for .pt tensor files
# Each item in this dataset corresponds to a weather frame tensor (input and target pair)
    def __init__(self, pt_file_list, input_mean, input_std, output_mean, output_std):
        self.data = []
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean.view(-1, 1, 1)
        self.output_std = output_std.view(-1, 1, 1)
        for pt_file in pt_file_list:
            tensor = torch.load(pt_file, map_location='cpu')
            self.data.append(tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
# Returns: input sequence (x_seq) and output tensor (target_tensor)
# x_seq is normalized using computed mean/std to ensure stable model input
# target_tensor is selected and normalized per TARGET_CHANNEL_INDEXES
        x_seq = self.data[idx][:13]  # shape: [T, C, H, W]
        raw_target = self.data[idx][0]
        print(f"🧪 raw_target shape: {raw_target.shape}")
        target_tensor = torch.index_select(raw_target, dim=0, index=torch.tensor(TARGET_CHANNEL_INDEXES, dtype=torch.long))
        x_seq = (x_seq - self.input_mean) / self.input_std
        output_mean = self.output_mean[TARGET_CHANNEL_INDEXES]
        output_std = self.output_std[TARGET_CHANNEL_INDEXES]
        print(f"🧪 target_tensor shape: {target_tensor.shape}")
        print(f"🧪 output_mean shape: {output_mean.shape}")
        print(f"🧪 output_std shape: {output_std.shape}")
        target_tensor = (target_tensor - output_mean) / output_std
        return x_seq, target_tensor

# ✅ Build test dataset
pt_files = glob.glob(os.path.join(TEST_DATA_DIR, '*.pt'))
test_dataset = ClimatePTChunkDataset(pt_files, INPUT_MEAN, INPUT_STD, OUTPUT_MEAN, OUTPUT_STD)
# Create the test dataset using file list and normalized parameters
# Use DataLoader for batch-wise processing
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ✅ Load model
model = ConvLSTMNet(13, 384, len(TARGET_CHANNEL_INDEXES), kernel_size=3)
# Initialize ConvLSTM model structure with 13 input channels, hidden units=384, output channels=11
# Load trained weights and set to eval mode — disables dropout, fixes batchnorm stats
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# ✅ Make predictions and apply denormalization
all_preds = []
all_targets = []
with torch.no_grad():
# Perform inference without gradient computation for efficiency
# Denormalize both prediction and target back to original scale
    for x_seq, y in test_loader:
        pred = model(x_seq)
        # Denormalization
        pred = pred * OUTPUT_STD.view(-1,1,1) + OUTPUT_MEAN.view(-1,1,1)
        y = y * OUTPUT_STD.view(-1,1,1) + OUTPUT_MEAN.view(-1,1,1)
        all_preds.append(pred.numpy())
        all_targets.append(y.numpy())

# ✅ Save predictions
all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)
np.save(OUTPUT_SAVE_PATH, {
# Store predictions and ground truths in .npy format
# Purpose: enables fast reloading for post-analysis or visualization
    'predictions': all_preds,
    'targets': all_targets
})
print(f"✅ Prediction output saved, total {len(all_preds)} samples.")

# ✅ Calculate RMSE and MAE for each variable
num_vars = all_preds.shape[1]
rmse_per_var = []
mae_per_var = []
for i in range(num_vars):
# Evaluate each variable separately using RMSE (penalizes large errors) and MAE (averages absolute deviation)
# Useful for understanding which variables the model predicts accurately vs. poorly
    pred_i = all_preds[:, i]
    target_i = all_targets[:, i]
    rmse = np.sqrt(np.mean((pred_i - target_i) ** 2))
    mae = np.mean(np.abs(pred_i - target_i))
    rmse_per_var.append(rmse)
    mae_per_var.append(mae)

var_names = [
    "tavg", "prcp", "Swnet", "Lwnet", "Qg", "Evap", "Qsm",
    "SnowDepth", "SoilMoist", "TVeg", "TWS"
]

print(" Evaluation results for each variable:")
import csv
with open("training_outputs_GA/metrics_summary.csv", "w", newline='') as f:
# Save all variable-wise RMSE and MAE scores to a CSV file for report and reproducibility
    writer = csv.writer(f)
    writer.writerow(["Variable", "RMSE", "MAE"])
    for i, (r, m) in enumerate(zip(rmse_per_var, mae_per_var)):
        name = var_names[i] if i < len(var_names) else f"var{i}"
        print(f"{name:12s} ▶ RMSE: {r:.4f} | MAE: {m:.4f}")
        writer.writerow([name, r, m])

print("📁 Saved error metrics to training_outputs_GA/metrics_summary.csv")

# ✅ Optional: visualize a prediction vs target image for a variable
idx = 0  # Sample Index
var = 0  # Index of the variable (e.g., 0 = tavg)
pred_map = all_preds[idx, var]
target_map = all_targets[idx, var]
plt.figure(figsize=(10,4))
# Optional visualization:
# Display predicted vs actual heatmap images for a chosen variable and sample index
# Useful for spatial error analysis
plt.subplot(1,2,1)
plt.imshow(pred_map, cmap='jet')
plt.title('Predicted')
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(target_map, cmap='jet')
plt.title('Target')
plt.colorbar()
plt.suptitle(f"Variable {var} Sample {idx}")
plt.savefig("training_outputs_GA/sample_comparison.png")
print("🖼️ Sample visualization saved.")
