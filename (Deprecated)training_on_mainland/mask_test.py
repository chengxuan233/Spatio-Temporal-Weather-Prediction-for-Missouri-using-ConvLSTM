# ✅ ConvLSTM Inference Script: Loads model prediction and saves CSV output with lat/lon mapping

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# === Path Configuration ===
TEST_DIR = r"G:/stream_split_dataset/test_sorted_pt_masked_chunks"
MODEL_DIR = r"E:/pythonProject/527 Climate Project/Training Process/actual-training/training_outputs_GA_log"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model_prcp_tavg_log.pt")
SAVE_CSV_PATH = "predicted_output_sample0.csv"
LAT_PATH = r"land_mask_lats.npy"
LON_PATH = r"land_mask_lons.npy"

SEQUENCE_LENGTH = 6
INPUT_CHANNELS = 14
TARGET_CHANNEL_INDEXES = [0, 1]

# === Load Normalization Stats and Hyperparameters ===
INPUT_MEAN = torch.load(os.path.join(MODEL_DIR, "input_mean.pt"))
INPUT_STD = torch.load(os.path.join(MODEL_DIR, "input_std.pt"))
OUTPUT_MEAN = torch.load(os.path.join(MODEL_DIR, "output_mean.pt"))
OUTPUT_STD = torch.load(os.path.join(MODEL_DIR, "output_std.pt"))

with open(os.path.join(MODEL_DIR, "best_hparams.json"), "r") as f:
    hparams = json.load(f)

ENCODER_OUT_CHANNELS = 256

# === ConvLSTM Cell ===
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding, bias=bias)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_output, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

# === ConvLSTM-based Model ===
class ConvLSTMNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, kernel_size=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, ENCODER_OUT_CHANNELS, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.convlstm1 = ConvLSTMCell(ENCODER_OUT_CHANNELS, hidden_channels, kernel_size)
        self.convlstm2 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
        self.convlstm3 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
        self.convlstm4 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, len(TARGET_CHANNEL_INDEXES), 3, padding=1)
        )

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        h1 = c1 = h2 = c2 = h3 = c3 = h4 = c4 = None
        for t in range(T):
            x = self.encoder(x_seq[:, t])
            if h1 is None:
                h1 = torch.zeros(B, hparams["hidden_size"], x.shape[2], x.shape[3], device=x.device)
                c1 = torch.zeros_like(h1)
                h2 = c2 = h3 = c3 = h4 = c4 = torch.zeros_like(h1)
            h1, c1 = self.convlstm1(x, h1, c1)
            h2, c2 = self.convlstm2(h1, h2, c2)
            h3, c3 = self.convlstm3(h2, h3, c3)
            h4, c4 = self.convlstm4(h3, h4, c4)
        return self.decoder(h4)

# === Dataset for Sequential Testing ===
class TestPTDataset(torch.utils.data.Dataset):
    def __init__(self, pt_files):
        self.pt_files = pt_files

    def __len__(self):
        return len(self.pt_files) - SEQUENCE_LENGTH

    def __getitem__(self, idx):
        files = self.pt_files[idx:idx + SEQUENCE_LENGTH]
        x_seq = []
        for file in files:
            t = torch.load(file)[0][:INPUT_CHANNELS]
            x_seq.append(t.unsqueeze(0))
        x_seq = torch.cat(x_seq, dim=0)
        x_seq = (x_seq - INPUT_MEAN.view(1, -1, 1, 1)) / INPUT_STD.view(1, -1, 1, 1)
        return x_seq, self.pt_files[idx + SEQUENCE_LENGTH]

# === Inference Pipeline ===
def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvLSTMNet(
        input_channels=INPUT_CHANNELS,
        hidden_channels=hparams["hidden_size"],
        output_channels=len(TARGET_CHANNEL_INDEXES),
        kernel_size=hparams["kernel_size"]
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    pt_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.pt")))
    dataset = TestPTDataset(pt_files)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    lats = np.load(LAT_PATH)
    lons = np.load(LON_PATH)

    with torch.no_grad():
        for i, (x_seq, target_path) in enumerate(loader):
            x_seq = x_seq.to(device)
            pred = model(x_seq).squeeze(0)
            pred = pred * OUTPUT_STD.view(-1, 1, 1).to(device) + OUTPUT_MEAN.view(-1, 1, 1).to(device)

            if i == 0:
                H, W = pred.shape[1], pred.shape[2]
                rows = []
                for i_ in range(H):
                    for j_ in range(W):
                        rows.append({
                            "i": i_,
                            "j": j_,
                            "lat": float(lats[i_]),
                            "lon": float(lons[j_]),
                            "prcp": pred[0, i_, j_].item(),
                            "tavg": pred[1, i_, j_].item()
                        })
                df = pd.DataFrame(rows)
                df.to_csv(SAVE_CSV_PATH, index=False)
                print(f"✅ Saved prediction to {SAVE_CSV_PATH}")
                break

if __name__ == "__main__":
    run_inference()
