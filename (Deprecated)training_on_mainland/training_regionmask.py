# ✅ Simplified and corrected training script: predicts prcp and tavg (channels 0 and 1)
# ✅ ConvLSTM layers reduced to 4; attention module removed
# ✅ Uses Weighted HuberLoss for robust, region-aware training
# ✅ Includes GA-based hyperparameter search, 3-Fold CV, EarlyStopping, and best model saving

# === Suppress FutureWarnings ===
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import glob
import csv
import random
import json
import time
import gc

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold

# === Global Parameters ===
INPUT_CHANNELS = 14  # Updated from 13 to 14: includes soft_region_mask
TARGET_CHANNEL_INDEXES = [0, 1]  # Predict only prcp and tavg
SEQUENCE_LENGTH = 6
OUTPUT_DIR = "training_outputs_GA_log"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Training Settings ===
class Args:
    train_dir = "G:/stream_split_dataset/train_sorted_pt_masked_chunks"
    val_dir = "G:/stream_split_dataset/val_sorted_pt_masked_chunks"
    batch_size = 8
    learning_rate = 1e-4
    epochs = 40
    model_save_path = os.path.join(OUTPUT_DIR, "best_model_prcp_tavg_log.pt")
    resume = False
args = Args()

# === Global Normalization Buffers ===
INPUT_MEAN = None
INPUT_STD = None
OUTPUT_MEAN = None
OUTPUT_STD = None

# === Region-based Weight Map (Loaded once globally) ===
region_mask = torch.from_numpy(np.load("region_soft_mask.npy")).float().cuda()

# === Custom Weighted Huber Loss ===
class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target, weight):
        error = pred - target
        abs_error = torch.abs(error)
        quadratic = torch.minimum(abs_error, torch.tensor(self.delta, device=error.device))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        weighted_loss = loss * weight.unsqueeze(1)
        return weighted_loss.mean()

# === Compute Dataset-wide Mean/Std ===
def compute_mean_std(file_list, channels, mode='input'):
    all_sum = torch.zeros(channels)
    all_sqsum = torch.zeros(channels)
    count = 0
    for file in tqdm(file_list, desc=f"⏳ Computing {mode} mean/std", ncols=100):
        tensor = torch.load(file, map_location='cpu')
        if tensor.ndim == 4:
            tensor = tensor[0]
        tensor = tensor[:INPUT_CHANNELS] if mode == 'input' else tensor[TARGET_CHANNEL_INDEXES]
        for c in range(tensor.shape[0]):
            all_sum[c] += tensor[c].sum()
            all_sqsum[c] += (tensor[c] ** 2).sum()
        count += tensor[0].numel()
    mean = all_sum / count
    std = (all_sqsum / count - mean ** 2).sqrt()
    return mean, std

# === Dataset Loader Class ===
class ClimatePTChunkDataset(Dataset):
    def __init__(self, pt_file_list):
        self.pt_files = pt_file_list
        self.seqlen = SEQUENCE_LENGTH

    def __len__(self):
        return len(self.pt_files) - self.seqlen

    def __getitem__(self, idx):
        seq_files = self.pt_files[idx: idx + self.seqlen]
        x_seq = []
        for file in seq_files:
            tensor = torch.load(file, map_location='cpu')
            if tensor.ndim == 4:
                tensor = tensor[0]
            x_seq.append(tensor[:INPUT_CHANNELS].unsqueeze(0))
        x_seq = torch.cat(x_seq, dim=0)
        x_seq = (x_seq - INPUT_MEAN.view(1, -1, 1, 1)) / INPUT_STD.view(1, -1, 1, 1)

        target_tensor = torch.load(self.pt_files[idx + self.seqlen], map_location='cpu')
        if target_tensor.ndim == 4:
            target_tensor = target_tensor[0]
        target_tensor = target_tensor[TARGET_CHANNEL_INDEXES]
        target_tensor[0] = torch.log1p(target_tensor[0])
        target_tensor = (target_tensor - OUTPUT_MEAN.view(-1, 1, 1)) / OUTPUT_STD.view(-1, 1, 1)

        return x_seq, target_tensor

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

# === ConvLSTM Model ===
class ConvLSTMNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, kernel_size=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.convlstm1 = ConvLSTMCell(256, hidden_channels, kernel_size)
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
            nn.Conv2d(64, output_channels, 3, padding=1)
        )

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        h1 = c1 = h2 = c2 = h3 = c3 = h4 = c4 = None
        for t in range(T):
            x = self.encoder(x_seq[:, t])
            if h1 is None:
                h1 = torch.zeros(B, self.convlstm1.conv.out_channels // 4, x.shape[2], x.shape[3], device=x.device)
                c1 = torch.zeros_like(h1)
                h2 = c2 = h3 = c3 = h4 = c4 = torch.zeros_like(h1)
            h1, c1 = self.convlstm1(x, h1, c1)
            h2, c2 = self.convlstm2(h1, h2, c2)
            h3, c3 = self.convlstm3(h2, h3, c3)
            h4, c4 = self.convlstm4(h3, h4, c4)
        return self.decoder(h4)

# === Genetic Algorithm Search ===
def random_hyperparameters():
    return {
        'batch_size': random.choice([4, 6]),
        'hidden_size': random.choice([192, 256, 320]),
        'kernel_size': random.choice([3, 5]),
        'learning_rate': random.choice([1e-4, 2e-4])
    }
