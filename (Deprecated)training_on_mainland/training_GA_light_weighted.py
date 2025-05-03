import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import glob
import csv
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold
import json
import time
import gc

# === Global Configuration ===
INPUT_CHANNELS = 13
TARGET_CHANNEL_INDEXES = [0, 1]  # Only predict prcp and tavg
SEQUENCE_LENGTH = 6
OUTPUT_DIR = "training_outputs_GA_log"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Training Argument Configuration ===
class Args:
    train_dir = "G:/stream_split_dataset/train_sorted_pt_masked_chunks"
    val_dir = "G:/stream_split_dataset/val_sorted_pt_masked_chunks"
    batch_size = 8
    learning_rate = 1e-4
    epochs = 30
    model_save_path = os.path.join(OUTPUT_DIR, "best_model_prcp_tavg_log.pt")
    resume = False
args = Args()

# === Mean/Std Buffers ===
INPUT_MEAN = None
INPUT_STD = None
OUTPUT_MEAN = None
OUTPUT_STD = None

# === Compute Global Mean and Std for Normalization ===
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

# === Dataset Class ===
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

        target_tensor[0] = torch.log1p(target_tensor[0])  # Apply log1p to prcp channel
        target_tensor = (target_tensor - OUTPUT_MEAN.view(-1, 1, 1)) / OUTPUT_STD.view(-1, 1, 1)

        return x_seq, target_tensor

# === ConvLSTM Core Cell ===
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size=kernel_size, padding=padding, bias=bias)

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

# === Full ConvLSTM-based Model ===
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

# === Random Hyperparameter Generator ===
def random_hyperparameters():
    return {
        'batch_size': random.choice([4, 6]),
        'hidden_size': random.choice([192, 256, 320]),
        'kernel_size': random.choice([3, 5]),
        'learning_rate': random.choice([1e-4, 2e-4])
    }

# === Early Stopping Mechanism ===
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# === GA Evaluation Function with 3-Fold Cross Validation ===
GA_SAMPLE_SIZE = 10

def evaluate_model(hparams, dataset):
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    val_losses = []
    for train_idx, val_idx in kfold.split(dataset):
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=hparams['batch_size'], shuffle=True, num_workers=2)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=hparams['batch_size'], shuffle=False, num_workers=2)

        model = ConvLSTMNet(INPUT_CHANNELS, hparams['hidden_size'], len(TARGET_CHANNEL_INDEXES), kernel_size=hparams['kernel_size']).to("cuda")
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])
        criterion = nn.HuberLoss()

        for _ in range(5):
            model.train()
            for x_seq, y in train_loader:
                x_seq, y = x_seq.cuda(), y.cuda()
                loss = criterion(model(x_seq), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        total_val_loss = sum(criterion(model(x_seq.cuda()), y.cuda()).item() for x_seq, y in val_loader)
        val_losses.append(total_val_loss / len(val_loader))

        del model
        torch.cuda.empty_cache()
        gc.collect()

    return np.mean(val_losses)

# === Genetic Algorithm Core ===
def genetic_algorithm(dataset, generations=4, population_size=5):
    population = [random_hyperparameters() for _ in range(population_size)]
    scores = [evaluate_model(ind, dataset) for ind in population]

    for gen in range(generations):
        sorted_population = [x for _, x in sorted(zip(scores, population))]
        top_half = sorted_population[:population_size // 2]
        new_population = top_half.copy()

        while len(new_population) < population_size:
            parent = random.choice(top_half)
            child = parent.copy()
            mutate_key = random.choice(list(child.keys()))
            child[mutate_key] = random_hyperparameters()[mutate_key]
            new_population.append(child)

        population = new_population
        scores = [evaluate_model(ind, dataset) for ind in population]

    best_hparams = population[np.argmin(scores)]
    with open("best_hparams.json", "w") as f:
        json.dump(best_hparams, f, indent=2)
    print("💾 Best hyperparameters saved to best_hparams_log.json")
    return best_hparams
