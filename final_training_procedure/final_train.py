import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from deap import base, creator, tools, algorithms
import random

HYPERPARAM_FILE = "best_hparams.json"
DATASET_DIR = "G:/stream_split_dataset_small"
LOG_DIR = "training_logs_dynamic"
os.makedirs(LOG_DIR, exist_ok=True)

SEARCH_CONFIG = {
    "hidden_size_choices": [192, 256],
    "lr_choices": [1e-4, 5e-4, 1e-3],
    "kernel_choices": [3, 5, 7],
    "population_size": 5,
    "generations": 3
}

# Global normalization buffers
y_mean = None
y_std = None

def init_normalization(tensor, output_channels=2):
    global y_mean, y_std
    all_y = tensor[:, 0:output_channels]  # shape [T, C, H, W]
    y_mean = all_y.mean(dim=(0, 2, 3))    # shape [C]
    y_std = all_y.std(dim=(0, 2, 3))      # shape [C]

def normalize(y):
    return (y - y_mean.to(y.device)[None, :, None, None]) / y_std.to(y.device)[None, :, None, None]

def denormalize(y):
    return y * y_std.to(y.device)[None, :, None, None] + y_mean.to(y.device)[None, :, None, None]

class DeeperConvLSTMNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, height, width, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(1, 3, 3), padding=(0, padding, padding)),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, padding, padding)),
            nn.ReLU(),
        )
        self.convlstm = None
        self.decoder = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_channels * height * width)
        )
        self.H = height
        self.W = width
        self.output_channels = output_channels

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = self.encoder(x)
        _, _, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B, T, -1)
        if self.convlstm is None:
            self.convlstm = nn.LSTM(input_size=64 * H * W, hidden_size=256, batch_first=True).to(x.device)
        x, _ = self.convlstm(x)
        x = x[:, -1, :]
        x = self.decoder(x)
        return x.view(B, self.output_channels, self.H, self.W)

def compute_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy().flatten()
    y_pred = y_pred.detach().cpu().numpy().flatten()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def load_split_pt_chunks(split_name, limit_chunks=None, verbose=True):
    folder = os.path.join(DATASET_DIR, split_name)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Path not found: {folder}")

    all_pt_files = sorted([f for f in os.listdir(folder) if f.endswith(".pt")])
    if not all_pt_files:
        raise ValueError(f"No .pt files found in folder: {folder}")

    if limit_chunks is not None:
        all_pt_files = all_pt_files[:limit_chunks]

    if verbose:
        print(f"Loading {len(all_pt_files)} files from: {folder}")

    tensors = [torch.load(os.path.join(folder, fname)) for fname in all_pt_files]
    X = torch.cat(tensors, dim=0)
    return X

def build_dataset(tensor, output_channels=2, sequence_length=12, forecast_horizon=1):
    x_seq, y_seq = [], []
    for i in range(tensor.shape[0] - sequence_length - forecast_horizon + 1):
        x_seq.append(tensor[i:i+sequence_length])
        y_seq.append(tensor[i+sequence_length:i+sequence_length+forecast_horizon, 0:output_channels])
    return TensorDataset(torch.stack(x_seq), torch.stack(y_seq))

def run_genetic_search(train_loader, val_loader, input_channels, output_channels, H, W, device):
    if os.path.exists(HYPERPARAM_FILE):
        with open(HYPERPARAM_FILE, 'r') as f:
            return json.load(f)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("hidden_size", random.choice, SEARCH_CONFIG["hidden_size_choices"])
    toolbox.register("lr", random.choice, SEARCH_CONFIG["lr_choices"])
    toolbox.register("kernel_size", random.choice, SEARCH_CONFIG["kernel_choices"])
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.hidden_size, toolbox.lr, toolbox.kernel_size), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        hs, lr, ks = int(ind[0]), ind[1], int(ind[2])
        model = DeeperConvLSTMNet(input_channels, hs, output_channels, H, W, ks).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.HuberLoss()
        model.train()
        for _ in range(3):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, normalize(yb[:, -1]))
                loss.backward()
                optimizer.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                mae, _ = compute_metrics(denormalize(yb[:, -1]), denormalize(preds))
                val_loss += mae
        return (val_loss / len(val_loader),)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=SEARCH_CONFIG["population_size"])
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=SEARCH_CONFIG["generations"], verbose=True)

    top = tools.selBest(pop, k=1)[0]
    best = {"hidden_size": int(top[0]), "lr": round(top[1], 5), "kernel_size": int(top[2])}
    with open(HYPERPARAM_FILE, 'w') as f:
        json.dump(best, f, indent=2)
    return best

if __name__ == '__main__':
    train_tensor = load_split_pt_chunks("train_sorted_pt_chunks", limit_chunks=10)
    val_tensor = load_split_pt_chunks("val_sorted_pt_chunks")

    _, C, H, W = train_tensor.shape
    sequence_length = 12
    forecast_horizon = 1
    output_channels = 2

    init_normalization(train_tensor, output_channels)

    train_dataset = build_dataset(train_tensor, output_channels, sequence_length, forecast_horizon)
    val_dataset = build_dataset(val_tensor, output_channels, sequence_length, forecast_horizon)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(HYPERPARAM_FILE):
        choice = input("📌 best_hparams.json exists, skip search? (y/n): ").strip().lower()
        if choice == "y":
            with open(HYPERPARAM_FILE, 'r') as f:
                best_hparams = json.load(f)
        else:
            best_hparams = run_genetic_search(train_loader, val_loader, C, output_channels, H, W, device)
    else:
        best_hparams = run_genetic_search(train_loader, val_loader, C, output_channels, H, W, device)

    model = DeeperConvLSTMNet(C, best_hparams['hidden_size'], output_channels, H, W, best_hparams['kernel_size']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_hparams['lr'])
    criterion = nn.L1Loss()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(LOG_DIR, f"final_model_{timestamp}.pt")
    log_path = os.path.join(LOG_DIR, f"final_log_{timestamp}.csv")
    fig_path = os.path.join(LOG_DIR, f"final_curve_{timestamp}.png")

    log_rows = []
    best_val_loss = float('inf')

    for epoch in range(30):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, normalize(yb[:, -1]))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, normalize(yb[:, -1]))
                val_loss += loss.item()
                if loss.item() < best_val_loss:
                    best_val_loss = loss.item()
                    torch.save(model.state_dict(), model_path)

        mae, rmse = compute_metrics(denormalize(yb[:, -1]), denormalize(preds))
        log_rows.append({"Epoch": epoch+1, "Train_Loss": total_loss, "Val_Loss": val_loss, "MAE": mae, "RMSE": rmse})
        print(f"Epoch {epoch+1}, Train Loss: {total_loss:.2f}, Val Loss: {val_loss:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    plt.figure(figsize=(10, 4))
    plt.plot([r['Epoch'] for r in log_rows], [r['Train_Loss'] for r in log_rows], label='Train Loss')
    plt.plot([r['Epoch'] for r in log_rows], [r['Val_Loss'] for r in log_rows], label='Val Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path)
