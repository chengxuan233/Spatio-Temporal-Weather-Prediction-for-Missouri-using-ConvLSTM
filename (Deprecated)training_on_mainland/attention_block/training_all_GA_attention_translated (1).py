# 🚀 ConvLSTM training framework with GA search and 5-Fold CV

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

# ✨ Configuration Area
INPUT_CHANNELS = 13
TARGET_CHANNEL_INDEXES = list(range(0, 11))  # 11 output features
SEQUENCE_LENGTH = 6
OUTPUT_DIR = "training_outputs_GA"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class Args:
    train_dir = "G:/stream_split_dataset/train_sorted_pt_chunks"
    val_dir = "G:/stream_split_dataset/val_sorted_pt_chunks"
    batch_size = 8
    learning_rate = 1e-4
    epochs = 30
    model_save_path = os.path.join(OUTPUT_DIR, "best_model_11var.pt")
    resume = False
args = Args()

# 🧩 Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale

# 🧩 Dataset
# 🧩 Dynamically loaded ClimatePTChunkDataset (memory safe and uniform time length)
# ✅ Control Dataset logging (only first 5 samples printed)
# 📦 Normalization parameters (example, can be tuned)
# 📦 Placeholders for dynamic mean and std computation
INPUT_MEAN = None
INPUT_STD = None
OUTPUT_MEAN = None
OUTPUT_STD = None

def compute_mean_std(file_list, channels, mode='input'):
    all_sum = torch.zeros(channels)
    all_sqsum = torch.zeros(channels)
    count = 0
    for file in tqdm(file_list, desc=f"⏳ Computing {mode} mean/std", ncols=100):
        tensor = torch.load(file, map_location='cpu')
        if tensor.ndim == 4:
            tensor = tensor[0]
        if mode == 'input':
            tensor = tensor[:1]  # Use only the first channel for x_seq
        else:
            tensor = tensor[:11]  # Use the first 11 variables as targets
        for c in range(tensor.shape[0]):
            all_sum[c] += tensor[c].sum()
            all_sqsum[c] += (tensor[c] ** 2).sum()
        count += tensor[0].numel()
    mean = all_sum / count
    std = (all_sqsum / count - mean ** 2).sqrt()
    return mean, std


class ClimatePTChunkDataset(Dataset):
    def __init__(self, pt_file_list):
        self.pt_files = pt_file_list
        self.seqlen = SEQUENCE_LENGTH
        self.verbose_limit = 5
        print(f"📂 Found {len(self.pt_files)} PT files for training/validation.")

    def __len__(self):
        return len(self.pt_files) - self.seqlen

    def __getitem__(self, idx):
        seq_files = self.pt_files[idx : idx + self.seqlen]
        x_seq = []
        for i, file in enumerate(seq_files):
            if idx < self.verbose_limit:
                print(f"🔄 Loading sequence: {file}")
            tensor = torch.load(file, map_location='cpu')
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            tensor = tensor[:1]
            x_seq.append(tensor)
        x_seq = torch.cat(x_seq, dim=0)

        target_file = self.pt_files[idx + self.seqlen]
        if idx < self.verbose_limit:
            print(f"🎯 Loading target: {target_file}")
        target_tensor = torch.load(target_file, map_location='cpu')
        if target_tensor.ndim == 3:
            target_tensor = target_tensor[:11]
        elif target_tensor.ndim == 4:
            target_tensor = target_tensor[0, :11]
        # Simple normalization (can be replaced by per-channel mean/std later)
        x_seq = (x_seq - INPUT_MEAN.view(-1, 1, 1)) / INPUT_STD.view(-1, 1, 1)
        target_tensor = (target_tensor - OUTPUT_MEAN.view(-1, 1, 1)) / OUTPUT_STD.view(-1, 1, 1)

        return x_seq, target_tensor


# 🚀 DataLoader wrapper with progress bar
def tqdm_dataloader(loader, desc=""):
    for batch in tqdm(loader, desc=desc, leave=False, ncols=100):
        yield batch

# 🚀 Updated ConvLSTMNet with GA-searchable hyperparameters and additional ConvLSTM layers
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return torch.relu(x + self.block(x))

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size=kernel_size, padding=padding, bias=bias
        )

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

# 🧠 Attention Module
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ConvLSTMNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, kernel_size=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            SEBlock(64),
            ResidualBlock(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            nn.MaxPool2d(2)
        )
        self.convlstm1 = ConvLSTMCell(128, hidden_channels, kernel_size)
        self.convlstm2 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
        self.convlstm3 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
        self.convlstm4 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
        self.convlstm5 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
        self.convlstm6 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
        self.attention = AttentionGate(F_g=hidden_channels, F_l=hidden_channels, F_int=hidden_channels // 2)
        self.decoder = nn.Sequential(
            SEBlock(hidden_channels),
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
        h1 = c1 = h2 = c2 = h3 = c3 = h4 = c4 = h5 = c5 = h6 = c6 = None
        for t in range(T):
            x = self.encoder(x_seq[:, t])
            if h1 is None:
                h1 = torch.zeros(B, self.convlstm1.conv.out_channels // 4, x.shape[2], x.shape[3], device=x.device)
                c1 = torch.zeros_like(h1)
                h2 = torch.zeros_like(h1)
                c2 = torch.zeros_like(h1)
                h3 = torch.zeros_like(h1)
                c3 = torch.zeros_like(h1)
                h4 = torch.zeros_like(h1)
                c4 = torch.zeros_like(h1)
                h5 = torch.zeros_like(h1)
                c5 = torch.zeros_like(h1)
                h6 = torch.zeros_like(h1)
                c6 = torch.zeros_like(h1)
            h1, c1 = self.convlstm1(x, h1, c1)
            h2, c2 = self.convlstm2(h1, h2, c2)
            h3, c3 = self.convlstm3(h2, h3, c3)
            h4, c4 = self.convlstm4(h3, h3, c3)
            h5, c5 = self.convlstm5(h4, h4, c4)
            h6, c6 = self.convlstm6(h5, h5, c5)
        att = self.attention(h6, h6)
        out = self.decoder(att)
        return out



# 🚀 3-Fold Cross Validation and Early Stopping
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

# 🚀 Genetic Algorithm Hyperparameter Search
def random_hyperparameters():
    return {
        'batch_size': random.choice([4, 6]),
        'hidden_size': random.choice([256, 320, 384, 448]),
        'kernel_size': random.choice([3]),  # Default using 3
        'learning_rate': random.choice([1e-4, 2e-4])
    }
# ✅ Save the best GA hyperparameter combination to file

def save_best_hparams(best_hparams, path="best_hparams.json"):
    with open(path, "w") as f:
        json.dump(best_hparams, f, indent=4)
    print(f"💾 The best hyperparameters have been saved to {path}")


import gc

# 🚀  Genetic Algorithm Hyperparameter Search Setting
# ✅ Recommended: 2 generations × 3 individuals × 3 folds × 5 epochs (1-2 hours)

def evaluate_model(hparams, dataset):
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)  # Originally 5-fold, now 3-fold
    val_losses = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=hparams['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=hparams['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

        model = ConvLSTMNet(INPUT_CHANNELS, hparams['hidden_size'], len(TARGET_CHANNEL_INDEXES), kernel_size=hparams['kernel_size']).to("cuda")
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])
        criterion = nn.MSELoss()

        for epoch in range(5):  # 🔻 Reduce the number of training epochs in the evaluation phase from 10 to 5
            model.train()
            for x_seq, y in train_loader:
                x_seq, y = x_seq.cuda(), y.cuda()
                pred = model(x_seq)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_seq, y in val_loader:
                x_seq, y = x_seq.cuda(), y.cuda()
                pred = model(x_seq)
                total_val_loss += criterion(pred, y).item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # 🚀 Release GPU memory after each fold
        del model
        torch.cuda.empty_cache()
        gc.collect()

    return np.mean(val_losses)

# 🎯 GA hyperparameters: fine-tuned around default values (retain search space)
def genetic_algorithm(dataset, generations=2, population_size=3):
    population = [random_hyperparameters() for _ in range(population_size)]
    scores = []
    for individual in population:
        score = evaluate_model(individual, dataset)
        scores.append(score)

    for gen in range(generations):
        print(f"Generation {gen+1}")
        sorted_population = [x for _, x in sorted(zip(scores, population))]
        top_half = sorted_population[:population_size//2]

        new_population = top_half.copy()
        while len(new_population) < population_size:
            parent = random.choice(top_half)
            child = parent.copy()
            mutation = random.choice(list(child.keys()))
            child[mutation] = random_hyperparameters()[mutation]
            new_population.append(child)

        population = new_population
        scores = []
        for individual in population:
            score = evaluate_model(individual, dataset)
            scores.append(score)

    best_idx = np.argmin(scores)
    save_best_hparams(population[best_idx])
    return population[best_idx]

# 🚀 Final training process using best GA hyperparameters
def final_train(args, best_hparams):
    global INPUT_MEAN, INPUT_STD, OUTPUT_MEAN, OUTPUT_STD
    print("Loading final training datasets...")
    # 🧠 Compute mean and std before training (only once)
    train_file_list = glob.glob(os.path.join(args.train_dir, '*.pt'))
    if INPUT_MEAN is None or INPUT_STD is None:
        INPUT_MEAN, INPUT_STD = compute_mean_std(train_file_list, channels=1, mode='input')
        OUTPUT_MEAN, OUTPUT_STD = compute_mean_std(train_file_list, channels=11, mode='target')
        print(f"📊 Input mean: {INPUT_MEAN.tolist()} 📊 Input std: {INPUT_STD.tolist()}")
        print(f"📊 Output mean: {OUTPUT_MEAN.tolist()} 📊 Output std: {OUTPUT_STD.tolist()}")

    train_dataset = ClimatePTChunkDataset(glob.glob(os.path.join(args.train_dir, '*.pt')))
    val_dataset = ClimatePTChunkDataset(glob.glob(os.path.join(args.val_dir, '*.pt')))

    train_loader = DataLoader(train_dataset, batch_size=best_hparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_hparams['batch_size'], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvLSTMNet(INPUT_CHANNELS, best_hparams['hidden_size'], len(TARGET_CHANNEL_INDEXES), kernel_size=best_hparams['kernel_size']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_hparams['learning_rate'])
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    early_stopping = EarlyStopping(patience=7)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    start_time = time.time()
    log_file = os.path.join(OUTPUT_DIR, "final_training_log.csv")
    with open(log_file, mode='w', newline='') as logfile:
        writer = csv.writer(logfile)
        writer.writerow(["Epoch", "Train Loss", "Val Loss"])

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            for x_seq, y in train_loader:
                x_seq, y = x_seq.to(device), y.to(device)
                pred = model(x_seq)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_seq, y in val_loader:
                    x_seq, y = x_seq.to(device), y.to(device)
                    pred = model(x_seq)
                    val_loss += criterion(pred, y).item()
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            scheduler.step(avg_val_loss)
            early_stopping(avg_val_loss)

            print(f"Epoch {epoch+1}/{args.epochs} Train Loss: {avg_train_loss:.6f} Val Loss: {avg_val_loss:.6f}")
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss])

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), args.model_save_path)
                print("📦 Best model saved!")

            # Save checkpoint every 10 epochs
            if (epoch+1) % 10 == 0:
                checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch+1}.pt")
                torch.save(model.state_dict(), checkpoint_path)

            if early_stopping.early_stop:
                print("⏹️ Early stopping triggered!")
                break

    # Plot training curves
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, "final_training_curve.png"))
    print("📈 Training curve saved!")

    # Print total training time
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"⏰ Total training time: {elapsed_minutes:.2f} minutes")

# 🚀 Main program entry
if __name__ == "__main__":
    print("🔍 Use existing hyperparameters? (y/n)")
    use_existing = input("➡️ Enter y to skip GA and load best_hparams.json: ").strip().lower()

    if use_existing == 'y':
        best_hparams_path = os.path.join("E:/pythonProject/527 Climate Project/Training Process", "best_hparams.json")
        if os.path.exists(best_hparams_path):
            with open(best_hparams_path, 'r') as f:
                best_hparams = json.load(f)
            print("✅ Loaded hyperparameters: ", best_hparams)
        else:
            print("⚠️ best_hparams.json not found, executing GA search")
            full_dataset = ClimatePTChunkDataset(glob.glob(os.path.join(args.train_dir, '*.pt')))
            best_hparams = genetic_algorithm(full_dataset, generations=2, population_size=3)
    else:
        full_dataset = ClimatePTChunkDataset(glob.glob(os.path.join(args.train_dir, '*.pt')))
        best_hparams = genetic_algorithm(full_dataset, generations=2, population_size=3)

    print("Best hyperparameters:", best_hparams)
    final_train(args, best_hparams)
