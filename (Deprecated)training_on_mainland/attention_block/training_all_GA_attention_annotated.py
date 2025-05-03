# 🚀 ConvLSTM training framework with Genetic Algorithm search and 5-Fold Cross Validation support

# --- Import Libraries ---
import os  # Handle directory and path operations
import glob  # Search for files using wildcards (e.g., "*.pt" files)
import csv  # Read and write CSV files for logging training results
import random  # Used in Genetic Algorithm for random hyperparameter mutation
import torch  # PyTorch main library for tensors and models
import torch.nn as nn  # Neural network module
from torch.utils.data import Dataset, DataLoader, Subset  # Data management utilities
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Plot training curves
from tqdm import tqdm  # Progress bar for loops
from sklearn.model_selection import KFold  # K-Fold Cross Validation
import json  # Save/load hyperparameter settings
import time  # Measure training duration

# --- Configuration Area ---
# Define important constants for dataset and model configuration
INPUT_CHANNELS = 13  # Number of input features (channels) per sample
TARGET_CHANNEL_INDEXES = list(range(0, 11))  # Select which channels to predict (11 output variables)
SEQUENCE_LENGTH = 6  # How many sequential frames to use as input
OUTPUT_DIR = "training_outputs_GA"  # Directory to save models, logs, and plots

# Create output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Argument Container ---
# Mimic argparse by manually defining training settings
class Args:
    train_dir = "G:/stream_split_dataset/train_sorted_pt_chunks"  # Directory containing training set
    val_dir = "G:/stream_split_dataset/val_sorted_pt_chunks"  # Directory containing validation set
    batch_size = 8  # Mini-batch size
    learning_rate = 1e-4  # Initial learning rate
    epochs = 30  # Number of training epochs
    model_save_path = os.path.join(OUTPUT_DIR, "best_model_11var.pt")  # File to save the best model
    resume = False  # Whether to resume training from a checkpoint
args = Args()

# --- Normalization Utilities ---
# These will store mean and std values for normalization and denormalization
INPUT_MEAN = None
INPUT_STD = None
OUTPUT_MEAN = None
OUTPUT_STD = None

# Function to compute mean and standard deviation across a list of .pt files
# This is important for normalizing data (z = (x - mean) / std) before training
def compute_mean_std(file_list, channels, mode='input'):
    all_sum = torch.zeros(channels)
    all_sqsum = torch.zeros(channels)
    count = 0
    for file in tqdm(file_list, desc=f"Computing {mode} mean/std", ncols=100):
        tensor = torch.load(file, map_location='cpu')
        if tensor.ndim == 4:
            tensor = tensor[0]  # (B, C, H, W) ➜ remove batch dimension
        if mode == 'input':
            tensor = tensor[:INPUT_CHANNELS]  # Select input channels
        else:
            tensor = tensor[TARGET_CHANNEL_INDEXES]  # Select output channels
        for c in range(tensor.shape[0]):
            all_sum[c] += tensor[c].sum()
            all_sqsum[c] += (tensor[c] ** 2).sum()
        count += tensor[0].numel()
    mean = all_sum / count
    std = (all_sqsum / count - mean ** 2).sqrt()
    return mean, std

# --- Custom Dataset ---
# This class loads a sequence of climate frames as input, and a future frame as target
# Dataset is used by PyTorch's DataLoader for batch training
class ClimatePTChunkDataset(Dataset):
    def __init__(self, pt_file_list):
        self.pt_files = pt_file_list
        self.seqlen = SEQUENCE_LENGTH  # How many timesteps per input
        self.verbose_limit = 5  # Limit console output to 5 samples for debugging
        print(f"Found {len(self.pt_files)} PT files for training/validation.")

    def __len__(self):
        # Each data point is a sequence of SEQUENCE_LENGTH frames ➜ shorten dataset accordingly
        return len(self.pt_files) - self.seqlen

    def __getitem__(self, idx):
        # Build input sequence
        seq_files = self.pt_files[idx: idx + self.seqlen]
        x_seq = []
        for file in seq_files:
            tensor = torch.load(file, map_location='cpu')
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)  # (H, W) ➜ (1, H, W)
            elif tensor.ndim == 4:
                tensor = tensor[0]  # Remove batch dimension
            if tensor.shape[0] < INPUT_CHANNELS:
                raise ValueError(f"File {file} has insufficient channels: {tensor.shape[0]}")
            tensor = tensor[:INPUT_CHANNELS]  # Select only relevant input channels
            x_seq.append(tensor)
        x_seq = torch.stack(x_seq, dim=0)  # Shape ➜ (T, C, H, W)

        # Normalize inputs using pre-computed mean/std
        x_seq = (x_seq - INPUT_MEAN[None, :, None, None]) / INPUT_STD[None, :, None, None]

        # Load target (future frame) and normalize
        target_file = self.pt_files[idx + self.seqlen]
        target_tensor = torch.load(target_file, map_location='cpu')
        if target_tensor.ndim == 4:
            raw_target = target_tensor[0]
        elif target_tensor.ndim == 3:
            raw_target = target_tensor
        else:
            raise ValueError(f"Invalid target tensor shape: {target_tensor.shape}")
        raw_target = raw_target[TARGET_CHANNEL_INDEXES]
        target_tensor = (raw_target - OUTPUT_MEAN[TARGET_CHANNEL_INDEXES][:, None, None]) / OUTPUT_STD[TARGET_CHANNEL_INDEXES][:, None, None]

        return x_seq, target_tensor

# --- Model Components ---

# Squeeze-and-Excitation Block: Helps the model learn which channels are more important dynamically
# It does so by learning per-channel weights via global average pooling and small FC network
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Output shape: (B, C, 1, 1)
            nn.Conv2d(channels, channels // reduction, 1),  # Reduce channel dimension
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),  # Restore channel dimension
            nn.Sigmoid()  # Output: weight for each channel ∈ [0,1]
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale  # Multiply weights to original input (channel-wise reweighting)

# Simple Residual Block: Helps mitigate vanishing gradients and learn identity mappings
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return torch.relu(x + self.block(x))  # Residual connection (skip connection)

# ConvLSTMCell: Main building block for spatiotemporal modeling
# Combines convolution with memory cells, allowing modeling of spatial and temporal dependencies
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2  # Padding to preserve spatial size
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size=kernel_size, padding=padding, bias=bias
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)  # Concatenate along channel dimension
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_output, 4, dim=1)  # Split into gates
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # Candidate memory
        c_next = f * c + i * g   # Cell state update
        h_next = o * torch.tanh(c_next)  # Hidden state update
        return h_next, c_next

# Attention Gate: Learns where to focus by comparing encoder and decoder feature maps
# Inspired by attention U-Net architecture, helps model focus on spatially important regions
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
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
        return x * psi  # Apply attention mask

# Complete ConvLSTM-based model with spatial encoder, temporal ConvLSTM, attention, and decoder
class ConvLSTMNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, kernel_size=3):
        super().__init__()
        # --- Encoder (spatial feature extraction using convolution) ---
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            SEBlock(64),
            ResidualBlock(64),
            nn.MaxPool2d(2),  # Downsample 2x
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            nn.MaxPool2d(2)   # Further downsample
        )

        # --- Temporal sequence modeling (6 ConvLSTM layers stacked) ---
        self.convlstm1 = ConvLSTMCell(128, hidden_channels, kernel_size)
        self.convlstm2 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
        self.convlstm3 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
        self.attention = AttentionGate(F_g=hidden_channels, F_l=hidden_channels, F_int=hidden_channels // 2)

        # --- Decoder (upsample and map back to output variables) ---
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
        h1 = c1 = h2 = c2 = h3 = c3 = None
        for t in range(T):
            x = self.encoder(x_seq[:, t])  # Apply encoder to each time step
            if h1 is None:
                # Initialize hidden and cell states
                h1 = torch.zeros(B, self.convlstm1.conv.out_channels // 4, x.shape[2], x.shape[3], device=x.device)
                c1 = torch.zeros_like(h1)
                h2 = torch.zeros_like(h1)
                c2 = torch.zeros_like(h1)
                h3 = torch.zeros_like(h1)
                c3 = torch.zeros_like(h1)
            h1, c1 = self.convlstm1(x, h1, c1)
            h2, c2 = self.convlstm2(h1, h2, c2)
            h3, c3 = self.convlstm3(h2, h3, c3)
        att = self.attention(h3, h3)  # Apply attention mechanism
        out = self.decoder(att)  # Decode to output channels
        return out

# --- EarlyStopping Utility ---
# This class monitors validation loss and stops training early if no improvement
# Prevents overfitting and saves time during final training
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience  # Number of epochs to wait before stopping
        self.counter = 0  # How many epochs have passed without improvement
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True  # Signal early stopping


# --- Genetic Algorithm (GA) Utilities ---

# Randomly sample a new hyperparameter combination for the population
def random_hyperparameters():
    return {
        'batch_size': random.choice([4, 6]),
        'hidden_size': random.choice([256, 320, 384, 448]),
        'kernel_size': random.choice([5]),  # Only use kernel size 5
        'learning_rate': random.choice([1e-4, 2e-4])
    }

# Save best hyperparameter set to a JSON file for later use
def save_best_hparams(best_hparams, path="best_hparams.json"):
    with open(path, "w") as f:
        json.dump(best_hparams, f, indent=4)
    print(f"Best hyperparameters saved to {path}")

import gc  # Used for manual garbage collection after GPU usage

# Evaluate a given hyperparameter set using 3-Fold Cross Validation
def evaluate_model(hparams, dataset):
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)  # 3 folds
    val_losses = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=hparams['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=hparams['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

        model = ConvLSTMNet(INPUT_CHANNELS, hparams['hidden_size'], len(TARGET_CHANNEL_INDEXES), kernel_size=hparams['kernel_size']).to("cuda")
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])
        criterion = nn.MSELoss()

        # Only train for 5 epochs during evaluation to save time
        for epoch in range(5):
            model.train()
            for x_seq, y in train_loader:
                x_seq, y = x_seq.cuda(), y.cuda()
                pred = model(x_seq)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate on validation set
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_seq, y in val_loader:
                x_seq, y = x_seq.cuda(), y.cuda()
                pred = model(x_seq)
                total_val_loss += criterion(pred, y).item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Clear GPU memory to prevent OOM errors
        del model
        torch.cuda.empty_cache()
        gc.collect()

    return np.mean(val_losses)  # Return average validation loss over all folds

# Genetic Algorithm main loop: evolves population to find best hyperparameter set
def genetic_algorithm(dataset, generations=2, population_size=3):
    population = [random_hyperparameters() for _ in range(population_size)]
    scores = [evaluate_model(individual, dataset) for individual in population]

    for gen in range(generations):
        print(f"Generation {gen+1}")
        # Rank population by score
        sorted_population = [x for _, x in sorted(zip(scores, population))]
        top_half = sorted_population[:population_size // 2]

        # Generate new children via mutation
        new_population = top_half.copy()
        while len(new_population) < population_size:
            parent = random.choice(top_half)
            child = parent.copy()
            mutation = random.choice(list(child.keys()))
            child[mutation] = random_hyperparameters()[mutation]  # Apply mutation
            new_population.append(child)

        # Evaluate new population
        population = new_population
        scores = [evaluate_model(individual, dataset) for individual in population]

    best_idx = np.argmin(scores)
    save_best_hparams(population[best_idx])
    return population[best_idx]  # Return best hyperparameter set

# --- Final Training Function ---
# After best hyperparameters are found via GA, this function performs full training
def final_train(args, best_hparams):
    global INPUT_MEAN, INPUT_STD, OUTPUT_MEAN, OUTPUT_STD
    print("Loading final training datasets...")

    # Compute dataset-wide mean and std if not already available
    train_file_list = glob.glob(os.path.join(args.train_dir, '*.pt'))
    if INPUT_MEAN is None or INPUT_STD is None:
        INPUT_MEAN, INPUT_STD = compute_mean_std(train_file_list, channels=INPUT_CHANNELS, mode='input')
        OUTPUT_MEAN, OUTPUT_STD = compute_mean_std(train_file_list, channels=13, mode='target')
        print(f"Input mean: {INPUT_MEAN.tolist()}  Input std: {INPUT_STD.tolist()}")
        print(f"Output mean: {OUTPUT_MEAN.tolist()}  Output std: {OUTPUT_STD.tolist()}")

    # Build datasets
    train_dataset = ClimatePTChunkDataset(train_file_list)
    val_dataset = ClimatePTChunkDataset(glob.glob(os.path.join(args.val_dir, '*.pt')))

    # Prepare dataloaders
    train_loader = DataLoader(train_dataset, batch_size=best_hparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_hparams['batch_size'], shuffle=False)

    # Initialize model, optimizer, and loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvLSTMNet(INPUT_CHANNELS, best_hparams['hidden_size'], len(TARGET_CHANNEL_INDEXES), kernel_size=best_hparams['kernel_size']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_hparams['learning_rate'])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    early_stopping = EarlyStopping(patience=7)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    # Logging setup
    log_file = os.path.join(OUTPUT_DIR, "final_training_log.csv")
    with open(log_file, mode='w', newline='') as logfile:
        writer = csv.writer(logfile)
        writer.writerow(["Epoch", "Train Loss", "Val Loss"])

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            for x_seq, y in train_loader:
                x_seq, y = x_seq.to(device), y.to(device)
                x_seq = x_seq.view(x_seq.size(0), SEQUENCE_LENGTH, INPUT_CHANNELS, 120, 240)
                pred = model(x_seq)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Evaluate on validation set
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_seq, y in val_loader:
                    x_seq, y = x_seq.to(device), y.to(device)
                    pred = model(x_seq)
                    val_loss += criterion(pred, y).item()

            # Compute average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # Adjust learning rate if plateau
            scheduler.step(avg_val_loss)
            early_stopping(avg_val_loss)

            # Logging
            print(f"Epoch {epoch+1}/{args.epochs} Train Loss: {avg_train_loss:.6f} Val Loss: {avg_val_loss:.6f}")
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss])

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), args.model_save_path)
                print("Best model saved!")

            # Save checkpoint every 10 epochs
            if (epoch+1) % 10 == 0:
                checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch+1}.pt")
                torch.save(model.state_dict(), checkpoint_path)

            # Early stopping
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

    # Plot training curves
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, "final_training_curve.png"))
    print("Training curve saved!")

# --- Main Program Entry ---
if __name__ == "__main__":
    print("Use existing hyperparameters? (y/n)")
    use_existing = input("Enter y to load best_hparams.json directly: ").strip().lower()

    if use_existing == 'y':
        best_hparams_path = os.path.join("best_hparams.json")
        if os.path.exists(best_hparams_path):
            with open(best_hparams_path, 'r') as f:
                best_hparams = json.load(f)
            print("Loaded hyperparameters:", best_hparams)
        else:
            print("best_hparams.json not found. Running GA search...")
            full_dataset = ClimatePTChunkDataset(glob.glob(os.path.join(args.train_dir, '*.pt')))
            best_hparams = genetic_algorithm(full_dataset, generations=2, population_size=3)
    else:
        full_dataset = ClimatePTChunkDataset(glob.glob(os.path.join(args.train_dir, '*.pt')))
        best_hparams = genetic_algorithm(full_dataset, generations=2, population_size=3)

    print("Best hyperparameters:", best_hparams)

    # Train model using best hyperparameters
    final_train(args, best_hparams)
