# ✅ 精简 + 修正后的训练脚本：预测 prcp 和 tavg（通道 0 和 1）
# ✅ ConvLSTM 层数减为 4 层；Attention 模块移除
# ✅ 使用 HuberLoss 替代 MSELoss，提高鲁棒性
# ✅ 保留 GA 超参数搜索、3-Fold CV、EarlyStopping、最优模型保存
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

INPUT_CHANNELS = 13
TARGET_CHANNEL_INDEXES = [0, 1]  # 仅预测 prcp 和 tavg
SEQUENCE_LENGTH = 6
OUTPUT_DIR = "training_outputs_GA_log"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class Args:
    train_dir = "G:/stream_split_dataset/train_sorted_pt_masked_chunks"
    val_dir = "G:/stream_split_dataset/val_sorted_pt_masked_chunks"
    batch_size = 8
    learning_rate = 1e-4
    epochs = 30
    model_save_path = os.path.join(OUTPUT_DIR, "best_model_prcp_tavg_log.pt")
    resume = False
args = Args()

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
            tensor = tensor[:INPUT_CHANNELS]
        else:
            tensor = tensor[TARGET_CHANNEL_INDEXES]
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

    def __len__(self):
        return len(self.pt_files) - self.seqlen

    def __getitem__(self, idx):
        seq_files = self.pt_files[idx: idx + self.seqlen]
        x_seq = []

        for file in seq_files:
            tensor = torch.load(file, map_location='cpu')
            if tensor.ndim == 4:
                tensor = tensor[0]
            tensor = tensor[:INPUT_CHANNELS]
            x_seq.append(tensor.unsqueeze(0))

        x_seq = torch.cat(x_seq, dim=0)  # [T, C, H, W]
        x_seq = (x_seq - INPUT_MEAN.view(1, -1, 1, 1)) / INPUT_STD.view(1, -1, 1, 1)

        # === target ===
        target_file = self.pt_files[idx + self.seqlen]
        target_tensor = torch.load(target_file, map_location='cpu')
        if target_tensor.ndim == 4:
            target_tensor = target_tensor[0]
        target_tensor = target_tensor[TARGET_CHANNEL_INDEXES]

        # ✅ 对 prcp 做 log1p 转换（只对通道 0）
        target_tensor[0] = torch.log1p(target_tensor[0])

        # ✅ 标准化（用输出 mean/std）
        target_tensor = (target_tensor - OUTPUT_MEAN.view(-1, 1, 1)) / OUTPUT_STD.view(-1, 1, 1)

        return x_seq, target_tensor


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

class ConvLSTMNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, kernel_size=3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # H/2 x W/2
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # H/4 x W/4
        )

        # 🧠 ConvLSTM 层（注意 input_dim 设置）
        self.convlstm1 = ConvLSTMCell(input_dim=256, hidden_dim=hidden_channels, kernel_size=kernel_size)
        self.convlstm2 = ConvLSTMCell(input_dim=hidden_channels, hidden_dim=hidden_channels, kernel_size=kernel_size)
        self.convlstm3 = ConvLSTMCell(input_dim=hidden_channels, hidden_dim=hidden_channels, kernel_size=kernel_size)
        self.convlstm4 = ConvLSTMCell(input_dim=hidden_channels, hidden_dim=hidden_channels, kernel_size=kernel_size)

        # 🎯 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # H/2
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # H
            nn.Conv2d(64, output_channels, 3, padding=1)
        )

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        h1 = c1 = h2 = c2 = h3 = c3 = h4 = c4 = None

        for t in range(T):
            x = self.encoder(x_seq[:, t])  # [B, 256, H/4, W/4]

            if h1 is None:
                h1 = torch.zeros(B, self.convlstm1.conv.out_channels // 4, x.shape[2], x.shape[3], device=x.device)
                c1 = torch.zeros_like(h1)
                h2 = c2 = h3 = c3 = h4 = c4 = torch.zeros_like(h1)

            h1, c1 = self.convlstm1(x, h1, c1)
            h2, c2 = self.convlstm2(h1, h2, c2)
            h3, c3 = self.convlstm3(h2, h3, c3)
            h4, c4 = self.convlstm4(h3, h4, c4)

        return self.decoder(h4)

def random_hyperparameters():
    return {
        'batch_size': random.choice([4, 6]),
        'hidden_size': random.choice([192, 256, 320]),
        'kernel_size': random.choice([3, 5]),
        'learning_rate': random.choice([1e-4, 2e-4])
    }

# ✅ EarlyStopping
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

GA_SAMPLE_SIZE = 10

# ✅ Genetic Algorithm 评估函数
def evaluate_model(hparams, dataset):
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    val_losses = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=hparams['batch_size'], shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=hparams['batch_size'], shuffle=False, num_workers=2)

        model = ConvLSTMNet(INPUT_CHANNELS, hparams['hidden_size'], len(TARGET_CHANNEL_INDEXES), kernel_size=hparams['kernel_size']).to("cuda")
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])
        criterion = nn.HuberLoss()

        for epoch in range(5):
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

        del model
        torch.cuda.empty_cache()
        gc.collect()

    return np.mean(val_losses)

# ✅ GA 主程序

def genetic_algorithm(dataset, generations=4, population_size=5):
    population = [random_hyperparameters() for _ in range(population_size)]
    scores = []
    print(f"\n🧪 Initial population evaluation:")
    for i, ind in enumerate(population):
        print(f"➡️ Evaluating Individual {i + 1}/{population_size}: {ind}")
        score = evaluate_model(ind, dataset)
        scores.append(score)

    for gen in range(generations):
        print(f"\n Generation {gen + 1}/{generations} ---------------------------")
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

    best_idx = np.argmin(scores)
    best_hparams = population[best_idx]
    with open("best_hparams.json", "w") as f:
        json.dump(best_hparams, f, indent=2)
    print("💾 最佳超参数已保存 best_hparams_log.json")
    return best_hparams

# ✅ 最终训练逻辑

def final_train(args, best_hparams, dataset):
    global INPUT_MEAN, INPUT_STD, OUTPUT_MEAN, OUTPUT_STD

    train_files = glob.glob(os.path.join(args.train_dir, "*.pt"))
    INPUT_MEAN, INPUT_STD = compute_mean_std(train_files, INPUT_CHANNELS, 'input')
    OUTPUT_MEAN, OUTPUT_STD = compute_mean_std(train_files, len(TARGET_CHANNEL_INDEXES), 'target')

    # ✅ 最终训练使用完整 dataset（不再用 subset）
    full_train_loader = DataLoader(dataset, batch_size=best_hparams['batch_size'], shuffle=True)
    val_files = glob.glob(os.path.join(args.val_dir, "*.pt"))
    val_set = ClimatePTChunkDataset(val_files)
    val_loader = DataLoader(val_set, batch_size=best_hparams['batch_size'], shuffle=False)

    model = ConvLSTMNet(INPUT_CHANNELS, best_hparams['hidden_size'],
                        len(TARGET_CHANNEL_INDEXES),
                        kernel_size=best_hparams['kernel_size']).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=best_hparams['learning_rate'])
    criterion = nn.HuberLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    early_stopping = EarlyStopping(patience=7)

    best_val_loss = float('inf')
    log_path = os.path.join(OUTPUT_DIR, "final_training_log_logged.csv")
    with open(log_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Val Loss"])

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            for x_seq, y in full_train_loader:
                x_seq, y = x_seq.cuda(), y.cuda()
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
                    x_seq, y = x_seq.cuda(), y.cuda()
                    pred = model(x_seq)
                    val_loss += criterion(pred, y).item()

            avg_train = train_loss / len(full_train_loader)
            avg_val = val_loss / len(val_loader)
            writer.writerow([epoch + 1, avg_train, avg_val])
            scheduler.step(avg_val)
            early_stopping(avg_val)

            print(f"Epoch {epoch+1}: Train {avg_train:.4f}, Val {avg_val:.4f}")
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), args.model_save_path)
                print("📦 Saved best model")

            if early_stopping.early_stop:
                print("⏹️ Early stopping triggered!")
                break

INPUT_MEAN, INPUT_STD = compute_mean_std(glob.glob(os.path.join(args.train_dir, '*.pt')), INPUT_CHANNELS, 'input')
OUTPUT_MEAN, OUTPUT_STD = compute_mean_std(glob.glob(os.path.join(args.train_dir, '*.pt')), len(TARGET_CHANNEL_INDEXES), 'target')

torch.save(INPUT_MEAN, os.path.join(OUTPUT_DIR, "input_mean.pt"))
torch.save(INPUT_STD, os.path.join(OUTPUT_DIR, "input_std.pt"))
torch.save(OUTPUT_MEAN, os.path.join(OUTPUT_DIR, "output_mean.pt"))
torch.save(OUTPUT_STD, os.path.join(OUTPUT_DIR, "output_std.pt"))

# ✅ 主程序入口
if __name__ == "__main__":
    print("🔍 使用已有超参配置？ y/n")
    use_existing = input("➡️ 输入 y 加载 best_hparams.json，否则将执行搜索: ").strip().lower()
    if use_existing == 'y' and os.path.exists("best_hparams.json"):
        with open("best_hparams.json", 'r') as f:
            best_hparams = json.load(f)
    else:
        # 🔁 使用全数据列表，但只随机抽样 GA_SAMPLE_SIZE 个用于 GA 搜索
        all_files = glob.glob(os.path.join(args.train_dir, "*.pt"))
        random.seed(42)  # 保证复现
        sampled_files = random.sample(all_files, min(GA_SAMPLE_SIZE, len(all_files)))

        # ✅ 使用样本子集进行 GA 搜索
        sample_dataset = ClimatePTChunkDataset(sampled_files)
        best_hparams = genetic_algorithm(sample_dataset)

    print("✅ 最佳超参:", best_hparams)

    # ✅ 用全部数据做最终训练
    full_files = glob.glob(os.path.join(args.train_dir, "*.pt"))
    full_dataset = ClimatePTChunkDataset(full_files)
    final_train(args, best_hparams, full_dataset)


