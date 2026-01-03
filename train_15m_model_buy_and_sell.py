
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from volume_profile import calculate_volume_profile

# -------------------------------------------------------------------------
# 1. MODEL ARCHITECTURE (Multi-Class Output)
# -------------------------------------------------------------------------
class PatternDetectorCNN(nn.Module):
    def __init__(self, window_size=100, vp_bins=80):
        super(PatternDetectorCNN, self).__init__()
        
        # --- Branch A: Time Series (Stochastic) ---
        self.ts_features = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
            
            nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(4) # Output: 2048 features
        )
        
        # --- Branch B: Market Structure (Volume Profile) ---
        self.vp_features = nn.Sequential(
            nn.Linear(vp_bins, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # --- Fusion Head ---
        combined_dim = (512 * 4) + 64 
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 3) # 3 Classes: 0=Hold, 1=Buy, 2=Sell
        )

    def forward(self, x_ts, x_vp):
        f_ts = self.ts_features(x_ts)
        f_ts = f_ts.view(f_ts.size(0), -1)
        f_vp = self.vp_features(x_vp)
        combined = torch.cat((f_ts, f_vp), dim=1)
        return self.classifier(combined)

# -------------------------------------------------------------------------
# 2. DATA ENGINEERING
# -------------------------------------------------------------------------
def calculate_stochastic(df, period_k=14, smooth_k=3):
    low_min = df['low'].rolling(window=period_k).min()
    high_max = df['high'].rolling(window=period_k).max()
    stoch = 100 * ((df['close'] - low_min) / (high_max - low_min))
    return stoch.rolling(window=smooth_k).mean() / 100.0

def get_channel_metrics(prices):
    y = prices.values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return model.coef_[0][0], model.score(x, y)

from joblib import Parallel, delayed

def process_single_window(i, df, window_size, lookforward):
    win_long = df['stoch_long'].iloc[i-window_size:i].values
    if np.isnan(win_long).any(): return None

    # Use minimal=True to skip heavy GMM calculations
    vp = calculate_volume_profile(df, start_idx=i-200, end_idx=i, num_bins=80, minimal=True)
    if vp is None: return None
    
    vp_profile = vp['profile'].values
    if len(vp_profile) != 80: return None
    
    vp_max = vp_profile.max()
    if vp_max == 0: return None
    vp_normalized = vp_profile / vp_max
    
    # Data from multiple scales
    s9 = df['stoch_9'].iloc[i-1]
    s14 = df['stoch_14'].iloc[i-1]
    s40 = df['stoch_40'].iloc[i-1]
    s60 = df['stoch_60'].iloc[i-1]
    
    # Calculate Alignment Metric (Standard Deviation or Spread)
    stoch_values = [s9, s14, s40, s60]
    alignment_spread = max(stoch_values) - min(stoch_values)
    is_aligned_low = (all(v < 0.25 for v in stoch_values)) and alignment_spread < 0.15
    is_aligned_high = (all(v > 0.75 for v in stoch_values)) and alignment_spread < 0.15
    
    # Labeling Logic
    price_window = df['close'].iloc[i-window_size:i]
    slope, r_sq = get_channel_metrics(price_window)
    future_return = (df['close'].iloc[i+lookforward] / df['close'].iloc[i]) - 1
    
    label = 0 # Default: HOLD
    
    # BUY Criteria: Quad Exhaustion + Downtrend + Future Upward Move
    if slope < 0 and r_sq > 0.35: 
        if is_aligned_low and future_return > 0.015:
            label = 1
    
    # SELL Criteria: Quad Overbought + Uptrend + Future Downward Move
    elif slope > 0 and r_sq > 0.35: 
        if is_aligned_high and future_return < -0.015:
            label = 2
            
    return (win_long.reshape(1, -1), vp_normalized, label)

def prepare_data(path, window_size=100, lookforward=32):
    print(f"Loading and processing {path}...")
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    
    df['stoch_9'] = calculate_stochastic(df, 9, 3)
    df['stoch_14'] = calculate_stochastic(df, 14, 3)
    df['stoch_40'] = calculate_stochastic(df, 40, 4)
    df['stoch_60'] = calculate_stochastic(df, 60, 10) # Original stoch_long
    
    # For the CNN window input, we'll keep using stoch_60 as the main trend feature
    df['stoch_long'] = df['stoch_60']
    
    print("Generating Features for Unified Model (Parallel Processing)...")
    
    indices = range(200, len(df) - lookforward, 2)
    
    # Use all available cores (n_jobs=-1)
    results = Parallel(n_jobs=-1)(
        delayed(process_single_window)(i, df, window_size, lookforward) 
        for i in tqdm(indices)
    )
    
    data_x_ts, data_x_vp, data_y = [], [], []
    for res in results:
        if res is not None:
            data_x_ts.append(res[0])
            data_x_vp.append(res[1])
            data_y.append(res[2])
        
    return np.array(data_x_ts), np.array(data_x_vp), np.array(data_y)

# -------------------------------------------------------------------------
# 3. TRAINING ENGINE
# -------------------------------------------------------------------------
class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, weight=None):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight # Class weights
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs, targets):
        log_pt = -self.ce(inputs, targets)
        pt = torch.exp(log_pt)
        focal_loss = -((1 - pt) ** self.gamma) * log_pt
        return focal_loss.mean()

def train_unified_model(X_ts, X_vp, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Unified Model on {device}")
    
    # Class Distribution
    classes, counts = np.unique(y, return_counts=True)
    dist = dict(zip(classes, counts))
    print(f"\nClass Distribution: {dist}")
    
    # Weights for Imbalance
    total = len(y)
    class_weights = []
    for c in range(3):
        count = dist.get(c, 0)
        if count == 0: 
            class_weights.append(1.0)
        else:
            # penalize HOLD (0) less, favor BUY (1) and SELL (2)
            weight = total / (3.0 * count)
            if c > 0: weight *= 2.0 # Extra boost for signals
            class_weights.append(weight)
            
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class Weights: {class_weights}")

    # Split
    idx_train, idx_val, y_train, y_val = train_test_split(np.arange(len(y)), y, test_size=0.15, stratify=y)
    
    train_ds = TensorDataset(
        torch.FloatTensor(X_ts[idx_train]).to(device),
        torch.FloatTensor(X_vp[idx_train]).to(device),
        torch.LongTensor(y_train).to(device)
    )
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    
    model = PatternDetectorCNN(window_size=X_ts.shape[2], vp_bins=X_vp.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    criterion = MultiClassFocalLoss(gamma=2.0, weight=weights_tensor)
    scaler = torch.amp.GradScaler('cuda')

    print("Starting Training...")
    for epoch in range(40): # More epochs for 3 classes
        model.train()
        total_loss = 0
        for b_ts, b_vp, b_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                pred = model(b_ts, b_vp)
                loss = criterion(pred, b_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "stoch_vp_unified_15m.pth")
    print("Unified model saved as stoch_vp_unified_15m.pth")

if __name__ == "__main__":
    PATH = "data/BTCUSDT_15m_data.csv"
    if os.path.exists(PATH):
        X_ts, X_vp, y = prepare_data(PATH)
        if len(np.unique(y)) < 3:
            print("WARNING: Not all classes (Hold, Buy, Sell) found. Check labeling logic.")
        train_unified_model(X_ts, X_vp, y)
    else:
        print(f"File not found: {PATH}")
