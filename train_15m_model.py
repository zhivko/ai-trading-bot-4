
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
from volume_profile import calculate_volume_profile  # Import VP logic

# -------------------------------------------------------------------------
# 1. MODEL ARCHITECTURE (Dual-Input: Time Series + Market Structure)
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
            nn.AdaptiveAvgPool1d(4) # Output: 512 * 4 = 2048 features
        )
        
        # --- Branch B: Market Structure (Volume Profile) ---
        self.vp_features = nn.Sequential(
            nn.Linear(vp_bins, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
            # Output: 64 features
        )
        
        # --- Fusion Head ---
        combined_dim = (512 * 4) + 64 
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x_ts, x_vp):
        # x_ts: [Batch, 1, Window]
        # x_vp: [Batch, Bins]
        
        # Branch A
        f_ts = self.ts_features(x_ts)
        f_ts = f_ts.view(f_ts.size(0), -1)
        
        # Branch B
        f_vp = self.vp_features(x_vp)
        
        # Concat
        combined = torch.cat((f_ts, f_vp), dim=1)
        
        return self.classifier(combined)

# -------------------------------------------------------------------------
# 2. DATA ENGINEERING & QUANTITATIVE CHANNEL DETECTION
# -------------------------------------------------------------------------
def calculate_stochastic(df, period_k=14, smooth_k=3):
    low_min = df['low'].rolling(window=period_k).min()
    high_max = df['high'].rolling(window=period_k).max()
    stoch = 100 * ((df['close'] - low_min) / (high_max - low_min))
    return stoch.rolling(window=smooth_k).mean() / 100.0

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_cp = np.abs(df['high'] - df['close'].shift())
    low_cp = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def get_channel_metrics(prices):
    """Returns slope and R-squared to define the 'Quality' of a downtrend."""
    y = prices.values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return model.coef_[0][0], model.score(x, y)

def prepare_data(path, window_size=100, lookforward=32):
    # lookforward=32 candles = 8 hours on 15m chart. 
    # Adjusted from 24 (6 hours) to give slightly more time for trend reversal verification.
    
    print(f"Loading and processing {path}...")
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    
    # Technical Indicators
    # Using larger periods for 15m to reduce noise? 
    # Or keep same logic but let model learn?
    # Keeping same strict parameters to find only HIGH QUALITY setups.
    df['stoch_long'] = calculate_stochastic(df, 60, 10) 
    df['atr'] = calculate_atr(df)
    
    # Volatility Adjustment: Higher ATR increases our 'low' threshold
    df['atr_ratio'] = df['atr'] / df['atr'].rolling(200).mean()
    df['dynamic_low'] = (0.20 * df['atr_ratio']).clip(0.15, 0.35)
    
    data_x_ts, data_x_vp, data_y = [], [], []
    
    print("Generating Features (TS + Volume Profile)... This may take a few minutes.")
    
    # Validation counters
    count_skipped_vp = 0
    count_skipped_nan = 0
    
    # Step size: 2 to reduce redundancy if dataset is huge, or 1 for max data
    # 15m data is huge (years). Step 4 (1 hour) might be enough?
    # Let's use step=2
    step = 2 
    
    for i in tqdm(range(200, len(df) - lookforward, step)): 
        # 1. Extract Window for CNN (Stochastic)
        win_long = df['stoch_long'].iloc[i-window_size:i].values
        
        if np.isnan(win_long).any(): 
            count_skipped_nan += 1
            continue

        # 2. Extract Volume Profile (Context)
        # Lookback 200 bars for market structure (approx 2 days of data on 15m)
        vp = calculate_volume_profile(df, start_idx=i-200, end_idx=i, num_bins=80)
        
        if vp is None: 
            count_skipped_vp += 1
            continue
        
        # Check VP validity
        vp_profile = vp['profile'].values
        if len(vp_profile) != 80: continue
        
        # Normalize VP (Scale Invariance)
        vp_max = vp_profile.max()
        if vp_max == 0: continue
        vp_normalized = vp_profile / vp_max
        
        # 3. Downtrend Channel Logic (Labeling) - RELAXED FOR 15M
        price_window = df['close'].iloc[i-window_size:i]
        slope, r_sq = get_channel_metrics(price_window)
        
        current_threshold = df['dynamic_low'].iloc[i]
        
        # RELAXED: Check if stochastics show ANY meaningful dip (not just deep oversold)
        # Instead of requiring 60% below dynamic_low, check if average of last 20 is low
        avg_stoch_recent = np.mean(win_long[-20:])
        is_low_stoch = avg_stoch_recent < 0.30  # Average below 30% is "low enough"
        
        # Also check for a dip pattern (recent low point)
        min_stoch_recent = np.min(win_long[-20:])
        has_dip = min_stoch_recent < 0.20  # At least one touch below 20%
        
        future_return = (df['close'].iloc[i+lookforward] / df['close'].iloc[i]) - 1
        
        label = 0
        # RELAXED CRITERIA:
        # - R² > 0.35 (was 0.50) - allows slightly messier downtrends
        # - Either low average OR clear dip
        # - Future return > 1.5% (was 2%) - more realistic for 15m
        if slope < 0 and r_sq > 0.35:
            if (is_low_stoch or has_dip) and future_return > 0.015:
                label = 1
        
        # Input TS: (1, WindowSize)
        data_x_ts.append(win_long.reshape(1, -1))
        # Input VP: (80, )
        data_x_vp.append(vp_normalized)
        
        data_y.append(label)
        
    return np.array(data_x_ts), np.array(data_x_vp), np.array(data_y)

# -------------------------------------------------------------------------
# 3. TRAINING ENGINE
# -------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal Loss to handle extreme class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_t * focal_weight
            
        loss = focal_weight * bce_loss
        return loss.mean()

def train_trading_bot(X_ts, X_vp, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True 

    # Handle Imbalance
    pos_count = sum(y)
    neg_count = len(y) - pos_count
    
    if pos_count == 0:
        print("CRITICAL ERROR: No positive examples found. Relax labeling criteria.")
        return

    pos_weight_value = (neg_count / pos_count) * 2.0
    pos_weight = torch.tensor([pos_weight_value]).to(device)
    
    print(f"\nClass Distribution:")
    print(f"  Positive examples: {pos_count} ({pos_count/len(y)*100:.2f}%)")
    print(f"  Negative examples: {neg_count} ({neg_count/len(y)*100:.2f}%)")
    print(f"  Imbalance ratio: 1:{neg_count/pos_count:.1f}")
    
    # Split
    indices = np.arange(len(y))
    idx_train, idx_val, y_train, y_val = train_test_split(indices, y, test_size=0.15, stratify=y)
    
    X_ts_train, X_vp_train = X_ts[idx_train], X_vp[idx_train]
    X_ts_val, X_vp_val = X_ts[idx_val], X_vp[idx_val]
    
    # Dataset
    train_ds = TensorDataset(
        torch.FloatTensor(X_ts_train).to(device),
        torch.FloatTensor(X_vp_train).to(device),
        torch.FloatTensor(y_train).unsqueeze(1).to(device)
    )
    # Larger batch size for bigger dataset
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    
    model = PatternDetectorCNN(window_size=X_ts.shape[2], vp_bins=X_vp.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    
    criterion = FocalLoss(alpha=0.75, gamma=2.0, pos_weight=pos_weight)
    scaler = torch.amp.GradScaler('cuda')

    print("Starting Training Loop...")
    for epoch in range(30): # 30 Epochs should be enough for 15m data (it's large)
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

    # SAVE WITH NEW NAME
    torch.save(model.state_dict(), "stoch_vp_detector_15m.pth")
    print("Model saved to stoch_vp_detector_15m.pth.")

if __name__ == "__main__":
    PATH = "data/BTCUSDT_15m_data.csv" 
    if os.path.exists(PATH):
        X_ts, X_vp, y = prepare_data(PATH)
        print(f"Data shapes: TS={X_ts.shape}, VP={X_vp.shape}, Y={y.shape}")
        if len(y) > 0:
            train_trading_bot(X_ts, X_vp, y)
    else:
        print(f"File not found: {PATH}")
