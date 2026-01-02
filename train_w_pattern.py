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

# -------------------------------------------------------------------------
# 1. MODEL ARCHITECTURE (RTX 5090 / Blackwell Optimized)
# -------------------------------------------------------------------------
class PatternDetectorCNN(nn.Module):
    def __init__(self, window_size=100):
        super(PatternDetectorCNN, self).__init__()
        # Input channels = 1 (Stochastic)
        self.features = nn.Sequential(
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
            nn.AdaptiveAvgPool1d(16) 
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

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

def prepare_data(path_1h, window_size=100, lookforward=24):
    print(f"Loading and processing data...")
    df = pd.read_csv(path_1h, index_col='timestamp', parse_dates=True)
    
    # Technical Indicators
    df['stoch_long'] = calculate_stochastic(df, 60, 10) # 1H "Slow"
    df['stoch_short'] = calculate_stochastic(df, 14, 3)  # 1H "Fast"
    df['atr'] = calculate_atr(df)
    
    # Volatility Adjustment: Higher ATR increases our 'low' threshold
    df['atr_ratio'] = df['atr'] / df['atr'].rolling(200).mean()
    df['dynamic_low'] = (0.20 * df['atr_ratio']).clip(0.15, 0.35)
    
    data_x, data_y = [], []
    
    for i in tqdm(range(window_size, len(df) - lookforward)):
        # 1. Extract Window for CNN
        win_long = df['stoch_long'].iloc[i-window_size:i].values
        win_short = df['stoch_short'].iloc[i-window_size:i].values
        
        if np.isnan(win_long).any() or np.isnan(win_short).any(): continue

        # 2. Downtrend Channel Logic
        price_window = df['close'].iloc[i-window_size:i]
        slope, r_sq = get_channel_metrics(price_window)
        
        # 3. Labeling Logic (The 'Boat Stick' Pattern) - RELAXED CRITERIA
        # Condition A: In a downtrend (negative slope + reasonable fit)
        # Condition B: Stochastic has been suppressed below dynamic threshold
        # Condition C: Future price breakout
        
        current_threshold = df['dynamic_low'].iloc[i]
        prolonged_low = np.mean(win_long[-20:] < current_threshold) > 0.6  # Relaxed from 0.8
        
        future_return = (df['close'].iloc[i+lookforward] / df['close'].iloc[i]) - 1
        
        label = 0
        if slope < 0 and r_sq > 0.50:  # Relaxed from 0.65 - allows looser downtrends
            if prolonged_low and future_return > 0.02:  # Relaxed from 0.03 - 2% move
                label = 1
        
        # Input is (1, WindowSize) - using only long stochastic
        data_x.append(win_long.reshape(1, -1))
        data_y.append(label)
        
    return np.array(data_x), np.array(data_y)

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

def train_trading_bot(X, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Optimization for Blackwell/5090
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True 

    # Handle Imbalance - Increase weight even more
    pos_count = sum(y)
    neg_count = len(y) - pos_count
    pos_weight_value = (neg_count / pos_count) * 2.0  # 2x multiplier for extra emphasis
    pos_weight = torch.tensor([pos_weight_value]).to(device)
    
    print(f"\nClass Distribution:")
    print(f"  Positive examples: {pos_count} ({pos_count/len(y)*100:.2f}%)")
    print(f"  Negative examples: {neg_count} ({neg_count/len(y)*100:.2f}%)")
    print(f"  Imbalance ratio: 1:{neg_count/pos_count:.1f}")
    print(f"  Positive class weight: {pos_weight_value:.1f}")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, stratify=y)
    
    train_ds = TensorDataset(torch.FloatTensor(X_train).to(device), torch.FloatTensor(y_train).unsqueeze(1).to(device))
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    
    model = PatternDetectorCNN(window_size=X.shape[2]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    
    # Use Focal Loss instead of BCE
    criterion = FocalLoss(alpha=0.75, gamma=2.0, pos_weight=pos_weight)
    scaler = torch.amp.GradScaler('cuda')

    print("Starting Training Loop...")
    for epoch in range(50):
        model.train()
        total_loss = 0
        for b_x, b_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                pred = model(b_x)
                loss = criterion(pred, b_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "stoch_low_detector_5090.pth")
    print("Model saved.")

if __name__ == "__main__":
    # Path to your 1H BTCUSDT Data
    PATH = "BTCUSDT_1h_data.csv" 
    if os.path.exists(PATH):
        X, y = prepare_data(PATH)
        train_trading_bot(X, y)
