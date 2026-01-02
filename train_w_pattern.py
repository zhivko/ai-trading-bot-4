import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import time

# -------------------------------------------------------------------------
# 1. MODEL ARCHITECTURE (RTX 5090 Optimized)
# -------------------------------------------------------------------------
class PatternDetectorCNN(nn.Module):
    def __init__(self, window_size=100):
        super(PatternDetectorCNN, self).__init__()
        # Deeper architecture for Blackwell architecture benefit
        self.features = nn.Sequential(
            # First block: Wide local feature detection
            nn.Conv1d(1, 128, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2), # -> 50
            
            # Second block
            nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2), # -> 25
            
            # Third block
            nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(16) # -> 16 global features per filter
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
            # Sigmoid removed for BCEWithLogitsLoss integration
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# -------------------------------------------------------------------------
# 2. DATA PREPARATION
# -------------------------------------------------------------------------
def prepare_training_data(filepath, window_size=100):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
    
    # Calculate Stoch 60,10
    low_min = df['low'].rolling(window=60, min_periods=1).min()
    high_max = df['high'].rolling(window=60, min_periods=1).max()
    k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
    stoch = k_percent.rolling(window=10, min_periods=1).mean()
    
    stoch = stoch / 100.0 # Normalize 0-1
    values = stoch.values
    
    data_x = []
    data_y = []
    
    print("Generating windows and labels...")
    # Slicing is faster than looping for dataset generation
    # Pre-calculating indices that match 'W' criteria
    for i in range(window_size, len(values)):
        window = values[i-window_size:i]
        if np.isnan(window).any():
            continue
            
        # --- REFINED HEURISTIC: "Shallow W near 20" ---
        # 1. Deep bottoms (around 15-25)
        # 2. Low amplitude peak (doesn't shoot to 80)
        # 3. Small breakout
        
        q2 = window[25:50]
        q3 = window[50:75]
        q4 = window[75:100]
        
        q2_min = q2.min()
        q4_min = q4.min()
        middle_max = q3.max()
        
        label = 0
        
        # Criteria:
        # Both bottoms must be in the "Oversold" zone but not necessarily 0
        is_near_20 = (q2_min < 0.30) and (q4_min < 0.35) 
        
        # Middle peak must be SHALLOW (the "lower amplitude" requirement)
        # If peak is > 0.5 (50), it's a "Normal" W, not the "Shallow" one the user wants.
        is_shallow = (middle_max < 0.55) and (middle_max > q2_min) and (middle_max > q4_min)
        
        # Amplitude check: The distance between bottom and peak should be modest
        amplitude = middle_max - min(q2_min, q4_min)
        is_correct_amplitude = (amplitude > 0.05) and (amplitude < 0.35)
        
        if is_near_20 and is_shallow and is_correct_amplitude:
            # We are currently at the end of the window. 
            # We want to label the point where the W is confirmed (breaking middle_max)
            if window[-1] > middle_max and window[-1] < 0.60:
                label = 1
        
        data_x.append(window)
        data_y.append(label)
        
    return np.array(data_x), np.array(data_y)

# -------------------------------------------------------------------------
# 3. HIGH-PERFORMANCE TRAINING ENGINE
# -------------------------------------------------------------------------
def train_model(X, y):
    device = torch.device("cuda")
    print(f"ULTRA TRAINING ON: {torch.cuda.get_device_name(0)}")
    
    # Optimization flags
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True # RTX 5090 loves TF32
    torch.backends.cudnn.allow_tf32 = True

    # Split and Transfer to GPU immediately
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # Load everything to VRAM (only ~200MB, 5090 has 32000MB)
    X_train = torch.FloatTensor(X_train).unsqueeze(1).to(device)
    y_train = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val = torch.FloatTensor(X_val).unsqueeze(1).to(device)
    y_val = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    train_ds = TensorDataset(X_train, y_train)
    # Huge batch size for 5090
    batch_size = 8192 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    model = PatternDetectorCNN().to(device)
    
    # UTILIZE TORCH COMPILE (Requires Triton/OpenMP, might be slow first run but fast later)
    # try:
    #     print("Compiling model for Blackwell architecture...")
    #     model = torch.compile(model)
    # except Exception as e:
    #     print(f"Skipping torch.compile: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, 
                                              steps_per_epoch=len(train_loader), 
                                              epochs=100)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda') # Use mixed precision

    epochs = 100
    print(f"Starting Ultra-Fast Training for {epochs} epochs (Batch Size: {batch_size})...")
    
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True) # Optimized zeroing
            
            with torch.amp.autocast('cuda'): # Mixed precision
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            
        # Validation
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                val_out = model(X_val)
                v_loss = criterion(val_out, y_val)
            
        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(v_loss.item())
        
        if (epoch + 1) % 10 == 0:
            avg_time = (time.time() - start_time) / (epoch + 1)
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {train_losses[-1]:.6f} | Val Loss: {val_losses[-1]:.6f} | {avg_time:.2f}s/epoch")

    total_time = time.time() - start_time
    print(f"Training Complete in {total_time:.2f} seconds!")
    
    # Save the model
    torch.save(model.state_dict(), "stoch_w_detector_5090.pth")
    return train_losses, val_losses

if __name__ == "__main__":
    data_path = r"c:\git\ai-trading-bot-4\data\BTCUSDT_15m_data.csv"
    
    if os.path.exists(data_path):
        X, y = prepare_training_data(data_path)
        print(f"Dataset summary: Total={len(X)}, Positives={sum(y)}")
        
        train_losses, val_losses = train_model(X, y)
        
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.yscale('log')
        plt.title('RTX 5090 Performance (Mixed Precision + Compiled)')
        plt.legend()
        plt.savefig('performance_plot.png')
        print("Model saved as stoch_w_detector_5090.pth")
    else:
        print(f"Data not found at {data_path}")
