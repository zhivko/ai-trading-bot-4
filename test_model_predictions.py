import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os

# Load the model
class PatternDetectorCNN(nn.Module):
    def __init__(self, window_size=100):
        super(PatternDetectorCNN, self).__init__()
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

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PatternDetectorCNN()
model.load_state_dict(torch.load("stoch_low_detector_5090.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

print(f"Model loaded on {device}")

# Load data
df = pd.read_csv('BTCUSDT_1h_data.csv', index_col='timestamp', parse_dates=True)

# Calculate stochastic the same way as training
period_k = 60
smooth_k = 10
low_min = df['low'].rolling(window=period_k).min()
high_max = df['high'].rolling(window=period_k).max()
stoch = 100 * ((df['close'] - low_min) / (high_max - low_min))
stoch_vals = (stoch.rolling(window=smooth_k).mean() / 100.0).values

print(f"\nStochastic stats:")
print(f"  Min: {np.nanmin(stoch_vals):.4f}, Max: {np.nanmax(stoch_vals):.4f}")
print(f"  Mean: {np.nanmean(stoch_vals):.4f}, Std: {np.nanstd(stoch_vals):.4f}")

# Test on a few random windows
window_size = 100
test_indices = [1000, 5000, 10000, 20000, 30000]

print(f"\nTesting model on random windows:")
for idx in test_indices:
    if idx < window_size or idx >= len(stoch_vals):
        continue
        
    window = stoch_vals[idx-window_size:idx]
    if np.isnan(window).any():
        continue
    
    with torch.no_grad():
        x_tensor = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0).to(device)
        logit = model(x_tensor)
        prob = torch.sigmoid(logit).item()
        
        timestamp = df.index[idx]
        stoch_val = stoch_vals[idx]
        print(f"  Index {idx} ({timestamp}): Stoch={stoch_val:.3f}, Confidence={prob*100:.2f}%")

# Find the highest confidence predictions across all data
print(f"\nScanning entire dataset for highest confidence predictions...")
all_probs = []
valid_indices = []

for i in range(window_size, len(stoch_vals)):
    window = stoch_vals[i-window_size:i]
    if not np.isnan(window).any():
        with torch.no_grad():
            x_tensor = torch.FloatTensor(window).unsqueeze(0).unsqueeze(0).to(device)
            logit = model(x_tensor)
            prob = torch.sigmoid(logit).item()
            all_probs.append(prob)
            valid_indices.append(i)

all_probs = np.array(all_probs)
print(f"\nPrediction statistics across {len(all_probs)} windows:")
print(f"  Min: {all_probs.min()*100:.4f}%")
print(f"  Max: {all_probs.max()*100:.4f}%")
print(f"  Mean: {all_probs.mean()*100:.4f}%")
print(f"  Median: {np.median(all_probs)*100:.4f}%")
print(f"  Std: {all_probs.std()*100:.4f}%")

# Show top 10 predictions
top_10_idx = np.argsort(all_probs)[-10:][::-1]
print(f"\nTop 10 highest confidence predictions:")
for rank, idx in enumerate(top_10_idx, 1):
    actual_idx = valid_indices[idx]
    timestamp = df.index[actual_idx]
    prob = all_probs[idx]
    stoch_val = stoch_vals[actual_idx]
    print(f"  {rank}. {timestamp}: {prob*100:.4f}% (stoch={stoch_val:.3f})")
