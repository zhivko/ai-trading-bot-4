
import pandas as pd
import numpy as np
import torch
import os
import sys

sys.path.append(os.getcwd())

from chart_generator import get_nn_model, calculate_volume_profile

def calculate_nn_for_timestamp(df, target_ts_str):
    target_ts = pd.to_datetime(target_ts_str)
    if df.index.tz is None and target_ts.tz is not None:
        target_ts = target_ts.tz_localize(None)
        
    try:
        idx = df.index.get_loc(target_ts)
    except KeyError:
        idx = df.index.searchsorted(target_ts)
        if idx >= len(df): idx = len(df) - 1

    print(f"Checking: {df.index[idx]}")
    
    model = get_nn_model()
    device = next(model.parameters()).device
    
    window_size = 100
    if idx < window_size:
        print("Not enough data")
        return 0.0
        
    # Calculate Stoch
    period_k = 60
    smooth_k = 10
    low_min = df['low'].rolling(window=period_k).min()
    high_max = df['high'].rolling(window=period_k).max()
    stoch = 100 * ((df['close'] - low_min) / (high_max - low_min))
    stoch_vals = (stoch.rolling(window=smooth_k).mean() / 100.0).values
    
    win_stoch = stoch_vals[idx-window_size:idx]
    
    # VP
    vp = calculate_volume_profile(df, start_idx=idx-200, end_idx=idx, num_bins=80)
    if not vp or len(vp['profile']) != 80:
        print("VP failed")
        return 0.0
        
    vp_profile = vp['profile'].values
    vp_max = vp_profile.max()
    vp_norm = vp_profile / vp_max if vp_max > 0 else vp_profile
    
    # Inference
    batch_ts = np.array([win_stoch])
    batch_vp = np.array([vp_norm])
    
    with torch.no_grad():
        t_ts = torch.FloatTensor(batch_ts).unsqueeze(1).to(device)
        t_vp = torch.FloatTensor(batch_vp).to(device)
        logits = model(t_ts, t_vp)
        prob = torch.sigmoid(logits).cpu().item()
        
    print(f"  NN Confidence: {prob*100:.2f}%")
    return prob

def main():
    filepath = "data/BTCUSDT_15m_data.csv"
    df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
    
    # Check dates around Dec 12-13
    targets = [
        "2025-12-12 12:00:00",
        "2025-12-12 18:00:00", 
        "2025-12-13 00:00:00",
        "2025-12-13 06:00:00",
        "2025-12-13 12:00:00"
    ]
    
    for t in targets:
        calculate_nn_for_timestamp(df, t)

if __name__ == "__main__":
    main()
