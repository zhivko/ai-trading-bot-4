import pandas as pd
import numpy as np

# Training script's stochastic calculation
def calc_stoch_training(df, period_k=14, smooth_k=3):
    low_min = df['low'].rolling(window=period_k).min()
    high_max = df['high'].rolling(window=period_k).max()
    stoch = 100 * ((df['close'] - low_min) / (high_max - low_min))
    return stoch.rolling(window=smooth_k).mean() / 100.0

# Chart generator's stochastic calculation  
def calc_stoch_chart(df, period_k, smooth_k, smooth_d):
    low_min = df['low'].rolling(window=period_k, min_periods=1).min()
    high_max = df['high'].rolling(window=period_k, min_periods=1).max()
    k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
    
    if smooth_k > 1:
        k_percent = k_percent.rolling(window=smooth_k, min_periods=1).mean()
    
    d_percent = k_percent.rolling(window=smooth_d, min_periods=1).mean()
    return d_percent

# Load test data
df = pd.read_csv('c:\\git\\ai-trading-bot-4\\data\\BTCUSDT_1h_data.csv', index_col='timestamp', parse_dates=True)
df_test = df.tail(200).copy()

# Calculate both ways
stoch_training = calc_stoch_training(df_test, 60, 10)
stoch_chart = calc_stoch_chart(df_test, 60, 10, 10)

print("Training script stochastic (60, 10):")
print(f"  Range: {stoch_training.min():.4f} to {stoch_training.max():.4f}")
print(f"  Last 5 values: {stoch_training.tail().values}")

print("\nChart generator stochastic (60, 10, 10):")
print(f"  Range: {stoch_chart.min():.4f} to {stoch_chart.max():.4f}")
print(f"  Last 5 values: {stoch_chart.tail().values}")

print("\nChart generator / 100:")
stoch_chart_normalized = stoch_chart / 100.0
print(f"  Range: {stoch_chart_normalized.min():.4f} to {stoch_chart_normalized.max():.4f}")
print(f"  Last 5 values: {stoch_chart_normalized.tail().values}")

print("\nDifference:")
print(f"  Max absolute difference: {np.abs(stoch_training - stoch_chart_normalized).max():.6f}")
