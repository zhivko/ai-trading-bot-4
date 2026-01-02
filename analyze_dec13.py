
import pandas as pd
import numpy as np
from scipy.stats import linregress

def calculate_stochastic(df, period_k, smooth_k):
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
    y = prices.values
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, r_value**2

filepath = "data/BTCUSDT_15m_data.csv"
df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)

target_str = "2025-12-13 00:00:00"
target_ts = pd.to_datetime(target_str)

idx = df.index.get_loc(target_ts)

print(f"\n--- Analysis for {df.index[idx]} (15M DATA) ---")

period_k = 60
smooth_k = 10

df['stoch'] = calculate_stochastic(df, period_k, smooth_k)
df['atr'] = calculate_atr(df)
df['atr_ratio'] = df['atr'] / df['atr'].rolling(200).mean()
df['dynamic_low'] = (0.20 * df['atr_ratio']).clip(0.15, 0.35)

window_size = 100

win_stoch = df['stoch'].iloc[idx-window_size:idx].values
win_prices = df['close'].iloc[idx-window_size:idx]

slope, r_sq = get_channel_metrics(win_prices)
current_threshold = df['dynamic_low'].iloc[idx]

last_20_stoch = win_stoch[-20:]
low_bars = np.sum(last_20_stoch < current_threshold)
low_ratio = low_bars / 20.0

# Check future return
lookforward = 32
future_return = (df['close'].iloc[idx+lookforward] / df['close'].iloc[idx]) - 1

print(f"1. Downtrend Quality (Current Req: Slope < 0, R2 > 0.50):")
print(f"   - Slope: {slope:.4f} {'[PASS]' if slope < 0 else '[FAIL]'}")
print(f"   - R-Squared: {r_sq:.4f} {'[PASS]' if r_sq > 0.5 else '[FAIL]'}")

print(f"2. Prolonged Low (Current Req: > 60% below {current_threshold:.2f}):")
print(f"   - Low Bars: {low_bars}/20")
print(f"   - Ratio: {low_ratio*100:.1f}% {'[PASS]' if low_ratio > 0.6 else '[FAIL]'}")

print(f"3. Future Return (+{lookforward} candles = 8 hours):")
print(f"   - Return: {future_return*100:.2f}% {'[PASS]' if future_return > 0.02 else '[FAIL]'}")

print(f"\n4. Would this be labeled as POSITIVE with relaxed criteria?")
print(f"   - R² > 0.35 (relaxed): {'YES' if r_sq > 0.35 else 'NO'}")
print(f"   - Low Ratio > 0.4 (relaxed): {'YES' if low_ratio > 0.4 else 'NO'}")
print(f"   - Future > 1.5%: {'YES' if future_return > 0.015 else 'NO'}")
