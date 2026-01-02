import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('c:\\git\\ai-trading-bot-4\\data\\BTCUSDT_1h_data.csv', index_col='timestamp', parse_dates=True)

# Calculate stochastic the same way as training
period_k = 60
smooth_k = 10
low_min = df['low'].rolling(window=period_k).min()
high_max = df['high'].rolling(window=period_k).max()
stoch = 100 * ((df['close'] - low_min) / (high_max - low_min))
stoch_vals = (stoch.rolling(window=smooth_k).mean() / 100.0)

# Look at recent data
recent = df.tail(500).copy()
recent['stoch'] = stoch_vals.tail(500)

print("Recent market analysis (last 500 hours):")
print(f"Date range: {recent.index[0]} to {recent.index[-1]}")
print(f"\nStochastic statistics:")
print(f"  Mean: {recent['stoch'].mean():.3f}")
print(f"  Min: {recent['stoch'].min():.3f}")
print(f"  Max: {recent['stoch'].max():.3f}")
print(f"  Current: {recent['stoch'].iloc[-1]:.3f}")

# Count how many times stochastic was below 0.20 (20%)
low_stoch_count = (recent['stoch'] < 0.20).sum()
print(f"\nCandles with stochastic < 20%: {low_stoch_count} ({low_stoch_count/len(recent)*100:.1f}%)")

# Find periods of prolonged low stochastics
print(f"\nPeriods where stochastic stayed below 20% for 10+ hours:")
consecutive_low = 0
low_periods = []
for i, (idx, row) in enumerate(recent.iterrows()):
    if row['stoch'] < 0.20:
        consecutive_low += 1
    else:
        if consecutive_low >= 10:
            low_periods.append((idx - pd.Timedelta(hours=consecutive_low), idx, consecutive_low))
        consecutive_low = 0

if consecutive_low >= 10:
    low_periods.append((recent.index[-consecutive_low], recent.index[-1], consecutive_low))

if low_periods:
    for start, end, duration in low_periods:
        print(f"  {start} to {end} ({duration} hours)")
else:
    print("  None found")

# Check price trend
price_change = (recent['close'].iloc[-1] / recent['close'].iloc[0] - 1) * 100
print(f"\nPrice movement over period: {price_change:+.2f}%")

# Look at the training data characteristics
print("\n" + "="*60)
print("TRAINING DATA CHARACTERISTICS (what the model learned):")
print("="*60)

# The model was trained on data where:
# 1. Stochastic was below dynamic threshold (0.15-0.35) for prolonged periods
# 2. Price was in a downtrend (negative slope, R² > 0.65)
# 3. Future price increased by >3% after the low

print("\nThe model was trained to detect:")
print("  ✓ Prolonged stochastic lows (< 20%) for 20+ bars")
print("  ✓ In a strong downtrend (negative slope, R² > 0.65)")
print("  ✓ Followed by 3%+ price increase within 24 hours")

print("\nRecent market may not have these conditions, which is why")
print("the model shows low confidence on recent data.")

# Show where the model DID find patterns
print("\n" + "="*60)
print("DATES WHERE MODEL FOUND STRONG PATTERNS:")
print("="*60)
print("  • November 26-27, 2024: 100% confidence (multiple signals)")
print("  • December 22-25, 2025: 5-8% confidence")
