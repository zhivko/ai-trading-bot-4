import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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
    y = prices.values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return model.coef_[0][0], model.score(x, y)

# Load data
df = pd.read_csv('BTCUSDT_1h_data.csv', index_col='timestamp', parse_dates=True)

# Calculate indicators
df['stoch_long'] = calculate_stochastic(df, 60, 10)
df['atr'] = calculate_atr(df)
df['atr_ratio'] = df['atr'] / df['atr'].rolling(200).mean()
df['dynamic_low'] = (0.20 * df['atr_ratio']).clip(0.15, 0.35)

window_size = 100
lookforward = 24

positive_count = 0
negative_count = 0
positive_examples = []

print("Analyzing training data labeling...")
print(f"Total candles: {len(df)}")
print(f"Analyzing windows from index {window_size} to {len(df) - lookforward}")

for i in range(window_size, len(df) - lookforward):
    win_long = df['stoch_long'].iloc[i-window_size:i].values
    
    if np.isnan(win_long).any():
        continue
    
    # Downtrend Channel Logic
    price_window = df['close'].iloc[i-window_size:i]
    slope, r_sq = get_channel_metrics(price_window)
    
    # Labeling Logic
    current_threshold = df['dynamic_low'].iloc[i]
    prolonged_low = np.mean(win_long[-20:] < current_threshold) > 0.6
    future_return = (df['close'].iloc[i+lookforward] / df['close'].iloc[i]) - 1
    
    label = 0
    if slope < 0 and r_sq > 0.50:
        if prolonged_low and future_return > 0.02:
            label = 1
            positive_count += 1
            positive_examples.append({
                'timestamp': df.index[i],
                'slope': slope,
                'r_sq': r_sq,
                'stoch_mean': np.mean(win_long[-20:]),
                'future_return': future_return * 100
            })
        else:
            negative_count += 1
    else:
        negative_count += 1

total = positive_count + negative_count
print(f"\n{'='*60}")
print(f"TRAINING DATA STATISTICS:")
print(f"{'='*60}")
print(f"Total valid windows: {total:,}")
print(f"Positive examples (pattern found): {positive_count:,} ({positive_count/total*100:.2f}%)")
print(f"Negative examples (no pattern): {negative_count:,} ({negative_count/total*100:.2f}%)")
print(f"\nClass imbalance ratio: 1:{negative_count/positive_count:.1f}")

if positive_count > 0:
    print(f"\n{'='*60}")
    print(f"POSITIVE EXAMPLES ANALYSIS:")
    print(f"{'='*60}")
    
    # Show first 10 and last 10 positive examples
    print(f"\nFirst 10 positive examples:")
    for i, ex in enumerate(positive_examples[:10], 1):
        print(f"  {i}. {ex['timestamp']}: Return={ex['future_return']:.2f}%, Stoch={ex['stoch_mean']:.3f}, R²={ex['r_sq']:.3f}")
    
    if len(positive_examples) > 10:
        print(f"\nLast 10 positive examples:")
        for i, ex in enumerate(positive_examples[-10:], len(positive_examples)-9):
            print(f"  {i}. {ex['timestamp']}: Return={ex['future_return']:.2f}%, Stoch={ex['stoch_mean']:.3f}, R²={ex['r_sq']:.3f}")
    
    # Statistics
    returns = [ex['future_return'] for ex in positive_examples]
    stochs = [ex['stoch_mean'] for ex in positive_examples]
    
    print(f"\n{'='*60}")
    print(f"POSITIVE EXAMPLES CHARACTERISTICS:")
    print(f"{'='*60}")
    print(f"Future returns:")
    print(f"  Mean: {np.mean(returns):.2f}%")
    print(f"  Median: {np.median(returns):.2f}%")
    print(f"  Min: {np.min(returns):.2f}%")
    print(f"  Max: {np.max(returns):.2f}%")
    print(f"\nStochastic values:")
    print(f"  Mean: {np.mean(stochs):.3f}")
    print(f"  Median: {np.median(stochs):.3f}")
    print(f"  Min: {np.min(stochs):.3f}")
    print(f"  Max: {np.max(stochs):.3f}")

print(f"\n{'='*60}")
print(f"RECOMMENDATIONS:")
print(f"{'='*60}")
if positive_count < 100:
    print("⚠️  Very few positive examples! Model will have low confidence.")
    print("   Solutions:")
    print("   1. Relax criteria further (lower R², lower future return threshold)")
    print("   2. Use more data (longer timeframe)")
    print("   3. Use data augmentation techniques")
elif positive_count < 1000:
    print("⚠️  Limited positive examples. Model may struggle.")
    print("   Consider relaxing criteria or using more data.")
else:
    print("✓  Sufficient positive examples for training.")
