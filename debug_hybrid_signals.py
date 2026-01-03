import pandas as pd
import numpy as np
from chart_generator import identify_quad_rotation_alarms

# Load data
df = pd.read_csv('BTCUSDT_15m_data.csv', index_col='timestamp', parse_dates=True)
df_slice = df.tail(10000).copy()

print("Analyzing signal quality...")
df_processed = identify_quad_rotation_alarms(df_slice, nn_threshold=30)

# Count signals
nn_buy_count = df_processed['nn_buy_alarm'].sum()
nn_sell_count = df_processed['nn_sell_alarm'].sum()
hybrid_buy_count = df_processed['hybrid_buy_alarm'].sum()
hybrid_sell_count = df_processed['hybrid_sell_alarm'].sum()

print(f"\n{'='*80}")
print(f"SIGNAL COMPARISON (Last 10,000 candles)")
print(f"{'='*80}")
print(f"Raw NN BUY signals:     {nn_buy_count}")
print(f"Raw NN SELL signals:    {nn_sell_count}")
print(f"Hybrid BUY signals:     {hybrid_buy_count}")
print(f"Hybrid SELL signals:    {hybrid_sell_count}")
print(f"\nHybrid Conversion Rate:")
print(f"  BUY:  {hybrid_buy_count}/{nn_buy_count} = {hybrid_buy_count/nn_buy_count*100 if nn_buy_count > 0 else 0:.1f}%")
print(f"  SELL: {hybrid_sell_count}/{nn_sell_count} = {hybrid_sell_count/nn_sell_count*100 if nn_sell_count > 0 else 0:.1f}%")

# Check intermediate conditions
breakout_count = df_processed['linreg_top'].notna().sum()
breakdown_count = df_processed['linreg_bot_up'].notna().sum()

print(f"\nChannel Detection:")
print(f"  Downward channels detected: {(df_processed['slope'] < 0).sum()}")
print(f"  Upward channels detected:   {(df_processed['slope_up'] > 0).sum()}")

# Sample some hybrid signals to see what's happening
if hybrid_buy_count > 0:
    print(f"\n{'='*80}")
    print(f"SAMPLE HYBRID BUY SIGNALS (First 5):")
    print(f"{'='*80}")
    hybrid_buys = df_processed[df_processed['hybrid_buy_alarm']].head(5)
    for idx, row in hybrid_buys.iterrows():
        print(f"\nTimestamp: {idx}")
        print(f"  Close: ${row['close']:.2f}")
        print(f"  Stoch 60: {row['stoch_60_10']*100:.1f}%")
        print(f"  Stoch 40: {row['stoch_40_4']*100:.1f}%")
        print(f"  Slope: {row['slope']:.6f}")

if hybrid_sell_count > 0:
    print(f"\n{'='*80}")
    print(f"SAMPLE HYBRID SELL SIGNALS (First 5):")
    print(f"{'='*80}")
    hybrid_sells = df_processed[df_processed['hybrid_sell_alarm']].head(5)
    for idx, row in hybrid_sells.iterrows():
        print(f"\nTimestamp: {idx}")
        print(f"  Close: ${row['close']:.2f}")
        print(f"  Stoch 60: {row['stoch_60_10']*100:.1f}%")
        print(f"  Stoch 40: {row['stoch_40_4']*100:.1f}%")
        print(f"  Slope Up: {row['slope_up']:.6f}")
