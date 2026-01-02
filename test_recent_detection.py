import pandas as pd
import sys
sys.path.insert(0, 'c:\\git\\ai-trading-bot-4')

from chart_generator import identify_quad_rotation_alarms

# Load test data - last 500 candles
df = pd.read_csv('c:\\git\\ai-trading-bot-4\\data\\BTCUSDT_1h_data.csv', index_col='timestamp', parse_dates=True)
df_test = df.tail(500).copy()

print(f"Testing NN detection on last 500 candles...")
print(f"Date range: {df_test.index[0]} to {df_test.index[-1]}")

# Test with different thresholds
for threshold in [10, 25, 50, 75, 85]:
    print(f"\n{'='*60}")
    print(f"Testing with threshold = {threshold}%")
    print(f"{'='*60}")
    
    df_result = identify_quad_rotation_alarms(df_test.copy(), nn_threshold=threshold)
    
    if 'nn_alarm' in df_result.columns:
        nn_count = df_result['nn_alarm'].sum()
        print(f"NN alarms found: {nn_count}")
        if nn_count > 0:
            print("\nNN alarm timestamps:")
            for ts in df_result[df_result['nn_alarm']].index:
                print(f"  - {ts}")
    else:
        print("WARNING: 'nn_alarm' column not found!")
