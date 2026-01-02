import pandas as pd
import sys
sys.path.insert(0, 'c:\\git\\ai-trading-bot-4')

from chart_generator import identify_quad_rotation_alarms

# Load test data
df = pd.read_csv('c:\\git\\ai-trading-bot-4\\data\\BTCUSDT_1h_data.csv', index_col='timestamp', parse_dates=True)

# Take last 200 candles for testing
df_test = df.tail(200).copy()

print(f"Testing NN detection on {len(df_test)} candles...")
print(f"Date range: {df_test.index[0]} to {df_test.index[-1]}")

# Run the detection with low threshold to see if anything triggers
df_result = identify_quad_rotation_alarms(df_test, nn_threshold=50)

print(f"\nColumns in result: {df_result.columns.tolist()}")

if 'nn_alarm' in df_result.columns:
    nn_count = df_result['nn_alarm'].sum()
    print(f"NN alarms found: {nn_count}")
    if nn_count > 0:
        print("\nNN alarm timestamps:")
        print(df_result[df_result['nn_alarm']].index.tolist())
else:
    print("WARNING: 'nn_alarm' column not found!")

if 'hybrid_alarm' in df_result.columns:
    hybrid_count = df_result['hybrid_alarm'].sum()
    print(f"Hybrid alarms found: {hybrid_count}")
else:
    print("WARNING: 'hybrid_alarm' column not found!")
