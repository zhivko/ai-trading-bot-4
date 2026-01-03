
import pandas as pd
import numpy as np
from volume_profile import calculate_volume_profile

def debug_gaps():
    print("Loading data...")
    try:
        df = pd.read_csv('BTCUSDT_15m_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    except FileNotFoundError:
        print("Data file not found.")
        return

    # Target specific date from screenshot
    target_date = "2025-12-29 04:15:00"
    
    try:
        target_idx = df.index.get_loc(target_date)
        print(f"Found target index: {target_idx} for date {target_date}")
    except KeyError:
        print(f"Target date {target_date} not found in data. Using last 200.")
        target_idx = len(df) - 1

    # Take 200 rows ENDING at target_idx
    start_idx = max(0, target_idx - 200)
    df_slice = df.iloc[start_idx:target_idx+1].copy()
    
    print(f"Data slice: {len(df_slice)} rows ({df_slice.index[0]} to {df_slice.index[-1]})")
    print(f"Price range: {df_slice['low'].min()} - {df_slice['high'].max()}")
    
    # Calculate VP with 80 bins
    vp = calculate_volume_profile(df_slice, precise=True, num_bins=80, verbose=True)
    
    if not vp:
        print("VP Calculation failed")
        return

    profile = vp['profile']
    bins = vp['bins']
    
    empty_bins = profile[profile == 0].index
    print(f"\nTotal Bins: {len(profile)}")
    print(f"Empty Bins: {len(empty_bins)}")
    
    if len(empty_bins) == 0:
        print("No empty bins found in this slice!")
        return

    print("\nAnalyzing Empty Bins...")
    
    # Check if any candle actually touches an empty bin
    bug_found = False
    
    for bin_idx in empty_bins:
        bin_low = bins[bin_idx]
        bin_high = bins[bin_idx+1]
        
        # Check for overlap
        overlapping = df_slice[
            (df_slice['high'] > bin_low) & (df_slice['low'] < bin_high)
        ]
        
        if not overlapping.empty:
            # Check edge cases - "touching" vs "overlapping"
            # np.digitize bins are [a, b). 
            # If high == bin_low, it doesn't enter.
            # precise logic uses:
            # low_bin = digitize(low) - 1
            # high_bin = digitize(high) - 1
            
            # Let's see the exact values of the overlaps
            print(f"BUG? Bin {bin_idx} ({bin_low:.4f} - {bin_high:.4f}) is empty but has overlaps:")
            print(overlapping[['open','high','low','close','volume']].head())
            bug_found = True
            
    if not bug_found:
        print("\nSUCCESS: All empty bins correspond to price gaps (no candle touched them).")
        print("The empty bins are accurately displaying lack of trading at those levels.")
    else:
        print("\nFAILURE: Logic bug confirmed. Some empty bins were touched by candles.")

if __name__ == "__main__":
    debug_gaps()
