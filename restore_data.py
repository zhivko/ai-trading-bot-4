
import os
import pandas as pd
from datetime import datetime
from data_updater import fetch_binance_klines, timeframe_to_binance_interval

def restore_btc_15m():
    symbol = "BTCUSDT"
    timeframe = "15m"
    filepath = os.path.join("data", "BTCUSDT_15m_data.csv")
    
    print(f"Restoring {filepath}...")
    
    # Fetch data. Try to fetch a good chunk.
    end_time = datetime.now()
    start_time = end_time - pd.Timedelta(days=20)
    
    interval = timeframe_to_binance_interval(timeframe)
    
    try:
        df = fetch_binance_klines(symbol, interval, start_time, end_time)
        
        if df is not None and not df.empty:
            # Ensure data dir exists
            os.makedirs("data", exist_ok=True)
            df.to_csv(filepath)
            print(f"Successfully restored {filepath} with {len(df)} candles.")
        else:
            print("Failed to fetch data: DataFrame is empty or None")
    except Exception as e:
        print(f"Error restoring data: {e}")

if __name__ == "__main__":
    restore_btc_15m()
