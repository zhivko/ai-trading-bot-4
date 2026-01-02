from chart_generator import generate_chart_data
import pandas as pd
import os
import traceback

def verify_gen():
    print("Testing generate_chart_data...")
    try:
        # Mimic 15m request
        symbol = "BTCUSDT"
        timeframe = "15m"
        filepath = os.path.join("data", f"{symbol}_{timeframe}_data.csv")
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return

        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        # Limit df for speed if needed, but error might only happen on full data
        # df = df.tail(1000) 
        
        print(f"Loaded {len(df)} rows. Generating chart data...")
        
        # Call the function
        # Note: arguments depend on the function signature in chart_generator.py
        # Based on file view, it likely takes: df, symbol, timeframe, ...
        # Let's check imports in app.py to see usage if possible, or just guess standard
        # app.py: generate_chart_data(df, symbol, timeframe) ??
        
        # Actually I need to check signature. But let's assume standard from context.
        # Wait, I'll inspect chart_generator.py signature first just to be safe.
        pass 
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    verify_gen()
