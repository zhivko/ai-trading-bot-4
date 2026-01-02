import pandas as pd
import os
import re
from datetime import datetime, timedelta
import requests
from binance.client import Client

def parse_filename(filename):
    """
    Parse filename to extract symbol and timeframe.
    Expected format: SYMBOL_TIMEFRAME_data.csv
    Example: BTCUSDT_1h_data.csv -> symbol: BTCUSDT, timeframe: 1h
    """
    pattern = r'^([A-Z]+)_(\d+[mhd])_data\.csv$'
    match = re.match(pattern, filename)

    if match:
        return {
            'symbol': match.group(1),
            'timeframe': match.group(2)
        }
    return None

def timeframe_to_minutes(timeframe):
    """Convert timeframe string to minutes"""
    value = int(timeframe[:-1])
    unit = timeframe[-1]

    if unit == 'm':
        return value
    elif unit == 'h':
        return value * 60
    elif unit == 'd':
        return value * 1440
    else:
        raise ValueError(f"Unknown timeframe unit: {unit}")

def timeframe_to_binance_interval(timeframe):
    """Convert timeframe to Binance API interval format"""
    mapping = {
        '1m': Client.KLINE_INTERVAL_1MINUTE,
        '3m': Client.KLINE_INTERVAL_3MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '30m': Client.KLINE_INTERVAL_30MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '2h': Client.KLINE_INTERVAL_2HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '6h': Client.KLINE_INTERVAL_6HOUR,
        '8h': Client.KLINE_INTERVAL_8HOUR,
        '12h': Client.KLINE_INTERVAL_12HOUR,
        '1d': Client.KLINE_INTERVAL_1DAY,
        '3d': Client.KLINE_INTERVAL_3DAY,
        '1w': Client.KLINE_INTERVAL_1WEEK,
    }
    return mapping.get(timeframe, Client.KLINE_INTERVAL_1HOUR)

def fetch_binance_klines(symbol, interval, start_time, end_time=None):
    """
    Fetch klines (OHLCV) data from Binance API
    Returns DataFrame with columns: timestamp, open, high, low, close, volume
    """
    try:
        # Initialize Binance client (no API key needed for public data)
        client = Client()

        # Fetch klines
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_time.strftime('%Y-%m-%d %H:%M:%S'),
            end_str=end_time.strftime('%Y-%m-%d %H:%M:%S') if end_time else None
        )

        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Keep only needed columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Convert price and volume to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        # Set timestamp as index
        df.set_index('timestamp', inplace=True)

        return df

    except Exception as e:
        print(f"Error fetching data from Binance: {e}")
        return None

def update_csv_file(filepath, symbol, timeframe):
    """
    Update CSV file with missing data from last entry to current time
    """
    print(f"Updating {filepath}...")

    # Read existing data
    df_existing = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)

    # Get last timestamp
    last_timestamp = df_existing.index[-1]
    current_time = datetime.now()

    # Calculate next expected candle time
    timeframe_minutes = timeframe_to_minutes(timeframe)
    next_candle_time = last_timestamp + timedelta(minutes=timeframe_minutes)

    # Check if we need to update
    if next_candle_time > current_time:
        print(f"  No update needed. Next candle at {next_candle_time}")
        return False

    print(f"  Last candle: {last_timestamp}")
    print(f"  Fetching data from {next_candle_time} to {current_time}")

    # Fetch new data from Binance
    binance_interval = timeframe_to_binance_interval(timeframe)
    df_new = fetch_binance_klines(symbol, binance_interval, next_candle_time, current_time)

    if df_new is None or len(df_new) == 0:
        print(f"  No new data available")
        return False

    # Remove incomplete last candle (current candle that's still forming)
    df_new = df_new.iloc[:-1]

    if len(df_new) == 0:
        print(f"  No complete candles to add")
        return False

    print(f"  Fetched {len(df_new)} new candles")

    # Merge with existing data
    df_combined = pd.concat([df_existing, df_new])

    # Remove duplicates (keep last)
    df_combined = df_combined[~df_combined.index.duplicated(keep='last')]

    # Sort by timestamp
    df_combined.sort_index(inplace=True)

    # Save back to CSV
    df_combined.to_csv(filepath)

    print(f"  Updated successfully. Total candles: {len(df_combined)}")
    return True

def update_data_files():
    """
    Scan data directory and update all CSV files with missing data
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return

    # Get all CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    print(f"\n{'='*80}")
    print(f"Starting data update for {len(csv_files)} files...")
    print(f"{'='*80}\n")

    updated_count = 0

    for filename in csv_files:
        # Parse filename to get symbol and timeframe
        parsed = parse_filename(filename)

        if parsed is None:
            print(f"Skipping {filename} - invalid filename format")
            continue

        symbol = parsed['symbol']
        timeframe = parsed['timeframe']
        filepath = os.path.join(data_dir, filename)

        try:
            if update_csv_file(filepath, symbol, timeframe):
                updated_count += 1
        except Exception as e:
            print(f"Error updating {filename}: {e}")

    print(f"\n{'='*80}")
    print(f"Update complete. {updated_count}/{len(csv_files)} files updated")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    # Test the updater
    update_data_files()
