import alarm_service
import os
import time
import pandas as pd
from datetime import datetime
from chart_generator import identify_quad_rotation_alarms
import backtest_strategy_3_lvn_acceleration as strat3

# Configuration
symbol = "BTCUSDT"
timeframe = "15m"
filepath = os.path.join(os.getcwd(), "data", f"{symbol}_{timeframe}_data.csv")

def find_last_real_alarm(filepath, symbol, timeframe):
    """Scan historical data for the absolute latest signal (any type)"""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
        
    df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
    if df.empty:
        return None
        
    # Apply detection logic
    from chart_generator import identify_quad_rotation_alarms
    df = identify_quad_rotation_alarms(df, nn_threshold=30)
    
    # Check all potential alarm columns
    # We want the LATEST one. 
    # Columns to check: 'alarm' (Quad), 'nn_buy_alarm', 'nn_sell_alarm'
    alarm_cols = [c for c in ['alarm', 'nn_buy_alarm', 'nn_sell_alarm'] if c in df.columns]
    if not alarm_cols:
        return None
        
    # Melt/Check where any column is True
    any_alarm_mask = df[alarm_cols].any(axis=1)
    alarm_rows = df[any_alarm_mask]
    
    if alarm_rows.empty:
        # Fallback
        last_time = df.index[-1]
        type_name = "Manual Test"
        is_buy = True
    else:
        # Pick the absolute latest row
        last_row = alarm_rows.iloc[-1]
        last_time = alarm_rows.index[-1]
        
        # Determine which signal it was (prioritize NN if both)
        if last_row.get('nn_sell_alarm'):
             type_name = "Neural Network"
             side_str = "SELL"
             is_buy = False
             emoji = "🔴"
        elif last_row.get('nn_buy_alarm'):
             type_name = "Neural Network"
             side_str = "BUY"
             is_buy = True
             emoji = "🟢"
        else:
             type_name = "Quad Rotation"
             side_str = "BUY"
             is_buy = True
             emoji = "🟢"
        
    # Standard format: Symbol_Time_Type
    # FIXED: Replace ALL spaces for Telegram command compatibility
    alarm_key = f"{symbol}_{timeframe}_{str(last_time).replace(' ', '-')}_{type_name.replace(' ', '-')}"
    
    # Extract trade levels for simulation
    df.loc[df.index[-1], 'nn_sell_alarm'] = not is_buy
    df.loc[df.index[-1], 'nn_buy_alarm'] = is_buy
    setup = strat3.get_trade_setup(df, len(df)-1)

    return {
        "key": alarm_key,
        "message": f"{emoji} **REAL SIGNAL SIMULATION**\nSymbol: {symbol}\nSignal: {type_name} {side_str}\nTime: {last_time}\nID: `{alarm_key}`\n\nThis simulation uses the absolute latest signal found in your data.",
        "symbol": symbol,
        "timeframe": timeframe,
        "is_buy": is_buy,
        "stop_loss": setup.get('stop_loss'),
        "take_profit": setup.get('take_profit_candidates', [None])[0] if setup.get('take_profit_candidates') else None
    }

print(f"Scanning {symbol} data for last alarm...")
alarm_data = find_last_real_alarm(filepath, symbol, timeframe)

if alarm_data:
    print(f"Found alarm: {alarm_data['key']}")
    print("Sending Telegram Notification...")
    try:
        alarm_service.send_telegram_alert(alarm_data, filepath)
        print("Simulation sent!")
    except Exception as e:
        print(f"Simulation failed: {e}")
else:
    print("Could not generate alarm data.")
