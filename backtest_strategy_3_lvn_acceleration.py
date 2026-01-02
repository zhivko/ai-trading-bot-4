import pandas as pd
import numpy as np
from volume_profile import calculate_volume_profile, is_in_lvn

def execute_trade(df, entry_idx, current_balance, position_size_pct=0.05):
    """
    Execute a single trade using Strategy 3: LVN Acceleration
    Exit before entering Low Volume Node (air pocket)
    
    Args:
        df: DataFrame with OHLCV data
        entry_idx: Index of the entry candle
        current_balance: Current account balance
        position_size_pct: Size of position relative to balance
        
    Returns:
        dict with trade results
    """
    entry_price = df['close'].iloc[entry_idx]
    entry_time = df.index[entry_idx]
    
    # Calculate position size
    position_usd = current_balance * position_size_pct
    btc_amount = position_usd / entry_price
    
    # Stop loss and Take Profit settings
    stop_loss_pct = 0.05
    stop_price = entry_price * (1 - stop_loss_pct)
    max_hold_candles = 24 * 4 * 7  # 7 days (15m candles)
    
    # Simulation state
    exit_price = entry_price # Default
    exit_time = entry_time
    exit_reason = "HOLD_LIMIT"
    exit_idx = entry_idx
    vp_last_updated = -999
    lvn_prices = []
    
    # Loop through future candles
    search_end = min(len(df), entry_idx + max_hold_candles)
    
    for i in range(entry_idx + 1, search_end):
        current_low = df['low'].iloc[i]
        current_high = df['high'].iloc[i]
        current_close = df['close'].iloc[i]
        current_time = df.index[i]
        
        # Check Stop Loss
        if current_low <= stop_price:
            exit_price = stop_price
            exit_time = current_time
            exit_reason = "STOP_LOSS"
            exit_idx = i
            break
            
        # Strategy Logic: Update VP every 20 bars to check for LVNs ahead
        if i - vp_last_updated >= 20:
            vp = calculate_volume_profile(df, start_idx=max(0, i-200), end_idx=i)
            if vp:
                # Find LVNs strictly ABOVE current price
                lvn_prices = [p for p in vp['lvn_prices'] if p > current_close]
                lvn_prices.sort()
            vp_last_updated = i
            
        # Check Exit Trigger: Entering LVN
        # We check if the HIGH price touched any LVN level
        triggered_lvn = None
        for lvn in lvn_prices:
            # Check if we are approaching LVN (within 0.2%)
            if current_high >= lvn * 0.998:
                triggered_lvn = lvn
                break
        
        # Only exit if profitable (at least 1%)
        if triggered_lvn is not None and triggered_lvn > entry_price * 1.01:
            exit_price = triggered_lvn # Exit at the LVN price
            exit_time = current_time
            exit_reason = "LVN_ENTRY"
            exit_idx = i
            break
            
        # Update exit values for max hold limit
        exit_price = current_close
        exit_time = current_time
        exit_idx = i
    
    # Calculate PnL
    exit_usd = btc_amount * exit_price
    pnl = exit_usd - position_usd
    pnl_pct = (exit_price / entry_price - 1) * 100
    
    # Hold duration in hours
    hold_duration = (exit_time - entry_time).total_seconds() / 3600
    
    return {
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_time': exit_time,
        'exit_price': exit_price,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'balance_after': current_balance + pnl,
        'hold_hours': hold_duration,
        'reason': exit_reason
    }

if __name__ == "__main__":
    print("Testing Strategy 3 module...")
    pass
