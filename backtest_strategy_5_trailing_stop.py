import pandas as pd
import numpy as np
from volume_profile import calculate_volume_profile

def execute_trade(df, entry_idx, current_balance, position_size_pct=0.05):
    """
    Execute a single trade using Strategy 5: Dynamic Trailing Stop
    Trails stop loss at HVNs below price
    
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
    
    # Initial Stop loss
    current_stop = entry_price * 0.95 # 5% initial stop
    max_hold_candles = 24 * 4 * 7  # 7 days (15m candles)
    
    # Simulation state
    exit_price = entry_price # Default
    exit_time = entry_time
    exit_reason = "HOLD_LIMIT"
    exit_idx = entry_idx
    vp_last_updated = -999
    hvn_prices = []
    
    # Loop through future candles
    search_end = min(len(df), entry_idx + max_hold_candles)
    
    for i in range(entry_idx + 1, search_end):
        current_low = df['low'].iloc[i]
        current_close = df['close'].iloc[i]
        current_time = df.index[i]
        
        # Check Stop Loss (Trailing)
        if current_low <= current_stop:
            exit_price = current_stop
            exit_time = current_time
            exit_reason = "TRAILING_STOP_HIT"
            exit_idx = i
            break
            
        # Strategy Logic: Update VP every 20 bars to find new support levels
        if i - vp_last_updated >= 20:
            vp = calculate_volume_profile(df, start_idx=max(0, i-200), end_idx=i)
            if vp:
                hvn_prices = vp['hvn_prices']
                # Try to move stop up
                # Find highest HVN below current price
                hvns_below = [p for p in hvn_prices if p < current_close * 0.99] # Keep 1% buffer
                
                if hvns_below:
                    potential_new_stop = max(hvns_below)
                    
                    # Conditions to move stop:
                    # 1. Must be higher than current stop
                    # 2. Must lock in at least 2% profit OR be better than initial stop (break-even logic etc)
                    # Let's simple rule: Only move stop if it locks in > 1.5% profit
                    
                    if potential_new_stop > current_stop:
                         if potential_new_stop > entry_price * 1.015:
                             current_stop = potential_new_stop
                
            vp_last_updated = i
            
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
    print("Testing Strategy 5 module...")
    pass
