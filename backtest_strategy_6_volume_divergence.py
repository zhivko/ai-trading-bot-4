import pandas as pd
import numpy as np
from volume_profile import calculate_volume_profile

def execute_trade(df, entry_idx, current_balance, position_size_pct=0.05):
    """
    Execute a single trade using Strategy 6: Volume Divergence
    Exit when price makes new high but POC volume decreases (divergence)
    
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
    
    # Stop loss
    stop_loss_pct = 0.05
    stop_price = entry_price * (1 - stop_loss_pct)
    max_hold_candles = 24 * 4 * 7  # 7 days (15m candles)
    
    # Simulation state
    exit_price = entry_price # Default
    exit_time = entry_time
    exit_reason = "HOLD_LIMIT"
    exit_idx = entry_idx
    vp_last_updated = -999
    
    # Loop through future candles
    search_end = min(len(df), entry_idx + max_hold_candles)
    
    # Need at least 50 bars to start checking context
    
    for i in range(entry_idx + 1, search_end):
        current_low = df['low'].iloc[i]
        current_close = df['close'].iloc[i]
        current_time = df.index[i]
        
        # Check Stop Loss
        if current_low <= stop_price:
            exit_price = stop_price
            exit_time = current_time
            exit_reason = "STOP_LOSS"
            exit_idx = i
            break
            
        # Strategy Logic: Update every 20 bars
        # Check for Volume Divergence:
        # 1. Price is making new highs (compared to previous window)
        # 2. POC Volume is decreasing
        
        if i - entry_idx > 100 and (i - vp_last_updated >= 20):
            # Define two windows: Recent (last 50) vs Previous (50 before that)
            recent_start = i - 50
            prev_start = i - 100
            
            # Check price trend
            recent_high = df['high'].iloc[recent_start:i].max()
            prev_high = df['high'].iloc[prev_start:recent_start].max()
            
            # Only worry if making new highs
            if recent_high > prev_high:
                # Calculate VPs
                vp_recent = calculate_volume_profile(df, start_idx=recent_start, end_idx=i)
                vp_prev = calculate_volume_profile(df, start_idx=prev_start, end_idx=recent_start)
                
                if vp_recent and vp_prev:
                    # Check volume divergence: significant drop in POC volume
                    # Normalize by number of bars (both are 50 here, but good practice)
                    # or just compare raw if windows are same size.
                    
                    if vp_recent['poc_volume'] < vp_prev['poc_volume'] * 0.75: # 25% drop in volume density
                        current_profit_pct = (current_close / entry_price - 1) * 100
                        if current_profit_pct > 1.0: # Only exit if profitable
                            exit_price = current_close
                            exit_time = current_time
                            exit_reason = "VOLUME_DIVERGENCE"
                            exit_idx = i
                            break
            
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
    print("Testing Strategy 6 module...")
    pass
