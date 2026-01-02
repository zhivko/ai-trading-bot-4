import pandas as pd
import numpy as np
from volume_profile import calculate_volume_profile

def execute_trade(df, entry_idx, current_balance, position_size_pct=0.05):
    """
    Execute a single trade using Strategy 2: VAH Exit
    
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
    
    # Calculate Volume Profile for context (last 200 bars)
    vp = calculate_volume_profile(df, start_idx=max(0, entry_idx-200), end_idx=entry_idx)
    if vp is None:
        return {
            'entry_time': entry_time,
            'exit_time': entry_time,
            'pnl': 0,
            'pnl_pct': 0,
            'balance_after': current_balance,
            'hold_hours': 0,
            'reason': 'VP_ERROR'
        }
    
    # Determine Profit Target (VAH)
    target_price = None
    target_type = "NONE"
    
    # If VAH is significantly above entry (> 1%)
    if vp['vah'] > entry_price * 1.01:
        target_price = vp['vah']
        target_type = "VAH"
    else:
        # Fallback: Use POC if above
        if vp['poc_price'] > entry_price * 1.01:
            target_price = vp['poc_price']
            target_type = "POC_FALLBACK"
        else:
            # Final fallback
            target_price = entry_price * 1.05
            target_type = "FALLBACK_5PCT"
    
    # Simulate trade
    exit_price = entry_price # Default
    exit_time = entry_time
    exit_reason = "HOLD_LIMIT"
    exit_idx = entry_idx
    
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
            
        # Check Profit Target
        if current_high >= target_price:
            exit_price = target_price
            exit_time = current_time
            exit_reason = f"TARGET_{target_type}"
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
        'reason': exit_reason,
        'target_price': target_price,
        'target_type': target_type
    }

if __name__ == "__main__":
    print("Testing Strategy 2 module...")
    pass
