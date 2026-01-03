import pandas as pd
import numpy as np
from volume_profile import calculate_volume_profile

def calculate_atr(df, period=14):
    """Calculate Average True Range (ATR)"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def execute_trade(df, entry_idx, current_balance, position_size_pct=0.5, precise: bool = False, is_sell: bool = False):
    """
    Execute a single trade using Strategy 5: Dynamic Trailing Stop
    Supports Long and Short positions.
    """
    entry_price = df['close'].iloc[entry_idx]
    entry_time = df.index[entry_idx]
    
    # Calculate position size
    position_usd = current_balance * position_size_pct
    btc_amount = position_usd / entry_price
    
    # Dynamic ATR Initial Stop Loss
    start_atr = max(0, entry_idx - 20)
    atr_window = df.iloc[start_atr:entry_idx+1].copy()
    atr_series = calculate_atr(atr_window, period=14)
    current_atr = atr_series.iloc[-1]
    
    if np.isnan(current_atr) or current_atr == 0:
        current_atr = entry_price * 0.01

    atr_multiplier = 4.5
    stop_distance = current_atr * atr_multiplier
    
    if is_sell:
        current_stop = entry_price + stop_distance
    else:
        current_stop = entry_price - stop_distance
        
    max_hold_candles = 24 * 4 * 7  # 7 days
    
    # Simulation state
    exit_price = entry_price
    exit_time = entry_time
    exit_reason = "HOLD_LIMIT"
    vp_last_updated = -999
    
    # Loop through future candles
    search_end = min(len(df), entry_idx + max_hold_candles)
    
    for i in range(entry_idx + 1, search_end):
        current_low = df['low'].iloc[i]
        current_high = df['high'].iloc[i]
        current_close = df['close'].iloc[i]
        current_time = df.index[i]
        
        # Check Stop Loss
        if is_sell:
            if current_high >= current_stop:
                exit_price = current_stop
                exit_time = current_time
                exit_reason = "TRAILING_STOP_HIT"
                break
        else:
            if current_low <= current_stop:
                exit_price = current_stop
                exit_time = current_time
                exit_reason = "TRAILING_STOP_HIT"
                break
            
        # Update trailing stop every 10 bars
        if i - vp_last_updated >= 10:
            vp = calculate_volume_profile(df, start_idx=max(0, i-200), end_idx=i, precise=precise)
            if vp:
                hvn_prices = vp['hvn_prices']
                if is_sell:
                    # Trail stop DOWN for Shorts: Find lowest HVN ABOVE current price but BELOW current stop
                    # Resistances
                    resistances = [p for p in hvn_prices if p > current_close * 1.01] # 1% buffer
                    if resistances:
                        potential_new_stop = min(resistances)
                        if potential_new_stop < current_stop:
                            if potential_new_stop < entry_price * 0.985: # Require 1.5% profit before trailing
                                current_stop = potential_new_stop
                else:
                    # Trail stop UP for Longs
                    supports = [p for p in hvn_prices if p < current_close * 0.99] # 1% buffer
                    if supports:
                        potential_new_stop = max(supports)
                        if potential_new_stop > current_stop:
                            if potential_new_stop > entry_price * 1.015:
                                current_stop = potential_new_stop
                
            vp_last_updated = i
            
        exit_price = current_close
        exit_time = current_time
    
    # Calculate PnL
    if is_sell:
        pnl_pct = (1 - exit_price / entry_price) * 100
    else:
        pnl_pct = (exit_price / entry_price - 1) * 100
        
    pnl = (position_usd * (pnl_pct / 100))
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
        'is_sell': is_sell
    }

if __name__ == "__main__":
    print("Testing Strategy 5 module...")
    pass
