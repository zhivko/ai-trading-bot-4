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
    Execute a single trade using Strategy 6: Hybrid Volume Divergence + Trail
    """
    entry_price = df['close'].iloc[entry_idx]
    entry_time = df.index[entry_idx]
    
    # Position sizing
    position_usd = current_balance * position_size_pct
    btc_amount = position_usd / entry_price
    
    # Dynamic ATR Stop Loss
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
    last_exit_time = entry_time
    profit_threshold_hit = False
    
    # Loop through future candles
    search_end = min(len(df), entry_idx + max_hold_candles)
    current_close = entry_price
    
    for i in range(entry_idx + 1, search_end):
        current_low = df['low'].iloc[i]
        current_high = df['high'].iloc[i]
        current_close = df['close'].iloc[i]
        current_time = df.index[i]
        last_exit_time = current_time
        exit_price = current_close
        
        current_profit = (1 - current_close / entry_price) * 100 if is_sell else (current_close / entry_price - 1) * 100
        if not profit_threshold_hit and current_profit > 2.0:
            profit_threshold_hit = True
            current_stop = entry_price # BE
        
        # 1. Check Stop Loss
        if is_sell:
            if current_high >= current_stop:
                exit_price = current_stop
                exit_reason = "STOP_LOSS" if not profit_threshold_hit else "TRAIL_STOP"
                break
        else:
            if current_low <= current_stop:
                exit_price = current_stop
                exit_reason = "STOP_LOSS" if not profit_threshold_hit else "TRAIL_STOP"
                break
            
        # 2. Update Trail
        if i - vp_last_updated >= 10:
            vp_i = calculate_volume_profile(df, start_idx=max(0, i-200), end_idx=i, precise=precise, minimal=True)
            if vp_i:
                hvn_prices = vp_i['hvn_prices']
                if is_sell:
                    resistances = [p for p in hvn_prices if p > current_close * 1.01]
                    if resistances:
                        new_stop = min(resistances)
                        if new_stop < current_stop: current_stop = new_stop
                else:
                    supports = [p for p in hvn_prices if p < current_close * 0.99]
                    if supports:
                        new_stop = max(supports)
                        if new_stop > current_stop: current_stop = new_stop
            vp_last_updated = i

        # 3. Secondary Exit: Volume Divergence
        # Only check after some holding time
        if i - entry_idx > 40:
            recent_start = i - 20
            prev_start = i - 40
            if is_sell:
                recent_low = df['low'].iloc[recent_start:i].min()
                prev_low = df['low'].iloc[prev_start:recent_start].min()
                if recent_low < prev_low:
                    vp_recent = calculate_volume_profile(df, start_idx=recent_start, end_idx=i, precise=precise)
                    vp_prev = calculate_volume_profile(df, start_idx=prev_start, end_idx=recent_start, precise=precise)
                    if vp_recent and vp_prev and vp_recent['poc_volume'] < vp_prev['poc_volume'] * 0.70:
                        if current_profit > 1.0:
                            exit_reason = "VOLUME_DIVERGENCE"
                            break
            else:
                recent_high = df['high'].iloc[recent_start:i].max()
                prev_high = df['high'].iloc[prev_start:recent_start].max()
                if recent_high > prev_high:
                    vp_recent = calculate_volume_profile(df, start_idx=recent_start, end_idx=i, precise=precise)
                    vp_prev = calculate_volume_profile(df, start_idx=prev_start, end_idx=recent_start, precise=precise)
                    if vp_recent and vp_prev and vp_recent['poc_volume'] < vp_prev['poc_volume'] * 0.70:
                        if current_profit > 1.0:
                            exit_reason = "VOLUME_DIVERGENCE"
                            break
            
    pnl_pct = (1 - exit_price / entry_price) * 100 if is_sell else (exit_price / entry_price - 1) * 100
    pnl = (position_usd * (pnl_pct / 100))
    
    return {
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_time': last_exit_time,
        'exit_price': exit_price,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'balance_after': current_balance + pnl,
        'hold_hours': (last_exit_time - entry_time).total_seconds() / 3600,
        'reason': exit_reason,
        'is_sell': is_sell
    }

if __name__ == "__main__":
    print("Testing Strategy 6 module...")
    pass
