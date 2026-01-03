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
    Execute a single trade using Strategy 1: Hybrid POC PPT + Trail
    """
    entry_price = df['close'].iloc[entry_idx]
    entry_time = df.index[entry_idx]
    
    # Calculate position size
    total_position_usd = current_balance * position_size_pct
    total_btc_amount = total_position_usd / entry_price
    remaining_btc = total_btc_amount
    
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
    
    # Calculate Volume Profile for targets
    vp = calculate_volume_profile(df, start_idx=max(0, entry_idx-200), end_idx=entry_idx, precise=precise)
    if vp is None:
        return {'entry_time': entry_time, 'exit_time': entry_time, 'pnl': 0, 'pnl_pct': 0, 'balance_after': current_balance, 'hold_hours': 0, 'reason': 'VP_ERROR'}
        
    # Determine Partial Profit Taking (PPT) Level
    ppt_price = None
    if is_sell:
        if vp['poc_price'] < entry_price * 0.99: ppt_price = vp['poc_price']
        elif vp['val'] < entry_price * 0.99: ppt_price = vp['val']
        else: ppt_price = entry_price * 0.96
    else:
        if vp['poc_price'] > entry_price * 1.01: ppt_price = vp['poc_price']
        elif vp['vah'] > entry_price * 1.01: ppt_price = vp['vah']
        else: ppt_price = entry_price * 1.04
    
    # Simulation state
    pnl_realized = 0
    exit_reason = "HOLD_LIMIT"
    ppt_hit = False
    vp_last_updated = -999
    last_exit_time = entry_time
    final_exit_price = entry_price
    
    # Loop through future candles
    search_end = min(len(df), entry_idx + max_hold_candles)
    current_close = entry_price
    
    for i in range(entry_idx + 1, search_end):
        current_low = df['low'].iloc[i]
        current_high = df['high'].iloc[i]
        current_close = df['close'].iloc[i]
        current_time = df.index[i]
        last_exit_time = current_time
        final_exit_price = current_close
        
        # 1. Check Stop Loss
        if is_sell:
            if current_high >= current_stop:
                pnl_realized += remaining_btc * (entry_price - current_stop)
                remaining_btc = 0
                exit_reason = "STOP_LOSS" if not ppt_hit else "TRAIL_STOP"
                final_exit_price = current_stop
                break
        else:
            if current_low <= current_stop:
                pnl_realized += remaining_btc * (current_stop - entry_price)
                remaining_btc = 0
                exit_reason = "STOP_LOSS" if not ppt_hit else "TRAIL_STOP"
                final_exit_price = current_stop
                break
            
        # 2. Check PPT (if not hit)
        if not ppt_hit:
            hit_ppt = False
            if is_sell:
                if current_low <= ppt_price: hit_ppt = True
            else:
                if current_high >= ppt_price: hit_ppt = True
                
            if hit_ppt:
                exit_amount = total_btc_amount * 0.50
                pnl_realized += exit_amount * (entry_price - ppt_price) if is_sell else exit_amount * (ppt_price - entry_price)
                remaining_btc -= exit_amount
                ppt_hit = True
                # Move SL to Break Even
                current_stop = entry_price
                
        # 3. Update Trailing Stop (if PPT hit)
        if ppt_hit and (i - vp_last_updated >= 10):
            vp_i = calculate_volume_profile(df, start_idx=max(0, i-200), end_idx=i, precise=precise)
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
                
        if remaining_btc < 1e-10:
            exit_reason = "ALL_TARGETS"
            break
            
    # Final cleanup
    if remaining_btc > 0:
        pnl_realized += remaining_btc * (entry_price - current_close) if is_sell else remaining_btc * (current_close - entry_price)
    
    pnl_pct = (pnl_realized / total_position_usd) * 100
    avg_exit_price = (entry_price - (pnl_realized / total_btc_amount)) if is_sell else (entry_price + (pnl_realized / total_btc_amount))
    
    return {
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_time': last_exit_time,
        'exit_price': avg_exit_price,
        'pnl': pnl_realized,
        'pnl_pct': pnl_pct,
        'balance_after': current_balance + pnl_realized,
        'hold_hours': (last_exit_time - entry_time).total_seconds() / 3600,
        'reason': exit_reason,
        'is_sell': is_sell
    }

if __name__ == "__main__":
    # Simple test logic
    print("Testing Strategy 1 module...")
    pass
