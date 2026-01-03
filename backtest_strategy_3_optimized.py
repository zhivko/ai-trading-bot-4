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
    Execute a single trade using Strategy 3 Optimized (v2): Heavy Runner + Volatility Buffer
    PPT (GMM Peak): 20%
    Runner (HVN Trail): 80%
    """
    entry_price = df['close'].iloc[entry_idx]
    entry_time = df.index[entry_idx]
    
    # Position sizing
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
    initial_stop = entry_price + stop_distance if is_sell else entry_price - stop_distance
    current_stop = initial_stop
    
    max_hold_candles = 24 * 4 * 7  # 7 days
    
    # Calculate Volume Profile for PPT target
    vp = calculate_volume_profile(df, start_idx=max(0, entry_idx-200), end_idx=entry_idx, precise=precise)
    if vp is None:
        return {'entry_time': entry_time, 'exit_time': entry_time, 'pnl': 0, 'pnl_pct': 0, 'balance_after': current_balance, 'hold_hours': 0, 'reason': 'VP_ERROR'}
    
    # PPT Target: First major GMM peak
    ppt_price = None
    if is_sell:
        peaks_below = sorted([p for p in vp['cluster_prices'] if p < entry_price * 0.99], reverse=True)
        ppt_price = peaks_below[0] if peaks_below else entry_price * 0.96
    else:
        peaks_above = sorted([p for p in vp['cluster_prices'] if p > entry_price * 1.01])
        ppt_price = peaks_above[0] if peaks_above else entry_price * 1.04
        
    # Simulation state
    pnl_realized = 0
    full_exit_reason = "HOLD_LIMIT"
    ppt_hit = False
    vp_last_updated = -999
    last_exit_time = entry_time
    current_close = entry_price
    
    # Loop through future candles
    search_end = min(len(df), entry_idx + max_hold_candles)
    
    for i in range(entry_idx + 1, search_end):
        current_low = df['low'].iloc[i]
        current_high = df['high'].iloc[i]
        current_close = df['close'].iloc[i]
        current_time = df.index[i]
        last_exit_time = current_time
        
        # 1. Stop Loss
        if is_sell:
            if current_high >= current_stop:
                pnl_realized += remaining_btc * (entry_price - current_stop)
                remaining_btc = 0
                full_exit_reason = "STOP_LOSS" if not ppt_hit else "TRAIL_STOP"
                break
        else:
            if current_low <= current_stop:
                pnl_realized += remaining_btc * (current_stop - entry_price)
                remaining_btc = 0
                full_exit_reason = "STOP_LOSS" if not ppt_hit else "TRAIL_STOP"
                break
            
        # 2. PPT (20%)
        if not ppt_hit:
            hit_ppt = False
            if is_sell:
                if current_low <= ppt_price: hit_ppt = True
            else:
                if current_high >= ppt_price: hit_ppt = True
                
            if hit_ppt:
                exit_amount = total_btc_amount * 0.20
                if exit_amount > remaining_btc: exit_amount = remaining_btc
                pnl_realized += exit_amount * (entry_price - ppt_price) if is_sell else exit_amount * (ppt_price - entry_price)
                remaining_btc -= exit_amount
                ppt_hit = True
                
                # VOLATILITY BUFFER: Move SL to -1 ATR (Protected)
                if is_sell:
                    current_stop = min(current_stop, entry_price + current_atr)
                else:
                    current_stop = max(current_stop, entry_price - current_atr)
                
        # 3. Update Trail (80%)
        if ppt_hit and (i - vp_last_updated >= 10):
            # Delayed Break-Even: After 5 more bars or another 0.5% move, move to BE
            if current_stop != entry_price:
                move_to_be = False
                if is_sell and current_close < entry_price * 0.995: move_to_be = True
                if not is_sell and current_close > entry_price * 1.005: move_to_be = True
                if move_to_be: current_stop = entry_price
            
            vp_i = calculate_volume_profile(df, start_idx=max(0, i-200), end_idx=i, precise=precise, minimal=True)
            if vp_i:
                hvn_prices = vp_i['hvn_prices']
                if is_sell:
                    resistances = [p for p in hvn_prices if p > current_close * 1.005]
                    if resistances:
                        new_stop = min(resistances)
                        if new_stop < current_stop: current_stop = new_stop
                else:
                    supports = [p for p in hvn_prices if p < current_close * 0.995]
                    if supports:
                        new_stop = max(supports)
                        if new_stop > current_stop: current_stop = new_stop
            vp_last_updated = i
                
        if remaining_btc < 1e-10:
            full_exit_reason = "ALL_TARGETS"
            break
            
    if remaining_btc > 0:
        pnl_realized += remaining_btc * (entry_price - current_close) if is_sell else remaining_btc * (current_close - entry_price)
        remaining_btc = 0
        
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
        'reason': full_exit_reason,
        'is_sell': is_sell
    }
