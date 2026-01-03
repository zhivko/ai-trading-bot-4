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
    Execute a single trade using Strategy 4: Hybrid Multi-Tier PPT + Trail
    Tier 1 & 2: Fixed Targets (33% each)
    Tier 3: Trailing Stop Runner (34%)
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
    
    if is_sell:
        current_stop = entry_price + stop_distance
    else:
        current_stop = entry_price - stop_distance
        
    max_hold_candles = 24 * 4 * 7  # 7 days
    
    # Calculate Volume Profile for targets
    vp = calculate_volume_profile(df, start_idx=max(0, entry_idx-200), end_idx=entry_idx, precise=precise)
    
    targets = []
    if vp:
        if is_sell:
            hvns_below = sorted([p for p in vp['hvn_prices'] if p < entry_price * 0.995], reverse=True)
            # Tier 1 (33%)
            targets.append({'price': hvns_below[0] if len(hvns_below) >= 1 else entry_price * 0.98, 'pct': 0.33, 'filled': False})
            # Tier 2 (33%)
            targets.append({'price': hvns_below[1] if len(hvns_below) >= 2 else entry_price * 0.96, 'pct': 0.33, 'filled': False})
        else:
            hvns_above = sorted([p for p in vp['hvn_prices'] if p > entry_price * 1.005])
            # Tier 1 (33%)
            targets.append({'price': hvns_above[0] if len(hvns_above) >= 1 else entry_price * 1.02, 'pct': 0.33, 'filled': False})
            # Tier 2 (33%)
            targets.append({'price': hvns_above[1] if len(hvns_above) >= 2 else entry_price * 1.04, 'pct': 0.33, 'filled': False})
    else:
        # Fallbacks
        mult = 0.98 if is_sell else 1.02
        targets = [
            {'price': entry_price * mult, 'pct': 0.33, 'filled': False},
            {'price': entry_price * (mult**2), 'pct': 0.33, 'filled': False}
        ]
            
    # Simulation state
    pnl_realized = 0
    full_exit_reason = "HOLD_LIMIT"
    vp_last_updated = -999
    last_exit_time = entry_time
    first_target_hit = False
    
    # Loop through future candles
    search_end = min(len(df), entry_idx + max_hold_candles)
    current_close = entry_price
    
    for i in range(entry_idx + 1, search_end):
        current_low = df['low'].iloc[i]
        current_high = df['high'].iloc[i]
        current_close = df['close'].iloc[i]
        current_time = df.index[i]
        last_exit_time = current_time
        
        # 1. Check Stop Loss
        if is_sell:
            if current_high >= current_stop:
                pnl_realized += remaining_btc * (entry_price - current_stop)
                remaining_btc = 0
                full_exit_reason = "STOP_LOSS" if not first_target_hit else "TRAIL_STOP"
                break
        else:
            if current_low <= current_stop:
                pnl_realized += remaining_btc * (current_stop - entry_price)
                remaining_btc = 0
                full_exit_reason = "STOP_LOSS" if not first_target_hit else "TRAIL_STOP"
                break
            
        # 2. Check Fixed Tiers
        for target in targets:
            if not target['filled']:
                hit = (current_low <= target['price']) if is_sell else (current_high >= target['price'])
                if hit:
                    exit_amt = total_btc_amount * target['pct']
                    if exit_amt > remaining_btc: exit_amt = remaining_btc
                    pnl_realized += exit_amt * (entry_price - target['price']) if is_sell else exit_amt * (target['price'] - entry_price)
                    remaining_btc -= exit_amt
                    target['filled'] = True
                    if not first_target_hit:
                        first_target_hit = True
                        current_stop = entry_price # BE
                
        # 3. Update Trail for Runner (Tier 3 - remaining 34%)
        if first_target_hit and (i - vp_last_updated >= 10):
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

if __name__ == "__main__":
    print("Testing Strategy 4 module...")
    pass
