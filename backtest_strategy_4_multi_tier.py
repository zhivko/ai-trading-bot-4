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

def execute_trade(df, entry_idx, current_balance, position_size_pct=0.05, precise: bool = False):
    """
    Execute a single trade using Strategy 4: Multi-Tier Scaling
    Exits: 33% at 1st HVN, 33% at 2nd HVN, 34% at VAH/POC
    
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
    total_position_usd = current_balance * position_size_pct
    total_btc_amount = total_position_usd / entry_price
    remaining_btc = total_btc_amount
    
    # Dynamic ATR Stop Loss
    start_atr = max(0, entry_idx - 20)
    atr_window = df.iloc[start_atr:entry_idx+1].copy()
    atr_series = calculate_atr(atr_window, period=14)
    current_atr = atr_series.iloc[-1]
    
    if np.isnan(current_atr) or current_atr == 0:
        current_atr = entry_price * 0.01
        
    atr_multiplier = 4.5
    stop_distance = current_atr * atr_multiplier
    stop_price = entry_price - stop_distance
    stop_loss_pct = stop_distance / entry_price
    max_hold_candles = 24 * 4 * 7  # 7 days (15m candles)
    
    # Calculate Volume Profile for targets
    vp = calculate_volume_profile(df, start_idx=max(0, entry_idx-200), end_idx=entry_idx, precise=precise)
    
    targets = []
    
    if vp:
        # Find potential targets
        hvns_above = sorted([p for p in vp['hvn_prices'] if p > entry_price * 1.01])
        final_target = max(vp['vah'], vp['poc_price'])
        
        # Tier 1: First HVN (33%)
        if len(hvns_above) >= 1:
            targets.append({'price': hvns_above[0], 'pct': 0.33, 'filled': False, 'name': 'HVN1'})
        else:
             targets.append({'price': entry_price * 1.02, 'pct': 0.33, 'filled': False, 'name': 'FALLBACK_2PCT'})
             
        # Tier 2: Second HVN (33%)
        if len(hvns_above) >= 2:
            targets.append({'price': hvns_above[1], 'pct': 0.33, 'filled': False, 'name': 'HVN2'})
        else:
             targets.append({'price': entry_price * 1.04, 'pct': 0.33, 'filled': False, 'name': 'FALLBACK_4PCT'})
             
        # Tier 3: Final Target (34%)
        if final_target > entry_price * 1.01:
             targets.append({'price': final_target, 'pct': 0.34, 'filled': False, 'name': 'FINAL_POC_VAH'})
        else:
             targets.append({'price': entry_price * 1.06, 'pct': 0.34, 'filled': False, 'name': 'FALLBACK_6PCT'})
    else:
        # Fallbacks if VP fails
        targets = [
            {'price': entry_price * 1.02, 'pct': 0.33, 'filled': False, 'name': 'FALLBACK_1'},
            {'price': entry_price * 1.04, 'pct': 0.33, 'filled': False, 'name': 'FALLBACK_2'},
            {'price': entry_price * 1.06, 'pct': 0.34, 'filled': False, 'name': 'FALLBACK_3'}
        ]
            
    # Simulation state
    pnl_realized = 0
    exit_events = []
    last_exit_time = entry_time
    full_exit_reason = "HOLD_LIMIT"
    
    # Loop through future candles
    search_end = min(len(df), entry_idx + max_hold_candles)
    
    for i in range(entry_idx + 1, search_end):
        current_low = df['low'].iloc[i]
        current_high = df['high'].iloc[i]
        current_close = df['close'].iloc[i]
        current_time = df.index[i]
        last_exit_time = current_time
        
        # Check Stop Loss (on remaining position)
        if current_low <= stop_price:
            # Exit remaining position
            exit_value = remaining_btc * stop_price
            pnl_realized += exit_value - (remaining_btc * entry_price)
            exit_events.append(f"STOP_LOSS @ {stop_price:.2f} ({remaining_btc:.6f} BTC)")
            remaining_btc = 0
            full_exit_reason = "STOP_LOSS"
            break
            
        # Check Targets
        for target in targets:
            if not target['filled'] and current_high >= target['price']:
                # Execute partial exit
                exit_amount = total_btc_amount * target['pct']
                if exit_amount > remaining_btc: exit_amount = remaining_btc # Cap
                
                exit_value = exit_amount * target['price']
                cost_basis = exit_amount * entry_price
                pnl_realized += exit_value - cost_basis
                
                remaining_btc -= exit_amount
                target['filled'] = True
                exit_events.append(f"{target['name']} @ {target['price']:.2f}")
                
        if remaining_btc < 0.00000001: # Effectively zero
            full_exit_reason = "ALL_TARGETS_HIT"
            break
    
    # If still holding at end (Time Limit), close position
    if remaining_btc > 0:
        exit_value = remaining_btc * current_close
        cost_basis = remaining_btc * entry_price
        pnl_realized += exit_value - cost_basis
        exit_events.append(f"TIME_LIMIT @ {current_close:.2f}")
        remaining_btc = 0
        
    pnl_pct = (pnl_realized / total_position_usd) * 100
    
    # Calculate average exit price (derived from PnL)
    # PnL = (AvgExit - Entry) * TotalAmount
    # PnL / TotalAmount = AvgExit - Entry
    # AvgExit = (PnL / TotalAmount) + Entry
    avg_exit_price = (pnl_realized / total_btc_amount) + entry_price
    
    hold_duration = (last_exit_time - entry_time).total_seconds() / 3600
    
    return {
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_time': last_exit_time,
        'exit_price': avg_exit_price, # Weighted average
        'pnl': pnl_realized,
        'pnl_pct': pnl_pct,
        'balance_after': current_balance + pnl_realized,
        'hold_hours': hold_duration,
        'reason': full_exit_reason
    }

if __name__ == "__main__":
    print("Testing Strategy 4 module...")
    pass
