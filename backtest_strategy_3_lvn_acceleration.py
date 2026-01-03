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
    Execute a single trade using Strategy 3: Hybrid LVN Accel PPT + Trail
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
    exit_reason = "HOLD_LIMIT"
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
                exit_reason = "STOP_LOSS" if not ppt_hit else "TRAIL_STOP"
                break
        else:
            if current_low <= current_stop:
                pnl_realized += remaining_btc * (current_stop - entry_price)
                remaining_btc = 0
                exit_reason = "STOP_LOSS" if not ppt_hit else "TRAIL_STOP"
                break
            
        # 2. PPT
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
                current_stop = entry_price # BE
                
        # 3. Update Trail
        if ppt_hit and (i - vp_last_updated >= 10):
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
            exit_reason = "ALL_TARGETS"
            break
            
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

def get_trade_setup(df, entry_idx):
    """Generate trade setup preview with volume profile data"""
    entry_price = df['close'].iloc[entry_idx]
    entry_time = df.index[entry_idx]
    is_sell = bool(df['nn_sell_alarm'].iloc[entry_idx]) if 'nn_sell_alarm' in df.columns else False
    
    # Calculate ATR for stop loss
    start_atr = max(0, entry_idx - 20)
    atr_window = df.iloc[start_atr:entry_idx+1].copy()
    atr_series = calculate_atr(atr_window, period=14)
    current_atr = atr_series.iloc[-1]
    
    if np.isnan(current_atr) or current_atr == 0:
        current_atr = entry_price * 0.01
    
    atr_multiplier = 4.5
    stop_distance = current_atr * atr_multiplier
    stop_loss = entry_price + stop_distance if is_sell else entry_price - stop_distance
    stop_loss_pct = abs((stop_loss - entry_price) / entry_price * 100)
    
    # Calculate Volume Profile
    # Reverted to 80 bins as requested, enabled verbose for debugging
    vp = calculate_volume_profile(df, start_idx=max(0, entry_idx-200), end_idx=entry_idx, precise=True, num_bins=80, verbose=True)
    
    # Get NN confidence if available
    nn_confidence = None
    if is_sell and 'nn_sell_confidence' in df.columns:
        nn_confidence = df['nn_sell_confidence'].iloc[entry_idx]
    elif 'nn_buy_confidence' in df.columns:
        nn_confidence = df['nn_buy_confidence'].iloc[entry_idx]
    
    # Determine take profit targets (Double Distribution Logic)
    take_profit_candidates = []
    if vp:
        # Helper to get volume for a price
        def get_vol(p):
            idx = np.abs(vp['bin_centers'] - p).argmin()
            return vp['profile'].iloc[idx]

        if is_sell:
            # GMM 2-Peak Detection
            from volume_profile import find_dual_distribution_peaks
            gmm_peaks = find_dual_distribution_peaks(vp['profile'], vp['bin_centers'])
            
            # Find the peak that is significantly BELOW entry
            targets_below = [p for p in gmm_peaks if p < entry_price * 0.995]
            
            if targets_below:
                # If multiple peaks below (unlikely with n=2), take the lowest one
                take_profit_candidates = [min(targets_below)]
            else:
                # Fallback: if GMM didn't find a lower peak, check standard clusters
                peaks_below = [p for p in vp['cluster_prices'] if p < entry_price * 0.995]
                if peaks_below:
                     take_profit_candidates = [sorted(peaks_below, key=get_vol, reverse=True)[0]]
                else:
                    take_profit_candidates = [entry_price * 0.96]

        else: # BUY
            # GMM 2-Peak Detection
            from volume_profile import find_dual_distribution_peaks
            gmm_peaks = find_dual_distribution_peaks(vp['profile'], vp['bin_centers'])
            
            # Find the peak that is significantly ABOVE entry
            targets_above = [p for p in gmm_peaks if p > entry_price * 1.005]
            
            if targets_above:
                # If multiple peaks above, take the highest one
                take_profit_candidates = [max(targets_above)]
            else:
                # Fallback
                peaks_above = [p for p in vp['cluster_prices'] if p > entry_price * 1.005]
                if peaks_above:
                    take_profit_candidates = [sorted(peaks_above, key=get_vol, reverse=True)[0]]
                else:
                    take_profit_candidates = [entry_price * 1.04]
    
    # Prepare volume profile data for mini-chart
    vp_data = None
    if vp:
        vp_data = {
            'prices': vp['bin_centers'].tolist(),  # Use bin_centers for X-axis prices
            'volumes': vp['profile'].values.tolist(),
            'poc': vp['poc_price'],
            'vah': vp['vah'],
            'val': vp['val']
        }
        
        # Use GMM PDF for the "orange line" to show the idealized 2-peak fit
        try:
            from volume_profile import get_dual_distribution_pdf
            pdf_curve = get_dual_distribution_pdf(vp['profile'], vp['bin_centers'])
            if pdf_curve is not None:
                 vp_data['volumes_smooth'] = pdf_curve.tolist()
        except ImportError:
             pass

    return {
        'timestamp': entry_time,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'stop_loss_pct': stop_loss_pct,
        'is_sell': is_sell,
        'take_profit_candidates': take_profit_candidates,
        'nn_confidence': nn_confidence,
        'vp_data': vp_data,
        'description': f"Hybrid Strategy 3: LVN Acceleration | {'SELL' if is_sell else 'BUY'} Signal"
    }

