import pandas as pd
import numpy as np
from volume_profile import calculate_volume_profile, is_in_lvn

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
    Execute a single trade using Strategy 3: LVN Acceleration
    Exit before entering Low Volume Node (air pocket)
    """
    entry_price = df['close'].iloc[entry_idx]
    entry_time = df.index[entry_idx]
    
    # Calculate position size
    position_usd = current_balance * position_size_pct
    btc_amount = position_usd / entry_price
    
    # Dynamic ATR Stop Loss
    # Calculate ATR up to entry point
    # Optimization: Calculate only needed window to avoid full DF recalc
    start_atr = max(0, entry_idx - 20)
    atr_window = df.iloc[start_atr:entry_idx+1].copy()
    atr_series = calculate_atr(atr_window, period=14)
    current_atr = atr_series.iloc[-1]
    
    if np.isnan(current_atr) or current_atr == 0:
        current_atr = entry_price * 0.01 # Fallback 1% if no data
        
    atr_multiplier = 4.5
    stop_distance = current_atr * atr_multiplier
    stop_price = entry_price - stop_distance
    stop_loss_pct = stop_distance / entry_price
    
    max_hold_candles = 24 * 4 * 7  # 7 days (15m candles)
    
    # Simulation state
    exit_price = entry_price # Default
    exit_time = entry_time
    exit_reason = "HOLD_LIMIT"
    exit_idx = entry_idx
    vp_last_updated = -999
    lvn_prices = []
    
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
            
        # Strategy Logic: Update VP every 20 bars to check for Clusters/LVNs ahead
        if i - vp_last_updated >= 20:
            vp = calculate_volume_profile(df, start_idx=max(0, i-200), end_idx=i, precise=precise)
            if vp:
                # Target Logic: Find next major Cluster Center (Peak) strictly ABOVE current price
                # If currently inside a cluster, we want the connection to the NEXT one.
                clusters = [c for c in vp['cluster_prices'] if c > current_close * 1.005] # Min 0.5% distance
                clusters.sort()
                
                # If no clusters found, fallback to LVNs (Air Pockets)
                if not clusters:
                     lvn_prices = [p for p in vp['lvn_prices'] if p > current_close]
                     lvn_prices.sort()
                     lvn_prices = lvn_prices
                else:
                    lvn_prices = clusters # Use clusters as primary interaction points for this strategy variant
            
            vp_last_updated = i
            
        # Check Exit Trigger: Touching Target
        triggered_target = None
        for target in lvn_prices:
            # Check if we hit the target (within 0.1%)
            if current_high >= target * 0.999:
                triggered_target = target
                break
        
        # Only exit if profitable (at least 1%)
        if triggered_target is not None and triggered_target > entry_price * 1.01:
            exit_price = triggered_target 
            exit_time = current_time
            exit_reason = "CLUSTER_TARGET" if "cluster" in str(lvn_prices) else "LVN_TARGET" # Loose debug
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
        'reason': exit_reason
    }

def get_trade_setup(df, entry_idx):
    """
    Get the trade setup details for visualization/debugging without executing simulation.
    """
    entry_price = df['close'].iloc[entry_idx]
    entry_time = df.index[entry_idx]
    
    # Calculate Dynamic ATR Stop Loss
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
    
    # Calculate VP Context
    # Lookback 400 candles before entry (increased from 200 to capture wider range)
    vp_start_idx = max(0, entry_idx-400)
    vp = calculate_volume_profile(df, start_idx=vp_start_idx, end_idx=entry_idx, verbose=True, precise=True)
    
    # Debug: Show VP calculation window
    vp_start_time = df.index[vp_start_idx]
    vp_end_time = df.index[entry_idx]
    vp_price_range = (df['low'].iloc[vp_start_idx:entry_idx].min(), df['high'].iloc[vp_start_idx:entry_idx].max())
    print(f"DEBUG VP Window: {vp_start_time} to {vp_end_time} ({entry_idx - vp_start_idx} candles)")
    print(f"DEBUG VP Price Range: {vp_price_range[0]:.2f} to {vp_price_range[1]:.2f}")
    print(f"DEBUG Entry: {entry_time} at {entry_price:.2f}")
    
    # Target Selection: Find LVN between GMM peaks
    # Since VP is calculated on 200 candles BEFORE entry, we need to find the gap
    # between the two volume clusters that price will need to cross
    
    tp_target = None
    tp_desc = None
    
    # Generate GMM to extract peak locations and find the gap between them
    from sklearn.mixture import GaussianMixture
    from scipy.optimize import minimize_scalar
    from scipy.stats import norm
    
    volume_profile = vp['profile']
    bin_centers = vp['bin_centers']
    
    # Prepare weighted data for GMM
    prices_weighted = []
    for i, vol in enumerate(volume_profile.values):
        if vol > 0:
            count = int(vol / volume_profile.sum() * 1000)
            prices_weighted.extend([bin_centers[i]] * max(1, count))
    
    if len(prices_weighted) >= 10:
        X = np.array(prices_weighted).reshape(-1, 1)
        
        # Fit 2-component GMM
        try:
            gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42, max_iter=100)
            gmm.fit(X)
            
            # Get the two peak locations (means)
            peaks = sorted(gmm.means_.flatten())
            lower_peak, upper_peak = peaks[0], peaks[1]
            
            print(f"DEBUG: GMM Peaks - Lower: {lower_peak:.2f}, Upper: {upper_peak:.2f}, Entry: {entry_price:.2f}")
            
            # Use upper peak if it's above entry and within reasonable range
            if entry_price < upper_peak < entry_price * 1.08:
                tp_target = upper_peak
                tp_desc = f"Upper GMM Peak: {tp_target:.2f}"
                print(f"DEBUG: Using upper GMM peak as target")
            else:
                print(f"DEBUG: Upper peak {upper_peak:.2f} not suitable (entry: {entry_price:.2f})")
        except Exception as e:
            print(f"DEBUG: GMM fitting failed: {e}")
            pass
    
    # Fallback logic if GMM fails
    if tp_target is None:
        # Try LVN as fallback (0.5% to 8% range)
        lvns = [l for l in vp['lvn_prices'] if entry_price * 1.005 < l < entry_price * 1.08]
        lvns.sort()
        
        if lvns:
            tp_target = lvns[0]
            tp_desc = f"LVN Gap: {tp_target:.2f}"
        else:
            # Try clusters (0.5% to 8% range)
            clusters = [c for c in vp['cluster_prices'] if entry_price * 1.005 < c < entry_price * 1.08]
            
            if clusters:
                cluster_volumes = {}
                for cluster_price in clusters:
                    bin_idx = np.argmin(np.abs(vp['bin_centers'] - cluster_price))
                    cluster_volumes[cluster_price] = vp['profile'].iloc[bin_idx]
                
                tp_target = max(cluster_volumes, key=cluster_volumes.get)
                tp_desc = f"Cluster: {tp_target:.2f} (Vol: {cluster_volumes[tp_target]:.0f})"
            else:
                # Last resort
                tp_target = entry_price * 1.03
                tp_desc = "Fixed 3% (No Structure)"

    # Get NN Confidence: Search back 20 candles to find the 'W' pattern peak that confirmed the entry
    nn_conf = 0.0
    if 'nn_confidence' in df.columns:
        # Search back up to 20 candles for the confirming pattern
        lookback_range = df['nn_confidence'].iloc[max(0, entry_idx-20):entry_idx+1]
        nn_conf = float(lookback_range.max())

    setup = {
        'timestamp': entry_time,
        'entry_price': entry_price,
        'stop_loss': stop_price,
        'stop_loss_pct': stop_loss_pct * 100,
        'nn_confidence': nn_conf,
        'take_profit_candidates': [tp_target],
        'description': f"Entry: {entry_price:.2f} | NN Conf: {nn_conf*100:.1f}% | SL: {stop_price:.2f} ({stop_loss_pct*100:.1f}%) | TP: {tp_desc}",
        'vp_data': {
            'prices': vp['bin_centers'].tolist(),  # Use actual price levels
            'volumes': vp['profile'].values.tolist(),
            'volumes_smooth': generate_gmm_curve(vp),  # Generate GMM curve on-demand
            'poc': vp['poc_price']
        }
    }
    
    # Debug: Check for zero-volume bins
    volumes = vp['profile'].values
    zero_bins = sum(1 for v in volumes if v == 0)
    near_zero_bins = sum(1 for v in volumes if 0 < v < volumes.max() * 0.01)
    print(f"DEBUG VP: Total bins: {len(volumes)}, Zero bins: {zero_bins}, Near-zero bins: {near_zero_bins}")
    print(f"DEBUG VP: Volume range: {volumes.min():.0f} to {volumes.max():.0f}")
        
    return setup

def generate_gmm_curve(vp):
    """Generate adaptive GMM fitted curve for visualization (expensive, only on marker click)"""
    from sklearn.mixture import GaussianMixture
    from scipy.stats import norm
    
    volume_profile = vp['profile']
    bin_centers = vp['bin_centers']
    
    # Prepare weighted data
    prices_weighted = []
    for i, vol in enumerate(volume_profile.values):
        if vol > 0:
            count = int(vol / volume_profile.sum() * 1000)
            prices_weighted.extend([bin_centers[i]] * max(1, count))
    
    if len(prices_weighted) < 10:
        return vp['profile_smooth'].tolist()  # Fallback to simple smooth
    
    X = np.array(prices_weighted).reshape(-1, 1)
    
    # Strongly prefer 2 components (typical VP pattern: accumulation + distribution)
    # Only try 2-3 components with heavy penalty for 3
    best_gmm = None
    best_score = np.inf
    best_n = 2
    
    for n in [2, 3]:  # Only try 2 or 3 peaks
        try:
            gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42, max_iter=100)
            gmm.fit(X)
            # Add strong complexity penalty to prefer 2 components
            complexity_penalty = (n - 2) * 1000  # Heavy penalty for n=3
            score = gmm.bic(X) + complexity_penalty
            if score < best_score:
                best_score = score
                best_gmm = gmm
                best_n = n
        except:
            continue
    
    if best_gmm is not None:
        # Properly reconstruct GMM curve by summing weighted Gaussian PDFs
        profile_smooth = np.zeros(len(bin_centers))
        
        for i in range(best_n):
            mean = best_gmm.means_[i][0]
            std = np.sqrt(best_gmm.covariances_[i][0][0])
            weight = best_gmm.weights_[i]
            
            # Evaluate Gaussian PDF for each bin center
            gaussian_pdf = norm.pdf(bin_centers, loc=mean, scale=std)
            profile_smooth += weight * gaussian_pdf
        
        # Normalize to match volume scale
        profile_smooth = profile_smooth / profile_smooth.max() * volume_profile.max()
        return profile_smooth.tolist()
    
    return vp['profile_smooth'].tolist()

if __name__ == "__main__":
    print("Testing Strategy 3 module...")
    pass
