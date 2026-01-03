import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Strategy Modules
try:
    import backtest_strategy_1_poc_target as strategy_1
    import backtest_strategy_2_vah_exit as strategy_2
    import backtest_strategy_3_lvn_acceleration as strategy_3
    import backtest_strategy_4_multi_tier as strategy_4
    import backtest_strategy_5_trailing_stop as strategy_5
    import backtest_strategy_6_volume_divergence as strategy_6
    STRATEGIES_AVAILABLE = True
except ImportError as e:
    print(f"Strategy modules import error: {e}")
    STRATEGIES_AVAILABLE = False

from scipy.stats import linregress
import torch
import torch.nn as nn

def calculate_stochastic(df, period_k, smooth_k, smooth_d):
    """Calculates Stochastic %D line"""
    low_min = df['low'].rolling(window=period_k, min_periods=1).min()
    high_max = df['high'].rolling(window=period_k, min_periods=1).max()
    k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))

    # Smooth K
    if smooth_k > 1:
        k_percent = k_percent.rolling(window=smooth_k, min_periods=1).mean()

    # Smooth D (This is the signal line we plot)
    d_percent = k_percent.rolling(window=smooth_d, min_periods=1).mean()
    return d_percent

# -------------------------------------------------------------------------
# NEURAL NETWORK PATTERN DETECTION (RTX 5090 Optimized Inference)
# -------------------------------------------------------------------------
import time
from threading import Lock
from volume_profile import calculate_volume_profile, get_raw_volume_histogram
class PatternDetectorCNN(nn.Module):
    def __init__(self, window_size=100, vp_bins=80):
        super(PatternDetectorCNN, self).__init__()
        
        # --- Branch A: Time Series (Stochastic) ---
        self.ts_features = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(4)
        )
        
        # --- Branch B: Market Structure (Volume Profile) ---
        self.vp_features = nn.Sequential(
            nn.Linear(vp_bins, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # --- Fusion Head ---
        combined_dim = (512 * 4) + 64 
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 3) # 3 Classes: 0=Hold, 1=Buy, 2=Sell
        )

    def forward(self, x_ts, x_vp):
        f_ts = self.ts_features(x_ts)
        f_ts = f_ts.view(f_ts.size(0), -1)
        f_vp = self.vp_features(x_vp)
        combined = torch.cat((f_ts, f_vp), dim=1)
        return self.classifier(combined)

from threading import Lock

_NN_MODEL = None
_NN_CACHE = {}  # Cache for NN detection results: {(filepath, data_len, threshold): results}
_NN_LOCK = Lock()

def get_nn_model():
    global _NN_MODEL
    # Double-checked locking pattern for thread safety
    if _NN_MODEL is None:
        with _NN_LOCK:
            if _NN_MODEL is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # Optimization: Load model tailored for 5090 inference
                model_path = os.path.join(os.path.dirname(__file__), "stoch_vp_unified_15m.pth")
                
                if os.path.exists(model_path):
                    try:
                        # Initialize Model Architecture (Must match training script)
                        model = PatternDetectorCNN().to(device)
                        state_dict = torch.load(model_path, map_location=device, weights_only=True)
                        
                        # Fix for DataParallel keys if training used it (just in case)
                        new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                        model.load_state_dict(new_state_dict)
                        model.to(device)
                        model.eval()
                        _NN_MODEL = model
                        print(f"NN Unified Model (3-Class) loaded successfully on {device}")
                    except Exception as e:
                        print(f"Error loading NN Model: {e}")
    return _NN_MODEL

def identify_nn_patterns(df, nn_threshold=30):
    """Detects 'W' patterns using the Dual-Input Neural Network"""
    # Load model
    model = get_nn_model()
    if model is None:
        df['nn_alarm'] = False
        return df

    device = next(model.parameters()).device
    
    # --- CACHE LOOKUP ---
    # Key: (Last Timestamp, Total Length, Threshold)
    if not df.empty:
        cache_key = (df.index[-1], len(df), nn_threshold)
        if cache_key in _NN_CACHE:
            cached_data = _NN_CACHE[cache_key]
            # Handle new cache format: (buy_alarm, sell_alarm, buy_confidence, sell_confidence)
            if isinstance(cached_data, tuple) and len(cached_data) == 4:
                df['nn_buy_alarm'], df['nn_sell_alarm'], df['nn_buy_confidence'], df['nn_sell_confidence'] = cached_data
                # For backward compatibility, map buy_alarm to nn_alarm
                df['nn_alarm'] = df['nn_buy_alarm']
                df['nn_confidence'] = df['nn_buy_confidence']
            return df
    
    # Stochastic Calculation (Same as training)
    period_k = 60
    smooth_k = 10
    low_min = df['low'].rolling(window=period_k).min()
    high_max = df['high'].rolling(window=period_k).max()
    stoch = 100 * ((df['close'] - low_min) / (high_max - low_min))
    stoch_vals = (stoch.rolling(window=smooth_k).mean() / 100.0).values
    
    nn_buy_results = np.zeros(len(df), dtype=bool)
    nn_sell_results = np.zeros(len(df), dtype=bool)
    nn_buy_confidences = np.zeros(len(df), dtype=float)
    nn_sell_confidences = np.zeros(len(df), dtype=float)
    
    window_size = 100
    # Optimization: For live chart interaction, we DO NOT need to calculate NN for the entire history (years of data).
    # We only need it for the recent view + some buffer.
    # Let's process only the last 2000 bars (approx 20 days of 15m data).
    # If users scroll back further, they won't see NN markers, which is acceptable for performance.
    
    SAFE_LOOKBACK = 3000 
    start_idx = max(200, len(df) - SAFE_LOOKBACK)
    
    # Batch Processing to handle Volume Profile speed
    valid_indices = []
    ts_windows = []
    vp_vectors = []
    
    print(f"Generating NN signals for last {len(df) - start_idx} candles (Optimization Active)...")
    
    t_loop_start = time.time()
    t_vp_total = 0
    t_prep_total = 0
    
    for i in range(start_idx, len(df)):
        t_p_start = time.time()
        # Stoch Window
        win_stoch = stoch_vals[i-window_size:i]
        
        if not np.isnan(win_stoch).any():
             # VP Window (200 bars)
             t_v_start = time.time()
             # Optimization: Use lightweight histogram for NN instead of full VP analysis
             vp_profile = get_raw_volume_histogram(df, i-200, i, num_bins=80)
             t_vp_total += (time.time() - t_v_start)
             
             if len(vp_profile) == 80:
                 vp_max = vp_profile.max()
                 if vp_max > 0:
                     vp_norm = vp_profile / vp_max
                     
                     valid_indices.append(i)
                     ts_windows.append(win_stoch)
                     vp_vectors.append(vp_norm)
        t_prep_total += (time.time() - t_p_start)
    
    t_loop_end = time.time()
    print(f"DEBUG: NN Loop Timing Breakdown (Total {t_loop_end - t_loop_start:.4f}s):")
    print(f"  - Total Data Prep/Stoch: {t_prep_total - t_vp_total:.4f}s")
    print(f"  - Total VP Calculation:  {t_vp_total:.4f}s")
    
    if ts_windows:
        t_inf_start = time.time()
        with torch.no_grad():
            has_cuda = torch.cuda.is_available()
            BATCH_SIZE = 1024 # Larger batch for GPU
            
            probs_all = []
            
            # Process in Batches
            for k in range(0, len(ts_windows), BATCH_SIZE):
                batch_ts = np.array(ts_windows[k:k+BATCH_SIZE])
                batch_vp = np.array(vp_vectors[k:k+BATCH_SIZE])
                
                with torch.amp.autocast('cuda') if has_cuda else torch.device('cpu'):
                    t_ts = torch.FloatTensor(batch_ts).unsqueeze(1).to(device)
                    t_vp = torch.FloatTensor(batch_vp).to(device)
                    
                    logits = model(t_ts, t_vp)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    probs_all.extend(probs)
            
            probs_all = np.array(probs_all) # Shape: [N, 3]
            
            # --- AGGRESSIVE SUPPRESSION LOGIC (NMS) ---
            threshold = nn_threshold / 100.0 
            
            buy_probs = probs_all[:, 1]
            sell_probs = probs_all[:, 2]
            
            final_buy = np.zeros_like(buy_probs, dtype=bool)
            final_sell = np.zeros_like(sell_probs, dtype=bool)
            cooldown = 30
            
            for idx in range(len(probs_all)):
                p_buy = buy_probs[idx]
                p_sell = sell_probs[idx]
                
                # Check BUY peak
                if p_buy > threshold:
                    start = max(0, idx - cooldown)
                    end = min(len(buy_probs), idx + cooldown + 1)
                    if p_buy == np.max(buy_probs[start:end]):
                        final_buy[idx] = True
                
                # Check SELL peak
                if p_sell > threshold:
                    start = max(0, idx - cooldown)
                    end = min(len(sell_probs), idx + cooldown + 1)
                    if p_sell == np.max(sell_probs[start:end]):
                        final_sell[idx] = True
            
            nn_buy_results[valid_indices] = final_buy
            nn_sell_results[valid_indices] = final_sell
            nn_buy_confidences[valid_indices] = buy_probs
            nn_sell_confidences[valid_indices] = sell_probs
            
            # Debug: Print detection stats
            print(f"NN Unified Detection: Found {np.sum(final_buy)} BUY, {np.sum(final_sell)} SELL patterns")
            
            # Show top 5 confidence scores for BUY signals
            if len(buy_probs) > 0:
                top_indices = np.argsort(buy_probs)[-5:][::-1]
                print(f"Top 5 NN BUY confidence scores:")
                for idx in top_indices:
                    actual_idx = valid_indices[idx]
                    timestamp = df.index[actual_idx]
                    print(f"  - {timestamp}: {buy_probs[idx]*100:.1f}% confidence")
            
            t_inf_total = time.time() - t_inf_start
            print(f"  - Total Model Inference: {t_inf_total:.4f}s")
                
    df['nn_buy_alarm'] = nn_buy_results
    df['nn_sell_alarm'] = nn_sell_results
    df['nn_buy_confidence'] = nn_buy_confidences
    df['nn_sell_confidence'] = nn_sell_confidences
    
    # Backward compatibility
    df['nn_alarm'] = nn_buy_results
    df['nn_confidence'] = nn_buy_confidences
    
    # --- CACHE STORAGE ---
    if not df.empty:
        cache_key = (df.index[-1], len(df), nn_threshold)
        _NN_CACHE[cache_key] = (nn_buy_results, nn_sell_results, nn_buy_confidences, nn_sell_confidences)
        
        # Limit cache size to prevent memory leaks (keep last 20)
        if len(_NN_CACHE) > 20:
             # Remove oldest item (simple FIFO not strictly ordered dict but ok for this)
             # Python 3.7+ dicts preserve insertion order, so next(iter(d)) is oldest.
             first_key = next(iter(_NN_CACHE))
             del _NN_CACHE[first_key]
             
    return df

def is_pivot(df, candle, window):
    """Detects if a candle is a pivot point (fractal)."""
    if candle - window < 0 or candle + window >= len(df):
        return 0
    
    pivot_high = 1
    pivot_low = 2
    for i in range(candle - window, candle + window + 1):
        if df['low'].iloc[candle] > df['low'].iloc[i]:
            pivot_low = 0
        if df['high'].iloc[candle] < df['high'].iloc[i]:
            pivot_high = 0
    
    if pivot_high and pivot_low: return 3
    if pivot_high: return 1
    if pivot_low: return 2
    return 0

def calculate_classic_channel(df, window=1, backcandles=35):
    """
    Calculates an ultra-sensitive parallel downward channel using pivot points.
    1. Finds pivot HIGHS (window=1) and fits a resistance line (slope).
    2. Defines a parallel support line touching the lowest low in the window.
    3. Optimized for capturing FULL steep descents (30-40 bars).
    """
    df['linreg_mid'] = np.nan
    df['linreg_top'] = np.nan
    df['linreg_bot'] = np.nan
    df['slope'] = np.nan
    
    l_mid = np.full(len(df), np.nan)
    l_top = np.full(len(df), np.nan)
    l_bot = np.full(len(df), np.nan)
    slopes = np.full(len(df), np.nan)
    
    # Vectorized Pivot Detection for HIGHS (Resistance)
    w_size = 2 * window + 1
    roll_high = df['high'].rolling(window=w_size, center=True).max()
    is_p_high = (df['high'] == roll_high).values
    
    low_vals = df['low'].values
    high_vals = df['high'].values
    indices = np.arange(len(df))
    
    for i in range(backcandles + window, len(df)):
        start = i - backcandles - window
        end = i - window
        mask_h = is_p_high[start:end]
        idx_p_high = indices[start:end][mask_h]
        vals_p_high = high_vals[start:end][mask_h]
        
        if len(idx_p_high) >= 2:
            try:
                # Resistance line (top) - derivation from local extreme highs
                slope, intercept, _, _, _ = linregress(idx_p_high, vals_p_high)
                
                # Downward slope only
                if slope < 0:
                    cur_top = slope * i + intercept
                    
                    # Parallel bottom line offset by min price distance in lookback
                    win_indices = np.arange(start, i)
                    baselines = slope * win_indices + intercept
                    offset = np.min(low_vals[start:i] - baselines)
                    
                    l_top[i] = cur_top
                    l_bot[i] = cur_top + offset
                    l_mid[i] = cur_top + (offset / 2)
                    slopes[i] = slope
            except: continue

    df['linreg_mid'] = l_mid
    df['linreg_top'] = l_top
    df['linreg_bot'] = l_bot
    df['slope'] = slopes
    return df

def detect_channel_breakout(df):
    """Detects when price breaks above the parallel classic channel top"""
    if 'linreg_top' not in df.columns or 'slope' not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    
    # Breakout criteria:
    # 1. Slope is negative (Down channel)
    # 2. Close is above the top of the channel
    # 3. Previous close was below/inside
    is_downtrend = df['slope'] < 0  # Classic channel slope is more sensitive
    is_above_top = df['close'] > df['linreg_top']
    was_below_top = df['close'].shift(1) <= df['linreg_top'].shift(1)
    
    return is_downtrend & is_above_top & was_below_top

def identify_hybrid_signals(df, nn_threshold=30):
    """
    Combines Classic Channel Breakouts with NN 'W' confirmation.
    Logic: Channel Establishment -> Breakout -> shallow W + Low Stochs.
    """
    if 'nn_alarm' not in df.columns:
        df = identify_nn_patterns(df, nn_threshold=nn_threshold)
    
    # Pre-calculate breakouts
    breakout_series = detect_channel_breakout(df)
    
    hybrid_signals = np.zeros(len(df), dtype=bool)
    
    # We now trigger on the BREAKOUT point if a 'W' was formed recently as confirmation.
    breakout_indices = df.index[breakout_series]
    
    for br_time in breakout_indices:
        # Look back for a 'W' confirmation in the last 20 candles
        start_search = br_time - pd.Timedelta(minutes=15 * 20)
        nn_in_window = df.loc[start_search:br_time, 'nn_alarm']
        
        if nn_in_window.any():
            # Found a 'W' followed by a breakout!
            # We must also check if stochastics were LOW ( < 40 ) at the time of the 'W'
            nn_times = nn_in_window.index[nn_in_window]
            for nn_time in nn_times:
                s_low = (df.loc[nn_time, 'stoch_60_10'] < 40) and (df.loc[nn_time, 'stoch_40_4'] < 40)
                if s_low:
                    hybrid_signals[df.index == br_time] = True
                    break

    df['hybrid_alarm'] = hybrid_signals
    return df

def identify_quad_rotation_alarms(df, nn_threshold=30):
    """
    Main aggregator: Calculates stochastics, classic channel, and identifies hybrid signals.
    Uses caching to avoid redundant NN detection on unchanged data.
    """
    global _NN_CACHE
    
    # Create cache key based on data fingerprint
    cache_key = (len(df), str(df.index[-1]) if len(df) > 0 else '', nn_threshold)
    
    # Check if we have cached results for this exact data
    if cache_key in _NN_CACHE:
        cached_df = _NN_CACHE[cache_key]
        # Verify cache is still valid (same index)
        if len(cached_df) == len(df) and (cached_df.index == df.index).all():
            print(f"[CACHE HIT] Using cached NN detection results (threshold={nn_threshold}%)")
            # Copy cached columns to current df
            cols = ['nn_buy_alarm', 'nn_sell_alarm', 'hybrid_alarm', 'alarm', 
                    'stoch_9_3', 'stoch_14_3', 'stoch_40_4', 'stoch_60_10', 'slope', 
                    'nn_buy_confidence', 'nn_sell_confidence']
            for col in cols:
                if col in cached_df.columns:
                    df[col] = cached_df[col]
            # Legacy mapping
            df['nn_alarm'] = df['nn_buy_alarm']
            df['nn_confidence'] = df['nn_buy_confidence']
            return df
    
    print(f"[CACHE MISS] Running full detection (threshold={nn_threshold}%)")
    
    # Calculate stochastics
    df['stoch_9_3'] = calculate_stochastic(df, 9, 1, 3)
    df['stoch_14_3'] = calculate_stochastic(df, 14, 1, 3)
    df['stoch_40_4'] = calculate_stochastic(df, 40, 1, 4)
    df['stoch_60_10'] = calculate_stochastic(df, 60, 10, 10)

    # Calculate Classic Price Channel (two lows + parallel shift)
    df = calculate_classic_channel(df)
    
    # Identify Quad Alarms (Old rule-based logic for comparison)
    cond_downtrend = df['slope'] < -0.5
    cond_quad_low = (df['stoch_9_3'] < 20) & (df['stoch_14_3'] < 25) & (df['stoch_40_4'] < 25) & (df['stoch_60_10'] < 25)
    cond_turn_up = df['stoch_9_3'] > df['stoch_9_3'].shift(1)
    df['alarm'] = cond_downtrend & cond_quad_low & cond_turn_up
    
    # Identify Hybrid Signals (New high-conviction logic)
    df = identify_hybrid_signals(df, nn_threshold=nn_threshold)
    
    # Cache the results
    _NN_CACHE[cache_key] = df.copy()
    
    # Limit cache size to prevent memory issues (keep last 10 unique datasets)
    if len(_NN_CACHE) > 10:
        # Remove oldest entry
        oldest_key = next(iter(_NN_CACHE))
        del _NN_CACHE[oldest_key]
        print(f"[CACHE] Evicted oldest entry, cache size: {len(_NN_CACHE)}")
    
    return df



def generate_chart_data(filepath, symbol, timeframe, num_candles=100, start_date=None, end_date=None, nn_threshold=30, exit_strategy_id=3):
    """
    Generate interactive OHLC chart data with stochastics using Plotly
    Returns JSON data for Plotly chart
    
    Args:
        nn_threshold: Neural network confidence threshold (0-100), default 30
    """
    import time
    t_start = time.time()
    # Read data
    df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
    t_load = time.time() - t_start

    # OPTIMIZATION: Slice DataFrame for processing
    process_start_index = 0
    buffer_size = 2000 # Increased from 500 to 2000 to ensure robust indicator calculation
    
    if start_date:
        try:
            s_ts = pd.to_datetime(start_date)
            if df.index.tz is None and s_ts.tz is not None:
                s_ts = s_ts.tz_localize(None)
            mask_idx = np.searchsorted(df.index, s_ts)
            process_start_index = max(0, mask_idx - buffer_size)
        except:
            process_start_index = max(0, len(df) - num_candles - buffer_size)
    else:
        process_start_index = max(0, len(df) - num_candles - buffer_size)

    # Create the working slice (copy to avoid SettingWithCopy warnings)
    df = df.iloc[process_start_index:].copy()
    # print(f"DEBUG: Processing slice from {df.index[0]} to {df.index[-1]} (Buffer={buffer_size})")

    # Calculate Indicators and Alarms on Working dataset
    t_alarms_start = time.time()
    df = identify_quad_rotation_alarms(df, nn_threshold=nn_threshold)
    t_alarms = time.time() - t_alarms_start
    t_alarms_end = t_alarms_start + t_alarms

    # -------------------------------------------------------------------------
    # Cross-Timeframe Logic (Show 1H Alarms on 15M Chart)
    # -------------------------------------------------------------------------
    df_1h_alarms = None
    debug_msg = "1H Overlay: Disabled"
    
    if timeframe == '15m':
        try:
            # Construct 1h filepath safely
            base_dir = os.path.dirname(filepath)
            filepath_1h = os.path.join(base_dir, f"{symbol}_1h_data.csv")
            
            if os.path.exists(filepath_1h):
                # Load 1h data
                df_1h = pd.read_csv(filepath_1h, index_col='timestamp', parse_dates=True)
                
                # Optimization: Slice 1h data to similar range as 15m view
                # INCREASED BUFFER to 500 to ensure Stochs have plenty of warm-up data
                if not df.empty:
                    start_time_limit = df.index[0]
                    slice_idx_1h = df_1h.index.searchsorted(start_time_limit)
                    start_idx_1h = max(0, slice_idx_1h - 500) 
                    df_1h = df_1h.iloc[start_idx_1h:].copy()

                # Calculate alarms on 1h
                t_1h_start = time.time()
                df_1h = identify_quad_rotation_alarms(df_1h, nn_threshold=nn_threshold)
                t_1h = time.time() - t_1h_start
                print(f"DEBUG: 1H Alarms calculation took {t_1h:.4f}s")
                
                # Filter for actual alarms
                df_1h_alarms = df_1h[df_1h['alarm']].copy()
                debug_msg = f"1H Overlay: Found {len(df_1h_alarms)} raw alarms"
            else:
                debug_msg = f"1H Overlay: File not found ({filepath_1h})"
        except Exception as e:
            debug_msg = f"1H Overlay Error: {str(e)}"
            print(f"Error loading 1h overlay: {e}")

    # Filter data based on parameters
    if start_date and end_date:
        # Date range filtering
        try:
            start_ts = pd.to_datetime(start_date)
            end_ts = pd.to_datetime(end_date)
            
            # Handle timezone mismatch
            if df.index.tz is None and start_ts.tz is not None:
                start_ts = start_ts.tz_localize(None)
                end_ts = end_ts.tz_localize(None)
                
            mask = (df.index >= start_ts) & (df.index <= end_ts)
            chart_df = df.loc[mask].copy()
            
            # Filter the 1h alarms to the view range too
            if df_1h_alarms is not None:
                 if df_1h_alarms.index.tz is None and start_ts.tz is not None:
                     pass
                 mask_1h = (df_1h_alarms.index >= start_ts) & (df_1h_alarms.index <= end_ts)
                 df_1h_alarms = df_1h_alarms.loc[mask_1h].copy()
                 debug_msg += f" | In View: {len(df_1h_alarms)}"

        except Exception as e:
            print(f"Error filtering dates: {e}")
            chart_df = df.tail(num_candles).copy()
        
        if len(chart_df) == 0:
             chart_df = df.tail(num_candles).copy()
    else:
        # Get last N candles
        chart_df = df.tail(num_candles).copy()
        if df_1h_alarms is not None and not chart_df.empty:
            # STRICT ALIGNMENT: Clamp 1H alarms to exactly the visible chart range
            min_ts = chart_df.index[0]
            max_ts = chart_df.index[-1]
            df_1h_alarms = df_1h_alarms[(df_1h_alarms.index >= min_ts) & (df_1h_alarms.index <= max_ts)].copy()
            debug_msg += f" | In View: {len(df_1h_alarms)}"

    # Create subplots: 1 for price+volume, 4 for stochastics
    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        row_heights=[0.40, 0.08, 0.10, 0.10, 0.10, 0.10],
    )

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=chart_df.index.strftime('%Y-%m-%dT%H:%M:%S'),
            open=chart_df['open'].tolist(),
            high=chart_df['high'].tolist(),
            low=chart_df['low'].tolist(),
            close=chart_df['close'].tolist(),
            name='OHLC',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            increasing_fillcolor='#26a69a',
            decreasing_fillcolor='#ef5350',
            line=dict(width=1),
            whiskerwidth=0.5
        ),
        row=1, col=1
    )
    t_plotly_start = time.time()
    


    # Calculate dynamic marker offset based on visible price range
    # This ensures markers are always visible regardless of price level
    price_range = chart_df['high'].max() - chart_df['low'].min()
    marker_offset = price_range * 0.015  # 1.5% of visible range
    
    # Add Alarm Markers (Current Timeframe)
    alarm_df = chart_df[chart_df['alarm']].copy()
    print(f"DEBUG: Found {len(alarm_df)} alarms in current view")
    
    # Distinct colors for different timeframes
    alarm_color = 'cyan' # Default (1h, 4h, etc)
    if timeframe == '15m':
        alarm_color = 'yellow'
        
    if not alarm_df.empty:
        fig.add_trace(
            go.Scatter(
                x=alarm_df.index.strftime('%Y-%m-%dT%H:%M:%S'),
                y=(alarm_df['low'] - marker_offset).tolist(), # Place just below candle low
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color=alarm_color),
                name=f'Quad Rotation Alarm (Timescale: {timeframe})',
                hoverinfo='x+y+name'
            ),
            row=1, col=1
        )

    # Add 1H Alarm Markers Overlay (if applicable)
    entry_points_df = None
    
    if df_1h_alarms is not None and not df_1h_alarms.empty:
         fig.add_trace(
            go.Scatter(
                x=df_1h_alarms.index.strftime('%Y-%m-%dT%H:%M:%S'),
                y=(df_1h_alarms['low'] - marker_offset * 1.5).tolist(), # Place slightly lower than current TF
                mode='markers',
                marker=dict(symbol='triangle-up', size=16, color='cyan'),
                name='Quad Rotation Alarm (Timescale: 1H)',
                hoverinfo='x+y+name'
            ),
            row=1, col=1
        )
         
         # ------------------------------------------------------------------
         # Calculate Entry Points (Green Dots) based on 1H Alarms
         # Strategy: After 1H Alarm, find first 15m candle where S40 < 20 and rising
         # ------------------------------------------------------------------
         entry_indices = []
         entry_prices = []
         
         # We need to scan 15m data for each 1H alarm
         # Ensure we have access to the full 'df' (which is the 15m data)
         
         for alarm_time in df_1h_alarms.index:
             # Look forward 24 hours max
             cutoff_time = alarm_time + pd.Timedelta(hours=24)
             
             # Slice 15m data after alarm
             # Note: df is the working slice of 15m data
             mask_entry = (df.index > alarm_time) & (df.index <= cutoff_time)
             potential_entries = df.loc[mask_entry].copy()
             
             if potential_entries.empty:
                 continue
                 
             # Vectorized check for entry condition on this slice
             # Condition: Stoch 40 < 20 AND Rising
             # We can pre-calculate these booleans for efficiency, or just loop this small slice
             
             # Let's simple loop to find the FIRST one (mimicking the script logic)
             found_entry = False
             for i in range(1, len(potential_entries)):
                 s40 = potential_entries['stoch_40_4'].iloc[i]
                 s40_prev = potential_entries['stoch_40_4'].iloc[i-1]
                 
                 if s40 < 20 and s40 > s40_prev:
                     # FOUND ENTRY
                     entry_time = potential_entries.index[i]
                     entry_price = potential_entries['close'].iloc[i]
                     
                     entry_indices.append(entry_time)
                     entry_prices.append(entry_price)
                     found_entry = True
                     break # Take only first valid entry per alarm
        
         if entry_indices:
             fig.add_trace(
                go.Scatter(
                    x=[t.strftime('%Y-%m-%dT%H:%M:%S') for t in entry_indices],
                    y=entry_prices,
                    mode='markers',
                    marker=dict(symbol='circle', size=10, color='#00FF00', line=dict(width=1, color='white')),
                    name='Best Entry (Timescale: 15m - S40 Turn)',
                    hoverinfo='x+y+name'
                ),
                row=1, col=1
            )

    # Add Neural Network detected patterns (Unified Model)
    if 'nn_buy_alarm' in chart_df.columns:
        # 1. BUY Patterns (Star markers)
        nn_buy_df = chart_df[chart_df['nn_buy_alarm'] & (~chart_df.get('hybrid_alarm', False))].copy()
        if not nn_buy_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=nn_buy_df.index.strftime('%Y-%m-%dT%H:%M:%S'),
                    y=(nn_buy_df['low'] - marker_offset * 1.5).tolist(),
                    mode='markers',
                    marker=dict(symbol='star', size=11, color='#00FF00', line=dict(width=1, color='white')),
                    name='NN: High-Conviction BUY',
                    hoverinfo='x+y+name'
                ),
                row=1, col=1
            )
            
        # 2. SELL Patterns (Star markers)
        nn_sell_df = chart_df[chart_df['nn_sell_alarm']].copy()
        if not nn_sell_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=nn_sell_df.index.strftime('%Y-%m-%dT%H:%M:%S'),
                    y=(nn_sell_df['high'] + marker_offset * 1.5).tolist(),
                    mode='markers',
                    marker=dict(symbol='star', size=11, color='#FF3131', line=dict(width=1, color='white')),
                    name='NN: High-Conviction SELL',
                    hoverinfo='x+y+name'
                ),
                row=1, col=1
            )

    # -------------------------------------------------------------------------
    # Add Hybrid Alarms (Trend Breakout + NN W)
    # -------------------------------------------------------------------------
    if 'hybrid_alarm' in chart_df.columns:
        hybrid_df = chart_df[chart_df['hybrid_alarm']].copy()
        if not hybrid_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=hybrid_df.index.strftime('%Y-%m-%dT%H:%M:%S'),
                    y=(hybrid_df['low'] - marker_offset * 2.5).tolist(), # Place below candle low
                    mode='markers',
                    marker=dict(
                        symbol='star', 
                        size=18, 
                        color='#00FF00', # Vibrand Green for confirmed
                        line=dict(width=2, color='white')
                    ),
                    name='👑 HYBRID SIGNAL: Trend Break + W-Confirmation',
                    hoverinfo='x+y+name'
                ),
                row=1, col=1
            )
    
    # -------------------------------------------------------------------------
    # Volume Profile Visualization - DISABLED PER USER REQUEST
    # -------------------------------------------------------------------------
    # try:
    #     # Import moved inside to avoid circular dependencies if any, 
    #     # but we should log clearly if it fails.
    #     import volume_profile
    #     from volume_profile import calculate_volume_profile
    #     
    #     # Calculate volume profile
    #     # User Request: Use strict 200-candle lookback (like NN Training) regardless of visible range.
    #     # This provides consistent Market Structure context.
    #     # We use the full 'df' to get the last 200 bars relative to the end of the chart view.
    #     # The chart view ends at chart_df.index[-1].
    #     
    #     last_view_idx = df.index.get_loc(chart_df.index[-1])
    #     start_vp_idx = max(0, last_view_idx - 200)
    #     
    #     # Create a slice for VP calculation (last 200 bars from the current view end)
    #     vp_source_df = df.iloc[start_vp_idx : last_view_idx + 1]
    #     
    #     vp_data = calculate_volume_profile(vp_source_df, num_bins=80)
    #     
    #     if vp_data is not None:
    #         # Add Volume Profile as horizontal bars on a secondary x-axis (Side-by-Side)
    #         # FORCE VALID LISTS: Convert numpy arrays to python lists to ensure JSON serialization is clean
    #         vp_x_data = vp_data['profile'].values.tolist()
    #         vp_y_data = vp_data['bin_centers'].tolist()
    #         
    #         fig.add_trace(
    #             go.Bar(
    #                 y=vp_y_data,
    #                 x=vp_x_data,
    #                 orientation='h',
    #                 marker=dict(
    #                     color='cyan', 
    #                     opacity=0.3, # Low opacity to be subtle
    #                     line=dict(width=0)
    #                 ),
    #                 name='Volume Profile',
    #                 showlegend=False,
    #                 xaxis='x99',  # Use safe ID to avoid conflict with subplot axes
    #                 yaxis='y',
    #                 hoverinfo='x+y'
    #             )
    #         )
    #         print(f"DEBUG: Volume Profile trace added. Points: {len(vp_x_data)}")
    #         
    #     else:
    #         print("DEBUG: Volume Profile data is None")
    #
    #
    # except ImportError as e:
    #     print(f"CRITICAL ERROR: volume_profile module not found: {e}")
    #     # Add a dummy trace to alert the user on the chart
    #     fig.add_annotation(
    #         text="Volume Profile Module Missing",
    #         xref="paper", yref="paper",
    #         x=0.95, y=0.95, showarrow=False, font=dict(color='red')
    #     )
    # except Exception as e:
    #     print(f"CRITICAL ERROR: Volume Profile generation failed: {e}")
    #     import traceback
    #     traceback.print_exc()



    # Add volume bars (standard volume at bottom)
    colors = ['#26a69a' if chart_df['close'].iloc[i] >= chart_df['open'].iloc[i] else '#ef5350'
              for i in range(len(chart_df))]

    fig.add_trace(
        go.Bar(
            x=chart_df.index.strftime('%Y-%m-%dT%H:%M:%S'),
            y=chart_df['volume'].tolist(),
            name='Volume',
            marker_color=colors,
            marker=dict(line=dict(width=0)),
            showlegend=False
        ),
        row=2, col=1
    )

    # Add Stochastic indicators
    stoch_configs = [
        {'col': 'stoch_9_3', 'row': 3, 'color': 'gold', 'name': 'S(9,3)'},
        {'col': 'stoch_14_3', 'row': 4, 'color': 'dodgerblue', 'name': 'S(14,3)'},
        {'col': 'stoch_40_4', 'row': 5, 'color': 'white', 'name': 'S(40,4)'},
        {'col': 'stoch_60_10', 'row': 6, 'color': 'magenta', 'name': 'S(60,10)'}
    ]

    for config in stoch_configs:
        # Add stochastic line
        fig.add_trace(
            go.Scatter(
                x=chart_df.index.strftime('%Y-%m-%dT%H:%M:%S'),
                y=chart_df[config['col']].tolist(),
                name=config['name'],
                line=dict(color=config['color'], width=2),
                showlegend=False
            ),
            row=config['row'], col=1
        )

        # Add horizontal lines at 20 and 80
        fig.add_hline(
            y=20, line_dash="dash", line_color="gray", line_width=1,
            row=config['row'], col=1
        )
        fig.add_hline(
            y=80, line_dash="dash", line_color="gray", line_width=1,
            row=config['row'], col=1
        )

        # Set y-axis range for stochastics
        fig.update_yaxes(range=[0, 100], row=config['row'], col=1)

    # -------------------------------------------------------------------------
    # STRATEGY EXIT VISUALIZATION
    # -------------------------------------------------------------------------
    if STRATEGIES_AVAILABLE and 'hybrid_alarm' in chart_df.columns:
        strategies = {
            1: ('POC Target', strategy_1, 'circle'),
            2: ('VAH Exit', strategy_2, 'diamond'),
            3: ('LVN Accel', strategy_3, 'star'),
            4: ('Multi-Tier', strategy_4, 'square'),
            5: ('Trail Stop', strategy_5, 'triangle-up'),
            6: ('Vol Diverg', strategy_6, 'x')
        }
        
        # Find entry signals in the visible chart
        entry_indices = np.where(chart_df['hybrid_alarm'] == True)[0]
        
        # We need to simulate trades for these entries
        # Note: execute_trade takes the whole DF usually, but here we pass chart_df 
        # which means it can only see what's visible. This is fine for visualization 
        # as long as we understand the limitation (can't see future if zoomed in past).
        # Actually, chart_df is a Slice. Indicies are relative to slice.
        # But execute_trade expects DataFrame.
        
        t_strat_loop_start = time.time()
        for strat_id, (name, module, symbol_marker) in strategies.items():
            exit_points_x = []
            exit_points_y = []
            hover_texts = []
            marker_colors = []
            
            is_active = (strat_id == exit_strategy_id)
            if not is_active:
                continue # Skip background strategies for performance
                
            base_color = '#00f2fe'
            opacity = 1.0
            size = 12
            
            for entry_idx in entry_indices:
                try:
                    # Run strategy
                    # Pass a copy to avoid side effects? No need.
                    # Balance 10000 dummy
                    trade = module.execute_trade(chart_df, entry_idx, 10000, precise=False)
                    
                    if trade['reason'] != 'VP_ERROR' and trade['exit_idx'] != entry_idx:
                        exit_points_x.append(trade['exit_time'])
                        exit_points_y.append(trade['exit_price'])
                        
                        pnl_str = f"{trade['pnl_pct']:.2f}%"
                        color = '#00ff88' if trade['pnl_pct'] > 0 else '#ff3366'
                        if not is_active: color = base_color # Greyscale for inactive
                        
                        marker_colors.append(color)
                        hover_texts.append(f"{name} Exit<br>PnL: {pnl_str}<br>Reason: {trade['reason']}")
                        
                except Exception as e:
                    # print(f"Strat {name} error: {e}")
                    pass
            
            if exit_points_x:
                fig.add_trace(
                    go.Scatter(
                        x=exit_points_x,
                        y=exit_points_y,
                        mode='markers',
                        name=f"{name} Exits",
                        marker=dict(
                            symbol=symbol_marker,
                            size=size,
                            color=marker_colors,
                            line=dict(width=1, color='white') if is_active else dict(width=0),
                            opacity=opacity
                        ),
                        hovertext=hover_texts,
                        hoverinfo='text'
                    ),
                    row=1, col=1
                )

    # -------------------------------------------------------------------------
    # FINAL LAYOUT AND MARKER POLISHING
    # -------------------------------------------------------------------------
    
    # Restore Dark Theme
    fig.update_layout(
         template='plotly_dark',
         autosize=True,
         showlegend=False,
         xaxis_rangeslider_visible=False,
         hovermode='x unified',
         paper_bgcolor='#0f0c29',
         plot_bgcolor='#1a1a2e',
         font=dict(color='white', size=12),
         margin=dict(l=40, r=40, t=10, b=70),
         dragmode='pan',
         uirevision='no_reset',
         hoverdistance=20, # Only activate when close (default)
         spikedistance=20,
         shapes=[{
            'type': 'line',
            'xref': 'x',
            'yref': 'paper',
            'x0': 0, 'x1': 0, 'y0': 0, 'y1': 1,
            'line': {'color': 'rgba(255, 255, 255, 0.5)', 'width': 1, 'dash': 'solid'},
            'visible': False,
            'name': 'crosshair_v'
         }]
    )

    # Force 100% width for all standard subplots (VP removed from main chart)
    layout_update = {}
    for i in range(1, 7):
        suffix = "" if i == 1 else str(i)
        key = f"xaxis{suffix}"
        # Links all axes to the first one for unified hover synchronization
        layout_update[key] = dict(domain=[0, 1.0], type='date', matches='x' if i > 1 else None)
    fig.update_layout(layout_update)

    # 1. Standard Axis Formatting (Dates) - MOVED AFTER LAYOUT UPDATE
    fig.update_xaxes(
        showgrid=True,
        gridcolor='#333333',
        gridwidth=0.5,
        color='white',
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikecolor='rgba(255, 255, 255, 0.8)',
        spikethickness=1,
        spikedash='solid',
        rangeslider_visible=False,
        type='date'
    )
    
    # Enable tick labels on the bottom row (row 6)
    fig.update_xaxes(showticklabels=True, row=6, col=1)

    # 2. Add POC/HVN/LVN Markers (Do this LAST to avoid axis pollution during layout updates)
    max_vp_vol = 100
    try:
        # Find the Volume Profile trace we added earlier
        vp_trace = None
        for trace in fig['data']:
            if trace['name'] == 'Volume Profile':
                vp_trace = trace
                max_vp_vol = max(list(trace['x'])) if trace['x'] else 100
                break
        
        # If we have volume profile data in the local scope variables (from earlier in function)
        # we can use it to add the extra lines
        if 'vp_data' in locals() and vp_data is not None:
            # Helper for line segments
            def _seg(levels, mx):
                xs, ys = [], []
                for p in levels:
                    xs.extend([0, mx, None])
                    ys.extend([p, p, None])
                return xs, ys

            # POC
            if vp_data.get('poc_price'):
                fig.add_trace(go.Scatter(
                    x=[0, max_vp_vol], y=[vp_data['poc_price'], vp_data['poc_price']],
                    mode='lines', line=dict(color='red', width=2),
                    name='Point of Control', xaxis='x99', yaxis='y'
                ))
            
            # HVN
            if vp_data.get('hvn_prices'):
                hx, hy = _seg(vp_data['hvn_prices'], max_vp_vol)
                fig.add_trace(go.Scatter(
                    x=hx, y=hy, mode='lines', 
                    line=dict(color='rgba(0, 255, 0, 0.5)', width=1, dash='dot'),
                    name='HVN', xaxis='x99', yaxis='y', showlegend=False
                ))

            # LVN
            if vp_data.get('lvn_prices'):
                lx, ly = _seg(vp_data['lvn_prices'], max_vp_vol)
                fig.add_trace(go.Scatter(
                    x=lx, y=ly, mode='lines', 
                    line=dict(color='rgba(255, 165, 0, 0.5)', width=1, dash='dot'),
                    name='LVN', xaxis='x99', yaxis='y', showlegend=False
                ))
    except Exception as e:
        print(f"DEBUG: Failed to add markers: {e}")

    # 3. Dedicated Volume Profile Axis (x99)
    # This must OVERRIDE the global update_xaxes settings
    fig.update_layout(
        xaxis99=dict(
            domain=[0.855, 1.0],
            anchor='y',
            type='linear',
            autorange=True,     # Let Plotly scale it to fit the 15% width
            showticklabels=False,
            visible=True,
            showgrid=False,
            zeroline=False,
            side='bottom',
            overlaying=None 
        )
    )

    # Final scaling for price and volume
    fig.update_yaxes(autorange=True, fixedrange=False, row=1, col=1)
    fig.update_yaxes(autorange=True, fixedrange=False, rangemode='tozero', row=2, col=1)

    # Return as JSON
    t_plot = time.time() - t_start - t_load - t_alarms
    json_result = fig.to_json()
    t_json = time.time() - t_start - t_load - t_alarms - t_plot
    
    print(f"DEBUG: generate_chart_data Breakdown:")
    print(f"  - Load CSV:    {t_load:.4f}s")
    print(f"  - Alarms/NN:   {t_alarms:.4f}s")
    print(f"  - Plotly Fig:  {t_plot:.4f}s")
    print(f"  - JSON Serial: {t_json:.4f}s")
    print(f"  - Total:       {time.time() - t_start:.4f}s")
    
    return json_result

def get_chart_metadata(filepath):
    """Get metadata about the chart data"""
    df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)

    return {
        'total_candles': len(df),
        'first_candle': df.index[0].strftime('%Y-%m-%d %H:%M:%S'),
        'last_candle': df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
        'latest_price': float(df['close'].iloc[-1]),
        'latest_volume': float(df['volume'].iloc[-1])
    }
