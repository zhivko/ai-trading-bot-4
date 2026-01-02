import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

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

# Pivot-based channel logic is now below

# -------------------------------------------------------------------------
# NEURAL NETWORK PATTERN DETECTION (RTX 5090 Optimized Inference)
# -------------------------------------------------------------------------
class PatternDetectorCNN(nn.Module):
    def __init__(self, window_size=100):
        super(PatternDetectorCNN, self).__init__()
        self.features = nn.Sequential(
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
            nn.AdaptiveAvgPool1d(16)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

_NN_MODEL = None
_NN_CACHE = {}  # Cache for NN detection results: {(filepath, data_len, threshold): results}

def get_nn_model():
    global _NN_MODEL
    if _NN_MODEL is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PatternDetectorCNN()
        model_path = os.path.join(os.path.dirname(__file__), "stoch_low_detector_5090.pth")
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device, weights_only=True)
                new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict)
                model.to(device)
                model.eval()
                _NN_MODEL = model
                print(f"NN Model loaded successfully on {device}")
            except Exception as e:
                print(f"Error loading NN Model: {e}")
    return _NN_MODEL

def identify_nn_patterns(df, nn_threshold=85):
    """Detects 'W' patterns using the trained Neural Network
    
    Args:
        nn_threshold: Confidence threshold (0-100), default 85
    """
    model = get_nn_model()
    if model is None:
        df['nn_alarm'] = False
        return df

    device = next(model.parameters()).device
    
    # Calculate stochastic the SAME WAY as training script (single smoothing, not double)
    # Training uses: stoch.rolling(smooth_k).mean() / 100.0
    # NOT the double-smoothed version from calculate_stochastic()
    period_k = 60
    smooth_k = 10
    low_min = df['low'].rolling(window=period_k).min()
    high_max = df['high'].rolling(window=period_k).max()
    stoch = 100 * ((df['close'] - low_min) / (high_max - low_min))
    stoch_vals = (stoch.rolling(window=smooth_k).mean() / 100.0).values
    nn_results = np.zeros(len(df), dtype=bool)
    
    window_size = 100
    start_idx = window_size
    valid_indices = []
    windows = []
    
    for i in range(start_idx, len(df)):
        window = stoch_vals[i-window_size:i]
        if not np.isnan(window).any():
            valid_indices.append(i)
            windows.append(window)
            
    if windows:
        with torch.no_grad():
            has_cuda = torch.cuda.is_available()
            # Handle float32/autocast
            with torch.amp.autocast('cuda') if has_cuda else torch.device('cpu'):
                x_tensor = torch.FloatTensor(np.array(windows)).unsqueeze(1).to(device)
                logits = model(x_tensor)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                
                # --- AGGRESSIVE SUPPRESSION LOGIC (NMS) ---
                # Use dynamic threshold from parameter
                threshold = nn_threshold / 100.0  # Convert from percentage to 0-1
                temp_results = probs > threshold
                
                # 2. Local Peak Detection / Cooldown (30 bars ~ 7.5 hours)
                final_results = np.zeros_like(probs, dtype=bool)
                cooldown = 30  # Reduced from 60 to allow more detections
                for idx in range(len(probs)):
                    if probs[idx] > threshold:
                        # Check if this is the maximum in [idx-cooldown, idx+cooldown]
                        start = max(0, idx - cooldown)
                        end = min(len(probs), idx + cooldown)
                        if probs[idx] == np.max(probs[start:end]):
                            final_results[idx] = True
                
                nn_results[valid_indices] = final_results
                
                # Debug: Print detection stats and top scores
                detected_count = np.sum(final_results)
                if detected_count > 0:
                    print(f"NN Stochastic Low Detection: Found {detected_count} patterns (threshold={nn_threshold}%)")
                
                # Show top 5 confidence scores to help tune threshold
                if len(probs) > 0:
                    top_indices = np.argsort(probs)[-5:][::-1]  # Top 5 highest scores
                    print(f"Top 5 NN confidence scores:")
                    for idx in top_indices:
                        actual_idx = valid_indices[idx]
                        timestamp = df.index[actual_idx]
                        print(f"  - {timestamp}: {probs[idx]*100:.1f}% confidence")
                
    df['nn_alarm'] = nn_results
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

def identify_hybrid_signals(df, nn_threshold=85):
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

def identify_quad_rotation_alarms(df, nn_threshold=85):
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
            for col in ['nn_alarm', 'hybrid_alarm', 'alarm', 'stoch_9_3', 'stoch_14_3', 'stoch_40_4', 'stoch_60_10', 'slope']:
                if col in cached_df.columns:
                    df[col] = cached_df[col]
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

def calculate_linear_regression_channel(df, window=40):
    """Calculate Linear Regression Channel for trend context (optimized for macro trends)"""
    df['linreg_mid'] = np.nan
    df['linreg_top'] = np.nan
    df['linreg_bot'] = np.nan
    df['slope'] = np.nan
    
    # Pre-calculate to avoid constant lookup
    closes = df['close'].values
    l_mid = np.full(len(df), np.nan)
    l_top = np.full(len(df), np.nan)
    l_bot = np.full(len(df), np.nan)
    slopes = np.full(len(df), np.nan)
    
    x = np.arange(window)
    for i in range(window, len(df)):
        y = closes[i-window:i]
        # Standard OLS
        slope, intercept, _, _, _ = linregress(x, y)
        current_mid = slope * (window - 1) + intercept
        std_dev = np.std(y)
        
        l_mid[i] = current_mid
        l_top[i] = current_mid + (2.0 * std_dev)
        l_bot[i] = current_mid - (2.0 * std_dev)
        slopes[i] = slope
        
    df['linreg_mid'] = l_mid
    df['linreg_top'] = l_top
    df['linreg_bot'] = l_bot
    df['slope'] = slopes
    return df

def identify_quad_rotation_alarms(df, nn_threshold=85):
    """
    Calculates 4 Stochastics, LinReg Channel, and identifies Quad Rotation Alarms.
    Returns the dataframe with an 'alarm' column.
    """
    # Calculate stochastics
    df['stoch_9_3'] = calculate_stochastic(df, 9, 1, 3)
    df['stoch_14_3'] = calculate_stochastic(df, 14, 1, 3)
    df['stoch_40_4'] = calculate_stochastic(df, 40, 1, 4)
    df['stoch_60_10'] = calculate_stochastic(df, 60, 10, 10)

    # Calculate Linear Regression Channel (Trend Context)
    df = calculate_linear_regression_channel(df, window=40)
    
    # Identify Alarms (Vectorized)
    # Condition A: Downward Channel (Slope < -0.5)
    cond_downtrend = df['slope'] < -0.5
    # Condition B: Quad Exhaustion (All low)
    cond_quad_low = (df['stoch_9_3'] < 20) & (df['stoch_14_3'] < 25) & (df['stoch_40_4'] < 25) & (df['stoch_60_10'] < 25)
    # Condition C: Turn Up (Fast Stoch curls up)
    cond_turn_up = df['stoch_9_3'] > df['stoch_9_3'].shift(1)
    
    df['alarm'] = cond_downtrend & cond_quad_low & cond_turn_up
    
    # Hybrid Pattern Detection (Breakout + NN W)
    df = identify_hybrid_signals(df, nn_threshold=nn_threshold)
    
    return df

def generate_chart_data(filepath, symbol, timeframe, num_candles=100, start_date=None, end_date=None, nn_threshold=85):
    """
    Generate interactive OHLC chart data with stochastics using Plotly
    Returns JSON data for Plotly chart
    
    Args:
        nn_threshold: Neural network confidence threshold (0-100), default 85
    """
    # Read data
    df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)

    # OPTIMIZATION: Slice DataFrame for processing
    process_start_index = 0
    if start_date:
        try:
            s_ts = pd.to_datetime(start_date)
            if df.index.tz is None and s_ts.tz is not None:
                s_ts = s_ts.tz_localize(None)
            mask_idx = np.searchsorted(df.index, s_ts)
            process_start_index = max(0, mask_idx - 500)
        except:
            process_start_index = max(0, len(df) - num_candles - 500)
    else:
        process_start_index = max(0, len(df) - num_candles - 500)

    # Create the working slice (copy to avoid SettingWithCopy warnings)
    df = df.iloc[process_start_index:].copy()

    # Calculate Indicators and Alarms on Working dataset
    df = identify_quad_rotation_alarms(df, nn_threshold=nn_threshold)

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
                df_1h = identify_quad_rotation_alarms(df_1h)
                
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
        if df_1h_alarms is not None:
            # rough align
            start_time = chart_df.index[0]
            df_1h_alarms = df_1h_alarms[df_1h_alarms.index >= start_time].copy()
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

    # Add Neural Network detected 'W' patterns
    if 'nn_alarm' in chart_df.columns:
        nn_df = chart_df[chart_df['nn_alarm'] & (~chart_df.get('hybrid_alarm', False))].copy()
        if not nn_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=nn_df.index.strftime('%Y-%m-%dT%H:%M:%S'),
                    y=(nn_df['low'] - marker_offset * 2).tolist(), # Place below candle low
                    mode='markers',
                    marker=dict(symbol='star', size=10, color='#4A90E2', line=dict(width=1, color='white')),
                    name='Neural Network: W-Pattern (Unconfirmed)',
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
    # Volume Profile Visualization
    # -------------------------------------------------------------------------
    try:
        from volume_profile import calculate_volume_profile
        
        # Calculate volume profile for visible chart range
        vp_data = calculate_volume_profile(chart_df, num_bins=80)
        
        if vp_data is not None:
            # Add Volume Profile as horizontal bars on a secondary x-axis (Side-by-Side)
            fig.add_trace(
                go.Bar(
                    y=vp_data['bin_centers'],
                    x=vp_data['profile'].values,
                    orientation='h',
                    marker=dict(color='cyan', opacity=0.3, line=dict(width=0)),
                    name='Volume Profile',
                    showlegend=False,
                    xaxis='x2',
                    yaxis='y',
                    hoverinfo='x+y'
                ),
                row=1, col=1
            )
            
            # Note: We removed the horizontal lines for POC/VAH/VAL as they were cluttering the view.
            # The profile bars themselves give a better visual representation.
            
    except ImportError:
        print("Warning: volume_profile module not found, skipping VP visualization")
    except Exception as e:
        print(f"Warning: Could not add volume profile visualization: {e}")


    # Add volume bars (standard volume at bottom)
    colors = ['#26a69a' if chart_df['close'].iloc[i] >= chart_df['open'].iloc[i] else '#ef5350'
              for i in range(len(chart_df))]

    fig.add_trace(
        go.Bar(
            x=chart_df.index.strftime('%Y-%m-%dT%H:%M:%S'),
            y=chart_df['volume'].tolist(),
            name='Volume',
            marker_color=colors,
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

    # Update layout for dark theme with SIDE-BY-SIDE Volume Profile
    fig.update_layout(
        template='plotly_dark',
        autosize=True,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        hovermode='x',
        paper_bgcolor='#0f0c29',
        plot_bgcolor='#1a1a2e',
        font=dict(color='white', size=12),
        margin=dict(l=40, r=40, t=10, b=70),
        dragmode='pan',
        uirevision='no_reset',
        hoverdistance=-1,
        spikedistance=-1,
        
        # Define layouts for axes to create side-by-side effect
        xaxis=dict(
            domain=[0, 0.85], # Main chart takes 85% width
            anchor='y'
        ),
        xaxis2=dict(
            domain=[0.86, 1.0], # Volume Profile takes right 14%
            anchor='y',
            showticklabels=False,
            visible=True
        )
    )

    # Update all x-axes
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
    
    # Explicitly enable tick labels on the bottom row (row 6)
    fig.update_xaxes(showticklabels=True, row=6, col=1)


    # Enable auto-range for price chart (row 1) and volume (row 2)
    fig.update_yaxes(
        autorange=True,
        fixedrange=False,
        row=1, col=1
    )  # Price chart - auto scale

    fig.update_yaxes(
        autorange=True,
        fixedrange=False,
        rangemode='tozero',
        row=2, col=1
    )  # Volume - auto scale

    # Return as JSON
    return fig.to_json()

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
