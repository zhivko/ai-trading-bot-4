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

def calculate_linear_regression_channel(df, window=20):
    """Calculate Linear Regression Channel for trend context"""
    df['linreg_mid'] = np.nan
    df['linreg_top'] = np.nan
    df['linreg_bot'] = np.nan
    df['slope'] = np.nan
    
    # We can optimize this loop later, but for 1H candles (typ. < 1000) it's fast enough
    # If using full history, iterating standard Python loop might be slow. 
    # Let's try to keep it efficient. 
    # For now, stick to the logic provided in the reference file.
    
    for i in range(window, len(df)):
        # Compute LinReg for Channel (Local window)
        y = df['close'].iloc[i-window:i]
        x = np.arange(window)
        slope, intercept, _, _, _ = linregress(x, y)
        
        # Store Channel Values
        current_mid = slope * (window - 1) + intercept
        std_dev = y.std()
        
        # Update DataFrame
        # Using .iat is faster than .loc/iloc for single value updates
        df.iat[i, df.columns.get_loc('linreg_mid')] = current_mid
        df.iat[i, df.columns.get_loc('linreg_top')] = current_mid + (2.0 * std_dev)
        df.iat[i, df.columns.get_loc('linreg_bot')] = current_mid - (2.0 * std_dev)
        df.iat[i, df.columns.get_loc('slope')] = slope
    
    return df

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
def get_nn_model():
    global _NN_MODEL
    if _NN_MODEL is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PatternDetectorCNN()
        model_path = os.path.join(os.path.dirname(__file__), "stoch_w_detector_5090.pth")
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

def identify_nn_patterns(df):
    """Detects 'W' patterns using the trained Neural Network"""
    model = get_nn_model()
    if model is None:
        df['nn_alarm'] = False
        return df

    device = next(model.parameters()).device
    if 'stoch_60_10' not in df.columns:
         df['stoch_60_10'] = calculate_stochastic(df, 60, 10, 10)
    
    stoch_vals = (df['stoch_60_10'].values / 100.0)
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
                nn_results[valid_indices] = probs > 0.85 # Confidence threshold
                
    df['nn_alarm'] = nn_results
    return df

def identify_quad_rotation_alarms(df):
    """
    Calculates 4 Stochastics, LinReg Channel, and identifies Quad Rotation Alarms.
    Returns the dataframe with an 'alarm' column.
    """
    # Calculate stochastics
    df['stoch_9_3'] = calculate_stochastic(df, 9, 1, 3)
    df['stoch_14_3'] = calculate_stochastic(df, 14, 1, 3)
    df['stoch_40_4'] = calculate_stochastic(df, 40, 1, 4)
    df['stoch_60_10'] = calculate_stochastic(df, 60, 10, 10)

    # Calculate Linear Regression Channel
    df = calculate_linear_regression_channel(df, window=20)
    
    # Identify Alarms (Vectorized)
    # Condition A: Downward Channel (Slope < -0.5)
    cond_downtrend = df['slope'] < -0.5
    # Condition B: Quad Exhaustion (All low)
    cond_quad_low = (df['stoch_9_3'] < 20) & (df['stoch_14_3'] < 25) & (df['stoch_40_4'] < 25) & (df['stoch_60_10'] < 25)
    # Condition C: Turn Up (Fast Stoch curls up)
    cond_turn_up = df['stoch_9_3'] > df['stoch_9_3'].shift(1)
    
    df['alarm'] = cond_downtrend & cond_quad_low & cond_turn_up
    
    # NN Pattern Detection Integration
    df = identify_nn_patterns(df)
    
    return df

def generate_chart_data(filepath, symbol, timeframe, num_candles=100, start_date=None, end_date=None):
    """
    Generate interactive OHLC chart data with stochastics using Plotly
    Returns JSON data for Plotly chart
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
    df = identify_quad_rotation_alarms(df)

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
        vertical_spacing=0.02,
        row_heights=[0.4, 0.1, 0.125, 0.125, 0.125, 0.125],
        subplot_titles=(
            f'{symbol} - {timeframe.upper()}',
            'Volume',
            'Stochastic (9,3)',
            'Stochastic (14,3)',
            'Stochastic (40,4)',
            'Stochastic (60,10)'
        )
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
            decreasing_fillcolor='#ef5350'
        ),
        row=1, col=1
    )
    
    # Add Linear Regression Channel
    fig.add_trace(
        go.Scatter(
            x=chart_df.index.strftime('%Y-%m-%dT%H:%M:%S'),
            y=chart_df['linreg_top'].tolist(),
            name='LinReg Top',
            line=dict(color='teal', width=1),
            hoverinfo='skip',
            showlegend=False
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df.index.strftime('%Y-%m-%dT%H:%M:%S'),
            y=chart_df['linreg_bot'].tolist(),
            name='LinReg Bot',
            line=dict(color='teal', width=1),
            hoverinfo='skip',
            showlegend=False
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df.index.strftime('%Y-%m-%dT%H:%M:%S'),
            y=chart_df['linreg_mid'].tolist(),
            name='LinReg Mid',
            line=dict(color='teal', width=1, dash='dash'),
            hoverinfo='skip',
            showlegend=False
        ),
        row=1, col=1
    )

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
                y=(alarm_df['low'] * 0.99).tolist(), # Place slightly below candle
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
                y=(df_1h_alarms['low'] * 0.98).tolist(), 
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

    # -------------------------------------------------------------------------
    # Add NN 'W' Pattern Markers (Deep Learning)
    # -------------------------------------------------------------------------
    if 'nn_alarm' in chart_df.columns:
        nn_alarm_df = chart_df[chart_df['nn_alarm']].copy()
        if not nn_alarm_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=nn_alarm_df.index.strftime('%Y-%m-%dT%H:%M:%S'),
                    y=(nn_alarm_df['low'] * 0.97).tolist(), # Slightly lower than triangles
                    mode='markers',
                    marker=dict(
                        symbol='star', 
                        size=14, 
                        color='#00FF00', 
                        line=dict(width=1, color='white')
                    ),
                    name='Neural Network: W-Pattern Detected',
                    hoverinfo='x+y+name'
                ),
                row=1, col=1
            )

    # Add volume bars
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

    # Update layout for dark theme
    fig.update_layout(
        template='plotly_dark',
        height=1000,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        paper_bgcolor='#0f0c29',
        plot_bgcolor='#1a1a2e',
        font=dict(color='white', size=12),

        margin=dict(l=50, r=50, t=80, b=50),
        dragmode='pan',
        uirevision='no_reset'
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
        spikecolor='rgba(255, 255, 255, 0.5)',
        spikethickness=1,
        spikedash='dash'
    )

    # Update all y-axes
    fig.update_yaxes(
        showgrid=True,
        gridcolor='#333333',
        gridwidth=0.5,
        color='white'
    )

    # Enable auto-range for price chart (row 1) and volume (row 2)
    fig.update_yaxes(
        autorange=True,
        fixedrange=False,
        row=1, col=1
    )  # Price chart - auto scale

    fig.update_yaxes(
        autorange=True,
        fixedrange=False,
        row=2, col=1
    )  # Volume - auto scale

    # Ensure x-axis shows all data
    fig.update_xaxes(
        rangeslider_visible=False,
        type='date'
    )

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
