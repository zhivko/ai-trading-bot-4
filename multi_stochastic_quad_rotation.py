import pandas as pd
import numpy as np
import mplfinance as mpf
import io
from scipy.stats import linregress
import datetime
import argparse
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from chart_generator import identify_nn_patterns

# Global variable to track alarm count for unique filenames
alarm_count = 0
first_chart_created = False  # Flag to break after first chart

# 2. INDICATOR SETUP
# -------------------------------------------------------------------------
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
    
    for i in range(window, len(df)):
        # Compute LinReg for Channel (Local window)
        y = df['close'].iloc[i-window:i]
        x = np.arange(window)
        slope, intercept, _, _, _ = linregress(x, y)
        
        # Store Channel Values
        current_mid = slope * (window - 1) + intercept
        std_dev = y.std()
        
        # Update DataFrame
        df.iat[i, df.columns.get_loc('linreg_mid')] = current_mid
        df.iat[i, df.columns.get_loc('linreg_top')] = current_mid + (2.0 * std_dev)
        df.iat[i, df.columns.get_loc('linreg_bot')] = current_mid - (2.0 * std_dev)
        df.iat[i, df.columns.get_loc('slope')] = slope
    
    return df

def calculate_stochastic_angle(df, column_name, window=5):
    """Calculate the angle of stochastic change over a window"""
    angles = []
    for i in range(window, len(df)):
        # Get values over the window
        values = df[column_name].iloc[i-window:i].values
        
        # Calculate angle using linear regression slope
        if len(values) >= 2 and not np.isnan(values).all():
            x = np.arange(len(values))
            # Remove NaN values for calculation
            valid_mask = ~np.isnan(values)
            if np.sum(valid_mask) >= 2:
                slope, _, _, _, _ = linregress(x[valid_mask], values[valid_mask])
                # Convert slope to angle in degrees
                angle = np.arctan(slope) * 180 / np.pi
                angles.append(angle)
            else:
                angles.append(0)
        else:
            angles.append(0)
    
    # Pad the beginning with zeros
    return [0] * window + angles

def create_1h_alarm_chart(df, alarm_index, alarm_price, alarm_count):
    """Create and save 1h chart when alarm is detected"""
    
    # Get data range around the alarm for context (center the alarm in the middle)
    total_candles = 60  # Total candles to show
    half_candles = total_candles // 2
    
    # Calculate start and end to center the alarm point
    start_idx = max(0, alarm_index - half_candles)
    end_idx = min(len(df), alarm_index + half_candles)
    
    # Adjust if we hit the boundaries to ensure we have enough candles
    if start_idx == 0:
        end_idx = min(len(df), total_candles)
    elif end_idx == len(df):
        start_idx = max(0, len(df) - total_candles)
    
    # Slice the data around the alarm
    chart_df = df.iloc[start_idx:end_idx].copy()
    
    # Create marker for the alarm
    chart_df['alarm_marker'] = np.nan
    chart_df.loc[chart_df.index[alarm_index - start_idx], 'alarm_marker'] = alarm_price * 0.93
    
    # Create marker for stochastics
    chart_df['s9_alarm'] = np.nan
    chart_df['s14_alarm'] = np.nan
    chart_df['s40_alarm'] = np.nan
    
    current_s9 = df['stoch_9_3'].iloc[alarm_index]
    current_s14 = df['stoch_14_3'].iloc[alarm_index]
    current_s40 = df['stoch_40_4'].iloc[alarm_index]
    
    chart_df.loc[chart_df.index[alarm_index - start_idx], 's9_alarm'] = current_s9
    chart_df.loc[chart_df.index[alarm_index - start_idx], 's14_alarm'] = current_s14
    chart_df.loc[chart_df.index[alarm_index - start_idx], 's40_alarm'] = current_s40
    
    # Create unique filename with timestamp
    alarm_timestamp = df.index[alarm_index].strftime('%Y%m%d_%H%M%S')
    filename = f"alarm_chart_{alarm_count:03d}_{alarm_timestamp}.png"
    
    # Setup plots
    apds = [
        # 1. Channel Lines (Teal/Cyan)
        mpf.make_addplot(chart_df['linreg_top'], color='teal', width=0.8, panel=0),
        mpf.make_addplot(chart_df['linreg_bot'], color='teal', width=0.8, panel=0),
        mpf.make_addplot(chart_df['linreg_mid'], color='teal', linestyle='--', width=0.5, panel=0),
        
        # 2. Alarm Marker on Price
        mpf.make_addplot(chart_df['alarm_marker'], type='scatter', markersize=100, marker='^', color='cyan', panel=0),

        # 3. Stochastics (Stacked Panels) - with white borders
        mpf.make_addplot(chart_df['stoch_9_3'], panel=2, color='gold', ylabel='S(9,3)', ylim=(0,100), width=1.5),
        mpf.make_addplot(chart_df['s9_alarm'], type='scatter', panel=2, markersize=50, marker='^', color='cyan'),
        
        mpf.make_addplot(chart_df['stoch_14_3'], panel=3, color='dodgerblue', ylabel='S(14,3)', ylim=(0,100), width=1.5),
        mpf.make_addplot(chart_df['s14_alarm'], type='scatter', panel=3, markersize=50, marker='^', color='cyan'),

        mpf.make_addplot(chart_df['stoch_40_4'], panel=4, color='white', ylabel='S(40,4)', ylim=(0,100), width=1.5),
        mpf.make_addplot(chart_df['s40_alarm'], type='scatter', panel=4, markersize=50, marker='^', color='cyan'),

        mpf.make_addplot(chart_df['stoch_60_10'], panel=5, color='magenta', ylabel='S(60,10)', ylim=(0,100), width=1.5),
    ]

    # Style Setup with white subplot borders
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit', wick='inherit', volume='in')
    s  = mpf.make_mpf_style(marketcolors=mc,
                            gridstyle=':',
                            y_on_right=True,
                            facecolor='black',
                            figcolor='black',
                            gridcolor='#333333',
                            rc={'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor':'white',
                                'axes.edgecolor':'white', 'axes.linewidth':0.8,
                                'legend.labelcolor':'white', 'text.color':'white'})

    # Generate chart
    fig, axes = mpf.plot(chart_df,
             type='candle',
             volume=True,
             style=s,
             addplot=apds,
             title=f"\nBTCUSDT - 1h Quad Rotation Alarm #{alarm_count} @ {alarm_price:.2f}",
             panel_ratios=(4,1,1,1,1,1),
             figscale=2.0,
             hlines=dict(hlines=[20,80], colors=['gray','gray'], linestyle='--', linewidths=0.5),
             datetime_format='%d.%m %H:%M:%S',
             returnfig=True
             )
             

    # Enable y-axis autoscale for the main price panel (panel 0)
    axes[0].relim()
    axes[0].autoscale_view()

    # Set legend text color to white for all axes
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_color('white')

    # Add alarm timestamp label below the chart
    alarm_datetime = df.index[alarm_index].strftime('%Y-%m-%d %H:%M:%S')
    fig.text(0.5, 0.02, f"ALARM DATETIME: {alarm_datetime}", 
             ha='center', va='bottom', fontsize=12, color='yellow', weight='bold',
             transform=fig.transFigure)
    
    # Save the figure
    fig.savefig(filename, dpi=600, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    
    print(f"1h chart saved: {filename}")
    return filename

def find_entry_opportunities_15m(df_15m, alarm_timestamp, lookforward_hours=24):
    """Find entry opportunities on 15m data after alarm timestamp using stoch_9_3<20 and rising condition"""

    # Find the index of alarm timestamp in 30m data
    alarm_time = pd.to_datetime(alarm_timestamp)

    # Calculate stochastics for full 30m data
    df_15m['stoch_9_3'] = calculate_stochastic(df_15m, 9, 1, 3)
    df_15m['stoch_14_3'] = calculate_stochastic(df_15m, 14, 1, 3)
    df_15m['stoch_40_4'] = calculate_stochastic(df_15m, 40, 1, 4)
    df_15m['stoch_60_10'] = calculate_stochastic(df_15m, 60, 10, 10)

    # Filter 30m data to start from alarm timestamp
    entry_df = df_15m[df_15m.index >= alarm_time].copy()
    print(entry_df.head)

    if len(entry_df) == 0:
        print(f"No 15m data available after alarm timestamp: {alarm_timestamp}")
        return None, None, None

    # Limit to lookforward period
    end_time = alarm_time + pd.Timedelta(hours=lookforward_hours)
    entry_df = entry_df[entry_df.index <= end_time]

    if len(entry_df) == 0:
        print(f"No 15m data available within {lookforward_hours} hours after alarm")
        return None, None, None
    
    # Find entry signals using simplified criteria
    entry_signals = []
    
    for i in range(5, len(entry_df)):  # Start from 5 to ensure enough data for comparison
        s9 = entry_df['stoch_9_3'].iloc[i]
        s40 = entry_df['stoch_40_4'].iloc[i]
        
        # Entry condition 1: S9 stochastic is below 20 (oversold)
        is_oversold = s40 < 20
        
        # Entry condition 2: S9 is rising (compare with previous values)
        s40_prev = entry_df['stoch_40_4'].iloc[i-1]
        s40_prev2 = entry_df['stoch_40_4'].iloc[i-2] if i >= 2 else s40_prev
        
        # Check if S9 is rising (current > previous OR showing upward momentum)
        is_rising = (s40 > s40_prev) or (s40 > s40_prev2) or (s40 - s40_prev > 0)
        
        # Find the point where S9 goes below 20 and then starts rising
        if is_oversold and is_rising:
            entry_time = entry_df.index[i]
            entry_price = entry_df['close'].iloc[i]
            entry_signals.append({
                'timestamp': entry_time,
                'price': entry_price,
                's9': s9,
                's14': entry_df['stoch_14_3'].iloc[i],
                's40': entry_df['stoch_40_4'].iloc[i],
                's60': entry_df['stoch_60_10'].iloc[i],
                's40_prev': s40_prev,
                's40_change': s40 - s40_prev,
                'index': df_15m.index.get_loc(entry_df.index[i])
            })
    
    if not entry_signals:
        print("No entry opportunities found in 15m data using stoch_9_3<20 and rising condition")
        return None, None, None
    
    # Choose the best entry (first one that meets criteria)
    best_entry = entry_signals[0]  # Take the first signal after alarm
    
    print(f"Best entry opportunity found (S40<20 and rising):")
    print(f"  Time: {best_entry['timestamp']}")
    print(f"  Price: {best_entry['price']:.2f}")
    print(f"  Stochastics: S40={best_entry['s40']:.1f}, S14={best_entry['s14']:.1f}, S40={best_entry['s40']:.1f}, S60={best_entry['s60']:.1f}")
    print(f"  S40 Change: {best_entry['s40_change']:.1f} (from {best_entry['s40_prev']:.1f})")
    
    return entry_df, best_entry, entry_signals
def create_15m_entry_chart(df_15m, alarm_timestamp, best_entry, entry_signals, alarm_count):
    """Create 15m chart showing entry opportunities"""
    
    if best_entry is None:
        print("No entry opportunity to chart")
        return None, None
    
    alarm_time = pd.to_datetime(alarm_timestamp)
    entry_time = best_entry['timestamp']
    entry_index = best_entry['index']
    
    # Get data range around the entry (show alarm time to entry time + context)
    total_candles = 80  # Total candles to show
    half_candles = total_candles // 2
    
    # Start from alarm time, end around entry + context
    start_idx = max(0, entry_index - half_candles)
    end_idx = min(len(df_15m), entry_index + half_candles)
    
    # Adjust to include alarm timestamp if possible
    alarm_idx_in_range = df_15m.index.get_loc(alarm_time) if alarm_time in df_15m.index else None
    
    # Slice the data around the entry
    chart_df = df_15m.iloc[start_idx:end_idx].copy()

    # Verify stochastic columns exist and have data
    if 'stoch_9_3' not in chart_df.columns or chart_df['stoch_9_3'].isna().all():
        print("Warning: Stochastic columns missing or empty in chart_df, recalculating...")
        chart_df['stoch_9_3'] = calculate_stochastic(chart_df, 9, 1, 3)
        chart_df['stoch_14_3'] = calculate_stochastic(chart_df, 14, 1, 3)
        chart_df['stoch_40_4'] = calculate_stochastic(chart_df, 40, 1, 4)
        chart_df['stoch_60_10'] = calculate_stochastic(chart_df, 60, 10, 10)

    # Initialize marker columns
    chart_df['alarm_marker'] = np.nan
    chart_df['entry_marker'] = np.nan
    chart_df['entry_signals'] = np.nan
    
    # Mark alarm time on chart (if in range)
    if alarm_idx_in_range is not None and start_idx <= alarm_idx_in_range < end_idx:
        chart_df.loc[chart_df.index[alarm_idx_in_range - start_idx], 'alarm_marker'] = df_15m['close'].iloc[alarm_idx_in_range] * 0.98
    
    # Mark best entry
    chart_df.loc[chart_df.index[entry_index - start_idx], 'entry_marker'] = best_entry['price']
    
    # Mark all entry signals
    for signal in entry_signals:
        signal_idx = signal['index']
        if start_idx <= signal_idx < end_idx:
            chart_df.loc[chart_df.index[signal_idx - start_idx], 'entry_signals'] = signal['price']
    
    # Create unique filename
    entry_timestamp = entry_time.strftime('%Y%m%d_%H%M%S')
    alarm_timestamp_str = alarm_time.strftime('%Y%m%d_%H%M%S')
    filename = f"entry_chart_{alarm_count:03d}_{alarm_timestamp_str}_to_{entry_timestamp}.png"
    

    # Calculate volume profile for 1 day before entry (96 candles)
    vp_start = entry_signals[0]['timestamp'] - pd.Timedelta(days=1)
    vp_df = df_15m[(df_15m.index >= vp_start) & (df_15m.index < entry_signals[0]['timestamp'])].copy()
    if len(vp_df) > 0:
        prices = vp_df['close']
        volumes = vp_df['volume']
        # Use chart_df's price range for bins to align with price chart
        chart_min_price = chart_df['close'].min()
        chart_max_price = chart_df['close'].max()
        num_bins = 80
        bins = np.linspace(chart_min_price, chart_max_price, num_bins + 1)
        vp_df['price_bin'] = pd.cut(prices, bins, labels=False, include_lowest=True)
        volume_profile = vp_df.groupby('price_bin')['volume'].sum()
        volume_profile = volume_profile.reindex(range(num_bins), fill_value=0)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # HVN, LVN, POC
        poc_idx = volume_profile.idxmax()
        poc_price = bin_centers[poc_idx]
        poc_volume = volume_profile.max()
        hvn_price = poc_price
        hvn_volume = poc_volume
        lvn_idx = volume_profile.idxmin()
        lvn_price = bin_centers[lvn_idx]
        lvn_volume = volume_profile.min()
    else:
        # Handle no data
        volume_profile = None

    # Setup plots
    apds = [
        # 1. Alarm Marker on Price
        mpf.make_addplot(chart_df['alarm_marker'], type='scatter', markersize=50, marker='v', color='red', panel=0),
        
        # 2. Entry Signals on Price
        mpf.make_addplot(chart_df['entry_signals'], type='scatter', markersize=80, marker='o', color='lime', panel=0),
        
        # 3. Best Entry Marker
        mpf.make_addplot(chart_df['entry_marker'], type='scatter', markersize=120, marker='^', color='yellow', panel=0),

        # 4. Stochastics (Stacked Panels)
        mpf.make_addplot(chart_df['stoch_9_3'], panel=2, color='gold', ylabel='S(9,3)', ylim=(0,100), width=1.5),
        
        mpf.make_addplot(chart_df['stoch_14_3'], panel=3, color='dodgerblue', ylabel='S(14,3)', ylim=(0,100), width=1.5),

        mpf.make_addplot(chart_df['stoch_40_4'], panel=4, color='white', ylabel='S(40,4)', ylim=(0,100), width=1.5),

        mpf.make_addplot(chart_df['stoch_60_10'], panel=5, color='magenta', ylabel='S(60,10)', ylim=(0,100), width=1.5),
    ]

    # Style Setup
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit', wick='inherit', volume='in')
    s  = mpf.make_mpf_style(marketcolors=mc,
                            gridstyle=':',
                            y_on_right=True,
                            facecolor='black',
                            figcolor='black',
                            gridcolor='#333333',
                            rc={'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor':'white',
                                'axes.edgecolor':'white', 'axes.linewidth':0.8,
                                'legend.labelcolor':'white', 'text.color':'white'})

    # Generate chart
    fig, axes = mpf.plot(chart_df,
             type='candle',
             volume=True,
             style=s,
             addplot=apds,
             title=f"\nBTCUSDT - 15m Entry Opportunities (Alarm: {alarm_timestamp_str} → Entry: {entry_timestamp})",
             panel_ratios=(4,1,1,1,1,1),
             figscale=2.0,
             hlines=dict(hlines=[20,80], colors=['gray','gray'], linestyle='--', linewidths=0.5),
             datetime_format='%d.%m %H:%M:%S',
             returnfig=True
             )

    # Add volume profile if available
    if volume_profile is not None:
        # Add volume profile to the right
        pos = axes[0].get_position()
        vp_width = 0.08
        vp_ax = fig.add_axes([pos.x1 + 0.05, pos.y0, vp_width, pos.height], sharey=axes[0])
        vp_ax.barh(bin_centers, volume_profile.values, height=(bins[1]-bins[0]), color='cyan', alpha=0.6)
        # Mark HVN, LVN, POC
        vp_ax.scatter([hvn_volume], [hvn_price], color='red', marker='^', s=100, label='HVN')
        vp_ax.scatter([lvn_volume], [lvn_price], color='green', marker='v', s=100, label='LVN')
        vp_ax.scatter([poc_volume], [poc_price], color='white', marker='o', s=100, label='POC')
        
        # Create legend outside the volume profile plot
        legend_handles = [mpatches.Patch(color='red', label='HVN (High Volume Node)'),
                          mpatches.Patch(color='green', label='LVN (Low Volume Node)'),
                          mpatches.Patch(color='white', label='POC (Point of Control)')]
        vp_ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(-0.2, 1.2), fontsize=6, facecolor='black', edgecolor='yellow', labelcolor='white')
        
        vp_ax.set_xlabel('Volume', color='white')
        vp_ax.set_title('VP (80 bins)', color='white', fontsize=10)
        vp_ax.tick_params(colors='white')
        vp_ax.grid(True, alpha=0.3, color='gray')
        vp_ax.yaxis.tick_right()  # Put ticks on the right side
        # Mark levels on price chart
        axes[0].axhline(y=hvn_price, color='red', linestyle='--', linewidth=1, alpha=0.7)
        axes[0].axhline(y=lvn_price, color='green', linestyle='--', linewidth=1, alpha=0.7)
        axes[0].axhline(y=poc_price, color='white', linestyle='--', linewidth=1, alpha=0.7)
        
        # Add marker legend for price plot
        from matplotlib.lines import Line2D
        price_legend_elements = [Line2D([0], [0], marker='v', color='red', label='Alarm', markersize=8, linestyle='None'),
                                 Line2D([0], [0], marker='o', color='lime', label='Entry Signal', markersize=8, linestyle='None'),
                                 Line2D([0], [0], marker='^', color='yellow', label='Best Entry', markersize=10, linestyle='None')]
        axes[0].legend(handles=price_legend_elements, loc='lower left', bbox_to_anchor=(0, 1.02), fontsize=8)

    # Set legend text color to white for all axes
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_color('white')

    # Add vertical line at best entry point
    entry_datetime = entry_time.strftime('%Y-%m-%d %H:%M:%S')
    entry_position = chart_df.index.get_loc(entry_time)
    for ax in axes:
        ax.axvline(x=entry_position, color='yellow', linestyle='-', linewidth=2, alpha=0.8)

    # Add alarm timestamp label below the chart
    fig.text(0.5, 0.02, f"ALARM DATETIME: {alarm_timestamp_str}",
             ha='center', va='bottom', fontsize=12, color='yellow', weight='bold',
             transform=fig.transFigure)
    
    # Add best entry timestamp label above the chart
    fig.text(0.5, 0.98, f"BEST ENTRY: {entry_datetime}",
             ha='center', va='top', fontsize=12, color='lime', weight='bold',
             transform=fig.transFigure)
    

    axes[0].relim()
    axes[0].autoscale_view()
    
    # Save the figure
    fig.savefig(filename, dpi=600, bbox_inches=None, facecolor='black')
    plt.close(fig)
    
    print(f"15m entry chart saved: {filename}")

    # Return filename and LVN price if available
    if volume_profile is not None:
        return filename, lvn_price
    else:
        return filename, None

def process_alarm_with_15m_analysis(df_1h, df_15m, alarm_index, alarm_price, alarm_timestamp):
    """Process alarm and generate both 1h and 30m analysis charts"""
    global alarm_count, first_chart_created
    
    alarm_count += 1
    
    print(f"\n=== PROCESSING ALARM #{alarm_count} ===")
    print(f"Alarm Time: {alarm_timestamp}")
    print(f"Alarm Price: {alarm_price:.2f}")
    
    # Create 1h alarm chart
    chart_1h = create_1h_alarm_chart(df_1h, alarm_index, alarm_price, alarm_count)
    
    # Find entry opportunities on 15m timeframe
    print(f"\nAnalyzing 15m data for entry opportunities (15° angle criterion)...")
    entry_df, best_entry, entry_signals = find_entry_opportunities_15m(df_15m, alarm_timestamp)
    
    # Create 15m entry chart if entry opportunities found
    chart_15m = None
    lvn_price = None
    if best_entry is not None:
        chart_15m, lvn_price = create_15m_entry_chart(df_15m, alarm_timestamp, best_entry, entry_signals, alarm_count)

    print(f"=== ALARM #{alarm_count} PROCESSING COMPLETE ===\n")

    return chart_1h, chart_15m, best_entry, lvn_price

def scan_for_alarms(df_1h, df_15m, break_on_first=False):
    """Main function to scan for alarms and process them"""
    global first_chart_created
    
    # Calculate The 4 Stochs (Quad) for 1h data
    df_1h['stoch_9_3'] = calculate_stochastic(df_1h, 9, 1, 3)     # Fast
    df_1h['stoch_14_3'] = calculate_stochastic(df_1h, 14, 1, 3)   # Standard
    df_1h['stoch_40_4'] = calculate_stochastic(df_1h, 40, 1, 4)   # Medium
    df_1h['stoch_60_10'] = calculate_stochastic(df_1h, 60, 10, 10) # Slow/Trend
    
    # Calculate Linear Regression Channel (Trend Context)
    window = 20
    df_1h = calculate_linear_regression_channel(df_1h, window)
    
    # NN Pattern Detection
    print("Running NN Pattern Detection on 1H data...")
    df_1h = identify_nn_patterns(df_1h)
    
    # Initialize marker columns
    df_1h['alarm_marker'] = np.nan
    df_1h['s9_marker'] = np.nan
    df_1h['s14_marker'] = np.nan
    df_1h['s40_marker'] = np.nan
    
    print("Processing Linear Regression and Alarms (this may take a moment)...")
    
    alarms_found = []
    
    for i in range(window, len(df_1h)):
        # Check Alarm Conditions
        s9 = df_1h['stoch_9_3'].iloc[i]
        s14 = df_1h['stoch_14_3'].iloc[i]
        s40 = df_1h['stoch_40_4'].iloc[i]
        s60 = df_1h['stoch_60_10'].iloc[i]
        s9_prev = df_1h['stoch_9_3'].iloc[i-1]
        slope = df_1h['slope'].iloc[i]
        
        # Condition A: Downward Channel (Slope is negative)
        is_downtrend = slope < -0.5 # Added small threshold to filter flat noise
        
        # Condition B: Quad Exhaustion (All low)
        is_quad_low = (s9 < 20) and (s14 < 25) and (s40 < 25) and (s60 < 25)
        
        # Condition C: Turn Up (Fast Stoch curls up)
        is_turn_up = (s9 > s9_prev)
        
        if is_downtrend and is_quad_low and is_turn_up:
            current_idx = df_1h.index[i]
            alarm_price = df_1h['close'].iloc[i]
            
            # Set marker values
            df_1h.iat[i, df_1h.columns.get_loc('alarm_marker')] = df_1h['low'].iloc[i] * 0.97
            df_1h.iat[i, df_1h.columns.get_loc('s9_marker')] = s9
            df_1h.iat[i, df_1h.columns.get_loc('s14_marker')] = s14
            df_1h.iat[i, df_1h.columns.get_loc('s40_marker')] = s40
            
            # Print alarm message
            print(f"[ALARM] {current_idx}: Quad Rotation in Downward Channel Detected @ {alarm_price:.2f}")

        # Check NN Alarm independently
        is_nn_w = df_1h['nn_alarm'].iloc[i] if 'nn_alarm' in df_1h.columns else False
        if is_nn_w:
             current_idx = df_1h.index[i]
             alarm_price = df_1h['close'].iloc[i]
             print(f"[NN ALARM] {current_idx}: Neural Network detected 'W' formation @ {alarm_price:.2f}")
             
             # If we haven't already processed this as a quad rotation, maybe do something?
             # For now, let's just mark it.
             df_1h.iat[i, df_1h.columns.get_loc('alarm_marker')] = df_1h['low'].iloc[i] * 0.95
            
            # Process alarm with 15m analysis
            chart_1h, chart_15m, best_entry, lvn_price = process_alarm_with_15m_analysis(df_1h, df_15m, i, alarm_price, current_idx)

            alarms_found.append({
                'timestamp': current_idx,
                'price': alarm_price,
                'chart_1h': chart_1h,
                'chart_15m': chart_15m,
                'best_entry': best_entry,
                'lvn_price': lvn_price
            })
            
            # Continue processing all alarms
            if break_on_first:
                first_chart_created = True
    
    return alarms_found

def simulate_trades(df_15m, alarms_found, initial_balance=10000, position_size_pct=0.05):
    """Simulate trades based on LVN entry and stochastic exit criteria"""

    trades = []
    balance = initial_balance
    btc_holdings = 0
    in_position = False
    entry_price = 0
    entry_timestamp = None
    trade_number = 0

    print("\n" + "="*80)
    print("TRADING SIMULATION REPORT")
    print("="*80)
    print(f"Initial Balance: ${balance:,.2f}")
    print(f"Position Size: {position_size_pct*100}% of networth per trade")
    print("Entry: LVN Price")
    print("Exit: When S9 > 80 AND S14 > 80 AND S40 > 80")
    print("="*80)

    for alarm in alarms_found:
        if alarm['best_entry'] is None or alarm['lvn_price'] is None:
            continue

        alarm_timestamp = alarm['timestamp']
        lvn_price = alarm['lvn_price']
        best_entry_timestamp = alarm['best_entry']['timestamp']

        # Find the LVN price timestamp - entry happens at LVN price
        # We look for when price reaches LVN after the alarm
        alarm_time = pd.to_datetime(alarm_timestamp)
        entry_window_df = df_15m[df_15m.index >= alarm_time].copy()

        if len(entry_window_df) == 0:
            continue

        # Find when price hits LVN level (within 1% tolerance)
        lvn_tolerance = 0.01
        entry_candidates = entry_window_df[
            (entry_window_df['low'] <= lvn_price * (1 + lvn_tolerance)) &
            (entry_window_df['high'] >= lvn_price * (1 - lvn_tolerance))
        ]

        if len(entry_candidates) == 0:
            # If price never hits LVN exactly, use the best entry price instead
            entry_idx = df_15m.index.get_loc(best_entry_timestamp)
            entry_price = lvn_price  # Still use LVN as entry price
            entry_timestamp = best_entry_timestamp
        else:
            # Use first time price hits LVN
            entry_idx = df_15m.index.get_loc(entry_candidates.index[0])
            entry_price = lvn_price
            entry_timestamp = entry_candidates.index[0]

        # Calculate position size in USD
        position_usd = balance * position_size_pct
        btc_amount = position_usd / entry_price

        # Look for exit signal after entry
        exit_window_df = df_15m[df_15m.index > entry_timestamp].copy()

        if len(exit_window_df) == 0:
            continue

        # Find exit: when all three stochastics are > 80
        exit_found = False
        for i in range(len(exit_window_df)):
            s9 = exit_window_df['stoch_9_3'].iloc[i]
            s14 = exit_window_df['stoch_14_3'].iloc[i]
            s40 = exit_window_df['stoch_40_4'].iloc[i]

            if s9 > 80 and s14 > 80 and s40 > 80:
                exit_timestamp = exit_window_df.index[i]
                exit_price = exit_window_df['close'].iloc[i]
                exit_found = True
                break

        if not exit_found:
            # No exit signal found, skip this trade
            continue

        # Calculate trade results
        trade_number += 1
        exit_usd = btc_amount * exit_price
        pnl_usd = exit_usd - position_usd
        pnl_pct = (exit_price / entry_price - 1) * 100
        balance += pnl_usd

        # Store trade
        trade = {
            'trade_number': trade_number,
            'alarm_time': alarm_timestamp,
            'entry_time': entry_timestamp,
            'entry_price': entry_price,
            'exit_time': exit_timestamp,
            'exit_price': exit_price,
            'position_usd': position_usd,
            'btc_amount': btc_amount,
            'pnl_usd': pnl_usd,
            'pnl_pct': pnl_pct,
            'balance_after': balance,
            'hold_duration': exit_timestamp - entry_timestamp
        }
        trades.append(trade)

        # Print trade details
        print(f"\nTrade #{trade_number}")
        print(f"  Alarm Time:    {alarm_timestamp}")
        print(f"  Entry Time:    {entry_timestamp}")
        print(f"  Entry Price:   ${entry_price:,.2f} (LVN)")
        print(f"  Position Size: ${position_usd:,.2f} ({btc_amount:.6f} BTC)")
        print(f"  Exit Time:     {exit_timestamp}")
        print(f"  Exit Price:    ${exit_price:,.2f}")
        print(f"  Hold Duration: {trade['hold_duration']}")
        print(f"  P&L:           ${pnl_usd:,.2f} ({pnl_pct:+.2f}%)")
        print(f"  Balance After: ${balance:,.2f}")
        print(f"  Exit Signals:  S9={s9:.1f}, S14={s14:.1f}, S40={s40:.1f}")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total Alarms Found:     {len(alarms_found)}")
    print(f"Total Trades Executed:  {len(trades)}")

    if len(trades) > 0:
        winning_trades = [t for t in trades if t['pnl_usd'] > 0]
        losing_trades = [t for t in trades if t['pnl_usd'] <= 0]

        total_pnl = sum(t['pnl_usd'] for t in trades)
        avg_pnl = total_pnl / len(trades)
        win_rate = len(winning_trades) / len(trades) * 100

        print(f"Winning Trades:         {len(winning_trades)} ({win_rate:.1f}%)")
        print(f"Losing Trades:          {len(losing_trades)}")

        if len(winning_trades) > 0:
            avg_win = sum(t['pnl_usd'] for t in winning_trades) / len(winning_trades)
            max_win = max(t['pnl_usd'] for t in winning_trades)
            print(f"Average Win:            ${avg_win:,.2f}")
            print(f"Max Win:                ${max_win:,.2f}")

        if len(losing_trades) > 0:
            avg_loss = sum(t['pnl_usd'] for t in losing_trades) / len(losing_trades)
            max_loss = min(t['pnl_usd'] for t in losing_trades)
            print(f"Average Loss:           ${avg_loss:,.2f}")
            print(f"Max Loss:               ${max_loss:,.2f}")

        print(f"\nInitial Balance:        ${initial_balance:,.2f}")
        print(f"Final Balance:          ${balance:,.2f}")
        print(f"Total P&L:              ${total_pnl:,.2f}")
        print(f"Return:                 {(balance/initial_balance - 1)*100:+.2f}%")
        print(f"Average P&L per Trade:  ${avg_pnl:,.2f}")

        avg_hold = sum(t['hold_duration'].total_seconds() for t in trades) / len(trades) / 3600
        print(f"Average Hold Duration:  {avg_hold:.1f} hours")

    print("="*80)

    return trades, balance

def main():
    """Main function with parameter support"""
    parser = argparse.ArgumentParser(description='Multi-Stochastic Quad Rotation Analysis')
    parser.add_argument('--timestamp', type=str, help='Specific timestamp to analyze (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--df', type=str, help='DataFrame CSV file to analyze (for single timestamp mode)')
    parser.add_argument('--break-on-first', action='store_true', default=True, help='Stop after first alarm (default: True)')
    parser.add_argument('--h1-file', type=str, default='BTCUSDT_1h_data.csv', help='1h data CSV file')
    parser.add_argument('--m15-file', type=str, default='BTCUSDT_15m_data.csv', help='15m data CSV file')


    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    try:
        df_1h = pd.read_csv(args.h1_file, index_col='timestamp', parse_dates=True)
        df_15m = pd.read_csv(args.m15_file, index_col='timestamp', parse_dates=True)
        print(f"Loaded {len(df_1h)} rows of 1h data and {len(df_15m)} rows of 15m data")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    if args.timestamp and args.df:
        # Single timestamp mode - analyze specific timestamp
        print(f"Analyzing specific timestamp: {args.timestamp}")
        
        # Load specific data file
        try:
            df_target = pd.read_csv(args.df, index_col='timestamp', parse_dates=True)
            print(f"Loaded {len(df_target)} rows from {args.df}")
        except Exception as e:
            print(f"Error loading target data: {e}")
            return
        
        # Find the timestamp in 1h data
        target_time = pd.to_datetime(args.timestamp)
        alarm_matches = df_1h[df_1h.index == target_time]
        
        if len(alarm_matches) == 0:
            print(f"Timestamp {args.timestamp} not found in 1h data")
            return
        
        alarm_index = alarm_matches.index[0]
        alarm_price = alarm_matches['close'].iloc[0]
        
        print(f"Found alarm at {alarm_index} @ {alarm_price:.2f}")
        
        # Process this specific alarm
        chart_1h, chart_15m, best_entry, lvn_price = process_alarm_with_15m_analysis(df_1h, df_15m,
                                                                          df_1h.index.get_loc(alarm_index),
                                                                          alarm_price, alarm_index)
        
        if best_entry:
            print(f"\nBest entry found: {best_entry['timestamp']} @ {best_entry['price']:.2f}")
        else:
            print("No suitable entry found in 15m data")
            
    else:
        # Full scan mode - look for alarms automatically
        print("Running full alarm scan...")
        alarms_found = scan_for_alarms(df_1h, df_15m, args.break_on_first)

        print(f"\nTotal alarms detected: {len(alarms_found)}")

        # Run trading simulation
        if len(alarms_found) > 0:
            trades, final_balance = simulate_trades(df_15m, alarms_found)

        if not args.break_on_first and len(alarms_found) > 0:
            # Generate summary chart if we processed all alarms
            print("Generating summary chart...")
            create_summary_chart(df_1h, len(alarms_found))

def create_summary_chart(df_1h, alarm_count):
    """Generate summary chart showing all alarms"""
    
    # Slice data for better visibility (last 300 candles)
    plot_df = df_1h.tail(300) 
    
    apds = [
        # 1. Channel Lines (Teal/Cyan)
        mpf.make_addplot(plot_df['linreg_top'], color='teal', width=0.8, panel=0),
        mpf.make_addplot(plot_df['linreg_bot'], color='teal', width=0.8, panel=0),
        mpf.make_addplot(plot_df['linreg_mid'], color='teal', linestyle='--', width=0.5, panel=0),
        
        # 2. Alarm Markers on Price
        mpf.make_addplot(plot_df['alarm_marker'], type='scatter', markersize=100, marker='^', color='cyan', panel=0),

        # 3. Stochastics (Stacked Panels) - with white borders
        mpf.make_addplot(plot_df['stoch_9_3'], panel=2, color='gold', ylabel='S(9,3)', ylim=(0,100), width=1.5),
        mpf.make_addplot(plot_df['s9_marker'], type='scatter', panel=2, markersize=50, marker='^', color='cyan'),
        
        mpf.make_addplot(plot_df['stoch_14_3'], panel=3, color='dodgerblue', ylabel='S(14,3)', ylim=(0,100), width=1.5),
        mpf.make_addplot(plot_df['s14_marker'], type='scatter', panel=3, markersize=50, marker='^', color='cyan'),

        mpf.make_addplot(plot_df['stoch_40_4'], panel=4, color='white', ylabel='S(40,4)', ylim=(0,100), width=1.5),
        mpf.make_addplot(plot_df['s40_marker'], type='scatter', panel=4, markersize=50, marker='^', color='cyan'),

        mpf.make_addplot(plot_df['stoch_60_10'], panel=5, color='magenta', ylabel='S(60,10)', ylim=(0,100), width=1.5),
    ]
    
    # Style Setup with white subplot borders
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='inherit', wick='inherit', volume='in')
    s  = mpf.make_mpf_style(marketcolors=mc, 
                            gridstyle=':', 
                            y_on_right=True, 
                            facecolor='black', 
                            figcolor='black', 
                            gridcolor='#333333', 
                            rc={'xtick.color':'white', 'ytick.color':'white', 'axes.labelcolor':'white',
                                'axes.edgecolor':'white', 'axes.linewidth':0.8})
    
    print("Generating Summary Chart...")
    mpf.plot(plot_df,
             type='candle',
             volume=True,
             style=s,
             addplot=apds,
             title=f"\nBTCUSDT - Downward Channel Quad Rotation Summary ({alarm_count} alarms detected)",
             panel_ratios=(4,1,1,1,1,1),
             figscale=2.5,
             hlines=dict(hlines=[20,80], colors=['gray','gray'], linestyle='--', linewidths=0.5),
             datetime_format='%d.%m %H:%M:%S',
             savefig=dict(fname="summary_chart.png", dpi=600, bbox_inches='tight', facecolor='black')
             )

if __name__ == "__main__":
    main()