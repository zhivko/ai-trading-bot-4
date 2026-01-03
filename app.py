from flask import Flask, render_template, jsonify, request
import pandas as pd
import os
import re
from datetime import datetime, timedelta
import threading
import time
import sys
from chart_generator import generate_chart_data, get_chart_metadata, get_nn_model
from data_updater import update_data_files

app = Flask(__name__)

# Pre-load Neural Network Model (STRICT)
# The app cannot function without the brain.
print("Pre-loading Neural Network Model...")
if get_nn_model() is None:
    print("CRITICAL ERROR: Failed to load Neural Network Model. Terminating...")
    sys.exit(1)

# Global variable to store available data files
data_files = []
last_update_time = None

def parse_filename(filename):
    """
    Parse filename to extract symbol and timeframe.
    Expected format: SYMBOL_TIMEFRAME_data.csv
    Example: BTCUSDT_1h_data.csv -> symbol: BTCUSDT, timeframe: 1h
    """
    pattern = r'^([A-Z]+)_(\d+[mhd])_data\.csv$'
    match = re.match(pattern, filename)

    if match:
        return {
            'symbol': match.group(1),
            'timeframe': match.group(2),
            'filename': filename
        }
    return None

def scan_data_directory():
    """Scan data directory for CSV files and parse their info"""
    global data_files
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return []

    files = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            parsed = parse_filename(filename)
            if parsed:
                filepath = os.path.join(data_dir, filename)
                try:
                    df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
                    parsed['last_candle_time'] = df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                    parsed['total_candles'] = len(df)
                    parsed['date_range'] = f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}"
                    files.append(parsed)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    data_files = files
    return files

@app.route('/')
def index():
    """Main dashboard showing all available data files"""
    scan_data_directory()
    return render_template('index.html', data_files=data_files, last_update=last_update_time)

@app.route('/api/files')
def get_files():
    """API endpoint to get list of available data files"""
    scan_data_directory()
    return jsonify({
        'files': data_files,
        'last_update': last_update_time.strftime('%Y-%m-%d %H:%M:%S') if last_update_time else None
    })

@app.route('/chart/<symbol>/<timeframe>')
def view_chart(symbol, timeframe):
    """Display chart for specific symbol and timeframe"""
    filename = f"{symbol}_{timeframe}_data.csv"
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        return f"Data file not found: {filename}", 404

    try:
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        last_candle = df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
        total_candles = len(df)

        return render_template('chart.html',
                             symbol=symbol,
                             timeframe=timeframe,
                             last_candle=last_candle,
                             total_candles=total_candles)
    except Exception as e:
        return f"Error loading data: {str(e)}", 500

@app.route('/api/chart/<symbol>/<timeframe>')
def get_chart_data(symbol, timeframe):
    """Generate and return interactive chart data as JSON"""
    filename = f"{symbol}_{timeframe}_data.csv"
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        return jsonify({'error': 'Data file not found'}), 404

    try:
        # Get number of candles from query parameter (default 572 for Dec 28 range)
        num_candles = request.args.get('candles', 572, type=int)
        
        # Get date range parameters
        start_date = request.args.get('start')
        end_date = request.args.get('end')
        
        # Get NN threshold parameter (default 30 - baseline)
        nn_threshold = request.args.get('nn_threshold', 30, type=int)
        
        # Get Exit Strategy parameter (default 3 - LVN Accel)
        exit_strategy = request.args.get('exit_strategy', 3, type=int)

        # Generate chart data as JSON
        start_time = time.time()
        chart_json_str = generate_chart_data(filepath, symbol, timeframe, num_candles, start_date, end_date, nn_threshold, exit_strategy)
        total_time = time.time() - start_time
        print(f"DEBUG: /api/chart/{symbol}/{timeframe} - Total Generation Time: {total_time:.4f}s")

        # Inject timing info into JSON if it's a valid JSON string
        try:
            import json
            chart_data = json.loads(chart_json_str)
            if 'metadata' not in chart_data:
                chart_data['metadata'] = {}
            chart_data['metadata']['perf_total_sec'] = round(total_time, 4)
            # Find and extract the internal timings if they were printed to stdout (hard to catch)
            # or just assume they are for internal logs. 
            # For now just return the total.
            chart_json_str = json.dumps(chart_data)
        except Exception as e:
            print(f"Warning: Failed to inject timing metadata: {e}")

        # Return raw JSON string with correct content type
        from flask import Response
        return Response(chart_json_str, mimetype='application/json')
    except Exception as e:
        import traceback
        print(f"ERROR in /api/chart/{symbol}/{timeframe}:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/metadata/<symbol>/<timeframe>')
def get_metadata(symbol, timeframe):
    """Get chart metadata"""
    filename = f"{symbol}_{timeframe}_data.csv"
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        return jsonify({'error': 'Data file not found'}), 404

    try:
        metadata = get_chart_metadata(filepath)
        return jsonify(metadata)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategy_preview/<strategy_id>/<timestamp>')
def get_strategy_preview(strategy_id, timestamp):
    """
    Get detailed trade setup for a specific strategy at a specific time.
    Used for interactive frontend visualization.
    """
    symbol = request.args.get('symbol', 'BTCUSDT')
    timeframe = request.args.get('timeframe', '15m')
    
    filename = f"{symbol}_{timeframe}_data.csv"
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Data file not found'}), 404
        
    try:
        # Load Data
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        
        # Find index for timestamp
        ts = pd.to_datetime(timestamp)
        if df.index.tz is None and ts.tz is not None:
            ts = ts.tz_localize(None)
            
        # Get specific candle index
        # We need exact match or nearest past match
        try:
            entry_idx = df.index.get_loc(ts)
        except KeyError:
             # Find nearest
             entry_idx = df.index.searchsorted(ts)
             if entry_idx >= len(df): entry_idx = len(df) - 1
        
        # Calculate NN Confidence for this window so sidebar is accurate
        from chart_generator import identify_nn_patterns
        # Process a small buffer around the entry point to ensure stochastics/NN have context
        # 400 candles is enough for stochastics warm-up + NN window
        subset_df = df.iloc[max(0, entry_idx-400):entry_idx+1].copy()
        subset_df = identify_nn_patterns(subset_df)
        
        # Dispatch to appropriate strategy module
        setup_data = {}
        if str(strategy_id) == '3':
            import backtest_strategy_3_lvn_acceleration as strat3
            setup_data = strat3.get_trade_setup(subset_df, len(subset_df)-1)
        else:
            setup_data = {
                'error': f"Strategy {strategy_id} preview not implemented yet",
                'description': "Preview not available for this strategy."
            }
            
        return jsonify(setup_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/latest/<symbol>/<timeframe>')
def get_latest_data(symbol, timeframe):
    """Get latest candle data for specific symbol/timeframe"""
    filename = f"{symbol}_{timeframe}_data.csv"
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        return jsonify({'error': 'Data file not found'}), 404

    try:
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        latest = df.iloc[-1]

        return jsonify({
            'timestamp': df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
            'open': float(latest['open']),
            'high': float(latest['high']),
            'low': float(latest['low']),
            'close': float(latest['close']),
            'volume': float(latest['volume'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/update')
def trigger_update():
    """Manually trigger data update"""
    global last_update_time
    try:
        update_data_files()
        last_update_time = datetime.now()
        return jsonify({
            'status': 'success',
            'message': 'Data updated successfully',
            'timestamp': last_update_time.strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def background_update_task():
    """Background task to update data files periodically"""
    global last_update_time
    while True:
        try:
            print(f"[{datetime.now()}] Running background data update...")
            update_data_files()
            last_update_time = datetime.now()
            print(f"[{datetime.now()}] Update completed successfully")
        except Exception as e:
            print(f"[{datetime.now()}] Error in background update: {e}")

        # Wait 15 minutes before next update
        time.sleep(900)  # 15 minutes = 900 seconds

if __name__ == '__main__':
    # Start background update task
    update_thread = threading.Thread(target=background_update_task, daemon=True)
    update_thread.start()

    # Scan data directory on startup
    scan_data_directory()

    # Run Flask app
    # Set reloader type to 'watchdog' to avoid watching site-packages
    os.environ['FLASK_RUN_RELOAD_TYPE'] = 'watchdog'
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)
