from flask import Flask, render_template, jsonify, request
import pandas as pd
import os
import re
from datetime import datetime, timedelta
import threading
import time
from chart_generator import generate_chart_data, get_chart_metadata
from data_updater import update_data_files

app = Flask(__name__)

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
        # Get number of candles from query parameter (default 100)
        num_candles = request.args.get('candles', 100, type=int)
        
        # Get date range parameters
        start_date = request.args.get('start')
        end_date = request.args.get('end')
        
        # Get NN threshold parameter (default 5 - lowered to show more signals)
        nn_threshold = request.args.get('nn_threshold', 5, type=int)

        # Generate chart data as JSON
        chart_json = generate_chart_data(filepath, symbol, timeframe, num_candles, start_date, end_date, nn_threshold)

        # Return raw JSON string with correct content type
        from flask import Response
        return Response(chart_json, mimetype='application/json')
    except Exception as e:
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
