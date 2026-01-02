# Crypto Price Monitor - Web Platform

A Flask-based web application for monitoring cryptocurrency OHLC (Open, High, Low, Close) prices with **interactive, zoomable, and scrollable** chart visualizations and automatic data updates.

## Features

- **Interactive Charts**: Fully interactive Plotly charts with zoom, pan, and scroll functionality
- **Real-time Price Monitoring**: Automatically updates CSV files with latest data from Binance
- **Beautiful Visualizations**: OHLC candlestick charts with 4 stochastic indicators
- **Smart File Parser**: Automatically detects symbol and timeframe from CSV filenames
- **Background Updates**: Automatically fetches missing data every 15 minutes
- **Responsive Dashboard**: Clean, modern web interface with real-time updates
- **Multiple Timeframes**: Supports any timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, etc.)
- **Dynamic Candle Selection**: Choose to display 50, 100, 200, 500, or 1000 candles

## File Structure

```
ai-trading-bot-4/
├── app.py                    # Main Flask application with API routes
├── chart_generator.py        # Interactive Plotly chart generation
├── data_updater.py          # Background task for updating CSV files
├── templates/
│   ├── index.html           # Dashboard homepage
│   └── chart.html           # Interactive chart viewer with Plotly.js
├── data/
│   ├── BTCUSDT_1h_data.csv  # Example: BTC 1-hour data
│   └── BTCUSDT_15m_data.csv # Example: BTC 15-minute data
└── requirements.txt         # Python dependencies
```

## CSV File Naming Convention

The application automatically parses CSV filenames to extract symbol and timeframe information.

**Format:** `SYMBOL_TIMEFRAME_data.csv`

**Examples:**
- `BTCUSDT_1h_data.csv` → Symbol: BTCUSDT, Timeframe: 1h
- `ETHUSDT_15m_data.csv` → Symbol: ETHUSDT, Timeframe: 15m
- `BNBUSDT_4h_data.csv` → Symbol: BNBUSDT, Timeframe: 4h

**Supported Timeframes:**
- Minutes: `1m`, `3m`, `5m`, `15m`, `30m`
- Hours: `1h`, `2h`, `4h`, `6h`, `8h`, `12h`
- Days: `1d`, `3d`
- Weeks: `1w`

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have CSV data files in the `data/` directory following the naming convention.

## Usage

### Running the Web Application

```bash
python app.py
```

The application will:
- Start on `http://localhost:5000`
- Automatically scan the `data/` directory for CSV files
- Start a background task to update data every 15 minutes

### Manually Update Data

Run the data updater independently:
```bash
python data_updater.py
```

This will:
- Scan all CSV files in `data/` directory
- Parse symbol and timeframe from filenames
- Fetch missing data from Binance API
- Update CSV files with new candles

## Web Interface

### Dashboard (`/`)
- Displays all available data files as cards
- Shows last candle time, total candles, and date range
- "Update Now" button to manually trigger data updates
- Auto-refreshes every 5 minutes

### Chart Viewer (`/chart/<symbol>/<timeframe>`)
- Displays OHLC chart with stochastic indicators
- Shows latest price data (Open, High, Low, Close, Volume)
- Auto-refreshes every 2 minutes
- Interactive refresh button

## API Endpoints

### `GET /api/files`
Returns list of available data files with metadata
```json
{
  "files": [
    {
      "symbol": "BTCUSDT",
      "timeframe": "1h",
      "filename": "BTCUSDT_1h_data.csv",
      "last_candle_time": "2025-12-25 12:00:00",
      "total_candles": 5000,
      "date_range": "2017-08-17 to 2025-12-25"
    }
  ],
  "last_update": "2025-12-25 12:15:00"
}
```

### `GET /api/chart/<symbol>/<timeframe>`
Returns chart image (PNG) for specified symbol and timeframe

### `GET /api/latest/<symbol>/<timeframe>`
Returns latest candle data
```json
{
  "timestamp": "2025-12-25 12:00:00",
  "open": 96500.0,
  "high": 96800.0,
  "low": 96400.0,
  "close": 96750.0,
  "volume": 1234.56
}
```

### `GET /api/update`
Manually trigger data update for all files

## Chart Visualization

Charts include:
- **OHLC Candlesticks**: Green (up) / Red (down) candles
- **Volume**: Volume bars at the bottom
- **Stochastic Indicators** (4 panels):
  - S(9,3) - Fast stochastic (gold)
  - S(14,3) - Standard stochastic (blue)
  - S(40,4) - Medium stochastic (white)
  - S(60,10) - Slow/Trend stochastic (magenta)
- **Horizontal Lines**: 20 (oversold) and 80 (overbought) levels

Chart style matches the original `create_1h_entry_chart` function with:
- Black background
- White borders and labels
- Professional trading platform appearance

## Background Data Updates

The application automatically:
1. Scans all CSV files in `data/` directory
2. Parses symbol and timeframe from filenames
3. Checks the last candle timestamp
4. Fetches missing data from Binance API
5. Appends new complete candles to CSV files
6. Runs every 15 minutes

**Note:** The updater only adds complete candles (excludes the current forming candle).

## How It Works

### File Parsing
```python
# Filename: BTCUSDT_1h_data.csv
# Parsed as:
{
  'symbol': 'BTCUSDT',      # Trading pair
  'timeframe': '1h'          # 1-hour candles
}
```

### Data Update Logic
1. Read last timestamp from CSV: `2025-12-25 10:00:00`
2. Calculate next expected candle: `2025-12-25 11:00:00` (for 1h timeframe)
3. Fetch data from Binance from `11:00:00` to `current_time`
4. Remove incomplete last candle (still forming)
5. Append complete candles to CSV
6. Save updated file

### Chart Generation
1. Load last 100 candles from CSV
2. Calculate 4 stochastic indicators
3. Generate chart using mplfinance
4. Save to `static/charts/` with timestamp
5. Return image path

## Configuration

### Change Update Interval
Edit [app.py](app.py:153):
```python
# Wait 15 minutes before next update
time.sleep(15 * 60)  # Change to desired interval in seconds
```

### Change Number of Candles in Chart
Edit [chart_generator.py](chart_generator.py:19):
```python
def generate_ohlc_chart(filepath, symbol, timeframe, num_candles=100):
    # Change num_candles to desired value
```

### Change Chart Resolution
Edit [chart_generator.py](chart_generator.py:107):
```python
fig.savefig(output_path, dpi=150, ...)  # Increase DPI for higher resolution
```

## Troubleshooting

### No Data Files Found
- Ensure CSV files are in the `data/` directory
- Check filename format: `SYMBOL_TIMEFRAME_data.csv`
- Symbol must be uppercase (e.g., `BTCUSDT`, not `btcusdt`)

### Update Fails
- Check internet connection
- Verify Binance API is accessible
- Check CSV file format (must have: timestamp, open, high, low, close, volume columns)

### Chart Generation Error
- Ensure CSV has enough data (at least 60 candles for stochastic calculations)
- Check CSV file is not corrupted
- Verify all required columns exist

## Development

### Adding New Features
- Add new routes in `app.py`
- Create new chart types in `chart_generator.py`
- Extend data sources in `data_updater.py`

### Custom Indicators
Edit `chart_generator.py` to add more technical indicators:
```python
# Add RSI, MACD, etc.
apds.append(mpf.make_addplot(...))
```

## Credits

Based on the chart visualization from `multi_stochastic_quad_rotation.py`, particularly the `create_1h_entry_chart` and `create_15m_entry_chart` functions.

## License

Use freely for personal and commercial projects.
