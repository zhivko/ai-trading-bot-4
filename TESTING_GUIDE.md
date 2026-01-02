# Testing Guide - Flask Web Monitor

Multiple ways to test that charts are rendering correctly with 100 candles and auto-ranging Y-axis.

## Quick Tests (No Installation Required)

### 1. Manual Browser Test (Easiest)

1. Start Flask app:
   ```bash
   .venv\Scripts\python.exe app.py
   ```

2. Open browser to: `http://localhost:5000`

3. Click on any data card (e.g., BTCUSDT - 1h)

4. **Verify**:
   - ✅ Chart loads and displays
   - ✅ Dropdown shows "100" selected (default)
   - ✅ Chart shows 6 panels (OHLC + Volume + 4 Stochastics)
   - ✅ You can zoom by dragging
   - ✅ You can pan by Shift+dragging
   - ✅ Prices are visible and scaled properly

### 2. Standalone HTML Test

1. Start Flask app (keep it running)

2. Open `test_chart_standalone.html` in your browser

3. **Verify**:
   - ✅ Status shows "Chart loaded successfully!"
   - ✅ Chart renders with all panels
   - ✅ Console shows no errors (press F12)

### 3. API Test (curl/browser)

Test if API returns data:

```bash
# Test chart data endpoint
curl http://localhost:5000/api/chart/BTCUSDT/1h?candles=100

# Should return JSON with "data" and "layout" fields
```

Or open in browser:
```
http://localhost:5000/api/chart/BTCUSDT/1h?candles=100
```

## Automated Tests

### 4. Python Test Script

Quick verification that modules load:

```bash
.venv\Scripts\python.exe test_app.py
```

**Expected output**:
```
✅ PASS - Package Imports
✅ PASS - Custom Modules
✅ PASS - Data Directory
✅ PASS - Chart Generation
```

### 5. Simple Puppeteer Test (Fast)

Quick visual test with screenshot:

**Prerequisites**: Install Node.js and run `npm install`

```bash
node test_simple.js
```

**Expected output**:
```
✅ Plotly loaded
✅ Chart is rendering correctly!
📸 Screenshot: quick_test.png
```

### 6. Full Puppeteer Test Suite (Comprehensive)

Complete browser automation test:

**Option A - Automated** (starts Flask automatically):
```bash
test_with_puppeteer.bat
```

**Option B - Manual** (Flask already running):
```bash
node test_charts_puppeteer.js
```

**Expected output**:
```
📊 Testing Dashboard...
   ✅ Dashboard loaded with 2 data card(s)

🔌 Testing API Endpoints...
   ✅ /api/files returned 2 file(s)
   ✅ Latest price: 96750.0
   ✅ Chart data: 6 traces, layout: true

📈 Testing Chart: BTCUSDT 1h...
   ✅ Plotly.js loaded
   ✅ Chart rendered successfully
   📊 Chart traces: 6
   🕯️  Candles displayed: 100
   ✅ All chart panels rendered
```

Screenshots saved to `screenshots/` folder.

## What to Verify

### Default Configuration
- ✅ **100 candles** displayed by default (not 200)
- ✅ Dropdown shows "100" selected
- ✅ Chart shows last 100 candles from CSV

### Y-Axis Auto-Range
- ✅ Price chart Y-axis scales to visible data
- ✅ When you zoom, Y-axis adjusts automatically
- ✅ No unnecessary whitespace above/below price chart
- ✅ Volume panel also auto-scales
- ✅ Stochastic panels fixed at 0-100 range

### Chart Panels (6 total)
1. ✅ **Panel 1**: OHLC Candlesticks (green/red)
2. ✅ **Panel 2**: Volume bars
3. ✅ **Panel 3**: Stochastic (9,3) - gold line
4. ✅ **Panel 4**: Stochastic (14,3) - blue line
5. ✅ **Panel 5**: Stochastic (40,4) - white line
6. ✅ **Panel 6**: Stochastic (60,10) - magenta line

### Interactivity
- ✅ Zoom by dragging
- ✅ Pan by Shift+dragging
- ✅ Scroll wheel zooms
- ✅ Double-click resets view
- ✅ Hover shows candle details
- ✅ Can change candle count (50, 100, 200, 500, 1000)

## Troubleshooting Tests

### Charts Not Rendering?

1. **Check Flask is running**:
   ```bash
   curl http://localhost:5000
   ```
   Should return HTML (dashboard page)

2. **Check API returns data**:
   ```bash
   curl http://localhost:5000/api/chart/BTCUSDT/1h?candles=100
   ```
   Should return JSON

3. **Check browser console** (F12):
   - Look for JavaScript errors
   - Look for "Plotly" undefined errors
   - Look for network errors (failed to load data)

4. **Check CSV files exist**:
   ```bash
   dir data\*.csv
   ```
   Should show at least one CSV file

### Puppeteer Test Fails?

1. **Node.js not installed**:
   - Download from https://nodejs.org/
   - Run `npm install`

2. **Flask not responding**:
   - Manually start Flask first
   - Wait 10 seconds before running test
   - Check Flask console for errors

3. **Timeout errors**:
   - Increase timeout in test script
   - Check your internet connection (CDN load)
   - Try `test_simple.js` first (faster)

4. **"Cannot read properties of undefined"**:
   - This is a Plotly internal error (usually harmless)
   - Check if screenshots show chart rendered
   - Try manual browser test instead

### Chart Shows But Has Issues?

1. **Wrong number of candles**:
   - Check dropdown selection
   - Check browser console for errors
   - Verify API URL includes `?candles=100`

2. **Y-axis not auto-ranging**:
   - Zoom in/out and check if it adjusts
   - Check [chart_generator.py:148-150](c:\git\ai-trading-bot-4\chart_generator.py#L148-L150)
   - Should see `autorange=True` for rows 1 and 2

3. **Missing panels**:
   - Should see 6 panels total
   - Check browser console for trace errors
   - Verify API returns 6+ traces

## Quick Fix Checklist

If charts aren't working:

- [ ] Flask app is running on port 5000
- [ ] CSV files exist in `data/` directory
- [ ] CSV files are named correctly (SYMBOL_TIMEFRAME_data.csv)
- [ ] Plotly.js CDN is accessible (check internet)
- [ ] Browser console shows no errors
- [ ] API endpoint returns JSON data
- [ ] Python dependencies installed (`pip install -r requirements.txt`)

## Test Results Interpretation

### Success Indicators
- ✅ All 6 chart panels visible
- ✅ Candles displayed (not just loading spinner)
- ✅ Can interact with chart (zoom, pan)
- ✅ Price labels on Y-axis
- ✅ Time labels on X-axis
- ✅ No console errors

### Failure Indicators
- ❌ Blank chart area
- ❌ "Loading..." spinner stuck
- ❌ Console errors about Plotly
- ❌ Console errors about data fetching
- ❌ Wrong number of candles
- ❌ Y-axis not scaling properly

## Performance Benchmarks

Expected performance on modern hardware:

- **Dashboard load**: < 1 second
- **Chart API response**: < 500ms (100 candles)
- **Chart render time**: < 2 seconds
- **Zoom/pan response**: Instant
- **Auto-refresh**: Every 2 minutes

## Test Coverage

The test suite verifies:

✅ Flask app starts and serves pages
✅ Dashboard displays data cards
✅ API endpoints return valid JSON
✅ Chart page loads HTML template
✅ Plotly.js CDN loads
✅ Chart data fetches from API
✅ Chart renders with 6 panels
✅ Default 100 candles displayed
✅ Y-axis auto-ranges
✅ Price data displays correctly
✅ Zoom functionality works
✅ Screenshots capture visual state

## Contact/Issues

If tests still fail after following this guide:
1. Check `screenshots/` folder for visual evidence
2. Check Flask console for server errors
3. Check browser console (F12) for JavaScript errors
4. Review error messages carefully
