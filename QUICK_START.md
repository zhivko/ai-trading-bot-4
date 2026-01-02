# Quick Start - Testing Charts

## ⚡ Fastest Way (30 seconds)

### Step 1: Start Flask
```powershell
.venv\Scripts\python.exe app.py
```

**You should see:**
```
 * Running on http://127.0.0.1:5000
```

### Step 2: Open Browser
Open: **http://localhost:5000**

### Step 3: Click Any Chart
Click on "BTCUSDT - 1h" (or any other card)

### Step 4: Verify Chart
✅ **SUCCESS if you see:**
- Chart with 6 panels (OHLC + Volume + 4 Stochastics)
- Dropdown shows "100" selected
- You can zoom by dragging
- Colors: Green/Red candles, Gold/Blue/White/Magenta stochastic lines

❌ **FAILED if you see:**
- Blank screen
- Loading spinner stuck
- Error message
- Less than 6 panels

---

## 🔍 Quick Checks

### Check 1: Default Candles
Look at the dropdown - should show "100" selected (not 200)

### Check 2: Auto-Range Y-Axis
1. Zoom into any area by dragging
2. Y-axis (price) should adjust to show only zoomed data
3. No big gaps above/below price candles

### Check 3: All Panels Present
Count the panels from top to bottom:
1. Price (candlesticks)
2. Volume (bars)
3. S(9,3) - Gold line
4. S(14,3) - Blue line
5. S(40,4) - White line
6. S(60,10) - Magenta line

### Check 4: Interactivity
- **Drag** = Zoom in
- **Shift+Drag** = Pan
- **Double-click** = Reset
- **Hover** = See details

---

## 🐛 If It Doesn't Work

### Flask Won't Start
```powershell
# Make sure virtual environment is activated
.venv\Scripts\python.exe -m pip install -r requirements.txt

# Try again
.venv\Scripts\python.exe app.py
```

### Chart Shows Blank
**Open browser console (F12):**
- Look for red errors
- Common: "Plotly is not defined" - CDN issue
- Common: "Failed to fetch" - API issue

**Test API directly:**
```
http://localhost:5000/api/chart/BTCUSDT/1h?candles=100
```

Should return JSON with `data` and `layout` fields.

### Wrong Number of Candles
1. Check dropdown selection
2. Refresh page (Ctrl+R)
3. Clear browser cache

---

## 📸 Visual Testing (With Screenshots)

If you installed Node.js and want automated screenshots:

```powershell
# Run the test
.\run_simple_test.ps1

# Check results
# - quick_test.png (if passed)
# - quick_test_error.png (if failed)
```

---

## ✅ Success Criteria

Your chart is working correctly if:
- ✅ 6 panels visible
- ✅ 100 candles showing (check dropdown)
- ✅ Y-axis scales automatically when zooming
- ✅ Can interact (zoom, pan, hover)
- ✅ Price labels visible on Y-axis
- ✅ Time labels visible on X-axis
- ✅ No JavaScript errors in console

---

## 🎯 Expected Behavior

**On Page Load:**
1. Dashboard shows data cards (1-2 seconds)
2. Click card → Chart page loads (1-2 seconds)
3. Plotly loads from CDN (1-2 seconds)
4. Chart renders (2-3 seconds)
5. **Total: ~5-10 seconds from click to fully rendered chart**

**After Loaded:**
- Zoom/Pan = Instant
- Hover = Instant
- Change candle count = 2-3 seconds to re-render

---

## 💡 Pro Tips

1. **Use Chrome/Edge** - Best Plotly support
2. **Check Internet** - Plotly loads from CDN
3. **Wait 5 seconds** - After page load before declaring failure
4. **F12 Console** - Shows helpful error messages
5. **Ctrl+Shift+R** - Hard refresh if cached

---

## 📞 Still Not Working?

1. Check `screenshots/` folder for visual evidence (if using Puppeteer)
2. Check Flask console output for errors
3. Check browser console (F12) for JavaScript errors
4. Verify CSV files exist in `data/` directory
5. Try the standalone test: Open `test_chart_standalone.html` in browser

---

**That's it! The chart should work in under 1 minute.**
