# AI Trading Bot v4 - Interactive Market Monitor

A sophisticated cryptocurrency trading bot and market visualization platform that combines deep learning pattern detection with advanced market structure analysis.

## 🚀 Key Technologies

### Backend (Python)
- **Framework**: Flask (Web API & Template Rendering)
- **Deep Learning**: PyTorch (1D CNN for Pattern Recognition)
- **Data Engineering**: Pandas & NumPy (High-performance vectorized analysis)
- **Market Structure**: Scikit-Learn (Gaussian Mixture Models for Volume Profile peak/node detection)
- **Signal Logic**: Vectorized implementation of Stochastic, ATR, and Linear Regression Channels.

### Frontend (Web UI)
- **Visualization**: Plotly.js (Professional-grade interactive charting)
- **Styling**: Vanilla CSS (Tailored glassmorphic dark mode aesthetics)
- **Logic**: Modern JavaScript (Asynchronous API polling & real-time UI updates)

---

## ✨ Features

- **Interactive Plotly Chart**: Smooth zooming, panning, and auto-following of live price action.
- **Neural Network "W-Pattern" Discovery**: A custom-trained PyTorch model detects high-probability "Stochastic Low" resets.
- **Dynamic Volume Profile Analysis**: Real-time calculation of Point of Control (POC), Value Area (VAH/VAL), and High/Low Volume Nodes.
- **Hybrid Signal Engine**: Combines technical channel breakouts with Deep Learning confirmations for higher accuracy.
- **Strategy Backtesting**: Simulate multiple exit strategies (LVN Acceleration, Multi-tier, Trailing Stop) directly in the UI.
- **Live Data Polling**: The dashboard and charts automatically update as new price data is fetched.

---

## 🛠️ Installation & Setup

1. **Environment Setup**:
   Ensure you have Python 3.9+ installed. It is recommended to use a virtual environment.
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Preparation**:
   Place your historical CSV data in the `data/` directory. Files must follow the format: `SYMBOL_TIMEFRAME_data.csv` (e.g., `BTCUSDT_15m_data.csv`).

4. **Run the Application**:
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000`.

---

## 📈 Usage Guide

1. **Dashboard**: Select a data set from the grid to launch the interactive chart.
2. **Chart Controls**:
   - **Visible Candles**: Adjust the zoom level (defaults to Dec 28 range).
   - **NN Confidence**: Filter signals based on the Neural Network's model certainty (recommended baseline: 30%).
   - **Exit Strategy**: Switch between different backtest strategies to see how they would have performed.
3. **Marker Interaction**:
   - **Click a Star Marker**: Opens the **Trade Setup** sidebar, displaying exact Entry, Stop Loss, and target LVNs.
   - **Mini-VP**: The sidebar renders a localized Volume Profile and GMM Gaussian curve for the trade context.
4. **Navigation**:
   - **Zoom**: Drag on any area of the chart.
   - **Pan**: Hold **Shift** and drag to scroll through historical data.
   - **Reset View**: Double-click anywhere on the chart.
