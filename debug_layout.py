from chart_generator import generate_chart_data
import json
import os

# Create dummy data file if needed, or use existing
filepath = "data/BTCUSDT_15m_data.csv"
if not os.path.exists(filepath):
    # Just list files to see if anything is there
    print("No data file found for debug")
else:
    result_json = generate_chart_data(filepath, "BTCUSDT", "15m", num_candles=100)
    data = json.loads(result_json)
    layout = data.get('layout', {})
    print(f"Hovermode: {layout.get('hovermode')}")
    print(f"Hoversubplots: {layout.get('hoversubplots')}")
    
    for i in range(1, 7):
        suffix = "" if i == 1 else str(i)
        key = f"xaxis{suffix}"
        xaxis = layout.get(key, {})
        print(f"{key}: showspikes={xaxis.get('showspikes')}, spikemode={xaxis.get('spikemode')}, matches={xaxis.get('matches')}")
