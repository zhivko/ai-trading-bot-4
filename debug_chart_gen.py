import pandas as pd
import json
import os
from chart_generator import generate_chart_data

def debug_chart():
    print("Running Debug Chart Generation...")
    
    # Check if data file exists
    filepath = 'data/BTCUSDT_15m_data.csv'
    if not os.path.exists(filepath):
        # Try root
        filepath = 'BTCUSDT_15m_data.csv'
    
    if not os.path.exists(filepath):
        print("Data file not found!")
        return

    try:
        # Generate chart
        json_str = generate_chart_data(filepath, 'BTCUSDT', '15m', num_candles=100)
        chart_data = json.loads(json_str)
        
        # Inspect traces
        traces = chart_data.get('data', [])
        print(f"Total Traces: {len(traces)}")
        
        vp_trace = None
        for i, trace in enumerate(traces):
            name = trace.get('name', 'Unnamed')
            print(f"Trace {i}: {name} | Type: {trace.get('type')} | Xaxis: {trace.get('xaxis')} | Yaxis: {trace.get('yaxis')}")
            if name == 'Volume Profile':
                vp_trace = trace
        
        if vp_trace:
            print("\nVolume Profile Trace Found!")
            print(f"X (Volume) sample: {vp_trace['x'][:5]}")
            print(f"Y (Price) sample: {vp_trace['y'][:5]}")
            print(f"Orientation: {vp_trace.get('orientation')}")
            print(f"X-Axis ID: {vp_trace.get('xaxis')}")
        else:
            print("\nERROR: Volume Profile Trace NOT FOUND in JSON!")
            
        # Inspect Layout
        layout = chart_data.get('layout', {})
        x99 = layout.get('xaxis99')
        if x99:
            print("\nXaxis99 Settings:")
            print(json.dumps(x99, indent=2))
        else:
            print("\nERROR: Xaxis99 NOT FOUND in Layout!")

    except Exception as e:
        print(f"Exception during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_chart()
