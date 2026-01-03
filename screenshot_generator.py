import os
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from chart_generator import get_chart_figure, identify_nn_patterns
import backtest_strategy_3_lvn_acceleration as strat3

# Kaleido setup is handled automatically by Plotly if installed

def generate_chart_image(filepath, symbol, timeframe, num_candles=100):
    """
    Generates a PNG image of the main chart.
    Returns the path to the temporary image file.
    """
    try:
        # Get the figure object from our existing logic
        # Using a smaller number of candles for the screenshot to make it readable on mobile
        fig, _, _ = get_chart_figure(filepath, symbol, timeframe, num_candles=num_candles)
        
        # Adjust layout for static image
        fig.update_layout(
            width=1000,
            height=800,
            template="plotly_dark",
            title=dict(
                text=f"{symbol} {timeframe} - AI Trading Signal",
                x=0.5,
                font=dict(size=24)
            ),
            # Remove range slider for cleaner image
            xaxis=dict(rangeslider=dict(visible=False))
        )
        
        # Save to temp file
        import tempfile
        tmp_dir = tempfile.gettempdir()
        output_path = os.path.join(tmp_dir, f"chart_{symbol}_{timeframe}.png")
        
        # Write image using Kaleido with explicit Scope to prevent hangs
        print(f"DEBUG: Writing chart image to {output_path}...")
        
        from kaleido.scopes.plotly import PlotlyScope
        scope = PlotlyScope()
        with open(output_path, "wb") as f:
            f.write(scope.transform(fig, format="png"))
            
        print("DEBUG: Chart image written.")
        return output_path
        
    except Exception as e:
        print(f"Error generating chart image: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_mini_vp_image(filepath, symbol, timeframe):
    """
    Generates a PNG image of the 'Mini Volume Profile' (sidebar view).
    """
    try:
        # Load and process data to get the VP setup
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        # Process recent data for NN/Stoch to ensure strat3 gets valid context
        buffer = 500
        if len(df) > buffer:
            subset_df = df.iloc[-buffer:].copy()
        else:
            subset_df = df.copy()
            
        # Identify Patterns (Prerequisite for Strat 3 info if any)
        from chart_generator import identify_nn_patterns
        subset_df = identify_nn_patterns(subset_df)
        
        # Get Strategy 3 setup data (which includes VP)
        # This will calculate VP on the fly for the latest candle
        setup_data = strat3.get_trade_setup(subset_df, len(subset_df)-1)
        
        if not setup_data or 'vp_data' not in setup_data or not setup_data['vp_data']:
            print("No VP data found in setup")
            return None
            
        vp_data = setup_data['vp_data']
        
        # Create Plotly Figure for MP
        fig = go.Figure()
        
        # 1. Filled Area (Volume Profile)
        fig.add_trace(go.Scatter(
            y=vp_data['prices'],
            x=vp_data['volumes'],
            mode='lines',
            fill='tozerox',
            line=dict(color='rgba(0, 242, 254, 0.8)', width=1),
            fillcolor='rgba(0, 242, 254, 0.3)',
            name='Volume'
        ))
        
        # 2. PDF Curve (if available)
        if 'volumes_smooth' in vp_data:
             fig.add_trace(go.Scatter(
                y=vp_data['prices'],
                x=vp_data['volumes_smooth'],
                mode='lines',
                line=dict(color='rgba(255, 165, 0, 0.8)', width=2),
                name='Gaussian Fit'
            ))

        # Add Entry Price Line
        if 'entry_price' in setup_data:
             fig.add_hline(y=setup_data['entry_price'], line_dash="dash", line_color="yellow", annotation_text="Entry")
             
        # Add Stop Loss
        if 'stop_loss' in setup_data:
             fig.add_hline(y=setup_data['stop_loss'], line_dash="dash", line_color="red", annotation_text="SL")

        # Add Targets
        if 'take_profit_candidates' in setup_data:
             for tp in setup_data['take_profit_candidates']:
                 fig.add_hline(y=tp, line_dash="dash", line_color="#39ff14", annotation_text="TP")

        # Layout
        fig.update_layout(
            width=400,
            height=600,
            template="plotly_dark",
            title=dict(text="Market Structure (VP)", x=0.5),
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=True, showticklabels=True),
            paper_bgcolor='rgba(15, 12, 41, 1)', # Match sidebar dark blue/purple
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        # Save
        import tempfile
        tmp_dir = tempfile.gettempdir()
        output_path = os.path.join(tmp_dir, f"vp_{symbol}_{timeframe}.png")
        
        from kaleido.scopes.plotly import PlotlyScope
        scope = PlotlyScope()
        with open(output_path, "wb") as f:
            f.write(scope.transform(fig, format="png"))
            
        return output_path

    except Exception as e:
        print(f"Error generating VP image: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test
    import sys
    # Assume data exists
    test_path = "c:\\git\\ai-trading-bot-4\\data\\BTCUSDT_15m_data.csv"
    if os.path.exists(test_path):
        print("Generating Main Chart...")
        p1 = generate_chart_image(test_path, "BTCUSDT", "15m", num_candles=100)
        print(f"Chart saved to: {p1}")
        
        print("Generating VP Chart...")
        p2 = generate_mini_vp_image(test_path, "BTCUSDT", "15m")
        print(f"VP saved to: {p2}")
