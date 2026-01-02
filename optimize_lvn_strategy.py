import pandas as pd
import numpy as np
from chart_generator import identify_quad_rotation_alarms
import backtest_strategy_3_lvn_acceleration as strategy_lvn

def run_optimization():
    print("Loading data...")
    try:
        df = pd.read_csv('BTCUSDT_15m_data.csv', index_col='timestamp', parse_dates=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Identifying Hybrid Signals...")
    df_slice = df.tail(50000).copy() # Last ~1.5 years
    df_processed = identify_quad_rotation_alarms(df_slice)
    entry_mask = df_processed['hybrid_alarm'] == True
    entry_indices = np.where(entry_mask)[0]
    
    print(f"Found {len(entry_indices)} High-Quality Entry Signals")
    
    # Scenarios to simulate
    scenarios = [
        {'name': 'Conservative (5% Size, 1x)', 'size': 0.05, 'lev': 1},
        {'name': 'Aggressive Spot (100% Size, 1x)', 'size': 1.0, 'lev': 1},
        {'name': 'Low Leverage (100% Size, 3x)', 'size': 1.0, 'lev': 3},
        {'name': 'High Leverage (100% Size, 5x)', 'size': 1.0, 'lev': 5},
        {'name': 'Degen Mode (100% Size, 10x)', 'size': 1.0, 'lev': 10},
    ]
    
    results = []
    
    for scen in scenarios:
        initial_capital = 10000
        balance = initial_capital
        effective_size_pct = scen['size'] * scen['lev']
        trades = []
        
        # Run backtest loop
        for entry_idx in entry_indices:
            # Execute with effective position size
            # Note: strategy module takes size_pct relative to current_balance
            trade = strategy_lvn.execute_trade(df_processed, entry_idx, balance, position_size_pct=effective_size_pct)
            
            if trade['reason'] != 'VP_ERROR':
                trades.append(trade)
                balance = trade['balance_after']
                
        # Metrics
        total_ret = (balance / initial_capital - 1) * 100
        max_dd = 0
        peak = initial_capital
        if trades:
            curves = [initial_capital]
            for t in trades: curves.append(t['balance_after'])
            curves = np.array(curves)
            peak = np.maximum.accumulate(curves)
            dds = (curves - peak) / peak
            max_dd = dds.min() * 100
            
        results.append({
            'Scenario': scen['name'],
            'Return (%)': total_ret,
            'Ending Balance ($)': balance,
            'Max Drawdown (%)': max_dd,
            'Win Rate (%)': len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100 if trades else 0
        })
    
    results_df = pd.DataFrame(results)
    print("\n" + "="*100)
    print("OPTIMIZATION RESULTS: STRATEGY 3 (LVN ACCELERATION)")
    print("="*100)
    print(results_df.to_string(index=False, float_format=lambda x: "{:.2f}".format(x)))

if __name__ == "__main__":
    run_optimization()
