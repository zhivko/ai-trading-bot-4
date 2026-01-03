import pandas as pd
import numpy as np
import backtest_strategy_4_optimized as strat4
import backtest_strategy_5_trailing_stop as strat5
from chart_generator import identify_nn_patterns

def run_focused_standardized_test():
    print("Loading data...")
    df = pd.read_csv('BTCUSDT_15m_data.csv', index_col='timestamp', parse_dates=True)
    df_slice = df.tail(10000).copy()
    
    print("Generating signals...")
    df_processed = identify_nn_patterns(df_slice, nn_threshold=0.30)
    buy_indices = np.where(df_processed['nn_buy_alarm'])[0]
    sell_indices = np.where(df_processed['nn_sell_alarm'])[0]
    
    # Combined indices
    all_indices = []
    for idx in buy_indices: all_indices.append((idx, False))
    for idx in sell_indices: all_indices.append((idx, True))
    all_indices.sort()
    
    results = []
    for name, module in [("4. Multi-Tier", strat4), ("5. Trail Stop", strat5)]:
        print(f"Testing {name} at 10% allocation...")
        balance = 10000
        trades = []
        for idx, is_sell in all_indices:
            res = module.execute_trade(df_processed, idx, balance, is_sell=is_sell)
            if res['reason'] != 'VP_ERROR':
                trades.append(res)
                balance = res['balance_after']
        
        ret = (balance / 10000 - 1) * 100
        wr = len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100 if trades else 0
        results.append((name, ret, wr, len(trades)))
        print(f"  Result: {ret:.2f}% Return, {wr:.1f}% WinRate")

    # Update CSV
    csv_path = 'strategy_comparison_results.csv'
    df_csv = pd.read_csv(csv_path)
    
    for name, ret, wr, trades in results:
        # Update name matching
        df_csv.loc[df_csv['strategy_name'] == name, 'total_return_pct'] = ret
        df_csv.loc[df_csv['strategy_name'] == name, 'win_rate'] = wr
        df_csv.loc[df_csv['strategy_name'] == name, 'total_pnl'] = (ret/100) * 10000
        df_csv.loc[df_csv['strategy_name'] == name, 'total_trades'] = trades

    df_csv.to_csv(csv_path, index=False)
    print(f"Updated {csv_path}")
    
    # Generate high-quality chart using the refactored function
    from backtest_all_strategies import generate_comparison_chart
    generate_comparison_chart(df_csv)

if __name__ == "__main__":
    run_focused_standardized_test()
