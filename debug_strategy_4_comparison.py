import pandas as pd
import numpy as np
import backtest_strategy_4_multi_tier as v1
import backtest_strategy_4_optimized as v2
from chart_generator import identify_nn_patterns

def run_focused_test():
    print("Loading data...")
    df = pd.read_csv('BTCUSDT_15m_data.csv', index_col='timestamp', parse_dates=True)
    df_slice = df.tail(10000).copy()
    
    print("Generating signals...")
    df_processed = identify_nn_patterns(df_slice, nn_threshold=0.30)
    buy_indices = np.where(df_processed['nn_buy_alarm'])[0]
    sell_indices = np.where(df_processed['nn_sell_alarm'])[0]
    
    versions = [("V1 (Scalp)", v1), ("V2 (Heavier Runner)", v2)]
    
    results = []
    for name, module in versions:
        print(f"\nEvaluating {name}...")
        balance = 10000
        trades = []
        
        # Combine buy/sell for simplicity
        all_indices = []
        for idx in buy_indices: all_indices.append((idx, False))
        for idx in sell_indices: all_indices.append((idx, True))
        all_indices.sort()
        
        for idx, is_sell in all_indices:
            res = module.execute_trade(df_processed, idx, balance, is_sell=is_sell)
            if res['reason'] != 'VP_ERROR':
                trades.append(res)
                balance = res['balance_after']
        
        ret = (balance / 10000 - 1) * 100
        wr = len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100 if trades else 0
        results.append({'Name': name, 'Return': f"{ret:.2f}%", 'WinRate': f"{wr:.1f}%", 'Trades': len(trades)})

    print("\n" + "="*40)
    print("STRATEGY 4: A/B COMPARISON")
    print("="*40)
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    run_focused_test()
