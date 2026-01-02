import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chart_generator import identify_hybrid_signals
import importlib

# Strategy Modules
import backtest_strategy_1_poc_target as strategy_1
import backtest_strategy_2_vah_exit as strategy_2
import backtest_strategy_3_lvn_acceleration as strategy_3
import backtest_strategy_4_multi_tier as strategy_4
import backtest_strategy_5_trailing_stop as strategy_5
import backtest_strategy_6_volume_divergence as strategy_6

def calculate_max_drawdown(trades, initial_capital):
    if not trades:
        return 0.0
    
    equity_curve = [initial_capital]
    for trade in trades:
        equity_curve.append(trade['balance_after'])
    
    equity_curve = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return drawdown.min() * 100

def calculate_sharpe_ratio(trades, risk_free_rate=0.0):
    if len(trades) < 2:
        return 0.0
    
    pnls = [t['pnl_pct'] for t in trades]
    returns = np.array(pnls) / 100
    
    if np.std(returns) == 0:
        return 0.0
        
    # Annualized Sharpe (assuming 1 trade every day roughly, or just simple ratio)
    # Simple Sharpe = Mean / Std
    sharpe = np.mean(returns) / np.std(returns)
    return sharpe

def run_backtest(strategy_module, df, entry_indices, initial_capital=10000):
    """Generic backtesting engine"""
    trades = []
    balance = initial_capital
    
    print(f"  Executing {len(entry_indices)} trades...")
    
    for i, entry_idx in enumerate(entry_indices):
        # Pass COPY of balance to avoid accumulating if we want independent trade stats?
        # Actually we want sequential simulation to track compounding/drawdown
        
        trade_result = strategy_module.execute_trade(df, entry_idx, balance)
        
        # Only record valid trades
        if trade_result['reason'] != 'VP_ERROR':
            trades.append(trade_result)
            balance = trade_result['balance_after']
            
    return trades, balance

def calculate_metrics(trades, initial_capital, final_balance):
    """Calculate performance metrics"""
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_return_pct': 0,
            'avg_pnl_per_trade': 0,
            'max_drawdown_pct': 0,
            'sharpe_ratio': 0,
            'avg_hold_time_hours': 0,
            'total_pnl': 0
        }
        
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] <= 0]
    
    metrics = {
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(winning_trades) / len(trades) * 100,
        'total_pnl': final_balance - initial_capital,
        'total_return_pct': (final_balance / initial_capital - 1) * 100,
        'avg_pnl_per_trade': (final_balance - initial_capital) / len(trades),
        'max_win': max([t['pnl'] for t in trades]) if trades else 0,
        'max_loss': min([t['pnl'] for t in trades]) if trades else 0,
        'avg_hold_time_hours': sum([t['hold_hours'] for t in trades]) / len(trades),
        'sharpe_ratio': calculate_sharpe_ratio(trades),
        'max_drawdown_pct': calculate_max_drawdown(trades, initial_capital)
    }
    return metrics

def compare_strategies():
    """Run all strategies and create comparison table"""
    print("Loading data...")
    # Load data
    try:
        df = pd.read_csv('BTCUSDT_15m_data.csv', index_col='timestamp', parse_dates=True)
        print(f"Loaded {len(df)} candles")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    # Identify Signals ONLY ONCE
    print("Identifying Hybrid Signals...")
    # We need stochastics and channels calculated
    # Chart generator logic usually does this inside, but let's assume identify_hybrid_signals handles it
    # Note: identify_hybrid_signals expects raw DF and does calculations if needed
    
    # We need to manually prep the DF as identify_hybrid_signals relies on columns existing?
    # Let's check chart_generator.py... 
    # It calls identify_quad_rotation_alarms -> identify_hybrid_signals
    # We should use identify_quad_rotation_alarms to be safe and get full setup
    
    from chart_generator import identify_quad_rotation_alarms
    
    # Process latest 5000 candles for speed, or full dataset?
    # Full dataset is 289k candles... might be slow.
    # Let's take last 50,000 candles (approx 1.5 years)
    df_slice = df.tail(50000).copy()
    
    print(f"Processing {len(df_slice)} candles...")
    df_processed = identify_quad_rotation_alarms(df_slice)
    
    # Filter for entries
    # Hybrid Alarm is our entry signal
    entry_mask = df_processed['hybrid_alarm'] == True
    entry_indices = np.where(entry_mask)[0] # Integer indices
    
    print(f"Found {len(entry_indices)} Hybrid Entry Signals")
    
    strategies = [
        ('1. POC Target', strategy_1),
        ('2. VAH Exit', strategy_2),
        ('3. LVN Accel', strategy_3),
        ('4. Multi-Tier', strategy_4),
        ('5. Trail Stop', strategy_5),
        ('6. Vol Diverg', strategy_6)
    ]
    
    results = []
    
    for name, module in strategies:
        print(f"\n{'='*60}")
        print(f"Running Strategy: {name}")
        print('='*60)
        
        # Run Backtest
        trades, final_balance = run_backtest(module, df_processed, entry_indices)
        metrics = calculate_metrics(trades, 10000, final_balance)
        metrics['strategy_name'] = name
        results.append(metrics)
        
        print(f"Return: {metrics['total_return_pct']:.2f}% | Win Rate: {metrics['win_rate']:.1f}% | PnL: ${metrics['total_pnl']:.2f}")
    
    # Create DataFrame
    comparison_df = pd.DataFrame(results)
    
    # Select key columns
    cols = ['strategy_name', 'total_return_pct', 'win_rate', 'total_pnl', 'max_drawdown_pct', 'sharpe_ratio', 'avg_hold_time_hours', 'total_trades']
    final_df = comparison_df[cols].sort_values('total_return_pct', ascending=False)
    
    print("\n" + "="*100)
    print("STRATEGY COMPARISON RESULTS (Last 50,000 Candles ~1.5 Years)")
    print("="*100)
    print(final_df.to_string(index=False, float_format=lambda x: "{:.2f}".format(x)))
    
    # Save to CSV
    final_df.to_csv('strategy_comparison_results.csv', index=False)
    print("\nResults saved to strategy_comparison_results.csv")
    
    return final_df

if __name__ == '__main__':
    compare_strategies()
