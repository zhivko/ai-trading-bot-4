import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chart_generator import identify_quad_rotation_alarms

import importlib
from concurrent.futures import ThreadPoolExecutor

# Strategy Modules
import backtest_strategy_1_poc_target as strategy_1
import backtest_strategy_2_vah_exit as strategy_2
import backtest_strategy_3_optimized as strategy_3
import backtest_strategy_4_optimized as strategy_4
import backtest_strategy_5_trailing_stop as strategy_5
import backtest_strategy_6_volume_divergence as strategy_6

def generate_comparison_chart(final_df, output_path='strategy_comparison_chart.png'):
    """Generate high-quality performance chart with legends and dual axes"""
    import matplotlib.pyplot as plt
    try:
        fig, ax1 = plt.subplots(figsize=(12, 7))
        plt.style.use('dark_background')
        fig.patch.set_facecolor('#0b0e14')
        ax1.set_facecolor('#0b0e14')
        
        # Primary axis: Total Return (%)
        bars = ax1.bar(final_df['strategy_name'], final_df['total_return_pct'], 
                       color='#00FFFF', alpha=0.7, edgecolor='white', label='Total Return (%)')
        
        ax1.set_title('High-Yield Strategy Comparison (50% Pos Size)\n(100 Days / 5% Drawdown Target)', 
                      fontsize=16, color='white', pad=30, fontweight='bold')
        ax1.set_ylabel('Total Return (%)', fontsize=12, color='#00FFFF', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#00FFFF')
        plt.xticks(rotation=25, ha='right', color='white', fontsize=11)
        
        # Secondary axis: Win Rate (%)
        ax2 = ax1.twinx()
        ax2.step(final_df['strategy_name'], final_df['win_rate'], where='mid', 
                 color='#39ff14', linewidth=3, marker='o', label='Win Rate (%)', alpha=0.9)
        ax2.set_ylabel('Win Rate (%)', fontsize=12, color='#39ff14', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#39ff14')
        ax2.set_ylim(0, 100)
        
        ax1.grid(axis='y', linestyle='--', alpha=0.2)
        
        for ax in [ax1, ax2]:
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white') 
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')

        # Add value labels for Return
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + (0.5 if yval > 0 else -1.5), 
                    f"{yval:.1f}%", va='bottom' if yval > 0 else 'top', ha='center', 
                    color='white', fontsize=10, fontweight='bold', 
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
            
        # Add values for Win Rate
        for i, val in enumerate(final_df['win_rate']):
            ax2.text(i, val + 2, f"{val:.1f}%", color='#39ff14', ha='center', fontsize=9, fontweight='bold')

        # Add Combined Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, 
                   facecolor='black', edgecolor='white', fontsize=10)
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison chart saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate comparison chart: {e}")

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

def run_backtest(strategy_module, df, entry_indices, initial_capital=10000, is_sell=False):
    """Generic backtesting engine with Shorting support"""
    trades = []
    balance = initial_capital
    
    direction_str = "SHORT" if is_sell else "LONG"
    print(f"  Executing {len(entry_indices)} {direction_str} trades...")
    
    for i, entry_idx in enumerate(entry_indices):
        # Pass is_sell to the strategy module
        trade_result = strategy_module.execute_trade(df, entry_idx, balance, is_sell=is_sell)
        
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

def run_strategy_evaluation(name, module, df_processed, buy_indices, sell_indices, initial_cap=10000):
    """Worker function to run bidirectional backtest for a single strategy"""
    print(f"Starting Thread for: {name}")
    
    # Run Longs
    long_trades, _ = run_backtest(module, df_processed, buy_indices, initial_capital=initial_cap, is_sell=False)
    
    # Run Shorts
    short_trades, _ = run_backtest(module, df_processed, sell_indices, initial_capital=initial_cap, is_sell=True)
    
    # Combine all trades for this strategy
    all_trades = long_trades + short_trades
    # Sort by entry time for accurate drawdown/sharpe
    all_trades.sort(key=lambda x: x['entry_time'])
    
    # Simulate sequential balance
    current_balance = initial_cap
    for t in all_trades:
        t['balance_after'] = current_balance + t['pnl']
        current_balance = t['balance_after']
        
    metrics = calculate_metrics(all_trades, initial_cap, current_balance)
    metrics['strategy_name'] = name
    
    print(f"Finished: {name} | Return: {metrics['total_return_pct']:.2f}%")
    return metrics

def compare_strategies():
    """Run all strategies and create comparison table"""
    print("Loading data...")
    try:
        df = pd.read_csv('BTCUSDT_15m_data.csv', index_col='timestamp', parse_dates=True)
        print(f"Loaded {len(df)} candles")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    from chart_generator import identify_nn_patterns, get_nn_model
    
    # Load Unified Model
    model = get_nn_model()
    if model is None:
        print("CRITICAL: Failed to load NN Unified Model. Backtest aborted.")
        return

    # Process dataset (last 100 days)
    df_slice = df.tail(10000).copy()
    print(f"Processing {len(df_slice)} candles for Unified Model Inference...")
    
    # Use raw NN signals (they already incorporate pattern detection)
    df_processed = identify_nn_patterns(df_slice, nn_threshold=30)
    
    # Extract signals
    buy_indices = np.where(df_processed['nn_buy_alarm'])[0]
    sell_indices = np.where(df_processed['nn_sell_alarm'])[0]
    
    print(f"\n{'='*80}")
    print(f"Signal Type: Raw NN (High-Confidence Patterns)")
    print(f"Found {len(buy_indices)} BUY Signals and {len(sell_indices)} SELL Signals")
    print(f"{'='*80}\n")

    
    # Base strategies to compare
    base_strategies = [
        ('1. POC Target', strategy_1),
        ('2. VAH Exit', strategy_2),
        ('3. LVN Accel', strategy_3),
        ('4. Multi-Tier', strategy_4),
        ('5. Trail Stop', strategy_5),
        ('6. Vol Diverg', strategy_6)
    ]
    
    results = []
    initial_cap = 10000
    
    # Use ThreadPoolExecutor for parallel strategy evaluation
    print(f"\nStarting Threaded Backtest for {len(base_strategies)} strategies...")
    with ThreadPoolExecutor(max_workers=len(base_strategies)) as executor:
        # Submit all strategy evaluations to the pool
        future_to_strategy = {
            executor.submit(run_strategy_evaluation, name, module, df_processed, buy_indices, sell_indices, initial_cap): name 
            for name, module in base_strategies
        }
        
        for future in future_to_strategy:
            try:
                metrics = future.result()
                results.append(metrics)
            except Exception as exc:
                name = future_to_strategy[future]
                print(f"Strategy {name} generated an exception: {exc}")

    # Create DataFrame
    comparison_df = pd.DataFrame(results)
    cols = ['strategy_name', 'total_return_pct', 'win_rate', 'total_pnl', 'max_drawdown_pct', 'sharpe_ratio', 'total_trades']
    final_df = comparison_df[cols].sort_values('total_return_pct', ascending=False)
    
    print("\n" + "="*110)
    print("BIDIRECTIONAL STRATEGY COMPARISON (Combined Longs + Shorts)")
    print("="*110)
    print(final_df.to_string(index=False, float_format=lambda x: "{:.2f}".format(x)))
    
    # Save to CSV
    final_df.to_csv('strategy_comparison_results.csv', index=False)
    print("\nResults saved to strategy_comparison_results.csv")
    
    # Generate Comparison Chart
    generate_comparison_chart(final_df)
    
    return final_df

if __name__ == '__main__':
    compare_strategies()
