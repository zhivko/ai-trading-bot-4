import pandas as pd
import matplotlib.pyplot as plt
import os

def create_comparison_chart():
    """Create bar charts comparing strategies from CSV results"""
    
    csv_path = 'strategy_comparison_results.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Sort by Total Return for consistency
    df = df.sort_values('total_return_pct', ascending=True)
    
    # Setup plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Volume Profile Exit Strategy Comparison', fontsize=16, weight='bold')
    
    # Metrics to map
    metrics = [
        ('total_return_pct', 'Total Return (%)', 'green'),
        ('win_rate', 'Win Rate (%)', 'blue'),
        ('total_pnl', 'Total PnL ($)', 'gold'),
        ('max_drawdown_pct', 'Max Drawdown (%)', 'red'),
        ('sharpe_ratio', 'Sharpe Ratio', 'purple'),
        ('avg_hold_time_hours', 'Avg Hold Time (Hours)', 'gray')
    ]
    
    for idx, (col, title, color) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        # Plot bars
        bars = ax.barh(df['strategy_name'], df[col], color=color, alpha=0.7)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width if width >= 0 else width - 5  # Adjust label position
            align = 'left' if width >= 0 else 'right'
            
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', 
                    ha=align, va='center', fontweight='bold')
            
        ax.set_title(title)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    save_path = 'strategy_comparison_chart.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison chart saved: {save_path}")

if __name__ == "__main__":
    create_comparison_chart()
