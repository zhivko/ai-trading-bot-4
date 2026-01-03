
import asyncio
from datetime import datetime
import json

# Mock Data
mock_tracking = {
    'BTC-USDT': {
        'side': 'SHORT',
        'last_sl': 93000.0,
        'entry_price': 90620.3,
        'current_price': 90450.0,
        'size': 0.005,
        'pnl_value': 0.85,  # (90620.3 - 90450) * 0.005 approx
        'pnl_percent': 0.18,
        'lowest_price': 90400.0,
        'open_time': '12:30:00 04-01'
    },
    'ETH-USDT': {
        'side': 'LONG',
        'last_sl': 2200.0,
        'entry_price': 2250.0,
        'current_price': 2210.0,
        'size': 0.1,
        'pnl_value': -4.0,
        'pnl_percent': -1.78,
        'highest_price': 2260.0,
        'open_time': '10:00:00 04-01'
    }
}

async def simulate_trades_handler():
    # Simulate the logic from telegram_bot.py
    msg = "📊 **Autonomous Trailing Monitor**\n\n"
    
    for symbol, trade in mock_tracking.items():
        side = trade.get('side', 'UNKNOWN')
        sl = trade.get('last_sl', 0)
        
        entry = trade.get('entry_price', 0)
        current = trade.get('current_price', 0)
        pnl_val = trade.get('pnl_value', 0)
        pnl_pct = trade.get('pnl_percent', 0)
        open_time = trade.get('open_time', 'Unknown')
        
        pnl_emoji = "🟢" if pnl_val >= 0 else "🔴"
        
        if side == 'LONG':
            best = trade.get('highest_price', 0)
            best_label = "Highest"
        else:
            best = trade.get('lowest_price', 0)
            best_label = "Lowest"
            
        msg += f"{pnl_emoji} **{symbol}** ({side})\n"
        msg += f"• ⏱ Opened: {open_time}\n"
        msg += f"• 🚪 Entry: {entry:.2f}\n"
        msg += f"• 📍 Current: {current:.2f}\n"
        msg += f"• 💰 PNL: **{pnl_pct:.2f}%** (${pnl_val:.2f})\n"
        msg += f"• {best_label}: {best:.2f}\n"
        msg += f"• 🛑 Active SL: {sl:.2f}\n\n"

    print("-" * 30)
    print("PREVIEW OF TELEGRAM MESSAGE:")
    print("-" * 30)
    print(msg)
    print("-" * 30)

if __name__ == "__main__":
    asyncio.run(simulate_trades_handler())
