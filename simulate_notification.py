import alarm_service
import os
import time
from datetime import datetime

# Fake data for simulation
symbol = "BTCUSDT"
timeframe = "15m"
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Fake alarm data
alarm_key = f"SIM_{int(time.time())}"
alarm_data = {
    "key": alarm_key,
    "message": f"🧪 **SIMULATION TEST**\nSymbol: {symbol}\nTime: {current_time}\nID: `{alarm_key}`\n\nThis is a test of the notification system with images.",
    "symbol": symbol,
    "timeframe": timeframe
}

# Use a real file path so screenshots work
filepath = os.path.join(os.getcwd(), "data", f"{symbol}_{timeframe}_data.csv")

print("Simulating Telegram Notification...")
try:
    alarm_service.send_telegram_alert(alarm_data, filepath)
    print("Simulation sent!")
except Exception as e:
    print(f"Simulation failed: {e}")
