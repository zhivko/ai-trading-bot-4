import pandas as pd
import os
import time
from datetime import datetime
from chart_generator import identify_quad_rotation_alarms
import screenshot_generator
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# Try to get Chat ID from env, otherwise we might need to find it
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Cache to prevent spamming the same alarm
ALARM_CACHE = {}

def get_chat_id_if_missing():
    """
    If CHAT_ID is missing, try to fetch it from updates.
    """
    global TELEGRAM_CHAT_ID
    if TELEGRAM_CHAT_ID:
        return TELEGRAM_CHAT_ID

    print("Attempting to auto-discover Telegram Chat ID...")
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
        resp = requests.get(url).json()
        
        if resp['ok'] and resp['result']:
            # Get the most recent chat ID
            latest = resp['result'][-1]
            if 'message' in latest:
                chat_id = str(latest['message']['chat']['id'])
                print(f"Discovered Chat ID: {chat_id}")
                TELEGRAM_CHAT_ID = chat_id
                return chat_id
    except Exception as e:
        print(f"Failed to discover Chat ID: {e}")
    
    return None

def check_signals(filepath, symbol, timeframe, nn_threshold=30):
    """
    Checks for trading signals in the latest data.
    Returns a list of dicts describing detected alarms.
    """
    if not os.path.exists(filepath):
        return []

    try:
        # Load data (last 1000 candles)
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        # Apply CET+1 Timezone shift for alerts
        df.index = df.index + pd.Timedelta(hours=1)
        
        if df.empty:
            return []
            
        if len(df) > 1000:
            df = df.iloc[-1000:].copy()

        # Run detection
        df = identify_quad_rotation_alarms(df, nn_threshold=nn_threshold)

        # Check latest candle (and maybe previous one to be safe/unsure about closing time)
        # We really care about the LAST COMPLETED candle usually, or the current forming one.
        # Since 'update_data_files' removes the incomplete candle, the last row IS the last completed candle.
        last_candle = df.iloc[-1]
        last_time = df.index[-1]
        
        alarms_found = []

        # Helper to construct alarm object
        def add_alarm(type_name, confidence=None, is_buy=True):
            # Create a unique key for this alarm to avoid duplicates: Symbol_Time_Type
            alarm_key = f"{symbol}_{timeframe}_{last_time}_{type_name}"
            
            # Message
            side = "BUY" if is_buy else "SELL"
            emoji = "🟢" if is_buy else "🔴"
            conf_str = f" ({confidence:.1f}%)" if confidence else ""
            
            msg = f"{emoji} **{symbol} {timeframe}**\nSignal: {type_name} {side}{conf_str}\nTime: {last_time}"
            
            alarms_found.append({
                "key": alarm_key,
                "message": msg,
                "symbol": symbol,
                "timeframe": timeframe
            })

        # Check for Quad Rotation Alarm
        if 'alarm' in last_candle and last_candle['alarm']:
            add_alarm("Quad Rotation", is_buy=True)

        # Check for NN Buy
        if 'nn_buy_alarm' in last_candle and last_candle['nn_buy_alarm']:
            conf = last_candle.get('nn_buy_confidence', 0) * 100
            add_alarm("Neural Network", confidence=conf, is_buy=True)

        # Check for NN Sell
        if 'nn_sell_alarm' in last_candle and last_candle['nn_sell_alarm']:
            conf = last_candle.get('nn_sell_confidence', 0) * 100
            add_alarm("Neural Network", confidence=conf, is_buy=False)

        return alarms_found

    except Exception as e:
        print(f"Error checking signals for {symbol}: {e}")
        return []

def send_telegram_alert(alarm_data, filepath):
    """
    Sends a message with images via Telegram.
    """
    global ALARM_CACHE
    
    key = alarm_data['key']
    
    # Check cache (debounce)
    # If we sent this alarm recently, skip
    if key in ALARM_CACHE:
        return

    chat_id = get_chat_id_if_missing()
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        print(f"Skipping Telegram notification (missing token or chat_id)\nMessage: {alarm_data['message']}")
        return

    print(f"Sending Telegram Alert: {alarm_data['message']}")
    
    try:
        # 1. Generate Images
        chart_img_path = screenshot_generator.generate_chart_image(
            filepath, alarm_data['symbol'], alarm_data['timeframe'], num_candles=80
        )
        vp_img_path = screenshot_generator.generate_mini_vp_image(
             filepath, alarm_data['symbol'], alarm_data['timeframe']
        )
        
        files = {}
        # Send as an album (MediaGroup) if both exist, or individual photos
        # Creating a media group is slightly complex with requests, simpler to send Text -> Image 1 -> Image 2
        
        # Send Text First
        url_msg = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(url_msg, data={"chat_id": chat_id, "text": alarm_data['message'], "parse_mode": "Markdown"})
        print(f"Text Response: {resp.status_code} - {resp.text}")
        
        # Send Chart
        if chart_img_path and os.path.exists(chart_img_path):
            with open(chart_img_path, 'rb') as f:
                url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
                resp = requests.post(url_photo, data={"chat_id": chat_id}, files={"photo": f})
                print(f"Photo Response: {resp.status_code} - {resp.text}")
            # Clean up
            try: os.remove(chart_img_path)
            except: pass

        # Send VP
        if vp_img_path and os.path.exists(vp_img_path):
            with open(vp_img_path, 'rb') as f:
                url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
                requests.post(url_photo, data={"chat_id": chat_id, "caption": "Volume Profile & Structure"}, files={"photo": f})
             # Clean up
            try: os.remove(vp_img_path)
            except: pass

        # Mark as sent
        ALARM_CACHE[key] = True
        
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")
        import traceback
        traceback.print_exc()

def check_and_notify_all():
    """
    Scans all data files and checks for alarms.
    Returns a tuple: (list_of_alarms, timestamp_of_check)
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    # Capture the timestamp of when we performed the check
    check_time = datetime.now()
    
    if not os.path.exists(data_dir):
        return [], check_time

    print(f"[{check_time}] Checking for alarms...")
    
    all_alarms = []

    for filename in os.listdir(data_dir):
        if filename.endswith("_data.csv"):
            parts = filename.split('_')
            if len(parts) >= 3:
                symbol = parts[0]
                timeframe = parts[1]
                
                # Filter to target timeframes if desired, e.g. '15m'
                if timeframe != '15m': 
                    continue

                filepath = os.path.join(data_dir, filename)
                alarms = check_signals(filepath, symbol, timeframe)
                
                for alarm in alarms:
                    send_telegram_alert(alarm, filepath)
                    all_alarms.append(alarm)
    
    return all_alarms, check_time
