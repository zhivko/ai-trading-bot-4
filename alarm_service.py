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
ALARM_DB_PATH = os.path.join("data", "alarms_db.json")

def save_alarm_to_disk(alarm_data):
    """Save alarm to a simple JSON file for the telegram bot to read"""
    import json
    try:
        db = {}
        if os.path.exists(ALARM_DB_PATH):
            with open(ALARM_DB_PATH, 'r') as f:
                db = json.load(f)
        
        # Add or update
        db[alarm_data['key']] = {
            "symbol": alarm_data['symbol'],
            "timeframe": alarm_data['timeframe'],
            "is_buy": alarm_data.get('is_buy', True),
            "stop_loss": alarm_data.get('stop_loss'),
            "take_profit": alarm_data.get('take_profit'),
            "time": datetime.now().isoformat()
        }
        
        # Keep only last 100
        if len(db) > 100:
            # Sort by time and keep newest
            sorted_keys = sorted(db.keys(), key=lambda k: db[k].get('time', ''), reverse=True)
            db = {k: db[k] for k in sorted_keys[:100]}

        with open(ALARM_DB_PATH, 'w') as f:
            json.dump(db, f, indent=2)
    except Exception as e:
        print(f"Error saving alarm to disk: {e}")

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

import backtest_strategy_3_lvn_acceleration as strat3

def check_signals_for_file(filepath, symbol, timeframe, nn_threshold=30):
    """
    Checks the given file for trading signals.
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
        last_candle_idx = len(df) - 1
        last_candle = df.iloc[last_candle_idx]
        last_time = df.index[-1]
        
        alarms_found = []

        # Helper to construct alarm object
        def add_alarm(type_name, confidence=None, is_buy=True):
            # Create a unique key for this alarm to avoid duplicates: Symbol_Time_Type
            # FIXED: Replace spaces with hyphens in both time and type name
            alarm_key = f"{symbol}_{timeframe}_{str(last_time).replace(' ', '-')}_{type_name.replace(' ', '-')}"
            
            # --- EXTRACT PRECISE TP/SL FROM STRATEGY ---
            # We temporarily ensure the side is set in the row for get_trade_setup
            df.loc[df.index[last_candle_idx], 'nn_sell_alarm'] = not is_buy
            df.loc[df.index[last_candle_idx], 'nn_buy_alarm'] = is_buy
            
            # Pass the full df to get_trade_setup so it has history for ATR/VP
            setup = strat3.get_trade_setup(df, last_candle_idx)
            stop_loss = setup.get('stop_loss')
            tp_list = setup.get('take_profit_candidates', [])
            take_profit = tp_list[0] if tp_list else None
            # -------------------------------------------

            # Message
            side = "BUY" if is_buy else "SELL"
            emoji = "🟢" if is_buy else "🔴"
            conf_str = f" ({confidence:.1f}%)" if confidence else ""
            
            msg = f"{emoji} **{symbol} {timeframe}**\nSignal: {type_name} {side}{conf_str}\nTime: {last_time}\nID: `{alarm_key}`"
            
            alarms_found.append({
                "key": alarm_key,
                "message": msg,
                "symbol": symbol,
                "timeframe": timeframe,
                "is_buy": is_buy,
                "stop_loss": stop_loss,
                "take_profit": take_profit
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
             filepath, alarm_data['symbol'], alarm_data['timeframe'], 
             is_sell=not alarm_data.get('is_buy', True)
        )
        
        media = []
        files = {}
        
        # Add Chart to Media Group
        if chart_img_path and os.path.exists(chart_img_path):
            media.append({
                "type": "photo",
                "media": "attach://chart",
                "caption": alarm_data['message'],
                "parse_mode": "Markdown"
            })
            files["chart"] = open(chart_img_path, 'rb')
        
        # Add VP to Media Group
        if vp_img_path and os.path.exists(vp_img_path):
            media.append({
                "type": "photo",
                "media": "attach://vp"
            })
            files["vp"] = open(vp_img_path, 'rb')
            
        # Persist alarm for the telegram bot to lookup
        save_alarm_to_disk(alarm_data)

        if media:
            # Send as Media Group (Album)
            url_media = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMediaGroup"
            import json
            resp = requests.post(url_media, data={"chat_id": chat_id, "media": json.dumps(media)}, files=files)
            print(f"Telegram MediaGroup Response: {resp.status_code} - {resp.text}")
        else:
            # Fallback to Text only if images failed
            url_msg = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            resp = requests.post(url_msg, data={"chat_id": chat_id, "text": alarm_data['message'], "parse_mode": "Markdown"})
            print(f"Telegram Text Fallback Response: {resp.status_code} - {resp.text}")

        # Clean up files and close handles
        for f_handle in files.values():
            f_handle.close()
            
        if chart_img_path and os.path.exists(chart_img_path):
            try: os.remove(chart_img_path)
            except: pass
        if vp_img_path and os.path.exists(vp_img_path):
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
                alarms = check_signals_for_file(filepath, symbol, timeframe)
                
                for alarm in alarms:
                    send_telegram_alert(alarm, filepath)
                    all_alarms.append(alarm)
    
    return all_alarms, check_time
