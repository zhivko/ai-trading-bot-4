
import logging
import os
import requests
import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

TRADING_SERVICE_URL = "http://localhost:8000"
DEFAULT_SYMBOL = "BTC-USDT"

async def check_auth(update: Update) -> bool:
    """Check if the user is authorized to use the bot."""
    if not TELEGRAM_CHAT_ID:
        logger.warning("TELEGRAM_CHAT_ID not set. allowing all users (NOT SAFe)")
        return True
    
    user_id = str(update.effective_chat.id)
    if user_id != str(TELEGRAM_CHAT_ID):
        logger.warning(f"Unauthorized access attempt from chat ID: {user_id}")
        await update.message.reply_text("⛔ Unauthorized access.")
        return False
    return True


import time

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_auth(update):
        return
    await context.bot.send_message(
        chat_id=update.effective_chat.id, 
        text="🚀 Apex Omni Trading Bot Interface Online\n\nCommands:\n/buy [alarm_id] [stop_loss] - Buy BTC-USDT\n/sell [alarm_id] [stop_loss] - Sell BTC-USDT\n\nIf alarm_id is omitted, one will be generated (e.g. buy_1700000000)."
    )

async def buy_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_auth(update):
        return

    # Default alarm_id to timestamp
    alarm_id = f"buy_{int(time.time())}"
    stop_loss = None

    if context.args:
        # Check if first arg looks like an alarm_id or a stop_loss?
        # User said: /buy {alarm_id} or /sell {alarm_id}
        # But logically someone might skip alarm_id. 
        # For simplicity, if args exist:
        # arg[0] = alarm_id
        # arg[1] = stop_loss
        alarm_id = context.args[0]
        if len(context.args) > 1:
            stop_loss = context.args[1]
    
    logger.info(f"Received /buy command. Alarm ID: {alarm_id}, Stop Loss: {stop_loss}")
    
    try:
        # Construct the URL
        url = f"{TRADING_SERVICE_URL}/buy/{DEFAULT_SYMBOL}"
        
        params = {}
        if stop_loss:
            params['stop_loss'] = stop_loss

        # We can pass params if needed, but for now just the basic call
        # If headers/auth logic is needed for the local service, add here
        response = requests.post(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            message = f"✅ BUY Order Placed!\n\nAlarm ID: {alarm_id}\nStop Loss: {stop_loss}\n\nDetails:\n{data}"
        else:
            message = f"❌ Buy Failed (Status: {response.status_code})\n\nResponse: {response.text}"
            
    except Exception as e:
        logger.error(f"Error calling trading service: {e}")
        message = f"❌ System Error: {str(e)}"

    await context.bot.send_message(chat_id=update.effective_chat.id, text=message)

async def sell_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_auth(update):
        return

    # Default alarm_id to timestamp
    alarm_id = f"sell_{int(time.time())}"
    stop_loss = None

    if context.args:
        alarm_id = context.args[0]
        if len(context.args) > 1:
            stop_loss = context.args[1]
    
    logger.info(f"Received /sell command. Alarm ID: {alarm_id}, Stop Loss: {stop_loss}")
    
    try:
        url = f"{TRADING_SERVICE_URL}/sell/{DEFAULT_SYMBOL}"
        
        params = {}
        if stop_loss:
            params['stop_loss'] = stop_loss

        response = requests.post(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            message = f"✅ SELL Order Placed!\n\nAlarm ID: {alarm_id}\nStop Loss: {stop_loss}\n\nDetails:\n{data}"
        else:
            message = f"❌ Sell Failed (Status: {response.status_code})\n\nResponse: {response.text}"
            
    except Exception as e:
        logger.error(f"Error calling trading service: {e}")
        message = f"❌ System Error: {str(e)}"

    await context.bot.send_message(chat_id=update.effective_chat.id, text=message)


# Global state for tracking active trades
# Format: {symbol: {'alarm_id': str, 'highest_price': float, 'entry_price': float, 'stop_loss': float, 'side': str}}
active_trades = {}

async def monitor_positions(context: ContextTypes.DEFAULT_TYPE):
    """
    Monitor active positions every 5 seconds.
    - Detect new positions (Entry).
    - Update trailing stops (Logic).
    - Alert on changes.
    """
    global active_trades
    
    if not TELEGRAM_CHAT_ID:
        return

    try:
        # Fetch current positions from trading service
        url = f"{TRADING_SERVICE_URL}/positions"
        response = requests.get(url, timeout=5)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch positions: {response.text}")
            return

        data = response.json()
        positions = data.get('positions', [])
        
        current_symbols = set()

        for pos in positions:
            symbol = pos.get('symbol')
            size = float(pos.get('size', 0))
            if size == 0:
                continue
                
            current_symbols.add(symbol)
            side = pos.get('side')
            entry_price = float(pos.get('entryPrice', 0))
            current_price = float(pos.get('current_price', 0)) # Assuming /positions returns this or we need to fetch ticker
            
            # If current_price missing in position object, fallback to entry (for initial) or fetch ticker if critical
            if current_price == 0:
                 current_price = entry_price # Temporary fallback

            # 1. New Position Detection
            if symbol not in active_trades:
                # Try to guess alarm_id or use default
                # In a real sync, we'd map orderID to alarmID, but here we assume latest
                alarm_id = f"detected_{int(time.time())}"
                
                # Check if we have a manually tracked one from /buy matching this symbol?
                # For now, create new tracking entry
                initial_stop_loss = 0.0 # Unknown unless we knew the order
                
                active_trades[symbol] = {
                    'alarm_id': alarm_id,
                    'highest_price': current_price if side == 'LONG' else 0, # For SHORT, we track lowest
                    'lowest_price': current_price if side == 'SHORT' else float('inf'),
                    'entry_price': entry_price,
                    'stop_loss': initial_stop_loss,
                    'side': side,
                    'last_update': time.time()
                }
                
                msg = f"🔔 **Entry Detected**\n\nSymbol: {symbol}\nSide: {side}\nEntry Price: {entry_price}\nSize: {size}"
                await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)


            # 2. Trailing Stop Management
            trade = active_trades[symbol]
            
            # Update high/low for trailing
            new_sl_value = None
            
            if side == 'LONG':
                if current_price > trade['highest_price']:
                    trade['highest_price'] = current_price
                    # Logic: If price moves up by X%, move SL up?
                    # Example: Trailing stop distance 2% (widened to avoid noise)
                    trail_percent = 0.02
                    new_sl = current_price * (1 - trail_percent)
                    
                    if new_sl > trade.get('stop_loss', 0):
                        new_sl_value = new_sl
                            
            elif side == 'SHORT':
                 if current_price < trade['lowest_price']:
                    trade['lowest_price'] = current_price
                    trail_percent = 0.02 
                    new_sl = current_price * (1 + trail_percent)
                    
                    current_sl = trade.get('stop_loss', float('inf'))
                    if new_sl < current_sl:
                        new_sl_value = new_sl

            # Apply Update if needed
            if new_sl_value:
                try:
                    update_url = f"{TRADING_SERVICE_URL}/position/update-sl/{symbol}"
                    resp = requests.post(update_url, params={'stop_loss': new_sl_value}, timeout=5)
                    
                    if resp.status_code == 200:
                        old_sl = trade.get('stop_loss', 0)
                        trade['stop_loss'] = new_sl_value
                        
                        # Notify
                        await context.bot.send_message(
                            chat_id=TELEGRAM_CHAT_ID, 
                            text=f"🔄 **Trailing Stop Updated** ({symbol})\n\nNew SL: {new_sl_value:.2f}\nPrice: {current_price:.2f}"
                        )
                    else:
                        logger.error(f"Failed to update SL for {symbol}: {resp.text}")
                except Exception as e:
                    logger.error(f"Exception updating SL: {e}")

        # 3. Closed Position Detection
        closed_symbols = []
        for sym in active_trades:
            if sym not in current_symbols:
                # Position closed
                closed_symbols.append(sym)
                await context.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=f"🏁 **Position Closed**\n\nSymbol: {sym}"
                )
        
        for sym in closed_symbols:
            del active_trades[sym]

    except Exception as e:
        logger.error(f"Error in monitor loop: {e}")

if __name__ == '__main__':
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not found in environment variables.")
        exit(1)

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    start_handler = CommandHandler('start', start)
    buy_cmd_handler = CommandHandler('buy', buy_handler)
    sell_cmd_handler = CommandHandler('sell', sell_handler)
    
    application.add_handler(start_handler)
    application.add_handler(buy_cmd_handler)
    application.add_handler(sell_cmd_handler)
    
    # Add JobQueue for monitoring
    if application.job_queue:
        application.job_queue.run_repeating(monitor_positions, interval=5, first=5)
        logger.info("Monitoring scheduled every 5 seconds.")
    else:
        logger.warning("JobQueue not available. Monitoring disabled.")
    
    logger.info("Bot is starting polling...")
    application.run_polling()
