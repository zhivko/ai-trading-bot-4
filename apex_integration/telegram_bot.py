
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

def get_alarm_info(alarm_id):
    """Lookup alarm details in the local JSON database."""
    import json
    try:
        db_path = os.path.join("..", "data", "alarms_db.json")
        if os.path.exists(db_path):
            with open(db_path, 'r') as f:
                db = json.load(f)
                return db.get(alarm_id)
    except Exception as e:
        logger.error(f"Error reading alarms_db.json: {e}")
    return None

async def buy_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_auth(update):
        return

    # Default values
    alarm_id = f"buy_{int(time.time())}"
    symbol = DEFAULT_SYMBOL
    stop_loss = None
    take_profit = None

    if context.args:
        # Intelligently handle ID potentially containing spaces
        full_potential_id = " ".join(context.args)
        
        # Check joined ID
        alarm_info = get_alarm_info(full_potential_id)
        if alarm_info:
            alarm_id = full_potential_id
            symbol = alarm_info.get('symbol', DEFAULT_SYMBOL)
            # Use precise SL/TP from strategy if available
            stop_loss = alarm_info.get('stop_loss')
            take_profit = alarm_info.get('take_profit')
        else:
            # Check if joining with hyphens 
            hyphenated_id = full_potential_id.replace(' ', '-')
            alarm_info = get_alarm_info(hyphenated_id)
            if alarm_info:
                alarm_id = hyphenated_id
                symbol = alarm_info.get('symbol', DEFAULT_SYMBOL)
                stop_loss = alarm_info.get('stop_loss')
                take_profit = alarm_info.get('take_profit')
            else:
                alarm_id = context.args[0]
                if len(context.args) > 1:
                    stop_loss = context.args[1]
    
    logger.info(f"Received /buy command. Alarm ID: {alarm_id}, Symbol: {symbol}, SL: {stop_loss}, TP: {take_profit}")
    
    try:
        url = f"{TRADING_SERVICE_URL}/buy/{symbol}"
        
        params = {}
        if stop_loss: params['stop_loss'] = stop_loss
        if take_profit: params['take_profit'] = take_profit

        response = requests.post(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            message = f"✅ BUY Order Placed!\n\nSymbol: {symbol}\nAlarm ID: {alarm_id}\n\nDetails:\n{json.dumps(data, indent=2)}"
        else:
            message = f"❌ Buy Failed (Status: {response.status_code})\n\nResponse: {response.text}"
            
    except Exception as e:
        logger.error(f"Error calling trading service: {e}")
        message = f"❌ System Error: {str(e)}"

    await context.bot.send_message(chat_id=update.effective_chat.id, text=message)

async def sell_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_auth(update):
        return

    # Default values
    alarm_id = f"sell_{int(time.time())}"
    symbol = DEFAULT_SYMBOL
    stop_loss = None
    take_profit = None

    if context.args:
        # Intelligently handle ID potentially containing spaces
        full_potential_id = " ".join(context.args)
        
        # Check joined ID
        alarm_info = get_alarm_info(full_potential_id)
        if alarm_info:
            alarm_id = full_potential_id
            symbol = alarm_info.get('symbol', DEFAULT_SYMBOL)
            # Use precise SL/TP from strategy
            stop_loss = alarm_info.get('stop_loss')
            take_profit = alarm_info.get('take_profit')
        else:
            # Check if joining with hyphens 
            hyphenated_id = full_potential_id.replace(' ', '-')
            alarm_info = get_alarm_info(hyphenated_id)
            if alarm_info:
                alarm_id = hyphenated_id
                symbol = alarm_info.get('symbol', DEFAULT_SYMBOL)
                stop_loss = alarm_info.get('stop_loss')
                take_profit = alarm_info.get('take_profit')
            else:
                alarm_id = context.args[0]
                if len(context.args) > 1:
                    stop_loss = context.args[1]
    
    logger.info(f"Received /sell command. Alarm ID: {alarm_id}, Symbol: {symbol}, SL: {stop_loss}, TP: {take_profit}")
    
    try:
        url = f"{TRADING_SERVICE_URL}/sell/{symbol}"
        
        params = {}
        if stop_loss: params['stop_loss'] = stop_loss
        if take_profit: params['take_profit'] = take_profit

        response = requests.post(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            message = f"✅ SELL Order Placed!\n\nSymbol: {symbol}\nAlarm ID: {alarm_id}\n\nDetails:\n{json.dumps(data, indent=2)}"
        else:
            message = f"❌ Sell Failed (Status: {response.status_code})\n\nResponse: {response.text}"
            
    except Exception as e:
        logger.error(f"Error calling trading service: {e}")
        message = f"❌ System Error: {str(e)}"

    await context.bot.send_message(chat_id=update.effective_chat.id, text=message)

async def trades_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show currently monitored trades and their trailing stop status from the service."""
    if not await check_auth(update):
        return

    try:
        url = f"{TRADING_SERVICE_URL}/tracking-status"
        response = requests.get(url, timeout=5)
        
        if response.status_code != 200:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"❌ Failed to fetch tracking: {response.status_code}")
            return

        data = response.json()
        tracking = data.get('tracking', {})
        
        if not tracking:
             await context.bot.send_message(chat_id=update.effective_chat.id, text="No active trades are currently being tracked for trailing stops.")
             return

        msg = "📊 **Autonomous Trailing Monitor**\n\n"
        for symbol, trade in tracking.items():
            side = trade.get('side', 'UNKNOWN')
            sl = trade.get('last_sl', 0)
            
            if side == 'LONG':
                best = trade.get('highest_price', 0)
                best_label = "Highest"
            else:
                best = trade.get('lowest_price', 0)
                best_label = "Lowest"
                
            msg += f"🔹 **{symbol}** ({side})\n"
            msg += f"• {best_label}: {best:.2f}\n"
            msg += f"• Active SL: {sl:.2f}\n\n"

        await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

    except Exception as e:
        logger.error(f"Error calling tracking-status: {e}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"❌ System Error: {str(e)}")

async def positions_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current exchange positions and P&L."""
    if not await check_auth(update):
        return

    try:
        url = f"{TRADING_SERVICE_URL}/positions"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            positions = data.get('positions', [])
            
            if not positions:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="No open positions found on the exchange.")
                return

            msg = "📈 **Active Positions**\n\n"
            for pos in positions:
                symbol = pos.get('symbol')
                side = pos.get('side')
                size = pos.get('size')
                entry = pos.get('entryPrice')
                current = pos.get('current_price')
                pnl_usd = pos.get('unrealized_pnl', 0)
                pnl_pct = pos.get('pnl_percentage', 0)
                
                emoji = "🟢" if pnl_usd >= 0 else "🔴"
                msg += f"{emoji} **{symbol}** ({side})\n"
                msg += f"• Size: {size}\n"
                msg += f"• Entry: {entry}\n"
                msg += f"• Mark: {current}\n"
                msg += f"• P&L: {pnl_pct:.2f}% (${pnl_usd:.2f})\n\n"
                
            await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"❌ Failed to fetch positions: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Error calling positions: {e}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"❌ System Error: {str(e)}")


if __name__ == '__main__':
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not found in environment variables.")
        exit(1)

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    start_handler = CommandHandler('start', start)
    buy_cmd_handler = CommandHandler('buy', buy_handler)
    sell_cmd_handler = CommandHandler('sell', sell_handler)
    trades_cmd_handler = CommandHandler('trades', trades_handler)
    positions_cmd_handler = CommandHandler('positions', positions_handler)
    
    application.add_handler(start_handler)
    application.add_handler(buy_cmd_handler)
    application.add_handler(sell_cmd_handler)
    application.add_handler(trades_cmd_handler)
    application.add_handler(positions_cmd_handler)
    
    # We NO LONGER need JobQueue here because the TRADING SERVICE handles monitoring autonomously.
    
    logger.info("Bot is starting polling...")
    application.run_polling()
