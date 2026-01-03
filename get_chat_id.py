import requests
import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TOKEN:
    print("Error: No telegram_token found in .env")
    exit(1)

url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
try:
    print(f"Querying: {url.replace(TOKEN, 'TOKEN')}...")
    resp = requests.get(url).json()
    
    if not resp.get('ok'):
        print(f"Error from API: {resp}")
        exit(1)
        
    results = resp.get('result', [])
    if not results:
        print("No updates found.")
        print("Please send a message (e.g., 'Hello') to your bot in Telegram and run this again.")
    else:
        # Get the last message
        last_update = results[-1]
        if 'message' in last_update:
            chat = last_update['message']['chat']
            chat_id = chat['id']
            username = chat.get('username', 'Unknown')
            first_name = chat.get('first_name', 'Unknown')
            print(f"\nSUCCESS! Found Chat ID: {chat_id}")
            print(f"User: {first_name} (@{username})")
            print(f"\nTo save this, add this line to your .env file:")
            print(f"telegram_chat_id={chat_id}")
        else:
            print("Found updates but no message with chat ID.")
            print(results)

except Exception as e:
    print(f"Request failed: {e}")
