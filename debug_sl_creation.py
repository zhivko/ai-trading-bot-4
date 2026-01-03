
import os
import json
import decimal
from dotenv import load_dotenv
# Compatibility Shim
try:
    from sympy.core.numbers import igcdex
except ImportError:
    try:
        from sympy.core.intfunc import igcdex
        import sympy.core.numbers
        sympy.core.numbers.igcdex = igcdex
    except ImportError:
        pass

from apexomni.http_private_v3 import HttpPrivate_v3
from apexomni.http_private_sign import HttpPrivateSign
from apexomni.constants import APEX_OMNI_HTTP_MAIN, NETWORKID_OMNI_MAIN_ARB

# Load env
load_dotenv(override=True)
load_dotenv("apex_integration/.env", override=True)

key = os.getenv('APEXPRO_API_KEY')
secret = os.getenv('APEXPRO_API_SECRET')
passphrase = os.getenv('APEXPRO_API_PASSPHRASE')

# Derive keys
eth_private_key = os.getenv('APEXPRO_ETH_PRIVATE_KEY')
if not eth_private_key.startswith('0x'): eth_private_key = '0x' + eth_private_key

temp_client = HttpPrivate_v3(APEX_OMNI_HTTP_MAIN, network_id=NETWORKID_OMNI_MAIN_ARB, eth_private_key=eth_private_key)
temp_client.configs_v3()
derived_keys = temp_client.derive_zk_key(temp_client.default_address)
zk_seeds = derived_keys['seeds']
l2_key = derived_keys['l2Key']

client = HttpPrivateSign(APEX_OMNI_HTTP_MAIN, network_id=NETWORKID_OMNI_MAIN_ARB,
                         zk_seeds=zk_seeds, zk_l2Key=l2_key,
                         api_key_credentials={'key': key, 'secret': secret, 'passphrase': passphrase})
client.configs_v3()

# Helper Functions
def round_size(size, step):
    step_decimal = decimal.Decimal(str(step))
    size_decimal = decimal.Decimal(str(size))
    rounded_size = (size_decimal // step_decimal) * step_decimal
    return f"{rounded_size:.{abs(step_decimal.as_tuple().exponent)}f}"

def format_decimal_for_apex(value):
    return "{:f}".format(decimal.Decimal(str(value)).normalize())

def get_symbol_config(configs, symbol):
    for s in configs:
        if s.get('symbol') == symbol:
            return s
    return None

# MAIN TEST
symbol = "BTC-USDT"
print(f"DEBUG: Attempting to place SL for {symbol}...")

# 1. Get Position
account = client.get_account_v3()
print("DEBUG: All Positions:", json.dumps(account.get('positions', []), indent=2))
positions = [p for p in account.get('positions', []) if p.get('symbol') == symbol and float(p.get('size', 0)) > 0]
if not positions:
    print("No active position found!")
    exit()

pos = positions[0]
side = pos.get('side')
size = pos.get('size')
entry_price = float(pos.get('entryPrice'))
print(f"Position: {side} {size} @ {entry_price}")

# 2. Calculate SL
sl_side = 'BUY' if side == 'SHORT' else 'SELL'
# Use 1% away
sl_price = entry_price * 1.01 if side == 'SHORT' else entry_price * 0.99
print(f"Proposed SL Price: {sl_price}")

# 3. Format
from apexomni.http_public import HttpPublic
client_public = HttpPublic(APEX_OMNI_HTTP_MAIN)

print("DEBUG: Checking Public Ticker...")
try:
    # Ticker usually contains symbol info or we can infer from it
    tickers = client_public.ticker_v3(symbol=symbol)
    print("DEBUG: Ticker Response:", json.dumps(tickers, indent=2))
except Exception as e:
    print(f"DEBUG: Public Ticker failed: {e}")

# If we can't get config, hardcode for now to test order placement?
# But we need tick size. 
# Let's try to get config from public client if it has such method
# Inspect public client methods
print("DEBUG: Public Client Methods:", [m for m in dir(client_public) if not m.startswith('_')])

# For now, let's assume standard tick size if we can't find it, just to see if we can place order
# BTC usually has 0.1 or 0.05 on Apex
if 'BTC' in symbol:
    tick_size = "0.1" 
    print(f"WARNING: Hardcoding tick size to {tick_size} for test")
else:
    print("ERROR: Cannot determine tick size.")
    exit()



print(f"Tick Size: {tick_size}")

sl_decimal = decimal.Decimal(str(sl_price))
rounded_sl = round_size(sl_decimal, tick_size)
formatted_sl = format_decimal_for_apex(rounded_sl)
print(f"Formatted SL: {formatted_sl}")

params = {
    'symbol': symbol,
    'side': sl_side,
    'type': 'STOP_MARKET',
    'size': size,
    'price': formatted_sl,
    'triggerPrice': formatted_sl,
    'triggerPriceType': 'MARKET',
    'reduceOnly': True
}

print("Sending Order Params:", json.dumps(params, indent=2))

try:
    resp = client.create_order_v3(**params)
    print("RESPONSE:", json.dumps(resp, indent=2))
except Exception as e:
    print("ERROR:", e)
