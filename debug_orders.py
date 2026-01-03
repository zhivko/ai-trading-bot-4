
import os
import json
import logging
from dotenv import load_dotenv
# Compatibility Shim
try:
    from sympy.core.numbers import igcdex
except ImportError:
    try:
        from sympy.core.intfunc import igcdex
        import sympy.core.numbers
        sympy.core.numbers.igcdex = igcdex
        print("Monkey-patched sympy.core.numbers.igcdex")
    except ImportError:
        pass

from apexomni.http_private_sign import HttpPrivateSign
from apexomni.constants import APEX_OMNI_HTTP_MAIN, NETWORKID_OMNI_MAIN_ARB

# Load env
load_dotenv(override=True)
load_dotenv("apex_integration/.env", override=True)

# Import ApexPro components
from apexomni.http_private_v3 import HttpPrivate_v3
from apexomni.http_public import HttpPublic

# Initialize ApexPro clients (same as fetch_account_value.py and trading_service.py)
key = os.getenv('APEXPRO_API_KEY')
secret = os.getenv('APEXPRO_API_SECRET')
passphrase = os.getenv('APEXPRO_API_PASSPHRASE')

# Derive ZK keys
eth_private_key = os.getenv('APEXPRO_ETH_PRIVATE_KEY')
if eth_private_key:
    # Ensure the private key has 0x prefix
    if not eth_private_key.startswith('0x'):
        eth_private_key = '0x' + eth_private_key

    # Use HttpPrivate_v3 for derivation
    temp_client = HttpPrivate_v3(APEX_OMNI_HTTP_MAIN, network_id=NETWORKID_OMNI_MAIN_ARB, eth_private_key=eth_private_key)
    temp_client.configs_v3()  # Initialize configuration

    # Derive ZK keys using default_address
    derived_keys = temp_client.derive_zk_key(temp_client.default_address)
    
    if derived_keys and 'seeds' in derived_keys and 'l2Key' in derived_keys:
        zk_seeds = derived_keys['seeds']
        l2_key = derived_keys['l2Key']
        print("ZK credentials derived successfully")
    else:
        print("ZK key derivation returned invalid format")
        zk_seeds = "dummy"
        l2_key = "dummy"
else:
    print("No ETH private key found")
    zk_seeds = "dummy"
    l2_key = "dummy"

client = HttpPrivateSign(APEX_OMNI_HTTP_MAIN, network_id=NETWORKID_OMNI_MAIN_ARB,
                         zk_seeds=zk_seeds, zk_l2Key=l2_key,
                         api_key_credentials={'key': key, 'secret': secret, 'passphrase': passphrase})
client.configs_v3()

print("-" * 50)
print("FETCHING ACCOUNT (POSITIONS)...")
try:
    account = client.get_account_v3()
    positions = account.get('positions', [])
    print(f"Total Positions: {len(positions)}")
    
    for i, pos in enumerate(positions):
        # Filter empty positions
        if float(pos.get('size', 0)) == 0: continue
        
        print(f"\n[Position {i}]")
        print(f"Symbol: {pos.get('symbol')}")
        print(f"Side: {pos.get('side')}")
        print(f"Size: {pos.get('size')}")
        print(f"EntryPrice: {pos.get('entryPrice')}")
        # Check for SL/TP fields directly on position
        print(f"Raw Position Keys: {list(pos.keys())}")
        print(f"Position JSON: {json.dumps(pos, indent=2)}")
        
except Exception as e:
    print(f"Error fetching account: {e}")

print("-" * 50)
print("FETCHING OPEN ORDERS...")
try:
    resp = client.open_orders_v3()
    data = resp.get('data', [])
    print(f"Total Open Orders: {len(data)}")
    
    for i, order in enumerate(data):
        print(f"\n[Order {i}]")
        print(f"Symbol: {order.get('symbol')}")
        print(f"Side: {order.get('side')}")
        print(f"Type: {order.get('type')}")
        print(f"TriggerPrice: {order.get('triggerPrice')}")
        print(f"Price: {order.get('price')}")
        print(f"Status: {order.get('status')}")
        print(f"Order JSON: {json.dumps(order, indent=2)}")
        
except Exception as e:
    print(f"Error fetching orders: {e}")

print("-" * 50)
print("FETCHING OPEN ORDERS (BTC-USDT)...")
try:
    resp = client.open_orders_v3(symbol="BTC-USDT")
    data = resp.get('data', [])
    print(f"Total Open Orders (Filtered): {len(data)}")
    print(json.dumps(resp, indent=2))
except Exception as e:
    print(f"Error fetching orders: {e}")

print("-" * 50)
print("FETCHING HISTORY ORDERS (Last 5)...")
try:
    # limit=5
    resp = client.history_orders_v3(symbol="BTC-USDT", limit=5)
    data = resp.get('data', [])
    print(f"Total History Orders: {len(data)}")
    for i, order in enumerate(data):
        print(f"\n[History {i}]")
        print(f"ID: {order.get('id')}")
        print(f"Status: {order.get('status')}")
        print(f"Type: {order.get('type')}")
        print(f"Side: {order.get('side')}")
        print(f"TriggerPrice: {order.get('triggerPrice')}")
        print(f"Price: {order.get('price')}")
        print(f"SL/TP Info: {json.dumps({k:v for k,v in order.items() if 'sl' in k.lower() or 'tp' in k.lower()})}")
except Exception as e:
    print(f"Error fetching history: {e}")
