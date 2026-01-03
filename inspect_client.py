
import os
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

from apexomni.http_private_sign import HttpPrivateSign
from apexomni.constants import APEX_OMNI_HTTP_MAIN, NETWORKID_OMNI_MAIN_ARB

# Load env
load_dotenv(override=True)
load_dotenv("apex_integration/.env", override=True)

key = os.getenv('APEXPRO_API_KEY')
secret = os.getenv('APEXPRO_API_SECRET')
passphrase = os.getenv('APEXPRO_API_PASSPHRASE')

client = HttpPrivateSign(
    APEX_OMNI_HTTP_MAIN, 
    network_id=NETWORKID_OMNI_MAIN_ARB,
    zk_seeds="dummy", 
    zk_l2Key="dummy", 
    api_key_credentials={'key': key, 'secret': secret, 'passphrase': passphrase}
)

client.configs_v3() # Initialize config

print("-" * 50)
print("AVAILABLE CLIENT METHODS (HttpPrivateSign):")
# Just list them, don't execute or getattr if risky
methods = [method_name for method_name in dir(client) if not method_name.startswith("__")]
for m in sorted(methods):
    print(m)
