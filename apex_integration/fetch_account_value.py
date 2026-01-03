import decimal
import time
import os
from dotenv import load_dotenv

from apexomni.constants import APEX_OMNI_HTTP_MAIN, NETWORKID_OMNI_MAIN_ARB
from apexomni.http_private_v3 import HttpPrivate_v3
from apexomni.http_public import HttpPublic

# Load environment variables
load_dotenv(override=True)

key = os.getenv('APEXPRO_API_KEY')
secret = os.getenv('APEXPRO_API_SECRET')
passphrase = os.getenv('APEXPRO_API_PASSPHRASE')

# Initialize private API client for Omni
client = HttpPrivate_v3(APEX_OMNI_HTTP_MAIN, network_id=NETWORKID_OMNI_MAIN_ARB,
                        api_key_credentials={'key': key, 'secret': secret, 'passphrase': passphrase})

# Get symbol configurations
symbol_list = client.configs().get("data").get("perpetualContract")
print("symbol_list:", symbol_list)

# Initialize data variables
wallets = []
openPositions = []
orders = []

# Get current account data
try:
    account = client.get_account_v3()
    print(f"Account response: {account}")
    if account is not None:
        if account.get("wallets") is not None:
            wallets = account.get("wallets")
        if account.get("openPositions") is not None:
            openPositions = account.get("openPositions")
    else:
        print("Warning: No account data")
except Exception as e:
    print(f"Error fetching account data: {e}")
    print("This indicates the Omni account needs to be registered/onboarded first (409 Conflict error).")
    print("You need to:")
    print("1. Log into https://omni.apex.exchange with your wallet")
    print("2. Complete any required KYC/verification")
    print("3. Visit https://omni.apex.exchange/keyManagement to create API keys")
    print("4. Or run account registration if this is a new account")
    print("See Apex Omni documentation for onboarding process.")
    account = None
    exit(1)

# Initialize public client
client_public = HttpPublic(APEX_OMNI_HTTP_MAIN)

# Display account information in tables
if account:
    print("\n" + "="*80)
    print("ACCOUNT OVERVIEW")
    print("="*80)

    # Account basic info
    print(f"Ethereum Address: {account.get('ethereumAddress', 'N/A')}")
    print(f"Account ID: {account.get('id', 'N/A')}")
    print(f"L2 Key: {account.get('l2Key', 'N/A')[:20]}...")

    # Spot Account
    spot = account.get('spotAccount', {})
    if spot:
        print("\n>> SPOT ACCOUNT")
        print("-" * 60)
        print(f"{'Status:':<20} {spot.get('status', 'N/A')}")
        print(f"{'Created At:':<20} {spot.get('createdAt', 'N/A')}")
        print(f"{'ZK Account ID:':<20} {spot.get('zkAccountId', 'N/A')}")
        print(f"{'Default Sub ID:':<20} {spot.get('defaultSubAccountId', 'N/A')}")

    # Contract Account
    contract = account.get('contractAccount', {})
    if contract:
        print("\n>> CONTRACT ACCOUNT")
        print("-" * 60)
        print(f"{'Status:':<20} {contract.get('status', 'N/A')}")
        print(f"{'Taker Fee Rate:':<20} {contract.get('takerFeeRate', 'N/A')}")
        print(f"{'Maker Fee Rate:':<20} {contract.get('makerFeeRate', 'N/A')}")

    # Spot Wallets
    spot_wallets = account.get('spotWallets', [])
    if spot_wallets:
        print("\n>> SPOT WALLETS")
        print("-" * 90)
        print(f"{'Token ID':<10} {'Balance':<25} {'Pending Deposit':<15} {'Pending Withdraw':<15}")
        print("-" * 90)
        for w in spot_wallets:
            token_id = w.get('tokenId', 'N/A')
            balance = f"{float(w.get('balance', '0')):,.6f}" if w.get('balance') else '0.000000'
            pending_dep = f"{float(w.get('pendingDepositAmount', '0')):,.6f}" if w.get('pendingDepositAmount') else '0.000000'
            pending_wd = f"{float(w.get('pendingWithdrawAmount', '0')):,.6f}" if w.get('pendingWithdrawAmount') else '0.000000'
            print(f"{token_id:<10} {balance:<25} {pending_dep:<15} {pending_wd:<15}")

    # Contract Wallets
    contract_wallets = account.get('contractWallets', [])
    if contract_wallets:
        print("\n>> CONTRACT WALLETS")
        print("-" * 100)
        print(f"{'Token':<10} {'Balance':<25} {'Pending Deposit':<15} {'Pending Withdraw':<15} {'Pending Transfer Out':<15}")
        print("-" * 100)
        for w in contract_wallets:
            token = w.get('token', 'N/A')
            balance = f"{float(w.get('balance', '0')):,.6f}" if w.get('balance') else '0.000000'
            pending_dep = f"{float(w.get('pendingDepositAmount', '0')):,.6f}" if w.get('pendingDepositAmount') else '0.000000'
            pending_wd = f"{float(w.get('pendingWithdrawAmount', '0')):,.6f}" if w.get('pendingWithdrawAmount') else '0.000000'
            pending_out = f"{float(w.get('pendingTransferOutAmount', '0')):,.6f}" if w.get('pendingTransferOutAmount') else '0.000000'
            print(f"{token:<10} {balance:<25} {pending_dep:<15} {pending_wd:<15} {pending_out:<15}")

    # Positions
    positions = account.get('positions', [])
    if positions:
        print("\n>> OPEN POSITIONS")
        print("-" * 130)
        print(f"{'Symbol':<12} {'Side':<8} {'Size':<15} {'Entry Price':<15} {'Custom Margin':<15} {'Fee':<15} {'Funding Fee':<12}")
        print("-" * 130)
        for p in positions:
            symbol = p.get('symbol', 'N/A')
            side = p.get('side', 'N/A')
            size = f"{float(p.get('size', '0')):,.3f}" if p.get('size') else '0.000'
            entry_price = f"{float(p.get('entryPrice', '0')):,.1f}" if p.get('entryPrice') else '0.0'
            custom_margin = p.get('customInitialMarginRate', 'N/A')
            fee = f"{float(p.get('fee', '0')):,.6f}" if p.get('fee') else '0.000000'
            funding_fee = f"{float(p.get('fundingFee', '0')):,.6f}" if p.get('fundingFee') else '0.000000'
            print(f"{symbol:<12} {side:<8} {size:<15} {entry_price:<15} {custom_margin:<15} {fee:<15} {funding_fee:<12}")

    print("="*80 + "\n")

    def get_symbol_config(symbol):
        for v in symbol_list:
            if v.get('symbol') == symbol or v.get('crossSymbolName') == symbol or v.get('symbolDisplayName') == symbol:
                return v

    def get_current_price(symbol):
        try:
            ticker_data = client_public.ticker_v3(symbol=symbol.replace('-USDT', '-USDC'))
            if ticker_data.get("data"):
                return ticker_data.get("data", {}).get("op", "0")
            return "0"
        except Exception as e:
            return "0"

    # Calculate real account value from API data
    print("\n--- ACCOUNT VALUE CALCULATION ---")

    # Get current prices for positions
    price_cache = {}
    for position in positions:
        symbol = position.get('symbol', '').replace('-USDT', '-USDC')  # Convert USDT to USDC for ticker
        if symbol and symbol not in price_cache:
            try:
                ticker_data = client_public.ticker_v3(symbol)
                if ticker_data.get("data"):
                    price_cache[symbol.replace('-USDC', '-USDT')] = ticker_data.get("data", {}).get("op", "0")
            except Exception as e:
                print(f"Error getting price for {symbol}: {e}")
                price_cache[position.get('symbol', '')] = "0"

    def get_current_price(symbol):
        return price_cache.get(symbol, "0")

    # Calculate wallet values
    totalWalletValue = decimal.Decimal('0.0')
    print("Wallet Balances:")
    for wallet in contract_wallets + spot_wallets:
        balance = decimal.Decimal(wallet.get('balance', '0'))
        token = wallet.get('token', wallet.get('tokenId', 'UNKNOWN'))

        # Convert token balances to USDT value
        if token == 'USDT' or token == '141':  # 141 might be USDT token ID
            value = balance
            print(f"  {token}: {balance:,.6f} USDT")
        elif token == 'ETH' or token == '36':  # ETH token
            try:
                eth_price = get_current_price('ETH-USDT')
                if eth_price and eth_price != "0":
                    value = balance * decimal.Decimal(eth_price)
                    print(f"  {token}: {balance:,.6f} ETH = {value:,.2f} USDT")
                else:
                    value = balance  # Keep in ETH if no price
                    print(f"  {token}: {balance:,.6f} ETH (no price available)")
            except:
                value = balance
                print(f"  {token}: {balance:,.6f} ETH")
        else:
            # Skip other tokens or treat as USDT value
            value = balance if token == 'USDT' else decimal.Decimal('0')
            print(f"  {token}: {balance:,.6f} (unsupported token)")

        totalWalletValue += value

    print(f"Total from wallets: {totalWalletValue:,.2f} USDT")

    # Calculate position values and unrealized P&L
    totalPositionValue = decimal.Decimal('0.0')
    totalUnrealizedPnL = decimal.Decimal('0.0')

    print("\nPosition Values:")
    for position in positions:
        symbol = position.get('symbol', '')
        side = position.get('side', '')
        size = decimal.Decimal(position.get('size', '0'))
        entry_price = decimal.Decimal(position.get('entryPrice', '0'))
        current_price_str = get_current_price(symbol)
        current_price = decimal.Decimal(current_price_str) if current_price_str else decimal.Decimal('0')

        if size == 0 or entry_price == 0:
            position_value = decimal.Decimal('0')
            pnl = decimal.Decimal('0')
        else:
            if side == 'LONG':
                position_value = size * current_price
                pnl = size * (current_price - entry_price)
            else:  # SHORT
                position_value = size * current_price * decimal.Decimal('-1')
                pnl = size * (entry_price - current_price)

        totalPositionValue += position_value
        totalUnrealizedPnL += pnl

        custom_margin = position.get('customInitialMarginRate', '0')
        print(f"{symbol:<12} {side:<8} Size:{size:,.3f} Entry:{entry_price:,.1f} Current:{current_price:,.1f} P&L:{pnl:,.2f} USDT")

    print(f"\nTotal position value: {totalPositionValue:,.2f} USDT")
    print(f"Total unrealized P&L: {totalUnrealizedPnL:,.2f} USDT")

    # Total account value
    totalAccountValue = totalWalletValue + totalUnrealizedPnL

    print(f"\n{'='*50}")
    print(f"TOTAL ACCOUNT VALUE: {totalAccountValue:,.2f} USDT")
    print(f"{'='*50}")

    # Calculate margin requirements for open positions with size > 0
    activePositions = [p for p in positions if decimal.Decimal(p.get('size', '0')) > 0]

    totalInitialMarginRequirement = decimal.Decimal('0.0')
    totalMaintenanceMarginRequirement = decimal.Decimal('0.0')

    for position in activePositions:
        symbol = position.get('symbol', '')
        size = decimal.Decimal(position.get('size', '0'))
        entry_price = decimal.Decimal(position.get('entryPrice', '0'))
        custom_margin = decimal.Decimal(position.get('customInitialMarginRate', '0.10'))
        config = get_symbol_config(symbol)
        maintenance_rate = decimal.Decimal(config.get('maintenanceMarginRate', '0.01')) if config else decimal.Decimal('0.01')

        # Initial margin based on custom rate
        initial_margin = entry_price * size * custom_margin
        totalInitialMarginRequirement += initial_margin

        # Maintenance margin based on current price
        current_price_str = get_current_price(symbol)
        current_price = decimal.Decimal(current_price_str) if current_price_str else entry_price
        maintenance_margin = current_price * size * maintenance_rate
        totalMaintenanceMarginRequirement += maintenance_margin

    print(f"\nMargin Requirements:")
    print(f"Total Initial Margin: {totalInitialMarginRequirement:,.2f} USDT")
    print(f"Total Maintenance Margin: {totalMaintenanceMarginRequirement:,.2f} USDT")

    availableValue = totalAccountValue - totalInitialMarginRequirement
    print(f"Available Value (for new positions): {availableValue:,.2f} USDT")

    # FIXED: Calculate liquidation analysis using correct perpetual futures formula
    btc_position = None
    for pos in activePositions:
        if pos.get('symbol') == 'BTC-USDT' and pos.get('size', '0') != '0' and pos.get('size', '0') != '0.000':
            btc_position = pos
            break

    if btc_position:
        size_str = btc_position.get('size', '0')
        entry_price_str = btc_position.get('entryPrice', '0')
        balance = totalWalletValue  # Account balance including contract wallets

        try:
            size = decimal.Decimal(size_str)
            entry_price = decimal.Decimal(entry_price_str)
            print(f"\n🚨 LIQUIDATION ANALYSIS for BTC-USDT LONG {size} contracts")

            # Get maintenance margin rate from symbol config
            btc_config = get_symbol_config('BTC-USDT')
            maintenance_margin_rate = decimal.Decimal(btc_config.get('maintenanceMarginRate', '0.007')) if btc_config else decimal.Decimal('0.007')

            print(f"Position Size: {size:,.4f} BTC")
            print(f"Entry Price: ${entry_price:,.2f}")
            print(f"Wallet Balance: ${balance:,.2f} USDT")
            print(f"Maintenance Rate: {maintenance_margin_rate * 100:.1f}%")

            # Calculate current position status
            current_price_str = get_current_price('BTC-USDT')
            current_price = decimal.Decimal(current_price_str) if current_price_str and current_price_str != "0" else entry_price
            unrealized_pnl = size * (current_price - entry_price)
            account_equity = balance + unrealized_pnl

            # Correct perpetual futures liquidation logic
            # For LONG positions: Liquidation occurs when:
            # Account Equity ≤ 0 OR Unrealized Loss > (Balance + Maintenance Margin Buffer)

            # Maintenance margin acts as buffer - liquidation when losses exceed available equity + buffer
            maintenance_buffer = entry_price * size * maintenance_margin_rate
            liquidation_threshold = abs(balance) + maintenance_buffer

            print(f"\nCurrent Stats:")
            print(f"Current BTC Price: ${current_price:,.2f}")
            print(f"Unrealized P&L: ${unrealized_pnl:,.2f} USDT")
            print(f"Account Equity: ${account_equity:,.2f} USDT")
            print(f"Maintenance Buffer: ${maintenance_buffer:,.2f} USDT")

            # Calculate liquidation price properly
            if balance <= 0:
                # When balance is negative, liquidation occurs when equity reaches zero
                liquidation_price = entry_price - (abs(balance) / size)
            else:
                # When balance is positive, liquidation occurs when losses exceed balance + maintenance_buffer
                liquidation_loss_needed = balance + maintenance_buffer
                liquidation_price = entry_price - (liquidation_loss_needed / size)

            if liquidation_price > 0:
                # Calculate how much BTC needs to drop from entry
                usd_drop_from_entry = entry_price - liquidation_price
                pct_drop = (usd_drop_from_entry / entry_price) * 100

                print(f"\n💀 LIQUIDATION PRICE: ${liquidation_price:,.2f}")
                print(f"USDT drop needed: ${usd_drop_from_entry:,.2f} ({pct_drop:.2f}%)")

                # Compare with current price
                current_drop = entry_price - current_price
                if current_price < liquidation_price:
                    print(f"⚠️  WARNING: Current price ${current_price:,.2f} is BELOW liquidation price!")
                else:
                    remaining_drop = liquidation_price - current_price
                    print(f"Distance to liquidation: ${remaining_drop:,.2f}")
            else:
                print("⚠️  LIQUIDATION RISK: Position should already be liquidated!")

        except Exception as e:
            print(f"❌ Error in liquidation analysis: {e}")

# Get open orders
try:
    get_orders = client.open_orders_v3()
    if get_orders.get("data") is not None:
        orders = get_orders.get("data")
except Exception as e:
    print(f"Error fetching open orders: {e}")

# Ticker prices will be fetched on demand

def get_symbol_config(symbol):
    for v in symbol_list:
        if v.get('symbol') == symbol or v.get('crossSymbolName') == symbol or v.get('symbolDisplayName') == symbol:
            return v

def get_symbol_price(symbol):
    try:
        ticker_data = client_public.ticker_v3(symbol.replace('-USDT', '-USDC'))  # Convert to USDC format
        return ticker_data.get("data", {}).get("op", "0")
    except:
        return None

print("\\n--- End of Account Value Calculation ---")
