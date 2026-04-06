#!/usr/bin/env python3
"""
LIVE Trading Engine for Auto-Trader (Spot Market)
WARNING: THIS SCRIPT TRADES REAL MONEY ON BINANCE

Designed to run via cron. 
Uses CCXT to execute market orders based on Neural Network predictions.
"""
import os
import sys
import json
import logging
import requests
import ccxt
from dotenv import load_dotenv
from datetime import datetime, timezone

# Load environment variables
load_dotenv()
WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from predict import get_signal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/live_trader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ──────────────── CONFIGURATION ────────────────
SYMBOL = "RONIN/USDT"
BASE_ASSET = "RONIN"    # What we buy
QUOTE_ASSET = "USDT"    # What we spend
STOP_LOSS_PCT = 0.03    # 3% Stop Loss
TAKE_PROFIT_PCT = 0.07  # 7% Take Profit
CONFIDENCE_THRESHOLD = 50.0 
STATE_FILE = "logs/live_state.json"
# ────────────────────────────────────────────────

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "position": None,  # None means we hold USDT. "LONG" means we hold RONIN
        "entry_price": 0.0,
        "entry_amount": 0.0,
        "last_signal": None,
        "total_trades": 0
    }

def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def send_webhook_notification(action, symbol, price, details):
    if not WEBHOOK_URL: return
    payload = {
        "action": action, "symbol": symbol, "price": price,
        "details": details, "timestamp": datetime.now(timezone.utc).isoformat()
    }
    try: requests.post(WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e: logger.error(f"Webhook failed: {e}")

def get_exchange():
    """Initialize Binance connection natively via CCXT."""
    binance = ccxt.binance({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_SECRET_KEY,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot'
        }
    })
    return binance

def run_live_trade():
    logger.info("=== STARTING LIVE TRADING CYCLE ===")
    
    if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
        logger.error("Missing Binance API keys in .env file! Aborting.")
        return

    try:
        exchange = get_exchange()
        # Fetch actual account balance
        balance = exchange.fetch_balance()
        free_usdt = balance[QUOTE_ASSET]['free'] if QUOTE_ASSET in balance else 0.0
        free_base = balance[BASE_ASSET]['free'] if BASE_ASSET in balance else 0.0
        
        logger.info(f"Binance Wallet -> Free USDT: ${free_usdt:.2f} | Free {BASE_ASSET}: {free_base:.4f}")
    except Exception as e:
        logger.error(f"Failed to connect to Binance API: {e}")
        return

    # 1. Fetch AI Signal
    try:
        logger.info("Fetching neural network signal...")
        result = get_signal()
        current_price = result["price"]
        signal = result["signal"]
        confidence = result["confidence"]
        logger.info(f"Signal: {signal} | Confidence: {confidence:.2f}% | Market Price: ${current_price:.6f}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return

    state = load_state()
    current_position = state["position"]
    action_taken = "HOLD"
    
    # Validation Check: Does our real balance match our JSON state?
    if current_position == "LONG" and free_base * current_price < 10.0:
        logger.warning(f"State mismatch: JSON says LONG but we have no {BASE_ASSET} in Binance. Resetting state to FLAT (Holding USDT).")
        current_position = None
        state["position"] = None
        save_state(state)
        
    if current_position is None and free_usdt < 10.0 and free_base * current_price > 10.0:
        logger.warning(f"State mismatch: JSON says FLAT but we have {BASE_ASSET} in Binance. Adopting position as LONG.")
        current_position = "LONG"
        state["position"] = "LONG"
        state["entry_price"] = current_price # Fallback entry price
        save_state(state)

    # 2. Reversals, Take Profit, and Stop Loss logic (If we are holding the coin)
    if current_position == "LONG":
        entry_price = state["entry_price"]
        pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
        logger.info(f"Current Trade PnL: {pnl_pct*100:.2f}% (Entry: ${entry_price:.6f})")
        
        should_close = False
        reason = ""
        
        # Check Stop Loss (-3%)
        if pnl_pct <= -STOP_LOSS_PCT:
            should_close = True; reason = "STOP_LOSS"
        # Check Take Profit (+7%)
        elif pnl_pct >= TAKE_PROFIT_PCT:
            should_close = True; reason = "TAKE_PROFIT"
        # Check AI Reversal (AI now predicts SHORT)
        elif signal == "SHORT" and confidence >= CONFIDENCE_THRESHOLD:
            should_close = True; reason = "REVERSAL_TO_SHORT"
            
        if should_close:
            logger.info(f"Attempting to CLOSE LONG position via MARKET_SELL. Reason: {reason}")
            try:
                # Sell 100% of our RONIN balance
                amount_to_sell = free_base
                order = exchange.create_market_sell_order(SYMBOL, amount_to_sell)
                fill_price = order['average'] if order.get('average') else current_price
                
                logger.info(f"Successfully Sold {amount_to_sell} {BASE_ASSET} at ~${fill_price}")
                send_webhook_notification("CLOSE_POSITION (LIVE)", SYMBOL, fill_price, f"Reason: {reason}")
                
                state["position"] = None
                state["total_trades"] += 1
                current_position = None
                action_taken = f"SELL ({reason})"
            except Exception as e:
                logger.error(f"Failed to execute MARKET_SELL: {e}")

    # 3. Entry Logic (If we are sitting in USDT and AI says LONG)
    if current_position is None:
        if signal == "LONG" and confidence >= CONFIDENCE_THRESHOLD:
            if free_usdt > 10.0: # Minimum $10 to trade on Binance normally
                # We use 99% of our USDT to buy RONIN (keeping 1% buffer for fractional fees)
                usdt_to_spend = free_usdt * 0.99
                logger.info(f"Attempting to OPEN LONG via MARKET_BUY. Investing: ${usdt_to_spend:.2f}")
                try:
                    try:
                        order = exchange.create_order(
                            SYMBOL, 'market', 'buy', 
                            amount=None, 
                            price=None, 
                            params={'quoteOrderQty': usdt_to_spend}
                        )
                    except Exception as fallback_e:
                        amount_to_buy = usdt_to_spend / current_price
                        order = exchange.create_market_buy_order(SYMBOL, amount_to_buy)
                        
                    fill_price = order['average'] if order.get('average') else current_price
                    logger.info(f"Successfully Bought {BASE_ASSET} at ~${fill_price}")
                    send_webhook_notification("OPEN_POSITION (LIVE)", SYMBOL, fill_price, f"Deployed ${usdt_to_spend:.2f} USDT")
                    
                    state["position"] = "LONG"
                    state["entry_price"] = float(fill_price)
                    action_taken = "BUY"
                except Exception as e:
                    logger.error(f"Failed to execute MARKET_BUY: {e}")
            else:
                logger.warning(f"AI signaled LONG, but free USDT (${free_usdt:.2f}) is too low to trade.")
        else:
            logger.info("Holding USDT. No entry conditions met.")
    
    save_state(state)
    logger.info(f"Cycle Complete. Action: {action_taken}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        state = load_state()
        print(f"[{SYMBOL}] Live Position: {state['position']} | Entry: ${state['entry_price']} | Total Executions: {state['total_trades']}")
    else:
        run_live_trade()
