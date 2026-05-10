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
import csv
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

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
TRAILING_ACTIVATION = 0.04  # Activate trailing stop after +4% gain
TRAILING_DROP_PCT = 0.015   # Sell when price drops 1.5% from peak
PARTIAL_TP_PCT = 0.04       # Trigger partial take profit at +4%
PARTIAL_TP_SELL_RATIO = 0.50  # Sell 50% of position at partial TP
COOLDOWN_HOURS = 8           # Hours to wait after a stop loss before re-entering
ANTI_PUMP_PCT = 0.08         # Don't buy if price pumped > 8% since last 4H check
STALE_HOURS = 16             # Hours before lowering TP threshold (4 x 4H cycles)
STALE_TP_PCT = 0.03          # Reduced TP when position is stale (3%)
CONFIDENCE_THRESHOLD = 50.0
NUM_BOTS = 2                 # Number of bots sharing the wallet
PEER_ASSETS = ["AXS"]        # Other bot's assets to include in portfolio valuation
ATR_FILTER_ENABLED = True    # Block buys when realized vol is in chop regime
ATR_FILTER_PERCENTILE = 25   # Block if current ATR% is below this percentile of recent history
POSITION_FRACTION = 1.00     # Fixed fraction of bot's capital to deploy per trade
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
        "entry_time": None,  # ISO timestamp of position open
        "last_signal": None,
        "total_trades": 0,
        "partial_tp_taken": False,
        "last_stop_loss_time": None  # ISO timestamp of last stop loss hit
    }

def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

TRADE_HISTORY_FILE = "logs/trade_history.csv"

def log_trade(action, price, pnl_pct, reason, balance_usdt):
    """Append a trade record to the CSV history."""
    file_exists = os.path.exists(TRADE_HISTORY_FILE)
    with open(TRADE_HISTORY_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "action", "price", "pnl_pct", "reason", "balance_usdt"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            action, f"{price:.6f}",
            f"{pnl_pct:.2f}" if pnl_pct is not None else "",
            reason, f"{balance_usdt:.2f}"
        ])

def generate_report():
    """Print a compact performance summary for sharing."""
    state = load_state()
    total_evals = state.get("total_evals", 0)
    winning = state.get("winning_trades", 0)
    win_rate = (winning / total_evals * 100) if total_evals > 0 else 0.0
    
    print(f"\n{'='*50}")
    print(f"  📊 RONIN/USDT — LIVE BOT PERFORMANCE REPORT")
    print(f"{'='*50}")
    print(f"  Position     : {state.get('position') or 'FLAT (Holding USDT)'}")
    if state.get('position') == 'LONG':
        print(f"  Entry Price  : ${state.get('entry_price', 0):.6f}")
        print(f"  Peak Price   : ${state.get('peak_price', 0):.6f}")
    print(f"  Total Trades : {state.get('total_trades', 0)}")
    print(f"  AI Win Rate  : {win_rate:.1f}% ({winning}/{total_evals})")
    print(f"{'─'*50}")
    
    if not os.path.exists(TRADE_HISTORY_FILE):
        print("  No trade history yet.")
        print(f"{'='*50}\n")
        return
    
    trades = []
    with open(TRADE_HISTORY_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(row)
    
    if not trades:
        print("  No trades recorded.")
        print(f"{'='*50}\n")
        return
    
    sells = [t for t in trades if t['action'] == 'SELL']
    buys = [t for t in trades if t['action'] == 'BUY']
    
    wins = [t for t in sells if float(t.get('pnl_pct', 0)) > 0]
    losses = [t for t in sells if float(t.get('pnl_pct', 0)) <= 0]
    pnls = [float(t['pnl_pct']) for t in sells if t.get('pnl_pct')]
    
    print(f"  Completed Trades: {len(sells)} ({len(wins)} wins / {len(losses)} losses)")
    if pnls:
        print(f"  Best Trade   : +{max(pnls):.2f}%")
        print(f"  Worst Trade  : {min(pnls):.2f}%")
        print(f"  Avg PnL/Trade: {sum(pnls)/len(pnls):.2f}%")
        
        # Estimate total return
        cumulative = 1.0
        for p in pnls:
            cumulative *= (1 + p/100)
        total_return = (cumulative - 1) * 100
        print(f"  Cumulative $$ : {total_return:+.2f}%")
    
    if trades:
        print(f"{'─'*50}")
        print(f"  Last 5 Trades:")
        for t in trades[-5:]:
            pnl = f" PnL:{t['pnl_pct']}%" if t.get('pnl_pct') else ""
            ts = t['timestamp'][:16].replace('T', ' ')
            print(f"    {ts} | {t['action']:4s} @ ${t['price']}{pnl} [{t.get('reason','')}]")
    
    print(f"{'='*50}\n")

def send_webhook_notification(action, symbol, price, details, extra=None):
    """Send enriched notification to n8n/Telegram webhook."""
    if not WEBHOOK_URL: return
    payload = {
        "action": action, "symbol": symbol, "price": f"${price:.4f}" if isinstance(price, float) else price,
        "details": details, "timestamp": datetime.now(timezone.utc).isoformat()
    }
    if extra:
        payload.update(extra)
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
        
        # Estimate total portfolio value (USDT + all crypto at market price)
        total_portfolio = free_usdt
        all_assets = [BASE_ASSET] + PEER_ASSETS
        for asset in all_assets:
            held = balance['free'].get(asset, 0.0)
            if held > 0:
                try:
                    ticker = exchange.fetch_ticker(f"{asset}/USDT")
                    total_portfolio += held * ticker['last']
                except: pass
        
        max_position = total_portfolio / NUM_BOTS
        logger.info(f"Binance Wallet -> Free USDT: ${free_usdt:.2f} | Free {BASE_ASSET}: {free_base:.4f} | Portfolio: ${total_portfolio:.2f} | Max/bot: ${max_position:.2f}")
    except Exception as e:
        logger.error(f"Failed to connect to Binance API: {e}")
        return

    current_time = datetime.now(timezone.utc)
    # Deduplicate: only treat as 4H boundary if we haven't processed this exact hour block yet
    state_peek = load_state()
    current_boundary_key = f"{current_time.strftime('%Y-%m-%d')}_{current_time.hour}"
    already_processed = state_peek.get("last_4h_boundary") == current_boundary_key
    is_4h_boundary = (current_time.hour % 4 == 0 and current_time.minute < 15 and not already_processed)

    state = load_state()
    current_position = state.get("position")
    action_taken = "HOLD"

    signal = state.get("last_signal")
    confidence = 0.0
    current_price = 0.0

    if is_4h_boundary:
        # 1. Fetch AI Signal
        try:
            logger.info("4H Boundary reached. Fetching neural network signal...")
            result = get_signal()
            current_price = result["price"]
            signal = result["signal"]
            confidence = result["confidence"]
            logger.info(f"Signal: {signal} | Confidence: {confidence:.2f}% | Market Price: ${current_price:.6f}")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return
            
        # Evaluate previous prediction accuracy
        last_signal = state.get("last_signal")
        last_price = state.get("last_price")
        
        if last_signal and last_price:
            actual_direction = "UP" if current_price > last_price else "DOWN"
            expected_direction = "UP" if last_signal == "LONG" else "DOWN"
            prev_correct = (actual_direction == expected_direction)
            
            state["total_evals"] = state.get("total_evals", 0) + 1
            if prev_correct:
                state["winning_trades"] = state.get("winning_trades", 0) + 1
                
            emoji = "✅" if prev_correct else "❌"
            logger.info(
                f"Previous prediction: {last_signal} at ${last_price:.6f} "
                f"→ Actual: {actual_direction} → {emoji}"
            )
    else:
        if current_position is not None:
            # Intermediate cycle just fetching price for Stops
            try:
                ticker = exchange.fetch_ticker(SYMBOL)
                current_price = ticker['last']
            except Exception as e:
                logger.error(f"Ticker fetch failed: {e}")
                return
        else:
            # Holding USDT and not a boundary, just completely exit to save output spam
            return
    
    # Validation Check: Does our real balance match our JSON state?
    if current_position == "LONG" and free_base * current_price < 1.0:
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

    # 2. Trailing Stop, Stop Loss, Partial TP, and Reversal logic (If we are holding the coin)
    if current_position == "LONG":
        entry_price = state["entry_price"]
        peak_price = state.get("peak_price", entry_price)
        partial_tp_taken = state.get("partial_tp_taken", False)
        
        # Update peak price tracker
        if current_price > peak_price:
            peak_price = current_price
            state["peak_price"] = peak_price
        
        pnl_pct = round((current_price - entry_price) / entry_price, 4) if entry_price > 0 else 0
        pnl_from_peak = (current_price - peak_price) / peak_price if peak_price > 0 else 0
        # ── STALE POSITION CHECK: Lower TP after 16h of lateralization ──
        entry_time = state.get("entry_time")
        is_stale = False
        if entry_time:
            entry_dt = datetime.fromisoformat(entry_time)
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            hours_held = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600
            if hours_held >= STALE_HOURS and not partial_tp_taken:
                is_stale = True
                logger.info(f"⏳ Stale position ({hours_held:.1f}h). TP reduced: {PARTIAL_TP_PCT*100:.0f}% → {STALE_TP_PCT*100:.0f}%")
        
        active_tp_pct = STALE_TP_PCT if is_stale else PARTIAL_TP_PCT
        active_trailing_activation = STALE_TP_PCT if is_stale else TRAILING_ACTIVATION
        peak_gain = round((peak_price - entry_price) / entry_price, 4) if entry_price > 0 else 0
        trailing_active = peak_gain >= active_trailing_activation
        
        tp_status = " [50% sold]" if partial_tp_taken else (" [STALE]" if is_stale else "")
        status_extra = f" | Peak: ${peak_price:.6f}{tp_status}" if trailing_active else tp_status
        logger.info(f"Current Trade PnL: {pnl_pct*100:.2f}% (Entry: ${entry_price:.6f}){status_extra}")
        
        if trailing_active:
            logger.info(f"🎯 Trailing Stop ACTIVE. Drop from peak: {pnl_from_peak*100:.2f}% (trigger at -{TRAILING_DROP_PCT*100:.1f}%)")
        
        # ── PARTIAL TAKE PROFIT: Sell 50% at TP threshold ──
        if not partial_tp_taken and pnl_pct >= active_tp_pct:
            partial_amount = free_base * PARTIAL_TP_SELL_RATIO
            logger.info(f"💰 PARTIAL TP triggered at +{pnl_pct*100:.1f}%! Selling {PARTIAL_TP_SELL_RATIO*100:.0f}% ({partial_amount:.4f} {BASE_ASSET})")
            try:
                order = exchange.create_market_sell_order(SYMBOL, partial_amount)
                fill_price = order['average'] if order.get('average') else current_price
                
                partial_pnl_pct = ((float(fill_price) - entry_price) / entry_price) * 100
                estimated_balance = float(fill_price) * partial_amount
                log_trade("SELL_PARTIAL", float(fill_price), partial_pnl_pct, f"PARTIAL_TP_50% (+{partial_pnl_pct:.1f}%)", estimated_balance)
                
                state["partial_tp_taken"] = True
                partial_tp_taken = True
                logger.info(f"✅ Locked +{partial_pnl_pct:.1f}% profit on 50%. Remaining 50% rides with trailing stop.")
                send_webhook_notification("PARTIAL_TP (LIVE)", SYMBOL, fill_price,
                    f"Sold 50% at +{partial_pnl_pct:.1f}%",
                    {"pnl_pct": f"+{partial_pnl_pct:.1f}%", "strategy": "Remaining 50% rides with trailing stop"})
                
                # Re-fetch balance after partial sell
                balance = exchange.fetch_balance()
                free_base = balance['free'].get(BASE_ASSET, 0.0)
            except Exception as e:
                logger.error(f"Failed to execute PARTIAL_TP sell: {e}")
        
        # ── FULL EXIT CHECKS ──
        should_close = False
        reason = ""
        
        # Check Stop Loss (-3%)
        if pnl_pct <= -STOP_LOSS_PCT:
            should_close = True; reason = "STOP_LOSS"
        # Check Trailing Stop (price dropped 1.5% from peak, only if trailing is active)
        elif trailing_active and pnl_from_peak <= -TRAILING_DROP_PCT:
            final_pnl = pnl_pct * 100
            should_close = True; reason = f"TRAILING_STOP (locked {final_pnl:.1f}% profit)"
        # Check AI Reversal (AI now predicts SHORT) - only on 4H boundaries
        elif is_4h_boundary and signal == "SHORT" and confidence >= CONFIDENCE_THRESHOLD:
            should_close = True; reason = "REVERSAL_TO_SHORT"
            
        if should_close:
            sell_label = "remaining " if partial_tp_taken else ""
            logger.info(f"Attempting to CLOSE {sell_label}LONG position via MARKET_SELL. Reason: {reason}")
            try:
                # Sell all remaining BASE_ASSET balance
                amount_to_sell = free_base
                order = exchange.create_market_sell_order(SYMBOL, amount_to_sell)
                fill_price = order['average'] if order.get('average') else current_price
                
                logger.info(f"Successfully Sold {amount_to_sell} {BASE_ASSET} at ~${fill_price}")
                
                final_pnl_pct = ((float(fill_price) - entry_price) / entry_price) * 100
                estimated_balance = float(fill_price) * amount_to_sell
                
                send_webhook_notification("CLOSE_POSITION (LIVE)", SYMBOL, fill_price,
                    f"Reason: {reason}",
                    {"pnl_pct": f"{final_pnl_pct:+.1f}%", "partial_tp": "Yes (50% already sold)" if partial_tp_taken else "No"})
                
                log_trade("SELL", float(fill_price), final_pnl_pct, reason, estimated_balance)
                
                state["position"] = None
                state["peak_price"] = 0.0
                state["partial_tp_taken"] = False
                state["total_trades"] += 1
                current_position = None
                action_taken = f"SELL ({reason})"
                
                # Record stop loss time for cooldown
                if reason == "STOP_LOSS":
                    state["last_stop_loss_time"] = datetime.now(timezone.utc).isoformat()
                    logger.info(f"⏸️ Cooldown activated: {COOLDOWN_HOURS}h before next entry.")
            except Exception as e:
                logger.error(f"Failed to execute MARKET_SELL: {e}")

    # 3. Entry Logic (Only execute if on 4H boundary)
    # Prevent Re-Entry: Only buy if we didn't just sell in this exact same cycle
    if is_4h_boundary and current_position is None and "SELL" not in action_taken:
        if signal == "LONG" and confidence >= CONFIDENCE_THRESHOLD:
            # ── COOLDOWN CHECK: Wait after stop loss (override if confidence >= 70%) ──
            last_sl_time = state.get("last_stop_loss_time")
            if last_sl_time:
                hours_since_sl = (datetime.now(timezone.utc) - datetime.fromisoformat(last_sl_time)).total_seconds() / 3600
                if hours_since_sl < COOLDOWN_HOURS and confidence < 70:
                    remaining = COOLDOWN_HOURS - hours_since_sl
                    logger.info(f"⏸️ Cooldown active: {remaining:.1f}h remaining after stop loss. Skipping entry.")
                else:
                    if confidence >= 70 and hours_since_sl < COOLDOWN_HOURS:
                        logger.info(f"🔥 High confidence ({confidence:.1f}%) overrides cooldown ({hours_since_sl:.1f}h/{COOLDOWN_HOURS}h).")
                    state["last_stop_loss_time"] = None  # Clear cooldown
            
            # ── ANTI-PUMP FILTER: Don't buy at the top ──
            last_known_price = state.get("last_price")
            if last_known_price and last_known_price > 0:
                price_change = (current_price - last_known_price) / last_known_price
                if price_change > ANTI_PUMP_PCT:
                    logger.info(f"🚫 Anti-pump filter: price already up +{price_change*100:.1f}% since last check. Too risky to enter.")
                    signal = "HOLD"

            # ── ANTI-CHOP FILTER: Don't buy when realized vol is in low percentile ──
            if signal == "LONG" and ATR_FILTER_ENABLED:
                atr_history = result.get("atr_pct_history") or []
                atr_current = result.get("atr_pct_current")
                if atr_current is not None and len(atr_history) >= 20:
                    sorted_atr = sorted(atr_history)
                    idx = max(0, int(len(sorted_atr) * ATR_FILTER_PERCENTILE / 100) - 1)
                    threshold = sorted_atr[idx]
                    if atr_current < threshold:
                        logger.info(
                            f"🌊 Anti-chop filter: ATR%={atr_current*100:.3f}% "
                            f"< P{ATR_FILTER_PERCENTILE}={threshold*100:.3f}% "
                            f"({len(atr_history)} samples) → skipping entry (chop regime)."
                        )
                        signal = "HOLD"

            cooldown_clear = not state.get("last_stop_loss_time")
            if signal == "LONG" and cooldown_clear and free_usdt > 10.0:
                # ── FIXED FRACTIONAL SIZING ──
                usdt_to_spend = min(free_usdt, max_position) * POSITION_FRACTION
                logger.info(f"Attempting to OPEN LONG via MARKET_BUY. Investing: ${usdt_to_spend:.2f} ({POSITION_FRACTION*100:.0f}% of ${max_position:.2f} | Confidence: {confidence:.1f}%)")
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
                    send_webhook_notification("OPEN_POSITION (LIVE)", SYMBOL, fill_price,
                        f"Deployed ${usdt_to_spend:.2f} USDT",
                        {"confidence": f"{confidence:.1f}%", "signal": signal, "size": f"{POSITION_FRACTION*100:.0f}%", "wallet_after": f"${free_usdt - usdt_to_spend:.2f} USDT"})

                    state["position"] = "LONG"
                    state["entry_price"] = float(fill_price)
                    state["peak_price"] = float(fill_price)
                    state["entry_time"] = datetime.now(timezone.utc).isoformat()
                    action_taken = "BUY"
                    log_trade("BUY", float(fill_price), None, f"AI_LONG ({confidence:.0f}% conf, {POSITION_FRACTION*100:.0f}% size)", free_usdt - usdt_to_spend)
                except Exception as e:
                    logger.error(f"Failed to execute MARKET_BUY: {e}")
            elif signal == "LONG" and not cooldown_clear:
                pass  # Already logged cooldown message above
            elif signal != "LONG":
                pass  # Anti-pump overrode to HOLD, already logged
            else:
                logger.warning(f"AI signaled LONG, but free USDT (${free_usdt:.2f}) is too low to trade.")
        else:
            logger.info("Holding USDT. No entry conditions met.")
    total_evals = state.get("total_evals", 0)
    winning_trades = state.get("winning_trades", 0)
    win_rate = (winning_trades / total_evals * 100) if total_evals > 0 else 0.0
    
    if is_4h_boundary:
        state["last_signal"] = signal
        state["last_price"] = current_price
        state["last_4h_boundary"] = current_boundary_key
        save_state(state)
        logger.info(f"4H Cycle Complete. Action: {action_taken} | Win Rate: {win_rate:.1f}% ({winning_trades}/{total_evals})")
    else:
        save_state(state)
        logger.info(f"15min Poll Complete. Action: {action_taken}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        state = load_state()
        print(f"[{SYMBOL}] Live Position: {state['position']} | Entry: ${state['entry_price']} | Total Executions: {state['total_trades']}")
    elif len(sys.argv) > 1 and sys.argv[1] == "--report":
        generate_report()
    else:
        run_live_trade()
