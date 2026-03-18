#!/usr/bin/env python3
"""
Automated Paper Trading Engine for Auto-Trader.
Designed to run every 4 hours via cron on a VPS.

Usage:
    python3 paper_trader.py              # Normal execution (cron mode)
    python3 paper_trader.py --status     # Show current portfolio status
    python3 paper_trader.py --reset      # Reset portfolio to initial balance
"""
import os
import sys
import csv
import json
import logging
from datetime import datetime, timezone

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from predict import get_signal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/paper_trader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ──────────────── CONFIGURATION ────────────────
INITIAL_BALANCE = 10000.0       # USD starting capital (fictitious)
RISK_PER_TRADE = 0.03           # 3% of capital risked per trade
STOP_LOSS_PCT = 0.05            # 5% stop loss
TAKE_PROFIT_PCT = 0.07          # 7% take profit
CONFIDENCE_THRESHOLD = 55.0     # Minimum confidence % to open a position
CSV_FILE = "logs/paper_trades.csv"
STATE_FILE = "logs/paper_state.json"
# ────────────────────────────────────────────────

CSV_HEADERS = [
    "timestamp",
    "btc_price",
    "signal",
    "confidence",
    "action",
    "position_type",
    "entry_price",
    "position_size_usd",
    "pnl_trade",
    "balance",
    "prev_signal",
    "prev_price",
    "prev_correct",
    "cumulative_wins",
    "cumulative_trades",
    "win_rate",
]


def load_state():
    """Load persistent state from JSON file."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "balance": INITIAL_BALANCE,
        "position": None,         # None or {"type": "LONG/SHORT", "entry_price": X, "size_usd": Y}
        "last_signal": None,
        "last_price": None,
        "total_trades": 0,
        "winning_trades": 0,
    }


def save_state(state):
    """Save persistent state to JSON file."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def append_csv(row_dict):
    """Append a row to the trades CSV."""
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    file_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def calculate_position_size(balance, price):
    """Calculate position size in USD based on risk rules."""
    risk_amount = balance * RISK_PER_TRADE
    return risk_amount / (STOP_LOSS_PCT)  # Leveraged position sizing


def check_stop_or_take_profit(position, current_price):
    """
    Check if current price triggers stop loss or take profit.
    Returns: (should_close, pnl_pct, reason)
    """
    entry = position["entry_price"]

    if position["type"] == "LONG":
        pnl_pct = (current_price - entry) / entry
    else:  # SHORT
        pnl_pct = (entry - current_price) / entry

    if pnl_pct <= -STOP_LOSS_PCT:
        return True, pnl_pct, "STOP_LOSS"
    elif pnl_pct >= TAKE_PROFIT_PCT:
        return True, pnl_pct, "TAKE_PROFIT"

    return False, pnl_pct, None


def run_paper_trade():
    """Main paper trading logic executed every 4 hours."""
    state = load_state()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # 1. Get the current neural network signal
    logger.info("Fetching neural network signal...")
    try:
        signal_data = get_signal()
    except Exception as e:
        logger.error(f"Failed to get signal: {e}")
        return

    signal = signal_data["signal"]
    confidence = signal_data["confidence"]
    price = signal_data["price"]

    logger.info(f"Signal: {signal} | Confidence: {confidence:.2f}% | RON: ${price:,.4f}")

    # 2. Check previous prediction accuracy
    prev_correct = None
    if state["last_signal"] and state["last_price"]:
        actual_direction = "UP" if price > state["last_price"] else "DOWN"
        expected_direction = "UP" if state["last_signal"] == "LONG" else "DOWN"
        prev_correct = actual_direction == expected_direction
        state["total_trades"] += 1
        if prev_correct:
            state["winning_trades"] += 1
        emoji = "✅" if prev_correct else "❌"
        logger.info(
            f"Previous prediction: {state['last_signal']} at ${state['last_price']:,.2f} "
            f"→ Actual: {actual_direction} → {emoji}"
        )

    # 3. Manage existing position
    action = "HOLD"
    pnl_trade = 0.0
    position_type = ""
    entry_price = 0.0
    position_size = 0.0

    if state["position"]:
        position_type = state["position"]["type"]
        entry_price = state["position"]["entry_price"]
        position_size = state["position"]["size_usd"]

        should_close, pnl_pct, reason = check_stop_or_take_profit(state["position"], price)

        # Also close if signal reverses
        signal_reversal = (
            (state["position"]["type"] == "LONG" and signal == "SHORT" and confidence >= CONFIDENCE_THRESHOLD)
            or (state["position"]["type"] == "SHORT" and signal == "LONG" and confidence >= CONFIDENCE_THRESHOLD)
        )

        if should_close or signal_reversal:
            pnl_trade = state["position"]["size_usd"] * pnl_pct
            state["balance"] += pnl_trade
            close_reason = reason if reason else f"REVERSAL→{signal}"
            action = f"CLOSE_{state['position']['type']}({close_reason})"
            logger.info(
                f"Closing {state['position']['type']} position | "
                f"Entry: ${entry_price:,.2f} → Exit: ${price:,.2f} | "
                f"PnL: ${pnl_trade:,.2f} ({pnl_pct*100:.2f}%) | "
                f"Reason: {close_reason}"
            )
            state["position"] = None

    # 4. Open new position if no position and signal is strong enough
    if state["position"] is None and confidence >= CONFIDENCE_THRESHOLD:
        pos_size = calculate_position_size(state["balance"], price)
        state["position"] = {
            "type": signal,
            "entry_price": price,
            "size_usd": pos_size,
        }
        action = f"OPEN_{signal}"
        position_type = signal
        entry_price = price
        position_size = pos_size
        logger.info(
            f"Opening {signal} position | Entry: ${price:,.2f} | "
            f"Size: ${pos_size:,.2f} | Balance: ${state['balance']:,.2f}"
        )
    elif state["position"] is None and confidence < CONFIDENCE_THRESHOLD:
        action = "SKIP_LOW_CONFIDENCE"
        logger.info(f"Signal {signal} at {confidence:.1f}% below threshold {CONFIDENCE_THRESHOLD}%. Skipping.")

    # 5. Update state
    state["last_signal"] = signal
    state["last_price"] = price

    win_rate = (
        (state["winning_trades"] / state["total_trades"] * 100)
        if state["total_trades"] > 0 else 0.0
    )

    # 6. Log to CSV
    row = {
        "timestamp": now,
        "btc_price": f"{price:.2f}",
        "signal": signal,
        "confidence": f"{confidence:.2f}",
        "action": action,
        "position_type": position_type,
        "entry_price": f"{entry_price:.2f}" if entry_price else "",
        "position_size_usd": f"{position_size:.2f}" if position_size else "",
        "pnl_trade": f"{pnl_trade:.2f}",
        "balance": f"{state['balance']:.2f}",
        "prev_signal": state["last_signal"] or "",
        "prev_price": f"{state['last_price']:.2f}" if state["last_price"] else "",
        "prev_correct": str(prev_correct) if prev_correct is not None else "",
        "cumulative_wins": state["winning_trades"],
        "cumulative_trades": state["total_trades"],
        "win_rate": f"{win_rate:.1f}",
    }
    append_csv(row)

    save_state(state)

    # 7. Summary
    logger.info(
        f"Portfolio: ${state['balance']:,.2f} | "
        f"Win Rate: {win_rate:.1f}% ({state['winning_trades']}/{state['total_trades']}) | "
        f"Action: {action}"
    )


def show_status():
    """Display current portfolio status."""
    state = load_state()
    win_rate = (
        (state["winning_trades"] / state["total_trades"] * 100)
        if state["total_trades"] > 0 else 0.0
    )
    pnl_total = state["balance"] - INITIAL_BALANCE
    pnl_pct = (pnl_total / INITIAL_BALANCE) * 100

    print("\n╔══════════════════════════════════════╗")
    print("║   AUTO-TRADER PAPER TRADING STATUS   ║")
    print("╠══════════════════════════════════════╣")
    print(f"║ Balance        : ${state['balance']:>12,.2f}     ║")
    print(f"║ Initial Capital: ${INITIAL_BALANCE:>12,.2f}     ║")
    print(f"║ PnL Total      : ${pnl_total:>+12,.2f}     ║")
    print(f"║ PnL %          : {pnl_pct:>+11.2f}%     ║")
    print(f"║ Total Signals  : {state['total_trades']:>12}     ║")
    print(f"║ Predictions ✅ : {state['winning_trades']:>12}     ║")
    print(f"║ Win Rate       : {win_rate:>11.1f}%     ║")
    if state["position"]:
        pos = state["position"]
        print(f"║ Open Position  : {pos['type']:>12}     ║")
        print(f"║ Entry Price    : ${pos['entry_price']:>12,.2f}     ║")
        print(f"║ Position Size  : ${pos['size_usd']:>12,.2f}     ║")
    else:
        print("║ Open Position  :         None     ║")
    print("╚══════════════════════════════════════╝\n")


def reset_portfolio():
    """Reset the paper trading state."""
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)
    print("Portfolio reset to initial balance.")


if __name__ == "__main__":
    if "--status" in sys.argv:
        show_status()
    elif "--reset" in sys.argv:
        reset_portfolio()
    else:
        run_paper_trade()
