import os
import json
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from hybrid_model import HybridLSTMTransformer

META_FILE = "champion_meta.json"

def set_seed(seed: int):
    """Pin all RNGs for reproducible training within a seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- Configuration ---
MAX_EXECUTION_TIME = 900  # 15m (Allows safe exit before 16m hard kill)
EPOCHS = 100
BATCH_SIZE = 64
SEQ_LEN = 37
LR = 0.00076
LOG_FILE = "results.log"

# --- Backtest config (must mirror live_trader.py) ---
SIM_FEE_PER_SIDE = 0.001
SIM_STOP_LOSS = 0.03
SIM_TRAILING_ACTIVATION = 0.04
SIM_TRAILING_DROP = 0.015
SIM_ENTRY_PROB_THRESHOLD = 0.5


def simulate_pnl(probs, closes, highs):
    """
    Sequential backtest mirroring live_trader.py logic.
    probs[i]/closes[i]/highs[i] are aligned: at decision step i, the model's
    prediction is probs[i]; closes[i] is the close at that step (entry/exit
    price); highs[i] is the high of the bar that just ended (used to update
    peak for trailing stop while holding).
    Returns (pnl_pct, n_trades, win_rate_pct).
    """
    cumulative = 1.0
    position = None
    entry_price = 0.0
    peak_price = 0.0
    trailing_active = False
    trades = 0
    wins = 0

    for i in range(len(probs)):
        # 1. If holding, evaluate exits at this bar.
        if position == 'LONG':
            if highs[i] > peak_price:
                peak_price = highs[i]

            current_close = closes[i]
            running_pnl = (current_close - entry_price) / entry_price
            should_close = False
            exit_pnl = running_pnl

            if running_pnl <= -SIM_STOP_LOSS:
                should_close = True
                exit_pnl = -SIM_STOP_LOSS  # approximate fill at SL level
            else:
                peak_gain = (peak_price - entry_price) / entry_price
                if peak_gain >= SIM_TRAILING_ACTIVATION:
                    trailing_active = True
                if trailing_active:
                    drop_from_peak = (current_close - peak_price) / peak_price
                    if drop_from_peak <= -SIM_TRAILING_DROP:
                        should_close = True
                        exit_pnl = running_pnl
                if not should_close and probs[i] < SIM_ENTRY_PROB_THRESHOLD:
                    should_close = True
                    exit_pnl = running_pnl

            if should_close:
                cumulative *= (1 + exit_pnl) * (1 - SIM_FEE_PER_SIDE)
                if exit_pnl > 0:
                    wins += 1
                trades += 1
                position = None
                trailing_active = False
                peak_price = 0.0

        # 2. If flat after potential exit, evaluate entry on this step's prob.
        if position is None and probs[i] >= SIM_ENTRY_PROB_THRESHOLD:
            entry_price = closes[i]
            position = 'LONG'
            peak_price = entry_price
            trailing_active = False
            cumulative *= (1 - SIM_FEE_PER_SIDE)

    # Force-close any open position at the last available close
    if position == 'LONG':
        final_pnl = (closes[-1] - entry_price) / entry_price
        cumulative *= (1 + final_pnl) * (1 - SIM_FEE_PER_SIDE)
        if final_pnl > 0:
            wins += 1
        trades += 1

    pnl_pct = (cumulative - 1) * 100
    win_rate = (wins / trades * 100) if trades > 0 else 0.0
    return pnl_pct, trades, win_rate

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main():
    start_time = time.time()
    seed = int(os.environ.get("TRAIN_SEED", "42"))
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | Seed: {seed}")

    # 1. Load Data
    data_path = "data/processed/RONINUSDT_4h_processed.csv"
    if not os.path.exists(data_path):
        # Allow fallback for local testing
        data_path = "../data/processed/RONINUSDT_4h_processed.csv"
        
    df = pd.read_csv(data_path)

    # We use 'log_return' as target (index 0) and OHLCV + ATR as features
    features = ['log_return', 'open', 'high', 'low', 'close', 'volume', 'atr_approx_14']
    df_features = df[features].dropna()

    # Extract unnormalized log_returns for accurate binary labeling
    true_returns = df_features['log_return'].values

    # Fit normalization stats ONLY on the train portion (first 80% of rows) to avoid val leak.
    row_split = int(len(df_features) * 0.8)
    train_rows = df_features.iloc[:row_split]
    mean = train_rows.mean()
    std = train_rows.std().replace(0, 1.0)
    df_normalized = (df_features - mean) / std
    data_np = df_normalized.values

    # 2. Sequence creation & Split
    X, y = [], []
    for i in range(len(data_np) - SEQ_LEN):
        X.append(data_np[i:i+SEQ_LEN])
        # Target: 1.0 if the next true log_return is implicitly bullish (>0) else 0.0
        label = 1.0 if true_returns[i+SEQ_LEN] > 0 else 0.0
        y.append(label)
        
    X = np.array(X)
    y = np.array(y)
    
    split_idx = int(len(X) * 0.8)
    
    train_dataset = TimeSeriesDataset(X[:split_idx], y[:split_idx])
    val_dataset = TimeSeriesDataset(X[split_idx:], y[split_idx:])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Model Initialization
    input_dim = data_np.shape[1]
    model = HybridLSTMTransformer(input_dim=input_dim, d_model=64, nhead=4, 
                                  num_lstm_layers=1, num_transformer_layers=2).to(device)
    
    # Binary Cross Entropy with Logits for stable classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # 4. Training Loop with Time Budgets
    best_val_loss = float('inf')
    early_exit = False

    print("Starting training...")
    for epoch in range(EPOCHS):
        # Time Enforcement check
        elapsed = time.time() - start_time
        if elapsed > MAX_EXECUTION_TIME:
            print(f"Time limit reached! ({elapsed:.2f}s). Exiting gracefully.")
            early_exit = True
            break
            
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X).view(-1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Mid-epoch timeout check (for large datasets)
            if time.time() - start_time > MAX_EXECUTION_TIME:
                early_exit = True
                break

        if early_exit:
            break

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).view(-1)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "best_model.pth")
            # Persist all metadata needed for inference alongside the weights
            meta = {
                "seq_len": SEQ_LEN,
                "batch_size": BATCH_SIZE,
                "feature_list": features,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "input_dim": input_dim,
                "d_model": 64,
                "nhead": 4,
                "num_lstm_layers": 1,
                "num_transformer_layers": 2,
            }
            with open(META_FILE, "w") as f_meta:
                json.dump(meta, f_meta, indent=2)
            print(f"Saved new best model with Val Loss: {avg_val:.4f}")
            
        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Elapsed: {time.time()-start_time:.1f}s")

    # 5. Backtest the best model on the validation set with realistic fees + trailing
    val_bpb = best_val_loss

    val_pnl_pct = 0.0
    val_trades = 0
    val_win_rate = 0.0
    try:
        # Reload best weights for evaluation (last epoch may not be the best)
        best_state = torch.load("best_model.pth", map_location=device, weights_only=True)
        eval_model = HybridLSTMTransformer(
            input_dim=input_dim, d_model=64, nhead=4,
            num_lstm_layers=1, num_transformer_layers=2
        ).to(device)
        eval_model.load_state_dict(best_state)
        eval_model.eval()

        # Predictions over the entire val set (one shot, batch_size doesn't matter)
        X_val_tensor = torch.tensor(X[split_idx:], dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = eval_model(X_val_tensor).view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

        # Aligned price arrays for the val period.
        # Sequence j in val (j in 0..n_val-1) corresponds to global row split_idx + j + SEQ_LEN - 1
        # which is the bar that just ended at decision time j.
        all_close = df_features['close'].values
        all_high = df_features['high'].values
        n_val = len(probs)
        val_start = split_idx + SEQ_LEN - 1
        sim_closes = all_close[val_start: val_start + n_val]
        sim_highs = all_high[val_start: val_start + n_val]

        if len(sim_closes) == n_val:
            val_pnl_pct, val_trades, val_win_rate = simulate_pnl(probs, sim_closes, sim_highs)
        else:
            print(f"⚠ Sim alignment off: probs={n_val}, closes={len(sim_closes)} — skipping.")
    except Exception as e:
        print(f"⚠ Backtest failed: {e}")

    # Composite fitness (Option B): PnL when the model actually trades; BCE-based
    # fallback otherwise so non-trading models can still be ranked by predictive skill.
    if val_trades > 0:
        val_fitness = val_pnl_pct
    else:
        val_fitness = -val_bpb * 100  # range ~ -70..-60, always worse than any positive PnL

    with open(LOG_FILE, "a") as f:
        f.write(f"val_bpb: {val_bpb:.6f}\n")
        f.write(f"val_pnl_pct: {val_pnl_pct:.6f}\n")
        f.write(f"val_trades: {val_trades}\n")
        f.write(f"val_win_rate: {val_win_rate:.2f}\n")
        f.write(f"val_fitness: {val_fitness:.6f}\n")

    print(f"Training finalized. val_bpb={val_bpb:.4f} | val_pnl={val_pnl_pct:+.2f}% over {val_trades} trades (win {val_win_rate:.1f}%) | fitness={val_fitness:+.4f}")

if __name__ == "__main__":
    main()
