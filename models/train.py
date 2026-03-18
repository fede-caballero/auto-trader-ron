import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from hybrid_model import HybridLSTMTransformer

# --- Configuration ---
MAX_EXECUTION_TIME = 900  # 15m (Allows safe exit before 16m hard kill)
EPOCHS = 100
BATCH_SIZE = 64
SEQ_LEN = 26
LR = 0.00124
LOG_FILE = "results.log"

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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
    
    # Normalize features (Standardization)
    mean = df_features.mean()
    std = df_features.std()
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
            print(f"Saved new best model with Val Loss: {avg_val:.4f}")
            
        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Elapsed: {time.time()-start_time:.1f}s")

    # 5. Reflection Logging (Distributed Karpathy Loop)
    # The tracker.py script will read this log file.
    
    # We log the best bits-per-byte (or MSE loss in this regression case)
    val_bpb = best_val_loss # For continuous matching, we log it under the expected variable name
    
    with open(LOG_FILE, "a") as f:
        f.write(f"val_bpb: {val_bpb:.6f}\n")
        
    print(f"Training finalized. Logged val_bpb: {val_bpb:.6f}")

if __name__ == "__main__":
    main()
