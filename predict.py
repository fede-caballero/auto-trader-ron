import os
import sys
import torch
import pandas as pd
import numpy as np

from core.prepare import DataPreparer
from models.hybrid_model import HybridLSTMTransformer

# When we import train.py, it expects hybrid_model to be in the same path
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
from models.train import SEQ_LEN, BATCH_SIZE

def get_normalization_params():
    """
    Recalculates the exact mean and std used during training.
    In a production system, these should be saved to a file during train.py!
    """
    processed_file = "data/processed/RONINUSDT_4h_processed.csv"
    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"Cannot find {processed_file}")
        
    df = pd.read_csv(processed_file)
    features_list = ['log_return', 'open', 'high', 'low', 'close', 'volume', 'atr_approx_14']
    features = df[features_list].values
    
    # 80% split logic used in train.py
    train_size = int(len(features) * 0.8)
    train_data = features[:train_size]
    
    # Convert to PyTorch tensors to exactly mirror train.py math
    train_data_t = torch.tensor(train_data, dtype=torch.float32)
    mean = train_data_t.mean(dim=0)
    std = train_data_t.std(dim=0)
    
    return mean, std

def get_signal():
    """
    Core inference function. Returns a dict with signal data.
    Used by both the interactive predict.py and the automated paper_trader.py.
    """
    mean, std = get_normalization_params()
    
    preparer = DataPreparer()
    preparer.raw_dir = "data/raw"
    preparer.processed_dir = "data/processed"
    
    raw_file = "data/raw/RONINUSDT_4h_live_raw.csv"
    processed_file = "data/processed/RONINUSDT_4h_live_processed.csv"
    
    out_raw = preparer.download_binance_klines("RONINUSDT", "4h", limit=200)
    if os.path.exists(out_raw):
        os.rename(out_raw, raw_file)
    
    out_processed = preparer.create_features(raw_file)
    if os.path.exists(out_processed):
        os.rename(out_processed, processed_file)
    
    df_live = pd.read_csv(processed_file)
    
    if len(df_live) < SEQ_LEN:
        raise ValueError(f"Not enough live data fetched! Need at least {SEQ_LEN} rows.")
    
    features_list = ['log_return', 'open', 'high', 'low', 'close', 'volume', 'atr_approx_14']
    latest_features = df_live[features_list].tail(SEQ_LEN).values
    tensor_features = torch.tensor(latest_features, dtype=torch.float32)
    
    normalized_features = (tensor_features - mean) / (std + 1e-8)
    input_sequence = normalized_features.unsqueeze(0)
    
    model_path = "models/champion_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Make sure to download {model_path} first!")
    
    model = HybridLSTMTransformer(
        input_dim=len(features_list), 
        d_model=64, 
        nhead=4, 
        num_lstm_layers=1,
        num_transformer_layers=2
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    
    with torch.no_grad():
        raw_logit = model(input_sequence).squeeze().item()
    
    probability_up = torch.sigmoid(torch.tensor(raw_logit)).item()
    current_price = df_live['close'].iloc[-1]
    
    if probability_up > 0.5:
        signal = "LONG"
        confidence = probability_up * 100
    else:
        signal = "SHORT"
        confidence = (1 - probability_up) * 100
    
    return {
        "signal": signal,
        "confidence": confidence,
        "probability_up": probability_up,
        "price": current_price,
        "raw_logit": raw_logit,
    }

def predict_latest_market():
    print("=== RON/USDT Auto-Trader Inference Engine ===")
    print(f"Loading champion model with optimized SEQ_LEN: {SEQ_LEN}")
    
    print("1. Extracting historical mean & std for normalization...")
    print("2. Fetching current LIVE market data from Binance...")
    
    result = get_signal()
    
    print("3. Normalizing live sequence...")
    print("4. Loading neural network weights...")
    
    print("\n--- INFERENCE RESULTS ---")
    print(f"Current RON/USDT Price  : ${result['price']:,.4f}")
    if result['signal'] == "LONG":
        print(f"Network Signal          : 🟢 LONG (BULLISH)")
    else:
        print(f"Network Signal          : 🔴 SHORT (BEARISH)")
    print(f"Confidence Level        : {result['confidence']:.2f}%")
    print("=================================")

if __name__ == "__main__":
    predict_latest_market()

