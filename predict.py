import os
import json
import logging
import torch
import pandas as pd

from core.prepare import DataPreparer
from models.hybrid_model import HybridLSTMTransformer

logger = logging.getLogger(__name__)

MODEL_PATH = "models/champion_model.pth"
META_PATH = "models/champion_meta.json"
DEFAULT_FEATURES = ['log_return', 'open', 'high', 'low', 'close', 'volume', 'atr_approx_14']

def load_champion_meta():
    """
    Load metadata saved alongside the champion model. If missing, fall back to
    re-deriving stats from the training CSV (legacy behavior, kept for safety).
    """
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            meta = json.load(f)
        meta["mean"] = torch.tensor(meta["mean"], dtype=torch.float32)
        meta["std"] = torch.tensor(meta["std"], dtype=torch.float32)
        return meta

    logger.warning(f"{META_PATH} not found — falling back to legacy stats from training CSV.")
    # Legacy fallback: recompute from processed training CSV using train-split only.
    from models.train import SEQ_LEN, BATCH_SIZE  # only imported in fallback path
    processed_file = "data/processed/RONINUSDT_4h_processed.csv"
    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"Cannot find {processed_file} and no {META_PATH}")
    df = pd.read_csv(processed_file)
    features = df[DEFAULT_FEATURES].dropna().values
    train_size = int(len(features) * 0.8)
    train_t = torch.tensor(features[:train_size], dtype=torch.float32)
    return {
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "feature_list": DEFAULT_FEATURES,
        "mean": train_t.mean(dim=0),
        "std": train_t.std(dim=0),
        "input_dim": len(DEFAULT_FEATURES),
        "d_model": 64,
        "nhead": 4,
        "num_lstm_layers": 1,
        "num_transformer_layers": 2,
    }

def get_signal():
    """
    Core inference function. Returns a dict with signal data.
    Used by both the interactive predict.py and the automated paper_trader.py.
    """
    meta = load_champion_meta()
    seq_len = meta["seq_len"]
    features_list = meta["feature_list"]
    mean = meta["mean"]
    std = meta["std"]

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

    if len(df_live) < seq_len:
        raise ValueError(f"Not enough live data fetched! Need at least {seq_len} rows.")

    latest_features = df_live[features_list].tail(seq_len).values
    tensor_features = torch.tensor(latest_features, dtype=torch.float32)

    normalized_features = (tensor_features - mean) / (std + 1e-8)
    input_sequence = normalized_features.unsqueeze(0)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Make sure to download {MODEL_PATH} first!")

    model = HybridLSTMTransformer(
        input_dim=meta["input_dim"],
        d_model=meta["d_model"],
        nhead=meta["nhead"],
        num_lstm_layers=meta["num_lstm_layers"],
        num_transformer_layers=meta["num_transformer_layers"],
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True))
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
    
    # Volatility snapshot for the anti-chop filter in live_trader.py
    atr_pct_series = (df_live['atr_approx_14'] / df_live['close']).dropna().tail(100)
    atr_pct_history = atr_pct_series.tolist()
    atr_pct_current = atr_pct_history[-1] if atr_pct_history else None

    return {
        "signal": signal,
        "confidence": confidence,
        "probability_up": probability_up,
        "price": current_price,
        "raw_logit": raw_logit,
        "atr_pct_current": atr_pct_current,
        "atr_pct_history": atr_pct_history,
    }

def predict_latest_market():
    print("=== RON/USDT Auto-Trader Inference Engine ===")
    meta = load_champion_meta()
    print(f"Loading champion model with SEQ_LEN: {meta['seq_len']}")

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

