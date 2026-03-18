import os
import requests
import pandas as pd
import logging
from datetime import datetime
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class DataPreparer:
    def __init__(self, data_dir: str = "data"):
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def download_binance_klines(self, symbol: str, interval: str, limit: int = 1000) -> Optional[str]:
        """
        Downloads OHLCV data from Binance Public API.
        """
        urls = [
            "https://data-api.binance.vision/api/v3/klines",
            "https://api.binance.com/api/v3/klines",
        ]
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        logger.info(f"Downloading {limit} klines for {symbol} at {interval} interval...")
        
        response = None
        for url in urls:
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                break  # Success, stop trying
            except requests.exceptions.RequestException:
                logger.warning(f"API {url} failed for {symbol}, trying next...")
                continue
        
        if response is None or response.status_code != 200:
            logger.error(f"All API endpoints failed for {symbol}")
            return None
        
        try:
            data = response.json()
            
            # Binance kline format
            columns = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
            
            df = pd.DataFrame(data, columns=columns)
            
            # Format and clean data
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            # Keep only essential columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            output_file = os.path.join(self.raw_dir, f"{symbol}_{interval}_raw.csv")
            df.to_csv(output_file, index=False)
            logger.info(f"Raw data saved to {output_file}")
            
            return output_file
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading data: {e}")
            return None

    def create_features(self, raw_file_path: str) -> Optional[str]:
        """
        Processes raw data, creates minimal features, and formats for model training.
        """
        if not os.path.exists(raw_file_path):
            logger.error(f"Raw file {raw_file_path} not found.")
            return None
            
        try:
            df = pd.read_csv(raw_file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort chronologically just in case
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Processing {len(df)} rows from {raw_file_path}...")
            
            # --- Feature Engineering ---
            # 1. Log returns (as recommended in the research)
            # Helps to normalize and make series stationary
            import numpy as np
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            
            # 2. Volatility Proxy (ATR simplified)
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['close'].shift(1))
            df['tr3'] = abs(df['low'] - df['close'].shift(1))
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr_approx_14'] = df['true_range'].rolling(window=14).mean()
            
            # Clean up intermediate cols
            df = df.drop(columns=['tr1', 'tr2', 'tr3', 'true_range'])
            
            # Drop NaNs created by shift/rolling
            df = df.dropna()
            
            # The model should mostly rely on raw OHLCV and Log Returns, 
            # avoiding too many traditional indicators that cause redundancy.
            
            base_name = os.path.basename(raw_file_path).replace("_raw.csv", "")
            output_file = os.path.join(self.processed_dir, f"{base_name}_processed.csv")
            
            df.to_csv(output_file, index=False)
            logger.info(f"Processed features saved to {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error processing features: {e}")
            return None

if __name__ == "__main__":
    preparer = DataPreparer()
    
    # 1. Download BTC data (e.g., 4h interval for swing trading focus)
    symbol = "RONINUSDT"
    interval = "4h"
    raw_file = preparer.download_binance_klines(symbol, interval, limit=1000)
    
    if raw_file:
        # 2. Process features
        processed_file = preparer.create_features(raw_file)
        
        # 3. Next Step: In the main orchestrator, we would use bridge_utils
        # to upload this `processed_file` to the Vast.ai instance:
        # bridge.sync_push(processed_file, "data/")
        logger.info("Data preparation complete. Ready for remote sync.")
