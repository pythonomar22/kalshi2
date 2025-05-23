# /Users/omarabul-hassan/Desktop/projects/kalshi/random/backtest/live_backtest_utils.py

import pandas as pd
import joblib
import json
import logging
import glob
import os
from pathlib import Path
import datetime as dt
from datetime import timezone
import re # Ensure re is imported

import live_backtest_config as config

# --- General Utilities (can be similar to historical) ---

def setup_daily_trade_logger(date_str: str, log_dir_path: Path = None, custom_header: str = None): # Added custom_header
    """Sets up a daily logger for trades, saving to CSV files."""
    if log_dir_path is None:
        log_dir_path = config.LOG_DIR

    log_file_path = log_dir_path / f"{date_str}_trades_live.csv" 
    logger_name = f"tradelog_live_{date_str}"
    trade_logger = logging.getLogger(logger_name)

    if not trade_logger.handlers:
        trade_logger.setLevel(logging.INFO) # Set level on logger itself
        # The handler is only for file output, not console for this specific logger
        # handler = logging.FileHandler(log_file_path, mode='w') # 'w' for a fresh log per run
        # trade_logger.addHandler(handler) # Don't add handler here, engine will write directly
        pass # Engine will handle file opening and writing

    # Write header if file is new or empty, or if custom_header is provided and different
    # This part is now handled by the engine to ensure dynamic header is written once
    if not log_file_path.exists() or os.path.getsize(log_file_path) == 0:
        if custom_header:
            header_to_write = custom_header
        else: # Default basic header if no custom one is provided
            header_to_write = ("trade_execution_time_utc,market_ticker,strike_price,resolution_time_ts,"
                               "decision_timestamp_s,time_to_resolution_minutes,action,predicted_prob_yes,"
                               "bet_cost_cents,contracts_traded,kalshi_outcome_target,pnl_cents\n")
        try:
            with open(log_file_path, 'w') as f: 
                f.write(header_to_write)
        except Exception as e:
            logging.error(f"Error writing header to trade log {log_file_path}: {e}")
                
    return trade_logger, log_file_path # Return path for direct writing by engine


def load_model_and_dependencies():
    """Loads the pre-trained model, scaler, and feature names."""
    try:
        model = joblib.load(config.MODEL_PATH)
        scaler = joblib.load(config.SCALER_PATH)
        with open(config.FEATURE_NAMES_PATH, 'r') as f:
            feature_names = json.load(f)
        logging.info(f"Successfully loaded model from {config.MODEL_PATH}, scaler, and {len(feature_names)} feature names.")
        return model, scaler, feature_names
    except FileNotFoundError as e:
        logging.error(f"Error loading model/scaler/features: {e}. Paths: M={config.MODEL_PATH}, S={config.SCALER_PATH}, F={config.FEATURE_NAMES_PATH}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred loading model components: {e}", exc_info=True)
        raise

def load_live_features_for_backtest():
    """Loads the latest CSV file containing features engineered from LIVE data."""
    list_of_feature_files = sorted(
        glob.glob(config.LATEST_LIVE_FEATURES_CSV_GLOB_PATTERN),
        key=os.path.getctime,
        reverse=True
    )
    if not list_of_feature_files:
        logging.error(f"No LIVE feature CSV files found: {config.LATEST_LIVE_FEATURES_CSV_GLOB_PATTERN}")
        raise FileNotFoundError("No live-engineered feature CSV files found. Run live_feature_engineering.py first.")

    latest_features_path = Path(list_of_feature_files[0])
    logging.info(f"Loading LIVE features for backtest from: {latest_features_path}")
    try:
        features_df = pd.read_csv(latest_features_path, low_memory=False)

        essential_cols = ['market_ticker', 'decision_timestamp_s', 'resolution_time_ts', 'target', 'strike_price', 'time_to_resolution_minutes']
        missing = [col for col in essential_cols if col not in features_df.columns]
        if missing:
            logging.error(f"Essential columns missing from live features_df: {missing}")
            raise ValueError(f"Live features_df missing essential columns: {missing}")

        for ts_col in ['decision_timestamp_s', 'resolution_time_ts']:
            features_df[ts_col] = pd.to_numeric(features_df[ts_col], errors='coerce').astype('Int64')

        features_df.dropna(subset=['decision_timestamp_s', 'resolution_time_ts'], inplace=True)
        logging.info(f"Loaded {len(features_df)} rows from live-engineered feature data.")
        return features_df
    except Exception as e:
        logging.error(f"Error loading live features from {latest_features_path}: {e}", exc_info=True)
        raise

def parse_live_kalshi_ticker_details(ticker_string: str) -> dict | None:
    if not ticker_string: return None
    match = re.match(r"^(.*?)-([A-Z0-9]+(?:[A-Z]{3}[0-9]{2}[0-9]{2}))-T(\d+\.?\d*)$", ticker_string)
    if not match:
        logging.warning(f"Could not parse ticker: {ticker_string} with primary pattern.")
        return None
    
    groups = match.groups()
    try:
        series = groups[0]
        session_id = groups[1] 
        strike_price = float(groups[2])
        return {"series": series, "session_id": session_id, "strike_price": strike_price}
    except Exception as e:
        logging.error(f"Error parsing components from ticker {ticker_string}: {e}")
        return None

def parse_iso_to_unix_timestamp(ds: str | None) -> int | None:
    if not ds: return None
    try:
        if 'Z' in ds:
            dt_obj = dt.datetime.fromisoformat(ds.replace('Z', '+00:00'))
        elif '+' in ds.split('T')[1] or '-' in ds.split('T')[1] and ':' in ds.split('T')[1]:
             dt_obj = dt.datetime.fromisoformat(ds)
        else: 
            dt_obj = dt.datetime.fromisoformat(ds)
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        return int(dt_obj.astimezone(timezone.utc).timestamp())
    except ValueError as ve:
        logging.warning(f"Could not parse ISO timestamp string '{ds}': {ve}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error parsing ISO timestamp '{ds}': {e}")
        return None

_binance_live_data_cache = {}

def load_and_preprocess_live_binance_data(binance_file_path: Path) -> pd.DataFrame | None:
    global _binance_live_data_cache
    if binance_file_path in _binance_live_data_cache:
        return _binance_live_data_cache[binance_file_path]

    if not binance_file_path.exists():
        logging.warning(f"Binance live data file not found: {binance_file_path}")
        _binance_live_data_cache[binance_file_path] = None
        return None
    try:
        df = pd.read_csv(binance_file_path)
        if df.empty:
            _binance_live_data_cache[binance_file_path] = pd.DataFrame()
            return pd.DataFrame()

        df['reception_timestamp_dt'] = pd.to_datetime(df['reception_timestamp_utc'], errors='coerce')
        df['kline_start_time_s'] = (df['kline_start_time_ms'] // 1000).astype('Int64')

        numeric_cols = ['open_price', 'close_price', 'high_price', 'low_price', 'base_asset_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['kline_start_time_s', 'reception_timestamp_dt', 'close_price'], inplace=True)
        
        df.sort_values(
            by=['kline_start_time_s', 'is_kline_closed', 'reception_timestamp_dt'],
            ascending=[True, False, False], 
            inplace=True
        )
        
        processed_df = df.drop_duplicates(subset=['kline_start_time_s'], keep='first')
        
        processed_df.set_index('kline_start_time_s', inplace=True)
        if not processed_df.index.is_monotonic_increasing:
             processed_df.sort_index(inplace=True)
        
        _binance_live_data_cache[binance_file_path] = processed_df
        # logging.info(f"Processed live Binance data from {binance_file_path.name}, {len(processed_df)} unique klines.") # Can be verbose
        return processed_df
    except Exception as e:
        logging.error(f"Error processing live Binance data from {binance_file_path}: {e}", exc_info=True)
        _binance_live_data_cache[binance_file_path] = None
        return None

def clear_binance_live_cache():
    global _binance_live_data_cache
    _binance_live_data_cache = {}
    logging.info("Cleared live Binance data cache.")