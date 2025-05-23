# /notebooks/backtest/backtest_utils.py

import pandas as pd
import joblib
import json
import logging
import glob
import os
from pathlib import Path
import datetime as dt

from backtest import backtest_config as config 

def setup_daily_trade_logger(date_str: str):
    log_file_path = config.LOG_DIR / f"{date_str}_trades.csv"
    logger_name = f"tradelog_{date_str}"
    trade_logger = logging.getLogger(logger_name)
    if not trade_logger.handlers:
        trade_logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file_path, mode='w')
        trade_logger.addHandler(handler)
        if not log_file_path.exists() or os.path.getsize(log_file_path) == 0:
             with open(log_file_path, 'w') as f: # Ensure header is written
                f.write("trade_execution_time_utc,market_ticker,strike_price,resolution_time_ts,decision_timestamp_s,time_to_resolution_minutes,action,predicted_prob_yes,bet_cost_cents,contracts_traded,kalshi_outcome_target,pnl_cents\n")
    return trade_logger, log_file_path # Return path for direct writing

def load_model_and_dependencies():
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

def load_features_for_backtest():
    list_of_feature_files = sorted(
        glob.glob(config.LATEST_FEATURES_CSV_GLOB_PATTERN), # Uses updated pattern
        key=os.path.getctime,
        reverse=True
    )
    if not list_of_feature_files:
        logging.error(f"No feature CSV files found: {config.LATEST_FEATURES_CSV_GLOB_PATTERN}")
        raise FileNotFoundError("No per-minute feature CSV files found.")
    
    latest_features_path = Path(list_of_feature_files[0])
    logging.info(f"Loading PER-MINUTE features for backtest from: {latest_features_path}")
    try:
        features_df = pd.read_csv(latest_features_path, low_memory=False)
        
        # Essential columns for per-minute structure
        essential_cols = ['market_ticker', 'decision_timestamp_s', 'resolution_time_ts', 'target', 'strike_price', 'time_to_resolution_minutes']
        if any(col not in features_df.columns for col in essential_cols):
            missing = [col for col in essential_cols if col not in features_df.columns]
            logging.error(f"Essential columns missing from features_df: {missing}")
            raise ValueError(f"Per-minute features_df missing essential columns: {missing}")

        # Ensure correct types for timestamp columns
        for ts_col in ['decision_timestamp_s', 'resolution_time_ts']:
            features_df[ts_col] = pd.to_numeric(features_df[ts_col], errors='coerce').astype('Int64')
        
        features_df.dropna(subset=['decision_timestamp_s', 'resolution_time_ts'], inplace=True)
        logging.info(f"Loaded {len(features_df)} rows from per-minute feature data.")
        return features_df
    except Exception as e:
        logging.error(f"Error loading features from {latest_features_path}: {e}", exc_info=True)
        raise

# get_features_for_market_at_decision_time IS NO LONGER NEEDED with the new feature structure.
# Each row of the loaded features_df *is* the set of features for a decision point.