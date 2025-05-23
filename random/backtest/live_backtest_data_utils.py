# /random/backtest/live_backtest_data_utils.py

import pandas as pd
from pathlib import Path
import logging
import pytz 
from datetime import timezone, datetime 
import datetime as dt 
import joblib
import json
import re

import live_backtest_config as live_cfg

logger = logging.getLogger("live_backtest_data_utils")

_live_kalshi_data_cache = {}
_live_binance_data_cache = {}

def clear_live_data_caches():
    global _live_kalshi_data_cache, _live_binance_data_cache
    _live_kalshi_data_cache = {}
    _live_binance_data_cache = {}
    logger.info("Cleared live Kalshi and Binance data caches.")

def parse_market_ticker_session_key(market_ticker: str) -> str | None:
    match = re.search(r"-([0-9]{2}[A-Z]{3}[0-9]{2})([0-9]{2})-T", market_ticker)
    if match:
        return match.group(1) + match.group(2)
    match_alt = re.search(r"-([0-9]{2}[A-Z]{3}[0-9]{4})-T", market_ticker)
    if match_alt:
        return match_alt.group(1)
    logger.warning(f"Could not parse session key from market_ticker: {market_ticker}")
    return None

def load_live_kalshi_market_data(market_ticker: str) -> pd.DataFrame | None:
    if market_ticker in _live_kalshi_data_cache:
        return _live_kalshi_data_cache[market_ticker]

    filepath = live_cfg.LIVE_KALSHI_DATA_DIR / f"{market_ticker}.csv"
    if not filepath.exists():
        logger.warning(f"Live Kalshi data file not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            logger.warning(f"Live Kalshi data file is empty: {filepath}")
            _live_kalshi_data_cache[market_ticker] = None
            return None
        local_tz = pytz.timezone(live_cfg.KALSHI_RAW_DATA_TIMEZONE)
        # Assuming df['timestamp'] does not have 'Z' and is naive local time
        df['timestamp_utc'] = pd.to_datetime(df['timestamp']).dt.tz_localize(local_tz).dt.tz_convert(pytz.utc)
        df['timestamp_s_utc'] = (df['timestamp_utc'].astype(int) // 10**9).astype('Int64')
        price_cols = ['yes_bid_price_cents', 'yes_ask_price_cents', 'delta_price_cents']
        for col in price_cols:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        qty_cols = ['yes_bid_qty', 'yes_ask_qty_on_no_side', 'delta_quantity']
        for col in qty_cols:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.sort_values(by=['timestamp_s_utc', 'sequence_num'], inplace=True)
        df.set_index('timestamp_s_utc', inplace=True) 
        _live_kalshi_data_cache[market_ticker] = df
        return df
    except Exception as e:
        logger.error(f"Error loading live Kalshi data from {filepath}: {e}", exc_info=True)
        _live_kalshi_data_cache[market_ticker] = None
        return None

def load_live_binance_data_for_session(session_key: str) -> pd.DataFrame | None:
    if session_key in _live_binance_data_cache:
        return _live_binance_data_cache[session_key]
    binance_filename = live_cfg.SESSION_TO_BINANCE_FILE_MAP.get(session_key)
    if not binance_filename:
        logger.error(f"No Binance file mapping found for session key: {session_key}")
        return None
    filepath = live_cfg.LIVE_BINANCE_DATA_DIR / binance_filename
    if not filepath.exists():
        logger.warning(f"Live Binance data file not found: {filepath} for session {session_key}")
        return None
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            logger.warning(f"Live Binance data file is empty: {filepath}")
            _live_binance_data_cache[session_key] = None
            return None
        df['timestamp_s_utc'] = (pd.to_numeric(df['kline_start_time_ms'], errors='coerce') // 1000).astype('Int64')
        df.dropna(subset=['timestamp_s_utc'], inplace=True)
        df.rename(columns={'open_price':'open', 'high_price':'high', 'low_price':'low', 'close_price':'close', 'base_asset_volume':'volume'}, inplace=True)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['reception_timestamp_dt_utc'] = pd.to_datetime(df['reception_timestamp_utc'])
        df.sort_values(by=['timestamp_s_utc', 'reception_timestamp_dt_utc'], inplace=True)
        df_processed = df.groupby('timestamp_s_utc').last().reset_index()
        df_processed.set_index('timestamp_s_utc', inplace=True)
        if not df_processed.index.is_monotonic_increasing: df_processed.sort_index(inplace=True)
        _live_binance_data_cache[session_key] = df_processed[['open', 'high', 'low', 'close', 'volume']]
        return _live_binance_data_cache[session_key]
    except Exception as e:
        logger.error(f"Error loading live Binance data from {filepath}: {e}", exc_info=True)
        _live_binance_data_cache[session_key] = None
        return None

def load_live_market_outcomes() -> pd.DataFrame | None:
    if not live_cfg.LIVE_MARKET_OUTCOMES_CSV.exists():
        logger.error(f"Live market outcomes CSV not found: {live_cfg.LIVE_MARKET_OUTCOMES_CSV}")
        return None
    try:
        df = pd.read_csv(live_cfg.LIVE_MARKET_OUTCOMES_CSV)
        
        # Convert ISO times to UTC pandas Datetime objects first
        # The 'Z' in your ISO strings means they are already UTC.
        df['open_time_dt_utc'] = pd.to_datetime(df['open_time_iso'], utc=True)
        df['close_time_dt_utc'] = pd.to_datetime(df['close_time_iso'], utc=True)

        # Now convert to Unix timestamp (seconds)
        df['open_time_ts_utc'] = (df['open_time_dt_utc'].astype('int64') // 10**9).astype('Int64')
        df['resolution_time_ts_utc'] = (df['close_time_dt_utc'].astype('int64') // 10**9).astype('Int64')
        
        df['target'] = df['result'].astype(str).str.lower().map({'yes': 1, 'no': 0})
        df.dropna(subset=['target', 'open_time_ts_utc', 'resolution_time_ts_utc'], inplace=True) # Also drop if timestamp conversion failed
        df['target'] = df['target'].astype(int)

        df['strike_price'] = pd.to_numeric(df['strike_price'], errors='coerce')
        df = df[df['status'].isin(['finalized', 'settled'])].copy()
        
        # Drop intermediate dt columns if not needed further
        df.drop(columns=['open_time_dt_utc', 'close_time_dt_utc'], inplace=True, errors='ignore')

        logger.info(f"Loaded {len(df)} settled/finalized market outcomes from {live_cfg.LIVE_MARKET_OUTCOMES_CSV}")
        return df
    except Exception as e:
        logger.error(f"Error loading live market outcomes: {e}", exc_info=True)
        return None

def load_model_and_dependencies():
    try:
        model = joblib.load(live_cfg.MODEL_PATH)
        scaler = joblib.load(live_cfg.SCALER_PATH)
        with open(live_cfg.FEATURE_NAMES_PATH, 'r') as f: feature_names = json.load(f)
        logger.info(f"Successfully loaded model, scaler, and {len(feature_names)} feature names from {live_cfg.MODEL_DIR}")
        return model, scaler, feature_names
    except FileNotFoundError as e:
        logger.error(f"Error loading model/scaler/features: {e}. Paths: M={live_cfg.MODEL_PATH}, S={live_cfg.SCALER_PATH}, F={live_cfg.FEATURE_NAMES_PATH}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred loading model components: {e}", exc_info=True)
        raise

def setup_hourly_trade_logger(decision_time_utc: dt.datetime, base_log_dir: Path) -> Path:
    date_str = decision_time_utc.strftime("%Y-%m-%d")
    hour_str = decision_time_utc.strftime("%H")
    date_specific_log_dir = base_log_dir / date_str
    date_specific_log_dir.mkdir(parents=True, exist_ok=True) 
    log_file_path = date_specific_log_dir / f"{hour_str}_live_trades.csv"
    if not log_file_path.exists() or log_file_path.stat().st_size == 0:
        with open(log_file_path, 'w') as f:
            f.write("trade_execution_time_utc,market_ticker,strike_price,resolution_time_ts,decision_timestamp_s,time_to_resolution_minutes,action,predicted_prob_yes,bet_cost_cents_per_contract,contracts_traded,actual_outcome_target,pnl_cents,current_capital_before_trade_cents,kelly_fraction_f_star,capital_to_risk_cents,trade_value_cents\n")
    return log_file_path