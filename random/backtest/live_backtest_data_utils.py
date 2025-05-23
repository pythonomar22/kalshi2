# /random/backtest/live_backtest_data_utils.py

import pandas as pd
from pathlib import Path
import logging
import pytz # For robust timezone conversion
from datetime import timezone, datetime
import joblib
import json
import re

# Import from live_backtest_config
try:
    from . import live_backtest_config as live_cfg
except ImportError: # For direct script execution if needed
    import live_backtest_config as live_cfg


logger = logging.getLogger("live_backtest_data_utils")

# Cache for loaded data to speed up repeated access if feature engineering calls per market
_live_kalshi_data_cache = {}
_live_binance_data_cache = {}

def clear_live_data_caches():
    global _live_kalshi_data_cache, _live_binance_data_cache
    _live_kalshi_data_cache = {}
    _live_binance_data_cache = {}
    logger.info("Cleared live Kalshi and Binance data caches.")

def parse_market_ticker_session_key(market_ticker: str) -> str | None:
    """
    Extracts the session key (e.g., "25MAY1920") from a Kalshi market ticker.
    Example: "KXBTCD-25MAY1920-T105499.99" -> "25MAY1920"
    """
    match = re.search(r"-([0-9]{2}[A-Z]{3}[0-9]{2})([0-9]{2})-T", market_ticker)
    if match:
        return match.group(1) + match.group(2) # e.g. 25MAY19 + 20 -> 25MAY1920
    # Fallback for slightly different ticker naming if needed, though yours are consistent
    match_alt = re.search(r"-([0-9]{2}[A-Z]{3}[0-9]{4})-T", market_ticker) # e.g. 25MAY2025
    if match_alt:
        return match_alt.group(1)
    logger.warning(f"Could not parse session key from market_ticker: {market_ticker}")
    return None

def load_live_kalshi_market_data(market_ticker: str) -> pd.DataFrame | None:
    """
    Loads data for a specific Kalshi market from the live collection.
    Parses timestamps from KALSHI_RAW_DATA_TIMEZONE to UTC.
    Returns a DataFrame sorted by UTC timestamp and sequence_num.
    """
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

        # Timestamp conversion
        local_tz = pytz.timezone(live_cfg.KALSHI_RAW_DATA_TIMEZONE)
        df['timestamp_utc'] = pd.to_datetime(df['timestamp']).dt.tz_localize(local_tz).dt.tz_convert(pytz.utc)
        df['timestamp_s_utc'] = (df['timestamp_utc'].astype(int) // 10**9).astype('Int64')


        # Ensure numeric types for price and quantity, converting cents to dollars for consistency if needed
        # Your historical features used cents/100.0 for prices from Kalshi candles.
        # Live data is already in cents.
        price_cols = ['yes_bid_price_cents', 'yes_ask_price_cents', 'delta_price_cents']
        for col in price_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        qty_cols = ['yes_bid_qty', 'yes_ask_qty_on_no_side', 'delta_quantity']
        for col in qty_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.sort_values(by=['timestamp_s_utc', 'sequence_num'], inplace=True)
        df.set_index('timestamp_s_utc', inplace=True) # Set index after sorting

        _live_kalshi_data_cache[market_ticker] = df
        return df
    except Exception as e:
        logger.error(f"Error loading live Kalshi data from {filepath}: {e}", exc_info=True)
        _live_kalshi_data_cache[market_ticker] = None
        return None

def load_live_binance_data_for_session(session_key: str) -> pd.DataFrame | None:
    """
    Loads Binance kline data corresponding to a Kalshi market session.
    Uses the SESSION_TO_BINANCE_FILE_MAP.
    Timestamps are kline_start_time_ms (convert to UTC seconds).
    """
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

        # Convert kline_start_time_ms to UTC seconds timestamp
        df['timestamp_s_utc'] = (pd.to_numeric(df['kline_start_time_ms'], errors='coerce') // 1000).astype('Int64')
        df.dropna(subset=['timestamp_s_utc'], inplace=True)

        # Select and rename columns to match historical Binance data structure if needed for feature eng.
        # Historical Binance used: "open", "high", "low", "close", "volume"
        # Live Binance has: "open_price", "high_price", "low_price", "close_price", "base_asset_volume"
        df.rename(columns={
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'close_price': 'close',
            'base_asset_volume': 'volume'
        }, inplace=True)

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Keep only necessary columns, sort by timestamp, and remove duplicates from multiple receptions
        # The live Binance data has multiple rows for the same kline before it's closed.
        # We need the state of the kline as of its start time.
        # If using 'is_kline_closed' == True, you only get one record per minute, but it arrives late.
        # For "as-of" feature engineering, you want the latest available data for a kline_start_time.
        # Let's sort by reception time and keep the last received update for each kline_start_time.
        df['reception_timestamp_dt_utc'] = pd.to_datetime(df['reception_timestamp_utc'])
        df.sort_values(by=['timestamp_s_utc', 'reception_timestamp_dt_utc'], inplace=True)
        
        # Keep the last update for each kline start time
        # df_processed = df.drop_duplicates(subset=['timestamp_s_utc'], keep='last')
        # The historical data was based on official Binance klines (one per minute).
        # The live stream gives updates. For backtesting features, you'd typically use the kline *as it was known* at its start or a point in time.
        # For simplicity and to align with how historical data was likely structured (one row per kline start time):
        # We'll take the version of the kline where is_kline_closed is True, or if not available, the last update.
        # However, your historical processing `load_binance_day_data` used the raw data, implying each row was a unique minute.
        # The `btcusdt_*kline_1m.csv` files from live collection might have multiple entries per `kline_start_time_ms` before `is_kline_closed` is true.
        # Let's use the kline as of its start time, taking the most recently received data for that start time.
        df_processed = df.groupby('timestamp_s_utc').last().reset_index()


        df_processed.set_index('timestamp_s_utc', inplace=True)
        if not df_processed.index.is_monotonic_increasing:
             df_processed.sort_index(inplace=True)

        _live_binance_data_cache[session_key] = df_processed[['open', 'high', 'low', 'close', 'volume']]
        return _live_binance_data_cache[session_key]
    except Exception as e:
        logger.error(f"Error loading live Binance data from {filepath}: {e}", exc_info=True)
        _live_binance_data_cache[session_key] = None
        return None

def load_live_market_outcomes() -> pd.DataFrame | None:
    """Loads the CSV containing outcomes for live collected markets."""
    if not live_cfg.LIVE_MARKET_OUTCOMES_CSV.exists():
        logger.error(f"Live market outcomes CSV not found: {live_cfg.LIVE_MARKET_OUTCOMES_CSV}")
        return None
    try:
        df = pd.read_csv(live_cfg.LIVE_MARKET_OUTCOMES_CSV)
        # Convert ISO times to UTC timestamp_s
        df['open_time_ts_utc'] = pd.to_datetime(df['open_time_iso']).dt.tz_convert(None).astype('int64') // 10**9
        df['resolution_time_ts_utc'] = pd.to_datetime(df['close_time_iso']).dt.tz_convert(None).astype('int64') // 10**9
        
        # Map 'yes'/'no' result to 1/0 target
        df['target'] = df['result'].astype(str).str.lower().map({'yes': 1, 'no': 0})
        df.dropna(subset=['target'], inplace=True) # Ensure only markets with valid outcomes
        df['target'] = df['target'].astype(int)

        df['strike_price'] = pd.to_numeric(df['strike_price'], errors='coerce')
        
        # Keep only markets that are finalized/settled
        df = df[df['status'].isin(['finalized', 'settled'])].copy()

        logger.info(f"Loaded {len(df)} settled/finalized market outcomes from {live_cfg.LIVE_MARKET_OUTCOMES_CSV}")
        return df
    except Exception as e:
        logger.error(f"Error loading live market outcomes: {e}", exc_info=True)
        return None

def load_model_and_dependencies():
    """Loads the model, scaler, and feature names for the backtest."""
    try:
        model = joblib.load(live_cfg.MODEL_PATH)
        scaler = joblib.load(live_cfg.SCALER_PATH)
        with open(live_cfg.FEATURE_NAMES_PATH, 'r') as f:
            feature_names = json.load(f)
        logger.info(f"Successfully loaded model, scaler, and {len(feature_names)} feature names from {live_cfg.MODEL_DIR}")
        return model, scaler, feature_names
    except FileNotFoundError as e:
        logger.error(f"Error loading model/scaler/features: {e}. Paths: M={live_cfg.MODEL_PATH}, S={live_cfg.SCALER_PATH}, F={live_cfg.FEATURE_NAMES_PATH}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred loading model components: {e}", exc_info=True)
        raise

def setup_daily_trade_logger(date_str: str, log_dir: Path):
    """Sets up a daily trade logger. Copied from historical backtest_utils."""
    log_file_path = log_dir / f"{date_str}_live_trades.csv" # Distinguish from historical
    # logger_name = f"livetradelog_{date_str}" # Avoid conflict if historical runs same day
    # trade_logger = logging.getLogger(logger_name)
    # if not trade_logger.handlers: # Check if handlers are already added
    #     trade_logger.propagate = False # Stop propagation to root logger for file-only
    #     trade_logger.setLevel(logging.INFO)
    #     handler = logging.FileHandler(log_file_path, mode='w') # 'w' to overwrite for a new run
    #     # No formatter needed for CSV, just write raw lines
    #     trade_logger.addHandler(handler)
    
    # Write header if file is new or empty
    if not log_file_path.exists() or log_file_path.stat().st_size == 0:
        with open(log_file_path, 'w') as f:
            f.write("trade_execution_time_utc,market_ticker,strike_price,resolution_time_ts,decision_timestamp_s,time_to_resolution_minutes,action,predicted_prob_yes,bet_cost_cents,contracts_traded,actual_outcome_target,pnl_cents\n")
    return log_file_path # Return path for direct writing