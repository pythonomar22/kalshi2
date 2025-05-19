# utils.py
import pandas as pd
import os
from pathlib import Path
import re
import datetime as dt
from datetime import timezone, timedelta
import logging
import numpy as np

logger = logging.getLogger(__name__)

# These will be set by backtest_v2.py or other calling scripts
BASE_PROJECT_DIR = Path.cwd() # Default, will be overridden
BINANCE_FLAT_DATA_DIR = BASE_PROJECT_DIR / "binance_data" # Default
KALSHI_DATA_DIR = BASE_PROJECT_DIR / "kalshi_data" # Default

# --- Feature Calculation Parameters (Defaults, can be overridden by backtest_v2.py) ---
# These need to match what the model was trained with.
BTC_MOMENTUM_WINDOWS = [5, 10, 15, 30]
BTC_VOLATILITY_WINDOW = 15
BTC_SMA_WINDOWS = [10, 30]
BTC_EMA_WINDOWS = [12, 26]
BTC_RSI_WINDOW = 14

# --- Caches ---
_binance_daily_raw_cache = {} # Cache for raw daily Binance CSVs
_binance_range_with_features_cache = None # Cache for a pre-calculated range
_cache_range_start_dt = None
_cache_range_end_dt = None

_kalshi_market_minute_data_cache = {}


def clear_binance_cache():
    global _binance_daily_raw_cache, _binance_range_with_features_cache
    global _cache_range_start_dt, _cache_range_end_dt
    _binance_daily_raw_cache = {}
    _binance_range_with_features_cache = None
    _cache_range_start_dt = None
    _cache_range_end_dt = None
    logger.info("Utils: All Binance data caches cleared.")

def clear_kalshi_cache():
    global _kalshi_market_minute_data_cache
    _kalshi_market_minute_data_cache = {}
    logger.info("Utils: Kalshi market minute data cache cleared.")


def parse_kalshi_ticker_info(ticker_string: str | None):
    """
    Parses Kalshi market or event ticker for series, date (YYMMMDD), hour (EDT), strike, and event_resolution_dt_utc.
    """
    if not ticker_string: return None
    market_match = re.match(r"^(.*?)-(\d{2}[A-Z]{3}\d{2})(\d{2})-(T(\d+\.?\d*))$", ticker_string)
    event_match = re.match(r"^(.*?)-(\d{2}[A-Z]{3}\d{2})(\d{2})$", ticker_string) # For event tickers

    match_to_use = market_match if market_match else event_match
    if not match_to_use:
        logger.debug(f"Utils: Ticker {ticker_string} did not match expected pattern.")
        return None

    groups = match_to_use.groups()
    series = groups[0]
    date_str_yymmmdd = groups[1] # e.g., 25MAY15
    hour_str_edt = groups[2]     # e.g., 22 (closing hour EDT)
    strike_price_val = float(groups[4]) if market_match and len(groups) > 4 and groups[4] else None

    try:
        year_int = 2000 + int(date_str_yymmmdd[:2])
        month_str = date_str_yymmmdd[2:5].upper()
        day_int = int(date_str_yymmmdd[5:])
        month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                     'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
        month_int = month_map[month_str]
        hour_edt_int = int(hour_str_edt)
        
        event_resolution_dt_naive_edt = dt.datetime(year_int, month_int, day_int, hour_edt_int, 0, 0)
        utc_offset_hours = 4 # Assuming EDT is UTC-4
        event_resolution_dt_utc_aware = event_resolution_dt_naive_edt.replace(tzinfo=timezone(timedelta(hours=-utc_offset_hours)))
        event_resolution_dt_utc = event_resolution_dt_utc_aware.astimezone(timezone.utc)
        
        return {
            "series": series,
            "date_str": date_str_yymmmdd, # Original YYMMMDD string
            "hour_str_EDT": hour_str_edt,
            "strike_price": strike_price_val,
            "event_resolution_dt_utc": event_resolution_dt_utc
        }
    except Exception as e:
        logger.error(f"Utils: Error parsing ticker {ticker_string}: {e}")
        return None


def _load_single_binance_day_raw(date_obj: dt.date) -> pd.DataFrame | None:
    """Loads a single day's raw Binance CSV from the flat directory structure."""
    global _binance_daily_raw_cache
    date_str = date_obj.strftime("%Y-%m-%d")
    if date_str in _binance_daily_raw_cache:
        df_copy = _binance_daily_raw_cache[date_str]
        return df_copy.copy() if df_copy is not None else None

    filepath = BINANCE_FLAT_DATA_DIR / f"BTCUSDT-1m-{date_str}.csv"
    if not filepath.exists():
        logger.warning(f"Utils: Binance raw data file not found for {date_str} at {filepath}")
        _binance_daily_raw_cache[date_str] = None
        return None
    try:
        column_names = ["open_time_raw", "open", "high", "low", "close", "volume",
                        "close_time_ms", "quote_asset_volume", "number_of_trades",
                        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
        df = pd.read_csv(filepath, header=None, names=column_names)
        if df.empty:
            _binance_daily_raw_cache[date_str] = None
            return None
        df['timestamp_s'] = df['open_time_raw'] // 1_000_000 # Start of minute
        for col in ['open', 'high', 'low', 'close', 'volume']: # Ensure numeric
             df[col] = pd.to_numeric(df[col], errors='coerce')
        df.set_index('timestamp_s', inplace=True)
        _binance_daily_raw_cache[date_str] = df
        return df.copy()
    except Exception as e:
        logger.error(f"Utils: Error loading raw Binance data from {filepath}: {e}")
        _binance_daily_raw_cache[date_str] = None
        return None

def _calculate_ta_features(df_btc: pd.DataFrame) -> pd.DataFrame:
    """Calculates TA features on a DataFrame with a 'close' column and timestamp_s index."""
    if df_btc.empty or 'close' not in df_btc.columns:
        return df_btc
    
    df = df_btc.copy() # Work on a copy
    df.dropna(subset=['close'], inplace=True)
    if df.empty: return df

    for window in BTC_MOMENTUM_WINDOWS:
        df[f'btc_mom_{window}m'] = df['close'].diff(periods=window)
    df[f'btc_vol_{BTC_VOLATILITY_WINDOW}m'] = df['close'].rolling(window=BTC_VOLATILITY_WINDOW, min_periods=2).std()
    for window in BTC_SMA_WINDOWS:
        df[f'btc_sma_{window}m'] = df['close'].rolling(window=window, min_periods=1).mean()
    for window in BTC_EMA_WINDOWS:
        df[f'btc_ema_{window}m'] = df['close'].ewm(span=window, adjust=False, min_periods=1).mean()
    if BTC_RSI_WINDOW > 0:
        delta = df['close'].diff(1)
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=BTC_RSI_WINDOW, min_periods=1).mean()
        avg_loss = loss.rolling(window=BTC_RSI_WINDOW, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-6) # Avoid division by zero
        df['btc_rsi'] = 100.0 - (100.0 / (1.0 + rs))
        df['btc_rsi'].fillna(50.0, inplace=True) # Neutral RSI for initial NaNs
    return df

def load_and_prepare_binance_range_with_features(start_dt: dt.datetime, end_dt: dt.datetime) -> pd.DataFrame | None:
    """
    Loads Binance data for a date range, concatenates, and calculates all TA features.
    Caches the result for the given range to avoid redundant processing.
    `start_dt` and `end_dt` are inclusive.
    """
    global _binance_range_with_features_cache, _cache_range_start_dt, _cache_range_end_dt
    
    # Normalize to date objects for range comparison and iteration
    start_date_obj = start_dt.date()
    end_date_obj = end_dt.date()

    if (_binance_range_with_features_cache is not None and
            _cache_range_start_dt == start_date_obj and
            _cache_range_end_dt == end_date_obj):
        logger.info(f"Utils: Using cached Binance features for range {start_date_obj} to {end_date_obj}")
        return _binance_range_with_features_cache.copy()

    logger.info(f"Utils: Preparing Binance data with features for range: {start_date_obj} to {end_date_obj}")
    all_daily_dfs = []
    current_date_obj = start_date_obj
    while current_date_obj <= end_date_obj:
        df_day_raw = _load_single_binance_day_raw(current_date_obj)
        if df_day_raw is not None and not df_day_raw.empty:
            all_daily_dfs.append(df_day_raw)
        current_date_obj += timedelta(days=1)

    if not all_daily_dfs:
        logger.error(f"Utils: No Binance data found for the entire range {start_date_obj} to {end_date_obj}")
        return None

    df_combined_raw = pd.concat(all_daily_dfs)
    if df_combined_raw.empty:
        logger.error(f"Utils: Combined Binance data is empty for range {start_date_obj} to {end_date_obj}")
        return None
        
    df_combined_raw.sort_index(inplace=True) # Ensure chronological order after concat
    df_combined_raw = df_combined_raw[~df_combined_raw.index.duplicated(keep='first')] # Remove potential overlaps

    logger.info(f"Utils: Calculating TA features on combined Binance data ({len(df_combined_raw)} rows)...")
    df_with_features = _calculate_ta_features(df_combined_raw)
    
    _binance_range_with_features_cache = df_with_features
    _cache_range_start_dt = start_date_obj
    _cache_range_end_dt = end_date_obj
    logger.info("Utils: Binance features calculated and cached for the range.")
    return df_with_features.copy()


def load_kalshi_market_minute_data(market_ticker: str, date_str_yymmmdd: str, hour_str_edt: str) -> pd.DataFrame | None:
    """Loads a specific Kalshi market's 1-minute data and caches it."""
    global _kalshi_market_minute_data_cache
    cache_key = f"{date_str_yymmmdd}_{hour_str_edt}_{market_ticker}"
    if cache_key in _kalshi_market_minute_data_cache:
        df_copy = _kalshi_market_minute_data_cache[cache_key]
        return df_copy.copy() if df_copy is not None else None

    filepath = KALSHI_DATA_DIR / date_str_yymmmdd / hour_str_edt.zfill(2) / f"{market_ticker}.csv"
    if not filepath.exists():
        logger.debug(f"Utils: Kalshi minute data file not found: {filepath}")
        _kalshi_market_minute_data_cache[cache_key] = None
        return None
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            _kalshi_market_minute_data_cache[cache_key] = pd.DataFrame() # Cache empty df
            return pd.DataFrame()
            
        df['timestamp_s'] = pd.to_numeric(df['timestamp_s'])
        df.set_index('timestamp_s', inplace=True)
        for col in df.columns:
            if 'cents' in col or 'volume' in col or 'interest' in col:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        _kalshi_market_minute_data_cache[cache_key] = df
        return df.copy()
    except Exception as e:
        logger.error(f"Utils: Error loading Kalshi market minute data from {filepath}: {e}")
        _kalshi_market_minute_data_cache[cache_key] = None
        return None

def get_kalshi_prices_at_decision(
    kalshi_market_df: pd.DataFrame | None, # DF for this specific market (already loaded)
    decision_timestamp_s: int, 
    max_staleness_seconds: int
) -> dict | None:
    """
    Gets Kalshi yes_bid/yes_ask at or just before decision_timestamp_s from the given market's DataFrame.
    """
    if kalshi_market_df is None or kalshi_market_df.empty:
        # logger.debug(f"Utils: Kalshi market_df is None or empty for decision at {decision_timestamp_s}.")
        return None
    
    try:
        # Find data at or immediately before the decision_timestamp_s
        relevant_rows = kalshi_market_df[kalshi_market_df.index <= decision_timestamp_s]
        if not relevant_rows.empty:
            latest_row = relevant_rows.iloc[-1]
            latest_row_ts = latest_row.name # This is the timestamp_s from the index
            
            time_diff_seconds = decision_timestamp_s - latest_row_ts
            
            if time_diff_seconds <= max_staleness_seconds:
                # logger.debug(f"Utils: Using Kalshi data from ts {latest_row_ts} (Staleness: {time_diff_seconds}s) for decision at {decision_timestamp_s}")
                return {
                    "yes_bid": latest_row.get('yes_bid_close_cents'), 
                    "yes_ask": latest_row.get('yes_ask_close_cents')
                }
            else:
                # logger.warning(f"Utils: No recent Kalshi prices for decision {decision_timestamp_s}. Latest is {time_diff_seconds}s old (Max staleness: {max_staleness_seconds}s).")
                return None
        else:
            # logger.warning(f"Utils: No Kalshi data at or before decision {decision_timestamp_s} in provided market_df.")
            return None
            
    except Exception as e:
        logger.error(f"Utils: Error in get_kalshi_prices_at_decision for {decision_timestamp_s}: {e}")
        return None