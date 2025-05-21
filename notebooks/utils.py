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

# These will be set by the main backtest script
BASE_PROJECT_DIR = Path.cwd() 
BINANCE_FLAT_DATA_DIR = BASE_PROJECT_DIR / "binance_data" 
KALSHI_DATA_DIR = BASE_PROJECT_DIR / "kalshi_data" 

# --- Feature Calculation Parameters (Single Source of Truth) ---
# These need to match what the model was trained with and what the strategy uses.
BTC_MOMENTUM_WINDOWS = [5, 10, 15, 30]
BTC_VOLATILITY_WINDOW = 15
BTC_SMA_WINDOWS = [10, 30]
BTC_EMA_WINDOWS = [12, 26]
BTC_RSI_WINDOW = 14
# KALSHI_PRICE_CHANGE_WINDOWS is used by strategy, not directly in utils features yet.

# --- Caches ---
_binance_daily_raw_cache = {} 
_binance_range_with_features_cache = None 
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
    event_match = re.match(r"^(.*?)-(\d{2}[A-Z]{3}\d{2})(\d{2})$", ticker_string) 

    match_to_use = market_match if market_match else event_match
    if not match_to_use:
        logger.debug(f"Utils: Ticker {ticker_string} did not match expected pattern.")
        return None

    groups = match_to_use.groups()
    series = groups[0]
    date_str_yymmmdd = groups[1] 
    hour_str_edt = groups[2]     
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
        # EDT is UTC-4. This is the standard offset for Kalshi's BTC market resolution times.
        edt_offset_from_utc = timedelta(hours=-4) 
        event_resolution_dt_edt_aware = event_resolution_dt_naive_edt.replace(tzinfo=timezone(edt_offset_from_utc))
        event_resolution_dt_utc = event_resolution_dt_edt_aware.astimezone(timezone.utc)
        
        return {
            "series": series,
            "date_str": date_str_yymmmdd, 
            "hour_str_EDT": hour_str_edt, # This is the closing hour in EDT as per Kalshi ticker convention
            "strike_price": strike_price_val,
            "event_resolution_dt_utc": event_resolution_dt_utc
        }
    except Exception as e:
        logger.error(f"Utils: Error parsing ticker {ticker_string}: {e}")
        return None

def get_session_key_from_market_row(market_row_series: pd.Series) -> str | None:
    """
    Generates a session key (e.g., '25MAY15_10') from a market row (DataFrame series).
    Assumes market_row_series['market_ticker'] exists.
    """
    ticker = market_row_series.get('market_ticker')
    if not ticker:
        return None
    parsed_info = parse_kalshi_ticker_info(ticker)
    if parsed_info and parsed_info.get('date_str') and parsed_info.get('hour_str_EDT'):
        return f"{parsed_info['date_str']}_{parsed_info['hour_str_EDT']}"
    return None


def _load_single_binance_day_raw(date_obj: dt.date) -> pd.DataFrame | None:
    """Loads a single day's raw Binance CSV from the flat directory structure."""
    global _binance_daily_raw_cache
    date_str = date_obj.strftime("%Y-%m-%d")
    if date_str in _binance_daily_raw_cache:
        df_copy = _binance_daily_raw_cache[date_str]
        return df_copy.copy() if df_copy is not None else None

    # Use BINANCE_FLAT_DATA_DIR which should be set by the calling script
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
            _binance_daily_raw_cache[date_str] = None # Cache None for empty files
            return None
        df['timestamp_s'] = df['open_time_raw'] // 1_000_000 
        for col in ['open', 'high', 'low', 'close', 'volume']: 
             df[col] = pd.to_numeric(df[col], errors='coerce')
        df.set_index('timestamp_s', inplace=True)
        df.dropna(subset=['close'], inplace=True) # Drop rows where close price is NaN after conversion
        _binance_daily_raw_cache[date_str] = df
        return df.copy()
    except Exception as e:
        logger.error(f"Utils: Error loading raw Binance data from {filepath}: {e}")
        _binance_daily_raw_cache[date_str] = None
        return None

def _calculate_ta_features(df_btc: pd.DataFrame) -> pd.DataFrame:
    """Calculates TA features on a DataFrame with a 'close' column and timestamp_s index."""
    if df_btc.empty or 'close' not in df_btc.columns:
        return df_btc.copy() # Return copy even if empty or no close
    
    df = df_btc.copy() 
    # Ensure 'close' is numeric; if not, TA features will fail or be incorrect
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df.dropna(subset=['close'], inplace=True) # Critical for TA calculations
    if df.empty: return df_btc.copy() # Return original (empty or no valid 'close') if all rows dropped

    # Use globally defined window sizes from this utils module
    for window in BTC_MOMENTUM_WINDOWS:
        df[f'btc_mom_{window}m'] = df['close'].diff(periods=window)
    if BTC_VOLATILITY_WINDOW > 0:
        df[f'btc_vol_{BTC_VOLATILITY_WINDOW}m'] = df['close'].rolling(window=BTC_VOLATILITY_WINDOW, min_periods=max(1, BTC_VOLATILITY_WINDOW // 2)).std()
    for window in BTC_SMA_WINDOWS:
        df[f'btc_sma_{window}m'] = df['close'].rolling(window=window, min_periods=1).mean()
    for window in BTC_EMA_WINDOWS:
        df[f'btc_ema_{window}m'] = df['close'].ewm(span=window, adjust=False, min_periods=1).mean()
    
    if BTC_RSI_WINDOW > 0:
        delta = df['close'].diff(1)
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0) # Ensure loss is positive for calculation
        
        # Use EWM for RSI calculation for smoother average, similar to many platforms
        avg_gain = gain.ewm(com=BTC_RSI_WINDOW - 1, min_periods=BTC_RSI_WINDOW).mean()
        avg_loss = loss.ewm(com=BTC_RSI_WINDOW - 1, min_periods=BTC_RSI_WINDOW).mean()
        
        rs = avg_gain / avg_loss.replace(0, 1e-9) # Avoid division by zero
        df['btc_rsi'] = 100.0 - (100.0 / (1.0 + rs))
        df['btc_rsi'].fillna(50.0, inplace=True) # Neutral RSI for initial NaNs (common practice)
    return df

def load_and_prepare_binance_range_with_features(start_dt: dt.datetime, end_dt: dt.datetime) -> pd.DataFrame | None:
    """
    Loads Binance data for a date range, concatenates, and calculates all TA features.
    Caches the result for the given range to avoid redundant processing.
    """
    global _binance_range_with_features_cache, _cache_range_start_dt, _cache_range_end_dt
    
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
    max_lookback_days = (max(BTC_MOMENTUM_WINDOWS + [BTC_VOLATILITY_WINDOW] + BTC_SMA_WINDOWS + BTC_EMA_WINDOWS + [BTC_RSI_WINDOW], default=0) // 1440) + 2 # days
    
    # Prepend data from earlier days to ensure enough history for rolling window calculations at the start of the range
    # Note: The TA functions use .diff(periods=window), so this pre-pending is more about correct std/mean at start.
    # For .diff(), it will just result in NaNs for the first `window` entries.
    start_date_for_load = start_date_obj - timedelta(days=max_lookback_days)

    current_load_date = start_date_for_load
    while current_load_date <= end_date_obj:
        df_day_raw = _load_single_binance_day_raw(current_load_date)
        if df_day_raw is not None and not df_day_raw.empty:
            all_daily_dfs.append(df_day_raw)
        current_load_date += timedelta(days=1)


    if not all_daily_dfs:
        logger.error(f"Utils: No Binance data found for the extended range {start_date_for_load} to {end_date_obj}")
        return None

    df_combined_raw = pd.concat(all_daily_dfs)
    if df_combined_raw.empty:
        logger.error(f"Utils: Combined Binance data is empty for extended range.")
        return None
        
    df_combined_raw.sort_index(inplace=True) 
    df_combined_raw = df_combined_raw[~df_combined_raw.index.duplicated(keep='first')]

    logger.info(f"Utils: Calculating TA features on combined Binance data ({len(df_combined_raw)} rows)...")
    df_with_features = _calculate_ta_features(df_combined_raw)
    
    # Filter back to the originally requested range AFTER feature calculation
    requested_start_ts = int(start_dt.replace(tzinfo=timezone.utc).timestamp())
    requested_end_ts = int(end_dt.replace(tzinfo=timezone.utc).timestamp())
    
    df_with_features_filtered_range = df_with_features[
        (df_with_features.index >= requested_start_ts) & 
        (df_with_features.index <= requested_end_ts)
    ]
    
    _binance_range_with_features_cache = df_with_features_filtered_range # Cache the filtered range
    _cache_range_start_dt = start_date_obj
    _cache_range_end_dt = end_date_obj
    logger.info(f"Utils: Binance features calculated and cached for the requested range {start_date_obj} to {end_date_obj}.")
    return df_with_features_filtered_range.copy()


def load_kalshi_market_minute_data(market_ticker: str, date_str_yymmmdd: str, hour_str_edt: str) -> pd.DataFrame | None:
    """Loads a specific Kalshi market's 1-minute data and caches it."""
    global _kalshi_market_minute_data_cache
    cache_key = f"{date_str_yymmmdd}_{hour_str_edt.zfill(2)}_{market_ticker}" # Ensure hour_str is padded
    if cache_key in _kalshi_market_minute_data_cache:
        df_copy = _kalshi_market_minute_data_cache[cache_key]
        return df_copy.copy() if df_copy is not None else None

    # Use KALSHI_DATA_DIR which should be set by the calling script
    filepath = KALSHI_DATA_DIR / date_str_yymmmdd / hour_str_edt.zfill(2) / f"{market_ticker}.csv"
    if not filepath.exists():
        logger.debug(f"Utils: Kalshi minute data file not found: {filepath}")
        _kalshi_market_minute_data_cache[cache_key] = None
        return None
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            _kalshi_market_minute_data_cache[cache_key] = pd.DataFrame() 
            return pd.DataFrame()
            
        df['timestamp_s'] = pd.to_numeric(df['timestamp_s'])
        df.set_index('timestamp_s', inplace=True)
        # Ensure relevant columns are numeric, coercing errors
        cols_to_numeric = ['yes_bid_close_cents', 'yes_ask_close_cents', 'volume', 'open_interest']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        _kalshi_market_minute_data_cache[cache_key] = df
        return df.copy()
    except Exception as e:
        logger.error(f"Utils: Error loading Kalshi market minute data from {filepath}: {e}")
        _kalshi_market_minute_data_cache[cache_key] = None
        return None

def get_kalshi_prices_at_decision(
    kalshi_market_df: pd.DataFrame | None, 
    decision_timestamp_s: int, 
    max_staleness_seconds: int
) -> dict | None:
    """
    Gets Kalshi yes_bid/yes_ask at or just before decision_timestamp_s.
    Uses 'yes_bid_close_cents' and 'yes_ask_close_cents' from the Kalshi minute data.
    """
    if kalshi_market_df is None or kalshi_market_df.empty:
        return None
    
    try:
        relevant_rows = kalshi_market_df[kalshi_market_df.index <= decision_timestamp_s]
        if not relevant_rows.empty:
            latest_row = relevant_rows.iloc[-1]
            latest_row_ts = latest_row.name 
            
            time_diff_seconds = decision_timestamp_s - latest_row_ts
            
            if time_diff_seconds <= max_staleness_seconds and time_diff_seconds >=0:
                return {
                    "yes_bid": latest_row.get('yes_bid_close_cents'), 
                    "yes_ask": latest_row.get('yes_ask_close_cents')
                    # Add other fields like volume if needed by strategy
                }
            # else:
                # logger.debug(f"Utils: Kalshi data too stale for {decision_timestamp_s}. Diff: {time_diff_seconds}s")
            return None
        return None
            
    except Exception as e:
        logger.error(f"Utils: Error in get_kalshi_prices_at_decision for {decision_timestamp_s}: {e}")
        return None