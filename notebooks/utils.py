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

# These will be set by the main backtest script if they run utils.py directly,
# or used by other modules importing from utils.
BASE_PROJECT_DIR = Path.cwd() 
BINANCE_FLAT_DATA_DIR = BASE_PROJECT_DIR / "binance_data" 
KALSHI_DATA_DIR = BASE_PROJECT_DIR / "kalshi_data" 

# --- Feature Calculation Parameters (Single Source of Truth) ---
# --- BTC Features ---
BTC_MOMENTUM_WINDOWS = [5, 10, 15, 30, 60] # Expanded
BTC_VOLATILITY_WINDOW = 15 
BTC_SMA_WINDOWS = [10, 30, 50]          # Expanded
BTC_EMA_WINDOWS = [12, 26, 50]          # Expanded
BTC_RSI_WINDOW = 14
BTC_ATR_WINDOW = 14                     # New

# --- Kalshi Features (Primarily for strategy logic, not directly calculated in _calculate_ta_features here) ---
# These are defined here so strategy modules can import them for consistency if needed.
# The actual calculation of Kalshi-specific features (like mid-price changes or volatility)
# happens within the feature engineering script or the strategy files themselves,
# as they require the Kalshi market data specific to each contract.
KALSHI_PRICE_CHANGE_WINDOWS = [1, 3, 5, 10] # Expanded
KALSHI_VOLATILITY_WINDOWS = [5, 10]       # New (for Kalshi mid-price volatility)
# Max staleness for Kalshi data when generating features (used in strategy/feature_engineering)
KALSHI_MAX_STALENESS_SECONDS_FOR_FEATURES = 120


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
        edt_offset_from_utc = timedelta(hours=-4) 
        event_resolution_dt_edt_aware = event_resolution_dt_naive_edt.replace(tzinfo=timezone(edt_offset_from_utc))
        event_resolution_dt_utc = event_resolution_dt_edt_aware.astimezone(timezone.utc)
        
        return {
            "series": series,
            "date_str": date_str_yymmmdd, 
            "hour_str_EDT": hour_str_edt, 
            "strike_price": strike_price_val,
            "event_resolution_dt_utc": event_resolution_dt_utc
        }
    except Exception as e:
        logger.error(f"Utils: Error parsing ticker {ticker_string}: {e}")
        return None

def get_session_key_from_market_row(market_row_series: pd.Series) -> str | None:
    ticker = market_row_series.get('market_ticker')
    if not ticker:
        return None
    parsed_info = parse_kalshi_ticker_info(ticker)
    if parsed_info and parsed_info.get('date_str') and parsed_info.get('hour_str_EDT'):
        return f"{parsed_info['date_str']}_{parsed_info['hour_str_EDT']}"
    return None


def _load_single_binance_day_raw(date_obj: dt.date) -> pd.DataFrame | None:
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
        df['timestamp_s'] = df['open_time_raw'] // 1_000_000 
        for col in ['open', 'high', 'low', 'close', 'volume']: 
             df[col] = pd.to_numeric(df[col], errors='coerce')
        df.set_index('timestamp_s', inplace=True)
        df.dropna(subset=['close', 'high', 'low'], inplace=True) # Ensure H,L,C for ATR
        _binance_daily_raw_cache[date_str] = df
        return df.copy()
    except Exception as e:
        logger.error(f"Utils: Error loading raw Binance data from {filepath}: {e}")
        _binance_daily_raw_cache[date_str] = None
        return None

def _calculate_ta_features(df_btc: pd.DataFrame) -> pd.DataFrame:
    if df_btc.empty or 'close' not in df_btc.columns:
        return df_btc.copy() 
    
    df = df_btc.copy() 
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    # Ensure high and low are also numeric for ATR
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df.dropna(subset=['close', 'high', 'low'], inplace=True) 
    if df.empty: return df_btc.copy()

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
        loss = -delta.where(delta < 0, 0.0) 
        avg_gain = gain.ewm(com=BTC_RSI_WINDOW - 1, min_periods=BTC_RSI_WINDOW).mean()
        avg_loss = loss.ewm(com=BTC_RSI_WINDOW - 1, min_periods=BTC_RSI_WINDOW).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9) 
        df['btc_rsi'] = 100.0 - (100.0 / (1.0 + rs))
        df['btc_rsi'] = df['btc_rsi'].fillna(50.0)

    # Calculate ATR
    if BTC_ATR_WINDOW > 0:
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift(1))
        low_close_prev = np.abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False) # skipna=False to ensure if any component is NaN, TR is NaN
        df[f'btc_atr_{BTC_ATR_WINDOW}'] = tr.ewm(alpha=1/BTC_ATR_WINDOW, adjust=False, min_periods=BTC_ATR_WINDOW).mean()
        # ATR can have initial NaNs, which should be handled by imputation in the feature engineering/strategy.
        # df[f'btc_atr_{BTC_ATR_WINDOW}'].fillna(method='bfill', inplace=True) # Or handle imputation later
    return df

def load_and_prepare_binance_range_with_features(start_dt: dt.datetime, end_dt: dt.datetime) -> pd.DataFrame | None:
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
    
    # Determine max lookback needed for all TA features
    all_btc_windows = BTC_MOMENTUM_WINDOWS + [BTC_VOLATILITY_WINDOW] + BTC_SMA_WINDOWS + BTC_EMA_WINDOWS + [BTC_RSI_WINDOW, BTC_ATR_WINDOW]
    max_lookback_ta_minutes = max(all_btc_windows, default=0)
    max_lookback_days = (max_lookback_ta_minutes // 1440) + 2 # Convert minutes to days, add buffer
    
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
    df_with_features = _calculate_ta_features(df_combined_raw) # This now calculates ATR too
    
    requested_start_ts = int(start_dt.replace(tzinfo=timezone.utc).timestamp())
    requested_end_ts = int(end_dt.replace(tzinfo=timezone.utc).timestamp())
    
    df_with_features_filtered_range = df_with_features[
        (df_with_features.index >= requested_start_ts) & 
        (df_with_features.index <= requested_end_ts)
    ]
    
    _binance_range_with_features_cache = df_with_features_filtered_range 
    _cache_range_start_dt = start_date_obj
    _cache_range_end_dt = end_date_obj
    logger.info(f"Utils: Binance features calculated and cached for the requested range {start_date_obj} to {end_date_obj}.")
    return df_with_features_filtered_range.copy()


def load_kalshi_market_minute_data(market_ticker: str, date_str_yymmmdd: str, hour_str_edt: str) -> pd.DataFrame | None:
    global _kalshi_market_minute_data_cache
    cache_key = f"{date_str_yymmmdd}_{hour_str_edt.zfill(2)}_{market_ticker}" 
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
            _kalshi_market_minute_data_cache[cache_key] = pd.DataFrame() 
            return pd.DataFrame()
            
        df['timestamp_s'] = pd.to_numeric(df['timestamp_s'])
        df.set_index('timestamp_s', inplace=True)
        
        cols_to_numeric = ['yes_bid_close_cents', 'yes_ask_close_cents', 'volume', 'open_interest']
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Pre-calculate mid_price if components exist, for potential use in Kalshi feature generation
        if 'yes_bid_close_cents' in df.columns and 'yes_ask_close_cents' in df.columns:
            bid = pd.to_numeric(df['yes_bid_close_cents'], errors='coerce')
            ask = pd.to_numeric(df['yes_ask_close_cents'], errors='coerce')
            df['mid_price'] = (bid + ask) / 2.0 # Will be NaN if bid or ask is NaN

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
    if kalshi_market_df is None or kalshi_market_df.empty:
        return None
    
    try:
        relevant_rows = kalshi_market_df[kalshi_market_df.index <= decision_timestamp_s]
        if not relevant_rows.empty:
            latest_row = relevant_rows.iloc[-1]
            latest_row_ts = latest_row.name 
            
            time_diff_seconds = decision_timestamp_s - latest_row_ts
            
            if 0 <= time_diff_seconds <= max_staleness_seconds:
                return {
                    "yes_bid": latest_row.get('yes_bid_close_cents'), 
                    "yes_ask": latest_row.get('yes_ask_close_cents')
                }
            return None
        return None
    except Exception as e:
        logger.error(f"Utils: Error in get_kalshi_prices_at_decision for {decision_timestamp_s}: {e}")
        return None