# live_backtester/live_utils.py
import pandas as pd
from pathlib import Path
import datetime as dt
from datetime import timezone, timedelta
import re
import logging
import numpy as np

logger = logging.getLogger(__name__)

# --- Parameters for Feature Calculation (from your historical setup) ---
BTC_MOMENTUM_WINDOWS = [5, 10, 15, 30]
BTC_VOLATILITY_WINDOW = 15
BTC_SMA_WINDOWS = [10, 30]
BTC_EMA_WINDOWS = [12, 26]
BTC_RSI_WINDOW = 14
KALSHI_PRICE_CHANGE_WINDOWS = [1, 3, 5]

# Define your local timezone's current offset from UTC.
# PST is UTC-8, PDT is UTC-7.
# IMPORTANT: This needs to be correct for the period your Kalshi data was collected.
# For "2025-05-20", if it's during Daylight Saving Time (PDT), it's UTC-7.
LOCAL_TIMEZONE_OFFSET_HOURS = -7 # Example for PDT (UTC-7)

def parse_kalshi_ticker_details_from_name(ticker_string: str | None) -> dict | None:
    """
    Parses Kalshi market ticker for series, date (YYMMMDD), hour (EDT), and strike.
    Also calculates the approximate market close/resolution time in UTC.
    """
    if not ticker_string: return None
    market_match = re.match(r"^(.*?)-(\d{2}[A-Z]{3}\d{2})(\d{2})-(T(\d+\.?\d*))$", ticker_string)
    
    if not market_match:
        logger.debug(f"LiveUtils: Ticker {ticker_string} did not match expected market pattern.")
        return None

    groups = market_match.groups()
    series = groups[0]
    date_str_yymmmdd = groups[1] 
    hour_str_edt = groups[2]     
    strike_price_val = float(groups[4]) if groups[4] else None

    try:
        year_int = 2000 + int(date_str_yymmmdd[:2])
        month_str = date_str_yymmmdd[2:5].upper()
        day_int = int(date_str_yymmmdd[5:])
        month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                     'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
        month_int = month_map[month_str]
        hour_edt_int = int(hour_str_edt)
        
        market_resolution_dt_naive_edt = dt.datetime(year_int, month_int, day_int, hour_edt_int, 0, 0)
        # EDT is UTC-4 (Eastern Daylight Time) for market close time definition
        edt_offset_hours = -4 
        market_resolution_dt_utc_aware_from_edt = market_resolution_dt_naive_edt.replace(tzinfo=timezone(timedelta(hours=edt_offset_hours)))
        market_resolution_dt_utc = market_resolution_dt_utc_aware_from_edt.astimezone(timezone.utc)
        
        return {
            "series": series,
            "date_str_yymmmdd": date_str_yymmmdd,
            "hour_str_edt": hour_str_edt,
            "strike_price": strike_price_val,
            "market_resolution_dt_utc": market_resolution_dt_utc 
        }
    except Exception as e:
        logger.error(f"LiveUtils: Error parsing ticker {ticker_string}: {e}")
        return None

def load_live_kalshi_csv(filepath: Path) -> pd.DataFrame | None:
    """Loads and preprocesses a live Kalshi data CSV, assuming timestamps are local."""
    if not filepath.exists():
        logger.warning(f"LiveUtils: Kalshi live data file not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            logger.warning(f"LiveUtils: Kalshi live data file is empty: {filepath}")
            return pd.DataFrame()
        
        df['timestamp_naive'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp_naive'], inplace=True)
        
        local_tz = timezone(timedelta(hours=LOCAL_TIMEZONE_OFFSET_HOURS))
        if isinstance(df['timestamp_naive'], pd.Series):
            df['timestamp_local_aware'] = df['timestamp_naive'].apply(lambda x: x.replace(tzinfo=local_tz) if pd.notnull(x) else x)
        else: 
            df['timestamp_local_aware'] = df['timestamp_naive'].replace(tzinfo=local_tz)

        df['timestamp'] = df['timestamp_local_aware'].dt.tz_convert(timezone.utc)
        
        df.set_index('timestamp', inplace=True) 
        df.sort_index(inplace=True)
        
        # if not df.empty:
        #     logger.debug(f"Kalshi data for {filepath.name}: First 5 UTC timestamps from CSV:")
        #     for ts_idx in df.index[:5]: logger.debug(ts_idx.isoformat())
        
        for col in ['yes_bid_price_cents', 'yes_bid_qty', 'yes_ask_price_cents', 'yes_ask_qty_on_no_side', 'sequence_num']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.drop(columns=['timestamp_naive', 'timestamp_local_aware'], errors='ignore')
    except Exception as e:
        logger.error(f"LiveUtils: Error loading live Kalshi data from {filepath}: {e}", exc_info=True)
        return None

def load_live_binance_csv_and_extract_closed_klines(filepath: Path) -> pd.DataFrame | None:
    """
    Loads a live Binance kline CSV and extracts the final state of each closed 1-minute kline.
    """
    if not filepath.exists():
        logger.warning(f"LiveUtils: Binance live data file not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            logger.warning(f"LiveUtils: Binance live data file is empty: {filepath}")
            return pd.DataFrame()

        df['kline_start_dt_utc'] = pd.to_datetime(df['kline_start_time_ms'], unit='ms', utc=True)
        
        # if not df.empty:
        #     logger.debug(f"Binance data for {filepath.name}: First 5 kline_start_dt_utc (from ms):")
        #     unique_starts = df['kline_start_dt_utc'].unique()
        #     for ts_val in unique_starts[:5]:
        #          if pd.notna(ts_val): logger.debug(ts_val.isoformat())
        
        df.sort_values(by=['kline_start_time_ms', 'reception_timestamp_utc'], inplace=True)
        df_closed_klines = df.groupby('kline_start_time_ms').last().reset_index()
        df_closed_klines['timestamp_s'] = df_closed_klines['kline_start_time_ms'] // 1000
        df_closed_klines.set_index('timestamp_s', inplace=True)
        df_closed_klines.sort_index(inplace=True)
        
        rename_map = {
            "open_price": "open", "high_price": "high", "low_price": "low",
            "close_price": "close", "base_asset_volume": "volume"
        }
        df_closed_klines.rename(columns=rename_map, inplace=True)
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
             if col in df_closed_klines.columns:
                df_closed_klines[col] = pd.to_numeric(df_closed_klines[col], errors='coerce')
        
        cols_to_keep = ['open', 'high', 'low', 'close', 'volume', 'kline_start_dt_utc', 'is_kline_closed']
        final_cols = [col for col in cols_to_keep if col in df_closed_klines.columns]
        
        return df_closed_klines[final_cols]

    except Exception as e:
        logger.error(f"LiveUtils: Error loading/processing live Binance data from {filepath}: {e}", exc_info=True)
        return None

def calculate_live_ta_features(df_btc_klines: pd.DataFrame, feature_order_list: list = None) -> pd.DataFrame:
    if df_btc_klines.empty or 'close' not in df_btc_klines.columns:
        return df_btc_klines 
    
    df = df_btc_klines.copy()
    df.dropna(subset=['close'], inplace=True) 
    if df.empty: return df

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
        df['btc_rsi'].fillna(50.0, inplace=True) 
    elif feature_order_list and 'btc_rsi' in feature_order_list : 
             df['btc_rsi'] = 50.0 
    
    return df

def get_kalshi_snapshot_at_decision(
    df_kalshi_live: pd.DataFrame, 
    decision_dt_utc: dt.datetime, 
    max_staleness_seconds: int = 5 
) -> dict | None:
    if df_kalshi_live is None or df_kalshi_live.empty:
        return None
    try:
        relevant_rows = df_kalshi_live[df_kalshi_live.index <= decision_dt_utc]
        if not relevant_rows.empty:
            latest_row = relevant_rows.iloc[-1]
            latest_row_dt_utc = latest_row.name 
            time_diff_seconds = (decision_dt_utc - latest_row_dt_utc).total_seconds()
            
            if time_diff_seconds <= max_staleness_seconds and time_diff_seconds >= 0:
                return {
                    "yes_bid": latest_row.get('yes_bid_price_cents'), 
                    "yes_ask": latest_row.get('yes_ask_price_cents'),
                    "yes_bid_qty": latest_row.get('yes_bid_qty'),
                    "yes_ask_qty_on_no_side": latest_row.get('yes_ask_qty_on_no_side'),
                    "timestamp_utc": latest_row_dt_utc 
                }
            return None
        return None
    except Exception as e:
        logger.error(f"LiveUtils: Error in get_kalshi_snapshot_at_decision for {decision_dt_utc}: {e}", exc_info=True)
        return None

# This function is no longer needed if we use the actual outcomes CSV for resolution
# def get_btc_price_at_resolution_from_live(...)