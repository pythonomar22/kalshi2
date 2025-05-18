# utils.py
import pandas as pd
import os
import re
import datetime as dt
from datetime import timezone, timedelta
import logging
import numpy as np

logger = logging.getLogger(__name__) 
BASE_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BINANCE_DATA_PATH_TEMPLATE = os.path.join(BASE_PROJECT_DIR, "binance_data/BTCUSDT-1m-{date_nodash}/BTCUSDT-1m-{date_nodash}.csv")
_binance_data_cache = {} 

def parse_kalshi_ticker_info(ticker_string: str):
    # (Keep the version from your last successful run)
    match = re.match(r"^(.*?)-(\d{2}[A-Z]{3}\d{2})(\d{2})-(T(\d+\.?\d*))$", ticker_string)
    if match:
        series, date_str_yyMmmdd, hour_str_EDT, strike_full, strike_val_str = match.groups()
        try:
            year_int = 2000 + int(date_str_yyMmmdd[:2]); month_str = date_str_yyMmmdd[2:5]
            day_int = int(date_str_yyMmmdd[5:])
            month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 
                         'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
            month_int = month_map[month_str.upper()]; hour_edt_int = int(hour_str_EDT)
            event_resolution_dt_naive_edt = dt.datetime(year_int, month_int, day_int, hour_edt_int, 0, 0)
            utc_offset_hours = 4 
            event_resolution_dt_utc_aware = event_resolution_dt_naive_edt.replace(tzinfo=timezone(timedelta(hours=-utc_offset_hours)))
            event_resolution_dt_utc = event_resolution_dt_utc_aware.astimezone(timezone.utc)
            logger.debug(f"Parsed {ticker_string}: DateStr={date_str_yyMmmdd}, HourEDT={hour_str_EDT} -> Resolves UTC: {event_resolution_dt_utc.isoformat()}")
            return {"series": series, "date_str": date_str_yyMmmdd, "hour_str_EDT": hour_str_EDT,
                    "strike_price": float(strike_val_str), "event_resolution_dt_utc": event_resolution_dt_utc}
        except Exception as e: logger.error(f"Error parsing ticker {ticker_string}: {e}"); return None
    logger.debug(f"Ticker {ticker_string} did not match expected pattern.")
    return None

def load_binance_data_for_date(date_nodash: str):
    # (Keep the version from your last successful run - it adds features)
    global _binance_data_cache
    if date_nodash in _binance_data_cache:
        if _binance_data_cache[date_nodash] is not None: return _binance_data_cache[date_nodash].copy()
        return None
    filepath = BINANCE_DATA_PATH_TEMPLATE.format(date_nodash=date_nodash)
    if not os.path.exists(filepath): logger.error(f"Binance file not found: {filepath}"); _binance_data_cache[date_nodash] = None; return None
    try:
        column_names = ["open_time_raw", "open", "high", "low", "close", "volume", "close_time_ms", "quote_asset_volume", 
                        "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
        df = pd.read_csv(filepath, header=None, names=column_names)
        if df.empty: logger.warning(f"Binance data file {filepath} is empty."); _binance_data_cache[date_nodash] = None; return None
        df['datetime_utc'] = pd.to_datetime(df['open_time_raw'], unit='us', utc=True)
        df['timestamp_s'] = (df['datetime_utc'] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta('1s')
        df.set_index('timestamp_s', inplace=True)
        df['close'] = pd.to_numeric(df['close'])
        df['btc_price_lag_1m'] = df['close'].shift(1)
        df['btc_price_change_1m'] = df['close'].diff(1)
        df['btc_price_change_5m'] = df['close'].diff(5)
        df['btc_price_change_15m'] = df['close'].diff(15)
        df['btc_volatility_15m'] = df['close'].rolling(window=15).std()
        min_ts, max_ts = df.index.min(), df.index.max()
        logger.info(f"Loaded Binance data for {date_nodash} ({len(df)} rows). Index (sec UTC): "
                     f"{dt.datetime.fromtimestamp(min_ts, tz=timezone.utc).isoformat()} to "
                     f"{dt.datetime.fromtimestamp(max_ts, tz=timezone.utc).isoformat()}")
        _binance_data_cache[date_nodash] = df 
        return df.copy()
    except Exception as e: logger.error(f"Error loading Binance data from {filepath}: {e}"); _binance_data_cache[date_nodash] = None; return None

def get_btc_features_for_signal(signal_timestamp_s: int):
    # (Keep the version from your last successful run)
    global _binance_data_cache
    signal_dt_utc = dt.datetime.fromtimestamp(signal_timestamp_s, tz=timezone.utc)
    date_nodash_needed = signal_dt_utc.strftime("%Y-%m-%d")
    if date_nodash_needed not in _binance_data_cache or _binance_data_cache[date_nodash_needed] is None:
        logger.info(f"Binance data for {date_nodash_needed} not in cache. Loading...")
        if load_binance_data_for_date(date_nodash_needed) is None: 
             logger.warning(f"Failed to load Binance for {date_nodash_needed} for signal at {signal_dt_utc.isoformat()}")
             return None 
    binance_df_for_date = _binance_data_cache.get(date_nodash_needed)
    if binance_df_for_date is None: return None
    try:
        if signal_timestamp_s in binance_df_for_date.index:
            row = binance_df_for_date.loc[signal_timestamp_s]
            features = {'btc_price': row['close'], 'btc_price_change_1m': row.get('btc_price_change_1m'), 
                        'btc_price_change_5m': row.get('btc_price_change_5m'), 'btc_price_change_15m': row.get('btc_price_change_15m'),
                        'btc_volatility_15m': row.get('btc_volatility_15m')}
            for key, val in features.items():
                if pd.isna(val):
                    logger.debug(f"NaN feature '{key}' at {signal_dt_utc.isoformat()}."); return None
            return features
        else: logger.debug(f"Exact BTC timestamp {signal_dt_utc.isoformat()} not in Binance index for {date_nodash_needed}."); return None
    except Exception as e: logger.error(f"Error in get_btc_features for {signal_timestamp_s} ({date_nodash_needed}): {e}"); return None

def load_kalshi_market_data(filepath: str):
    # (Keep the version from your last successful run)
    if not os.path.exists(filepath): logger.warning(f"Kalshi file not found: {filepath}"); return None
    try:
        df = pd.read_csv(filepath)
        if df.empty: logger.debug(f"Kalshi file {filepath} is empty."); return None
        df['timestamp_s'] = pd.to_numeric(df['timestamp_s'])
        for col in ['yes_bid_close_cents', 'yes_ask_close_cents']: 
            if col in df.columns: df[col] = pd.to_numeric(df[col])
        df.set_index('timestamp_s', inplace=True)
        return df
    except Exception as e: logger.error(f"Error loading Kalshi {filepath}: {e}"); return None

def get_kalshi_prices_at_decision(kalshi_df: pd.DataFrame, decision_timestamp_s: int, max_staleness_seconds: int):
    # (Keep the version from your last successful run)
    decision_dt_str = dt.datetime.fromtimestamp(decision_timestamp_s, tz=timezone.utc).isoformat()
    logger.debug(f"Attempting Kalshi prices for decision: {decision_dt_str}")
    if kalshi_df is None or kalshi_df.empty: logger.debug(f"Kalshi df is None or empty for {decision_dt_str}."); return None
    try:
        if decision_timestamp_s in kalshi_df.index:
            row = kalshi_df.loc[decision_timestamp_s]
            logging.debug(f"Exact Kalshi prices at {decision_dt_str}")
            return {"yes_bid": row.get('yes_bid_close_cents'), "yes_ask": row.get('yes_ask_close_cents')}
        else:
            relevant_rows = kalshi_df[kalshi_df.index <= decision_timestamp_s]
            if not relevant_rows.empty:
                latest_row = relevant_rows.iloc[-1]; latest_row_ts = latest_row.name
                time_diff = decision_timestamp_s - latest_row_ts
                if time_diff <= max_staleness_seconds:
                    logging.debug(f"Using stale Kalshi data from {dt.datetime.fromtimestamp(latest_row_ts, tz=timezone.utc)} (Staleness: {time_diff}s)")
                    return {"yes_bid": latest_row.get('yes_bid_close_cents'), "yes_ask": latest_row.get('yes_ask_close_cents')}
                else:
                    logging.warning(f"No recent Kalshi prices for {decision_dt_str}. Latest is {time_diff}s old. Max staleness: {max_staleness_seconds}s.")
                    return None
            else:
                min_idx_ts = kalshi_df.index.min() if not kalshi_df.empty else 'N/A'; max_idx_ts = kalshi_df.index.max() if not kalshi_df.empty else 'N/A'
                min_dt_str = dt.datetime.fromtimestamp(min_idx_ts, tz=timezone.utc).isoformat() if isinstance(min_idx_ts, (int, float)) else min_idx_ts
                max_dt_str = dt.datetime.fromtimestamp(max_idx_ts, tz=timezone.utc).isoformat() if isinstance(max_idx_ts, (int, float)) else max_idx_ts
                logging.warning(f"No Kalshi data at or before {decision_dt_str}. File covers: {min_dt_str} to {max_dt_str}.")
                return None
    except Exception as e: logger.error(f"Error in get_kalshi_prices for {decision_timestamp_s}: {e}"); return None

def clear_binance_cache():
    global _binance_data_cache
    _binance_data_cache = {}
    logger.info("Binance data cache cleared.")