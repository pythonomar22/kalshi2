# /Users/omarabul-hassan/Desktop/projects/kalshi/random/backtest/live_feature_engineering.py

import pandas as pd
import numpy as np
import datetime as dt
from datetime import timezone, timedelta
from pathlib import Path
import logging
import json
import re
from tqdm import tqdm
import os

# Assuming this script is in random/backtest/
import live_backtest_config as config
import live_backtest_utils as utils

# --- Logging Setup ---
logger = logging.getLogger("live_feature_engineering")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

RECORDED_KALSHI_CSV_TIMEZONE = 'America/Los_Angeles' 
logger.info(f"Assuming Kalshi CSV timestamps are recorded in timezone: {RECORDED_KALSHI_CSV_TIMEZONE}")

# +++ ADDED FOR SINGLE SESSION TESTING +++
TARGET_SESSION_ID_FOR_TESTING = "25MAY2015" # Set to None or remove to process all sessions
# +++++++++++++++++++++++++++++++++++++++

# In random/backtest/live_feature_engineering.py

def get_live_btc_kline_snapshot( # Consider renaming to get_previous_completed_btc_kline
    binance_day_df: pd.DataFrame,
    kalshi_decision_ts_sec: int,
    current_market_ticker_for_debug: str = None,
    is_market_being_debugged: bool = False
    ) -> pd.Series | None:
    if binance_day_df is None or binance_day_df.empty:
        return None
    try:
        # Determine the start of the minute for the kline that would have *just closed*
        # before or at the beginning of the minute containing kalshi_decision_ts_sec.
        # e.g., if kalshi_decision_ts_sec is 10:00:05, its minute starts at 10:00:00.
        # The kline that just closed started at 09:59:00.
        decision_minute_start_ts = (kalshi_decision_ts_sec // 60) * 60
        target_kline_start_ts = decision_minute_start_ts - 60 # Start time of the previous, completed kline

        if target_kline_start_ts not in binance_day_df.index:
            if is_market_being_debugged:
                logger.info(f"  [DEBUG LFE - BTC KLINE {current_market_ticker_for_debug}] Target PREVIOUS kline start_ts {target_kline_start_ts} ({dt.datetime.fromtimestamp(target_kline_start_ts, tz=timezone.utc).isoformat()}) not found in Binance data for decision at {dt.datetime.fromtimestamp(kalshi_decision_ts_sec, tz=timezone.utc).isoformat()}.")
            return None
        
        # Retrieve the specific kline by its start time (which is the index)
        kline_data = binance_day_df.loc[target_kline_start_ts]
        # load_and_preprocess_live_binance_data should ensure index is unique

        if is_market_being_debugged:
             logger.info(f"  [DEBUG LFE - BTC KLINE {current_market_ticker_for_debug}] For decision {dt.datetime.fromtimestamp(kalshi_decision_ts_sec, tz=timezone.utc).isoformat()}, using Binance kline that *started* at {dt.datetime.fromtimestamp(kline_data.name, tz=timezone.utc).isoformat()} (close: {kline_data.get('close_price')}) as the *previous completed* kline.")
        return kline_data # kline_data.name will be target_kline_start_ts

    except KeyError: # If target_kline_start_ts is not in index
        if is_market_being_debugged:
            logger.warning(f"  [DEBUG LFE - BTC KLINE {current_market_ticker_for_debug}] KeyError: Target PREVIOUS kline start_ts {target_kline_start_ts} not found for decision at {dt.datetime.fromtimestamp(kalshi_decision_ts_sec, tz=timezone.utc).isoformat()}.")
        return None
    except Exception as e:
        # Keep existing error logging for other unexpected issues
        if is_market_being_debugged:
            logger.error(f"  [DEBUG LFE - BTC KLINE {current_market_ticker_for_debug}] Error getting PREVIOUS Binance kline for decision_ts {kalshi_decision_ts_sec} (target_kline_start_ts {target_kline_start_ts if 'target_kline_start_ts' in locals() else 'N/A'}): {e}", exc_info=True)
        return None

def main():
    logger.info("Starting LIVE feature engineering process...")
    if TARGET_SESSION_ID_FOR_TESTING:
        logger.info(f"+++ TESTING MODE: Processing ONLY for session ID: {TARGET_SESSION_ID_FOR_TESTING} +++")

    utils.clear_binance_live_cache()

    if not config.LIVE_MARKET_OUTCOMES_CSV.exists():
        logger.critical(f"Live market outcomes file not found: {config.LIVE_MARKET_OUTCOMES_CSV}. Run fetch.py.")
        return
    try:
        outcomes_df = pd.read_csv(config.LIVE_MARKET_OUTCOMES_CSV)
        outcomes_df['target'] = outcomes_df['result'].apply(lambda x: 1 if str(x).lower() == 'yes' else (0 if str(x).lower() == 'no' else np.nan))
        outcomes_df.dropna(subset=['target'], inplace=True)
        outcomes_df['target'] = outcomes_df['target'].astype(int)
        outcomes_df['resolution_time_ts'] = outcomes_df['close_time_iso'].apply(utils.parse_iso_to_unix_timestamp)
        outcomes_df['market_open_ts'] = outcomes_df['open_time_iso'].apply(utils.parse_iso_to_unix_timestamp)
        outcomes_df.rename(columns={'strike_price': 'kalshi_strike_price'}, inplace=True)
        outcomes_df.dropna(subset=['market_ticker', 'resolution_time_ts', 'kalshi_strike_price', 'target', 'market_open_ts'], inplace=True)
        
        # +++ FILTER OUTCOMES FOR TARGET SESSION +++
        if TARGET_SESSION_ID_FOR_TESTING:
            outcomes_df = outcomes_df[outcomes_df['market_ticker'].str.contains(f"-{TARGET_SESSION_ID_FOR_TESTING}-T")]
            if outcomes_df.empty:
                logger.critical(f"No market outcomes found for target session {TARGET_SESSION_ID_FOR_TESTING}. Check session ID and outcomes file.")
                return
            logger.info(f"Filtered outcomes to {len(outcomes_df)} markets for session {TARGET_SESSION_ID_FOR_TESTING}.")
        # +++++++++++++++++++++++++++++++++++++++++

        logger.info(f"Loaded {len(outcomes_df)} settled markets with outcomes (after potential session filter).")
        if outcomes_df.empty:
            logger.warning("No settled markets in outcomes file (after potential session filter). Exiting feature engineering.")
            return
    except Exception as e:
        logger.critical(f"Error loading or processing outcomes CSV {config.LIVE_MARKET_OUTCOMES_CSV}: {e}", exc_info=True)
        return

    all_decision_point_features_list = []
    processed_kalshi_files = 0
    filtered_out_market_debug_count = 0 
    MAX_FILTERED_MARKETS_TO_DEBUG = 5 

    # --- Iterate through Kalshi market data files ---
    # +++ FILTER KALSHI CSV FILES FOR TARGET SESSION +++
    if TARGET_SESSION_ID_FOR_TESTING:
        kalshi_csv_files = list(config.LIVE_KALSHI_DATA_DIR.glob(f"*-{TARGET_SESSION_ID_FOR_TESTING}-T*.csv"))
        if not kalshi_csv_files:
            logger.critical(f"No Kalshi CSV files found for target session {TARGET_SESSION_ID_FOR_TESTING} in {config.LIVE_KALSHI_DATA_DIR}.")
            return
    else:
        kalshi_csv_files = list(config.LIVE_KALSHI_DATA_DIR.glob("KXBTCD-*.csv"))
    # +++++++++++++++++++++++++++++++++++++++++++++++
    
    logger.info(f"Found {len(kalshi_csv_files)} live Kalshi market CSV files to process (after potential session filter).")

    DEBUG_MAX_KALSHI_FILES = 0 
    DEBUG_MAX_DECISION_POINTS_PER_FILE = 0 

    for kalshi_file_path in tqdm(kalshi_csv_files, desc="Processing Kalshi Market Files"):
        market_ticker_from_filename = kalshi_file_path.stem
        
        market_outcome_info = outcomes_df[outcomes_df['market_ticker'] == market_ticker_from_filename]
        if market_outcome_info.empty:
            logger.debug(f"No outcome data for {market_ticker_from_filename} (it might have been filtered by session). Skipping.")
            continue
        
        market_outcome_info = market_outcome_info.iloc[0]
        target_outcome = market_outcome_info['target']
        resolution_time_ts = int(market_outcome_info['resolution_time_ts'])
        kalshi_strike_price = float(market_outcome_info['kalshi_strike_price'])
        market_open_ts = int(market_outcome_info['market_open_ts'])

        ticker_details = utils.parse_live_kalshi_ticker_details(market_ticker_from_filename)
        if not ticker_details or "session_id" not in ticker_details:
            logger.warning(f"Could not parse session_id from {market_ticker_from_filename}, skipping.")
            continue
        
        session_id = ticker_details['session_id']
        # +++ CHECK IF SESSION_ID MATCHES TARGET (should already be filtered by kalshi_csv_files glob) +++
        if TARGET_SESSION_ID_FOR_TESTING and session_id != TARGET_SESSION_ID_FOR_TESTING:
            logger.debug(f"Skipping {market_ticker_from_filename} as its session_id '{session_id}' does not match target '{TARGET_SESSION_ID_FOR_TESTING}'.")
            continue
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        binance_filename = config.SESSION_TO_BINANCE_FILE_MAP.get(session_id)
        if not binance_filename:
            logger.warning(f"No Binance file mapping for session {session_id} (market {market_ticker_from_filename}), skipping.")
            continue
        
        binance_live_df_path = config.LIVE_BINANCE_DATA_DIR / binance_filename
        current_binance_processed_df = utils.load_and_preprocess_live_binance_data(binance_live_df_path)
        if current_binance_processed_df is None or current_binance_processed_df.empty:
            logger.warning(f"No Binance data loaded or processed for {binance_filename} (market {market_ticker_from_filename}), skipping.")
            continue

        try:
            kalshi_market_df_raw = pd.read_csv(kalshi_file_path)
            if kalshi_market_df_raw.empty:
                logger.info(f"Kalshi market data file {kalshi_file_path.name} is empty.")
                continue

            kalshi_market_df_raw['decision_dt_naive'] = pd.to_datetime(kalshi_market_df_raw['timestamp'], errors='coerce')
            kalshi_market_df_raw.dropna(subset=['decision_dt_naive'], inplace=True)

            try:
                kalshi_market_df_raw['decision_dt_localized'] = kalshi_market_df_raw['decision_dt_naive'].dt.tz_localize(RECORDED_KALSHI_CSV_TIMEZONE)
            except Exception as tz_localize_err:
                 logger.error(f"Error localizing timestamps for {kalshi_file_path.name} to {RECORDED_KALSHI_CSV_TIMEZONE}: {tz_localize_err}. Falling back to UTC assumption for this file or skipping.")
                 if kalshi_market_df_raw['decision_dt_naive'].dt.tz is not None:
                     kalshi_market_df_raw['decision_dt_localized'] = kalshi_market_df_raw['decision_dt_naive'].dt.tz_convert('UTC')
                 else: 
                     logger.warning(f"  Assuming UTC for {kalshi_file_path.name} due to localization error.")
                     kalshi_market_df_raw['decision_dt_localized'] = kalshi_market_df_raw['decision_dt_naive'].dt.tz_localize('UTC')

            kalshi_market_df_raw['decision_dt_utc'] = kalshi_market_df_raw['decision_dt_localized'].dt.tz_convert('UTC')
            kalshi_market_df_raw['decision_timestamp_s'] = kalshi_market_df_raw['decision_dt_utc'].apply(lambda x: int(x.timestamp()))
            
            kalshi_market_df_raw.sort_values(by='decision_timestamp_s', inplace=True)
            
            raw_min_decision_ts = kalshi_market_df_raw['decision_timestamp_s'].min()
            raw_max_decision_ts = kalshi_market_df_raw['decision_timestamp_s'].max()

            first_valid_decision_ts = market_open_ts 
            last_valid_decision_ts = resolution_time_ts - (config.MIN_MINUTES_BEFORE_RESOLUTION_FOR_DECISION * 60)
            
            kalshi_market_df_filtered = kalshi_market_df_raw[
                (kalshi_market_df_raw['decision_timestamp_s'] >= first_valid_decision_ts) &
                (kalshi_market_df_raw['decision_timestamp_s'] <= last_valid_decision_ts)
            ]

            if kalshi_market_df_filtered.empty:
                if filtered_out_market_debug_count < MAX_FILTERED_MARKETS_TO_DEBUG:
                    logger.warning(f"--- FILTERING DEBUG FOR MARKET: {market_ticker_from_filename} ---")
                    logger.warning(f"  Market Open TS (Outcome File, UTC): {market_open_ts} ({dt.datetime.fromtimestamp(market_open_ts, tz=timezone.utc).isoformat()})")
                    logger.warning(f"  Market Resolution TS (Outcome File, UTC): {resolution_time_ts} ({dt.datetime.fromtimestamp(resolution_time_ts, tz=timezone.utc).isoformat()})")
                    logger.warning(f"  Calculated First Valid Decision TS (UTC): {first_valid_decision_ts} ({dt.datetime.fromtimestamp(first_valid_decision_ts, tz=timezone.utc).isoformat()})")
                    logger.warning(f"  Calculated Last Valid Decision TS (UTC): {last_valid_decision_ts} ({dt.datetime.fromtimestamp(last_valid_decision_ts, tz=timezone.utc).isoformat()})")
                    logger.warning(f"  MIN Decision TS in RAW Kalshi CSV (Converted to UTC): {raw_min_decision_ts} ({dt.datetime.fromtimestamp(raw_min_decision_ts, tz=timezone.utc).isoformat() if pd.notna(raw_min_decision_ts) else 'N/A'})")
                    logger.warning(f"  MAX Decision TS in RAW Kalshi CSV (Converted to UTC): {raw_max_decision_ts} ({dt.datetime.fromtimestamp(raw_max_decision_ts, tz=timezone.utc).isoformat() if pd.notna(raw_max_decision_ts) else 'N/A'})")
                    logger.warning(f"  Number of points in raw Kalshi CSV: {len(kalshi_market_df_raw)}")
                    logger.warning(f"  Result: All points filtered out.")
                    logger.warning(f"--- END FILTERING DEBUG FOR MARKET: {market_ticker_from_filename} ---")
                    filtered_out_market_debug_count +=1
                else:
                    logger.info(f"No valid decision points for {market_ticker_from_filename} after time filtering (further debug logs suppressed).")
                continue 
            
            logger.info(f"Processing {len(kalshi_market_df_filtered)} decision points for {market_ticker_from_filename}")
            kalshi_market_df = kalshi_market_df_filtered 

        except Exception as e:
            logger.error(f"Error reading or processing Kalshi file {kalshi_file_path.name}: {e}", exc_info=True)
            continue
        
        decision_points_count_for_file = 0
        is_current_market_debugged = DEBUG_MAX_KALSHI_FILES > 0 and processed_kalshi_files < DEBUG_MAX_KALSHI_FILES

        for _, kalshi_row in kalshi_market_df.iterrows():
            if DEBUG_MAX_DECISION_POINTS_PER_FILE > 0 and decision_points_count_for_file >= DEBUG_MAX_DECISION_POINTS_PER_FILE:
                break

            decision_ts_sec = kalshi_row['decision_timestamp_s']
            
            features = {
                'market_ticker': market_ticker_from_filename,
                'decision_timestamp_s': decision_ts_sec,
                'resolution_time_ts': resolution_time_ts,
                'strike_price': kalshi_strike_price,
                'target': target_outcome,
                'time_to_resolution_minutes': round((resolution_time_ts - decision_ts_sec) / 60.0, 2)
            }

            current_btc_kline_info = get_live_btc_kline_snapshot(current_binance_processed_df, decision_ts_sec, market_ticker_from_filename, is_current_market_debugged)

            if current_btc_kline_info is not None and pd.notna(current_btc_kline_info.get('close_price')):
                features['current_btc_price'] = float(current_btc_kline_info['close_price'])
                features['current_dist_strike_abs'] = features['current_btc_price'] - kalshi_strike_price
                features['current_dist_strike_pct'] = (features['current_dist_strike_abs'] / kalshi_strike_price) if kalshi_strike_price != 0 else np.nan
                
                btc_price_history_for_stats = current_binance_processed_df[
                    current_binance_processed_df.index <= current_btc_kline_info.name 
                ]['close_price'].copy()

                if not btc_price_history_for_stats.empty:
                    temp_series_for_asof = pd.Series(
                        btc_price_history_for_stats.values,
                        index=pd.to_datetime(btc_price_history_for_stats.index, unit='s', utc=True)
                    )
                    
                    current_kline_start_dt_for_asof = pd.Timestamp(current_btc_kline_info.name, unit='s', tz='utc')

                    for lag in config.LAG_WINDOWS_MINUTES:
                        target_lag_dt = current_kline_start_dt_for_asof - timedelta(minutes=lag)
                        past_price_at_lag_kline = temp_series_for_asof.asof(target_lag_dt)
                        
                        if pd.notna(past_price_at_lag_kline) and pd.notna(features.get('current_btc_price')):
                            features[f'btc_price_change_pct_{lag}m'] = (features['current_btc_price'] - past_price_at_lag_kline) / past_price_at_lag_kline if past_price_at_lag_kline != 0 else np.nan
                        else:
                            features[f'btc_price_change_pct_{lag}m'] = np.nan
                    
                    for window in config.ROLLING_WINDOWS_MINUTES:
                        if len(btc_price_history_for_stats) >= window:
                            std_val = btc_price_history_for_stats.iloc[-window:].std()
                        elif len(btc_price_history_for_stats) >= 2:
                            std_val = btc_price_history_for_stats.std()
                        else:
                            std_val = np.nan
                        features[f'btc_volatility_{window}m'] = std_val
                else: 
                    for lag in config.LAG_WINDOWS_MINUTES: features[f'btc_price_change_pct_{lag}m'] = np.nan
                    for window in config.ROLLING_WINDOWS_MINUTES: features[f'btc_volatility_{window}m'] = np.nan
            else: 
                features.update({f:np.nan for f in ['current_btc_price','current_dist_strike_abs','current_dist_strike_pct']})
                for lag in config.LAG_WINDOWS_MINUTES: features[f'btc_price_change_pct_{lag}m'] = np.nan
                for window in config.ROLLING_WINDOWS_MINUTES: features[f'btc_volatility_{window}m'] = np.nan

            features['current_kalshi_yes_bid'] = kalshi_row.get('yes_bid_price_cents', np.nan) 
            if pd.notna(features['current_kalshi_yes_bid']): features['current_kalshi_yes_bid'] /= 100.0
            
            features['current_kalshi_yes_ask'] = kalshi_row.get('yes_ask_price_cents', np.nan)
            if pd.notna(features['current_kalshi_yes_ask']): features['current_kalshi_yes_ask'] /= 100.0

            if pd.notna(features['current_kalshi_yes_bid']) and pd.notna(features['current_kalshi_yes_ask']):
                features['current_kalshi_mid_price'] = (features['current_kalshi_yes_bid'] + features['current_kalshi_yes_ask']) / 2.0
                features['current_kalshi_spread_abs'] = features['current_kalshi_yes_ask'] - features['current_kalshi_yes_bid']
                features['current_kalshi_spread_pct'] = (features['current_kalshi_spread_abs'] / features['current_kalshi_mid_price']) if features['current_kalshi_mid_price'] != 0 else np.nan
            else:
                features.update({f:np.nan for f in ['current_kalshi_mid_price','current_kalshi_spread_abs','current_kalshi_spread_pct']})
            
            features['current_kalshi_volume'] = np.nan 
            features['current_kalshi_oi'] = np.nan

            all_decision_point_features_list.append(features)
            decision_points_count_for_file += 1
        
        processed_kalshi_files += 1
        if DEBUG_MAX_KALSHI_FILES > 0 and processed_kalshi_files >= DEBUG_MAX_KALSHI_FILES:
            logger.info(f"Reached DEBUG_MAX_KALSHI_FILES ({DEBUG_MAX_KALSHI_FILES}). Stopping.")
            break

    if all_decision_point_features_list:
        output_features_df = pd.DataFrame(all_decision_point_features_list)
        timestamp_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # +++ MODIFY FILENAME IF TESTING SINGLE SESSION +++
        if TARGET_SESSION_ID_FOR_TESTING:
            features_filename = f"kalshi_live_decision_features_SESSION_{TARGET_SESSION_ID_FOR_TESTING}_{timestamp_str}.csv"
        else:
            features_filename = f"kalshi_live_decision_features_{timestamp_str}.csv"
        # +++++++++++++++++++++++++++++++++++++++++++++++

        features_filepath = config.LIVE_FEATURES_DIR / features_filename
        try:
            output_features_df.to_csv(features_filepath, index=False)
            logger.info(f"Successfully saved {len(output_features_df)} LIVE decision point features to: {features_filepath}")
            if not output_features_df.empty:
                print(f"Live features saved to: {features_filepath.resolve()}")
                print("Sample of generated features:")
                print(output_features_df.head().to_string())
            
            nan_summary = (output_features_df.isnull().sum() / len(output_features_df) * 100)
            nan_summary = nan_summary[nan_summary > 0].sort_values(ascending=False)
            if not nan_summary.empty:
                logger.info("NaN percentage per column in generated live features:")
                print(nan_summary.to_string())
            else:
                logger.info("No NaNs found in generated live features.")
        except Exception as e:
            logger.error(f"Error saving live features: {e}", exc_info=True)
    else:
        logger.warning("No live decision point features were generated.")

    logger.info("Live feature engineering process complete.")

if __name__ == "__main__":
    main()