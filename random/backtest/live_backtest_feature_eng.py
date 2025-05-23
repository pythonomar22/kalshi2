# /random/backtest/live_backtest_feature_eng.py

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone, timedelta

# Import from live_backtest_config and live_backtest_data_utils
try:
    from . import live_backtest_config as live_cfg
    from . import live_backtest_data_utils as live_data_utils
except ImportError: # For direct script execution if needed
    import live_backtest_config as live_cfg
    import live_backtest_data_utils as live_data_utils

logger = logging.getLogger("live_backtest_feature_eng")

def get_kalshi_state_at_or_before_ts(kalshi_market_df: pd.DataFrame, decision_ts_utc: int) -> pd.Series | None:
    """
    Gets the latest Kalshi market state (bid/ask) from the live data at or before the decision_ts_utc.
    Assumes kalshi_market_df is indexed by 'timestamp_s_utc' and sorted.
    """
    if kalshi_market_df is None or kalshi_market_df.empty:
        return None
    try:
        # Use searchsorted to find the insertion point for decision_ts_utc
        idx_pos = kalshi_market_df.index.searchsorted(decision_ts_utc, side='right')
        if idx_pos == 0:
            return None # No data at or before this time
        
        # The actual data point is at index idx_pos - 1
        latest_kalshi_state = kalshi_market_df.iloc[idx_pos - 1]
        
        # Ensure the found data is not too old (e.g., more than a few minutes)
        # This helps avoid using stale data if there are gaps in live collection
        if decision_ts_utc - latest_kalshi_state.name > (3 * 60): # More than 3 mins old
            # logger.debug(f"Stale Kalshi data for ts {decision_ts_utc}, found data is from {latest_kalshi_state.name}")
            return None
        return latest_kalshi_state

    except IndexError:
        return None
    except Exception as e:
        logger.error(f"Error in get_kalshi_state_at_or_before_ts for ts {decision_ts_utc}: {e}", exc_info=False)
        return None


def get_binance_kline_at_or_before_ts(binance_session_df: pd.DataFrame, decision_ts_utc: int) -> pd.Series | None:
    """
    Gets the latest Binance kline data from the live session data at or before the decision_ts_utc.
    Assumes binance_session_df is indexed by 'timestamp_s_utc' (kline start time) and sorted.
    The kline's data is valid for the minute that starts at its index.
    So, for a decision_ts_utc, we need the kline whose start time is <= decision_ts_utc.
    """
    if binance_session_df is None or binance_session_df.empty:
        return None
    try:
        idx_pos = binance_session_df.index.searchsorted(decision_ts_utc, side='right')
        if idx_pos == 0:
            return None
        
        # This kline started at or before decision_ts_utc
        # Its data (open, close, etc.) pertains to the interval [kline.name, kline.name + 59s]
        kline_data = binance_session_df.iloc[idx_pos - 1]

        # Ensure the kline is not for a future minute relative to decision_ts_utc
        # This should be handled by searchsorted, but double check
        if kline_data.name > decision_ts_utc :
             logger.warning(f"Binance kline lookahead: kline_ts {kline_data.name} > decision_ts {decision_ts_utc}")
             return None # Should not happen with side='right' and iloc[idx_pos-1]
        
        return kline_data

    except IndexError:
        return None
    except Exception as e:
        logger.error(f"Error in get_binance_kline_at_or_before_ts: {e}", exc_info=False)
        return None

def generate_features_for_market_live(
    market_info_row: pd.Series, # Row from live_market_outcomes_df
    model_feature_names: list
) -> pd.DataFrame:
    """
    Generates features for a single market using live-collected data.
    Iterates minute-by-minute from market open to close (resolution - buffer).
    """
    market_ticker = market_info_row['market_ticker']
    kalshi_strike_price = market_info_row['strike_price']
    market_open_ts_utc = market_info_row['open_time_ts_utc']
    resolution_time_ts_utc = market_info_row['resolution_time_ts_utc']
    actual_target_outcome = market_info_row['target']

    session_key = live_data_utils.parse_market_ticker_session_key(market_ticker)
    if not session_key:
        logger.error(f"Could not determine session key for {market_ticker}. Skipping feature gen.")
        return pd.DataFrame()

    # Load necessary data for this market
    # Caching within these load functions will prevent re-reading from disk for same session/market
    live_kalshi_df = live_data_utils.load_live_kalshi_market_data(market_ticker)
    live_binance_df = live_data_utils.load_live_binance_data_for_session(session_key)

    if live_kalshi_df is None or live_binance_df is None:
        logger.warning(f"Missing Kalshi or Binance live data for {market_ticker} (session {session_key}). Skipping.")
        return pd.DataFrame()

    market_decision_features = []

    # Iterate minute by minute from market open up to X minutes before resolution
    # Decision timestamps should be at the START of the minute for which we are making a decision
    # e.g., if market opens 12:00:00, first decision could be for 12:01:00 (using data up to 12:00:59)
    
    # The Kalshi market open_time is when it's listed. Trades might start slightly after.
    # Let's start making decisions from one full minute after market_open_ts_utc.
    current_decision_ts_utc = market_open_ts_utc + 60 
    # current_decision_ts_utc = market_open_ts_utc # Or start right at open_time
    
    last_possible_decision_ts = resolution_time_ts_utc - (live_cfg.MIN_MINUTES_BEFORE_RESOLUTION_FOR_DECISION * 60)

    while current_decision_ts_utc <= last_possible_decision_ts:
        features = {
            'market_ticker': market_ticker,
            'decision_timestamp_s': current_decision_ts_utc, # This is the time of decision
            'resolution_time_ts': resolution_time_ts_utc,
            'strike_price': kalshi_strike_price,
            'target': actual_target_outcome, # Actual outcome for P&L calculation later
            'time_to_resolution_minutes': round((resolution_time_ts_utc - current_decision_ts_utc) / 60.0, 2)
        }

        # --- Binance Features (BTC price, lags, volatility) ---
        # We need Binance kline data at or *just before* current_decision_ts_utc
        # The kline whose start_time is current_decision_ts_utc - 60s gives data for the minute ending *at* current_decision_ts_utc
        btc_kline_for_decision = get_binance_kline_at_or_before_ts(live_binance_df, current_decision_ts_utc -1) # data for minute just ending
        
        if btc_kline_for_decision is not None and pd.notna(btc_kline_for_decision['close']):
            features['current_btc_price'] = float(btc_kline_for_decision['close'])
            features['current_dist_strike_abs'] = features['current_btc_price'] - kalshi_strike_price
            features['current_dist_strike_pct'] = (features['current_dist_strike_abs'] / kalshi_strike_price) if kalshi_strike_price != 0 else np.nan
            
            # For lags and rolling vols, we need a history series ending at btc_kline_for_decision.name
            # The live_binance_df is already per-minute klines for the session.
            # We need to slice it up to and including btc_kline_for_decision.name
            
            # Max lookback needed for any feature
            max_hist_window_min = max(live_cfg.LAG_WINDOWS_MINUTES + live_cfg.ROLLING_WINDOWS_MINUTES)
            history_start_ts = btc_kline_for_decision.name - (max_hist_window_min * 60)
            
            # Slice the session's Binance data for relevant history
            btc_price_series_for_stats = live_binance_df.loc[
                (live_binance_df.index >= history_start_ts) &
                (live_binance_df.index <= btc_kline_for_decision.name)
            ]['close']
            
            if not btc_price_series_for_stats.empty:
                # Convert index to DatetimeIndex for asof if not already (it should be from load_live_binance)
                # temp_series_for_asof = pd.Series(btc_price_series_for_stats.values,
                #                                  index=pd.to_datetime(btc_price_series_for_stats.index, unit='s', utc=True))
                # live_binance_df index is already timestamp_s_utc (numeric)
                temp_series_for_asof = btc_price_series_for_stats # Already numeric index

                for lag in live_cfg.LAG_WINDOWS_MINUTES:
                    target_lag_ts = btc_kline_for_decision.name - (lag * 60)
                    # Find price at or before target_lag_ts using searchsorted on numeric index
                    idx_pos = temp_series_for_asof.index.searchsorted(target_lag_ts, side='right')
                    past_price = np.nan
                    if idx_pos > 0:
                        past_price_timestamp = temp_series_for_asof.index[idx_pos - 1]
                        # Ensure the found past price is indeed for the target lag period
                        if past_price_timestamp <= target_lag_ts: # and target_lag_ts - past_price_timestamp < 60 (within the minute)
                             past_price = temp_series_for_asof.iloc[idx_pos - 1]

                    if pd.notna(past_price) and pd.notna(features.get('current_btc_price')) and past_price != 0:
                        features[f'btc_price_change_pct_{lag}m'] = (features['current_btc_price'] - past_price) / past_price
                    else:
                        features[f'btc_price_change_pct_{lag}m'] = np.nan
                
                for window in live_cfg.ROLLING_WINDOWS_MINUTES:
                    # We need 'window' number of 1-minute klines ending at btc_kline_for_decision.name
                    # Slice the series for the window period
                    rolling_window_data = temp_series_for_asof.loc[
                        (temp_series_for_asof.index > btc_kline_for_decision.name - (window * 60)) &
                        (temp_series_for_asof.index <= btc_kline_for_decision.name)
                    ]
                    if len(rolling_window_data) >= 2 : # Min periods for std
                        std_val = rolling_window_data.std()
                    else:
                        std_val = np.nan
                    features[f'btc_volatility_{window}m'] = std_val
            else: # Not enough btc history
                for lag in live_cfg.LAG_WINDOWS_MINUTES: features[f'btc_price_change_pct_{lag}m'] = np.nan
                for window in live_cfg.ROLLING_WINDOWS_MINUTES: features[f'btc_volatility_{window}m'] = np.nan
        else: # No current BTC kline found
            features.update({f:np.nan for f in ['current_btc_price','current_dist_strike_abs','current_dist_strike_pct']})
            for lag in live_cfg.LAG_WINDOWS_MINUTES: features[f'btc_price_change_pct_{lag}m'] = np.nan
            for window in live_cfg.ROLLING_WINDOWS_MINUTES: features[f'btc_volatility_{window}m'] = np.nan

        # --- Kalshi Market Features ---
        # Get Kalshi state at or just before current_decision_ts_utc
        current_kalshi_state = get_kalshi_state_at_or_before_ts(live_kalshi_df, current_decision_ts_utc -1) # Data for minute just ending

        if current_kalshi_state is not None:
            # Prices are already in cents from live data collection
            yes_bid = current_kalshi_state.get('yes_bid_price_cents', np.nan)
            yes_ask = current_kalshi_state.get('yes_ask_price_cents', np.nan)
            
            # Convert to dollars for feature consistency if model expects that (historical features did this)
            features['current_kalshi_yes_bid'] = yes_bid / 100.0 if pd.notna(yes_bid) else np.nan
            features['current_kalshi_yes_ask'] = yes_ask / 100.0 if pd.notna(yes_ask) else np.nan
            
            # Volume and OI are not used by 'no_vol_oi' model, but if they were:
            # features['current_kalshi_volume'] = current_kalshi_state.get('volume', np.nan) # Need to define how 'volume' is aggregated for the minute
            # features['current_kalshi_oi'] = current_kalshi_state.get('open_interest', np.nan) # OI is usually in snapshots

            if pd.notna(features['current_kalshi_yes_bid']) and pd.notna(features['current_kalshi_yes_ask']):
                features['current_kalshi_mid_price'] = (features['current_kalshi_yes_bid'] + features['current_kalshi_yes_ask']) / 2.0
                features['current_kalshi_spread_abs'] = features['current_kalshi_yes_ask'] - features['current_kalshi_yes_bid']
                features['current_kalshi_spread_pct'] = (features['current_kalshi_spread_abs'] / features['current_kalshi_mid_price']) if features.get('current_kalshi_mid_price', 0) != 0 else np.nan
            else:
                features.update({f:np.nan for f in ['current_kalshi_mid_price','current_kalshi_spread_abs','current_kalshi_spread_pct']})
        else: # No current Kalshi state found
            features.update({f:np.nan for f in ['current_kalshi_yes_bid','current_kalshi_yes_ask','current_kalshi_mid_price','current_kalshi_spread_abs','current_kalshi_spread_pct']})
            # also volume/oi if used: 'current_kalshi_volume','current_kalshi_oi'
        
        # Ensure all model features are present, filling with NaN if calculated one is missing
        final_feature_row = {}
        for f_name in model_feature_names:
            final_feature_row[f_name] = features.get(f_name, np.nan)
        
        # Add back non-model essential columns for the engine
        final_feature_row['market_ticker'] = features['market_ticker']
        final_feature_row['decision_timestamp_s'] = features['decision_timestamp_s']
        final_feature_row['resolution_time_ts'] = features['resolution_time_ts']
        final_feature_row['target'] = features['target']
        # 'strike_price' is already in model_feature_names
        # 'time_to_resolution_minutes' is already in model_feature_names

        market_decision_features.append(final_feature_row)
        current_decision_ts_utc += 60 # Move to the next minute

    return pd.DataFrame(market_decision_features)