# /random/backtest/live_backtest_feature_eng.py

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone, timedelta # Keep this
# import datetime as dt # Not strictly needed if using datetime.datetime directly

# Changed from relative to direct import
import live_backtest_config as live_cfg
import live_backtest_data_utils as live_data_utils

logger = logging.getLogger("live_backtest_feature_eng")

def get_kalshi_state_at_or_before_ts(kalshi_market_df: pd.DataFrame, decision_ts_utc: int) -> pd.Series | None:
    if kalshi_market_df is None or kalshi_market_df.empty: return None
    try:
        idx_pos = kalshi_market_df.index.searchsorted(decision_ts_utc, side='right')
        if idx_pos == 0: return None 
        latest_kalshi_state = kalshi_market_df.iloc[idx_pos - 1]
        if decision_ts_utc - latest_kalshi_state.name > (3 * 60): return None
        return latest_kalshi_state
    except IndexError: return None
    except Exception as e:
        logger.error(f"Error in get_kalshi_state_at_or_before_ts for ts {decision_ts_utc}: {e}", exc_info=False)
        return None

def get_binance_kline_at_or_before_ts(binance_session_df: pd.DataFrame, decision_ts_utc: int) -> pd.Series | None:
    if binance_session_df is None or binance_session_df.empty: return None
    try:
        idx_pos = binance_session_df.index.searchsorted(decision_ts_utc, side='right')
        if idx_pos == 0: return None
        kline_data = binance_session_df.iloc[idx_pos - 1]
        if kline_data.name > decision_ts_utc :
             logger.warning(f"Binance kline lookahead: kline_ts {kline_data.name} > decision_ts {decision_ts_utc}")
             return None 
        return kline_data
    except IndexError: return None
    except Exception as e:
        logger.error(f"Error in get_binance_kline_at_or_before_ts: {e}", exc_info=False)
        return None

def generate_features_for_market_live(
    market_info_row: pd.Series, model_feature_names: list
) -> pd.DataFrame:
    market_ticker = market_info_row['market_ticker']
    kalshi_strike_price = market_info_row['strike_price']
    market_open_ts_utc = market_info_row['open_time_ts_utc']
    resolution_time_ts_utc = market_info_row['resolution_time_ts_utc']
    actual_target_outcome = market_info_row['target']
    session_key = live_data_utils.parse_market_ticker_session_key(market_ticker)
    if not session_key:
        logger.error(f"Could not determine session key for {market_ticker}. Skipping feature gen.")
        return pd.DataFrame()
    live_kalshi_df = live_data_utils.load_live_kalshi_market_data(market_ticker)
    live_binance_df = live_data_utils.load_live_binance_data_for_session(session_key)
    if live_kalshi_df is None or live_binance_df is None:
        logger.warning(f"Missing Kalshi or Binance live data for {market_ticker} (session {session_key}). Skipping.")
        return pd.DataFrame()
    market_decision_features = []
    current_decision_ts_utc = market_open_ts_utc + 60 
    last_possible_decision_ts = resolution_time_ts_utc - (live_cfg.MIN_MINUTES_BEFORE_RESOLUTION_FOR_DECISION * 60)
    while current_decision_ts_utc <= last_possible_decision_ts:
        features = {'market_ticker':market_ticker,'decision_timestamp_s':current_decision_ts_utc,'resolution_time_ts':resolution_time_ts_utc,'strike_price':kalshi_strike_price,'target':actual_target_outcome,'time_to_resolution_minutes':round((resolution_time_ts_utc - current_decision_ts_utc)/60.0,2)}
        btc_kline_for_decision = get_binance_kline_at_or_before_ts(live_binance_df, current_decision_ts_utc -1) 
        if btc_kline_for_decision is not None and pd.notna(btc_kline_for_decision['close']):
            features['current_btc_price'] = float(btc_kline_for_decision['close'])
            features['current_dist_strike_abs'] = features['current_btc_price'] - kalshi_strike_price
            features['current_dist_strike_pct'] = (features['current_dist_strike_abs']/kalshi_strike_price) if kalshi_strike_price!=0 else np.nan
            max_hist_window_min = max(live_cfg.LAG_WINDOWS_MINUTES + live_cfg.ROLLING_WINDOWS_MINUTES)
            history_start_ts = btc_kline_for_decision.name - (max_hist_window_min * 60)
            btc_price_series_for_stats = live_binance_df.loc[(live_binance_df.index >= history_start_ts) & (live_binance_df.index <= btc_kline_for_decision.name)]['close']
            if not btc_price_series_for_stats.empty:
                temp_series_for_asof = btc_price_series_for_stats 
                for lag in live_cfg.LAG_WINDOWS_MINUTES:
                    target_lag_ts = btc_kline_for_decision.name - (lag * 60)
                    idx_pos = temp_series_for_asof.index.searchsorted(target_lag_ts, side='right')
                    past_price = np.nan
                    if idx_pos > 0:
                        past_price_timestamp = temp_series_for_asof.index[idx_pos - 1]
                        if past_price_timestamp <= target_lag_ts: past_price = temp_series_for_asof.iloc[idx_pos - 1]
                    if pd.notna(past_price) and pd.notna(features.get('current_btc_price')) and past_price != 0: features[f'btc_price_change_pct_{lag}m'] = (features['current_btc_price'] - past_price) / past_price
                    else: features[f'btc_price_change_pct_{lag}m'] = np.nan
                for window in live_cfg.ROLLING_WINDOWS_MINUTES:
                    rolling_window_data = temp_series_for_asof.loc[(temp_series_for_asof.index > btc_kline_for_decision.name - (window * 60)) & (temp_series_for_asof.index <= btc_kline_for_decision.name)]
                    if len(rolling_window_data) >= 2 : std_val = rolling_window_data.std()
                    else: std_val = np.nan
                    features[f'btc_volatility_{window}m'] = std_val
            else: 
                for lag in live_cfg.LAG_WINDOWS_MINUTES: features[f'btc_price_change_pct_{lag}m'] = np.nan
                for window in live_cfg.ROLLING_WINDOWS_MINUTES: features[f'btc_volatility_{window}m'] = np.nan
        else: 
            features.update({f:np.nan for f in ['current_btc_price','current_dist_strike_abs','current_dist_strike_pct']})
            for lag in live_cfg.LAG_WINDOWS_MINUTES: features[f'btc_price_change_pct_{lag}m'] = np.nan
            for window in live_cfg.ROLLING_WINDOWS_MINUTES: features[f'btc_volatility_{window}m'] = np.nan
        current_kalshi_state = get_kalshi_state_at_or_before_ts(live_kalshi_df, current_decision_ts_utc -1) 
        if current_kalshi_state is not None:
            yes_bid = current_kalshi_state.get('yes_bid_price_cents', np.nan); yes_ask = current_kalshi_state.get('yes_ask_price_cents', np.nan)
            features['current_kalshi_yes_bid'] = yes_bid / 100.0 if pd.notna(yes_bid) else np.nan
            features['current_kalshi_yes_ask'] = yes_ask / 100.0 if pd.notna(yes_ask) else np.nan
            if pd.notna(features['current_kalshi_yes_bid']) and pd.notna(features['current_kalshi_yes_ask']):
                features['current_kalshi_mid_price'] = (features['current_kalshi_yes_bid'] + features['current_kalshi_yes_ask']) / 2.0
                features['current_kalshi_spread_abs'] = features['current_kalshi_yes_ask'] - features['current_kalshi_yes_bid']
                features['current_kalshi_spread_pct'] = (features['current_kalshi_spread_abs'] / features.get('current_kalshi_mid_price', 0)) if features.get('current_kalshi_mid_price', 0) != 0 else np.nan
            else: features.update({f:np.nan for f in ['current_kalshi_mid_price','current_kalshi_spread_abs','current_kalshi_spread_pct']})
        else: features.update({f:np.nan for f in ['current_kalshi_yes_bid','current_kalshi_yes_ask','current_kalshi_mid_price','current_kalshi_spread_abs','current_kalshi_spread_pct']})
        final_feature_row = {}
        for f_name in model_feature_names: final_feature_row[f_name] = features.get(f_name, np.nan)
        final_feature_row['market_ticker'] = features['market_ticker']; final_feature_row['decision_timestamp_s'] = features['decision_timestamp_s']; final_feature_row['resolution_time_ts'] = features['resolution_time_ts']; final_feature_row['target'] = features['target']
        market_decision_features.append(final_feature_row)
        current_decision_ts_utc += 60 
    return pd.DataFrame(market_decision_features)