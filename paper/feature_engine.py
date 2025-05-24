# /paper/feature_engine.py

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Deque

import paper.config as cfg
from paper.data_models import BinanceKline, KalshiMarketState, KalshiMarketInfo

logger = logging.getLogger("paper_feature_engine")

def get_latest_kalshi_state_for_features(
    kalshi_market_state: Optional[KalshiMarketState],
    decision_ts_utc: datetime
) -> Optional[Dict[str, any]]:
    """
    Extracts relevant Kalshi features from the in-memory KalshiMarketState.
    Ensures data is not from the future relative to decision_ts_utc.
    The KalshiMarketState itself should be the latest snapshot/delta update *before or at* decision_ts_utc.
    """
    if kalshi_market_state is None:
        return None

    # The KalshiMarketState's timestamp_utc should be <= decision_ts_utc due to how it's updated
    # We assume kalshi_market_state.update_ui_bid_ask() has been called after its last update.
    
    # Convert prices from cents (as stored in KalshiMarketState) to 0-1 scale for model
    # if the model was trained on 0-1 scale. Your historical feature engineering
    # divided by 100.0.
    # Your live_backtest_feature_eng.py for Kalshi:
    # features['current_kalshi_yes_bid'] = yes_bid / 100.0 if pd.notna(yes_bid) else np.nan
    
    kalshi_features = {}
    bid_cents = kalshi_market_state.ui_yes_bid_cents
    ask_cents = kalshi_market_state.ui_yes_ask_cents

    kalshi_features['current_kalshi_yes_bid'] = bid_cents / 100.0 if bid_cents is not None else np.nan
    kalshi_features['current_kalshi_yes_ask'] = ask_cents / 100.0 if ask_cents is not None else np.nan
    
    # Volume and OI are not used by 'no_vol_oi' model, but we can include them if present
    # kalshi_features['current_kalshi_volume'] = kalshi_market_state.some_volume_field # If available
    # kalshi_features['current_kalshi_oi'] = kalshi_market_state.some_oi_field # If available

    if pd.notna(kalshi_features['current_kalshi_yes_bid']) and pd.notna(kalshi_features['current_kalshi_yes_ask']):
        mid_price = (kalshi_features['current_kalshi_yes_bid'] + kalshi_features['current_kalshi_yes_ask']) / 2.0
        spread_abs = kalshi_features['current_kalshi_yes_ask'] - kalshi_features['current_kalshi_yes_bid']
        kalshi_features['current_kalshi_mid_price'] = mid_price
        kalshi_features['current_kalshi_spread_abs'] = spread_abs
        kalshi_features['current_kalshi_spread_pct'] = (spread_abs / mid_price) if mid_price != 0 else np.nan
    else:
        kalshi_features.update({
            'current_kalshi_mid_price': np.nan,
            'current_kalshi_spread_abs': np.nan,
            'current_kalshi_spread_pct': np.nan
        })
    return kalshi_features


def get_latest_btc_kline_and_history_for_features(
    binance_kline_history: Deque[BinanceKline], # A deque of recent BinanceKline objects, newest at right
    decision_ts_utc: datetime
) -> Optional[Dict[str, any]]:
    """
    Extracts BTC price, calculates lags and rolling volatility from in-memory kline history.
    The `decision_ts_utc` is the "current moment" for the decision.
    We need the kline that *closed just before* this moment.
    """
    if not binance_kline_history:
        return None

    # Find the latest kline whose kline_end_time < decision_ts_utc
    # Binance klines are per minute. If decision_ts_utc is e.g., 10:01:00,
    # we need the kline that covers 10:00:00 to 10:00:59 (ends at 10:00:59.999).
    # Its kline_start_time would be 10:00:00.
    
    target_kline_start_ts_seconds = int((decision_ts_utc - timedelta(minutes=1)).replace(second=0, microsecond=0).timestamp())

    current_btc_kline_for_decision: Optional[BinanceKline] = None
    for kline in reversed(binance_kline_history): # Search from newest
        if kline.timestamp_s_utc == target_kline_start_ts_seconds:
            current_btc_kline_for_decision = kline
            break
    
    if current_btc_kline_for_decision is None or not current_btc_kline_for_decision.is_closed:
        logger.warning(f"No suitable closed BTC kline found for decision_ts {decision_ts_utc.isoformat()} (target start: {target_kline_start_ts_seconds}). History length: {len(binance_kline_history)}")
        if binance_kline_history:
            latest_hist_kline = binance_kline_history[-1]
            logger.debug(f"Latest kline in history: start_ts={latest_hist_kline.timestamp_s_utc}, end_ts={latest_hist_kline.kline_end_time//1000}, closed={latest_hist_kline.is_closed}")
        return None

    btc_features = {}
    current_btc_price = current_btc_kline_for_decision.close
    btc_features['current_btc_price'] = current_btc_price

    # Prepare a series of close prices from the history for lags/rolling stats
    # The series should end with `current_btc_kline_for_decision`
    
    # Max lookback needed in minutes for any feature
    max_lag_rolling_min = max(cfg.LAG_WINDOWS_MINUTES + cfg.ROLLING_WINDOWS_MINUTES, default=0)
    
    # We need `max_lag_rolling_min` klines *before or including* current_btc_kline_for_decision
    # The timestamps in BinanceKline are kline_start_time.
    # current_btc_kline_for_decision.timestamp_s_utc is the start of the current minute's kline.
    
    historical_closes_for_stats = []
    timestamps_for_stats = []

    # Iterate through history to get closes up to and including the current one
    # Ensure they are sorted by time if deque doesn't guarantee it (it should if appended chronologically)
    for kline_hist_item in binance_kline_history:
        if kline_hist_item.timestamp_s_utc <= current_btc_kline_for_decision.timestamp_s_utc:
            historical_closes_for_stats.append(kline_hist_item.close)
            timestamps_for_stats.append(pd.Timestamp(kline_hist_item.timestamp_s_utc, unit='s', tz='utc'))
    
    if not historical_closes_for_stats:
        logger.warning(f"No historical BTC closes found for stats up to {current_btc_kline_for_decision.timestamp_s_utc}")
        # Fill missing BTC features with NaN
        for lag in cfg.LAG_WINDOWS_MINUTES: btc_features[f'btc_price_change_pct_{lag}m'] = np.nan
        for window in cfg.ROLLING_WINDOWS_MINUTES: btc_features[f'btc_volatility_{window}m'] = np.nan
        return btc_features


    # Create a Pandas Series with DatetimeIndex for easy `asof` and `rolling`
    # Index should be kline_start_time_utc
    btc_price_series = pd.Series(historical_closes_for_stats, index=pd.Index(timestamps_for_stats, name="timestamp_s_utc"))
    btc_price_series = btc_price_series[~btc_price_series.index.duplicated(keep='last')].sort_index()


    # Calculate Lag Features
    # `current_btc_price` is from `current_btc_kline_for_decision`
    # The timestamp for `current_btc_price` is `current_btc_kline_for_decision.timestamp_s_utc`
    current_price_timestamp_pd = pd.Timestamp(current_btc_kline_for_decision.timestamp_s_utc, unit='s', tz='utc')

    for lag in cfg.LAG_WINDOWS_MINUTES:
        # Target timestamp for the past price (start of that minute's kline)
        target_lag_timestamp_pd = current_price_timestamp_pd - pd.Timedelta(minutes=lag)
        
        # Use .asof to get the price at or before the target_lag_timestamp_pd
        past_price_s = btc_price_series.asof(target_lag_timestamp_pd)
        
        if pd.notna(past_price_s) and past_price_s != 0:
            btc_features[f'btc_price_change_pct_{lag}m'] = (current_btc_price - past_price_s) / past_price_s
        else:
            btc_features[f'btc_price_change_pct_{lag}m'] = np.nan

    # Calculate Rolling Volatility Features
    # Volatility up to and including the `current_btc_kline_for_decision`
    for window in cfg.ROLLING_WINDOWS_MINUTES:
        # Ensure series is long enough for the window
        if len(btc_price_series) >= window:
            # Get the last `window` number of observations from the series
            # The series is already sorted and ends at current_price_timestamp_pd
            rolling_data = btc_price_series.loc[btc_price_series.index <= current_price_timestamp_pd].tail(window)
            if len(rolling_data) >= 2: # Min periods for std
                 btc_features[f'btc_volatility_{window}m'] = rolling_data.std()
            else:
                 btc_features[f'btc_volatility_{window}m'] = np.nan
        elif len(btc_price_series) >=2 : # Fallback if series shorter than window but >=2
            btc_features[f'btc_volatility_{window}m'] = btc_price_series.loc[btc_price_series.index <= current_price_timestamp_pd].std()
        else:
            btc_features[f'btc_volatility_{window}m'] = np.nan
            
    return btc_features


def generate_live_features(
    market_info: KalshiMarketInfo,
    decision_ts_utc: datetime,
    latest_kalshi_state: Optional[KalshiMarketState],
    binance_kline_history: Deque[BinanceKline], # Deque of BinanceKline, newest at right
    model_feature_names: List[str]
) -> Optional[Dict[str, any]]:
    """
    Generates all features for a given market at a specific decision timestamp.
    """
    features = {}
    
    # --- Basic Market/Time Features ---
    features['market_ticker'] = market_info.ticker # For logging/identification, not a model feature
    features['decision_timestamp_s'] = int(decision_ts_utc.timestamp()) # For logging
    features['resolution_time_ts'] = int(market_info.close_time_utc.timestamp()) # For logging
    
    features['strike_price'] = market_info.strike_price
    time_to_res_seconds = (market_info.close_time_utc - decision_ts_utc).total_seconds()
    features['time_to_resolution_minutes'] = round(time_to_res_seconds / 60.0, 2)

    if features['time_to_resolution_minutes'] < cfg.MIN_MINUTES_BEFORE_RESOLUTION_FOR_DECISION:
        # logger.debug(f"Too close to resolution for {market_info.ticker}. TTR: {features['time_to_resolution_minutes']}m")
        return None # Decision point is too close to market resolution

    # --- BTC Features ---
    btc_features = get_latest_btc_kline_and_history_for_features(binance_kline_history, decision_ts_utc)
    if btc_features:
        features.update(btc_features)
        if pd.notna(features.get('current_btc_price')) and pd.notna(features.get('strike_price')):
            current_btc_p = features['current_btc_price']
            strike_p = features['strike_price']
            features['current_dist_strike_abs'] = current_btc_p - strike_p
            features['current_dist_strike_pct'] = (current_btc_p - strike_p) / strike_p if strike_p != 0 else np.nan
        else:
            features['current_dist_strike_abs'] = np.nan
            features['current_dist_strike_pct'] = np.nan
    else: # Fill with NaNs if BTC data is missing
        logger.warning(f"BTC features could not be generated for {market_info.ticker} at {decision_ts_utc.isoformat()}")
        potential_btc_keys = ['current_btc_price', 'current_dist_strike_abs', 'current_dist_strike_pct'] + \
                             [f'btc_price_change_pct_{lag}m' for lag in cfg.LAG_WINDOWS_MINUTES] + \
                             [f'btc_volatility_{win}m' for win in cfg.ROLLING_WINDOWS_MINUTES]
        for key in potential_btc_keys: features[key] = np.nan


    # --- Kalshi Market Features ---
    kalshi_features = get_latest_kalshi_state_for_features(latest_kalshi_state, decision_ts_utc)
    if kalshi_features:
        features.update(kalshi_features)
    else: # Fill with NaNs if Kalshi data is missing
        logger.warning(f"Kalshi features could not be generated for {market_info.ticker} at {decision_ts_utc.isoformat()}")
        potential_kalshi_keys = ['current_kalshi_yes_bid', 'current_kalshi_yes_ask', 
                                 'current_kalshi_mid_price', 'current_kalshi_spread_abs', 
                                 'current_kalshi_spread_pct'] 
                                # Add 'current_kalshi_volume', 'current_kalshi_oi' if model uses them
        for key in potential_kalshi_keys: features[key] = np.nan

    # --- Select and Order Features for Model ---
    # And ensure all expected model features are present, filling with NaN if somehow missed
    final_model_input_features = {}
    any_critical_nan = False
    for f_name in model_feature_names:
        val = features.get(f_name, np.nan)
        final_model_input_features[f_name] = val
        if pd.isna(val):
            # Log only if it's a feature the model truly expects (already filtered by model_feature_names)
            logger.warning(f"NaN found for expected model feature '{f_name}' for {market_info.ticker} at {decision_ts_utc.isoformat()}")
            any_critical_nan = True # Decide later if this is fatal for this decision point

    if any_critical_nan:
        logger.error(f"Critical NaNs found in model input features for {market_info.ticker} at {decision_ts_utc.isoformat()}. Cannot make prediction.")
        # You might choose to return None or the dict with NaNs based on downstream handling
        # For now, let's return the dict and let the trading_logic handle NaN checks before prediction
        # return None
    
    # Add back identifying information for the output, not for the model
    final_model_input_features['market_ticker'] = market_info.ticker
    final_model_input_features['decision_timestamp_s'] = int(decision_ts_utc.timestamp())
    final_model_input_features['resolution_time_ts'] = int(market_info.close_time_utc.timestamp())
    # 'target' is not known at decision time for live trading

    return final_model_input_features