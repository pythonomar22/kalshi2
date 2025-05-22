# live_backtester/live_strategy.py
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import logging
import datetime as dt
from datetime import timezone, timedelta

from live_backtester.live_utils import (
    BTC_MOMENTUM_WINDOWS, BTC_VOLATILITY_WINDOW, 
    BTC_SMA_WINDOWS, BTC_EMA_WINDOWS, BTC_RSI_WINDOW,
    KALSHI_PRICE_CHANGE_WINDOWS 
)
# This should match the value used in feature_engineering.ipynb Cell 1 for consistency
KALSHI_MAX_STALENESS_SECONDS_FOR_FEATURES = 120 

logger = logging.getLogger(__name__)

MODEL_BASE_DIR = Path(".") 
# These will be set by live_backtest_main.py based on MODEL_TYPE_TO_RUN
MIN_MODEL_PROB_FOR_CONSIDERATION = 0.5 # Default, will be overridden
EDGE_THRESHOLD_FOR_TRADE = 0.0     # Default, will be overridden

_scaler = None
_model = None 
_feature_order = None
_model_classes = None
_model_type_loaded = "None" 

def load_classifier_artifacts(model_artifacts_directory: Path):
    global _scaler, _model, _feature_order, _model_classes, MODEL_BASE_DIR, _model_type_loaded
    MODEL_BASE_DIR = model_artifacts_directory
    logger.info(f"LiveStrategy: Attempting to load CLASSIFIER artifacts from {MODEL_BASE_DIR.resolve()}")

    scaler_path = MODEL_BASE_DIR / "feature_scaler_classifier_v1.joblib"
    feature_order_path = MODEL_BASE_DIR / "feature_columns_classifier_v1.json"
    
    model_type_indicator = MODEL_BASE_DIR.name.lower()
    if "logreg" in model_type_indicator:
        model_filename = "logistic_regression_btc_classifier_v1.joblib"
        _model_type_loaded = "logistic_regression"
    elif "rf" in model_type_indicator or "random_forest" in model_type_indicator:
        model_filename = "RandomForest_classifier_v1.joblib"
        _model_type_loaded = "random_forest"
    else:
        logger.error(f"LiveStrategy: Could not determine model type from directory: {MODEL_BASE_DIR}. Defaulting to LogReg.")
        model_filename = "logistic_regression_btc_classifier_v1.joblib"
        _model_type_loaded = "logistic_regression" # Fallback

    model_path = MODEL_BASE_DIR / model_filename
    logger.info(f"LiveStrategy: Determined model type '{_model_type_loaded}', attempting to load model file '{model_filename}'.")

    artifacts_loaded = True
    try:
        if scaler_path.exists(): _scaler = joblib.load(scaler_path); logger.debug("Scaler loaded.")
        else: logger.error(f"LiveStrategy: Scaler not found: {scaler_path}"); artifacts_loaded = False
        
        if model_path.exists(): 
            _model = joblib.load(model_path); logger.debug(f"Model '{model_filename}' loaded.")
            if hasattr(_model, 'classes_'): _model_classes = _model.classes_
            else: logger.warning(f"Model type {_model_type_loaded} lacks 'classes_'. Assuming [0, 1]."); _model_classes = np.array([0,1])
        else: logger.error(f"LiveStrategy: Model '{model_filename}' not found: {model_path}"); artifacts_loaded = False
        
        if feature_order_path.exists():
            with open(feature_order_path, 'r') as f: _feature_order = json.load(f); logger.debug("Feature order loaded.")
        else: logger.error(f"LiveStrategy: Feature order not found: {feature_order_path}"); artifacts_loaded = False
            
        if artifacts_loaded: logger.info(f"LiveStrategy: All '{_model_type_loaded}' CLASSIFIER artifacts loaded successfully.")
        else: logger.error(f"LiveStrategy: One or more '{_model_type_loaded}' CLASSIFIER artifacts failed to load.")

    except Exception as e:
        logger.critical(f"LiveStrategy: CRITICAL error loading '{_model_type_loaded}' CLASSIFIER artifacts: {e}", exc_info=True)
        return False
    return artifacts_loaded

def generate_features_from_live_data(
    df_closed_btc_klines_history: pd.DataFrame, 
    df_kalshi_live_market_history: pd.DataFrame, 
    kalshi_strike_price: float,
    decision_dt_utc: dt.datetime, 
    kalshi_market_close_dt_utc: dt.datetime
) -> pd.Series | None:
    if _feature_order is None:
        logger.error(f"LiveStrategy ({_model_type_loaded}): Feature order not loaded.")
        return None
    
    features = pd.Series(index=_feature_order, dtype=float)
    signal_features_dt_utc = (decision_dt_utc - timedelta(minutes=1)).replace(second=0, microsecond=0)
    signal_features_ts_s = int(signal_features_dt_utc.timestamp())

    if df_closed_btc_klines_history.empty or signal_features_ts_s not in df_closed_btc_klines_history.index:
        logger.debug(f"LiveStrategy ({_model_type_loaded}): BTC kline data for features_ts {signal_features_ts_s} (decision: {decision_dt_utc.isoformat()}) not found.")
        return None
        
    btc_row_at_signal = df_closed_btc_klines_history.loc[signal_features_ts_s]
    btc_price_t_minus_1 = btc_row_at_signal.get('close')
    if pd.isna(btc_price_t_minus_1):
        logger.debug(f"LiveStrategy ({_model_type_loaded}): btc_price_t_minus_1 is NaN for ts {signal_features_ts_s}.")
        return None
    features['btc_price_t_minus_1'] = btc_price_t_minus_1

    for window in BTC_MOMENTUM_WINDOWS:
        col = f'btc_mom_{window}m'; features[col] = btc_row_at_signal.get(col, 0.0)
    if BTC_VOLATILITY_WINDOW > 0:
        col = f'btc_vol_{BTC_VOLATILITY_WINDOW}m'; features[col] = btc_row_at_signal.get(col, 0.0)
    for window in BTC_SMA_WINDOWS:
        col = f'btc_sma_{window}m'; features[col] = btc_row_at_signal.get(col, btc_price_t_minus_1)
    for window in BTC_EMA_WINDOWS:
        col = f'btc_ema_{window}m'; features[col] = btc_row_at_signal.get(col, btc_price_t_minus_1)
    if BTC_RSI_WINDOW > 0:
        col = 'btc_rsi'; features[col] = btc_row_at_signal.get(col, 50.0)
    elif 'btc_rsi' in _feature_order: features['btc_rsi'] = 50.0

    features['distance_to_strike'] = btc_price_t_minus_1 - kalshi_strike_price

    _yes_bid_t_minus_1, _yes_ask_t_minus_1, _spread_t_minus_1, _mid_t_minus_1 = 0.0, 100.0, 100.0, 50.0
    # For live data, 'volume' and 'open_interest' from streamer are per-tick, not per-minute aggregates.
    # Training used per-minute aggregates. To maintain consistency, we should impute them as 0 here
    # unless the live streamer data is processed to provide minute aggregates.
    _vol_t_minus_1, _oi_t_minus_1 = 0.0, 0.0 
    
    kalshi_snapshot_t_minus_1 = None
    if df_kalshi_live_market_history is not None and not df_kalshi_live_market_history.empty:
        relevant_kalshi_rows_t_minus_1 = df_kalshi_live_market_history[df_kalshi_live_market_history.index <= signal_features_dt_utc]
        if not relevant_kalshi_rows_t_minus_1.empty:
            latest_kalshi_row_t_minus_1 = relevant_kalshi_rows_t_minus_1.iloc[-1]
            latest_kalshi_dt_t_minus_1 = latest_kalshi_row_t_minus_1.name 
            
            if (signal_features_dt_utc - latest_kalshi_dt_t_minus_1).total_seconds() <= KALSHI_MAX_STALENESS_SECONDS_FOR_FEATURES:
                kalshi_snapshot_t_minus_1 = latest_kalshi_row_t_minus_1

    if kalshi_snapshot_t_minus_1 is not None:
        _yes_bid_val = kalshi_snapshot_t_minus_1.get('yes_bid_price_cents')
        _yes_ask_val = kalshi_snapshot_t_minus_1.get('yes_ask_price_cents')

        if pd.notna(_yes_bid_val) and pd.notna(_yes_ask_val) and \
           0 <= _yes_bid_val <= 100 and 0 <= _yes_ask_val <= 100 and _yes_bid_val <= _yes_ask_val:
            _yes_bid_t_minus_1 = _yes_bid_val
            _yes_ask_t_minus_1 = _yes_ask_val
            _spread_t_minus_1 = _yes_ask_t_minus_1 - _yes_bid_t_minus_1
            _mid_t_minus_1 = (_yes_bid_t_minus_1 + _yes_ask_t_minus_1) / 2.0
        # If your live Kalshi CSV had 'volume'/'open_interest' fields that are comparable to training, map them:
        # _vol_t_minus_1 = kalshi_snapshot_t_minus_1.get('volume_comparable_to_training', 0.0)
        # _oi_t_minus_1 = kalshi_snapshot_t_minus_1.get('oi_comparable_to_training', 0.0)

    if 'kalshi_yes_bid' in _feature_order: features['kalshi_yes_bid'] = _yes_bid_t_minus_1
    if 'kalshi_yes_ask' in _feature_order: features['kalshi_yes_ask'] = _yes_ask_t_minus_1
    if 'kalshi_spread' in _feature_order: features['kalshi_spread'] = _spread_t_minus_1
    if 'kalshi_mid_price' in _feature_order: features['kalshi_mid_price'] = _mid_t_minus_1
    if 'kalshi_volume_t_minus_1' in _feature_order: features['kalshi_volume_t_minus_1'] = _vol_t_minus_1 
    if 'kalshi_open_interest_t_minus_1' in _feature_order: features['kalshi_open_interest_t_minus_1'] = _oi_t_minus_1

    for window_min in KALSHI_PRICE_CHANGE_WINDOWS:
        col_name = f'kalshi_mid_chg_{window_min}m'
        features[col_name] = 0.0 
        if col_name in _feature_order and pd.notna(_mid_t_minus_1) and \
           df_kalshi_live_market_history is not None and not df_kalshi_live_market_history.empty:
            past_dt_for_chg_calc = signal_features_dt_utc - timedelta(minutes=window_min)
            past_kalshi_rows_for_chg = df_kalshi_live_market_history[df_kalshi_live_market_history.index <= past_dt_for_chg_calc]
            if not past_kalshi_rows_for_chg.empty:
                past_kalshi_candle_for_chg = past_kalshi_rows_for_chg.iloc[-1]
                past_bid_for_chg = past_kalshi_candle_for_chg.get('yes_bid_price_cents')
                past_ask_for_chg = past_kalshi_candle_for_chg.get('yes_ask_price_cents')
                if pd.notna(past_bid_for_chg) and pd.notna(past_ask_for_chg):
                    past_mid_for_chg = (past_bid_for_chg + past_ask_for_chg) / 2.0
                    features[col_name] = _mid_t_minus_1 - past_mid_for_chg
    
    features['time_until_market_close_min'] = (kalshi_market_close_dt_utc - decision_dt_utc).total_seconds() / 60
    features['hour_of_day_utc'] = decision_dt_utc.hour
    features['day_of_week_utc'] = decision_dt_utc.weekday()
    try: 
        edt_tz = timezone(timedelta(hours=-4)) 
        features['hour_of_day_edt'] = decision_dt_utc.astimezone(edt_tz).hour
    except Exception: features['hour_of_day_edt'] = (decision_dt_utc.hour - 4 + 24) % 24
    
    for f_name in _feature_order:
        if pd.isna(features.get(f_name)):
            # Heuristic imputation based on common defaults for these features
            if 'rsi' in f_name: features[f_name] = 50.0
            elif f_name == 'kalshi_yes_ask': features[f_name] = 100.0
            elif f_name == 'kalshi_spread': features[f_name] = 100.0 # Max spread if unknown
            elif f_name == 'kalshi_mid_price': features[f_name] = 50.0 # Neutral mid if unknown
            elif f_name == 'kalshi_yes_bid': features[f_name] = 0.0
            elif 'kalshi_volume' in f_name or 'kalshi_open_interest' in f_name: features[f_name] = 0.0
            elif 'btc_mom' in f_name or 'kalshi_mid_chg' in f_name: features[f_name] = 0.0 # No change if unknown
            else: features[f_name] = 0.0 # General fallback for other numeric features
            logger.debug(f"LiveStrategy ({_model_type_loaded}): Filled NaN for feature '{f_name}' with {features[f_name]}.")
            
    return features.reindex(_feature_order)

def calculate_model_prediction_proba(feature_vector: pd.Series) -> float | None:
    if not all([_scaler, _model, _feature_order, _model_classes is not None]):
        logger.error(f"LiveStrategy ({_model_type_loaded}): Model/scaler/features not fully loaded.")
        return None
    if feature_vector is None or feature_vector.empty:
        logger.warning(f"LiveStrategy ({_model_type_loaded}): Empty feature vector for prediction.")
        return None
    try:
        feature_df_ordered = feature_vector.to_frame().T[_feature_order] 
        if feature_df_ordered.isnull().values.any():
            nan_cols = feature_df_ordered.columns[feature_df_ordered.isnull().any()].tolist()
            logger.warning(f"LiveStrategy ({_model_type_loaded}): NaNs in feature vector before scaling: {nan_cols}. Imputing.")
            for col in nan_cols: 
                if 'rsi' in col: feature_df_ordered[col].fillna(50.0, inplace=True)
                elif col == 'kalshi_yes_ask': feature_df_ordered[col].fillna(100.0, inplace=True)
                elif col == 'kalshi_spread': feature_df_ordered[col].fillna(100.0, inplace=True)
                elif col == 'kalshi_mid_price': feature_df_ordered[col].fillna(50.0, inplace=True)
                else: feature_df_ordered[col].fillna(0, inplace=True)

        scaled_features_array = _scaler.transform(feature_df_ordered)
        
        positive_class_idx = np.where(_model_classes == 1)[0]
        if not positive_class_idx.size:
            logger.error(f"LiveStrategy ({_model_type_loaded}): Positive class (1) not found in model.classes_ ({_model_classes}).")
            return None
            
        proba_yes = _model.predict_proba(scaled_features_array)[0, positive_class_idx[0]]
        return float(proba_yes)
        
    except Exception as e:
        logger.error(f"LiveStrategy ({_model_type_loaded}): Error during probability prediction: {e}", exc_info=True)
        return None

def get_trade_decision(
    predicted_proba_yes: float | None,
    current_kalshi_bid_at_decision_t: float | None, 
    current_kalshi_ask_at_decision_t: float | None  
):
    if predicted_proba_yes is None: return None, None, None
    trade_action, model_prob_for_chosen_side, entry_price_for_chosen_side = None, None, None

    if pd.notna(current_kalshi_ask_at_decision_t) and 0 < current_kalshi_ask_at_decision_t < 100:
        implied_proba_yes_at_ask = current_kalshi_ask_at_decision_t / 100.0
        if predicted_proba_yes > MIN_MODEL_PROB_FOR_CONSIDERATION and \
           predicted_proba_yes > (implied_proba_yes_at_ask + EDGE_THRESHOLD_FOR_TRADE):
            trade_action = "BUY_YES"
            model_prob_for_chosen_side = predicted_proba_yes
            entry_price_for_chosen_side = current_kalshi_ask_at_decision_t

    if trade_action is None and pd.notna(current_kalshi_bid_at_decision_t) and 0 < current_kalshi_bid_at_decision_t < 100:
        predicted_proba_no = 1.0 - predicted_proba_yes
        cost_of_no_contract_cents = 100.0 - current_kalshi_bid_at_decision_t
        if 0 < cost_of_no_contract_cents < 100:
            implied_proba_no_at_bid = cost_of_no_contract_cents / 100.0
            if predicted_proba_no > MIN_MODEL_PROB_FOR_CONSIDERATION and \
               predicted_proba_no > (implied_proba_no_at_bid + EDGE_THRESHOLD_FOR_TRADE):
                trade_action = "BUY_NO"
                model_prob_for_chosen_side = predicted_proba_no
                entry_price_for_chosen_side = cost_of_no_contract_cents
    return trade_action, model_prob_for_chosen_side, entry_price_for_chosen_side