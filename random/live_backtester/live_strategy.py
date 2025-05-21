# live_backtester/live_strategy.py
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import logging
import datetime as dt
from datetime import timezone, timedelta

from live_backtester import live_utils # Adjusted import

logger = logging.getLogger(__name__)

MODEL_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 
MODEL_BASE_DIR = MODEL_PROJECT_ROOT / "notebooks" / "trained_models"

SCALER_PATH = MODEL_BASE_DIR / "feature_scaler_v1.joblib"
MODEL_PARAMS_PATH = MODEL_BASE_DIR / "lr_model_params_v1.json"
FEATURE_ORDER_PATH = MODEL_BASE_DIR / "feature_columns_v1.json"

_scaler = None
_model_intercept = None
_model_coefficients_dict = None
_feature_order = None

try:
    if SCALER_PATH.exists():
        _scaler = joblib.load(SCALER_PATH)
        logger.info(f"LiveStrategy: Loaded scaler from {SCALER_PATH}")
    else:
        logger.error(f"LiveStrategy: Scaler file not found at {SCALER_PATH}")

    if MODEL_PARAMS_PATH.exists():
        with open(MODEL_PARAMS_PATH, 'r') as f:
            params = json.load(f)
        _model_intercept = params.get('intercept')
        _model_coefficients_dict = params.get('coefficients')
        logger.info(f"LiveStrategy: Loaded model parameters (intercept, coefs) from {MODEL_PARAMS_PATH}")
    else:
        logger.error(f"LiveStrategy: Model params file not found at {MODEL_PARAMS_PATH}")
        
    if FEATURE_ORDER_PATH.exists():
        with open(FEATURE_ORDER_PATH, 'r') as f:
            _feature_order = json.load(f)
        logger.info(f"LiveStrategy: Loaded feature order ({len(_feature_order)} features) from {FEATURE_ORDER_PATH}")
    else:
        logger.error(f"LiveStrategy: Feature order JSON not found at {FEATURE_ORDER_PATH}. This is critical.")

except Exception as e:
    logger.critical(f"LiveStrategy: CRITICAL error loading model/scaler/features: {e}", exc_info=True)
    _scaler = _model_intercept = _model_coefficients_dict = _feature_order = None


PRED_THRESHOLD_BUY_YES = 750.0
PRED_THRESHOLD_BUY_NO = -750.0

def generate_features_from_live_data(
    df_closed_btc_klines_history: pd.DataFrame, 
    current_kalshi_snapshot: dict | None, 
    df_kalshi_mid_price_history: pd.DataFrame, 
    kalshi_strike_price: float,
    decision_dt_utc: dt.datetime, 
    kalshi_market_close_dt_utc: dt.datetime # Changed from market_resolution to market_close for clarity
) -> pd.Series | None:
    if _feature_order is None:
        logger.error("LiveStrategy: Feature order not loaded. Cannot generate features.")
        return None
    
    features = pd.Series(index=_feature_order, dtype=float)
    signal_btc_kline_start_ts_s = int((decision_dt_utc - timedelta(minutes=1)).replace(second=0, microsecond=0).timestamp())

    if df_closed_btc_klines_history.empty or signal_btc_kline_start_ts_s not in df_closed_btc_klines_history.index:
        logger.warning(f"LiveStrategy: BTC kline data insufficient or signal_ts {signal_btc_kline_start_ts_s} "
                       f"(for decision {decision_dt_utc.isoformat()}) not in df_closed_btc_klines_history. Index type: {df_closed_btc_klines_history.index.dtype}")
        return None
        
    btc_row_at_signal = df_closed_btc_klines_history.loc[signal_btc_kline_start_ts_s]
    
    features['btc_price_t_minus_1'] = btc_row_at_signal.get('close')
    if pd.isna(features['btc_price_t_minus_1']):
        logger.warning(f"LiveStrategy: btc_price_t_minus_1 is NaN for ts {signal_btc_kline_start_ts_s}.")
        return None

    for window in live_utils.BTC_MOMENTUM_WINDOWS: # Use live_utils for consistency
        col = f'btc_mom_{window}m'; features[col] = btc_row_at_signal.get(col, 0.0)
    if live_utils.BTC_VOLATILITY_WINDOW > 0:
        col = f'btc_vol_{live_utils.BTC_VOLATILITY_WINDOW}m'; features[col] = btc_row_at_signal.get(col, 0.0)
    for window in live_utils.BTC_SMA_WINDOWS:
        col = f'btc_sma_{window}m'; features[col] = btc_row_at_signal.get(col, features['btc_price_t_minus_1'])
    for window in live_utils.BTC_EMA_WINDOWS:
        col = f'btc_ema_{window}m'; features[col] = btc_row_at_signal.get(col, features['btc_price_t_minus_1'])
    if live_utils.BTC_RSI_WINDOW > 0:
        col = 'btc_rsi'; features[col] = btc_row_at_signal.get(col, 50.0)
    elif 'btc_rsi' in _feature_order: # Add placeholder if model expects it
        features['btc_rsi'] = 50.0


    features['distance_to_strike'] = features['btc_price_t_minus_1'] - kalshi_strike_price

    if current_kalshi_snapshot and \
       pd.notna(current_kalshi_snapshot.get('yes_bid')) and \
       pd.notna(current_kalshi_snapshot.get('yes_ask')):
        features['kalshi_yes_bid'] = current_kalshi_snapshot['yes_bid']
        features['kalshi_yes_ask'] = current_kalshi_snapshot['yes_ask']
        features['kalshi_spread'] = features['kalshi_yes_ask'] - features['kalshi_yes_bid']
        features['kalshi_mid_price'] = (features['kalshi_yes_bid'] + features['kalshi_yes_ask']) / 2.0
        features['kalshi_volume_t_minus_1'] = 0 
        features['kalshi_open_interest_t_minus_1'] = 0
    else:
        features['kalshi_yes_bid'] = 0; features['kalshi_yes_ask'] = 100
        features['kalshi_spread'] = 100; features['kalshi_mid_price'] = 50
        features['kalshi_volume_t_minus_1'] = 0; features['kalshi_open_interest_t_minus_1'] = 0

    current_mid_price = features.get('kalshi_mid_price')
    if pd.notna(current_mid_price) and not df_kalshi_mid_price_history.empty:
        for window_min in live_utils.KALSHI_PRICE_CHANGE_WINDOWS:
            col_name = f'kalshi_mid_chg_{window_min}m'
            if col_name in _feature_order:
                past_dt_utc = decision_dt_utc - timedelta(minutes=window_min)
                past_mid_rows = df_kalshi_mid_price_history[df_kalshi_mid_price_history.index <= past_dt_utc]
                if not past_mid_rows.empty:
                    past_mid_val = past_mid_rows.iloc[-1]['kalshi_mid_price']
                    if pd.notna(past_mid_val):
                        features[col_name] = current_mid_price - past_mid_val
                    else: features[col_name] = 0.0
                else: features[col_name] = 0.0
    else: 
        for window_min in live_utils.KALSHI_PRICE_CHANGE_WINDOWS:
            col_name = f'kalshi_mid_chg_{window_min}m'
            if col_name in _feature_order: features[col_name] = 0.0

    features['time_until_market_close_min'] = (kalshi_market_close_dt_utc - decision_dt_utc).total_seconds() / 60
    features['hour_of_day_utc'] = decision_dt_utc.hour
    features['day_of_week_utc'] = decision_dt_utc.weekday()
    decision_dt_edt = decision_dt_utc.astimezone(timezone(timedelta(hours=-4))) 
    features['hour_of_day_edt'] = decision_dt_edt.hour
    
    for f_name in _feature_order:
        if pd.isna(features.get(f_name)):
            if 'rsi' in f_name: features[f_name] = 50.0
            elif 'kalshi_yes_ask' == f_name : features[f_name] = 100.0
            elif 'kalshi_spread' == f_name : features[f_name] = 100.0
            elif 'kalshi_mid_price' == f_name : features[f_name] = 50.0
            else: features[f_name] = 0.0
            
    return features.reindex(_feature_order)

def calculate_model_prediction(feature_vector: pd.Series) -> float | None:
    if not all([_scaler, _model_intercept is not None, _model_coefficients_dict, _feature_order]):
        logger.error("LiveStrategy: Model/scaler/params not fully loaded. Cannot predict.")
        return None
    if feature_vector is None or feature_vector.empty:
        logger.warning("LiveStrategy: Empty feature vector for prediction.")
        return None
    try:
        feature_df_ordered = feature_vector.to_frame().T 
        if feature_df_ordered.isnull().values.any():
            nan_cols = feature_df_ordered.columns[feature_df_ordered.isnull().any()].tolist()
            logger.warning(f"LiveStrategy: NaNs in feature vector before scaling: {nan_cols}. Filling with 0.")
            feature_df_ordered.fillna(0, inplace=True) 

        scaled_features_array = _scaler.transform(feature_df_ordered)
        coefficient_array = np.array([_model_coefficients_dict.get(f, 0) for f in _feature_order])
        
        if len(scaled_features_array[0]) != len(coefficient_array):
            logger.error(f"LiveStrategy: Mismatch in scaled features ({len(scaled_features_array[0])}) and coefficients ({len(coefficient_array)}) count.")
            return None
            
        prediction = _model_intercept + np.dot(scaled_features_array[0], coefficient_array)
        return float(prediction)
    except Exception as e:
        logger.error(f"LiveStrategy: Error during prediction: {e}", exc_info=True)
        logger.error(f"Feature vector that caused error: {feature_vector.to_dict()}")
        return None

def get_trade_decision(model_prediction_score: float | None):
    if model_prediction_score is None: return None, None
    trade_action = None 
    if model_prediction_score > PRED_THRESHOLD_BUY_YES:
        trade_action = "BUY_YES"
    elif model_prediction_score < PRED_THRESHOLD_BUY_NO:
        trade_action = "BUY_NO"
    return trade_action, model_prediction_score