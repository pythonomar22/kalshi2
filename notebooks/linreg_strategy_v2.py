# linreg_strategy_v2.py
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import logging
import datetime as dt
from datetime import timezone, timedelta

logger = logging.getLogger(__name__)

MODEL_DIR = Path("./trained_models")
SCALER_PATH = MODEL_DIR / "feature_scaler_v1.joblib"
MODEL_PARAMS_PATH = MODEL_DIR / "lr_model_params_v1.json"

_scaler = None
_model_intercept = None
_model_coefficients_dict = None
_feature_order = None

try:
    if SCALER_PATH.exists():
        _scaler = joblib.load(SCALER_PATH)
        logger.info(f"Strategy: Loaded scaler from {SCALER_PATH}")
    else:
        logger.error(f"Strategy: Scaler file not found at {SCALER_PATH}")

    if MODEL_PARAMS_PATH.exists():
        with open(MODEL_PARAMS_PATH, 'r') as f:
            params = json.load(f)
        _model_intercept = params.get('intercept')
        _model_coefficients_dict = params.get('coefficients')
        _feature_order = params.get('feature_order')
        logger.info(f"Strategy: Loaded model parameters from {MODEL_PARAMS_PATH}")
        if _feature_order:
            logger.info(f"Strategy: Expected feature order ({len(_feature_order)} features): {_feature_order[:5]}...")
    else:
        logger.error(f"Strategy: Model params file not found at {MODEL_PARAMS_PATH}")
except Exception as e:
    logger.critical(f"Strategy: CRITICAL error loading model/scaler: {e}", exc_info=True)
    _scaler = _model_intercept = _model_coefficients_dict = _feature_order = None

BTC_MOMENTUM_WINDOWS = [5, 10, 15, 30]
BTC_VOLATILITY_WINDOW = 15
BTC_SMA_WINDOWS = [10, 30]
BTC_EMA_WINDOWS = [12, 26]
BTC_RSI_WINDOW = 14
KALSHI_PRICE_CHANGE_WINDOWS = [1, 3, 5]

PRED_THRESHOLD_BUY_YES = 50.0
PRED_THRESHOLD_BUY_NO = -50.0

def generate_live_features(
    btc_price_history_df: pd.DataFrame,
    current_kalshi_bid: float | None,
    current_kalshi_ask: float | None,
    kalshi_market_history_df: pd.DataFrame | None,
    kalshi_strike_price: float,
    decision_point_dt_utc: dt.datetime,
    kalshi_market_close_dt_utc: dt.datetime
) -> pd.Series | None:
    if _feature_order is None:
        logger.error("Strategy: Feature order not loaded. Cannot generate features.")
        return None
    features = pd.Series(index=_feature_order, dtype=float)
    signal_ts_utc = int(decision_point_dt_utc.timestamp()) - 60

    if btc_price_history_df.empty or signal_ts_utc not in btc_price_history_df.index:
        logger.warning(f"Strategy: BTC data insufficient or signal_ts {signal_ts_utc} not in history for {decision_point_dt_utc.isoformat()}")
        return None
    if 'close' not in btc_price_history_df.columns:
        logger.error("Strategy: 'close' column missing in btc_price_history_df.")
        return None
        
    features['btc_price_t_minus_1'] = btc_price_history_df.loc[signal_ts_utc, 'close']
    if pd.isna(features['btc_price_t_minus_1']): return None

    for window in BTC_MOMENTUM_WINDOWS:
        col_name = f'btc_mom_{window}m'
        if col_name in _feature_order:
            features[col_name] = btc_price_history_df.loc[signal_ts_utc, col_name] if col_name in btc_price_history_df.columns else 0
    
    row_at_signal_ts = btc_price_history_df.loc[signal_ts_utc]
    for f_name in [f'btc_vol_{BTC_VOLATILITY_WINDOW}m'] + \
                  [f'btc_sma_{w}m' for w in BTC_SMA_WINDOWS] + \
                  [f'btc_ema_{w}m' for w in BTC_EMA_WINDOWS] + \
                  (['btc_rsi'] if BTC_RSI_WINDOW > 0 else []):
        if f_name in _feature_order:
            features[f_name] = row_at_signal_ts.get(f_name, 50 if 'rsi' in f_name else 0)

    features['distance_to_strike'] = features['btc_price_t_minus_1'] - kalshi_strike_price

    if current_kalshi_bid is not None and current_kalshi_ask is not None:
        features['kalshi_yes_bid'] = current_kalshi_bid
        features['kalshi_yes_ask'] = current_kalshi_ask
        features['kalshi_spread'] = current_kalshi_ask - current_kalshi_bid
        features['kalshi_mid_price'] = (current_kalshi_bid + current_kalshi_ask) / 2.0
    else:
        features['kalshi_yes_bid'] = 0; features['kalshi_yes_ask'] = 100
        features['kalshi_spread'] = 100; features['kalshi_mid_price'] = 50

    if kalshi_market_history_df is not None and not kalshi_market_history_df.empty:
        relevant_kalshi_rows = kalshi_market_history_df[kalshi_market_history_df.index <= signal_ts_utc]
        if not relevant_kalshi_rows.empty:
            latest_kalshi_candle = relevant_kalshi_rows.iloc[-1]
            features['kalshi_volume_t_minus_1'] = latest_kalshi_candle.get('volume', 0)
            features['kalshi_open_interest_t_minus_1'] = latest_kalshi_candle.get('open_interest', 0)
            current_mid = features.get('kalshi_mid_price')
            if pd.notna(current_mid):
                for window in KALSHI_PRICE_CHANGE_WINDOWS:
                    col_name = f'kalshi_mid_chg_{window}m'
                    if col_name in _feature_order:
                        past_ts = signal_ts_utc - (window * 60)
                        past_kalshi_rows = kalshi_market_history_df[kalshi_market_history_df.index <= past_ts]
                        if not past_kalshi_rows.empty:
                            past_k_candle = past_kalshi_rows.iloc[-1]
                            past_bid = past_k_candle.get('yes_bid_close_cents'); past_ask = past_k_candle.get('yes_ask_close_cents')
                            if pd.notna(past_bid) and pd.notna(past_ask):
                                features[col_name] = current_mid - ((past_bid + past_ask) / 2.0)
                            else: features[col_name] = 0
                        else: features[col_name] = 0
    # Fallback fills for any features not explicitly calculated or if source data missing
    for f_name in _feature_order:
        if pd.isna(features.get(f_name)):
            # logger.warning(f"Strategy: Feature '{f_name}' was NaN post-gen. Filling default.")
            if 'rsi' in f_name: features[f_name] = 50
            elif 'kalshi_yes_ask' == f_name : features[f_name] = 100
            elif 'kalshi_spread' == f_name : features[f_name] = 100
            elif 'kalshi_mid_price' == f_name : features[f_name] = 50
            else: features[f_name] = 0
    return features

def calculate_model_prediction(feature_vector: pd.Series) -> float | None:
    if not all([_scaler, _model_intercept is not None, _model_coefficients_dict, _feature_order]):
        logger.error("Strategy: Model/scaler/params not fully loaded. Cannot predict.")
        return None
    if feature_vector is None or feature_vector.empty:
        logger.warning("Strategy: Empty feature vector for prediction.")
        return None
    try:
        # Ensure feature_vector is a DataFrame with one row and correct column order for scaler
        feature_df_ordered = feature_vector.reindex(_feature_order).to_frame().T
        
        if feature_df_ordered.isnull().any().any(): # Check any NaNs in the DataFrame
            logger.warning(f"Strategy: NaNs in feature vector before scaling: {feature_df_ordered.columns[feature_df_ordered.isnull().any()].tolist()}. Filling with 0 for scaling.")
            feature_df_ordered.fillna(0, inplace=True)

        scaled_features_array = _scaler.transform(feature_df_ordered) # Pass DataFrame
        
        coefficient_array = np.array([_model_coefficients_dict.get(f, 0) for f in _feature_order])
        prediction = _model_intercept + np.dot(scaled_features_array[0], coefficient_array)
        return float(prediction)
    except Exception as e:
        logger.error(f"Strategy: Error during prediction: {e}", exc_info=True)
        return None

def get_trade_decision(model_prediction_score: float | None):
    if model_prediction_score is None: return None, None
    trade_action = None
    if model_prediction_score > PRED_THRESHOLD_BUY_YES: trade_action = "BUY_YES"
    elif model_prediction_score < PRED_THRESHOLD_BUY_NO: trade_action = "BUY_NO"
    return trade_action, model_prediction_score