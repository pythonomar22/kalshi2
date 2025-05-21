# live_backtester/live_strategy.py
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import logging
import datetime as dt
from datetime import timezone, timedelta

# Import TA parameters from live_utils to ensure consistency for feature generation
from live_backtester.live_utils import (
    BTC_MOMENTUM_WINDOWS, BTC_VOLATILITY_WINDOW, 
    BTC_SMA_WINDOWS, BTC_EMA_WINDOWS, BTC_RSI_WINDOW,
    KALSHI_PRICE_CHANGE_WINDOWS # Strategy needs this for its Kalshi features
)

logger = logging.getLogger(__name__)

# --- Configuration for Model Artifacts ---
# Path to the directory containing the *classifier* model, scaler, and feature order
# This will be set by live_backtest_main.py by pointing to the 'logreg' subdirectory
MODEL_BASE_DIR = Path(".") # Placeholder, will be updated by live_backtest_main.py

# --- Trade Decision Parameters ---
MIN_MODEL_PROB_FOR_CONSIDERATION = 0.90 # Model must be at least this confident
EDGE_THRESHOLD_FOR_TRADE = 0.40     # Model's P(event) must be this much > Kalshi's Implied P(event)

# --- Global variables for loaded model artifacts ---
_scaler = None
_model = None # Will hold the LogisticRegression model
_feature_order = None
_model_classes = None

def load_classifier_artifacts(model_artifacts_directory: Path):
    """Loads the scaler, logistic regression model, and feature order for the live strategy."""
    global _scaler, _model, _feature_order, _model_classes, MODEL_BASE_DIR
    MODEL_BASE_DIR = model_artifacts_directory # Update global if needed, or just use the param

    logger.info(f"LiveStrategy: Attempting to load CLASSIFIER artifacts from {MODEL_BASE_DIR.resolve()}")

    scaler_path = MODEL_BASE_DIR / "feature_scaler_classifier_v1.joblib"
    model_path = MODEL_BASE_DIR / "logistic_regression_btc_classifier_v1.joblib"
    feature_order_path = MODEL_BASE_DIR / "feature_columns_classifier_v1.json"
    
    artifacts_loaded = True
    try:
        if scaler_path.exists(): _scaler = joblib.load(scaler_path)
        else: logger.error(f"LiveStrategy: Scaler not found: {scaler_path}"); artifacts_loaded = False
        
        if model_path.exists(): 
            _model = joblib.load(model_path)
            _model_classes = _model.classes_
        else: logger.error(f"LiveStrategy: Classifier Model not found: {model_path}"); artifacts_loaded = False
        
        if feature_order_path.exists():
            with open(feature_order_path, 'r') as f: _feature_order = json.load(f)
        else: logger.error(f"LiveStrategy: Feature order not found: {feature_order_path}"); artifacts_loaded = False
            
        if artifacts_loaded: logger.info("LiveStrategy: All CLASSIFIER artifacts loaded successfully.")
        else: logger.error("LiveStrategy: One or more CLASSIFIER artifacts failed to load.")

    except Exception as e:
        logger.critical(f"LiveStrategy: CRITICAL error loading CLASSIFIER artifacts: {e}", exc_info=True)
        return False
    return artifacts_loaded

def generate_features_from_live_data(
    df_closed_btc_klines_history: pd.DataFrame, 
    current_kalshi_snapshot: dict | None, 
    df_kalshi_mid_price_history: pd.DataFrame, # This is resampled to 1min, last, ffill
    kalshi_strike_price: float,
    decision_dt_utc: dt.datetime, 
    kalshi_market_close_dt_utc: dt.datetime
) -> pd.Series | None:
    if _feature_order is None:
        logger.error("LiveStrategy: Feature order not loaded. Cannot generate features.")
        return None
    
    features = pd.Series(index=_feature_order, dtype=float)
    # Features are based on data available *before* the decision_dt_utc.
    # For 1-min klines, this means the kline that closed at decision_dt_utc - 1 minute.
    # Its start timestamp is decision_dt_utc - 1 minute (if decision_dt_utc is top of minute).
    signal_btc_kline_start_ts_s = int((decision_dt_utc - timedelta(minutes=1)).replace(second=0, microsecond=0).timestamp())

    if df_closed_btc_klines_history.empty or signal_btc_kline_start_ts_s not in df_closed_btc_klines_history.index:
        logger.debug(f"LiveStrategy: BTC kline data insufficient or signal_ts {signal_btc_kline_start_ts_s} "
                       f"(for decision {decision_dt_utc.isoformat()}) not in df_closed_btc_klines_history.")
        return None
        
    btc_row_at_signal = df_closed_btc_klines_history.loc[signal_btc_kline_start_ts_s]
    
    current_btc_price = btc_row_at_signal.get('close')
    if pd.isna(current_btc_price):
        logger.debug(f"LiveStrategy: btc_price_t_minus_1 (close) is NaN for ts {signal_btc_kline_start_ts_s}.")
        return None
    features['btc_price_t_minus_1'] = current_btc_price

    # Populate BTC TA features
    for window in BTC_MOMENTUM_WINDOWS:
        col = f'btc_mom_{window}m'; features[col] = btc_row_at_signal.get(col, 0.0)
    if BTC_VOLATILITY_WINDOW > 0:
        col = f'btc_vol_{BTC_VOLATILITY_WINDOW}m'; features[col] = btc_row_at_signal.get(col, 0.0)
    for window in BTC_SMA_WINDOWS:
        col = f'btc_sma_{window}m'; features[col] = btc_row_at_signal.get(col, current_btc_price)
    for window in BTC_EMA_WINDOWS:
        col = f'btc_ema_{window}m'; features[col] = btc_row_at_signal.get(col, current_btc_price)
    if BTC_RSI_WINDOW > 0:
        col = 'btc_rsi'; features[col] = btc_row_at_signal.get(col, 50.0)
    elif 'btc_rsi' in _feature_order: features['btc_rsi'] = 50.0


    features['distance_to_strike'] = current_btc_price - kalshi_strike_price

    # Kalshi market price features
    _yes_bid_feat, _yes_ask_feat, _spread_feat, _mid_feat = 0.0, 100.0, 100.0, 50.0 # Imputed defaults
    if current_kalshi_snapshot and \
       pd.notna(current_kalshi_snapshot.get('yes_bid')) and \
       pd.notna(current_kalshi_snapshot.get('yes_ask')) and \
       0 < current_kalshi_snapshot['yes_bid'] < 100 and \
       0 < current_kalshi_snapshot['yes_ask'] < 100 and \
       current_kalshi_snapshot['yes_bid'] < current_kalshi_snapshot['yes_ask']:
        _yes_bid_feat = current_kalshi_snapshot['yes_bid']
        _yes_ask_feat = current_kalshi_snapshot['yes_ask']
        _spread_feat = _yes_ask_feat - _yes_bid_feat
        _mid_feat = (_yes_bid_feat + _yes_ask_feat) / 2.0
    
    if 'kalshi_yes_bid' in _feature_order: features['kalshi_yes_bid'] = _yes_bid_feat
    if 'kalshi_yes_ask' in _feature_order: features['kalshi_yes_ask'] = _yes_ask_feat
    if 'kalshi_spread' in _feature_order: features['kalshi_spread'] = _spread_feat
    if 'kalshi_mid_price' in _feature_order: features['kalshi_mid_price'] = _mid_feat
    
    # Kalshi historical (volume, OI, mid_chg)
    # For live, volume/OI from snapshot might not be candle volume/OI.
    # Using 0 for these as live snapshot might not have aggregated candle vol/OI.
    # If your `current_kalshi_snapshot` includes these, map them.
    features['kalshi_volume_t_minus_1'] = 0.0 # Placeholder for live scenario
    features['kalshi_open_interest_t_minus_1'] = 0.0 # Placeholder for live scenario

    # Kalshi mid price changes
    # df_kalshi_mid_price_history should be indexed by UTC timestamp (datetime object)
    current_mid_price_for_chg = features.get('kalshi_mid_price') # From current snapshot
    if pd.notna(current_mid_price_for_chg) and not df_kalshi_mid_price_history.empty:
        for window_min in KALSHI_PRICE_CHANGE_WINDOWS:
            col_name = f'kalshi_mid_chg_{window_min}m'
            if col_name in _feature_order:
                # Target past timestamp for mid price history lookup
                past_dt_utc_for_chg = decision_dt_utc - timedelta(minutes=window_min)
                # Find row at or just before this past_dt_utc in the resampled history
                # The history df is already 1-min resampled, last(), ffill()
                past_mid_rows = df_kalshi_mid_price_history[df_kalshi_mid_price_history.index <= past_dt_utc_for_chg]
                if not past_mid_rows.empty:
                    past_mid_val = past_mid_rows.iloc[-1]['kalshi_mid_price'] # Mid price from history
                    if pd.notna(past_mid_val):
                        features[col_name] = current_mid_price_for_chg - past_mid_val
                    else: features[col_name] = 0.0
                else: features[col_name] = 0.0
    else: # If no current mid or no history, changes are 0
        for window_min in KALSHI_PRICE_CHANGE_WINDOWS:
            col_name = f'kalshi_mid_chg_{window_min}m'
            if col_name in _feature_order: features[col_name] = 0.0

    # Time features
    features['time_until_market_close_min'] = (kalshi_market_close_dt_utc - decision_dt_utc).total_seconds() / 60
    features['hour_of_day_utc'] = decision_dt_utc.hour
    features['day_of_week_utc'] = decision_dt_utc.weekday()
    try: # EDT conversion
        edt_tz = timezone(timedelta(hours=-4)) # Standard EDT offset
        features['hour_of_day_edt'] = decision_dt_utc.astimezone(edt_tz).hour
    except Exception: features['hour_of_day_edt'] = (decision_dt_utc.hour - 4 + 24) % 24
    
    # Ensure all features in _feature_order are present, fill any remaining NaNs with heuristic
    for f_name in _feature_order:
        if pd.isna(features.get(f_name)):
            if 'rsi' in f_name: features[f_name] = 50.0
            elif f_name == 'kalshi_yes_ask': features[f_name] = 100.0
            elif f_name == 'kalshi_spread': features[f_name] = 100.0
            elif f_name == 'kalshi_mid_price': features[f_name] = 50.0
            elif f_name == 'kalshi_yes_bid': features[f_name] = 0.0
            elif 'kalshi_volume' in f_name or 'kalshi_open_interest' in f_name: features[f_name] = 0.0
            else: features[f_name] = 0.0 # General fallback for other features
            logger.debug(f"LiveStrategy: Filled NaN for feature '{f_name}' with heuristic value.")
            
    return features.reindex(_feature_order) # Final reindex to ensure order and presence

def calculate_model_prediction_proba(feature_vector: pd.Series) -> float | None:
    """Calculates the probability of the market resolving YES using the classifier."""
    if not all([_scaler, _model, _feature_order, _model_classes is not None]):
        logger.error("LiveStrategy: Classifier Model/scaler/features not fully loaded. Cannot predict probability.")
        return None
    if feature_vector is None or feature_vector.empty:
        logger.warning("LiveStrategy: Empty feature vector for probability prediction.")
        return None
    try:
        feature_df_ordered = feature_vector.to_frame().T 
        if feature_df_ordered.isnull().values.any():
            nan_cols = feature_df_ordered.columns[feature_df_ordered.isnull().any()].tolist()
            logger.warning(f"LiveStrategy: NaNs in feature vector before scaling: {nan_cols}. Imputing with 0.")
            feature_df_ordered.fillna(0, inplace=True) 

        scaled_features_array = _scaler.transform(feature_df_ordered)
        
        positive_class_idx = np.where(_model_classes == 1)[0]
        if not positive_class_idx.size:
            logger.error(f"LiveStrategy: Positive class (1) not found in model.classes_ ({_model_classes}).")
            return None
            
        proba_yes = _model.predict_proba(scaled_features_array)[0, positive_class_idx[0]]
        return float(proba_yes)
        
    except Exception as e:
        logger.error(f"LiveStrategy: Error during probability prediction: {e}", exc_info=True)
        # logger.error(f"Feature vector that caused error (first 5): {feature_vector.head().to_dict()}")
        return None

def get_trade_decision(
    predicted_proba_yes: float | None,
    current_kalshi_bid: float | None,
    current_kalshi_ask: float | None
):
    """
    Determines trade action based on predicted P(YES) vs Kalshi implied odds + edge.
    Returns: (trade_action_str, model_prob_for_chosen_side, entry_price_cents_for_chosen_side)
    """
    if predicted_proba_yes is None:
        return None, None, None

    trade_action = None
    model_prob_for_chosen_side = None # This will be P(YES) or P(NO) depending on action
    entry_price_for_chosen_side = None

    # Potential BUY YES
    if pd.notna(current_kalshi_ask) and 0 < current_kalshi_ask < 100:
        implied_proba_yes_at_ask = current_kalshi_ask / 100.0
        if predicted_proba_yes > MIN_MODEL_PROB_FOR_CONSIDERATION and \
           predicted_proba_yes > (implied_proba_yes_at_ask + EDGE_THRESHOLD_FOR_TRADE):
            trade_action = "BUY_YES"
            model_prob_for_chosen_side = predicted_proba_yes
            entry_price_for_chosen_side = current_kalshi_ask
            # logger.debug(f"Decision Eval: BUY_YES? P(model_yes)={predicted_proba_yes:.3f} vs P(implied_ask)={implied_proba_yes_at_ask:.3f} + edge={EDGE_THRESHOLD_FOR_TRADE:.3f}")

    # Potential BUY NO (only if no BUY_YES decision)
    if trade_action is None and pd.notna(current_kalshi_bid) and 0 < current_kalshi_bid < 100:
        predicted_proba_no = 1.0 - predicted_proba_yes
        
        cost_of_no_contract_cents = 100.0 - current_kalshi_bid
        if 0 < cost_of_no_contract_cents < 100: # Ensure cost of NO is valid
            implied_proba_no_at_bid = cost_of_no_contract_cents / 100.0
            
            if predicted_proba_no > MIN_MODEL_PROB_FOR_CONSIDERATION and \
               predicted_proba_no > (implied_proba_no_at_bid + EDGE_THRESHOLD_FOR_TRADE):
                trade_action = "BUY_NO"
                model_prob_for_chosen_side = predicted_proba_no # Store P(NO)
                entry_price_for_chosen_side = cost_of_no_contract_cents
                # logger.debug(f"Decision Eval: BUY_NO? P(model_no)={predicted_proba_no:.3f} vs P(implied_bid_for_no)={implied_proba_no_at_bid:.3f} + edge={EDGE_THRESHOLD_FOR_TRADE:.3f}")

    # if trade_action is None:
    #     logger.debug(f"Decision Eval: NO_TRADE | P(model_yes)={predicted_proba_yes:.3f}, Ask:{current_kalshi_ask}, Bid:{current_kalshi_bid}")

    return trade_action, model_prob_for_chosen_side, entry_price_for_chosen_side