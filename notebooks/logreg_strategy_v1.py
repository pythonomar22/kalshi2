# logreg_strategy_v1.py
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import logging
import datetime as dt
from datetime import timezone, timedelta

# Import TA parameters from utils to ensure consistency
from utils import (
    BTC_MOMENTUM_WINDOWS, BTC_VOLATILITY_WINDOW, 
    BTC_SMA_WINDOWS, BTC_EMA_WINDOWS, BTC_RSI_WINDOW
)
# KALSHI_PRICE_CHANGE_WINDOWS is specific to this strategy's feature generation logic
KALSHI_PRICE_CHANGE_WINDOWS = [1, 3, 5]


logger = logging.getLogger(__name__)

MODEL_DIR = Path("./trained_models/logreg") # Default, can be overridden by backtester

# --- Trade Decision Parameters ---
# Min probability from model to even consider a trade direction
MIN_MODEL_PROB_FOR_CONSIDERATION = 0.52 # e.g., model must be at least 52% confident for YES or NO

# Min edge: Model's P(event) must be this much > Kalshi's Implied P(event)
EDGE_THRESHOLD_FOR_TRADE = 0.03  # e.g., 3% edge required

# --- Global variables for loaded model artifacts ---
_scaler = None
_model = None 
_feature_order = None
_model_classes = None

def load_model_artifacts(model_base_dir_override: Path = None):
    global _scaler, _model, _feature_order, _model_classes, MODEL_DIR

    current_model_dir = model_base_dir_override if model_base_dir_override else MODEL_DIR
    # Try to find it relative to this script's parent if default isn't working
    if not current_model_dir.is_dir(): 
        # Assuming strategy is in notebooks/ and trained_models is also in notebooks/
        # Then current_model_dir would be relative to notebooks/
        alt_model_dir = Path(__file__).resolve().parent / "trained_models" / "logreg"
        if alt_model_dir.exists():
            current_model_dir = alt_model_dir
        else: 
            alt_model_dir_2 = Path(__file__).resolve().parent.parent / "trained_models" / "logreg"
            if alt_model_dir_2.exists():
                current_model_dir = alt_model_dir_2
            else:
                logger.error(f"Strategy: Model directory not found at {current_model_dir.resolve()} or common alternatives. Please set MODEL_DIR in backtester.")
                return False
    
    logger.info(f"Strategy: Attempting to load artifacts from {current_model_dir.resolve()}")

    scaler_path = current_model_dir / "feature_scaler_classifier_v1.joblib"
    model_path = current_model_dir / "logistic_regression_btc_classifier_v1.joblib"
    feature_order_path = current_model_dir / "feature_columns_classifier_v1.json" # From training
    
    artifacts_loaded = True
    try:
        if scaler_path.exists(): _scaler = joblib.load(scaler_path)
        else: logger.error(f"Scaler not found: {scaler_path}"); artifacts_loaded = False
        
        if model_path.exists(): 
            _model = joblib.load(model_path)
            _model_classes = _model.classes_
        else: logger.error(f"Model not found: {model_path}"); artifacts_loaded = False
        
        if feature_order_path.exists():
            with open(feature_order_path, 'r') as f: _feature_order = json.load(f)
        else: logger.error(f"Feature order not found: {feature_order_path}"); artifacts_loaded = False
            
        if artifacts_loaded: logger.info("Strategy: All artifacts loaded successfully.")
        else: logger.error("Strategy: One or more artifacts failed to load.")

    except Exception as e:
        logger.critical(f"Strategy: CRITICAL error loading artifacts: {e}", exc_info=True)
        return False
    return artifacts_loaded

if not load_model_artifacts(): # Attempt to load on module import
    logger.warning("Strategy: Failed to load all model artifacts on import. Predictions may fail if MODEL_DIR not set by caller.")


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
        logger.error("Strategy (Classifier): Feature order not loaded. Cannot generate features.")
        return None
    
    features = pd.Series(index=_feature_order, dtype=float)
    signal_ts_utc = int(decision_point_dt_utc.timestamp()) - 60 

    if btc_price_history_df.empty or signal_ts_utc not in btc_price_history_df.index:
        logger.debug(f"Strategy (Classifier): BTC data insufficient or signal_ts {signal_ts_utc} not in history for {decision_point_dt_utc.isoformat()}")
        return None
    if 'close' not in btc_price_history_df.columns: 
        logger.error("Strategy (Classifier): 'close' column missing in btc_price_history_df.")
        return None
        
    features['btc_price_t_minus_1'] = btc_price_history_df.loc[signal_ts_utc, 'close']
    if pd.isna(features['btc_price_t_minus_1']):
        logger.debug(f"Strategy (Classifier): btc_price_t_minus_1 is NaN at {signal_ts_utc}.")
        return None

    row_at_signal_ts = btc_price_history_df.loc[signal_ts_utc]
    for window in BTC_MOMENTUM_WINDOWS:
        col_name = f'btc_mom_{window}m'
        if col_name in _feature_order: features[col_name] = row_at_signal_ts.get(col_name, 0.0)
    
    if BTC_VOLATILITY_WINDOW > 0:
        col_name = f'btc_vol_{BTC_VOLATILITY_WINDOW}m'
        if col_name in _feature_order: features[col_name] = row_at_signal_ts.get(col_name, 0.0)

    for window in BTC_SMA_WINDOWS:
        col_name = f'btc_sma_{window}m'
        if col_name in _feature_order: features[col_name] = row_at_signal_ts.get(col_name, features['btc_price_t_minus_1'])
    for window in BTC_EMA_WINDOWS:
        col_name = f'btc_ema_{window}m'
        if col_name in _feature_order: features[col_name] = row_at_signal_ts.get(col_name, features['btc_price_t_minus_1'])
    
    if BTC_RSI_WINDOW > 0:
        col_name = 'btc_rsi'
        if col_name in _feature_order: features[col_name] = row_at_signal_ts.get(col_name, 50.0)
    elif 'btc_rsi' in _feature_order: features['btc_rsi'] = 50.0

    if 'distance_to_strike' in _feature_order:
        features['distance_to_strike'] = features['btc_price_t_minus_1'] - kalshi_strike_price

    _yes_bid_feat, _yes_ask_feat, _spread_feat, _mid_feat = 0.0, 100.0, 100.0, 50.0 # Imputed defaults
    if current_kalshi_bid is not None and current_kalshi_ask is not None and \
       0 < current_kalshi_bid < 100 and 0 < current_kalshi_ask < 100 and current_kalshi_bid < current_kalshi_ask:
        _yes_bid_feat = current_kalshi_bid
        _yes_ask_feat = current_kalshi_ask
        _spread_feat = current_kalshi_ask - current_kalshi_bid
        _mid_feat = (current_kalshi_bid + current_kalshi_ask) / 2.0
    
    if 'kalshi_yes_bid' in _feature_order: features['kalshi_yes_bid'] = _yes_bid_feat
    if 'kalshi_yes_ask' in _feature_order: features['kalshi_yes_ask'] = _yes_ask_feat
    if 'kalshi_spread' in _feature_order: features['kalshi_spread'] = _spread_feat
    if 'kalshi_mid_price' in _feature_order: features['kalshi_mid_price'] = _mid_feat
    
    _latest_volume, _latest_oi = 0.0, 0.0
    # Default mid_chg features to 0
    for window in KALSHI_PRICE_CHANGE_WINDOWS:
        col_name = f'kalshi_mid_chg_{window}m'
        if col_name in _feature_order: features[col_name] = 0.0

    if kalshi_market_history_df is not None and not kalshi_market_history_df.empty:
        relevant_kalshi_rows = kalshi_market_history_df[kalshi_market_history_df.index <= signal_ts_utc]
        if not relevant_kalshi_rows.empty:
            latest_kalshi_candle = relevant_kalshi_rows.iloc[-1]
            _latest_volume = latest_kalshi_candle.get('volume', 0.0)
            _latest_oi = latest_kalshi_candle.get('open_interest', 0.0)
            
            current_mid_for_chg = features.get('kalshi_mid_price')
            if pd.notna(current_mid_for_chg): # Only calculate if current mid is valid
                for window in KALSHI_PRICE_CHANGE_WINDOWS:
                    col_name = f'kalshi_mid_chg_{window}m'
                    if col_name in _feature_order:
                        past_ts = signal_ts_utc - (window * 60)
                        past_kalshi_mid_rows = kalshi_market_history_df[kalshi_market_history_df.index <= past_ts]
                        if not past_kalshi_mid_rows.empty:
                            past_k_candle_for_chg = past_kalshi_mid_rows.iloc[-1]
                            past_bid_chg = past_k_candle_for_chg.get('yes_bid_close_cents')
                            past_ask_chg = past_k_candle_for_chg.get('yes_ask_close_cents')
                            if pd.notna(past_bid_chg) and pd.notna(past_ask_chg):
                                past_mid_val = (past_bid_chg + past_ask_chg) / 2.0
                                features[col_name] = current_mid_for_chg - past_mid_val
                            # else features[col_name] remains 0 (defaulted above)
                        # else features[col_name] remains 0
                        
    if 'kalshi_volume_t_minus_1' in _feature_order: features['kalshi_volume_t_minus_1'] = _latest_volume
    if 'kalshi_open_interest_t_minus_1' in _feature_order: features['kalshi_open_interest_t_minus_1'] = _latest_oi

    if 'time_until_market_close_min' in _feature_order:
        features['time_until_market_close_min'] = (kalshi_market_close_dt_utc - decision_point_dt_utc).total_seconds() / 60
    if 'hour_of_day_utc' in _feature_order: features['hour_of_day_utc'] = decision_point_dt_utc.hour
    if 'day_of_week_utc' in _feature_order: features['day_of_week_utc'] = decision_point_dt_utc.weekday()
    
    if 'hour_of_day_edt' in _feature_order:
        try:
            edt_tz = timezone(timedelta(hours=-4)) # Standard EDT offset
            features['hour_of_day_edt'] = decision_point_dt_utc.astimezone(edt_tz).hour
        except Exception: features['hour_of_day_edt'] = (decision_point_dt_utc.hour - 4 + 24) % 24

    # Final check for NaNs for features that should always have a value
    for f_name in _feature_order:
        if pd.isna(features.get(f_name)):
            # This block is a safety net. Ideally, all features are handled above.
            # logger.warning(f"Strategy (Classifier): Feature '{f_name}' was NaN at end of generation. Defaulting to 0 or specific heuristic.")
            if 'rsi' in f_name: features[f_name] = 50.0
            elif f_name == 'kalshi_yes_ask': features[f_name] = 100.0
            elif f_name == 'kalshi_spread': features[f_name] = 100.0
            elif f_name == 'kalshi_mid_price': features[f_name] = 50.0
            else: features[f_name] = 0.0
            
    return features.reindex(_feature_order)


def calculate_model_prediction_proba(feature_vector: pd.Series) -> float | None:
    if not all([_scaler, _model, _feature_order, _model_classes is not None]):
        logger.error("Strategy (Classifier): Model/scaler/features not fully loaded. Cannot predict probability.")
        return None
    if feature_vector is None or feature_vector.empty:
        logger.warning("Strategy (Classifier): Empty feature vector for probability prediction.")
        return None
    try:
        feature_df_ordered = feature_vector.to_frame().T
        
        if feature_df_ordered.isnull().values.any():
            nan_cols = feature_df_ordered.columns[feature_df_ordered.isnull().any()].tolist()
            logger.warning(f"Strategy (Classifier): NaNs in feature vector before scaling: {nan_cols}. Filling with 0.")
            feature_df_ordered.fillna(0, inplace=True)

        scaled_features_array = _scaler.transform(feature_df_ordered)
        
        positive_class_idx = np.where(_model_classes == 1)[0]
        if not positive_class_idx.size: # Check if empty
            logger.error(f"Strategy (Classifier): Positive class (1) not found in model.classes_ ({_model_classes}).")
            return None
            
        proba_yes = _model.predict_proba(scaled_features_array)[0, positive_class_idx[0]]
        return float(proba_yes)
        
    except Exception as e:
        logger.error(f"Strategy (Classifier): Error during probability prediction: {e}", exc_info=True)
        return None

def get_trade_decision(
    predicted_proba_yes: float | None,
    current_kalshi_bid: float | None, # Yes bid price in cents
    current_kalshi_ask: float | None  # Yes ask price in cents
):
    """
    Determines trade action based on predicted probability vs Kalshi implied odds + edge.
    Returns: (trade_action_str, model_prob_for_chosen_side, entry_price_cents_for_chosen_side)
             Returns (None, None, None) if no trade.
    """
    if predicted_proba_yes is None:
        return None, None, None

    trade_action = None
    model_prob_for_chosen_side = None
    entry_price_for_chosen_side = None

    # Consider BUY YES
    if pd.notna(current_kalshi_ask) and 0 < current_kalshi_ask < 100:
        implied_proba_yes_at_ask = current_kalshi_ask / 100.0
        if predicted_proba_yes > MIN_MODEL_PROB_FOR_CONSIDERATION and \
           predicted_proba_yes > (implied_proba_yes_at_ask + EDGE_THRESHOLD_FOR_TRADE):
            trade_action = "BUY_YES"
            model_prob_for_chosen_side = predicted_proba_yes
            entry_price_for_chosen_side = current_kalshi_ask
            logger.debug(f"Decision: BUY_YES | P(model_yes)={predicted_proba_yes:.3f} vs P(implied_yes_ask)={implied_proba_yes_at_ask:.3f} + edge={EDGE_THRESHOLD_FOR_TRADE:.3f}")


    # Consider BUY NO (only if no BUY_YES decision was made)
    if trade_action is None and pd.notna(current_kalshi_bid) and 0 < current_kalshi_bid < 100:
        predicted_proba_no = 1.0 - predicted_proba_yes
        
        # Cost to buy NO is 100 - current_kalshi_bid
        # Implied P(NO) from this cost is (100 - current_kalshi_bid) / 100.0
        cost_of_no_contract_cents = 100.0 - current_kalshi_bid
        implied_proba_no_at_bid = cost_of_no_contract_cents / 100.0
        
        if predicted_proba_no > MIN_MODEL_PROB_FOR_CONSIDERATION and \
           predicted_proba_no > (implied_proba_no_at_bid + EDGE_THRESHOLD_FOR_TRADE):
            trade_action = "BUY_NO"
            model_prob_for_chosen_side = predicted_proba_no
            entry_price_for_chosen_side = cost_of_no_contract_cents
            logger.debug(f"Decision: BUY_NO | P(model_no)={predicted_proba_no:.3f} vs P(implied_no_bid)={implied_proba_no_at_bid:.3f} + edge={EDGE_THRESHOLD_FOR_TRADE:.3f}")

    if trade_action is None:
        logger.debug(f"Decision: NO_TRADE | P(model_yes)={predicted_proba_yes:.3f}, Ask:{current_kalshi_ask}, Bid:{current_kalshi_bid}")


    return trade_action, model_prob_for_chosen_side, entry_price_for_chosen_side