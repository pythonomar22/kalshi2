# notebooks/classifier_strategy.py
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import logging
import datetime as dt
from datetime import timezone, timedelta

# Import TA parameters from utils to ensure consistency
try:
    from . import utils as main_utils 
    BTC_MOMENTUM_WINDOWS = main_utils.BTC_MOMENTUM_WINDOWS
    BTC_VOLATILITY_WINDOW = main_utils.BTC_VOLATILITY_WINDOW
    BTC_SMA_WINDOWS = main_utils.BTC_SMA_WINDOWS
    BTC_EMA_WINDOWS = main_utils.BTC_EMA_WINDOWS
    BTC_RSI_WINDOW = main_utils.BTC_RSI_WINDOW
    logger_strat_utils = logging.getLogger(__name__)
    logger_strat_utils.debug("Strategy: Successfully imported TA parameters from main_utils.")
except ImportError:
    logger_strat_utils = logging.getLogger(__name__)
    logger_strat_utils.warning("Strategy: Could not import TA parameters from main_utils via relative import. Trying direct.")
    try:
        import utils as main_utils 
        BTC_MOMENTUM_WINDOWS = main_utils.BTC_MOMENTUM_WINDOWS
        BTC_VOLATILITY_WINDOW = main_utils.BTC_VOLATILITY_WINDOW
        BTC_SMA_WINDOWS = main_utils.BTC_SMA_WINDOWS
        BTC_EMA_WINDOWS = main_utils.BTC_EMA_WINDOWS
        BTC_RSI_WINDOW = main_utils.BTC_RSI_WINDOW
        logger_strat_utils.info("Strategy: Successfully imported TA parameters from main_utils via direct import.")
    except ImportError:
        logger_strat_utils.error("Strategy: CRITICAL - Could not import TA parameters from main_utils. Using local defaults.")
        BTC_MOMENTUM_WINDOWS = [5, 10, 15, 30]; BTC_VOLATILITY_WINDOW = 15
        BTC_SMA_WINDOWS = [10, 30]; BTC_EMA_WINDOWS = [12, 26]; BTC_RSI_WINDOW = 14

KALSHI_PRICE_CHANGE_WINDOWS = [1, 3, 5]
logger = logging.getLogger(__name__)
MODEL_ARTIFACTS_BASE_DIR = Path(".")
MIN_MODEL_PROB_FOR_CONSIDERATION = 0.55
EDGE_THRESHOLD_FOR_TRADE = 0.05
_scaler = None; _model = None; _feature_order = None; _model_classes = None
_model_type_loaded = "None"

def load_model_artifacts(artifacts_directory: Path, model_type_name: str):
    global _scaler, _model, _feature_order, _model_classes, _model_type_loaded, MODEL_ARTIFACTS_BASE_DIR
    MODEL_ARTIFACTS_BASE_DIR = artifacts_directory
    logger.info(f"Strategy: Attempting to load '{model_type_name}' artifacts from {MODEL_ARTIFACTS_BASE_DIR.resolve()}")
    scaler_path = MODEL_ARTIFACTS_BASE_DIR / "feature_scaler_classifier_v1.joblib"
    feature_order_path = MODEL_ARTIFACTS_BASE_DIR / "feature_columns_classifier_v1.json"
    
    if model_type_name == "logistic_regression": model_filename = "logistic_regression_btc_classifier_v1.joblib"
    elif model_type_name == "random_forest": model_filename = "RandomForest_classifier_v1.joblib"
    else: logger.error(f"Strategy: Unknown model_type_name '{model_type_name}'."); return False
        
    model_path = MODEL_ARTIFACTS_BASE_DIR / model_filename
    artifacts_loaded_successfully = True
    try:
        if scaler_path.exists(): _scaler = joblib.load(scaler_path); logger.debug(f"Scaler loaded.")
        else: logger.error(f"Scaler not found: {scaler_path}"); artifacts_loaded_successfully = False
        
        if model_path.exists(): 
            _model = joblib.load(model_path); logger.debug(f"Model '{model_filename}' loaded.")
            if hasattr(_model, 'classes_'): _model_classes = _model.classes_
            else: logger.warning(f"Model type {model_type_name} lacks 'classes_'. Assuming [0, 1]."); _model_classes = np.array([0,1])
        else: logger.error(f"Model file '{model_filename}' not found: {model_path}"); artifacts_loaded_successfully = False
        
        if feature_order_path.exists():
            with open(feature_order_path, 'r') as f: _feature_order = json.load(f)
            logger.debug(f"Feature order loaded.")
        else: logger.error(f"Feature order not found: {feature_order_path}"); artifacts_loaded_successfully = False
            
        if artifacts_loaded_successfully:
            _model_type_loaded = model_type_name
            logger.info(f"Strategy: All '{_model_type_loaded}' artifacts loaded successfully.")
        else: logger.error(f"Strategy: Artifact load failed for '{model_type_name}'.")
    except Exception as e:
        logger.critical(f"Strategy: CRITICAL error loading '{model_type_name}' artifacts: {e}", exc_info=True)
        return False
    return artifacts_loaded_successfully

def generate_live_features(
    btc_price_history_df: pd.DataFrame, current_kalshi_bid: float | None, current_kalshi_ask: float | None,
    kalshi_market_history_df: pd.DataFrame | None, kalshi_strike_price: float,
    decision_point_dt_utc: dt.datetime, kalshi_market_close_dt_utc: dt.datetime
) -> pd.Series | None:
    if _feature_order is None: logger.error(f"Strategy ({_model_type_loaded}): Feature order not loaded."); return None
    features = pd.Series(index=_feature_order, dtype=float)
    signal_ts_utc = int((decision_point_dt_utc - timedelta(minutes=1)).replace(second=0, microsecond=0).timestamp())

    if btc_price_history_df.empty or signal_ts_utc not in btc_price_history_df.index:
        logger.debug(f"Strategy ({_model_type_loaded}): BTC data insufficient for ts {signal_ts_utc}."); return None
    if 'close' not in btc_price_history_df.columns: 
        logger.error(f"Strategy ({_model_type_loaded}): 'close' col missing in btc_price_history_df."); return None
        
    btc_row_at_signal = btc_price_history_df.loc[signal_ts_utc]
    current_btc_price = btc_row_at_signal.get('close')
    if pd.isna(current_btc_price): logger.debug(f"Strategy ({_model_type_loaded}): btc_price_t_minus_1 NaN."); return None
    features['btc_price_t_minus_1'] = current_btc_price

    for window in BTC_MOMENTUM_WINDOWS:
        col = f'btc_mom_{window}m'; features[col] = btc_row_at_signal.get(col, 0.0) if col in _feature_order else np.nan
    if BTC_VOLATILITY_WINDOW > 0:
        col = f'btc_vol_{BTC_VOLATILITY_WINDOW}m'; features[col] = btc_row_at_signal.get(col, 0.0) if col in _feature_order else np.nan
    for window in BTC_SMA_WINDOWS:
        col = f'btc_sma_{window}m'; features[col] = btc_row_at_signal.get(col, current_btc_price) if col in _feature_order else np.nan
    for window in BTC_EMA_WINDOWS:
        col = f'btc_ema_{window}m'; features[col] = btc_row_at_signal.get(col, current_btc_price) if col in _feature_order else np.nan
    if BTC_RSI_WINDOW > 0:
        col = 'btc_rsi'; features[col] = btc_row_at_signal.get(col, 50.0) if col in _feature_order else np.nan
    elif 'btc_rsi' in _feature_order: features['btc_rsi'] = 50.0

    if 'distance_to_strike' in _feature_order: features['distance_to_strike'] = current_btc_price - kalshi_strike_price
    
    _yes_bid_f, _yes_ask_f, _spread_f, _mid_f = 0.0, 100.0, 100.0, 50.0
    if pd.notna(current_kalshi_bid) and pd.notna(current_kalshi_ask) and \
       0 <= current_kalshi_bid <= 100 and 0 <= current_kalshi_ask <= 100 and current_kalshi_bid <= current_kalshi_ask:
        _yes_bid_f, _yes_ask_f = current_kalshi_bid, current_kalshi_ask
        _spread_f = _yes_ask_f - _yes_bid_f
        _mid_f = (_yes_bid_f + _yes_ask_f) / 2.0 if (_yes_bid_f !=0 or _yes_ask_f !=100) else 50.0
    
    if 'kalshi_yes_bid' in _feature_order: features['kalshi_yes_bid'] = _yes_bid_f
    if 'kalshi_yes_ask' in _feature_order: features['kalshi_yes_ask'] = _yes_ask_f
    if 'kalshi_spread' in _feature_order: features['kalshi_spread'] = _spread_f
    if 'kalshi_mid_price' in _feature_order: features['kalshi_mid_price'] = _mid_f
    
    _vol, _oi = 0.0, 0.0
    for w in KALSHI_PRICE_CHANGE_WINDOWS:
        col = f'kalshi_mid_chg_{w}m'; features[col] = 0.0 if col in _feature_order else np.nan

    if kalshi_market_history_df is not None and not kalshi_market_history_df.empty:
        hist_k_rows = kalshi_market_history_df[kalshi_market_history_df.index <= signal_ts_utc]
        if not hist_k_rows.empty:
            latest_hist_k = hist_k_rows.iloc[-1]
            _vol, _oi = latest_hist_k.get('volume', 0.0), latest_hist_k.get('open_interest', 0.0)
            curr_mid_chg = features.get('kalshi_mid_price')
            if pd.notna(curr_mid_chg):
                for w in KALSHI_PRICE_CHANGE_WINDOWS:
                    col = f'kalshi_mid_chg_{w}m'
                    if col in _feature_order:
                        past_ts_chg = signal_ts_utc - (w * 60)
                        past_k_mid_rows = kalshi_market_history_df[kalshi_market_history_df.index <= past_ts_chg]
                        if not past_k_mid_rows.empty:
                            past_k_candle_chg = past_k_mid_rows.iloc[-1]
                            past_b, past_a = past_k_candle_chg.get('yes_bid_close_cents'), past_k_candle_chg.get('yes_ask_close_cents')
                            if pd.notna(past_b) and pd.notna(past_a): features[col] = curr_mid_chg - ((past_b + past_a) / 2.0)
    if 'kalshi_volume_t_minus_1' in _feature_order: features['kalshi_volume_t_minus_1'] = _vol
    if 'kalshi_open_interest_t_minus_1' in _feature_order: features['kalshi_open_interest_t_minus_1'] = _oi

    if 'time_until_market_close_min' in _feature_order:
        features['time_until_market_close_min'] = (kalshi_market_close_dt_utc - decision_point_dt_utc).total_seconds() / 60
    if 'hour_of_day_utc' in _feature_order: features['hour_of_day_utc'] = decision_point_dt_utc.hour
    if 'day_of_week_utc' in _feature_order: features['day_of_week_utc'] = decision_point_dt_utc.weekday()
    if 'hour_of_day_edt' in _feature_order:
        try: features['hour_of_day_edt'] = decision_point_dt_utc.astimezone(timezone(timedelta(hours=-4))).hour
        except Exception: features['hour_of_day_edt'] = (decision_point_dt_utc.hour - 4 + 24) % 24

    for f_name in _feature_order: 
        if pd.isna(features.get(f_name)):
            if 'rsi' in f_name: features[f_name] = 50.0
            elif f_name == 'kalshi_yes_ask': features[f_name] = 100.0
            elif f_name == 'kalshi_spread': features[f_name] = 100.0
            elif f_name == 'kalshi_mid_price': features[f_name] = 50.0
            else: features[f_name] = 0.0 
            logger.debug(f"Strategy ({_model_type_loaded}): Heuristically filled final NaN for '{f_name}'.")
    return features.reindex(_feature_order)

def calculate_model_prediction_proba(feature_vector: pd.Series) -> float | None:
    if not all([_scaler, _model, _feature_order, _model_classes is not None]):
        logger.error(f"Strategy ({_model_type_loaded}): Model/scaler/features not fully loaded."); return None
    if feature_vector is None or feature_vector.empty:
        logger.warning(f"Strategy ({_model_type_loaded}): Empty feature vector for prediction."); return None
    try:
        feature_df = pd.DataFrame([feature_vector], columns=_feature_order)
        if feature_df.isnull().values.any():
            nan_cols = feature_df.columns[feature_df.isnull().any()].tolist()
            logger.warning(f"Strategy ({_model_type_loaded}): NaNs in feature vector pre-scaling: {nan_cols}. Imputing.")
            for col in nan_cols: 
                if 'rsi' in col: feature_df[col].fillna(50.0, inplace=True)
                else: feature_df[col].fillna(0, inplace=True)
        scaled_features_array = _scaler.transform(feature_df)
        scaled_features_df_with_names = pd.DataFrame(scaled_features_array, columns=_feature_order)
        positive_class_idx = np.where(_model_classes == 1)[0]
        if not positive_class_idx.size: 
            logger.error(f"Strategy ({_model_type_loaded}): Positive class (1) not found in model.classes_ ({_model_classes})."); return None
        proba_yes = _model.predict_proba(scaled_features_df_with_names)[0, positive_class_idx[0]]
        return float(proba_yes)
    except Exception as e:
        logger.error(f"Strategy ({_model_type_loaded}): Error during probability prediction: {e}", exc_info=True)
        return None

def get_trade_decision(
    predicted_proba_yes: float | None, current_kalshi_bid: float | None, current_kalshi_ask: float | None
):
    # Ensure all return variables are initialized at the start of the function's scope
    trade_action = None
    model_prob_for_chosen_side = None
    entry_price_for_chosen_side = None

    if predicted_proba_yes is None:
        # Still return the initialized None values
        return trade_action, model_prob_for_chosen_side, entry_price_for_chosen_side

    # --- Try BUY_YES ---
    if pd.notna(current_kalshi_ask) and 0 < current_kalshi_ask < 100:
        implied_proba_yes_at_ask = current_kalshi_ask / 100.0
        if predicted_proba_yes > MIN_MODEL_PROB_FOR_CONSIDERATION and \
           predicted_proba_yes > (implied_proba_yes_at_ask + EDGE_THRESHOLD_FOR_TRADE):
            trade_action = "BUY_YES"
            model_prob_for_chosen_side = predicted_proba_yes
            entry_price_for_chosen_side = current_kalshi_ask

    # --- Try BUY_NO (only if no BUY_YES action) ---
    if trade_action is None: # Check if a BUY_YES decision was already made
        if pd.notna(current_kalshi_bid) and 0 < current_kalshi_bid < 100:
            predicted_proba_no = 1.0 - predicted_proba_yes
            cost_of_no_contract_cents = 100.0 - current_kalshi_bid
            if 0 < cost_of_no_contract_cents < 100:
                implied_proba_no_at_bid = cost_of_no_contract_cents / 100.0
                if predicted_proba_no > MIN_MODEL_PROB_FOR_CONSIDERATION and \
                   predicted_proba_no > (implied_proba_no_at_bid + EDGE_THRESHOLD_FOR_TRADE):
                    trade_action = "BUY_NO"
                    model_prob_for_chosen_side = predicted_proba_no
                    entry_price_for_chosen_side = cost_of_no_contract_cents
    
    # If trade_action is still None here, it means no trade conditions were met.
    # model_prob_for_chosen_side and entry_price_for_chosen_side will correctly be None.
    return trade_action, model_prob_for_chosen_side, entry_price_for_chosen_side