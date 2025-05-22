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
    BTC_ATR_WINDOW = getattr(main_utils, 'BTC_ATR_WINDOW', 14) # Get ATR if defined, else default
    KALSHI_PRICE_CHANGE_WINDOWS = getattr(main_utils, 'KALSHI_PRICE_CHANGE_WINDOWS', [1, 3, 5, 10])
    KALSHI_VOLATILITY_WINDOWS = getattr(main_utils, 'KALSHI_VOLATILITY_WINDOWS', [5, 10])
    KALSHI_MAX_STALENESS_SECONDS_FOR_FEATURES = 120 
    logger_strat_utils = logging.getLogger(__name__)
    logger_strat_utils.debug("Strategy: Successfully imported TA parameters from main_utils.")
except ImportError:
    logger_strat_utils = logging.getLogger(__name__)
    logger_strat_utils.warning("Strategy: Could not import TA parameters from main_utils. Using local defaults.")
    # Ensure all new TA params from feature_engineering are here too
    BTC_MOMENTUM_WINDOWS = [5, 10, 15, 30, 60] 
    BTC_VOLATILITY_WINDOW = 15
    BTC_SMA_WINDOWS = [10, 30, 50] 
    BTC_EMA_WINDOWS = [12, 26, 50] 
    BTC_RSI_WINDOW = 14
    BTC_ATR_WINDOW = 14
    KALSHI_PRICE_CHANGE_WINDOWS = [1, 3, 5, 10]
    KALSHI_VOLATILITY_WINDOWS = [5, 10]
    KALSHI_MAX_STALENESS_SECONDS_FOR_FEATURES = 120

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
    
    _model_type_loaded = model_type_name # Store the passed model_type_name

    # Determine filenames based on model_type_name and versioning convention
    # For now, assume v2 for RF, v1 for LogReg, but this could be more robust
    if _model_type_loaded == "logistic_regression":
        model_filename = "logistic_regression_btc_classifier_v1.joblib" # Or your v2 LogReg name
        scaler_filename = "feature_scaler_classifier_v1.joblib" # Assuming LogReg uses v1 features/scaler
        feature_order_filename = "feature_columns_classifier_v1.json"
    elif _model_type_loaded == "random_forest":
        model_filename = "CalibratedRandomForest_classifier_v2.joblib" 
        scaler_filename = "feature_scaler_classifier_v2.joblib"       # *** USE V2 SCALER ***
        feature_order_filename = "feature_columns_classifier_v2.json" # *** USE V2 FEATURES LIST ***
    else:
        logger.error(f"Strategy: Unknown model_type_name '{_model_type_loaded}'."); return False
        
    model_path = MODEL_ARTIFACTS_BASE_DIR / model_filename
    scaler_path = MODEL_ARTIFACTS_BASE_DIR / scaler_filename
    feature_order_path = MODEL_ARTIFACTS_BASE_DIR / feature_order_filename
    
    logger.info(f"Strategy: Loading model='{model_filename}', scaler='{scaler_filename}', features='{feature_order_filename}'")

    artifacts_loaded_successfully = True
    try:
        if scaler_path.exists(): _scaler = joblib.load(scaler_path); logger.debug(f"Scaler '{scaler_filename}' loaded.")
        else: logger.error(f"Scaler '{scaler_filename}' not found: {scaler_path}"); artifacts_loaded_successfully = False
        
        if model_path.exists(): 
            _model = joblib.load(model_path); logger.debug(f"Model '{model_filename}' loaded.")
            if hasattr(_model, 'classes_'): _model_classes = _model.classes_
            else: logger.warning(f"Model type {_model_type_loaded} lacks 'classes_'. Assuming [0, 1]."); _model_classes = np.array([0,1])
        else: logger.error(f"Model '{model_filename}' not found: {model_path}"); artifacts_loaded_successfully = False
        
        if feature_order_path.exists():
            with open(feature_order_path, 'r') as f: _feature_order = json.load(f)
            logger.debug(f"Feature order '{feature_order_filename}' loaded ({len(_feature_order)} features).")
        else: logger.error(f"Feature order '{feature_order_filename}' not found: {feature_order_path}"); artifacts_loaded_successfully = False
            
        if artifacts_loaded_successfully:
            logger.info(f"Strategy: All '{_model_type_loaded}' (version specific) artifacts loaded successfully.")
        else: logger.error(f"Strategy: Artifact load failed for '{_model_type_loaded}'.")
    except Exception as e:
        logger.critical(f"Strategy: CRITICAL error loading '{_model_type_loaded}' artifacts: {e}", exc_info=True)
        return False
    return artifacts_loaded_successfully

def generate_live_features(
    btc_price_history_df: pd.DataFrame, 
    kalshi_market_history_df: pd.DataFrame | None, 
    kalshi_strike_price: float,
    decision_point_dt_utc: dt.datetime, 
    kalshi_market_close_dt_utc: dt.datetime
) -> pd.Series | None:
    # This function should now generate all 40 features as defined in feature_engineering.ipynb (Cell 3).
    # The previous version of this function (from the "give me full files only" response for V2 features)
    # should be largely correct if _feature_order (now correctly loaded as the 40-feature list) is used.

    if _feature_order is None: logger.error(f"Strategy ({_model_type_loaded}): Feature order not loaded."); return None
    features = pd.Series(index=_feature_order, dtype=float) # Series will be indexed by all 40 feature names
    
    signal_dt_for_features_utc = (decision_point_dt_utc - timedelta(minutes=1)).replace(second=0, microsecond=0)
    signal_ts_utc = int(signal_dt_for_features_utc.timestamp())

    if btc_price_history_df.empty or signal_ts_utc not in btc_price_history_df.index:
        logger.debug(f"Strategy ({_model_type_loaded}): BTC data insufficient for ts {signal_ts_utc} (decision: {decision_point_dt_utc.isoformat()}).")
        return None
        
    btc_row_at_signal = btc_price_history_df.loc[signal_ts_utc]
    current_btc_price_t_minus_1 = btc_row_at_signal.get('close')
    if pd.isna(current_btc_price_t_minus_1): 
        logger.debug(f"Strategy ({_model_type_loaded}): btc_price_t_minus_1 NaN at {signal_ts_utc}.")
        return None
    
    # Populate all features - ensure this logic matches feature_engineering.ipynb Cell 3 for V2 features
    features['btc_price_t_minus_1'] = current_btc_price_t_minus_1
    for window in BTC_MOMENTUM_WINDOWS:
        col = f'btc_mom_{window}m'
        if col in _feature_order: features[col] = btc_row_at_signal.get(col, 0.0) 
    if BTC_VOLATILITY_WINDOW > 0 and f'btc_vol_{BTC_VOLATILITY_WINDOW}m' in _feature_order:
        features[f'btc_vol_{BTC_VOLATILITY_WINDOW}m'] = btc_row_at_signal.get(f'btc_vol_{BTC_VOLATILITY_WINDOW}m', 0.0)
    
    for window in BTC_SMA_WINDOWS: 
        sma_col = f'btc_sma_{window}m'
        vs_sma_col = f'btc_price_vs_sma_{window}m'
        if sma_col in _feature_order:
            sma_val = btc_row_at_signal.get(sma_col, current_btc_price_t_minus_1) # Impute SMA with current price if missing
            features[sma_col] = sma_val
            if vs_sma_col in _feature_order:
                features[vs_sma_col] = current_btc_price_t_minus_1 / sma_val if pd.notna(sma_val) and sma_val != 0 else 1.0
                
    for window in BTC_EMA_WINDOWS:
        ema_col = f'btc_ema_{window}m'
        vs_ema_col = f'btc_price_vs_ema_{window}m'
        if ema_col in _feature_order:
            ema_val = btc_row_at_signal.get(ema_col, current_btc_price_t_minus_1) # Impute EMA with current price
            features[ema_col] = ema_val
            if vs_ema_col in _feature_order:
                features[vs_ema_col] = current_btc_price_t_minus_1 / ema_val if pd.notna(ema_val) and ema_val != 0 else 1.0

    if BTC_RSI_WINDOW > 0 and 'btc_rsi' in _feature_order:
        features['btc_rsi'] = btc_row_at_signal.get('btc_rsi', 50.0)
    elif 'btc_rsi' in _feature_order: features['btc_rsi'] = 50.0 # Default if not calculated

    atr_col = f'btc_atr_{BTC_ATR_WINDOW}'
    if BTC_ATR_WINDOW > 0 and atr_col in _feature_order:
        features[atr_col] = btc_row_at_signal.get(atr_col, np.nan) # Keep NaN for now, impute later

    # --- Kalshi Features (from t-1) ---
    _yes_bid_t_minus_1, _yes_ask_t_minus_1, _mid_t_minus_1 = 0.0, 100.0, 50.0
    _vol_t_minus_1, _oi_t_minus_1 = 0.0, 0.0
    
    kalshi_row_at_signal = None
    if kalshi_market_history_df is not None and not kalshi_market_history_df.empty:
        relevant_kalshi_rows_t_minus_1 = kalshi_market_history_df[kalshi_market_history_df.index <= signal_ts_utc]
        if not relevant_kalshi_rows_t_minus_1.empty:
            latest_kalshi_row_t_minus_1 = relevant_kalshi_rows_t_minus_1.iloc[-1]
            latest_kalshi_ts_t_minus_1 = latest_kalshi_row_t_minus_1.name
            if (signal_ts_utc - latest_kalshi_ts_t_minus_1) <= KALSHI_MAX_STALENESS_SECONDS_FOR_FEATURES:
                kalshi_row_at_signal = latest_kalshi_row_t_minus_1
    
    if kalshi_row_at_signal is not None:
        _yes_bid_val = kalshi_row_at_signal.get('yes_bid_close_cents')
        _yes_ask_val = kalshi_row_at_signal.get('yes_ask_close_cents')
        if pd.notna(_yes_bid_val) and pd.notna(_yes_ask_val) and 0 <= _yes_bid_val <= 100 and 0 <= _yes_ask_val <= 100 and _yes_bid_val <= _yes_ask_val:
            _yes_bid_t_minus_1 = _yes_bid_val
            _yes_ask_t_minus_1 = _yes_ask_val
            _mid_t_minus_1 = (_yes_bid_t_minus_1 + _yes_ask_t_minus_1) / 2.0
        _vol_t_minus_1 = kalshi_row_at_signal.get('volume', 0.0)
        _oi_t_minus_1 = kalshi_row_at_signal.get('open_interest', 0.0)

    if 'kalshi_yes_bid' in _feature_order: features['kalshi_yes_bid'] = _yes_bid_t_minus_1
    if 'kalshi_yes_ask' in _feature_order: features['kalshi_yes_ask'] = _yes_ask_t_minus_1
    if 'kalshi_mid_price' in _feature_order: features['kalshi_mid_price'] = _mid_t_minus_1
    if 'kalshi_spread' in _feature_order: features['kalshi_spread'] = _yes_ask_t_minus_1 - _yes_bid_t_minus_1 if pd.notna(_yes_ask_t_minus_1) and pd.notna(_yes_bid_t_minus_1) else 100.0
    if 'kalshi_volume_t_minus_1' in _feature_order: features['kalshi_volume_t_minus_1'] = _vol_t_minus_1
    if 'kalshi_open_interest_t_minus_1' in _feature_order: features['kalshi_open_interest_t_minus_1'] = _oi_t_minus_1

    for w_min in KALSHI_PRICE_CHANGE_WINDOWS:
        col = f'kalshi_mid_chg_{w_min}m'
        if col in _feature_order:
            features[col] = 0.0 
            if kalshi_market_history_df is not None and not kalshi_market_history_df.empty and pd.notna(_mid_t_minus_1):
                past_ts_for_chg_calc = signal_ts_utc - (w_min * 60)
                past_kalshi_mid_rows_for_chg = kalshi_market_history_df[kalshi_market_history_df.index <= past_ts_for_chg_calc]
                if not past_kalshi_mid_rows_for_chg.empty:
                    past_k_candle_chg = past_kalshi_mid_rows_for_chg.iloc[-1]
                    past_mid_for_chg = past_k_candle_chg.get('mid_price') # Assumes 'mid_price' column was added in load_kalshi_market_data
                    if pd.notna(past_mid_for_chg): features[col] = _mid_t_minus_1 - past_mid_for_chg
    
    if 'mid_price' in kalshi_market_history_df.columns: # Check if pre-calculated
        kalshi_hist_for_vol = kalshi_market_history_df[kalshi_market_history_df.index <= signal_ts_utc]['mid_price']
        for window in KALSHI_VOLATILITY_WINDOWS:
            vol_col = f'kalshi_mid_vol_{window}m'
            if vol_col in _feature_order:
                features[vol_col] = kalshi_hist_for_vol.tail(window).std() if len(kalshi_hist_for_vol) >= window else np.nan

    # --- Interaction and Relative Features ---
    if 'distance_to_strike' in _feature_order: features['distance_to_strike'] = current_btc_price_t_minus_1 - kalshi_strike_price
    
    current_atr_val = features.get(atr_col) # Get potentially NaN ATR
    if 'distance_to_strike_norm_atr' in _feature_order:
        if pd.notna(current_atr_val) and current_atr_val > 1e-6 and pd.notna(features['distance_to_strike']):
            features['distance_to_strike_norm_atr'] = features['distance_to_strike'] / current_atr_val
        else: features['distance_to_strike_norm_atr'] = np.nan # Keep NaN, impute later

    if 'kalshi_vs_btc_implied_spread' in _feature_order:
        if pd.notna(_mid_t_minus_1) and pd.notna(current_atr_val) and current_atr_val > 1e-6 and pd.notna(features['distance_to_strike']):
            btc_implied_value_offset = np.clip( (features['distance_to_strike'] / current_atr_val) * 15, -45, 45)
            btc_equiv_kalshi_price = 50 + btc_implied_value_offset
            features['kalshi_vs_btc_implied_spread'] = _mid_t_minus_1 - btc_equiv_kalshi_price
        else: features['kalshi_vs_btc_implied_spread'] = np.nan


    # --- Time Features ---
    if 'time_until_market_close_min' in _feature_order: features['time_until_market_close_min'] = (kalshi_market_close_dt_utc - decision_point_dt_utc).total_seconds() / 60
    if 'hour_of_day_utc' in _feature_order: features['hour_of_day_utc'] = decision_point_dt_utc.hour
    if 'day_of_week_utc' in _feature_order: features['day_of_week_utc'] = decision_point_dt_utc.weekday()
    if 'hour_of_day_edt' in _feature_order:
        try: features['hour_of_day_edt'] = decision_point_dt_utc.astimezone(timezone(timedelta(hours=-4))).hour
        except Exception: features['hour_of_day_edt'] = (decision_point_dt_utc.hour - 4 + 24) % 24

    # --- Final Imputation (Consistent with train.ipynb Cell 2) ---
    for f_name in _feature_order: 
        if pd.isna(features.get(f_name)): # Check if it's still NaN after attempted calculations
            # Apply same imputation logic as in train.ipynb Cell 2
            if 'btc_mom' in f_name or 'kalshi_mid_chg' in f_name: features[f_name] = 0.0
            elif 'btc_vol' in f_name or 'kalshi_mid_vol' in f_name: features[f_name] = 0.0 # Median was used in train, 0 is simpler here, or pass medians
            elif 'btc_sma' in f_name or 'btc_ema' in f_name: features[f_name] = features.get('btc_price_t_minus_1', 0.0) # Impute with price
            elif 'btc_price_vs_sma' in f_name or 'btc_price_vs_ema' in f_name: features[f_name] = 1.0
            elif 'btc_rsi' in f_name: features[f_name] = 50.0
            elif 'btc_atr' in f_name: features[f_name] = 0.0 # Median was used in train
            elif f_name == 'distance_to_strike_norm_atr': features[f_name] = 0.0
            elif f_name == 'kalshi_vs_btc_implied_spread': features[f_name] = 0.0
            elif f_name == 'kalshi_yes_bid': features[f_name] = 0.0
            elif f_name == 'kalshi_yes_ask': features[f_name] = 100.0
            elif f_name == 'kalshi_spread': features[f_name] = 100.0
            elif f_name == 'kalshi_mid_price': features[f_name] = 50.0
            elif 'kalshi_volume_t_minus_1' in f_name or 'kalshi_open_interest_t_minus_1' in f_name: features[f_name] = 0.0
            else: features[f_name] = 0.0 
            # logger.debug(f"Strategy ({_model_type_loaded}): Imputed NaN for feature '{f_name}' with {features[f_name]}.")
            
    return features.reindex(_feature_order) # Ensures all expected columns are present and ordered

# ... (calculate_model_prediction_proba and get_trade_decision remain the same as your last version) ...
def calculate_model_prediction_proba(feature_vector: pd.Series) -> float | None:
    if not all([_scaler, _model, _feature_order, _model_classes is not None]):
        logger.error(f"Strategy ({_model_type_loaded}): Model/scaler/features not fully loaded."); return None
    if feature_vector is None or feature_vector.empty:
        logger.warning(f"Strategy ({_model_type_loaded}): Empty feature vector for prediction."); return None
    try:
        # Ensure feature_vector is a DataFrame with columns in _feature_order before scaling
        feature_df = pd.DataFrame([feature_vector])[_feature_order] # Reorder and select
        
        if feature_df.isnull().values.any():
            nan_cols = feature_df.columns[feature_df.isnull().any()].tolist()
            logger.warning(f"Strategy ({_model_type_loaded}): NaNs in feature vector pre-scaling: {nan_cols}. Imputing specific or with 0.")
            for col in nan_cols: 
                # Imputation logic should ideally mirror the final imputation in generate_live_features if any NaNs slip through
                if 'rsi' in col: feature_df[col].fillna(50.0, inplace=True)
                elif col == 'kalshi_yes_ask': feature_df[col].fillna(100.0, inplace=True)
                elif col == 'kalshi_spread': feature_df[col].fillna(100.0, inplace=True)
                elif col == 'kalshi_mid_price': feature_df[col].fillna(50.0, inplace=True)
                elif 'btc_price_vs' in col : feature_df[col].fillna(1.0, inplace=True)
                else: feature_df[col].fillna(0, inplace=True) 
        
        scaled_features_array = _scaler.transform(feature_df)
        
        positive_class_idx = np.where(_model_classes == 1)[0]
        if not positive_class_idx.size: 
            logger.error(f"Strategy ({_model_type_loaded}): Positive class (1) not found in model.classes_ ({_model_classes})."); return None
        
        proba_yes = _model.predict_proba(scaled_features_array)[0, positive_class_idx[0]]
        return float(proba_yes)
    except Exception as e:
        # Check if the error is about feature count
        if "features" in str(e) and ("expected" in str(e) or "input" in str(e)):
             logger.error(f"Strategy ({_model_type_loaded}): Feature count mismatch during prediction. Model expects {_model.n_features_in_ if hasattr(_model, 'n_features_in_') else 'N/A'} features. Input shape: {scaled_features_array.shape if 'scaled_features_array' in locals() else 'N/A'}. Feature order length: {len(_feature_order) if _feature_order else 'N/A'}")
        logger.error(f"Strategy ({_model_type_loaded}): Error during probability prediction: {e}", exc_info=True)
        return None

def get_trade_decision(
    predicted_proba_yes: float | None, 
    current_kalshi_bid_at_decision_t: float | None, 
    current_kalshi_ask_at_decision_t: float | None  
):
    trade_action = None
    model_prob_for_chosen_side = None
    entry_price_for_chosen_side = None

    if predicted_proba_yes is None:
        return trade_action, model_prob_for_chosen_side, entry_price_for_chosen_side

    if pd.notna(current_kalshi_ask_at_decision_t) and 0 < current_kalshi_ask_at_decision_t < 100:
        implied_proba_yes_at_ask = current_kalshi_ask_at_decision_t / 100.0
        if predicted_proba_yes > MIN_MODEL_PROB_FOR_CONSIDERATION and \
           predicted_proba_yes > (implied_proba_yes_at_ask + EDGE_THRESHOLD_FOR_TRADE):
            trade_action = "BUY_YES"
            model_prob_for_chosen_side = predicted_proba_yes
            entry_price_for_chosen_side = current_kalshi_ask_at_decision_t

    if trade_action is None: 
        if pd.notna(current_kalshi_bid_at_decision_t) and 0 < current_kalshi_bid_at_decision_t < 100:
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