# notebooks/classifier_strategy.py
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import logging
import datetime as dt
from datetime import timezone, timedelta

import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")


# Import TA parameters from utils to ensure consistency
try:
    from . import utils as main_utils
    BTC_MOMENTUM_WINDOWS = main_utils.BTC_MOMENTUM_WINDOWS
    BTC_VOLATILITY_WINDOW = main_utils.BTC_VOLATILITY_WINDOW
    BTC_SMA_WINDOWS = main_utils.BTC_SMA_WINDOWS
    BTC_EMA_WINDOWS = main_utils.BTC_EMA_WINDOWS
    BTC_RSI_WINDOW = main_utils.BTC_RSI_WINDOW
    BTC_ATR_WINDOW = getattr(main_utils, 'BTC_ATR_WINDOW', 14)
    KALSHI_PRICE_CHANGE_WINDOWS = getattr(main_utils, 'KALSHI_PRICE_CHANGE_WINDOWS', [1, 3, 5, 10])
    KALSHI_VOLATILITY_WINDOWS = getattr(main_utils, 'KALSHI_VOLATILITY_WINDOWS', [5, 10])
    KALSHI_MAX_STALENESS_SECONDS_FOR_FEATURES = getattr(main_utils, 'KALSHI_MAX_STALENESS_SECONDS_FOR_FEATURES', 120)
    logger_strat_utils = logging.getLogger(__name__)
    # logger_strat_utils.debug("Strategy: Successfully imported TA parameters from main_utils.") # Can be noisy
except ImportError:
    logger_strat_utils = logging.getLogger(__name__)
    logger_strat_utils.warning("Strategy: Could not import TA parameters from main_utils. Using local defaults.")
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

# --- Strategy Thresholds ---
# These will be SET by the backtesting script (backtest_classifier_v1.py)
MIN_MODEL_PROB_FOR_CONSIDERATION = 0.85 # Default, to be overridden
EDGE_THRESHOLD_FOR_TRADE = 0.10      # Default, to be overridden

# --- NEW FILTERS (Defaults, can also be set/tuned in backtesting script if needed) ---
MAX_TIME_UNTIL_CLOSE_FOR_TRADE_MIN = 120 # Max 2 hours before close
MAX_KALSHI_SPREAD_FOR_TRADE_CENTS = 15   # Max 15 cents spread

_scaler = None; _model = None; _feature_order = None; _model_classes = None; _imputation_values = None
_model_type_loaded = "None"

def load_model_artifacts(artifacts_directory: Path, model_type_name: str):
    global _scaler, _model, _feature_order, _model_classes, _model_type_loaded, MODEL_ARTIFACTS_BASE_DIR, _imputation_values
    MODEL_ARTIFACTS_BASE_DIR = artifacts_directory
    logger.info(f"Strategy: Attempting to load '{model_type_name}' artifacts from {MODEL_ARTIFACTS_BASE_DIR.resolve()}")
    _model_type_loaded = model_type_name
    imputation_values_filename = None
    if _model_type_loaded == "logistic_regression":
        model_filename = "logistic_regression_btc_classifier_v1.joblib"
        scaler_filename = "feature_scaler_classifier_v1.joblib"
        feature_order_filename = "feature_columns_classifier_v1.json"
        imputation_values_filename = "imputation_values_classifier_v1.json"
    elif _model_type_loaded == "random_forest":
        model_filename = "CalibratedRandomForest_classifier_v2.joblib"
        scaler_filename = "feature_scaler_classifier_v2.joblib"
        feature_order_filename = "feature_columns_classifier_v2.json"
        imputation_values_filename = "imputation_values_classifier_v2.json"
    else:
        logger.error(f"Strategy: Unknown model_type_name '{_model_type_loaded}'."); return False

    model_path = MODEL_ARTIFACTS_BASE_DIR / model_filename
    scaler_path = MODEL_ARTIFACTS_BASE_DIR / scaler_filename
    feature_order_path = MODEL_ARTIFACTS_BASE_DIR / feature_order_filename
    imputation_values_path = MODEL_ARTIFACTS_BASE_DIR / imputation_values_filename
    logger.info(f"Strategy: Loading model='{model_filename}', scaler='{scaler_filename}', features='{feature_order_filename}', imputation_map='{imputation_values_filename}'")
    artifacts_loaded_successfully = True
    try:
        if scaler_path.exists(): _scaler = joblib.load(scaler_path); # logger.debug(f"Scaler '{scaler_filename}' loaded.")
        else: logger.error(f"Scaler '{scaler_filename}' not found: {scaler_path}"); artifacts_loaded_successfully = False
        if model_path.exists():
            _model = joblib.load(model_path); # logger.debug(f"Model '{model_filename}' loaded.")
            if hasattr(_model, 'classes_'): _model_classes = _model.classes_
            else: logger.warning(f"Model type {_model_type_loaded} lacks 'classes_'. Assuming [0, 1]."); _model_classes = np.array([0,1])
        else: logger.error(f"Model '{model_filename}' not found: {model_path}"); artifacts_loaded_successfully = False
        if feature_order_path.exists():
            with open(feature_order_path, 'r') as f: _feature_order = json.load(f)
            # logger.debug(f"Feature order '{feature_order_filename}' loaded ({len(_feature_order)} features).")
        else: logger.error(f"Feature order '{feature_order_filename}' not found: {feature_order_path}"); artifacts_loaded_successfully = False
        if imputation_values_path.exists():
            with open(imputation_values_path, 'r') as f: _imputation_values = json.load(f)
            # logger.debug(f"Imputation values map '{imputation_values_filename}' loaded ({len(_imputation_values if _imputation_values else 0)} values).")
        else:
            logger.error(f"Imputation values map '{imputation_values_filename}' not found: {imputation_values_path}")
            _imputation_values = {}
            artifacts_loaded_successfully = False
        if artifacts_loaded_successfully: logger.info(f"Strategy: All '{_model_type_loaded}' artifacts loaded successfully.")
        else: logger.error(f"Strategy: Artifact load failed for '{_model_type_loaded}'.")
    except Exception as e:
        logger.critical(f"Strategy: CRITICAL error loading '{_model_type_loaded}' artifacts: {e}", exc_info=True)
        _scaler, _model, _feature_order, _model_classes, _imputation_values = None, None, None, None, None
        return False
    return artifacts_loaded_successfully

def generate_live_features(
    btc_price_history_df: pd.DataFrame,
    kalshi_market_history_df: pd.DataFrame | None,
    kalshi_strike_price: float,
    decision_point_dt_utc: dt.datetime,
    kalshi_market_close_dt_utc: dt.datetime
) -> pd.Series | None:
    # This function remains unchanged from the version that correctly uses _imputation_values
    # (as provided in my response to "great, let's address these first: Inconsistent NaN Imputation (High Impact)")
    # Ensure that version is used here. For brevity, I'm not repeating the full function body.
    # Key is that it uses _feature_order and _imputation_values correctly.

    if _feature_order is None:
        logger.error(f"Strategy ({_model_type_loaded}): Feature order not loaded."); return None
    if _imputation_values is None: 
        logger.error(f"Strategy ({_model_type_loaded}): Imputation values map not loaded. Cannot reliably generate features."); return None

    features = pd.Series(index=_feature_order, dtype=float) 
    signal_dt_for_features_utc = (decision_point_dt_utc - timedelta(minutes=1)).replace(second=0, microsecond=0)
    signal_ts_utc = int(signal_dt_for_features_utc.timestamp())

    if btc_price_history_df.empty or signal_ts_utc not in btc_price_history_df.index:
        # logger.debug(f"Strategy ({_model_type_loaded}): BTC data insufficient for ts {signal_ts_utc}. Relying on imputation for BTC features.")
        for f_name in _feature_order:
            if f_name.startswith("btc_"): 
                 features[f_name] = _imputation_values.get(f_name) # Let imputation handle if key missing
    else:
        btc_row_at_signal = btc_price_history_df.loc[signal_ts_utc]
        current_btc_price_t_minus_1 = btc_row_at_signal.get('close')
        if pd.isna(current_btc_price_t_minus_1):
            pass # Will be imputed later
        else:
            features['btc_price_t_minus_1'] = current_btc_price_t_minus_1
        for window in BTC_MOMENTUM_WINDOWS:
            col = f'btc_mom_{window}m',
            if col in _feature_order: features[col] = btc_row_at_signal.get(col)
        if BTC_VOLATILITY_WINDOW > 0 and f'btc_vol_{BTC_VOLATILITY_WINDOW}m' in _feature_order:
            features[f'btc_vol_{BTC_VOLATILITY_WINDOW}m'] = btc_row_at_signal.get(f'btc_vol_{BTC_VOLATILITY_WINDOW}m')
        for window in BTC_SMA_WINDOWS:
            sma_col = f'btc_sma_{window}m'; vs_sma_col = f'btc_price_vs_sma_{window}m'
            if sma_col in _feature_order:
                sma_val = btc_row_at_signal.get(sma_col); features[sma_col] = sma_val
                if vs_sma_col in _feature_order:
                    price_for_vs = features.get('btc_price_t_minus_1', np.nan)
                    if pd.notna(sma_val) and sma_val != 0 and pd.notna(price_for_vs): features[vs_sma_col] = price_for_vs / sma_val
                    else: features[vs_sma_col] = np.nan
        for window in BTC_EMA_WINDOWS:
            ema_col = f'btc_ema_{window}m'; vs_ema_col = f'btc_price_vs_ema_{window}m'
            if ema_col in _feature_order:
                ema_val = btc_row_at_signal.get(ema_col); features[ema_col] = ema_val
                if vs_ema_col in _feature_order:
                    price_for_vs = features.get('btc_price_t_minus_1', np.nan)
                    if pd.notna(ema_val) and ema_val != 0 and pd.notna(price_for_vs): features[vs_ema_col] = price_for_vs / ema_val
                    else: features[vs_ema_col] = np.nan
        if BTC_RSI_WINDOW > 0 and 'btc_rsi' in _feature_order: features['btc_rsi'] = btc_row_at_signal.get('btc_rsi')
        atr_col_name = f'btc_atr_{BTC_ATR_WINDOW}'
        if BTC_ATR_WINDOW > 0 and atr_col_name in _feature_order: features[atr_col_name] = btc_row_at_signal.get(atr_col_name)

    _yes_bid_t_minus_1, _yes_ask_t_minus_1, _mid_t_minus_1 = np.nan, np.nan, np.nan
    _vol_t_minus_1, _oi_t_minus_1 = np.nan, np.nan
    kalshi_row_at_signal = None
    if kalshi_market_history_df is not None and not kalshi_market_history_df.empty:
        relevant_kalshi_rows_t_minus_1 = kalshi_market_history_df[kalshi_market_history_df.index <= signal_ts_utc]
        if not relevant_kalshi_rows_t_minus_1.empty:
            latest_kalshi_row_t_minus_1 = relevant_kalshi_rows_t_minus_1.iloc[-1]
            latest_kalshi_ts_t_minus_1 = latest_kalshi_row_t_minus_1.name
            if (signal_ts_utc - latest_kalshi_ts_t_minus_1) <= KALSHI_MAX_STALENESS_SECONDS_FOR_FEATURES:
                kalshi_row_at_signal = latest_kalshi_row_t_minus_1
    if kalshi_row_at_signal is not None:
        _yes_bid_val = kalshi_row_at_signal.get('yes_bid_close_cents'); _yes_ask_val = kalshi_row_at_signal.get('yes_ask_close_cents')
        if pd.notna(_yes_bid_val) and pd.notna(_yes_ask_val) and 0 <= _yes_bid_val <= 100 and 0 <= _yes_ask_val <= 100 and _yes_bid_val <= _yes_ask_val:
            _yes_bid_t_minus_1 = _yes_bid_val; _yes_ask_t_minus_1 = _yes_ask_val
            _mid_t_minus_1 = (_yes_bid_t_minus_1 + _yes_ask_t_minus_1) / 2.0
        else: _yes_bid_t_minus_1 = np.nan if pd.isna(_yes_bid_val) else _yes_bid_val; _yes_ask_t_minus_1 = np.nan if pd.isna(_yes_ask_val) else _yes_ask_val; _mid_t_minus_1 = np.nan
        _vol_t_minus_1 = kalshi_row_at_signal.get('volume'); _oi_t_minus_1 = kalshi_row_at_signal.get('open_interest')
    if 'kalshi_yes_bid' in _feature_order: features['kalshi_yes_bid'] = _yes_bid_t_minus_1
    if 'kalshi_yes_ask' in _feature_order: features['kalshi_yes_ask'] = _yes_ask_t_minus_1
    if 'kalshi_mid_price' in _feature_order: features['kalshi_mid_price'] = _mid_t_minus_1
    if 'kalshi_spread' in _feature_order:
        if pd.notna(_yes_ask_t_minus_1) and pd.notna(_yes_bid_t_minus_1): features['kalshi_spread'] = _yes_ask_t_minus_1 - _yes_bid_t_minus_1
        else: features['kalshi_spread'] = np.nan
    if 'kalshi_volume_t_minus_1' in _feature_order: features['kalshi_volume_t_minus_1'] = _vol_t_minus_1
    if 'kalshi_open_interest_t_minus_1' in _feature_order: features['kalshi_open_interest_t_minus_1'] = _oi_t_minus_1
    for w_min in KALSHI_PRICE_CHANGE_WINDOWS:
        col_name = f'kalshi_mid_chg_{w_min}m'
        if col_name in _feature_order:
            val_to_set = np.nan
            if kalshi_market_history_df is not None and not kalshi_market_history_df.empty and pd.notna(_mid_t_minus_1):
                past_ts_for_chg_calc = signal_ts_utc - (w_min * 60)
                past_kalshi_mid_rows_for_chg = kalshi_market_history_df[kalshi_market_history_df.index <= past_ts_for_chg_calc]
                if not past_kalshi_mid_rows_for_chg.empty:
                    past_k_candle_chg = past_kalshi_mid_rows_for_chg.iloc[-1]
                    past_mid_for_chg = past_k_candle_chg.get('mid_price')
                    if pd.notna(past_mid_for_chg): val_to_set = _mid_t_minus_1 - past_mid_for_chg
            features[col_name] = val_to_set
    if kalshi_market_history_df is not None and 'mid_price' in kalshi_market_history_df.columns:
        kalshi_hist_for_vol = kalshi_market_history_df[kalshi_market_history_df.index <= signal_ts_utc]['mid_price'].dropna()
        for window in KALSHI_VOLATILITY_WINDOWS:
            vol_col_name = f'kalshi_mid_vol_{window}m'
            if vol_col_name in _feature_order:
                if len(kalshi_hist_for_vol) >= window: features[vol_col_name] = kalshi_hist_for_vol.tail(window).std()
                else: features[vol_col_name] = np.nan
    btc_price_for_derived = features.get('btc_price_t_minus_1', np.nan)
    if 'distance_to_strike' in _feature_order:
        if pd.notna(btc_price_for_derived): features['distance_to_strike'] = btc_price_for_derived - kalshi_strike_price
        else: features['distance_to_strike'] = np.nan
    atr_val_for_derived = features.get(f'btc_atr_{BTC_ATR_WINDOW}', np.nan)
    dist_strike_val_for_derived = features.get('distance_to_strike', np.nan)
    if 'distance_to_strike_norm_atr' in _feature_order:
        if pd.notna(atr_val_for_derived) and atr_val_for_derived > 1e-6 and pd.notna(dist_strike_val_for_derived): features['distance_to_strike_norm_atr'] = dist_strike_val_for_derived / atr_val_for_derived
        else: features['distance_to_strike_norm_atr'] = np.nan
    mid_price_for_derived = features.get('kalshi_mid_price', np.nan)
    if 'kalshi_vs_btc_implied_spread' in _feature_order:
        if pd.notna(mid_price_for_derived) and pd.notna(atr_val_for_derived) and atr_val_for_derived > 1e-6 and pd.notna(dist_strike_val_for_derived):
            btc_implied_value_offset = np.clip( (dist_strike_val_for_derived / atr_val_for_derived) * 15, -45, 45)
            btc_equiv_kalshi_price = 50 + btc_implied_value_offset
            features['kalshi_vs_btc_implied_spread'] = mid_price_for_derived - btc_equiv_kalshi_price
        else: features['kalshi_vs_btc_implied_spread'] = np.nan
    if 'time_until_market_close_min' in _feature_order: features['time_until_market_close_min'] = (kalshi_market_close_dt_utc - decision_point_dt_utc).total_seconds() / 60
    if 'hour_of_day_utc' in _feature_order: features['hour_of_day_utc'] = decision_point_dt_utc.hour
    if 'day_of_week_utc' in _feature_order: features['day_of_week_utc'] = decision_point_dt_utc.weekday()
    if 'hour_of_day_edt' in _feature_order:
        try: features['hour_of_day_edt'] = decision_point_dt_utc.astimezone(timezone(timedelta(hours=-4))).hour
        except Exception: features['hour_of_day_edt'] = (decision_point_dt_utc.hour - 4 + 24) % 24
    if pd.isna(features.get('kalshi_yes_bid')): features['kalshi_yes_bid'] = _imputation_values.get('kalshi_yes_bid')
    if pd.isna(features.get('kalshi_yes_ask')): features['kalshi_yes_ask'] = _imputation_values.get('kalshi_yes_ask')
    imputed_bid_for_derived = features.get('kalshi_yes_bid'); imputed_ask_for_derived = features.get('kalshi_yes_ask')
    if 'kalshi_spread' in _feature_order and pd.isna(features.get('kalshi_spread')):
        if pd.notna(imputed_ask_for_derived) and pd.notna(imputed_bid_for_derived): features['kalshi_spread'] = imputed_ask_for_derived - imputed_bid_for_derived
    if 'kalshi_mid_price' in _feature_order and pd.isna(features.get('kalshi_mid_price')):
        if pd.notna(imputed_ask_for_derived) and pd.notna(imputed_bid_for_derived): features['kalshi_mid_price'] = (imputed_bid_for_derived + imputed_ask_for_derived) / 2.0
    for f_name in _feature_order:
        if pd.isna(features.get(f_name)):
            if f_name in _imputation_values: features[f_name] = _imputation_values[f_name]
            else: logger.warning(f"Strategy ({_model_type_loaded}): Feature '{f_name}' not in imputation map. Filling with 0.0."); features[f_name] = 0.0
    if features.isnull().any():
        nan_cols_final = features[features.isnull()].index.tolist()
        logger.error(f"Strategy ({_model_type_loaded}): NaNs still present AFTER imputation: {nan_cols_final}. Defaulting to 0.")
        features.fillna(0.0, inplace=True)
    return features.reindex(_feature_order)

def calculate_model_prediction_proba(feature_vector: pd.Series) -> float | None:
    # This function remains unchanged from the version that correctly uses _imputation_values
    if not all([_scaler, _model, _feature_order, _model_classes is not None]):
        logger.error(f"Strategy ({_model_type_loaded}): Model/scaler/features not fully loaded."); return None
    if feature_vector is None or feature_vector.empty:
        logger.warning(f"Strategy ({_model_type_loaded}): Empty feature vector for prediction."); return None
    try:
        feature_df = pd.DataFrame([feature_vector])[_feature_order]
        if feature_df.isnull().values.any():
            nan_cols = feature_df.columns[feature_df.isnull().any()].tolist()
            logger.error(f"Strategy ({_model_type_loaded}): NaNs found in feature vector JUST BEFORE SCALING: {nan_cols}. THIS IS UNEXPECTED. Imputing with _imputation_values or 0 as emergency fallback.")
            for col_nan in nan_cols: feature_df[col_nan].fillna(_imputation_values.get(col_nan, 0.0), inplace=True)
        scaled_features_array = _scaler.transform(feature_df)
        positive_class_idx = np.where(_model_classes == 1)[0]
        if not positive_class_idx.size:
            logger.error(f"Strategy ({_model_type_loaded}): Positive class (1) not found in model.classes_ ({_model_classes})."); return None
        proba_yes = _model.predict_proba(scaled_features_array)[0, positive_class_idx[0]]
        return float(proba_yes)
    except Exception as e:
        n_features_expected = _model.n_features_in_ if hasattr(_model, 'n_features_in_') else 'N/A'
        logger.error(f"Strategy ({_model_type_loaded}): Error during probability prediction. Model expects {n_features_expected} features. Input shape: {feature_df.shape if 'feature_df' in locals() else 'N/A'}. Error: {e}", exc_info=True)
        return None

def get_trade_decision(
    predicted_proba_yes: float | None,
    current_kalshi_bid_at_decision_t: float | None, # This is Kalshi YES bid
    current_kalshi_ask_at_decision_t: float | None, # This is Kalshi YES ask
    # NEWLY ADDED from feature_vector for filtering:
    time_until_market_close_min: float | None,
    current_kalshi_spread_cents: float | None # Calculated as ask - bid for YES contract
):
    trade_action = None
    model_prob_for_chosen_side = None
    entry_price_for_chosen_side = None
    rejection_reason = "NONE" # For logging

    if predicted_proba_yes is None:
        rejection_reason = "MODEL_PRED_FAILED"
        return trade_action, model_prob_for_chosen_side, entry_price_for_chosen_side, rejection_reason

    # --- Apply new pre-trade filters ---
    if time_until_market_close_min is not None and time_until_market_close_min >= MAX_TIME_UNTIL_CLOSE_FOR_TRADE_MIN:
        rejection_reason = f"FILTER_TIME_CLOSE (Actual: {time_until_market_close_min:.0f} >= {MAX_TIME_UNTIL_CLOSE_FOR_TRADE_MIN})"
        logger.debug(f"Trade rejected for {rejection_reason}")
        return trade_action, model_prob_for_chosen_side, entry_price_for_chosen_side, rejection_reason
    
    if current_kalshi_spread_cents is not None and current_kalshi_spread_cents > MAX_KALSHI_SPREAD_FOR_TRADE_CENTS:
        rejection_reason = f"FILTER_SPREAD (Actual: {current_kalshi_spread_cents:.0f} > {MAX_KALSHI_SPREAD_FOR_TRADE_CENTS})"
        logger.debug(f"Trade rejected for {rejection_reason}")
        return trade_action, model_prob_for_chosen_side, entry_price_for_chosen_side, rejection_reason
    
    # --- Original Trade Logic (BUY_YES) ---
    if pd.notna(current_kalshi_ask_at_decision_t) and 0 < current_kalshi_ask_at_decision_t < 100:
        implied_proba_yes_at_ask = current_kalshi_ask_at_decision_t / 100.0
        if predicted_proba_yes >= MIN_MODEL_PROB_FOR_CONSIDERATION and \
           predicted_proba_yes > (implied_proba_yes_at_ask + EDGE_THRESHOLD_FOR_TRADE):
            trade_action = "BUY_YES"
            model_prob_for_chosen_side = predicted_proba_yes
            entry_price_for_chosen_side = current_kalshi_ask_at_decision_t
            # No rejection_reason needed if trade is considered
        elif predicted_proba_yes < MIN_MODEL_PROB_FOR_CONSIDERATION:
            rejection_reason = f"NO_TRADE_MIN_PROB_YES (P(Y):{predicted_proba_yes:.2f} < {MIN_MODEL_PROB_FOR_CONSIDERATION})"
        else: # Edge not met
            edge_val = predicted_proba_yes - implied_proba_yes_at_ask
            rejection_reason = f"NO_TRADE_EDGE_YES (Edge:{edge_val:.2f} < {EDGE_THRESHOLD_FOR_TRADE})"


    # --- Original Trade Logic (BUY_NO) ---
    if trade_action is None: # Only consider BUY_NO if BUY_YES was not chosen
        if pd.notna(current_kalshi_bid_at_decision_t) and 0 < current_kalshi_bid_at_decision_t < 100:
            predicted_proba_no = 1.0 - predicted_proba_yes
            cost_of_no_contract_cents = 100.0 - current_kalshi_bid_at_decision_t # This is the entry price for BUY_NO
            
            if 0 < cost_of_no_contract_cents < 100: # Ensure valid cost for NO contract
                implied_proba_no_at_bid = cost_of_no_contract_cents / 100.0 # Implied prob of NO winning if you buy NO at this price
                
                if predicted_proba_no >= MIN_MODEL_PROB_FOR_CONSIDERATION and \
                   predicted_proba_no > (implied_proba_no_at_bid + EDGE_THRESHOLD_FOR_TRADE):
                    trade_action = "BUY_NO"
                    model_prob_for_chosen_side = predicted_proba_no
                    entry_price_for_chosen_side = cost_of_no_contract_cents
                    rejection_reason = "NONE" # Reset if BUY_NO is chosen
                elif rejection_reason == "NONE" or "MIN_PROB_YES" in rejection_reason or "EDGE_YES" in rejection_reason : # Only update if no prior filter hit or was due to YES side
                    if predicted_proba_no < MIN_MODEL_PROB_FOR_CONSIDERATION:
                        rejection_reason = f"NO_TRADE_MIN_PROB_NO (P(N):{predicted_proba_no:.2f} < {MIN_MODEL_PROB_FOR_CONSIDERATION})"
                    else: # Edge not met for NO
                        edge_val_no = predicted_proba_no - implied_proba_no_at_bid
                        rejection_reason = f"NO_TRADE_EDGE_NO (Edge:{edge_val_no:.2f} < {EDGE_THRESHOLD_FOR_TRADE})"
            elif rejection_reason == "NONE": # If cost_of_no_contract_cents is invalid
                rejection_reason = "INVALID_NO_CONTRACT_PRICE"

    if trade_action is None and rejection_reason == "NONE": # If no trade action and no specific rejection, it's general threshold not met
        rejection_reason = "NO_TRADE_THRESHOLDS_NOT_MET"

    return trade_action, model_prob_for_chosen_side, entry_price_for_chosen_side, rejection_reason