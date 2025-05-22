# notebooks/backtest_classifier_v1.py
import pandas as pd
import os
from pathlib import Path
import datetime as dt
from datetime import timezone, timedelta
import logging
import numpy as np
import json
import warnings 
from sklearn.utils.validation import DataConversionWarning 

warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn.utils.validation')

try:
    script_path = Path(os.path.abspath(__file__)) 
    BASE_PROJECT_DIR = script_path.parent.parent 
    if not (BASE_PROJECT_DIR / "notebooks").exists(): 
        BASE_PROJECT_DIR = Path.cwd()
        if not (BASE_PROJECT_DIR / "notebooks").exists() and (BASE_PROJECT_DIR.parent / "notebooks").exists():
            BASE_PROJECT_DIR = BASE_PROJECT_DIR.parent
except Exception as path_e:
    BASE_PROJECT_DIR = Path.cwd()

NOTEBOOKS_DIR = BASE_PROJECT_DIR / "notebooks"
KALSHI_DATA_ROOT_DIR = NOTEBOOKS_DIR / "kalshi_data"
BINANCE_FLAT_DATA_DIR = NOTEBOOKS_DIR / "binance_data" 
MODEL_ARTIFACTS_ROOT_DIR = NOTEBOOKS_DIR / "trained_models" 

MODEL_TYPE_TO_RUN = "random_forest" # Ensure this is set to test the new RF model

if MODEL_TYPE_TO_RUN == "logistic_regression": 
    CLASSIFIER_MODEL_SUBDIR_NAME = "logreg"
    # If you have a v2 LogReg, update its name in classifier_strategy.py
elif MODEL_TYPE_TO_RUN == "random_forest": 
    CLASSIFIER_MODEL_SUBDIR_NAME = "rf" # This directory should contain CalibratedRandomForest_classifier_v2.joblib
else: 
    raise ValueError(f"Unsupported MODEL_TYPE_TO_RUN: {MODEL_TYPE_TO_RUN}")

MODEL_ARTIFACTS_SPECIFIC_DIR = MODEL_ARTIFACTS_ROOT_DIR / CLASSIFIER_MODEL_SUBDIR_NAME
LOGS_BASE_DIR = NOTEBOOKS_DIR / "logs" 
BACKTEST_LOGS_DIR = LOGS_BASE_DIR / f"backtest_{MODEL_TYPE_TO_RUN}_calibrated_v2_15m_offset" # More descriptive
BACKTEST_LOGS_DIR.mkdir(parents=True, exist_ok=True)
run_timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')

log_formatter_console = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s')
console_handler_main = logging.StreamHandler()
console_handler_main.setFormatter(log_formatter_console)
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.root.addHandler(console_handler_main)
logging.root.setLevel(logging.INFO) 

logger = logging.getLogger(f"Backtest_{MODEL_TYPE_TO_RUN}_CalV2_15m") 
logger.info(f"BASE_PROJECT_DIR: {BASE_PROJECT_DIR.resolve()}")
logger.info(f"Attempting to load {MODEL_TYPE_TO_RUN} model artifacts from: {MODEL_ARTIFACTS_SPECIFIC_DIR.resolve()}")
logger.info(f"Backtest session logs will be saved under: {BACKTEST_LOGS_DIR.resolve()}")

import utils 
import sizing 
import classifier_strategy as strategy 

# Pass the specific model_type_name to load_model_artifacts
# This ensures classifier_strategy.py knows which .joblib file to look for.
if not strategy.load_model_artifacts(MODEL_ARTIFACTS_SPECIFIC_DIR, model_type_name=MODEL_TYPE_TO_RUN):
    logger.critical(f"CRITICAL: Failed to load '{MODEL_TYPE_TO_RUN}' model artifacts. Aborting.")
    exit()
else:
    logger.info(f"Successfully loaded '{strategy._model_type_loaded}' model and artifacts for backtesting.")


try:
    if not KALSHI_DATA_ROOT_DIR.exists(): raise FileNotFoundError(f"Kalshi data root missing: {KALSHI_DATA_ROOT_DIR.resolve()}")
    outcomes_files = sorted(KALSHI_DATA_ROOT_DIR.glob("kalshi_btc_hourly_NTM_filtered_market_outcomes_*.csv"), key=os.path.getctime, reverse=True)
    if not outcomes_files: raise FileNotFoundError(f"No NTM outcomes CSV in {KALSHI_DATA_ROOT_DIR.resolve()}")
    MARKET_OUTCOMES_CSV_PATH = outcomes_files[0]
    logger.info(f"Backtest: Using market outcomes from {MARKET_OUTCOMES_CSV_PATH.resolve()}")
except FileNotFoundError as e: logger.critical(f"Backtest CRITICAL - {e}. Outcomes file required."); exit()
except Exception as e: logger.critical(f"Backtest CRITICAL - Error finding outcomes CSV: {e}", exc_info=True); exit()

BACKTEST_START_DATE_STR = "2025-05-12"; BACKTEST_END_DATE_STR = "2025-05-12" 
DECISION_INTERVAL_MINUTES = 1 
TRADE_DECISION_OFFSET_MINUTES_FROM_MARKET_CLOSE = 15 # MUST MATCH feature_engineering.ipynb
KALSHI_MAX_STALENESS_SECONDS_FOR_EXECUTION_PRICES = 120 
INITIAL_CAPITAL_USD = 500.0

sizing.KELLY_FRACTION = 0.05 # More conservative Kelly
sizing.MAX_CAPITAL_ALLOCATION_PERCENTAGE_OF_TOTAL = 0.05 
sizing.MAX_CONTRACTS_PER_TRADE = 50 

# Strategy parameters for the Calibrated Random Forest (v2)
# Adjust these based on the new probability distribution.
# Mean P(YES) is ~0.63. Max is ~0.93.
# Try slightly more relaxed thresholds to see if YES trades appear.
if MODEL_TYPE_TO_RUN == "random_forest":
    strategy.MIN_MODEL_PROB_FOR_CONSIDERATION = 0.65  # Example: Lowered slightly
    strategy.EDGE_THRESHOLD_FOR_TRADE = 0.05        # Example: Lowered slightly
elif MODEL_TYPE_TO_RUN == "logistic_regression":
    strategy.MIN_MODEL_PROB_FOR_CONSIDERATION = 0.60 
    strategy.EDGE_THRESHOLD_FOR_TRADE = 0.10      

logger.info(f"Strategy thresholds set for {MODEL_TYPE_TO_RUN}: MinProb={strategy.MIN_MODEL_PROB_FOR_CONSIDERATION}, Edge={strategy.EDGE_THRESHOLD_FOR_TRADE}")
logger.info(f"Sizing params: KellyFrac={sizing.KELLY_FRACTION}, MaxAllocPct={sizing.MAX_CAPITAL_ALLOCATION_PERCENTAGE_OF_TOTAL}, MaxContracts={sizing.MAX_CONTRACTS_PER_TRADE}")

detailed_log_counter_per_market = {} # Reset for each run of the script

def run_hourly_session_backtest(session_markets_df, binance_session_features_df, global_config_state):
    # ... (The rest of this function remains IDENTICAL to the one in your previous "give me full files only" request)
    # ... (It already includes the detailed feature cross-check logging and decision_eval_log saving)
    session_key = utils.get_session_key_from_market_row(session_markets_df.iloc[0])
    logger.info(f"--- Starting Backtest for Session: {session_key} ---")
    session_trades_log, session_pnl_cents, session_contracts_traded = [], 0, 0
    decision_eval_log = [] 
    
    global detailed_log_counter_per_market 

    for idx, market_row in session_markets_df.iterrows():
        market_ticker, kalshi_strike_price = market_row['market_ticker'], market_row['kalshi_strike_price']
        market_open_dt_utc, market_close_dt_utc = market_row['market_open_time_iso'], market_row['market_close_time_iso']
        actual_market_result = market_row['result']
        
        if market_ticker not in detailed_log_counter_per_market: # Initialize if not present (e.g. script re-run without full restart)
            detailed_log_counter_per_market[market_ticker] = 0
        else:
            detailed_log_counter_per_market[market_ticker] = 0 # Reset for each new market processing within a session run

        parsed_ticker_info = utils.parse_kalshi_ticker_info(market_ticker)
        if not parsed_ticker_info: logger.warning(f"    Could not parse ticker {market_ticker}. Skipping."); continue
        
        df_kalshi_market_minute_data = utils.load_kalshi_market_minute_data(
            market_ticker, parsed_ticker_info['date_str'], parsed_ticker_info['hour_str_EDT']
        )
        if df_kalshi_market_minute_data is None: df_kalshi_market_minute_data = pd.DataFrame()

        current_decision_dt_utc = market_open_dt_utc.replace(second=0, microsecond=0)
        latest_permissible_decision_dt_utc = (market_close_dt_utc - 
            timedelta(minutes=global_config_state['trade_decision_offset_minutes_from_market_close'])).replace(second=0, microsecond=0)

        while current_decision_dt_utc <= latest_permissible_decision_dt_utc:
            signal_dt_for_features_utc = (current_decision_dt_utc - timedelta(minutes=1)).replace(second=0, microsecond=0)
            signal_ts_utc_for_features = int(signal_dt_for_features_utc.timestamp())
            
            if signal_ts_utc_for_features not in binance_session_features_df.index:
                current_decision_dt_utc += timedelta(minutes=global_config_state['decision_interval_minutes']); continue
            
            btc_history_for_features = binance_session_features_df[binance_session_features_df.index <= signal_ts_utc_for_features].copy()
            if btc_history_for_features.empty:
                current_decision_dt_utc += timedelta(minutes=global_config_state['decision_interval_minutes']); continue
                
            kalshi_prices_at_decision_t = utils.get_kalshi_prices_at_decision(
                df_kalshi_market_minute_data, 
                int(current_decision_dt_utc.timestamp()), 
                global_config_state['kalshi_max_staleness_seconds_for_execution']
            )
            current_yes_bid_at_t, current_yes_ask_at_t = (kalshi_prices_at_decision_t.get('yes_bid'), 
                                                          kalshi_prices_at_decision_t.get('yes_ask')) if kalshi_prices_at_decision_t else (None, None)

            feature_vector_series = strategy.generate_live_features(
                btc_price_history_df=btc_history_for_features, 
                kalshi_market_history_df=df_kalshi_market_minute_data, 
                kalshi_strike_price=kalshi_strike_price, 
                decision_point_dt_utc=current_decision_dt_utc, 
                kalshi_market_close_dt_utc=market_close_dt_utc
            )
            
            predicted_proba_yes = None
            if feature_vector_series is not None:
                predicted_proba_yes = strategy.calculate_model_prediction_proba(feature_vector_series)

            log_this_iteration_detail = False
            if detailed_log_counter_per_market.get(market_ticker, 0) < 3 or \
               (predicted_proba_yes is not None and (predicted_proba_yes < 0.1 or predicted_proba_yes > 0.9)) or \
               feature_vector_series is None:
                log_this_iteration_detail = True

            if log_this_iteration_detail:
                logger.debug(f"--- Detailed Feature Check @ {current_decision_dt_utc.isoformat()} for {market_ticker} (Strike: {kalshi_strike_price}) ---")
                logger.debug(f"  Signal datetime for t-1 features: {signal_dt_for_features_utc.isoformat()} (ts: {signal_ts_utc_for_features})")
                if not btc_history_for_features.empty and signal_ts_utc_for_features in btc_history_for_features.index:
                    raw_btc_t_minus_1_row = btc_history_for_features.loc[signal_ts_utc_for_features]
                    logger.debug(f"  Raw BTC @ t-1 ({signal_ts_utc_for_features}): Open={raw_btc_t_minus_1_row.get('open')}, Close={raw_btc_t_minus_1_row.get('close')}")
                else:
                    logger.debug(f"  Raw BTC @ t-1 ({signal_ts_utc_for_features}): Data point not found.")
                kalshi_t_minus_1_row = None
                if df_kalshi_market_minute_data is not None and not df_kalshi_market_minute_data.empty:
                    relevant_kalshi_rows = df_kalshi_market_minute_data[df_kalshi_market_minute_data.index <= signal_ts_utc_for_features]
                    if not relevant_kalshi_rows.empty:
                        kalshi_t_minus_1_row = relevant_kalshi_rows.iloc[-1]
                        time_diff_staleness = signal_ts_utc_for_features - kalshi_t_minus_1_row.name
                        is_stale_for_features = time_diff_staleness > strategy.KALSHI_MAX_STALENESS_SECONDS_FOR_FEATURES
                        logger.debug(f"  Raw Kalshi @ t-1 (ts: {kalshi_t_minus_1_row.name}, stale: {is_stale_for_features}, diff: {time_diff_staleness}s): "
                                     f"YesBid={kalshi_t_minus_1_row.get('yes_bid_close_cents')}, YesAsk={kalshi_t_minus_1_row.get('yes_ask_close_cents')}")
                    else: logger.debug(f"  Raw Kalshi @ t-1: No data at or before {signal_ts_utc_for_features}.")
                else: logger.debug(f"  Raw Kalshi @ t-1: df_kalshi_market_minute_data is None or empty.")
                if feature_vector_series is not None:
                    logger.debug(f"  Generated Feature Subset (Key Features):")
                    key_feat_subset = {k: feature_vector_series.get(k) for k in ['btc_price_t_minus_1', 'distance_to_strike', 'kalshi_mid_price', 'time_until_market_close_min', 'btc_atr_14', 'distance_to_strike_norm_atr', 'kalshi_vs_btc_implied_spread'] if k in feature_vector_series}
                    logger.debug(f"    {key_feat_subset}")
                    if predicted_proba_yes is not None: logger.debug(f"  Model P(YES) from these features: {predicted_proba_yes:.4f}")
                else: logger.debug(f"  Feature vector generation FAILED.")
                detailed_log_counter_per_market[market_ticker] = detailed_log_counter_per_market.get(market_ticker,0) + 1
            
            log_entry = {
                "decision_dt_utc": current_decision_dt_utc.isoformat(), "market_ticker": market_ticker,
                "predicted_proba_yes": None, "kalshi_yes_bid_at_t": current_yes_bid_at_t,
                "kalshi_yes_ask_at_t": current_yes_ask_at_t, "implied_proba_yes_at_ask": None,
                "edge_for_yes": None, "predicted_proba_no": None,
                "implied_proba_no_at_bid": None, "edge_for_no": None,
                "considered_action": "FEATURE_GEN_FAILED" if feature_vector_series is None else ("MODEL_PRED_FAILED" if predicted_proba_yes is None else "NONE")
            }

            if feature_vector_series is None:
                decision_eval_log.append(log_entry)
                current_decision_dt_utc += timedelta(minutes=global_config_state['decision_interval_minutes']); continue
            
            if predicted_proba_yes is not None:
                log_entry["predicted_proba_yes"] = round(predicted_proba_yes, 4)
                log_entry["predicted_proba_no"] = round(1.0 - predicted_proba_yes, 4)
                if pd.notna(current_yes_ask_at_t) and 0 < current_yes_ask_at_t < 100:
                    log_entry["implied_proba_yes_at_ask"] = round(current_yes_ask_at_t / 100.0, 4)
                    log_entry["edge_for_yes"] = round(predicted_proba_yes - log_entry["implied_proba_yes_at_ask"], 4)
                if pd.notna(current_yes_bid_at_t) and 0 < current_yes_bid_at_t < 100:
                    cost_of_no_cents = 100.0 - current_yes_bid_at_t
                    if 0 < cost_of_no_cents < 100:
                        log_entry["implied_proba_no_at_bid"] = round(cost_of_no_cents / 100.0, 4)
                        if log_entry["predicted_proba_no"] is not None:
                             log_entry["edge_for_no"] = round(log_entry["predicted_proba_no"] - log_entry["implied_proba_no_at_bid"], 4)
            else: 
                decision_eval_log.append(log_entry)
                current_decision_dt_utc += timedelta(minutes=global_config_state['decision_interval_minutes']); continue
            
            trade_action, model_prob_chosen_side, entry_price_chosen_side = strategy.get_trade_decision(
                predicted_proba_yes, current_yes_bid_at_t, current_yes_ask_at_t
            )
            log_entry["considered_action"] = trade_action if trade_action else "NO_TRADE_THRESHOLD_NOT_MET"

            num_contracts, pnl_this_trade_cents = 0, np.nan
            if trade_action and pd.notna(model_prob_chosen_side) and pd.notna(entry_price_chosen_side):
                num_contracts = sizing.calculate_kelly_position_size(
                    model_prob_win=model_prob_chosen_side, entry_price_cents=entry_price_chosen_side,
                    available_capital_usd=global_config_state['current_capital_usd']
                )
                if num_contracts > 0:
                    pnl_per_contract = 0
                    if actual_market_result.lower() == 'yes':
                        pnl_per_contract = (100 - entry_price_chosen_side) if trade_action == "BUY_YES" else (-entry_price_chosen_side)
                    elif actual_market_result.lower() == 'no':
                        pnl_per_contract = (-entry_price_chosen_side) if trade_action == "BUY_YES" else (100 - entry_price_chosen_side)
                    pnl_this_trade_cents = pnl_per_contract * num_contracts
                    global_config_state['current_capital_usd'] += (pnl_this_trade_cents / 100.0) 
                    session_pnl_cents += pnl_this_trade_cents; session_contracts_traded += num_contracts
                    logger.info(f"TRADE: {market_ticker} @ {current_decision_dt_utc.strftime('%H:%M')} | P(YES):{predicted_proba_yes:.3f} "
                               f"P({trade_action.split('_')[1]}):{model_prob_chosen_side:.3f} | {trade_action} x{num_contracts}@{entry_price_chosen_side:.0f}c "
                               f"(Exec B:{current_yes_bid_at_t if pd.notna(current_yes_bid_at_t) else 'N/A'}, Exec A:{current_yes_ask_at_t if pd.notna(current_yes_ask_at_t) else 'N/A'}) | "
                               f"Outcome:{actual_market_result.upper()} | PNL:{pnl_this_trade_cents:.0f}c | Cap:${global_config_state['current_capital_usd']:.2f}")
                else: 
                    log_entry["considered_action"] = "NO_TRADE_SIZE_ZERO" 
            
            if not trade_action or num_contracts == 0 :
                if log_entry["considered_action"] not in ["FEATURE_GEN_FAILED", "MODEL_PRED_FAILED", "NO_TRADE_SIZE_ZERO"]:
                     log_entry["considered_action"] = "NO_TRADE_THRESHOLD_NOT_MET"

            if log_entry["considered_action"].startswith("NO_TRADE"):
                 logger.debug(f"NO_TRADE_EVAL: {market_ticker} @ {current_decision_dt_utc.strftime('%H:%M')} | P(YES):{log_entry['predicted_proba_yes']:.4f} | "
                              f"Reason:{log_entry['considered_action']} | "
                              f"ExecB:{current_yes_bid_at_t}, ExecA:{current_yes_ask_at_t} | "
                              f"EdgeY:{log_entry['edge_for_yes']}, EdgeN:{log_entry['edge_for_no']}")
            
            decision_eval_log.append(log_entry)
            session_trades_log.append({
                "decision_timestamp_utc": current_decision_dt_utc.isoformat(), "market_ticker": market_ticker,
                "kalshi_strike_price": kalshi_strike_price, 
                "predicted_proba_yes": round(predicted_proba_yes, 4) if pd.notna(predicted_proba_yes) else None,
                "model_prob_chosen_side": round(model_prob_chosen_side, 4) if pd.notna(model_prob_chosen_side) else None,
                "kalshi_yes_bid_at_decision_t": current_yes_bid_at_t, 
                "kalshi_yes_ask_at_decision_t": current_yes_ask_at_t, 
                "executed_trade_action": trade_action if num_contracts > 0 else log_entry["considered_action"],
                "num_contracts_sim": num_contracts, 
                "simulated_entry_price_cents": entry_price_chosen_side if pd.notna(entry_price_chosen_side) and num_contracts > 0 else None,
                "pnl_cents": pnl_this_trade_cents, "actual_market_result": actual_market_result,
                "session_capital_after_trade": round(global_config_state['current_capital_usd'],2)
            })
            current_decision_dt_utc += timedelta(minutes=global_config_state['decision_interval_minutes'])
    
    if decision_eval_log:
        df_decision_eval = pd.DataFrame(decision_eval_log)
        eval_log_filename = f"decision_eval_log_{session_key}_{MODEL_TYPE_TO_RUN}_{run_timestamp}.csv"
        eval_log_path = BACKTEST_LOGS_DIR / eval_log_filename
        df_decision_eval.to_csv(eval_log_path, index=False)
        logger.info(f"Saved detailed decision evaluation log to: {eval_log_path.name}")

    return session_trades_log, session_pnl_cents, session_contracts_traded

def main_backtest_loop():
    # ... (This function remains the same as your previous version, including the DEBUG level for file_handler_session) ...
    if not MARKET_OUTCOMES_CSV_PATH or not MARKET_OUTCOMES_CSV_PATH.exists(): logger.critical("Market outcomes CSV not found. Aborting."); return
    df_all_market_details = pd.read_csv(MARKET_OUTCOMES_CSV_PATH)
    df_all_market_details['market_open_time_iso'] = pd.to_datetime(df_all_market_details['market_open_time_iso'], utc=True, errors='coerce')
    df_all_market_details['market_close_time_iso'] = pd.to_datetime(df_all_market_details['market_close_time_iso'], utc=True, errors='coerce')
    df_all_market_details.dropna(subset=['market_open_time_iso', 'market_close_time_iso', 'kalshi_strike_price', 'market_ticker', 'result'], inplace=True)
    
    backtest_start_dt_utc = pd.to_datetime(BACKTEST_START_DATE_STR, utc=True).normalize()
    backtest_end_dt_utc = (pd.to_datetime(BACKTEST_END_DATE_STR, utc=True) + timedelta(days=1)).normalize() - timedelta(seconds=1)
    df_all_market_details = df_all_market_details[
        (df_all_market_details['market_close_time_iso'] >= backtest_start_dt_utc) &
        (df_all_market_details['market_close_time_iso'] <= backtest_end_dt_utc)
    ].copy()
    if df_all_market_details.empty: logger.warning(f"No markets for period: {BACKTEST_START_DATE_STR} to {BACKTEST_END_DATE_STR}"); return

    df_all_market_details['session_key'] = df_all_market_details.apply(utils.get_session_key_from_market_row, axis=1)
    df_all_market_details.dropna(subset=['session_key'], inplace=True)
    grouped_sessions = df_all_market_details.groupby('session_key')
    logger.info(f"Backtest: Identified {len(grouped_sessions)} unique hourly sessions.")

    global_config_state = {
        'initial_capital_usd': INITIAL_CAPITAL_USD, 
        'current_capital_usd': INITIAL_CAPITAL_USD, 
        'trade_decision_offset_minutes_from_market_close': TRADE_DECISION_OFFSET_MINUTES_FROM_MARKET_CLOSE,
        'decision_interval_minutes': DECISION_INTERVAL_MINUTES, 
        'kalshi_max_staleness_seconds_for_execution': KALSHI_MAX_STALENESS_SECONDS_FOR_EXECUTION_PRICES
    }
    overall_trades_log, overall_pnl_cents, overall_contracts_traded = [], 0, 0
    
    utils.BASE_PROJECT_DIR = BASE_PROJECT_DIR
    utils.BINANCE_FLAT_DATA_DIR = BINANCE_FLAT_DATA_DIR
    utils.KALSHI_DATA_DIR = KALSHI_DATA_ROOT_DIR 

    def sort_key_func(sk_str):
        pi = utils.parse_kalshi_ticker_info(f"DUMMY-{sk_str.replace('_', '')}-T1") 
        return pi['event_resolution_dt_utc'] if pi and 'event_resolution_dt_utc' in pi else dt.datetime.min.replace(tzinfo=timezone.utc)
    
    sorted_session_keys = sorted(grouped_sessions.groups.keys(), key=sort_key_func)
    
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s')
    global console_handler_main 
    if console_handler_main: console_handler_main.setFormatter(log_formatter)


    for session_key in sorted_session_keys:
        current_file_handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)]
        for h_file in current_file_handlers: h_file.close(); logging.getLogger().removeHandler(h_file)

        session_log_file_name = f"session_{session_key}_{MODEL_TYPE_TO_RUN}_{run_timestamp}.log"
        session_log_file_path = BACKTEST_LOGS_DIR / session_log_file_name
        file_handler_session = logging.FileHandler(str(session_log_file_path), mode='w') 
        file_handler_session.setFormatter(log_formatter)
        file_handler_session.setLevel(logging.DEBUG) 
        logging.getLogger().addHandler(file_handler_session) 
        
        logger.info(f"Processing session {session_key} (logs: {session_log_file_path.name})")

        session_markets_df = grouped_sessions.get_group(session_key)
        utils.clear_binance_cache(); utils.clear_kalshi_cache()
        if session_markets_df.empty: logger.info(f"No markets in session {session_key}. Skipping."); continue
        
        min_market_open_dt_session = session_markets_df['market_open_time_iso'].min()
        max_market_close_dt_session = session_markets_df['market_close_time_iso'].max()
        
        max_ta_lookback_minutes = 0 
        if hasattr(utils, 'BTC_MOMENTUM_WINDOWS') and utils.BTC_MOMENTUM_WINDOWS: max_ta_lookback_minutes = max(max_ta_lookback_minutes, max(utils.BTC_MOMENTUM_WINDOWS))
        max_ta_lookback_minutes = max(max_ta_lookback_minutes, getattr(utils, 'BTC_VOLATILITY_WINDOW', 0))
        if hasattr(utils, 'BTC_SMA_WINDOWS') and utils.BTC_SMA_WINDOWS: max_ta_lookback_minutes = max(max_ta_lookback_minutes, max(utils.BTC_SMA_WINDOWS))
        if hasattr(utils, 'BTC_EMA_WINDOWS') and utils.BTC_EMA_WINDOWS: max_ta_lookback_minutes = max(max_ta_lookback_minutes, max(utils.BTC_EMA_WINDOWS))
        max_ta_lookback_minutes = max(max_ta_lookback_minutes, getattr(utils, 'BTC_RSI_WINDOW', 0))
        if hasattr(utils, 'BTC_ATR_WINDOW') and utils.BTC_ATR_WINDOW : max_ta_lookback_minutes = max(max_ta_lookback_minutes, utils.BTC_ATR_WINDOW) # Include ATR window
        max_ta_lookback_minutes = max(max_ta_lookback_minutes, 60) 
        buffer_minutes = 60 
        
        binance_hist_start_dt = min_market_open_dt_session - timedelta(minutes=max_ta_lookback_minutes + buffer_minutes) 
        binance_hist_end_dt = max_market_close_dt_session + timedelta(minutes=buffer_minutes) 
        logger.info(f"  Session {session_key}: Binance data range for TA: {binance_hist_start_dt.strftime('%Y-%m-%d %H:%M')} to {binance_hist_end_dt.strftime('%Y-%m-%d %H:%M')}")
        
        binance_session_features_df = utils.load_and_prepare_binance_range_with_features(binance_hist_start_dt, binance_hist_end_dt)
        if binance_session_features_df is None or binance_session_features_df.empty:
            logger.critical(f"  Binance history load or feature calculation failed for session {session_key}. Skipping."); continue

        session_trades, session_pnl, session_contracts = run_hourly_session_backtest(
            session_markets_df.sort_values(by='market_close_time_iso'), binance_session_features_df, global_config_state 
        )
        overall_trades_log.extend(session_trades); overall_pnl_cents += session_pnl; overall_contracts_traded += session_contracts
        
        logger.info(f"--- Session {session_key} Summary ---")
        decision_eval_file = BACKTEST_LOGS_DIR / f'decision_eval_log_{session_key}_{MODEL_TYPE_TO_RUN}_{run_timestamp}.csv'
        num_decision_points = len(pd.read_csv(decision_eval_file)) if decision_eval_file.exists() else 0
        logger.info(f"  Decision Points Logged in decision_eval_log: {num_decision_points}")
        logger.info(f"  Executed Trades in session_trades_log: {sum(1 for t in session_trades if t.get('num_contracts_sim',0) > 0 and t.get('executed_trade_action') not in ['FEATURE_GEN_FAILED', 'MODEL_PRED_FAILED', 'NO_TRADE_THRESHOLD_NOT_MET', 'NO_TRADE_SIZE_ZERO'])}")
        logger.info(f"  Contracts Traded: {session_contracts}")
        logger.info(f"  P&L: ${session_pnl/100:.2f}, Capital After: ${global_config_state['current_capital_usd']:.2f}")
        
        if file_handler_session: file_handler_session.close(); logging.getLogger().removeHandler(file_handler_session)
            
    logger.info(f"\n\n======= OVERALL HISTORICAL BACKTEST SUMMARY ({MODEL_TYPE_TO_RUN.upper()}) =======")
    logger.info(f"Period: {BACKTEST_START_DATE_STR} to {BACKTEST_END_DATE_STR}")
    logger.info(f"Initial Capital: ${INITIAL_CAPITAL_USD:.2f}, Final Capital: ${global_config_state['current_capital_usd']:.2f}")
    num_executed_trades_overall = sum(1 for t in overall_trades_log if t.get('num_contracts_sim', 0) > 0 and t.get('executed_trade_action') not in ['FEATURE_GEN_FAILED', 'MODEL_PRED_FAILED', 'NO_TRADE_THRESHOLD_NOT_MET', 'NO_TRADE_SIZE_ZERO'])
    logger.info(f"Total Decisions Logged (in ALL_TRADES_EVALS): {len(overall_trades_log)}")
    logger.info(f"Total Executed Trades: {num_executed_trades_overall}, Total Contracts: {overall_contracts_traded}")
    logger.info(f"Total P&L: ${overall_pnl_cents/100:.2f}")
    if num_executed_trades_overall > 0: logger.info(f"Avg P&L/Executed Trade: ${overall_pnl_cents/num_executed_trades_overall/100:.4f}")
    if overall_contracts_traded > 0: logger.info(f"Avg P&L/Contract: ${overall_pnl_cents/overall_contracts_traded/100:.4f}")

    if overall_trades_log:
        overall_results_csv_filename = f"ALL_TRADES_WITH_EVALS_{MODEL_TYPE_TO_RUN}_{run_timestamp}.csv"
        overall_results_csv_path = BACKTEST_LOGS_DIR.parent / overall_results_csv_filename 
        df_overall_results = pd.DataFrame(overall_trades_log)
        df_overall_results.to_csv(overall_results_csv_path, index=False)
        logger.info(f"Overall backtest trade results (incl. non-executed evals) saved to: {overall_results_csv_path.resolve()}")
    
    logging.shutdown()

if __name__ == "__main__":
    if strategy._model is None or strategy._scaler is None or strategy._feature_order is None:
        logger.critical(f"Strategy for '{MODEL_TYPE_TO_RUN}' lacks artifacts. Aborting.")
    elif not MARKET_OUTCOMES_CSV_PATH:
        logger.critical("MARKET_OUTCOMES_CSV_PATH not set. Aborting.")
    else:
        logger.info(f"Starting Historical Backtest: {MODEL_TYPE_TO_RUN} Model (Calibrated_v2_15m_offset), Kelly Sizing...")
        main_backtest_loop()