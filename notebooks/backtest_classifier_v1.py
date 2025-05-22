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
# from sklearn.utils.validation import DataConversionWarning # No longer needed if versions match

# Filter specific UserWarnings from sklearn if necessary, but try to resolve underlying issues first
# warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn.utils.validation')

# --- Project Path Setup ---
try:
    # Try to set BASE_PROJECT_DIR relative to this script file
    script_path = Path(os.path.abspath(__file__))
    # Assuming this script is in /notebooks/, so .parent is /notebooks/, .parent.parent is project root
    BASE_PROJECT_DIR = script_path.parent.parent
    if not (BASE_PROJECT_DIR / "notebooks").exists(): # Fallback if structure is different
        BASE_PROJECT_DIR = Path.cwd() # Assume PWD is project root
        if not (BASE_PROJECT_DIR / "notebooks").exists() and (BASE_PROJECT_DIR.parent / "notebooks").exists():
             BASE_PROJECT_DIR = BASE_PROJECT_DIR.parent # If PWD is /notebooks
except NameError: # __file__ is not defined (e.g. running in basic Python interpreter or some IDEs)
    BASE_PROJECT_DIR = Path.cwd()
    # Attempt to find the 'notebooks' directory if CWD is project root or inside 'notebooks'
    if not (BASE_PROJECT_DIR / "notebooks").exists() and (BASE_PROJECT_DIR.parent / "notebooks").exists():
        BASE_PROJECT_DIR = BASE_PROJECT_DIR.parent
    print(f"Note: __file__ not defined, BASE_PROJECT_DIR set to: {BASE_PROJECT_DIR.resolve()}")


NOTEBOOKS_DIR = BASE_PROJECT_DIR / "notebooks"
KALSHI_DATA_ROOT_DIR = NOTEBOOKS_DIR / "kalshi_data"
BINANCE_FLAT_DATA_DIR = NOTEBOOKS_DIR / "binance_data"
MODEL_ARTIFACTS_ROOT_DIR = NOTEBOOKS_DIR / "trained_models"

# --- Model and Log Configuration ---
MODEL_TYPE_TO_RUN = "logistic_regression" # CHANGED
# MODEL_TYPE_TO_RUN = "random_forest"

if MODEL_TYPE_TO_RUN == "logistic_regression":
    CLASSIFIER_MODEL_SUBDIR_NAME = "logreg" # CHANGED - this is where LogReg artifacts are saved
    BACKTEST_LOG_SUBDIR_PREFIX = "backtest_logreg_calibrated_v2_15m_offset" # CHANGED
elif MODEL_TYPE_TO_RUN == "random_forest":
    CLASSIFIER_MODEL_SUBDIR_NAME = "rf"
    BACKTEST_LOG_SUBDIR_PREFIX = "backtest_random_forest_calibrated_v2_15m_offset"
else:
    raise ValueError(f"Unsupported MODEL_TYPE_TO_RUN: {MODEL_TYPE_TO_RUN}")

MODEL_ARTIFACTS_SPECIFIC_DIR = MODEL_ARTIFACTS_ROOT_DIR / CLASSIFIER_MODEL_SUBDIR_NAME
LOGS_BASE_DIR = NOTEBOOKS_DIR / "logs"
BACKTEST_LOGS_DIR = LOGS_BASE_DIR / BACKTEST_LOG_SUBDIR_PREFIX # Specific log subdir for this model type
BACKTEST_LOGS_DIR.mkdir(parents=True, exist_ok=True)
run_timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')

# --- Logging Setup ---
log_formatter_console = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s')
console_handler_main = logging.StreamHandler()
console_handler_main.setFormatter(log_formatter_console)
# Clear existing root handlers to avoid duplicate logs if script is re-run in same session
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.root.addHandler(console_handler_main)
logging.root.setLevel(logging.INFO) # Set root logger level

# Specific logger for this backtest instance
logger = logging.getLogger(f"Backtest_{MODEL_TYPE_TO_RUN.replace('_',' ').title()}_CalV2_15m")
logger.info(f"BASE_PROJECT_DIR: {BASE_PROJECT_DIR.resolve()}")
logger.info(f"NOTEBOOKS_DIR: {NOTEBOOKS_DIR.resolve()}")
logger.info(f"Attempting to load {MODEL_TYPE_TO_RUN} model artifacts from: {MODEL_ARTIFACTS_SPECIFIC_DIR.resolve()}")
logger.info(f"Backtest session logs will be saved under: {BACKTEST_LOGS_DIR.resolve()}")

# --- Import Custom Modules ---
# Ensure utils.py can find its data directories if it's imported by backtest_classifier_v1.py
# This assumes utils.py uses relative paths or can accept these as overrides.
# For simplicity, we can re-assign them in utils if utils.py is structured to allow it,
# or ensure utils.py can determine BASE_PROJECT_DIR itself.
# If utils.py is in the same directory (notebooks), relative imports in utils might rely on how this script is run.
# To be safe, ensure utils.py also has robust path detection or accepts overrides.
try:
    from . import utils # If utils is in the same package/directory
    from . import sizing
    from . import classifier_strategy as strategy
except ImportError: # Fallback for running as script or if '.' relative import fails
    import utils
    import sizing
    import classifier_strategy as strategy

# Set paths for utils module if it relies on global vars (safer if utils handles its own paths)
utils.BASE_PROJECT_DIR = BASE_PROJECT_DIR
utils.BINANCE_FLAT_DATA_DIR = BINANCE_FLAT_DATA_DIR
utils.KALSHI_DATA_DIR = KALSHI_DATA_ROOT_DIR


# --- Load Model Artifacts ---
# Pass the specific model_type_name to load_model_artifacts
if not strategy.load_model_artifacts(MODEL_ARTIFACTS_SPECIFIC_DIR, model_type_name=MODEL_TYPE_TO_RUN):
    logger.critical(f"CRITICAL: Failed to load '{MODEL_TYPE_TO_RUN}' model artifacts from {MODEL_ARTIFACTS_SPECIFIC_DIR.resolve()}. Aborting.")
    exit()
else:
    logger.info(f"Successfully loaded '{strategy._model_type_loaded}' model and artifacts for backtesting.")


# --- Load Market Outcomes Data ---
try:
    if not KALSHI_DATA_ROOT_DIR.exists():
        raise FileNotFoundError(f"Kalshi data root directory missing: {KALSHI_DATA_ROOT_DIR.resolve()}")
    # Find the latest NTM outcomes CSV
    outcomes_files = sorted(
        KALSHI_DATA_ROOT_DIR.glob("kalshi_btc_hourly_NTM_filtered_market_outcomes_*.csv"),
        key=os.path.getctime,
        reverse=True
    )
    if not outcomes_files:
        raise FileNotFoundError(f"No NTM outcomes CSV files found in {KALSHI_DATA_ROOT_DIR.resolve()}")
    MARKET_OUTCOMES_CSV_PATH = outcomes_files[0]
    logger.info(f"Backtest: Using market outcomes from {MARKET_OUTCOMES_CSV_PATH.resolve()}")
except FileNotFoundError as e:
    logger.critical(f"Backtest CRITICAL - {e}. Outcomes file required for backtesting.")
    exit()
except Exception as e:
    logger.critical(f"Backtest CRITICAL - Error finding/validating outcomes CSV: {e}", exc_info=True)
    exit()

# --- Backtest Configuration ---
BACKTEST_START_DATE_STR = "2025-05-12"  # Inclusive
BACKTEST_END_DATE_STR = "2025-05-12"    # Inclusive
DECISION_INTERVAL_MINUTES = 1
TRADE_DECISION_OFFSET_MINUTES_FROM_MARKET_CLOSE = 15 # Must match feature_engineering.ipynb
KALSHI_MAX_STALENESS_SECONDS_FOR_EXECUTION_PRICES = 120
INITIAL_CAPITAL_USD = 500.0

# --- Sizing Configuration ---
sizing.KELLY_FRACTION = 0.05 # More conservative for initial tests
sizing.MAX_CAPITAL_ALLOCATION_PERCENTAGE_OF_TOTAL = 0.05
sizing.MAX_CONTRACTS_PER_TRADE = 50

# --- Strategy Parameters ---
# These need to be tuned for Logistic Regression based on its probability distribution
if MODEL_TYPE_TO_RUN == "logistic_regression":
    strategy.MIN_MODEL_PROB_FOR_CONSIDERATION = 0.85 # Start high, observe LogReg's high-confidence predictions
    strategy.EDGE_THRESHOLD_FOR_TRADE = 0.10        # Start with a reasonable edge
elif MODEL_TYPE_TO_RUN == "random_forest": # Keep RF params for reference if switching back
    strategy.MIN_MODEL_PROB_FOR_CONSIDERATION = 0.85 # Example from previous discussion
    strategy.EDGE_THRESHOLD_FOR_TRADE = 0.10       # Example from previous discussion

logger.info(f"Strategy thresholds set for {MODEL_TYPE_TO_RUN}: MinProb={strategy.MIN_MODEL_PROB_FOR_CONSIDERATION}, Edge={strategy.EDGE_THRESHOLD_FOR_TRADE}")
logger.info(f"Sizing params: KellyFrac={sizing.KELLY_FRACTION}, MaxAllocPct={sizing.MAX_CAPITAL_ALLOCATION_PERCENTAGE_OF_TOTAL}, MaxContracts={sizing.MAX_CONTRACTS_PER_TRADE}")

detailed_log_counter_per_market = {} # Reset for each run of the script

def run_hourly_session_backtest(session_markets_df, binance_session_features_df, global_config_state):
    session_key = utils.get_session_key_from_market_row(session_markets_df.iloc[0])
    logger.info(f"--- Starting Backtest for Session: {session_key} ---")
    session_trades_log, session_pnl_cents, session_contracts_traded = [], 0, 0
    decision_eval_log = [] # For detailed logging of each decision point

    global detailed_log_counter_per_market

    for idx, market_row in session_markets_df.iterrows():
        market_ticker = market_row['market_ticker']
        kalshi_strike_price = market_row['kalshi_strike_price']
        market_open_dt_utc = market_row['market_open_time_iso']
        market_close_dt_utc = market_row['market_close_time_iso']
        actual_market_result = market_row['result']

        detailed_log_counter_per_market[market_ticker] = 0 # Reset for each new market

        parsed_ticker_info = utils.parse_kalshi_ticker_info(market_ticker)
        if not parsed_ticker_info:
            logger.warning(f"    Could not parse ticker {market_ticker}. Skipping market.")
            continue

        df_kalshi_market_minute_data = utils.load_kalshi_market_minute_data(
            market_ticker, parsed_ticker_info['date_str'], parsed_ticker_info['hour_str_EDT']
        )
        if df_kalshi_market_minute_data is None:
            df_kalshi_market_minute_data = pd.DataFrame() # Use empty DF if no data

        current_decision_dt_utc = market_open_dt_utc.replace(second=0, microsecond=0)
        latest_permissible_decision_dt_utc = (market_close_dt_utc -
            timedelta(minutes=global_config_state['trade_decision_offset_minutes_from_market_close'])).replace(second=0, microsecond=0)

        while current_decision_dt_utc <= latest_permissible_decision_dt_utc:
            signal_dt_for_features_utc = (current_decision_dt_utc - timedelta(minutes=1)).replace(second=0, microsecond=0)
            signal_ts_utc_for_features = int(signal_dt_for_features_utc.timestamp())

            if signal_ts_utc_for_features not in binance_session_features_df.index:
                logger.debug(f"  Binance features not available for signal_ts {signal_ts_utc_for_features} @ {current_decision_dt_utc}. Skipping decision point.")
                current_decision_dt_utc += timedelta(minutes=global_config_state['decision_interval_minutes'])
                continue

            # Use up to and including signal_ts for features
            btc_history_for_features = binance_session_features_df[binance_session_features_df.index <= signal_ts_utc_for_features].copy()
            if btc_history_for_features.empty:
                logger.debug(f"  BTC history for features empty for signal_ts {signal_ts_utc_for_features} @ {current_decision_dt_utc}. Skipping.")
                current_decision_dt_utc += timedelta(minutes=global_config_state['decision_interval_minutes'])
                continue

            kalshi_prices_at_decision_t = utils.get_kalshi_prices_at_decision(
                df_kalshi_market_minute_data,
                int(current_decision_dt_utc.timestamp()),
                global_config_state['kalshi_max_staleness_seconds_for_execution']
            )
            current_yes_bid_at_t = kalshi_prices_at_decision_t.get('yes_bid') if kalshi_prices_at_decision_t else None
            current_yes_ask_at_t = kalshi_prices_at_decision_t.get('yes_ask') if kalshi_prices_at_decision_t else None

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
            
            # Detailed logging for first few iterations or extreme predictions
            log_this_iteration_detail = False
            if detailed_log_counter_per_market.get(market_ticker, 0) < 3 or \
               (predicted_proba_yes is not None and (predicted_proba_yes < 0.1 or predicted_proba_yes > 0.9)) or \
               feature_vector_series is None or predicted_proba_yes is None:
                log_this_iteration_detail = True

            if log_this_iteration_detail:
                logger.debug(f"--- Detailed Feature Check @ {current_decision_dt_utc.isoformat()} for {market_ticker} (Strike: {kalshi_strike_price}) ---")
                logger.debug(f"  Signal datetime for t-1 features: {signal_dt_for_features_utc.isoformat()} (ts: {signal_ts_utc_for_features})")
                # ... (rest of detailed feature logging can be kept if desired) ...
                if feature_vector_series is not None:
                    key_feat_subset = {k: feature_vector_series.get(k) for k in ['btc_price_t_minus_1', 'distance_to_strike', 'kalshi_mid_price', 'time_until_market_close_min', 'btc_atr_14', 'distance_to_strike_norm_atr'] if k in feature_vector_series}
                    logger.debug(f"    Key Features: {key_feat_subset}")
                    if predicted_proba_yes is not None: logger.debug(f"  Model P(YES) from these features: {predicted_proba_yes:.4f}")
                    else: logger.debug("  Model P(YES) calculation FAILED.")
                else: logger.debug(f"  Feature vector generation FAILED.")
                detailed_log_counter_per_market[market_ticker] = detailed_log_counter_per_market.get(market_ticker,0) + 1


            # Prepare log entry for decision_eval_log
            log_entry_eval = {
                "decision_dt_utc": current_decision_dt_utc.isoformat(),
                "market_ticker": market_ticker,
                "predicted_proba_yes": None, # Default
                "kalshi_yes_bid_at_t": current_yes_bid_at_t,
                "kalshi_yes_ask_at_t": current_yes_ask_at_t,
                "implied_proba_yes_at_ask": None, "edge_for_yes": None,
                "predicted_proba_no": None, "implied_proba_no_at_bid": None, "edge_for_no": None,
                "considered_action": "FEATURE_GEN_FAILED"
            }

            if feature_vector_series is None: # Feature generation failed
                decision_eval_log.append(log_entry_eval)
                # Add to main trades log as well to show it was considered
                session_trades_log.append({
                    "decision_timestamp_utc": current_decision_dt_utc.isoformat(), "market_ticker": market_ticker,
                    "kalshi_strike_price": kalshi_strike_price, "predicted_proba_yes": None,
                    "model_prob_chosen_side": None, "kalshi_yes_bid_at_decision_t": current_yes_bid_at_t,
                    "kalshi_yes_ask_at_decision_t": current_yes_ask_at_t, "executed_trade_action": "FEATURE_GEN_FAILED",
                    "num_contracts_sim": 0, "simulated_entry_price_cents": None, "pnl_cents": np.nan,
                    "actual_market_result": actual_market_result, "session_capital_after_trade": global_config_state['current_capital_usd']
                })
                current_decision_dt_utc += timedelta(minutes=global_config_state['decision_interval_minutes']); continue

            log_entry_eval["considered_action"] = "MODEL_PRED_FAILED" # Default if pred fails
            if predicted_proba_yes is not None:
                log_entry_eval["predicted_proba_yes"] = round(predicted_proba_yes, 4)
                log_entry_eval["predicted_proba_no"] = round(1.0 - predicted_proba_yes, 4)
                if pd.notna(current_yes_ask_at_t) and 0 < current_yes_ask_at_t < 100:
                    log_entry_eval["implied_proba_yes_at_ask"] = round(current_yes_ask_at_t / 100.0, 4)
                    log_entry_eval["edge_for_yes"] = round(predicted_proba_yes - log_entry_eval["implied_proba_yes_at_ask"], 4)
                if pd.notna(current_yes_bid_at_t) and 0 < current_yes_bid_at_t < 100:
                    cost_of_no_cents_eval = 100.0 - current_yes_bid_at_t
                    if 0 < cost_of_no_cents_eval < 100:
                        log_entry_eval["implied_proba_no_at_bid"] = round(cost_of_no_cents_eval / 100.0, 4)
                        if log_entry_eval["predicted_proba_no"] is not None: # Check if P(NO) was calculable
                             log_entry_eval["edge_for_no"] = round(log_entry_eval["predicted_proba_no"] - log_entry_eval["implied_proba_no_at_bid"], 4)
                log_entry_eval["considered_action"] = "NONE" # Base if prediction was successful
            else: # Model prediction failed
                decision_eval_log.append(log_entry_eval)
                session_trades_log.append({
                    "decision_timestamp_utc": current_decision_dt_utc.isoformat(), "market_ticker": market_ticker,
                    "kalshi_strike_price": kalshi_strike_price, "predicted_proba_yes": None,
                    "model_prob_chosen_side": None, "kalshi_yes_bid_at_decision_t": current_yes_bid_at_t,
                    "kalshi_yes_ask_at_decision_t": current_yes_ask_at_t, "executed_trade_action": "MODEL_PRED_FAILED",
                    "num_contracts_sim": 0, "simulated_entry_price_cents": None, "pnl_cents": np.nan,
                    "actual_market_result": actual_market_result, "session_capital_after_trade": global_config_state['current_capital_usd']
                })
                current_decision_dt_utc += timedelta(minutes=global_config_state['decision_interval_minutes']); continue

            trade_action, model_prob_chosen_side, entry_price_chosen_side = strategy.get_trade_decision(
                predicted_proba_yes, current_yes_bid_at_t, current_yes_ask_at_t
            )
            if trade_action: # If a trade is considered (BUY_YES or BUY_NO)
                 log_entry_eval["considered_action"] = trade_action
            else: # If no trade action from strategy thresholds
                 log_entry_eval["considered_action"] = "NO_TRADE_THRESHOLD_NOT_MET"


            num_contracts, pnl_this_trade_cents = 0, np.nan
            executed_action_for_log = log_entry_eval["considered_action"] # Default to what was considered

            if trade_action and pd.notna(model_prob_chosen_side) and pd.notna(entry_price_chosen_side):
                num_contracts = sizing.calculate_kelly_position_size(
                    model_prob_win=model_prob_chosen_side,
                    entry_price_cents=entry_price_chosen_side,
                    available_capital_usd=global_config_state['current_capital_usd']
                )
                if num_contracts > 0:
                    executed_action_for_log = trade_action # Confirmed trade
                    pnl_per_contract = 0
                    if actual_market_result.lower() == 'yes':
                        pnl_per_contract = (100.0 - entry_price_chosen_side) if trade_action == "BUY_YES" else (-entry_price_chosen_side)
                    elif actual_market_result.lower() == 'no':
                        pnl_per_contract = (-entry_price_chosen_side) if trade_action == "BUY_YES" else (100.0 - entry_price_chosen_side)
                    
                    pnl_this_trade_cents = pnl_per_contract * num_contracts
                    global_config_state['current_capital_usd'] += (pnl_this_trade_cents / 100.0)
                    session_pnl_cents += pnl_this_trade_cents
                    session_contracts_traded += num_contracts
                    logger.info(f"TRADE: {market_ticker} @ {current_decision_dt_utc.strftime('%H:%M')} | P(YES):{predicted_proba_yes:.3f} "
                               f"P({trade_action.split('_')[1]}):{model_prob_chosen_side:.3f} | {trade_action} x{num_contracts}@{entry_price_chosen_side:.0f}c "
                               f"(Exec B:{current_yes_bid_at_t if pd.notna(current_yes_bid_at_t) else 'N/A'}, Exec A:{current_yes_ask_at_t if pd.notna(current_yes_ask_at_t) else 'N/A'}) | "
                               f"Outcome:{actual_market_result.upper()} | PNL:{pnl_this_trade_cents:.0f}c | Cap:${global_config_state['current_capital_usd']:.2f}")
                else: # Sizing resulted in 0 contracts
                    log_entry_eval["considered_action"] = "NO_TRADE_SIZE_ZERO"
                    executed_action_for_log = "NO_TRADE_SIZE_ZERO"
            
            # For cases where trade_action was None from the start, or num_contracts was 0
            if log_entry_eval["considered_action"] not in ["FEATURE_GEN_FAILED", "MODEL_PRED_FAILED", "NO_TRADE_SIZE_ZERO"] and not trade_action:
                 log_entry_eval["considered_action"] = "NO_TRADE_THRESHOLD_NOT_MET" # Ensure correct reason if it passed model pred
                 executed_action_for_log = "NO_TRADE_THRESHOLD_NOT_MET"


            if log_entry_eval["considered_action"].startswith("NO_TRADE"):
                 logger.debug(f"NO_TRADE_EVAL: {market_ticker} @ {current_decision_dt_utc.strftime('%H:%M')} | P(YES):{log_entry_eval['predicted_proba_yes']:.4f} | "
                              f"Reason:{log_entry_eval['considered_action']} | "
                              f"ExecB:{current_yes_bid_at_t}, ExecA:{current_yes_ask_at_t} | "
                              f"EdgeY:{log_entry_eval['edge_for_yes']}, EdgeN:{log_entry_eval['edge_for_no']}")

            decision_eval_log.append(log_entry_eval) # Add to the detailed eval log

            # Add to main overall trades log
            session_trades_log.append({
                "decision_timestamp_utc": current_decision_dt_utc.isoformat(),
                "market_ticker": market_ticker,
                "kalshi_strike_price": kalshi_strike_price,
                "predicted_proba_yes": round(predicted_proba_yes, 4) if pd.notna(predicted_proba_yes) else None,
                "model_prob_chosen_side": round(model_prob_chosen_side, 4) if pd.notna(model_prob_chosen_side) else None,
                "kalshi_yes_bid_at_decision_t": current_yes_bid_at_t,
                "kalshi_yes_ask_at_decision_t": current_yes_ask_at_t,
                "executed_trade_action": executed_action_for_log, # This is the final outcome reason
                "num_contracts_sim": num_contracts,
                "simulated_entry_price_cents": entry_price_chosen_side if pd.notna(entry_price_chosen_side) and num_contracts > 0 else None,
                "pnl_cents": pnl_this_trade_cents if pd.notna(pnl_this_trade_cents) else np.nan, # Ensure NaN if no trade
                "actual_market_result": actual_market_result,
                "session_capital_after_trade": round(global_config_state['current_capital_usd'],2)
            })
            current_decision_dt_utc += timedelta(minutes=global_config_state['decision_interval_minutes'])

    if decision_eval_log: # Save detailed decision log for this session
        df_decision_eval = pd.DataFrame(decision_eval_log)
        eval_log_filename = f"decision_eval_log_{session_key}_{MODEL_TYPE_TO_RUN}_{run_timestamp}.csv"
        eval_log_path = BACKTEST_LOGS_DIR / eval_log_filename
        df_decision_eval.to_csv(eval_log_path, index=False)
        logger.info(f"Saved detailed decision evaluation log ({len(df_decision_eval)} rows) to: {eval_log_path.name}")

    return session_trades_log, session_pnl_cents, session_contracts_traded


def main_backtest_loop():
    if not MARKET_OUTCOMES_CSV_PATH or not MARKET_OUTCOMES_CSV_PATH.exists():
        logger.critical(f"Market outcomes CSV not found at {MARKET_OUTCOMES_CSV_PATH}. Aborting.")
        return

    df_all_market_details = pd.read_csv(MARKET_OUTCOMES_CSV_PATH)
    df_all_market_details['market_open_time_iso'] = pd.to_datetime(df_all_market_details['market_open_time_iso'], utc=True, errors='coerce')
    df_all_market_details['market_close_time_iso'] = pd.to_datetime(df_all_market_details['market_close_time_iso'], utc=True, errors='coerce')
    df_all_market_details.dropna(subset=['market_open_time_iso', 'market_close_time_iso', 'kalshi_strike_price', 'market_ticker', 'result'], inplace=True)

    backtest_start_dt_utc = pd.to_datetime(BACKTEST_START_DATE_STR, utc=True).normalize()
    backtest_end_dt_utc = (pd.to_datetime(BACKTEST_END_DATE_STR, utc=True) + timedelta(days=1)).normalize() - timedelta(seconds=1) # End of day

    df_all_market_details = df_all_market_details[
        (df_all_market_details['market_close_time_iso'] >= backtest_start_dt_utc) &
        (df_all_market_details['market_close_time_iso'] <= backtest_end_dt_utc)
    ].copy()

    if df_all_market_details.empty:
        logger.warning(f"No markets found for the period: {BACKTEST_START_DATE_STR} to {BACKTEST_END_DATE_STR}")
        return

    df_all_market_details['session_key'] = df_all_market_details.apply(utils.get_session_key_from_market_row, axis=1)
    df_all_market_details.dropna(subset=['session_key'], inplace=True) # Drop if session key could not be parsed
    grouped_sessions = df_all_market_details.groupby('session_key')
    logger.info(f"Backtest: Identified {len(grouped_sessions)} unique hourly sessions for the specified date range.")

    global_config_state = {
        'initial_capital_usd': INITIAL_CAPITAL_USD,
        'current_capital_usd': INITIAL_CAPITAL_USD,
        'trade_decision_offset_minutes_from_market_close': TRADE_DECISION_OFFSET_MINUTES_FROM_MARKET_CLOSE,
        'decision_interval_minutes': DECISION_INTERVAL_MINUTES,
        'kalshi_max_staleness_seconds_for_execution': KALSHI_MAX_STALENESS_SECONDS_FOR_EXECUTION_PRICES
    }
    overall_trades_log, overall_pnl_cents, overall_contracts_traded = [], 0, 0

    # Sort session keys chronologically
    def sort_key_func(sk_str): # sk_str is like "25MAY11_20"
        try:
            # Simplified parsing just for sorting by date and hour
            date_part = sk_str.split('_')[0] # "25MAY11"
            hour_part_edt = int(sk_str.split('_')[1]) # 20
            
            day = int(date_part[:2])
            month_str = date_part[2:5]
            year_suffix = int(date_part[5:])
            year = 2000 + year_suffix
            
            month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                         'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
            month = month_map[month_str.upper()]
            
            # Create a naive datetime object for sorting purposes (timezone doesn't matter for sorting)
            return dt.datetime(year, month, day, hour_part_edt)
        except Exception as e:
            logger.warning(f"Could not parse session key '{sk_str}' for sorting: {e}. Using min datetime.")
            return dt.datetime.min # Fallback for unparseable keys

    sorted_session_keys = sorted(grouped_sessions.groups.keys(), key=sort_key_func)
    
    log_formatter_file = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s')

    for session_key in sorted_session_keys:
        # Remove previous file handlers for sessions to avoid logging to multiple files
        # Keep the console handler
        current_file_handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)]
        for h_file in current_file_handlers:
            h_file.close()
            logging.getLogger().removeHandler(h_file)

        session_log_file_name = f"session_{session_key}_{MODEL_TYPE_TO_RUN}_{run_timestamp}.log"
        session_log_file_path = BACKTEST_LOGS_DIR / session_log_file_name
        file_handler_session = logging.FileHandler(str(session_log_file_path), mode='w')
        file_handler_session.setFormatter(log_formatter_file)
        file_handler_session.setLevel(logging.DEBUG) # Log DEBUG level to session files
        logging.getLogger().addHandler(file_handler_session) # Add to the root logger for this session

        logger.info(f"Processing session {session_key} (detailed logs will be in: {session_log_file_path.name})")

        session_markets_df = grouped_sessions.get_group(session_key)
        utils.clear_binance_cache() # Clear per session to manage memory for raw daily Binance files
        utils.clear_kalshi_cache()  # Clear Kalshi market data cache per session

        if session_markets_df.empty:
            logger.info(f"No markets in session {session_key}. Skipping."); continue

        min_market_open_dt_session = session_markets_df['market_open_time_iso'].min()
        max_market_close_dt_session = session_markets_df['market_close_time_iso'].max()

        # Determine TA lookback needed from utils module constants
        all_ta_windows_minutes = utils.BTC_MOMENTUM_WINDOWS + [utils.BTC_VOLATILITY_WINDOW] + \
                                 utils.BTC_SMA_WINDOWS + utils.BTC_EMA_WINDOWS + \
                                 [utils.BTC_RSI_WINDOW, utils.BTC_ATR_WINDOW]
        max_ta_lookback_minutes = max(filter(None, all_ta_windows_minutes), default=60) # default 60 if list is empty/all zeros
        buffer_minutes = 60 # Additional buffer for feature stability

        binance_hist_start_dt = min_market_open_dt_session - timedelta(minutes=max_ta_lookback_minutes + buffer_minutes)
        binance_hist_end_dt = max_market_close_dt_session + timedelta(minutes=buffer_minutes)
        logger.info(f"  Session {session_key}: Binance data range for TA features: {binance_hist_start_dt.strftime('%Y-%m-%d %H:%M')} to {binance_hist_end_dt.strftime('%Y-%m-%d %H:%M')}")

        binance_session_features_df = utils.load_and_prepare_binance_range_with_features(binance_hist_start_dt, binance_hist_end_dt)
        if binance_session_features_df is None or binance_session_features_df.empty:
            logger.critical(f"  Binance history load or feature calculation FAILED for session {session_key}. Cannot proceed with this session."); continue

        session_trades, session_pnl, session_contracts = run_hourly_session_backtest(
            session_markets_df.sort_values(by='market_close_time_iso'), # Process markets chronologically within session
            binance_session_features_df,
            global_config_state
        )
        overall_trades_log.extend(session_trades)
        overall_pnl_cents += session_pnl
        overall_contracts_traded += session_contracts

        logger.info(f"--- Session {session_key} Summary ---")
        decision_eval_file_check = BACKTEST_LOGS_DIR / f'decision_eval_log_{session_key}_{MODEL_TYPE_TO_RUN}_{run_timestamp}.csv'
        num_decision_points_logged = 0
        if decision_eval_file_check.exists():
            try: num_decision_points_logged = len(pd.read_csv(decision_eval_file_check))
            except Exception: pass # Ignore if file empty or corrupt
        logger.info(f"  Decision Points Logged in decision_eval_log: {num_decision_points_logged}")
        num_executed_session_trades = sum(1 for t in session_trades if t.get('num_contracts_sim',0) > 0 and isinstance(t.get('executed_trade_action'), str) and t.get('executed_trade_action').startswith('BUY_'))
        logger.info(f"  Executed Trades in this session: {num_executed_session_trades}")
        logger.info(f"  Contracts Traded in this session: {session_contracts}")
        logger.info(f"  P&L for this session: ${session_pnl/100.0:.2f}")
        logger.info(f"  Capital after this session: ${global_config_state['current_capital_usd']:.2f}")
        
        if file_handler_session: # Close and remove handler for this session's file
            file_handler_session.close()
            logging.getLogger().removeHandler(file_handler_session)

    logger.info(f"\n\n======= OVERALL HISTORICAL BACKTEST SUMMARY ({MODEL_TYPE_TO_RUN.upper()}) =======")
    logger.info(f"Backtesting Period: {BACKTEST_START_DATE_STR} to {BACKTEST_END_DATE_STR}")
    logger.info(f"Initial Capital: ${INITIAL_CAPITAL_USD:.2f}")
    logger.info(f"Final Capital: ${global_config_state['current_capital_usd']:.2f}")
    
    num_executed_trades_overall = sum(1 for t in overall_trades_log if t.get('num_contracts_sim', 0) > 0 and isinstance(t.get('executed_trade_action'), str) and t.get('executed_trade_action').startswith('BUY_'))
    logger.info(f"Total Decision Records Logged (in ALL_TRADES_EVALS): {len(overall_trades_log)}")
    logger.info(f"Total Executed Trades Overall: {num_executed_trades_overall}")
    logger.info(f"Total Contracts Traded Overall: {overall_contracts_traded}")
    logger.info(f"Total P&L Overall: ${overall_pnl_cents/100.0:.2f}")

    if num_executed_trades_overall > 0:
        logger.info(f"Average P&L per Executed Trade: ${overall_pnl_cents/num_executed_trades_overall/100.0:.4f}")
    if overall_contracts_traded > 0:
        logger.info(f"Average P&L per Contract: ${overall_pnl_cents/overall_contracts_traded/100.0:.4f}")

    if overall_trades_log:
        # Save to the main LOGS_BASE_DIR, not the specific subdirectory for this run's session logs
        overall_results_csv_filename = f"ALL_TRADES_WITH_EVALS_{MODEL_TYPE_TO_RUN}_{run_timestamp}.csv"
        overall_results_csv_path = LOGS_BASE_DIR / overall_results_csv_filename
        df_overall_results = pd.DataFrame(overall_trades_log)
        df_overall_results.to_csv(overall_results_csv_path, index=False)
        logger.info(f"Overall backtest trade results (incl. non-executed evals) saved to: {overall_results_csv_path.resolve()}")
    else:
        logger.info("No trades or decision points were logged in overall_trades_log.")
    
    logging.shutdown() # Gracefully close all handlers

if __name__ == "__main__":
    if strategy._model is None or strategy._scaler is None or strategy._feature_order is None or strategy._imputation_values is None:
        logger.critical(f"Strategy for '{MODEL_TYPE_TO_RUN}' is missing critical artifacts (model, scaler, feature_order, or imputation_values). Aborting.")
    elif not MARKET_OUTCOMES_CSV_PATH: # Check if MARKET_OUTCOMES_CSV_PATH was successfully set
        logger.critical("MARKET_OUTCOMES_CSV_PATH not set. Aborting backtest.")
    else:
        logger.info(f"Starting Historical Backtest for: {MODEL_TYPE_TO_RUN} Model, Kelly Sizing...")
        main_backtest_loop()