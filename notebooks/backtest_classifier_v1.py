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

MODEL_TYPE_TO_RUN = "logistic_regression" 

if MODEL_TYPE_TO_RUN == "logistic_regression": CLASSIFIER_MODEL_SUBDIR_NAME = "logreg"
elif MODEL_TYPE_TO_RUN == "random_forest": CLASSIFIER_MODEL_SUBDIR_NAME = "rf"
else: raise ValueError(f"Unsupported MODEL_TYPE_TO_RUN: {MODEL_TYPE_TO_RUN}")

MODEL_ARTIFACTS_SPECIFIC_DIR = MODEL_ARTIFACTS_ROOT_DIR / CLASSIFIER_MODEL_SUBDIR_NAME
LOGS_BASE_DIR = NOTEBOOKS_DIR / "logs" 
BACKTEST_LOGS_DIR = LOGS_BASE_DIR / f"backtest_{MODEL_TYPE_TO_RUN}_{CLASSIFIER_MODEL_SUBDIR_NAME}"
BACKTEST_LOGS_DIR.mkdir(parents=True, exist_ok=True)
run_timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')

# --- Root Logger for Initial Setup (Console Only) ---
# This ensures console output during setup, and then per-session file handlers take over for detailed logs.
log_formatter_console = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s')
console_handler_main = logging.StreamHandler()
console_handler_main.setFormatter(log_formatter_console)

# Clear any existing handlers from the root logger to avoid conflicts or duplicate messages
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.root.addHandler(console_handler_main) # Add our configured console handler
logging.root.setLevel(logging.INFO) # Set root logger level

logger = logging.getLogger(f"Backtest_{MODEL_TYPE_TO_RUN}") 
logger.info(f"BASE_PROJECT_DIR: {BASE_PROJECT_DIR.resolve()}")
logger.info(f"Attempting to load {MODEL_TYPE_TO_RUN} model artifacts from: {MODEL_ARTIFACTS_SPECIFIC_DIR.resolve()}")
logger.info(f"Backtest session logs will be saved under: {BACKTEST_LOGS_DIR.resolve()}")

import utils 
import sizing 
import classifier_strategy as strategy 

if not strategy.load_model_artifacts(MODEL_ARTIFACTS_SPECIFIC_DIR, model_type_name=MODEL_TYPE_TO_RUN):
    logger.critical(f"CRITICAL: Failed to load '{MODEL_TYPE_TO_RUN}' model artifacts. Aborting.")
    exit()
else:
    logger.info(f"Successfully loaded '{MODEL_TYPE_TO_RUN}' model and artifacts.")

try:
    if not KALSHI_DATA_ROOT_DIR.exists(): raise FileNotFoundError(f"Kalshi data root missing: {KALSHI_DATA_ROOT_DIR.resolve()}")
    outcomes_files = sorted(KALSHI_DATA_ROOT_DIR.glob("kalshi_btc_hourly_NTM_filtered_market_outcomes_*.csv"), key=os.path.getctime, reverse=True)
    if not outcomes_files: raise FileNotFoundError(f"No NTM outcomes CSV in {KALSHI_DATA_ROOT_DIR.resolve()}")
    MARKET_OUTCOMES_CSV_PATH = outcomes_files[0]
    logger.info(f"Backtest: Using market outcomes from {MARKET_OUTCOMES_CSV_PATH.resolve()}")
except FileNotFoundError as e: logger.critical(f"Backtest CRITICAL - {e}. Outcomes file required."); exit()
except Exception as e: logger.critical(f"Backtest CRITICAL - Error finding outcomes CSV: {e}", exc_info=True); exit()

BACKTEST_START_DATE_STR = "2025-05-12"; BACKTEST_END_DATE_STR = "2025-05-12"   
DECISION_INTERVAL_MINUTES = 1; TRADE_DECISION_OFFSET_MINUTES_FROM_MARKET_CLOSE = 5 
KALSHI_MAX_STALENESS_SECONDS = 120; INITIAL_CAPITAL_USD = 500.0
sizing.KELLY_FRACTION = 0.1; sizing.MAX_CAPITAL_ALLOCATION_PERCENTAGE_OF_TOTAL = 0.10 
sizing.MAX_CONTRACTS_PER_TRADE = 200 

if MODEL_TYPE_TO_RUN == "logistic_regression":
    strategy.MIN_MODEL_PROB_FOR_CONSIDERATION = 0.70 
    strategy.EDGE_THRESHOLD_FOR_TRADE = 0.10      
elif MODEL_TYPE_TO_RUN == "random_forest":
    strategy.MIN_MODEL_PROB_FOR_CONSIDERATION = 0.85 
    strategy.EDGE_THRESHOLD_FOR_TRADE = 0.20
logger.info(f"Strategy thresholds set for {MODEL_TYPE_TO_RUN}: MinProb={strategy.MIN_MODEL_PROB_FOR_CONSIDERATION}, Edge={strategy.EDGE_THRESHOLD_FOR_TRADE}")

def run_hourly_session_backtest(session_markets_df, binance_session_features_df, global_config_state):
    session_key = utils.get_session_key_from_market_row(session_markets_df.iloc[0])
    logger.info(f"--- Starting Backtest for Session: {session_key} ---")
    session_trades_log, session_pnl_cents, session_contracts_traded = [], 0, 0

    for idx, market_row in session_markets_df.iterrows():
        market_ticker, kalshi_strike_price = market_row['market_ticker'], market_row['kalshi_strike_price']
        market_open_dt_utc, market_close_dt_utc = market_row['market_open_time_iso'], market_row['market_close_time_iso']
        actual_market_result = market_row['result']
        logger.debug(f"  Processing Market: {market_ticker}, Strike: {kalshi_strike_price}")

        parsed_ticker_info = utils.parse_kalshi_ticker_info(market_ticker)
        if not parsed_ticker_info: logger.warning(f"    Could not parse ticker {market_ticker}. Skipping."); continue
        
        kalshi_market_day_data_df = utils.load_kalshi_market_minute_data(
            market_ticker, parsed_ticker_info['date_str'], parsed_ticker_info['hour_str_EDT']
        )
        if kalshi_market_day_data_df is None: kalshi_market_day_data_df = pd.DataFrame()

        current_decision_dt_utc = market_open_dt_utc.replace(second=0, microsecond=0)
        latest_permissible_decision_dt_utc = (market_close_dt_utc - 
            timedelta(minutes=global_config_state['trade_decision_offset_minutes_from_market_close'])).replace(second=0, microsecond=0)

        while current_decision_dt_utc <= latest_permissible_decision_dt_utc:
            signal_ts_utc_for_features = int((current_decision_dt_utc - timedelta(minutes=1)).replace(second=0, microsecond=0).timestamp())
            if signal_ts_utc_for_features not in binance_session_features_df.index:
                current_decision_dt_utc += timedelta(minutes=global_config_state['decision_interval_minutes']); continue
            
            btc_history_for_features = binance_session_features_df[binance_session_features_df.index <= signal_ts_utc_for_features].copy()
            if btc_history_for_features.empty:
                current_decision_dt_utc += timedelta(minutes=global_config_state['decision_interval_minutes']); continue
                
            current_kalshi_prices = utils.get_kalshi_prices_at_decision(
                kalshi_market_day_data_df, int(current_decision_dt_utc.timestamp()), 
                global_config_state['kalshi_max_staleness_seconds']
            )
            current_yes_bid, current_yes_ask = (current_kalshi_prices.get('yes_bid'), current_kalshi_prices.get('yes_ask')) if current_kalshi_prices else (None, None)

            feature_vector_series = strategy.generate_live_features(
                btc_price_history_df=btc_history_for_features, current_kalshi_bid=current_yes_bid, 
                current_kalshi_ask=current_yes_ask, kalshi_market_history_df=kalshi_market_day_data_df, 
                kalshi_strike_price=kalshi_strike_price, decision_point_dt_utc=current_decision_dt_utc,
                kalshi_market_close_dt_utc=market_close_dt_utc
            )
            if feature_vector_series is None:
                current_decision_dt_utc += timedelta(minutes=global_config_state['decision_interval_minutes']); continue
            
            predicted_proba_yes = strategy.calculate_model_prediction_proba(feature_vector_series)
            if predicted_proba_yes is None:
                current_decision_dt_utc += timedelta(minutes=global_config_state['decision_interval_minutes']); continue
            
            trade_action, model_prob_chosen_side, entry_price_chosen_side = strategy.get_trade_decision(
                predicted_proba_yes, current_yes_bid, current_yes_ask
            )
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
                               f"(B:{current_yes_bid if pd.notna(current_yes_bid) else 'N/A'},A:{current_yes_ask if pd.notna(current_yes_ask) else 'N/A'}) | "
                               f"Outcome:{actual_market_result.upper()} | PNL:{pnl_this_trade_cents:.0f}c | Cap:${global_config_state['current_capital_usd']:.2f}")
                else: trade_action = "NO_TRADE" 
            if not trade_action or num_contracts == 0: 
                 logger.debug(f"NO_TRADE: {market_ticker} @ {current_decision_dt_utc.strftime('%H:%M')} | P(model_yes)={predicted_proba_yes:.3f} | Action:{trade_action if trade_action else 'None'} | B:{current_yes_bid}, A:{current_yes_ask}")
            session_trades_log.append({
                "decision_timestamp_utc": current_decision_dt_utc.isoformat(), "market_ticker": market_ticker,
                "kalshi_strike_price": kalshi_strike_price, "predicted_proba_yes": round(predicted_proba_yes, 4) if pd.notna(predicted_proba_yes) else None,
                "model_prob_chosen_side": round(model_prob_chosen_side, 4) if pd.notna(model_prob_chosen_side) else None,
                "kalshi_yes_bid_at_decision": current_yes_bid, "kalshi_yes_ask_at_decision": current_yes_ask,
                "executed_trade_action": trade_action if num_contracts > 0 else "NO_TRADE", "num_contracts_sim": num_contracts,
                "simulated_entry_price_cents": entry_price_chosen_side if pd.notna(entry_price_chosen_side) and num_contracts > 0 else None,
                "pnl_cents": pnl_this_trade_cents, "actual_market_result": actual_market_result,
                "session_capital_after_trade": round(global_config_state['current_capital_usd'],2)
            })
            current_decision_dt_utc += timedelta(minutes=global_config_state['decision_interval_minutes'])
    return session_trades_log, session_pnl_cents, session_contracts_traded

def main_backtest_loop():
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
        'initial_capital_usd': INITIAL_CAPITAL_USD, 'current_capital_usd': INITIAL_CAPITAL_USD, 
        'trade_decision_offset_minutes_from_market_close': TRADE_DECISION_OFFSET_MINUTES_FROM_MARKET_CLOSE,
        'decision_interval_minutes': DECISION_INTERVAL_MINUTES, 'kalshi_max_staleness_seconds': KALSHI_MAX_STALENESS_SECONDS
    }
    overall_trades_log, overall_pnl_cents, overall_contracts_traded = [], 0, 0
    utils.BASE_PROJECT_DIR, utils.BINANCE_FLAT_DATA_DIR, utils.KALSHI_DATA_DIR = BASE_PROJECT_DIR, BINANCE_FLAT_DATA_DIR, KALSHI_DATA_ROOT_DIR

    def sort_key_func(sk_str):
        pi = utils.parse_kalshi_ticker_info(f"DUMMY-{sk_str.replace('_', '')}-T1")
        return pi['event_resolution_dt_utc'] if pi else dt.datetime.min.replace(tzinfo=timezone.utc)
    sorted_session_keys = sorted(grouped_sessions.groups.keys(), key=sort_key_func)
    
    # General log formatter (used by console and session files)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s')
    # Ensure console handler uses this formatter
    global console_handler_main # Reference the one set up globally
    if console_handler_main:
        console_handler_main.setFormatter(log_formatter)

    for session_key in sorted_session_keys:
        # Remove previous session's file handler, if any
        # This ensures only one file handler is active at a time for session-specific logs
        current_file_handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)]
        for h_file in current_file_handlers:
            h_file.close()
            logging.getLogger().removeHandler(h_file)

        # Add new file handler for the current session
        session_log_file_name = f"session_{session_key}_{MODEL_TYPE_TO_RUN}_{run_timestamp}.log"
        session_log_file_path = BACKTEST_LOGS_DIR / session_log_file_name
        
        file_handler_session = logging.FileHandler(str(session_log_file_path), mode='w') 
        file_handler_session.setFormatter(log_formatter)
        logging.getLogger().addHandler(file_handler_session) # Add to root logger
        
        logger.info(f"Processing session {session_key} (logs: {session_log_file_path.name})")

        session_markets_df = grouped_sessions.get_group(session_key)
        utils.clear_binance_cache(); utils.clear_kalshi_cache()
        if session_markets_df.empty: logger.info(f"No markets in session {session_key}. Skipping."); continue
        
        min_market_open_dt_session = session_markets_df['market_open_time_iso'].min()
        max_market_close_dt_session = session_markets_df['market_close_time_iso'].max()
        max_ta_lookback_minutes = 0 # Recalculate max lookback needed
        if hasattr(utils, 'BTC_MOMENTUM_WINDOWS') and utils.BTC_MOMENTUM_WINDOWS: max_ta_lookback_minutes = max(max_ta_lookback_minutes, max(utils.BTC_MOMENTUM_WINDOWS))
        max_ta_lookback_minutes = max(max_ta_lookback_minutes, getattr(utils, 'BTC_VOLATILITY_WINDOW', 0))
        if hasattr(utils, 'BTC_SMA_WINDOWS') and utils.BTC_SMA_WINDOWS: max_ta_lookback_minutes = max(max_ta_lookback_minutes, max(utils.BTC_SMA_WINDOWS))
        if hasattr(utils, 'BTC_EMA_WINDOWS') and utils.BTC_EMA_WINDOWS: max_ta_lookback_minutes = max(max_ta_lookback_minutes, max(utils.BTC_EMA_WINDOWS))
        max_ta_lookback_minutes = max(max_ta_lookback_minutes, getattr(utils, 'BTC_RSI_WINDOW', 0))
        max_ta_lookback_minutes = max(max_ta_lookback_minutes, 60) 
        buffer_minutes = 60 
        binance_hist_start_dt = min_market_open_dt_session - timedelta(minutes=max_ta_lookback_minutes + buffer_minutes) 
        binance_hist_end_dt = max_market_close_dt_session + timedelta(minutes=buffer_minutes) 
        logger.info(f"  Session {session_key}: Binance data {binance_hist_start_dt.strftime('%Y-%m-%d %H:%M')} to {binance_hist_end_dt.strftime('%Y-%m-%d %H:%M')}")
        
        binance_session_features_df = utils.load_and_prepare_binance_range_with_features(binance_hist_start_dt, binance_hist_end_dt)
        if binance_session_features_df is None or binance_session_features_df.empty:
            logger.critical(f"  Binance history load failed for session {session_key}. Skipping."); continue

        session_trades, session_pnl, session_contracts = run_hourly_session_backtest(
            session_markets_df.sort_values(by='market_close_time_iso'), binance_session_features_df, global_config_state 
        )
        overall_trades_log.extend(session_trades); overall_pnl_cents += session_pnl; overall_contracts_traded += session_contracts
        
        logger.info(f"--- Session {session_key} Summary ---")
        logger.info(f"  Decision Points: {len(session_trades)}")
        logger.info(f"  Executed Trades: {sum(1 for t in session_trades if t.get('num_contracts_sim',0) > 0)}")
        logger.info(f"  Contracts Traded: {session_contracts}")
        logger.info(f"  P&L: ${session_pnl/100:.2f}, Capital After: ${global_config_state['current_capital_usd']:.2f}")
        
        if file_handler_session: # Close and remove current session's file handler
            file_handler_session.close()
            logging.getLogger().removeHandler(file_handler_session)
            
    # Final summary to console (root logger should still have only the console handler active)
    logger.info(f"\n\n======= OVERALL HISTORICAL BACKTEST SUMMARY ({MODEL_TYPE_TO_RUN.upper()}) =======")
    logger.info(f"Period: {BACKTEST_START_DATE_STR} to {BACKTEST_END_DATE_STR}")
    logger.info(f"Initial Capital: ${INITIAL_CAPITAL_USD:.2f}, Final Capital: ${global_config_state['current_capital_usd']:.2f}")
    num_executed_trades_overall = sum(1 for t in overall_trades_log if t.get('num_contracts_sim', 0) > 0)
    logger.info(f"Total Decisions: {len(overall_trades_log)}, Total Executed Trades: {num_executed_trades_overall}, Total Contracts: {overall_contracts_traded}")
    logger.info(f"Total P&L: ${overall_pnl_cents/100:.2f}")
    if num_executed_trades_overall > 0: logger.info(f"Avg P&L/Executed Trade: ${overall_pnl_cents/num_executed_trades_overall/100:.4f}")
    if overall_contracts_traded > 0: logger.info(f"Avg P&L/Contract: ${overall_pnl_cents/overall_contracts_traded/100:.4f}")

    if overall_trades_log:
        overall_results_csv_filename = f"ALL_TRADES_{MODEL_TYPE_TO_RUN}_{run_timestamp}.csv"
        overall_results_csv_path = BACKTEST_LOGS_DIR.parent / overall_results_csv_filename 
        df_overall_results = pd.DataFrame(overall_trades_log)
        df_overall_results.to_csv(overall_results_csv_path, index=False)
        logger.info(f"Overall backtest trade results saved to: {overall_results_csv_path.resolve()}")
    
    logging.shutdown() # Clean up all logging resources

if __name__ == "__main__":
    if strategy._model is None or strategy._scaler is None or strategy._feature_order is None:
        logger.critical(f"Strategy for '{MODEL_TYPE_TO_RUN}' lacks artifacts. Aborting.")
    elif not MARKET_OUTCOMES_CSV_PATH:
        logger.critical("MARKET_OUTCOMES_CSV_PATH not set. Aborting.")
    else:
        logger.info(f"Starting Historical Backtest: {MODEL_TYPE_TO_RUN} Model, Kelly Sizing...")
        main_backtest_loop()