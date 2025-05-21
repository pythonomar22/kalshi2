# live_backtester/live_backtest_main.py
import pandas as pd
import numpy as np
from pathlib import Path
import datetime as dt
from datetime import timezone, timedelta
import logging
import time 
import os 

import sys
current_script_path = Path(__file__).resolve()
live_backtester_dir = current_script_path.parent
project_root_level1 = live_backtester_dir.parent 
project_root_level2 = project_root_level1.parent 

if str(project_root_level1) not in sys.path:
    sys.path.insert(0, str(project_root_level1))
if str(project_root_level2) not in sys.path:
    sys.path.insert(0, str(project_root_level2))
    
from live_backtester import live_utils
from live_backtester import live_strategy

try:
    from notebooks import sizing as historical_sizing
    logger_sizing = logging.getLogger("SizingModule") 
    historical_sizing.MAX_CAPITAL_ALLOCATION_PER_TRADE_USD = 100.0 
    historical_sizing.BASE_CAPITAL_ALLOCATION_PER_TRADE_USD = 5.0  
    historical_sizing.MAX_CONTRACTS_PER_TRADE = 50 
    historical_sizing.MODEL_SCORE_THRESHOLD_BUY_YES = 1.0 
    historical_sizing.MODEL_SCORE_THRESHOLD_BUY_NO = 1.0
    historical_sizing.PRACTICAL_MAX_MODEL_SCORE_FOR_SCALING = 5.0

    def map_prediction_to_sizing_score(prediction_usd_diff: float, PRED_THRESHOLD_BUY_YES, PRED_THRESHOLD_BUY_NO) -> float:
        if pd.isna(prediction_usd_diff): return 0.0
        abs_diff = abs(prediction_usd_diff)
        min_conf_diff_threshold_yes = PRED_THRESHOLD_BUY_YES
        min_conf_diff_threshold_no = abs(PRED_THRESHOLD_BUY_NO) 
        if prediction_usd_diff > 0 and abs_diff < min_conf_diff_threshold_yes: return 0.0
        if prediction_usd_diff < 0 and abs_diff < min_conf_diff_threshold_no: return 0.0
        min_score_scaling_diff = min_conf_diff_threshold_yes if prediction_usd_diff > 0 else min_conf_diff_threshold_no
        max_score_scaling_diff = 1000.0 
        if abs_diff >= max_score_scaling_diff: return 5.0
        scale_range = max_score_scaling_diff - min_score_scaling_diff
        if scale_range <= 0: return 1.0 
        score = 1.0 + 4.0 * ((abs_diff - min_score_scaling_diff) / scale_range)
        return max(1.0, min(score, 5.0))
except ImportError:
    logger.warning("Could not import 'sizing' module. Sizing will be fixed at 1 contract.")
    historical_sizing = None
    map_prediction_to_sizing_score = None

# --- Configuration ---
RUN_CONFIG = {
    "kalshi_yyyymmddhh_session_key": "25MAY2018", # Key to match Kalshi CSVs and Binance map
    "decision_interval_minutes": 1, 
    "trade_decision_offset_minutes_from_market_close": 5, 
    "kalshi_quote_max_staleness_seconds": 5, 
    "binance_history_needed_minutes": 120,
    "initial_capital_usd": 500.0 
}
# Path to the outcomes CSV you generated
LIVE_SESSIONS_OUTCOMES_CSV_PATH = live_backtester_dir / "live_sessions_market_outcomes.csv" 

# --- Path Definitions ---
LIVE_DATA_ROOT_DIR = project_root_level1 
KALSHI_LIVE_LOGS_DIR = LIVE_DATA_ROOT_DIR / "market_data_logs"
BINANCE_LIVE_LOGS_DIR = LIVE_DATA_ROOT_DIR / "binance_market_data_logs"
BACKTEST_LOGS_DIR = live_backtester_dir / "logs"
BACKTEST_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging Setup ---
run_timestamp_str = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file_name_template = "live_backtest_{kalshi_session}_{run_ts}.log" 
log_file_path = Path(".") 

for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()] 
)
logger = logging.getLogger("LiveBacktestMain")

# --- Main Backtest Function ---
def run_live_data_backtest(current_run_config, df_all_market_outcomes: pd.DataFrame):
    kalshi_session_key = current_run_config['kalshi_yyyymmddhh_session_key']
    binance_csv_name_for_session = current_run_config['binance_csv_name']

    logger.info(f"--- Starting Live Data Backtest for Session: {kalshi_session_key} ---")
    logger.info(f"Using Binance data: {binance_csv_name_for_session}")
    logger.info(f"Model PRED_THRESHOLD_BUY_YES: {live_strategy.PRED_THRESHOLD_BUY_YES}")
    logger.info(f"Model PRED_THRESHOLD_BUY_NO: {live_strategy.PRED_THRESHOLD_BUY_NO}")
    logger.info(f"Initial Capital: ${current_run_config['initial_capital_usd']:.2f}")

    binance_live_csv_path = BINANCE_LIVE_LOGS_DIR / binance_csv_name_for_session
    df_binance_all_closed_klines = live_utils.load_live_binance_csv_and_extract_closed_klines(binance_live_csv_path)
    if df_binance_all_closed_klines is None or df_binance_all_closed_klines.empty:
        logger.critical(f"Failed to load or process Binance live data from {binance_live_csv_path}. Aborting.")
        return
    logger.info(f"Loaded {len(df_binance_all_closed_klines)} closed 1-min Binance klines for session.")
    
    df_binance_with_ta = live_utils.calculate_live_ta_features(
        df_binance_all_closed_klines,
        feature_order_list=live_strategy._feature_order
    )
    logger.info(f"Calculated TA features for {len(df_binance_with_ta)} Binance klines for session.")
    
    # Filter outcomes for the current session's markets
    # Assuming kalshi_yyyymmddhh_session_key (e.g., "25MAY2019") is part of the market_ticker
    relevant_outcomes = df_all_market_outcomes[
        df_all_market_outcomes['market_ticker'].str.contains(f"-{kalshi_session_key}-T")
    ].set_index('market_ticker') # Index by ticker for easy lookup

    if relevant_outcomes.empty:
        logger.warning(f"No market outcomes found in the outcomes CSV for session key {kalshi_session_key}. P&L cannot be calculated accurately.")
        # Decide if you want to proceed without P&L or abort
        # For now, we'll proceed and trades will have NaN P&L if outcome is missing
    else:
        logger.info(f"Loaded {len(relevant_outcomes)} market outcomes for session {kalshi_session_key}.")


    kalshi_session_pattern = f"KXBTCD-{kalshi_session_key}-T*.csv"
    kalshi_market_files = sorted(list(KALSHI_LIVE_LOGS_DIR.glob(kalshi_session_pattern)))
    
    if not kalshi_market_files:
        logger.warning(f"No Kalshi market files found for session pattern {kalshi_session_pattern} in {KALSHI_LIVE_LOGS_DIR}. Aborting.")
        return
    logger.info(f"Found {len(kalshi_market_files)} Kalshi market CSVs for session {kalshi_session_key}.")

    if not all([live_strategy._scaler, live_strategy._model_intercept is not None, 
                live_strategy._model_coefficients_dict, live_strategy._feature_order]):
        logger.critical("Model artifacts not loaded in live_strategy.py. Aborting backtest.")
        return
    logger.info("Live strategy model artifacts (scaler, params, feature_order) appear to be loaded.")

    session_trades_log = []
    current_capital_usd = current_run_config['initial_capital_usd']
    total_contracts_traded_session = 0

    for kalshi_csv_path in kalshi_market_files:
        market_ticker = kalshi_csv_path.stem 
        logger.info(f"\n--- Processing Kalshi Market: {market_ticker} ---")

        # Get actual market outcome for P&L
        actual_market_result_str = None
        market_outcome_details = None
        if market_ticker in relevant_outcomes.index:
            market_outcome_details = relevant_outcomes.loc[market_ticker]
            actual_market_result_str = market_outcome_details.get('result') # 'yes' or 'no'
            logger.info(f"  Actual historical outcome for {market_ticker}: {actual_market_result_str}")
        else:
            logger.warning(f"  No historical outcome found for {market_ticker} in outcomes CSV. P&L will be NaN.")


        df_kalshi_live_market = live_utils.load_live_kalshi_csv(kalshi_csv_path)
        if df_kalshi_live_market is None or df_kalshi_live_market.empty:
            logger.warning(f"Could not load or empty Kalshi data for {market_ticker}. Skipping.")
            continue

        # Use market close time from the loaded outcomes if available, otherwise parse from name
        # The one from outcomes CSV is more reliable as it's from API directly.
        if market_outcome_details is not None and pd.notna(market_outcome_details.get('close_time_iso')):
            kalshi_market_resolution_dt_utc = pd.to_datetime(market_outcome_details['close_time_iso'], utc=True)
        else: # Fallback to parsing from ticker name
            parsed_details = live_utils.parse_kalshi_ticker_details_from_name(market_ticker)
            if not parsed_details or not parsed_details.get("market_resolution_dt_utc"):
                 logger.warning(f"Could not determine market resolution time for {market_ticker} from name. Skipping.")
                 continue
            kalshi_market_resolution_dt_utc = parsed_details["market_resolution_dt_utc"]
        
        kalshi_strike_price = market_outcome_details.get('strike_price') if market_outcome_details is not None else \
                              (live_utils.parse_kalshi_ticker_details_from_name(market_ticker) or {}).get('strike_price')
        
        if kalshi_strike_price is None:
            logger.warning(f"Could not determine strike price for {market_ticker}. Skipping.")
            continue
            
        if df_kalshi_live_market.index.empty:
            logger.warning(f"Kalshi live data for {market_ticker} has no timestamps after processing. Skipping.")
            continue
            
        actual_data_start_dt_utc = df_kalshi_live_market.index.min()
        actual_data_end_dt_utc = df_kalshi_live_market.index.max()

        loop_start_dt_utc = actual_data_start_dt_utc.replace(second=0, microsecond=0)
        if actual_data_start_dt_utc.second > 0 or actual_data_start_dt_utc.microsecond > 0 :
             loop_start_dt_utc += timedelta(minutes=1) 

        theoretical_loop_end_dt_utc = (kalshi_market_resolution_dt_utc - 
                           timedelta(minutes=current_run_config["trade_decision_offset_minutes_from_market_close"])).replace(second=0, microsecond=0)
        effective_data_end_for_loop = actual_data_end_dt_utc.replace(second=0, microsecond=0)
        loop_end_dt_utc = min(theoretical_loop_end_dt_utc, effective_data_end_for_loop)

        logger.info(f"Market {market_ticker}: Strike={kalshi_strike_price}, Resolves At (UTC)={kalshi_market_resolution_dt_utc.isoformat()}")
        logger.info(f"  Recorded Kalshi data available from: {actual_data_start_dt_utc.isoformat()} to {actual_data_end_dt_utc.isoformat()}")
        logger.info(f"  Decision loop will run from: {loop_start_dt_utc.isoformat()} to {loop_end_dt_utc.isoformat()} (Interval: {current_run_config['decision_interval_minutes']} min)")

        if loop_start_dt_utc > loop_end_dt_utc:
            logger.info(f"  No valid decision window for market {market_ticker}. Skipping market.")
            continue
        
        market_trades_log_for_this_market = [] # Store trades for *this market* to calculate P&L at its end
        current_decision_dt_utc = loop_start_dt_utc
        
        while current_decision_dt_utc <= loop_end_dt_utc:
            # ... (rest of the decision loop: get Kalshi snapshot, Binance history, features, prediction) ...
            current_kalshi_snapshot = live_utils.get_kalshi_snapshot_at_decision(
                df_kalshi_live_market, current_decision_dt_utc,
                current_run_config["kalshi_quote_max_staleness_seconds"]
            )
            if not current_kalshi_snapshot:
                current_decision_dt_utc += timedelta(minutes=current_run_config["decision_interval_minutes"])
                continue

            end_of_binance_hist_for_features_ts_s = int((current_decision_dt_utc - timedelta(minutes=1)).replace(second=0, microsecond=0).timestamp())
            start_of_binance_hist_for_features_ts_s = end_of_binance_hist_for_features_ts_s - (current_run_config["binance_history_needed_minutes"] * 60)
            df_btc_history_for_features = df_binance_with_ta[
                (df_binance_with_ta.index >= start_of_binance_hist_for_features_ts_s) &
                (df_binance_with_ta.index <= end_of_binance_hist_for_features_ts_s)
            ].copy()

            if df_btc_history_for_features.empty or end_of_binance_hist_for_features_ts_s not in df_btc_history_for_features.index:
                current_decision_dt_utc += timedelta(minutes=current_run_config["decision_interval_minutes"])
                continue
            
            df_kalshi_mid_history = pd.DataFrame()
            if 'yes_bid_price_cents' in df_kalshi_live_market.columns and 'yes_ask_price_cents' in df_kalshi_live_market.columns:
                kalshi_history_subset = df_kalshi_live_market[df_kalshi_live_market.index <= current_decision_dt_utc].copy()
                kalshi_history_subset['kalshi_mid_price'] = (kalshi_history_subset['yes_bid_price_cents'] + kalshi_history_subset['yes_ask_price_cents']) / 2.0
                kalshi_history_subset.dropna(subset=['kalshi_mid_price'], inplace=True)
                if not kalshi_history_subset.empty and pd.api.types.is_datetime64_any_dtype(kalshi_history_subset.index):
                    df_kalshi_mid_history = kalshi_history_subset[['kalshi_mid_price']].resample('1T').last().ffill()
            
            feature_vector = live_strategy.generate_features_from_live_data(
                df_closed_btc_klines_history=df_btc_history_for_features,
                current_kalshi_snapshot=current_kalshi_snapshot,
                df_kalshi_mid_price_history=df_kalshi_mid_history, 
                kalshi_strike_price=kalshi_strike_price,
                decision_dt_utc=current_decision_dt_utc,
                kalshi_market_close_dt_utc=kalshi_market_resolution_dt_utc 
            )

            if feature_vector is None:
                current_decision_dt_utc += timedelta(minutes=current_run_config["decision_interval_minutes"])
                continue
                
            predicted_btc_diff = live_strategy.calculate_model_prediction(feature_vector)
            if predicted_btc_diff is None:
                current_decision_dt_utc += timedelta(minutes=current_run_config["decision_interval_minutes"])
                continue

            trade_action, _ = live_strategy.get_trade_decision(predicted_btc_diff)
            
            num_contracts = 0
            entry_price_cents = np.nan 
            contract_cost_cents = np.nan

            if trade_action == "BUY_YES" and pd.notna(current_kalshi_snapshot.get('yes_ask')):
                entry_price_cents = current_kalshi_snapshot['yes_ask']
                contract_cost_cents = entry_price_cents
            elif trade_action == "BUY_NO" and pd.notna(current_kalshi_snapshot.get('yes_bid')):
                entry_price_cents = 100 - current_kalshi_snapshot['yes_bid']
                contract_cost_cents = entry_price_cents
            
            if trade_action and pd.notna(contract_cost_cents) and 0 < contract_cost_cents < 100:
                if historical_sizing and map_prediction_to_sizing_score:
                    sizing_score = map_prediction_to_sizing_score(
                        predicted_btc_diff, 
                        live_strategy.PRED_THRESHOLD_BUY_YES,
                        live_strategy.PRED_THRESHOLD_BUY_NO
                    )
                    num_contracts = historical_sizing.calculate_position_size_capital_based(
                        sizing_score, contract_cost_cents, current_capital_usd
                    )
                else: 
                    num_contracts = 1 if contract_cost_cents > 0 else 0 
                
                if num_contracts > 0:
                    total_trade_cost_usd = (contract_cost_cents * num_contracts) / 100.0
                    if total_trade_cost_usd > current_capital_usd: 
                        num_contracts = 0 
                        # logger.info(f"[{current_decision_dt_utc.strftime('%H:%M:%S')}] {market_ticker.split('-T')[1]} | {trade_action} | Insufficient capital for {num_contracts} @ {contract_cost_cents:.0f}c. Needed ${total_trade_cost_usd:.2f}, Have ${current_capital_usd:.2f}")
                else: # num_contracts became 0 after sizing
                    trade_action = "NO_TRADE" # Override to NO_TRADE if size is 0
            else: 
                trade_action = "NO_TRADE" 
                num_contracts = 0

            # Log decision before P&L (P&L comes after market closes)
            log_entry_for_now = {
                "decision_timestamp_utc": current_decision_dt_utc.isoformat(),
                "market_ticker": market_ticker,
                "kalshi_strike_price": kalshi_strike_price,
                "predicted_btc_diff_from_strike": round(predicted_btc_diff, 2), 
                "trade_action": trade_action, # This might be NO_TRADE now if sizing was 0
                "num_contracts_sim": num_contracts,
                "kalshi_yes_bid_at_decision": current_kalshi_snapshot.get('yes_bid'),
                "kalshi_yes_ask_at_decision": current_kalshi_snapshot.get('yes_ask'),
                "simulated_entry_price_cents": entry_price_cents if pd.notna(entry_price_cents) else None,
                "pnl_cents": np.nan 
            }
            # Add to this market's specific log, P&L will be updated later
            market_trades_log_for_this_market.append(log_entry_for_now)


            log_msg_prefix = f"TICK: {current_decision_dt_utc.strftime('%H:%M:%S')} | {market_ticker.split('-T')[1]} | PredD:{predicted_btc_diff:.2f}"
            if trade_action != "NO_TRADE" and num_contracts > 0:
                logger.info(f"{log_msg_prefix} -> {trade_action} x{num_contracts} "
                            f"@ {entry_price_cents:.0f}c "
                            f"(B:{current_kalshi_snapshot.get('yes_bid')}, A:{current_kalshi_snapshot.get('yes_ask')}) | Cap Before: ${current_capital_usd:.2f}")
            elif trade_action != "NO_TRADE" and num_contracts == 0: 
                 logger.info(f"{log_msg_prefix} -> {trade_action} SIGNALED but NO CONTRACTS (Cost:{contract_cost_cents:.0f}c or Cap issue)")
            else: 
                reason = ""
                if live_strategy.PRED_THRESHOLD_BUY_NO <= predicted_btc_diff <= live_strategy.PRED_THRESHOLD_BUY_YES:
                    reason = f"score between NO_THRESH ({live_strategy.PRED_THRESHOLD_BUY_NO:.2f}) and YES_THRESH ({live_strategy.PRED_THRESHOLD_BUY_YES:.2f})"
                # else, it was a signal but sizing resulted in 0 contracts or invalid price
                logger.info(f"{log_msg_prefix} -> NO_TRADE (Reason: {reason if reason else 'Sizing/Price issue'})")

            current_decision_dt_utc += timedelta(minutes=current_run_config["decision_interval_minutes"])
        
        # --- After iterating through all decision points for THIS market, calculate PNL for its trades ---
        if market_trades_log_for_this_market and actual_market_result_str:
            for trade_idx, trade in enumerate(market_trades_log_for_this_market):
                if trade["trade_action"] != "NO_TRADE" and pd.notna(trade["simulated_entry_price_cents"]) and trade["num_contracts_sim"] > 0:
                    entry_p = trade["simulated_entry_price_cents"]
                    contracts = trade["num_contracts_sim"]
                    pnl_per_c = 0
                    if actual_market_result_str.lower() == "yes":
                        pnl_per_c = (100 - entry_p) if trade["trade_action"] == "BUY_YES" else (-entry_p)
                    elif actual_market_result_str.lower() == "no":
                        pnl_per_c = (-entry_p) if trade["trade_action"] == "BUY_YES" else (100 - entry_p)
                    
                    calculated_pnl_cents = pnl_per_c * contracts
                    market_trades_log_for_this_market[trade_idx]["pnl_cents"] = calculated_pnl_cents # Update the dict
                    
                    # Update capital *after* each trade is resolved for this market
                    current_capital_usd += (calculated_pnl_cents / 100.0)
                    total_contracts_traded_session += contracts
                    logger.info(f"  PNL_UPDATE for {market_ticker} | Trade @ {trade['decision_timestamp_utc'].split('T')[1][:8]}: "
                                f"{trade['trade_action']} x{contracts} -> {calculated_pnl_cents:.0f}c. New Cap: ${current_capital_usd:.2f}")
        elif market_trades_log_for_this_market and not actual_market_result_str:
             logger.warning(f"  PNL for {market_ticker} trades could not be calculated (missing actual_market_result_str).")
            
        session_trades_log.extend(market_trades_log_for_this_market)


    if session_trades_log:
        df_trades = pd.DataFrame(session_trades_log)
        results_csv_path = BACKTEST_LOGS_DIR / f"live_backtest_results_{kalshi_session_key}_{run_timestamp_str}.csv"
        df_trades.to_csv(results_csv_path, index=False)
        logger.info(f"Live backtest results saved to: {results_csv_path}")
        
        # Executed trades are those with num_contracts_sim > 0
        executed_trades_df = df_trades[df_trades['num_contracts_sim'] > 0]
        num_executed_trades = len(executed_trades_df)
        total_pnl_session_cents = executed_trades_df['pnl_cents'].sum() # Sum PNL only for executed trades
        
        logger.info(f"\n--- Live Data Backtest Summary for Session: {kalshi_session_key} ---")
        logger.info(f"Total decision points evaluated across all markets in session: {len(df_trades)}")
        logger.info(f"Total trades EXECUTED (contracts > 0): {num_executed_trades}")
        if num_executed_trades > 0:
            logger.info(f"  BUY_YES executed: {len(executed_trades_df[executed_trades_df['trade_action'] == 'BUY_YES'])}")
            logger.info(f"  BUY_NO executed: {len(executed_trades_df[executed_trades_df['trade_action'] == 'BUY_NO'])}")
            logger.info(f"Total Contracts Traded in session: {total_contracts_traded_session}") # This was already cumulative
            logger.info(f"Initial Capital: ${current_run_config['initial_capital_usd']:.2f}, Final Capital: ${current_capital_usd:.2f}")
            logger.info(f"Total P&L for Session (from actual outcomes): ${total_pnl_session_cents/100:.2f}")
            if total_contracts_traded_session > 0 :
                 logger.info(f"Avg P&L/Contract (from actual outcomes): ${total_pnl_session_cents/total_contracts_traded_session/100:.4f}")
    else:
        logger.info("No trade decisions were logged during the backtest for this session.")
    
if __name__ == "__main__":
    logger.info("Live Data Backtester Initializing...")
    
    session_to_binance_file_map = {
        "25MAY1920": "btcusdt_kline_1m.csv",
        "25MAY2015": "btcusdt_2kline_1m.csv",
        "25MAY2016": "btcusdt_3kline_1m.csv",
        "25MAY2017": "btcusdt_4kline_1m.csv",
        "25MAY2018": "btcusdt_5kline_1m.csv",
        "25MAY2019": "btcusdt_6kline_1m.csv",
    }
    
    session_key_to_run = "25MAY1920" 

    _current_run_config = RUN_CONFIG.copy() 

    if session_key_to_run in session_to_binance_file_map:
        _current_run_config["kalshi_yyyymmddhh_session_key"] = session_key_to_run # Use the correct key
        _current_run_config["binance_csv_name"] = session_to_binance_file_map[session_key_to_run]
        
        _log_file_name = log_file_name_template.format(
            kalshi_session=_current_run_config['kalshi_yyyymmddhh_session_key'],
            run_ts=run_timestamp_str
        )
        current_log_file_path = BACKTEST_LOGS_DIR / _log_file_name
        
        root_logger = logging.getLogger() 
        file_handler_global_ref = None 

        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                if handler.baseFilename != str(current_log_file_path):
                    handler.close()
                    root_logger.removeHandler(handler)
        
        already_has_handler_for_this_file = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == str(current_log_file_path) 
            for h in root_logger.handlers
        )

        if not already_has_handler_for_this_file:
            file_handler = logging.FileHandler(str(current_log_file_path))
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            file_handler_global_ref = file_handler 
            logger.info(f"Logging for this session will be in: {current_log_file_path}")
        else:
            logger.info(f"File handler for {current_log_file_path} already exists. Not adding duplicate.")

        # Load the outcomes CSV
        if not LIVE_SESSIONS_OUTCOMES_CSV_PATH.exists():
            logger.critical(f"Live session outcomes CSV not found at {LIVE_SESSIONS_OUTCOMES_CSV_PATH}. P&L will be inaccurate. Please run `fetch_live_session_outcomes.py`.")
            df_all_market_outcomes_loaded = pd.DataFrame() # Empty df
        else:
            try:
                df_all_market_outcomes_loaded = pd.read_csv(LIVE_SESSIONS_OUTCOMES_CSV_PATH)
                logger.info(f"Loaded {len(df_all_market_outcomes_loaded)} market outcomes from {LIVE_SESSIONS_OUTCOMES_CSV_PATH}")
            except Exception as e:
                logger.critical(f"Error loading outcomes CSV {LIVE_SESSIONS_OUTCOMES_CSV_PATH}: {e}", exc_info=True)
                df_all_market_outcomes_loaded = pd.DataFrame()


        try:
            run_live_data_backtest(
                current_run_config=_current_run_config,
                df_all_market_outcomes=df_all_market_outcomes_loaded
            )
        except Exception as e:
            logger.critical("Unhandled exception during backtest execution:", exc_info=True)
        finally:
            logger.info(f"Full log file for this run: {current_log_file_path}")
            if file_handler_global_ref and file_handler_global_ref in root_logger.handlers: 
                file_handler_global_ref.close()
                root_logger.removeHandler(file_handler_global_ref)
    else:
        logger.error(f"Session key '{session_key_to_run}' not found in session_to_binance_file_map. Cannot run backtest.")
        logger.error(f"Available session keys: {list(session_to_binance_file_map.keys())}")