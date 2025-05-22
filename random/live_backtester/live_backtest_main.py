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

# --- Path Setup for Imports ---
current_script_path = Path(__file__).resolve()
live_backtester_dir = current_script_path.parent
project_root_level1 = live_backtester_dir.parent 
project_root_level2 = project_root_level1.parent 

if str(project_root_level1) not in sys.path: sys.path.insert(0, str(project_root_level1))
if str(project_root_level2) not in sys.path: sys.path.insert(0, str(project_root_level2))
if str(project_root_level2 / "notebooks") not in sys.path: sys.path.insert(0, str(project_root_level2 / "notebooks"))

from live_backtester import live_utils
from live_backtester import live_strategy as classifier_live_strategy

try:
    from notebooks import sizing as main_sizing_module
    logger_sizing_init = logging.getLogger("SizingModuleInit")
except ImportError:
    logging.getLogger("SizingModuleInit").critical("Could not import 'main_sizing_module' from notebooks. Kelly Sizing will fail.")
    main_sizing_module = None

# --- Configuration ---
RUN_CONFIG = {
    "decision_interval_minutes": 1,
    "trade_decision_offset_minutes_from_market_close": 5, # Must match feature_engineering
    "kalshi_quote_max_staleness_seconds_for_execution": 5, 
    "binance_history_needed_minutes": 120, 
    "initial_capital_usd": 500.0 
}
LIVE_SESSIONS_OUTCOMES_CSV_PATH = live_backtester_dir / "live_sessions_market_outcomes.csv"

LIVE_DATA_ROOT_DIR = project_root_level1
KALSHI_LIVE_LOGS_DIR = LIVE_DATA_ROOT_DIR / "market_data_logs"
BINANCE_LIVE_LOGS_DIR = LIVE_DATA_ROOT_DIR / "binance_market_data_logs"

# --- CHOOSE MODEL TYPE TO BACKTEST ---
MODEL_TYPE_TO_RUN = "random_forest" # "logistic_regression" or "random_forest"

if MODEL_TYPE_TO_RUN == "logistic_regression":
    MODEL_ARTIFACTS_SUBDIR_NAME = "logreg"
    BACKTEST_LOG_SUBDIR_NAME = "logs_logreg_kelly_live" # Changed suffix
    LOG_FILE_NAME_PREFIX = "live_bktst_LogRegKelly"
    classifier_live_strategy.MIN_MODEL_PROB_FOR_CONSIDERATION = 0.60 
    classifier_live_strategy.EDGE_THRESHOLD_FOR_TRADE = 0.15 
    if main_sizing_module:
        main_sizing_module.KELLY_FRACTION = 0.05 
        main_sizing_module.MAX_CAPITAL_ALLOCATION_PERCENTAGE_OF_TOTAL = 0.05
        main_sizing_module.MAX_CONTRACTS_PER_TRADE = 50
elif MODEL_TYPE_TO_RUN == "random_forest":
    MODEL_ARTIFACTS_SUBDIR_NAME = "rf"
    BACKTEST_LOG_SUBDIR_NAME = "logs_rf_kelly_live" # Changed suffix
    LOG_FILE_NAME_PREFIX = "live_bktst_RFKelly"
    # These are crucial and should be tuned based on the new model's characteristics
    # Start with values that might have worked for the historical backtest or slightly adjusted
    classifier_live_strategy.MIN_MODEL_PROB_FOR_CONSIDERATION = 0.55 # Example for RF
    classifier_live_strategy.EDGE_THRESHOLD_FOR_TRADE = 0.05     # Example for RF
    if main_sizing_module:
        main_sizing_module.KELLY_FRACTION = 0.05 # Example, might need to be more conservative initially
        main_sizing_module.MAX_CAPITAL_ALLOCATION_PERCENTAGE_OF_TOTAL = 0.05 # Example
        main_sizing_module.MAX_CONTRACTS_PER_TRADE = 50 # Example
else:
    raise ValueError(f"Unsupported MODEL_TYPE_TO_RUN: {MODEL_TYPE_TO_RUN}")

BACKTEST_LOGS_DIR = live_backtester_dir / BACKTEST_LOG_SUBDIR_NAME
BACKTEST_LOGS_DIR.mkdir(parents=True, exist_ok=True)

run_timestamp_str = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file_name_template = LOG_FILE_NAME_PREFIX + "_{kalshi_session}_{run_ts}.log"

for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(f"LiveBktst_{MODEL_TYPE_TO_RUN}") 

MODEL_ARTIFACTS_DIR_FOR_STRATEGY = project_root_level2 / "notebooks" / "trained_models" / MODEL_ARTIFACTS_SUBDIR_NAME
if not classifier_live_strategy.load_classifier_artifacts(MODEL_ARTIFACTS_DIR_FOR_STRATEGY):
    logger.critical(f"CRITICAL: Failed to load {MODEL_TYPE_TO_RUN} artifacts from {MODEL_ARTIFACTS_DIR_FOR_STRATEGY}. Aborting.")
    exit()
else:
    logger.info(f"Successfully loaded {MODEL_TYPE_TO_RUN} model artifacts for live strategy.")


def run_live_data_backtest(current_run_config_session, df_all_market_outcomes: pd.DataFrame, global_capital_state: dict):
    kalshi_session_key = current_run_config_session['kalshi_yyyymmddhh_session_key']
    binance_csv_name_for_session = current_run_config_session['binance_csv_name']

    logger.info(f"--- Starting Live Data Backtest ({MODEL_TYPE_TO_RUN}) for Session: {kalshi_session_key} ---")
    logger.info(f"Using Binance data: {binance_csv_name_for_session}")
    logger.info(f"Strategy Params: MinProb={classifier_live_strategy.MIN_MODEL_PROB_FOR_CONSIDERATION}, Edge={classifier_live_strategy.EDGE_THRESHOLD_FOR_TRADE}")
    if main_sizing_module:
        logger.info(f"Sizing Params: KellyFrac={main_sizing_module.KELLY_FRACTION}, MaxAllocPct={main_sizing_module.MAX_CAPITAL_ALLOCATION_PERCENTAGE_OF_TOTAL}, MaxContracts={main_sizing_module.MAX_CONTRACTS_PER_TRADE}")
    logger.info(f"Session Start Capital: ${global_capital_state['current_capital_usd']:.2f}")

    binance_live_csv_path = BINANCE_LIVE_LOGS_DIR / binance_csv_name_for_session
    df_binance_all_closed_klines = live_utils.load_live_binance_csv_and_extract_closed_klines(binance_live_csv_path)
    if df_binance_all_closed_klines is None or df_binance_all_closed_klines.empty:
        logger.critical(f"Binance live data failed for {binance_live_csv_path}. Aborting session {kalshi_session_key}.")
        return [], 0, 0

    df_binance_with_ta = live_utils.calculate_live_ta_features(
        df_binance_all_closed_klines,
        feature_order_list=classifier_live_strategy._feature_order
    )
    logger.info(f"Calculated TA features for {len(df_binance_with_ta)} Binance klines for session {kalshi_session_key}.")

    relevant_outcomes = df_all_market_outcomes[
        df_all_market_outcomes['market_ticker'].str.contains(f"-{kalshi_session_key}-T", na=False)
    ].set_index('market_ticker')

    kalshi_session_pattern = f"KXBTCD-{kalshi_session_key}-T*.csv"
    kalshi_market_files = sorted(list(KALSHI_LIVE_LOGS_DIR.glob(kalshi_session_pattern)))
    if not kalshi_market_files:
        logger.warning(f"No Kalshi market CSVs for session pattern {kalshi_session_pattern}. Skipping session.")
        return [], 0, 0

    session_trades_log, session_pnl_cents_total, session_contracts_traded_total = [], 0, 0

    for kalshi_csv_path in kalshi_market_files:
        market_ticker = kalshi_csv_path.stem
        logger.info(f"\n--- Processing Live Kalshi Market: {market_ticker} ---")

        actual_market_result_str, market_outcome_details = None, None
        if market_ticker in relevant_outcomes.index:
            market_outcome_details = relevant_outcomes.loc[market_ticker]
            actual_market_result_str = market_outcome_details.get('result')
        else: logger.warning(f"  No outcome found for {market_ticker} in outcomes CSV.")

        df_kalshi_live_market = live_utils.load_live_kalshi_csv(kalshi_csv_path)
        if df_kalshi_live_market is None or df_kalshi_live_market.empty:
            logger.warning(f"No Kalshi data in {kalshi_csv_path} for {market_ticker}. Skipping market."); continue

        kalshi_market_resolution_dt_utc = None
        if market_outcome_details is not None and pd.notna(market_outcome_details.get('close_time_iso')):
            kalshi_market_resolution_dt_utc = pd.to_datetime(market_outcome_details['close_time_iso'], utc=True)
        else:
            parsed_details = live_utils.parse_kalshi_ticker_details_from_name(market_ticker)
            if parsed_details and parsed_details.get("market_resolution_dt_utc"):
                kalshi_market_resolution_dt_utc = parsed_details["market_resolution_dt_utc"]
        if not kalshi_market_resolution_dt_utc:
            logger.warning(f"Cannot determine resolution time for {market_ticker}. Skipping."); continue

        kalshi_strike_price = (market_outcome_details.get('strike_price') if market_outcome_details is not None 
                              else (live_utils.parse_kalshi_ticker_details_from_name(market_ticker) or {}).get('strike_price'))
        if kalshi_strike_price is None: 
            logger.warning(f"Cannot determine strike price for {market_ticker}. Skipping."); continue
        
        if not isinstance(df_kalshi_live_market.index, pd.DatetimeIndex) or df_kalshi_live_market.index.tzinfo != timezone.utc:
            logger.warning(f"Kalshi data for {market_ticker} index is not a UTC DatetimeIndex. Skipping."); continue

        actual_data_start_dt_utc = df_kalshi_live_market.index.min()
        actual_data_end_dt_utc = df_kalshi_live_market.index.max()
        
        # Align loop start/end with decision interval and offset
        loop_start_dt_utc = actual_data_start_dt_utc.ceil(freq=f"{current_run_config_session['decision_interval_minutes']}T")
        theoretical_loop_end_dt_utc = (kalshi_market_resolution_dt_utc -
                           timedelta(minutes=current_run_config_session["trade_decision_offset_minutes_from_market_close"])).floor(freq='min')
        effective_data_end_for_loop = actual_data_end_dt_utc.floor(freq='min')
        loop_end_dt_utc = min(theoretical_loop_end_dt_utc, effective_data_end_for_loop)

        logger.info(f"  Market {market_ticker}: Strike={kalshi_strike_price}, Resolves={kalshi_market_resolution_dt_utc.isoformat()}")
        logger.info(f"  Kalshi Data (UTC): {actual_data_start_dt_utc.isoformat()} to {actual_data_end_dt_utc.isoformat()}")
        logger.info(f"  Decision Loop (UTC): {loop_start_dt_utc.isoformat()} to {loop_end_dt_utc.isoformat()}")

        if loop_start_dt_utc > loop_end_dt_utc: logger.info(f"  No valid decision window for {market_ticker}. Skipping."); continue

        current_decision_dt_utc = loop_start_dt_utc
        while current_decision_dt_utc <= loop_end_dt_utc:
            kalshi_snapshot_at_t = live_utils.get_kalshi_snapshot_at_decision(
                df_kalshi_live_market, current_decision_dt_utc,
                current_run_config_session["kalshi_quote_max_staleness_seconds_for_execution"]
            )
            cs_bid_at_t = kalshi_snapshot_at_t.get('yes_bid') if kalshi_snapshot_at_t else None
            cs_ask_at_t = kalshi_snapshot_at_t.get('yes_ask') if kalshi_snapshot_at_t else None

            signal_btc_features_ts_s = int((current_decision_dt_utc - timedelta(minutes=1)).replace(second=0, microsecond=0).timestamp())
            if signal_btc_features_ts_s not in df_binance_with_ta.index:
                current_decision_dt_utc += timedelta(minutes=current_run_config_session['decision_interval_minutes']); continue
            
            df_btc_history_for_features = df_binance_with_ta[df_binance_with_ta.index <= signal_btc_features_ts_s].copy()
            if df_btc_history_for_features.empty:
                current_decision_dt_utc += timedelta(minutes=current_run_config_session['decision_interval_minutes']); continue
            
            feature_vector = classifier_live_strategy.generate_features_from_live_data(
                df_closed_btc_klines_history=df_btc_history_for_features,
                df_kalshi_live_market_history=df_kalshi_live_market, 
                kalshi_strike_price=kalshi_strike_price,
                decision_dt_utc=current_decision_dt_utc, 
                kalshi_market_close_dt_utc=kalshi_market_resolution_dt_utc
            )
            if feature_vector is None:
                current_decision_dt_utc += timedelta(minutes=current_run_config_session["decision_interval_minutes"]); continue

            predicted_proba_yes = classifier_live_strategy.calculate_model_prediction_proba(feature_vector)
            if predicted_proba_yes is None:
                current_decision_dt_utc += timedelta(minutes=current_run_config_session["decision_interval_minutes"]); continue

            trade_action, model_prob_chosen_side, entry_price_chosen_side = \
                classifier_live_strategy.get_trade_decision(predicted_proba_yes, cs_bid_at_t, cs_ask_at_t)

            num_contracts, pnl_this_trade_cents = 0, np.nan
            if trade_action and main_sizing_module and pd.notna(model_prob_chosen_side) and pd.notna(entry_price_chosen_side) :
                num_contracts = main_sizing_module.calculate_kelly_position_size(
                    model_prob_win=model_prob_chosen_side, entry_price_cents=entry_price_chosen_side,
                    available_capital_usd=global_capital_state['current_capital_usd']
                )
                if num_contracts > 0:
                    if actual_market_result_str: 
                        pnl_per_contract = 0
                        if actual_market_result_str.lower() == "yes":
                            pnl_per_contract = (100 - entry_price_chosen_side) if trade_action == "BUY_YES" else (-entry_price_chosen_side)
                        elif actual_market_result_str.lower() == "no":
                            pnl_per_contract = (-entry_price_chosen_side) if trade_action == "BUY_YES" else (100 - entry_price_chosen_side)
                        pnl_this_trade_cents = pnl_per_contract * num_contracts
                        global_capital_state['current_capital_usd'] += (pnl_this_trade_cents / 100.0)
                        session_pnl_cents_total += pnl_this_trade_cents
                    session_contracts_traded_total += num_contracts
                    logger.info(f"TRADE: {market_ticker} @ {current_decision_dt_utc.strftime('%H:%M')} | P(YES):{predicted_proba_yes:.3f} P({trade_action.split('_')[1]}):{model_prob_chosen_side:.3f} | "
                                f"{trade_action} x{num_contracts}@{entry_price_chosen_side:.0f}c "
                                f"(Exec B:{cs_bid_at_t if cs_bid_at_t is not None else 'N/A'},Exec A:{cs_ask_at_t if cs_ask_at_t is not None else 'N/A'}) | Outcome:{actual_market_result_str.upper() if actual_market_result_str else 'N/A'} | PNL:{pnl_this_trade_cents:.0f}c | GlobalCap:${global_capital_state['current_capital_usd']:.2f}")
                else: trade_action = "NO_TRADE"

            if not trade_action or num_contracts == 0:
                logger.debug(f"NO_TRADE: {market_ticker} @ {current_decision_dt_utc.strftime('%H:%M')} | P(model_yes)={predicted_proba_yes:.3f} | Exec B:{cs_bid_at_t if cs_bid_at_t is not None else 'N/A'} Exec A:{cs_ask_at_t if cs_ask_at_t is not None else 'N/A'}")

            session_trades_log.append({
                "decision_timestamp_utc": current_decision_dt_utc.isoformat(), "market_ticker": market_ticker,
                "kalshi_strike_price": kalshi_strike_price, "predicted_proba_yes": round(predicted_proba_yes,4) if pd.notna(predicted_proba_yes) else None,
                "model_prob_chosen_side": round(model_prob_chosen_side,4) if pd.notna(model_prob_chosen_side) else None,
                "kalshi_yes_bid_at_decision_t": cs_bid_at_t, "kalshi_yes_ask_at_decision_t": cs_ask_at_t,
                "executed_trade_action": trade_action if num_contracts > 0 else "NO_TRADE",
                "num_contracts_sim": num_contracts, "simulated_entry_price_cents": entry_price_chosen_side if num_contracts > 0 else None,
                "pnl_cents": pnl_this_trade_cents, "actual_market_result": actual_market_result_str,
                "capital_after_trade": round(global_capital_state['current_capital_usd'],2)
            })
            current_decision_dt_utc += timedelta(minutes=current_run_config_session["decision_interval_minutes"])
    return session_trades_log, session_pnl_cents_total, session_contracts_traded_total

def main():
    logger.info(f"Live Data Backtester ({MODEL_TYPE_TO_RUN}) Initializing...")
    logger.info(f"Artifacts loaded from: {MODEL_ARTIFACTS_DIR_FOR_STRATEGY.resolve()}")
    logger.info(f"Live logs will be stored under: {BACKTEST_LOGS_DIR.resolve()}")

    session_to_binance_file_map = { 
        "25MAY1920": "btcusdt_kline_1m.csv", "25MAY2015": "btcusdt_2kline_1m.csv",
        "25MAY2016": "btcusdt_3kline_1m.csv", "25MAY2017": "btcusdt_4kline_1m.csv",
        "25MAY2018": "btcusdt_5kline_1m.csv", "25MAY2019": "btcusdt_6kline_1m.csv",
    }
    sessions_to_run_keys = ["25MAY1920", "25MAY2015", "25MAY2016", "25MAY2017", "25MAY2018", "25MAY2019"]
    # sessions_to_run_keys = ["25MAY1920"] # For single session testing

    if not LIVE_SESSIONS_OUTCOMES_CSV_PATH.exists():
        logger.critical(f"Live session outcomes CSV {LIVE_SESSIONS_OUTCOMES_CSV_PATH} not found. Run fetch.py. Aborting."); return
    df_all_market_outcomes_loaded = pd.read_csv(LIVE_SESSIONS_OUTCOMES_CSV_PATH)

    overall_trades_log_all_sessions, session_pnl_list_dollars = [], []
    global_capital_state = { 'current_capital_usd': RUN_CONFIG['initial_capital_usd'] }
    log_formatter_for_file = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s')

    for session_key_to_run in sessions_to_run_keys:
        if session_key_to_run not in session_to_binance_file_map:
            logger.error(f"Binance CSV for session '{session_key_to_run}' not mapped. Skipping."); continue

        current_run_config_session = RUN_CONFIG.copy()
        current_run_config_session["kalshi_yyyymmddhh_session_key"] = session_key_to_run
        current_run_config_session["binance_csv_name"] = session_to_binance_file_map[session_key_to_run]
        
        session_log_file_name = log_file_name_template.format(kalshi_session=session_key_to_run, run_ts=run_timestamp_str)
        current_session_log_path = BACKTEST_LOGS_DIR / session_log_file_name
        root_logger = logging.getLogger()
        active_file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
        for h in active_file_handlers: h.close(); root_logger.removeHandler(h)
        file_handler = logging.FileHandler(str(current_session_log_path)); file_handler.setFormatter(log_formatter_for_file)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging for session '{session_key_to_run}' to: {current_session_log_path.name}")

        session_trades, session_pnl_cents, session_contracts = run_live_data_backtest(
            current_run_config_session, df_all_market_outcomes_loaded, global_capital_state
        )
        if session_trades: overall_trades_log_all_sessions.extend(session_trades)
        session_pnl_dollars = session_pnl_cents / 100.0
        session_pnl_list_dollars.append(session_pnl_dollars)

        logger.info(f"--- Summary for Session: {session_key_to_run} ({MODEL_TYPE_TO_RUN}) ---")
        logger.info(f"  Decision Points: {len(session_trades) if session_trades else 0}")
        logger.info(f"  Executed Trades: {sum(1 for t in session_trades if t.get('num_contracts_sim',0) > 0) if session_trades else 0}")
        logger.info(f"  Contracts Traded: {session_contracts}")
        logger.info(f"  Session P&L: ${session_pnl_dollars:.2f}, Global Capital: ${global_capital_state['current_capital_usd']:.2f}")
        if file_handler: file_handler.close(); root_logger.removeHandler(file_handler)

    if not any(isinstance(h, logging.StreamHandler) for h in logging.root.handlers): 
        console_handler = logging.StreamHandler(); console_handler.setFormatter(log_formatter_for_file)
        logging.root.addHandler(console_handler)

    logger.info(f"\n\n======= OVERALL LIVE BACKTEST SUMMARY ({MODEL_TYPE_TO_RUN}) =======")
    logger.info(f"Sessions: {', '.join(sessions_to_run_keys)}")
    logger.info(f"Initial Capital: ${RUN_CONFIG['initial_capital_usd']:.2f}, Final Capital: ${global_capital_state['current_capital_usd']:.2f}")
    num_executed_trades_overall = sum(1 for trade in overall_trades_log_all_sessions if trade.get('num_contracts_sim', 0) > 0)
    total_pnl_overall_dollars = sum(session_pnl_list_dollars)
    total_contracts_overall = sum(t['num_contracts_sim'] for t in overall_trades_log_all_sessions if t.get('num_contracts_sim',0) > 0)
    logger.info(f"Total Decisions: {len(overall_trades_log_all_sessions)}, Executed Trades: {num_executed_trades_overall}, Contracts: {total_contracts_overall}")
    logger.info(f"Total P&L: ${total_pnl_overall_dollars:.2f}")
    if num_executed_trades_overall > 0: logger.info(f"Avg P&L/Trade: ${total_pnl_overall_dollars / num_executed_trades_overall:.4f}")
    if total_contracts_overall > 0: logger.info(f"Avg P&L/Contract: ${total_pnl_overall_dollars / total_contracts_overall:.4f}")

    if len(session_pnl_list_dollars) > 1 :
        s_returns = pd.Series(session_pnl_list_dollars); mean_s_ret, std_s_ret = s_returns.mean(), s_returns.std()
        if std_s_ret != 0 and not pd.isna(std_s_ret):
            sharpe = mean_s_ret / std_s_ret
            # Approx annualization, depends on how many independent sessions one might trade in a year
            sessions_per_year_est = 252 * 8 # (Trading days * active hours/day) - very rough
            ann_factor = np.sqrt(sessions_per_year_est / len(sessions_to_run_keys) if sessions_to_run_keys else 1)
            logger.info(f"Sharpe (session P&Ls, rf=0): {sharpe:.4f} (Raw), {sharpe * ann_factor:.4f} (Annualized Approx.)")
    else: logger.info("Sharpe Ratio: N/A (Not enough P&L data points)")

    if overall_trades_log_all_sessions:
        df_overall_results = pd.DataFrame(overall_trades_log_all_sessions)
        overall_results_csv_path = BACKTEST_LOGS_DIR.parent / f"{LOG_FILE_NAME_PREFIX}_overall_live_results_{run_timestamp_str}.csv" 
        df_overall_results.to_csv(overall_results_csv_path, index=False)
        logger.info(f"Overall live backtest results saved to: {overall_results_csv_path.resolve()}")
    logging.shutdown()

if __name__ == "__main__":
    if not main_sizing_module: logger.warning("Sizing module (main_sizing_module) not imported. Kelly Sizing might be unavailable.")
    if classifier_live_strategy._model is None or classifier_live_strategy._scaler is None or classifier_live_strategy._feature_order is None:
        logger.critical(f"{MODEL_TYPE_TO_RUN} model artifacts not loaded in live_strategy. Aborting.")
    else:
        main()