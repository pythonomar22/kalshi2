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
project_root_level1 = live_backtester_dir.parent # 'random' directory
project_root_level2 = project_root_level1.parent # Main project directory 'kalshi'

if str(project_root_level1) not in sys.path:
    sys.path.insert(0, str(project_root_level1))

if str(project_root_level2) not in sys.path:
    sys.path.insert(0, str(project_root_level2))
if str(project_root_level2 / "notebooks") not in sys.path:
    sys.path.insert(0, str(project_root_level2 / "notebooks"))

from live_backtester import live_utils
from live_backtester import live_strategy as classifier_live_strategy

try:
    from notebooks import sizing as main_sizing_module
    logger_sizing_init = logging.getLogger("SizingModuleInit")
    # === CONSERVATIVE SETTINGS ===
    main_sizing_module.KELLY_FRACTION = 0.05 # Example: Reduced from 0.1
    main_sizing_module.MAX_CAPITAL_ALLOCATION_PERCENTAGE_OF_TOTAL = 0.05 # Example: Reduced from 0.10
    main_sizing_module.MAX_CONTRACTS_PER_TRADE = 50
    logger_sizing_init.info("Sizing module imported and CONSERVATIVE Kelly params set for live backtest.")
except ImportError:
    logging.getLogger("SizingModuleInit").critical("Could not import 'main_sizing_module' from notebooks. Kelly Sizing will fail.")
    main_sizing_module = None

# --- Configuration ---
RUN_CONFIG = {
    "kalshi_yyyymmddhh_session_key": "25MAY1920",
    "decision_interval_minutes": 1,
    "trade_decision_offset_minutes_from_market_close": 5,
    "kalshi_quote_max_staleness_seconds": 5,
    "binance_history_needed_minutes": 120,
    "initial_capital_usd": 500.0 # Start with 500 for this test
}
LIVE_SESSIONS_OUTCOMES_CSV_PATH = live_backtester_dir / "live_sessions_market_outcomes.csv"

LIVE_DATA_ROOT_DIR = project_root_level1
KALSHI_LIVE_LOGS_DIR = LIVE_DATA_ROOT_DIR / "market_data_logs"
BINANCE_LIVE_LOGS_DIR = LIVE_DATA_ROOT_DIR / "binance_market_data_logs"
BACKTEST_LOGS_DIR = live_backtester_dir / "logs_classifier_kelly_conservative" # New log subdir
BACKTEST_LOGS_DIR.mkdir(parents=True, exist_ok=True)

run_timestamp_str = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file_name_template = "live_bktst_LogRegKellyConserv_{kalshi_session}_{run_ts}.log"

for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("LiveClassifyKellyConservBktst") # Updated logger name

MODEL_ARTIFACTS_DIR_FOR_STRATEGY = project_root_level2 / "notebooks" / "trained_models" / "logreg"
if not classifier_live_strategy.load_classifier_artifacts(MODEL_ARTIFACTS_DIR_FOR_STRATEGY):
    logger.critical("CRITICAL: Failed to load classifier artifacts in live_strategy. Aborting.")
    exit()

# === CONSERVATIVE STRATEGY SETTINGS ===
classifier_live_strategy.MIN_MODEL_PROB_FOR_CONSIDERATION = 0.60 # Was 0.55
classifier_live_strategy.EDGE_THRESHOLD_FOR_TRADE = 0.20     # Was 0.05

def run_live_data_backtest(current_run_config_session, df_all_market_outcomes: pd.DataFrame, global_capital_state: dict):
    kalshi_session_key = current_run_config_session['kalshi_yyyymmddhh_session_key']
    binance_csv_name_for_session = current_run_config_session['binance_csv_name']

    logger.info(f"--- Starting Live Data Backtest (Classifier + Kelly - Conservative) for Session: {kalshi_session_key} ---")
    logger.info(f"Using Binance data: {binance_csv_name_for_session}")
    logger.info(f"Strategy: Min Model Prob: {classifier_live_strategy.MIN_MODEL_PROB_FOR_CONSIDERATION}, Edge Threshold: {classifier_live_strategy.EDGE_THRESHOLD_FOR_TRADE}")
    logger.info(f"Sizing: Kelly Frac: {main_sizing_module.KELLY_FRACTION if main_sizing_module else 'N/A'}, Max Alloc Pct: {main_sizing_module.MAX_CAPITAL_ALLOCATION_PERCENTAGE_OF_TOTAL if main_sizing_module else 'N/A'}")
    logger.info(f"Session Start Capital: ${global_capital_state['current_capital_usd']:.2f}")

    binance_live_csv_path = BINANCE_LIVE_LOGS_DIR / binance_csv_name_for_session
    df_binance_all_closed_klines = live_utils.load_live_binance_csv_and_extract_closed_klines(binance_live_csv_path)
    if df_binance_all_closed_klines is None or df_binance_all_closed_klines.empty:
        logger.critical(f"Failed to load Binance live data from {binance_live_csv_path}. Aborting session.")
        return [], 0, 0

    df_binance_with_ta = live_utils.calculate_live_ta_features(
        df_binance_all_closed_klines,
        feature_order_list=classifier_live_strategy._feature_order
    )
    logger.info(f"Calculated TA features for {len(df_binance_with_ta)} Binance klines.")

    relevant_outcomes = df_all_market_outcomes[
        df_all_market_outcomes['market_ticker'].str.contains(f"-{kalshi_session_key}-T", na=False)
    ].set_index('market_ticker')
    if relevant_outcomes.empty:
        logger.warning(f"No market outcomes found for session key {kalshi_session_key}.")

    kalshi_session_pattern = f"KXBTCD-{kalshi_session_key}-T*.csv"
    kalshi_market_files = sorted(list(KALSHI_LIVE_LOGS_DIR.glob(kalshi_session_pattern)))

    if not kalshi_market_files:
        logger.warning(f"No Kalshi market files for session {kalshi_session_pattern}. Skipping session.")
        return [], 0, 0

    session_trades_log = []
    session_pnl_cents_total = 0
    session_contracts_traded_total = 0

    for kalshi_csv_path in kalshi_market_files:
        market_ticker = kalshi_csv_path.stem
        logger.info(f"\n--- Processing Kalshi Market: {market_ticker} ---")

        actual_market_result_str = None; market_outcome_details = None
        if market_ticker in relevant_outcomes.index:
            market_outcome_details = relevant_outcomes.loc[market_ticker]
            actual_market_result_str = market_outcome_details.get('result')
            logger.info(f"  Actual outcome: {actual_market_result_str}")
        else:
            logger.warning(f"  No outcome found for {market_ticker}. P&L will be NaN for this market.")

        df_kalshi_live_market = live_utils.load_live_kalshi_csv(kalshi_csv_path)
        if df_kalshi_live_market is None or df_kalshi_live_market.empty:
            logger.warning(f"No Kalshi data for {market_ticker}. Skipping.")
            continue

        if market_outcome_details is not None and pd.notna(market_outcome_details.get('close_time_iso')):
            kalshi_market_resolution_dt_utc = pd.to_datetime(market_outcome_details['close_time_iso'], utc=True)
        else:
            parsed_details = live_utils.parse_kalshi_ticker_details_from_name(market_ticker)
            if not parsed_details or not parsed_details.get("market_resolution_dt_utc"):
                 logger.warning(f"Cannot get resolution time for {market_ticker}. Skipping.")
                 continue
            kalshi_market_resolution_dt_utc = parsed_details["market_resolution_dt_utc"]

        kalshi_strike_price = market_outcome_details.get('strike_price') if market_outcome_details is not None else \
                              (live_utils.parse_kalshi_ticker_details_from_name(market_ticker) or {}).get('strike_price')
        if kalshi_strike_price is None:
            logger.warning(f"Cannot get strike price for {market_ticker}. Skipping.")
            continue

        if df_kalshi_live_market.index.empty or not isinstance(df_kalshi_live_market.index, pd.DatetimeIndex):
            logger.warning(f"Kalshi data for {market_ticker} has no valid UTC timestamps. Skipping.")
            continue

        actual_data_start_dt_utc = df_kalshi_live_market.index.min()
        actual_data_end_dt_utc = df_kalshi_live_market.index.max()
        loop_start_dt_utc = actual_data_start_dt_utc.ceil(freq='min')
        theoretical_loop_end_dt_utc = (kalshi_market_resolution_dt_utc -
                           timedelta(minutes=current_run_config_session["trade_decision_offset_minutes_from_market_close"])).floor(freq='min')
        effective_data_end_for_loop = actual_data_end_dt_utc.floor(freq='min')
        loop_end_dt_utc = min(theoretical_loop_end_dt_utc, effective_data_end_for_loop)

        logger.info(f"Market {market_ticker}: Strike={kalshi_strike_price}, Resolves (UTC)={kalshi_market_resolution_dt_utc.isoformat()}")
        logger.info(f"  Kalshi data: {actual_data_start_dt_utc.isoformat()} to {actual_data_end_dt_utc.isoformat()}")
        logger.info(f"  Decision loop: {loop_start_dt_utc.isoformat()} to {loop_end_dt_utc.isoformat()}")

        if loop_start_dt_utc > loop_end_dt_utc:
            logger.info(f"  No valid decision window for market {market_ticker}. Skipping.")
            continue

        current_decision_dt_utc = loop_start_dt_utc
        while current_decision_dt_utc <= loop_end_dt_utc:
            current_kalshi_snapshot = live_utils.get_kalshi_snapshot_at_decision(
                df_kalshi_live_market, current_decision_dt_utc,
                current_run_config_session["kalshi_quote_max_staleness_seconds"]
            )
            features_for_ts_utc = int((current_decision_dt_utc - timedelta(minutes=1)).timestamp())

            if features_for_ts_utc not in df_binance_with_ta.index:
                current_decision_dt_utc += timedelta(minutes=current_run_config_session['decision_interval_minutes'])
                continue

            df_btc_history_for_features = df_binance_with_ta[df_binance_with_ta.index <= features_for_ts_utc].copy()
            if df_btc_history_for_features.empty:
                current_decision_dt_utc += timedelta(minutes=current_run_config_session['decision_interval_minutes'])
                continue

            df_kalshi_mid_history_for_features = pd.DataFrame()
            if 'yes_bid_price_cents' in df_kalshi_live_market.columns and 'yes_ask_price_cents' in df_kalshi_live_market.columns:
                _hist = df_kalshi_live_market[df_kalshi_live_market.index <= current_decision_dt_utc].copy()
                if not _hist.empty:
                    _hist['kalshi_mid_price'] = (_hist['yes_bid_price_cents'] + _hist['yes_ask_price_cents']) / 2.0
                    _hist.dropna(subset=['kalshi_mid_price'], inplace=True)
                    if not _hist.empty:
                        df_kalshi_mid_history_for_features = _hist[['kalshi_mid_price']].resample('1T').last().ffill()

            feature_vector = classifier_live_strategy.generate_features_from_live_data(
                df_closed_btc_klines_history=df_btc_history_for_features,
                current_kalshi_snapshot=current_kalshi_snapshot,
                df_kalshi_mid_price_history=df_kalshi_mid_history_for_features,
                kalshi_strike_price=kalshi_strike_price,
                decision_dt_utc=current_decision_dt_utc,
                kalshi_market_close_dt_utc=kalshi_market_resolution_dt_utc
            )

            if feature_vector is None:
                current_decision_dt_utc += timedelta(minutes=current_run_config_session["decision_interval_minutes"])
                continue

            predicted_proba_yes = classifier_live_strategy.calculate_model_prediction_proba(feature_vector)
            if predicted_proba_yes is None:
                current_decision_dt_utc += timedelta(minutes=current_run_config_session["decision_interval_minutes"])
                continue

            cs_bid = current_kalshi_snapshot.get('yes_bid') if current_kalshi_snapshot else None
            cs_ask = current_kalshi_snapshot.get('yes_ask') if current_kalshi_snapshot else None

            trade_action, model_prob_chosen_side, entry_price_chosen_side = \
                classifier_live_strategy.get_trade_decision(predicted_proba_yes, cs_bid, cs_ask)

            num_contracts = 0; pnl_this_trade_cents = np.nan

            if trade_action and main_sizing_module and pd.notna(model_prob_chosen_side) and pd.notna(entry_price_chosen_side) :
                num_contracts = main_sizing_module.calculate_kelly_position_size(
                    model_prob_win=model_prob_chosen_side,
                    entry_price_cents=entry_price_chosen_side,
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
                                    f"(B:{cs_bid},A:{cs_ask}) | Outcome:{actual_market_result_str.upper()} | PNL:{pnl_this_trade_cents:.0f}c | GlobalCap:${global_capital_state['current_capital_usd']:.2f}")
                    else:
                         logger.info(f"TRADE_INTENT (No Outcome): {market_ticker} @ {current_decision_dt_utc.strftime('%H:%M')} | P(YES):{predicted_proba_yes:.3f} P({trade_action.split('_')[1]}):{model_prob_chosen_side:.3f} | "
                                    f"{trade_action} x{num_contracts}@{entry_price_chosen_side:.0f}c (B:{cs_bid},A:{cs_ask})")
                else:
                    trade_action = "NO_TRADE"

            if not trade_action or num_contracts == 0:
                logger.debug(f"NO_TRADE: {market_ticker} @ {current_decision_dt_utc.strftime('%H:%M')} | P(model_yes)={predicted_proba_yes:.3f} | Bid:{cs_bid} Ask:{cs_ask}")

            session_trades_log.append({
                "decision_timestamp_utc": current_decision_dt_utc.isoformat(), "market_ticker": market_ticker,
                "kalshi_strike_price": kalshi_strike_price, "predicted_proba_yes": round(predicted_proba_yes,4) if pd.notna(predicted_proba_yes) else None,
                "model_prob_chosen_side": round(model_prob_chosen_side,4) if pd.notna(model_prob_chosen_side) else None,
                "kalshi_yes_bid_at_decision": cs_bid, "kalshi_yes_ask_at_decision": cs_ask,
                "executed_trade_action": trade_action if num_contracts > 0 else "NO_TRADE",
                "num_contracts_sim": num_contracts, "simulated_entry_price_cents": entry_price_chosen_side if num_contracts > 0 else None,
                "pnl_cents": pnl_this_trade_cents, "actual_market_result": actual_market_result_str,
                "capital_after_trade": round(global_capital_state['current_capital_usd'],2)
            })
            current_decision_dt_utc += timedelta(minutes=current_run_config_session["decision_interval_minutes"])

    return session_trades_log, session_pnl_cents_total, session_contracts_traded_total


def main():
    logger.info("Live Data Backtester (Classifier + Kelly - Conservative) Initializing...")

    session_to_binance_file_map = {
        "25MAY1920": "btcusdt_kline_1m.csv", "25MAY2015": "btcusdt_2kline_1m.csv",
        "25MAY2016": "btcusdt_3kline_1m.csv", "25MAY2017": "btcusdt_4kline_1m.csv",
        "25MAY2018": "btcusdt_5kline_1m.csv", "25MAY2019": "btcusdt_6kline_1m.csv",
    }

    sessions_to_run_keys = ["25MAY1920", "25MAY2015", "25MAY2016", "25MAY2017", "25MAY2018", "25MAY2019"] # Run all
    # sessions_to_run_keys = ["25MAY2019"] # Example: run only one specific session for testing

    if not LIVE_SESSIONS_OUTCOMES_CSV_PATH.exists():
        logger.critical(f"Live session outcomes CSV not found at {LIVE_SESSIONS_OUTCOMES_CSV_PATH}. Aborting.")
        return
    df_all_market_outcomes_loaded = pd.read_csv(LIVE_SESSIONS_OUTCOMES_CSV_PATH)
    logger.info(f"Loaded {len(df_all_market_outcomes_loaded)} market outcomes from {LIVE_SESSIONS_OUTCOMES_CSV_PATH}")

    overall_trades_log_all_sessions = []
    session_pnl_list_dollars = [] # For Sharpe Ratio calculation

    global_capital_state = { 'current_capital_usd': RUN_CONFIG['initial_capital_usd'] }

    for session_key_to_run in sessions_to_run_keys:
        if session_key_to_run not in session_to_binance_file_map:
            logger.error(f"Session key '{session_key_to_run}' not in map. Skipping.")
            continue

        current_run_config_session = RUN_CONFIG.copy()
        current_run_config_session["kalshi_yyyymmddhh_session_key"] = session_key_to_run
        current_run_config_session["binance_csv_name"] = session_to_binance_file_map[session_key_to_run]
        # global_capital_state is passed by reference and updated within run_live_data_backtest

        session_log_file_name = log_file_name_template.format(
            kalshi_session=current_run_config_session['kalshi_yyyymmddhh_session_key'],
            run_ts=run_timestamp_str
        )
        current_session_log_path = BACKTEST_LOGS_DIR / session_log_file_name

        root_logger = logging.getLogger()
        active_file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
        for h in active_file_handlers: h.close(); root_logger.removeHandler(h)

        file_handler = logging.FileHandler(str(current_session_log_path))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging for session '{session_key_to_run}' will be in: {current_session_log_path}")

        session_trades, session_pnl_cents, session_contracts = run_live_data_backtest(
            current_run_config_session,
            df_all_market_outcomes_loaded,
            global_capital_state # Pass the mutable dict
        )
        if session_trades:
            overall_trades_log_all_sessions.extend(session_trades)
        
        session_pnl_dollars = session_pnl_cents / 100.0
        session_pnl_list_dollars.append(session_pnl_dollars) # Store session P&L in dollars

        logger.info(f"--- Summary for Session: {session_key_to_run} ---")
        executed_session_trades = sum(1 for t in session_trades if t.get('num_contracts_sim',0) > 0) if session_trades else 0
        logger.info(f"  Decision Points Logged: {len(session_trades) if session_trades else 0}")
        logger.info(f"  Executed Trades: {executed_session_trades}")
        logger.info(f"  Contracts Traded: {session_contracts}")
        logger.info(f"  Session P&L: ${session_pnl_dollars:.2f}")
        logger.info(f"  Global Capital after session: ${global_capital_state['current_capital_usd']:.2f}")

        if file_handler: file_handler.close(); root_logger.removeHandler(file_handler)

    if not any(isinstance(h, logging.StreamHandler) for h in logging.root.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter) # Use existing formatter from file_handler
        logging.root.addHandler(console_handler)

    logger.info(f"\n\n======= OVERALL LIVE BACKTEST SUMMARY (Classifier + Kelly - Conservative) =======")
    logger.info(f"Ran for sessions: {', '.join(sessions_to_run_keys)}")
    logger.info(f"Initial Capital: ${RUN_CONFIG['initial_capital_usd']:.2f}, Final Capital: ${global_capital_state['current_capital_usd']:.2f}")

    num_executed_trades_overall = sum(1 for trade in overall_trades_log_all_sessions if trade.get('num_contracts_sim', 0) > 0)
    total_pnl_overall_dollars = sum(session_pnl_list_dollars)
    total_contracts_overall = sum(t['num_contracts_sim'] for t in overall_trades_log_all_sessions if t.get('num_contracts_sim',0) > 0)

    logger.info(f"Total Decisions Logged: {len(overall_trades_log_all_sessions)}")
    logger.info(f"Total Trades Executed: {num_executed_trades_overall}, Total Contracts: {total_contracts_overall}")
    logger.info(f"Total P&L: ${total_pnl_overall_dollars:.2f}")

    if num_executed_trades_overall > 0:
        avg_pnl_trade = total_pnl_overall_dollars / num_executed_trades_overall
        logger.info(f"Avg P&L/Executed Trade: ${avg_pnl_trade:.4f}")
    if total_contracts_overall > 0:
        avg_pnl_contract = total_pnl_overall_dollars / total_contracts_overall
        logger.info(f"Avg P&L/Contract: ${avg_pnl_contract:.4f}")

    # Sharpe Ratio Calculation (using per-session P&L as returns)
    if len(session_pnl_list_dollars) > 1 : # Need at least 2 data points for std dev
        session_returns_series = pd.Series(session_pnl_list_dollars)
        mean_session_return = session_returns_series.mean()
        std_session_return = session_returns_series.std()
        if std_session_return != 0 and not pd.isna(std_session_return):
            sharpe_ratio_raw = mean_session_return / std_session_return
            # To annualize, need to know how many "sessions" make a year.
            # Assuming hourly sessions, and ~8 active trading hours a day, 252 days a year
            # This is a rough estimate and depends on actual market availability.
            sessions_per_year_approx = 8 * 252 
            annualization_factor = np.sqrt(sessions_per_year_approx / len(sessions_to_run_keys) if sessions_to_run_keys else 1) # Scale by number of sessions in sample
            sharpe_ratio_annualized = sharpe_ratio_raw * annualization_factor
            logger.info(f"Sharpe Ratio (based on session P&Ls, rf=0): {sharpe_ratio_raw:.4f} (Raw)")
            logger.info(f"Sharpe Ratio (Annualized Approx. @ {sessions_per_year_approx} sessions/year): {sharpe_ratio_annualized:.4f}")

        else:
            logger.info("Sharpe Ratio: N/A (Std Dev of session returns is 0 or NaN)")
    else:
        logger.info("Sharpe Ratio: N/A (Not enough session P&Ls to calculate)")


    if overall_trades_log_all_sessions:
        df_overall_results = pd.DataFrame(overall_trades_log_all_sessions)
        overall_results_csv_path = BACKTEST_LOGS_DIR.parent / f"live_bktst_overall_LogRegKellyConserv_{run_timestamp_str}.csv" # In parent of session logs
        df_overall_results.to_csv(overall_results_csv_path, index=False)
        logger.info(f"Overall live backtest trade results saved to: {overall_results_csv_path.resolve()}")

    logging.shutdown()


if __name__ == "__main__":
    if not main_sizing_module:
        logger.critical("Sizing module (with Kelly) could not be imported. Live backtest cannot proceed correctly.")
    elif classifier_live_strategy._model is None or classifier_live_strategy._scaler is None or classifier_live_strategy._feature_order is None:
        logger.critical("Classifier model artifacts not loaded in live_strategy module. Aborting.")
    else:
        main()