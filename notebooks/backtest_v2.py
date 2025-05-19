# backtest_v2.py
import pandas as pd
import os
from pathlib import Path
import datetime as dt
from datetime import timezone, timedelta
import logging
import numpy as np
import json

# --- Setup Logger EARLIER ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s')
logger_early_setup = logging.getLogger(__name__)

# --- Configuration ---
# Determine BASE_PROJECT_DIR
# This logic assumes `backtest_v2.py` is inside the `notebooks` directory,
# and that `notebooks` is a direct child of the project root.
try:
    script_path = Path(os.path.abspath(__file__)) # Full path to this script
    # If this script is /path/to/project_root/notebooks/backtest_v2.py
    # then script_path.parent is /path/to/project_root/notebooks/
    # and script_path.parent.parent is /path/to/project_root/
    if script_path.parent.name == "notebooks":
        BASE_PROJECT_DIR = script_path.parent.parent
    else:
        # Fallback if the script is not in a directory named 'notebooks'
        # This could happen if you run it from a different location or copied it.
        # Try to find a 'notebooks' dir in CWD or parent of CWD.
        cwd = Path.cwd()
        if (cwd / "notebooks").exists() and (cwd / "notebooks" / "kalshi_data").exists(): # Running from project root
            BASE_PROJECT_DIR = cwd
        elif cwd.name == "notebooks" and (cwd / "kalshi_data").exists(): # Running from notebooks dir
            BASE_PROJECT_DIR = cwd.parent
        else: # Best guess: current working directory's parent is project root
            logger_early_setup.warning(
                f"Could not reliably determine project root structure based on 'notebooks' directory. "
                f"Script path: {script_path}, CWD: {cwd}. "
                f"Assuming CWD's parent '{cwd.parent}' is project root or CWD is project root."
            )
            # Check if data dirs exist relative to cwd.parent, then cwd
            if (cwd.parent / "notebooks" / "kalshi_data").exists():
                 BASE_PROJECT_DIR = cwd.parent
            elif (cwd / "notebooks" / "kalshi_data").exists():
                 BASE_PROJECT_DIR = cwd
            else: # Final fallback
                 BASE_PROJECT_DIR = cwd


    logger_early_setup.info(f"Determined BASE_PROJECT_DIR: {BASE_PROJECT_DIR.resolve()}")
except Exception as path_e:
    logger_early_setup.error(f"Error determining BASE_PROJECT_DIR: {path_e}. Defaulting to CWD's parent.")
    BASE_PROJECT_DIR = Path.cwd().parent # A common fallback

# --- Define data and artifact directories RELATIVE TO BASE_PROJECT_DIR ---
# Your data structure is:
# BASE_PROJECT_DIR/notebooks/kalshi_data/
# BASE_PROJECT_DIR/notebooks/binance_data/
# BASE_PROJECT_DIR/notebooks/logs/ (new, will be created)
# BASE_PROJECT_DIR/notebooks/trained_models/

NOTEBOOKS_DIR = BASE_PROJECT_DIR / "notebooks" # Define notebooks directory

KALSHI_DATA_ROOT_DIR = NOTEBOOKS_DIR / "kalshi_data"
BINANCE_FLAT_DATA_DIR = NOTEBOOKS_DIR / "binance_data"
LOGS_DIR = NOTEBOOKS_DIR / "logs" # Logs will be inside notebooks/logs
MODEL_ARTIFACTS_DIR = NOTEBOOKS_DIR / "trained_models"

# --- Refined Logging Setup (AFTER paths are defined) ---
LOGS_DIR.mkdir(parents=True, exist_ok=True)
run_timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file_path = LOGS_DIR / f"backtest_v2_log_LR_{run_timestamp}.txt"

for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s',
    handlers=[logging.FileHandler(str(log_file_path)), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

logger.info(f"Final BASE_PROJECT_DIR: {BASE_PROJECT_DIR.resolve()}")
logger.info(f"Notebooks Dir: {NOTEBOOKS_DIR.resolve()}")
logger.info(f"Kalshi Data Dir: {KALSHI_DATA_ROOT_DIR.resolve()}")
logger.info(f"Binance Data Dir: {BINANCE_FLAT_DATA_DIR.resolve()}")
logger.info(f"Model Artifacts Dir: {MODEL_ARTIFACTS_DIR.resolve()}")
logger.info(f"Logs will be saved in: {LOGS_DIR.resolve()}")


# --- Import project modules AFTER path setup and initial logging ---
import utils
import sizing
import linreg_strategy_v2 as strategy

# --- Ensure strategy module can find its model artifacts ---
strategy.MODEL_DIR = MODEL_ARTIFACTS_DIR.resolve()

# Find the latest Kalshi NTM outcomes CSV
try:
    if not KALSHI_DATA_ROOT_DIR.exists():
        raise FileNotFoundError(f"Kalshi data root directory does not exist: {KALSHI_DATA_ROOT_DIR.resolve()}")
    outcomes_files = sorted(
        KALSHI_DATA_ROOT_DIR.glob("kalshi_btc_hourly_NTM_filtered_market_outcomes_*.csv"),
        key=os.path.getctime, reverse=True
    )
    if not outcomes_files:
        raise FileNotFoundError(f"No NTM outcomes CSV found in {KALSHI_DATA_ROOT_DIR.resolve()} matching pattern 'kalshi_btc_hourly_NTM_filtered_market_outcomes_*.csv'")
    MARKET_OUTCOMES_CSV_PATH = outcomes_files[0]
    logger.info(f"Backtest: Using market outcomes from {MARKET_OUTCOMES_CSV_PATH.resolve()}")
except FileNotFoundError as e:
    logger.critical(f"Backtest: CRITICAL - {e}. Market outcomes file is required to run the backtest.")
    MARKET_OUTCOMES_CSV_PATH = None
except Exception as e:
    logger.critical(f"Backtest: CRITICAL - Error finding outcomes CSV: {e}", exc_info=True)
    MARKET_OUTCOMES_CSV_PATH = None

# Backtest specific configurations
BACKTEST_START_DATE_STR = "2025-05-10"
BACKTEST_END_DATE_STR = "2025-05-15" # Adjust as per your available outcome data

TRADE_DECISION_OFFSET_MINUTES = 5
KALSHI_MAX_STALENESS_SECONDS = 120
TOTAL_CAPITAL_USD = 1000.0

# --- Sizing Configuration ---
sizing.MAX_CAPITAL_ALLOCATION_PER_TRADE_USD = 100.0
sizing.BASE_CAPITAL_ALLOCATION_PER_TRADE_USD = 20.0
sizing.MAX_CONTRACTS_PER_TRADE = 200

def map_prediction_to_sizing_score(prediction_usd_diff: float) -> float:
    if pd.isna(prediction_usd_diff): return 0.0
    abs_diff = abs(prediction_usd_diff)
    min_conf_diff_threshold = strategy.PRED_THRESHOLD_BUY_YES if hasattr(strategy, 'PRED_THRESHOLD_BUY_YES') else 50.0
    if abs_diff < min_conf_diff_threshold: return 0.0
    min_score_scaling_diff = min_conf_diff_threshold
    max_score_scaling_diff = 1000.0
    if abs_diff >= max_score_scaling_diff: return 5.0
    scale_range = max_score_scaling_diff - min_score_scaling_diff
    if scale_range <= 0: return 1.0
    score = 1.0 + 4.0 * ((abs_diff - min_score_scaling_diff) / scale_range)
    return max(1.0, min(score, 5.0))

sizing.MODEL_SCORE_THRESHOLD_BUY_YES = 1.0
sizing.MODEL_SCORE_THRESHOLD_BUY_NO = 1.0
sizing.PRACTICAL_MAX_MODEL_SCORE_FOR_SCALING = 5.0

# --- Main Backtest Logic ---
def run_backtest():
    if not MARKET_OUTCOMES_CSV_PATH or not MARKET_OUTCOMES_CSV_PATH.exists():
        logger.critical("Market outcomes CSV not found or path not set. Aborting backtest.")
        return

    try:
        df_all_market_details = pd.read_csv(MARKET_OUTCOMES_CSV_PATH)
        df_all_market_details['event_resolution_time_iso'] = pd.to_datetime(df_all_market_details['event_resolution_time_iso'], utc=True)
        df_all_market_details['market_open_time_iso'] = pd.to_datetime(df_all_market_details['market_open_time_iso'], utc=True)
        df_all_market_details['market_close_time_iso'] = pd.to_datetime(df_all_market_details['market_close_time_iso'], utc=True)
        df_all_market_details.dropna(subset=['market_open_time_iso', 'market_close_time_iso', 'kalshi_strike_price', 'market_ticker', 'result'], inplace=True)
        logger.info(f"Backtest: Loaded {len(df_all_market_details)} market details from outcomes CSV.")
    except Exception as e:
        logger.critical(f"Backtest: CRITICAL - Error loading market outcomes: {e}. Abort.", exc_info=True)
        return

    backtest_start_dt = pd.to_datetime(BACKTEST_START_DATE_STR, utc=True).normalize()
    backtest_end_dt = (pd.to_datetime(BACKTEST_END_DATE_STR, utc=True) + timedelta(days=1)).normalize() - timedelta(seconds=1)
    
    markets_for_backtest = df_all_market_details[
        (df_all_market_details['market_close_time_iso'] >= backtest_start_dt) &
        (df_all_market_details['market_close_time_iso'] <= backtest_end_dt)
    ].copy()
    
    if markets_for_backtest.empty:
        logger.warning(f"No markets found for backtest period: {BACKTEST_START_DATE_STR} to {BACKTEST_END_DATE_STR}")
        return
        
    logger.info(f"Backtest: Processing {len(markets_for_backtest)} markets for period {BACKTEST_START_DATE_STR} to {BACKTEST_END_DATE_STR}.")
    markets_for_backtest.sort_values(by='market_close_time_iso', inplace=True)

    utils.clear_binance_cache()
    utils.clear_kalshi_cache()
    current_capital_usd = TOTAL_CAPITAL_USD
    
    total_pnl_cents = 0; trades_executed = 0; total_contracts_traded = 0

    if markets_for_backtest.empty:
        logger.warning("No markets to process, skipping Binance data load.")
        _binance_full_history_with_features = pd.DataFrame()
    else:
        max_lookback_needed = max(strategy.BTC_MOMENTUM_WINDOWS + [strategy.BTC_VOLATILITY_WINDOW] + strategy.BTC_SMA_WINDOWS + strategy.BTC_EMA_WINDOWS + [strategy.BTC_RSI_WINDOW], default=30)
        buffer_minutes = 60
        valid_open_times = markets_for_backtest['market_open_time_iso'].dropna()
        if valid_open_times.empty: logger.error("No valid market open times. Aborting."); return
        min_event_open_dt = valid_open_times.min()
        max_event_close_dt = markets_for_backtest['market_close_time_iso'].dropna().max()
        if pd.isna(min_event_open_dt) or pd.isna(max_event_close_dt): logger.error("Could not determine Binance data range. Aborting."); return
        binance_hist_start_dt = min_event_open_dt - timedelta(minutes=max_lookback_needed + buffer_minutes)
        binance_hist_end_dt = max_event_close_dt + timedelta(minutes=buffer_minutes)
        logger.info(f"Backtest: Pre-loading Binance data with features from {binance_hist_start_dt.strftime('%Y-%m-%d')} to {binance_hist_end_dt.strftime('%Y-%m-%d')}.")
        _binance_full_history_with_features = utils.load_and_prepare_binance_range_with_features(binance_hist_start_dt, binance_hist_end_dt)
        if _binance_full_history_with_features is None or _binance_full_history_with_features.empty:
            logger.critical("Failed to load Binance history for backtest. Aborting."); return

    for idx, market_row in markets_for_backtest.iterrows():
        market_ticker = market_row['market_ticker']
        kalshi_strike_price = market_row['kalshi_strike_price']
        market_close_dt_utc = market_row['market_close_time_iso']
        actual_market_result = market_row['result']
        # logger.info(f"--- Processing market: {market_ticker} (Closes UTC: {market_close_dt_utc.isoformat()}) ---")
        decision_point_dt_utc = market_close_dt_utc - timedelta(minutes=TRADE_DECISION_OFFSET_MINUTES)
        signal_ts_utc = int(decision_point_dt_utc.timestamp()) - 60
        parsed_kalshi_ticker = utils.parse_kalshi_ticker_info(market_ticker)
        if not parsed_kalshi_ticker: logger.warning(f"Could not parse {market_ticker}. Skipping."); continue
        kalshi_market_day_data_df = utils.load_kalshi_market_minute_data(market_ticker, parsed_kalshi_ticker['date_str'], parsed_kalshi_ticker['hour_str_EDT'])
        if kalshi_market_day_data_df is None: kalshi_market_day_data_df = pd.DataFrame()
        current_kalshi_prices = utils.get_kalshi_prices_at_decision(kalshi_market_day_data_df, signal_ts_utc, KALSHI_MAX_STALENESS_SECONDS)
        current_yes_bid, current_yes_ask = (current_kalshi_prices.get('yes_bid'), current_kalshi_prices.get('yes_ask')) if current_kalshi_prices else (None, None)
        btc_history_for_features = _binance_full_history_with_features[_binance_full_history_with_features.index <= signal_ts_utc].copy()
        if btc_history_for_features.empty or signal_ts_utc not in btc_history_for_features.index:
            logger.warning(f"Insufficient BTC history for {market_ticker} at {signal_ts_utc}. Skipping."); continue
        feature_vector_series = strategy.generate_live_features(btc_history_for_features, current_yes_bid, current_yes_ask, kalshi_market_day_data_df, kalshi_strike_price, decision_point_dt_utc, market_close_dt_utc)
        if feature_vector_series is None: logger.warning(f"Feature generation failed for {market_ticker}. Skipping."); continue
        predicted_btc_diff_from_strike = strategy.calculate_model_prediction(feature_vector_series)
        if predicted_btc_diff_from_strike is None: logger.warning(f"Model prediction failed for {market_ticker}. Skipping."); continue
        # logger.info(f"Market {market_ticker}: Strike={kalshi_strike_price:.2f}, Pred Diff: {predicted_btc_diff_from_strike:.2f}")
        trade_action, _ = strategy.get_trade_decision(predicted_btc_diff_from_strike)
        entry_price_cents = 0; contract_cost_cents = 0
        if trade_action == "BUY_YES":
            if pd.isna(current_yes_ask) or not (0 < current_yes_ask < 100): trade_action = None
            else: entry_price_cents = current_yes_ask; contract_cost_cents = current_yes_ask
        elif trade_action == "BUY_NO":
            if pd.isna(current_yes_bid) or not (0 < current_yes_bid < 100) or pd.isna(current_yes_ask) or not (0 < current_yes_ask < 100) or current_yes_bid >= current_yes_ask : trade_action = None
            else: entry_price_cents = 100 - current_yes_bid; contract_cost_cents = 100 - current_yes_bid
        if trade_action:
            sizing_score_input = map_prediction_to_sizing_score(predicted_btc_diff_from_strike)
            position_size = sizing.calculate_position_size_capital_based(sizing_score_input, contract_cost_cents, current_capital_usd)
            if position_size == 0: trade_action = None
            if trade_action and position_size > 0:
                total_trade_cost_usd = (contract_cost_cents * position_size) / 100.0
                if total_trade_cost_usd > current_capital_usd:
                    position_size = int(np.floor((current_capital_usd * 100) / contract_cost_cents))
                    if position_size == 0: continue
                pnl_per_contract = 0
                if actual_market_result == 'yes': pnl_per_contract = (100 - entry_price_cents) if trade_action == "BUY_YES" else (-entry_price_cents)
                elif actual_market_result == 'no': pnl_per_contract = (-entry_price_cents) if trade_action == "BUY_YES" else (100 - entry_price_cents)
                else: logger.warning(f"Unknown outcome '{actual_market_result}' for {market_ticker}. Skip P&L."); continue
                pnl_this_trade_total_cents = pnl_per_contract * position_size
                total_pnl_cents += pnl_this_trade_total_cents; current_capital_usd += (pnl_this_trade_total_cents / 100.0)
                trades_executed += 1; total_contracts_traded += position_size
                logger.info(f"TRADE: {market_ticker} | PredD:{predicted_btc_diff_from_strike:.2f} | {trade_action} x{position_size}@{entry_price_cents:.0f}c (B:{current_yes_bid if pd.notna(current_yes_bid) else 'N/A'},A:{current_yes_ask if pd.notna(current_yes_ask) else 'N/A'}) | Outcome:{actual_market_result.upper()} | PNL:{pnl_this_trade_total_cents:.0f}c | Cap:${current_capital_usd:.2f}")
        # else: logger.info(f"NO TRADE for {market_ticker} (PredD:{predicted_btc_diff_from_strike if predicted_btc_diff_from_strike is not None else 'N/A'}, Act:{trade_action})")

    logger.info(f"\n--- Backtest Summary (New LR Model v2) ---")
    logger.info(f"Period: {BACKTEST_START_DATE_STR} to {BACKTEST_END_DATE_STR}")
    logger.info(f"Initial Capital: ${TOTAL_CAPITAL_USD:.2f}, Final Capital: ${current_capital_usd:.2f}")
    logger.info(f"Trades: {trades_executed}, Contracts: {total_contracts_traded}, Total P&L: ${total_pnl_cents/100:.2f}")
    if trades_executed > 0:
        avg_pnl_per_trade_str = f"${total_pnl_cents/trades_executed/100:.4f}"
        
        # Calculate avg_pnl_per_contract separately to handle the conditional formatting
        if total_contracts_traded > 0:
            avg_pnl_per_contract_val = total_pnl_cents / total_contracts_traded / 100
            avg_pnl_per_contract_str = f"${avg_pnl_per_contract_val:.4f}"
        else:
            avg_pnl_per_contract_str = "$0.0000" # Or "N/A"
            
        logger.info(f"Avg P&L/Trade: {avg_pnl_per_trade_str}, Avg P&L/Contract: {avg_pnl_per_contract_str}")
    logger.info(f"Log file: {log_file_path.resolve()}")

if __name__ == "__main__":
    # Check if strategy loaded its artifacts correctly. strategy.MODEL_DIR should be set.
    if strategy._scaler is None or strategy._model_intercept is None:
        logger.critical("Model artifacts (scaler/params) not loaded in strategy module. Backtest cannot run. "
                        f"Check strategy.MODEL_DIR ({strategy.MODEL_DIR if hasattr(strategy, 'MODEL_DIR') else 'Not Set'}) "
                        "and paths in linreg_strategy_v2.py.")
    elif not MARKET_OUTCOMES_CSV_PATH:
        logger.critical("MARKET_OUTCOMES_CSV_PATH is not set. Cannot run backtest.")
    else:
        utils.BASE_PROJECT_DIR = BASE_PROJECT_DIR
        utils.BINANCE_FLAT_DATA_DIR = BINANCE_FLAT_DATA_DIR
        utils.KALSHI_DATA_DIR = KALSHI_DATA_ROOT_DIR
        
        utils.BTC_MOMENTUM_WINDOWS = strategy.BTC_MOMENTUM_WINDOWS
        utils.BTC_VOLATILITY_WINDOW = strategy.BTC_VOLATILITY_WINDOW
        utils.BTC_SMA_WINDOWS = strategy.BTC_SMA_WINDOWS
        utils.BTC_EMA_WINDOWS = strategy.BTC_EMA_WINDOWS
        utils.BTC_RSI_WINDOW = strategy.BTC_RSI_WINDOW
        
        logger.info("Starting Backtest V2 with Trained Linear Regression Model...")
        run_backtest()