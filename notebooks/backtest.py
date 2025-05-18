# backtest.py
import pandas as pd
import os
import datetime as dt
from datetime import timezone 
import logging
import numpy as np

import utils 
import sizing 

# --- Configuration ---
BASE_PROJECT_DIR = utils.BASE_PROJECT_DIR
KALSHI_ORGANIZED_DATA_DIR = os.path.join(BASE_PROJECT_DIR, "organized_market_data")
MARKET_OUTCOMES_CSV_PATH = os.path.join(BASE_PROJECT_DIR, "market_candlestick_data/kalshi_btc_hourly_market_outcomes.csv") 

BACKTEST_DATE_YYMMMDD = "25MAY15"
TRADE_DECISION_OFFSET_MINUTES = 5   
KALSHI_MAX_STALENESS_SECONDS = 240  
NTM_PERCENTAGE_THRESHOLD = 0.025    

# --- Simulated Linear Regression Model Parameters ---
# These would normally be loaded from a trained model file
MODEL_PARAMS = {
    'intercept': 0.0,
    'coef_btc_price_change_1m': 0.05,    # Positive short-term momentum = slightly bullish
    'coef_btc_price_change_5m': 0.1,     # Positive mid-term momentum = more bullish
    'coef_btc_price_change_15m': 0.02,   # Longer-term momentum, less weight
    'coef_btc_volatility_15m': -0.01,  # Higher volatility might slightly decrease confidence (negative weight)
    'coef_distance_to_strike': 0.0001  # If current price is far above strike, more bullish for YES
                                      # This needs careful scaling or normalization in a real model
}
# Thresholds for taking action based on model score
MODEL_SCORE_THRESHOLD_BUY_YES = 0.2  # Lowered threshold a bit
MODEL_SCORE_THRESHOLD_BUY_NO = -0.2 # Lowered threshold a bit

# --- Sizing Parameters (can be moved to sizing.py or config file later) ---
sizing.MODEL_SCORE_THRESHOLD_BUY_YES = MODEL_SCORE_THRESHOLD_BUY_YES # Make available to sizing module
sizing.MODEL_SCORE_THRESHOLD_BUY_NO = MODEL_SCORE_THRESHOLD_BUY_NO   # Make available to sizing module
sizing.MIN_MODEL_SCORE_FOR_SCALING = 0.5 # Start scaling if abs(score) > 0.5
sizing.CONFIDENCE_SCALING_FACTOR = 1.0   # For every 1 point of score above min_scaling, add 1 contract
sizing.BASE_POSITION_SIZE = 1
sizing.MAX_POSITION_SIZE = 3 

# --- Logging Setup ---
LOGS_DIR = os.path.join(BASE_PROJECT_DIR, "logs") 
os.makedirs(LOGS_DIR, exist_ok=True)
log_file_path = os.path.join(LOGS_DIR, f"backtest_log_ActualLR_{BACKTEST_DATE_YYMMMDD}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s',
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Main Backtest Logic ---
def run_backtest():
    utils.clear_binance_cache() 

    logger.info(f"Starting backtest with Actual LR model & Sizing for date: {BACKTEST_DATE_YYMMMDD}")
    total_pnl_cents = 0; trades_executed = 0; markets_processed_total = 0
    markets_considered_for_trade = 0 
    markets_skipped_no_btc_data = 0; markets_skipped_no_kalshi_data = 0
    markets_skipped_relevance = 0; markets_skipped_wide_spread = 0
    markets_skipped_no_model_signal = 0
    total_contracts_traded = 0

    primary_backtest_date_dt = dt.datetime.strptime(f"20{BACKTEST_DATE_YYMMMDD}", "%Y%b%d")
    primary_date_nodash = primary_backtest_date_dt.strftime("%Y-%m-%d")
    if utils.load_binance_data_for_date(primary_date_nodash) is None: 
        logger.critical(f"CRITICAL: Primary Binance data load failed. Abort."); return
    try:
        market_outcomes_df = pd.read_csv(MARKET_OUTCOMES_CSV_PATH).set_index('market_ticker')
        logger.info(f"Loaded {len(market_outcomes_df)} market outcomes.")
    except Exception as e: logger.critical(f"CRITICAL: Market outcomes load failed: {e}. Abort."); return

    kalshi_date_dir_path = os.path.join(KALSHI_ORGANIZED_DATA_DIR, BACKTEST_DATE_YYMMMDD)
    if not os.path.isdir(kalshi_date_dir_path): logger.error(f"Kalshi data dir not found: {kalshi_date_dir_path}"); return

    for hour_edt_str_dir_name in sorted(os.listdir(kalshi_date_dir_path)):
        hour_dir_full_path = os.path.join(kalshi_date_dir_path, hour_edt_str_dir_name)
        if not os.path.isdir(hour_dir_full_path): continue
        logger.info(f"--- Processing Event Resolution Hour (EDT): {hour_edt_str_dir_name} ---")

        for market_filename in sorted(os.listdir(hour_dir_full_path)):
            if not market_filename.endswith(".csv"): continue
            market_ticker = market_filename[:-4]; markets_processed_total += 1
            logger.debug(f"Processing market file: {market_ticker}")
            
            parsed_ticker_info = utils.parse_kalshi_ticker_info(market_ticker)
            if not parsed_ticker_info: logger.warning(f"Parse failed for {market_ticker}. Skip."); continue

            if parsed_ticker_info["hour_str_EDT"] != hour_edt_str_dir_name.zfill(2):
                logger.error(f"Mismatch! Folder hour {hour_edt_str_dir_name} vs Ticker EDT hour {parsed_ticker_info['hour_str_EDT']} for {market_ticker}. Skipping.")
                continue

            kalshi_strike_price = parsed_ticker_info["strike_price"]
            event_resolution_dt_utc = parsed_ticker_info["event_resolution_dt_utc"] 
            kalshi_market_close_ts = int(event_resolution_dt_utc.timestamp())
            decision_timestamp_s = kalshi_market_close_ts - (TRADE_DECISION_OFFSET_MINUTES * 60)
            btc_signal_timestamp_s = decision_timestamp_s - 60 

            btc_features = utils.get_btc_features_for_signal(btc_signal_timestamp_s)
            if btc_features is None: # get_btc_features_for_signal now checks for NaNs internally
                logger.warning(f"No/incomplete BTC features for {market_ticker} at {dt.datetime.fromtimestamp(btc_signal_timestamp_s, tz=timezone.utc).isoformat()}. Skip.")
                markets_skipped_no_btc_data +=1; continue
            
            btc_price_at_signal_time = btc_features['btc_price']
            
            lower_bound_strike = btc_price_at_signal_time * (1 - NTM_PERCENTAGE_THRESHOLD)
            upper_bound_strike = btc_price_at_signal_time * (1 + NTM_PERCENTAGE_THRESHOLD)
            if not (lower_bound_strike <= kalshi_strike_price <= upper_bound_strike):
                logger.debug(f"Market {market_ticker} strike {kalshi_strike_price:.2f} (BTC: {btc_price_at_signal_time:.2f}) out of NTM range. Skip.")
                markets_skipped_relevance += 1; continue
            
            markets_considered_for_trade += 1
            kalshi_market_filepath = os.path.join(hour_dir_full_path, market_filename)
            kalshi_df = utils.load_kalshi_market_data(kalshi_market_filepath)
            if kalshi_df is None or kalshi_df.empty:
                logger.warning(f"No Kalshi data in file for {market_ticker}. Skip.")
                markets_skipped_no_kalshi_data +=1; continue
            
            kalshi_prices = utils.get_kalshi_prices_at_decision(kalshi_df, decision_timestamp_s, KALSHI_MAX_STALENESS_SECONDS)
            if kalshi_prices is None or pd.isna(kalshi_prices['yes_ask']) or pd.isna(kalshi_prices['yes_bid']):
                logger.warning(f"No valid Kalshi prices for {market_ticker} at decision {dt.datetime.fromtimestamp(decision_timestamp_s, tz=timezone.utc).isoformat()}. Skip.")
                markets_skipped_no_kalshi_data +=1; continue
            
            yes_ask_price, yes_bid_price = kalshi_prices['yes_ask'], kalshi_prices['yes_bid']
            if not (0 < yes_ask_price < 100 and 0 < yes_bid_price < 100 and yes_bid_price < yes_ask_price):
                logger.info(f"Market {market_ticker} invalid spread: Ask={yes_ask_price:.0f}, Bid={yes_bid_price:.0f}. Skip.")
                markets_skipped_wide_spread +=1; continue

            # --- Linear Regression Model Prediction ---
            # Feature: distance from strike (normalized or scaled might be better in real model)
            # Positive distance means BTC price is above Kalshi strike
            distance_to_strike = btc_price_at_signal_time - kalshi_strike_price 

            model_prediction_score = MODEL_PARAMS['intercept'] + \
                                     (MODEL_PARAMS['coef_btc_price_change_1m'] * btc_features['btc_price_change_1m']) + \
                                     (MODEL_PARAMS['coef_btc_price_change_5m'] * btc_features['btc_price_change_5m']) + \
                                     (MODEL_PARAMS['coef_btc_price_change_15m'] * btc_features['btc_price_change_15m']) + \
                                     (MODEL_PARAMS['coef_btc_volatility_15m'] * btc_features['btc_volatility_15m']) + \
                                     (MODEL_PARAMS['coef_distance_to_strike'] * distance_to_strike)
            
            feature_str = (f"BTC Last={btc_price_at_signal_time:.2f}, Chg1m={btc_features['btc_price_change_1m']:.2f}, "
                           f"Chg5m={btc_features['btc_price_change_5m']:.2f}, Chg15m={btc_features['btc_price_change_15m']:.2f}, "
                           f"Vol15m={btc_features['btc_volatility_15m']:.2f}, DistToStrike={distance_to_strike:.2f}")
            logger.debug(f"{market_ticker}: {feature_str}, ModelScore={model_prediction_score:.4f}")

            trade_action = None; entry_price = 0; position_size = 0
            if model_prediction_score > MODEL_SCORE_THRESHOLD_BUY_YES:
                trade_action = "BUY_YES"; entry_price = yes_ask_price
                position_size = sizing.calculate_position_size(model_prediction_score)
            elif model_prediction_score < MODEL_SCORE_THRESHOLD_BUY_NO:
                trade_action = "BUY_NO"; entry_price = 100 - yes_bid_price
                position_size = sizing.calculate_position_size(model_prediction_score) 
            else:
                markets_skipped_no_model_signal +=1

            if trade_action and position_size > 0 :
                try: market_outcome_result = market_outcomes_df.loc[market_ticker, 'result']
                except KeyError: logger.error(f"Outcome for {market_ticker} not found. Skip P&L."); continue
                
                pnl_per_contract = 0
                if market_outcome_result == 'yes': 
                    pnl_per_contract = (100 - entry_price) if trade_action == "BUY_YES" else (0 - entry_price)
                elif market_outcome_result == 'no': 
                    pnl_per_contract = (0 - entry_price) if trade_action == "BUY_YES" else (100 - entry_price)
                else: logger.warning(f"Unknown outcome '{market_outcome_result}' for {market_ticker}."); continue
                
                pnl_this_trade = pnl_per_contract * position_size
                total_pnl_cents += pnl_this_trade; trades_executed += 1
                total_contracts_traded += position_size
                logging.info(f"TRADE (LR): {market_ticker} | Score={model_prediction_score:.2f} | {feature_str} | Strike: {kalshi_strike_price:.2f} | "
                             f"{trade_action} x{position_size} @ {entry_price:.0f}c (A:{yes_ask_price:.0f} B:{yes_bid_price:.0f}) | Outcome: {market_outcome_result.upper()} | P&L: {pnl_this_trade:.0f}c")
    
    # (Summary logging remains the same)
    logger.info(f"\n--- Backtest Summary (Actual LR Model with Sizing) for Date: {BACKTEST_DATE_YYMMMDD} ---")
    logger.info(f"Total Kalshi market files found: {markets_processed_total}")
    logger.info(f"Markets skipped - Not NTM: {markets_skipped_relevance}")
    logger.info(f"Markets considered for trading (NTM): {markets_considered_for_trade}")
    logger.info(f"  Of NTM, skipped - No BTC data/features: {markets_skipped_no_btc_data}")
    logger.info(f"  Of NTM, skipped - No/Stale Kalshi data: {markets_skipped_no_kalshi_data}")
    logger.info(f"  Of NTM, skipped - Wide Spread: {markets_skipped_wide_spread}")
    logger.info(f"  Of NTM, skipped - No Model Signal (score too low/neutral): {markets_skipped_no_model_signal}")
    logger.info(f"Total trades executed: {trades_executed}")
    logger.info(f"Total contracts traded: {total_contracts_traded}")
    logger.info(f"Total P&L: {total_pnl_cents:.2f} cents (${total_pnl_cents/100:.2f})")
    if trades_executed > 0:
        logger.info(f"Average P&L per trade: {total_pnl_cents/trades_executed:.2f} cents")
        if total_contracts_traded > 0: # Avoid division by zero
            logger.info(f"Average P&L per contract: {total_pnl_cents/total_contracts_traded:.2f} cents")

    print(f"\n--- Backtest Summary (Actual LR Model with Sizing) for Date: {BACKTEST_DATE_YYMMMDD} ---")
    print(f"Total Kalshi market files found: {markets_processed_total}")
    print(f"Markets skipped (Not NTM): {markets_skipped_relevance}")
    print(f"Markets considered for trading (NTM): {markets_considered_for_trade}")
    print(f"  Skipped (No BTC Data/Features): {markets_skipped_no_btc_data}")
    print(f"  Skipped (No/Stale Kalshi Data): {markets_skipped_no_kalshi_data}")
    print(f"  Skipped (Wide Spread): {markets_skipped_wide_spread}")
    print(f"  Skipped (No Model Signal): {markets_skipped_no_model_signal}")
    print(f"Total Trades Executed: {trades_executed}")
    print(f"Total Contracts Traded: {total_contracts_traded}")
    print(f"Total P&L: ${total_pnl_cents/100:.2f}")
    if trades_executed > 0:
        print(f"Avg P&L per Trade: ${total_pnl_cents/trades_executed/100:.4f}")
        if total_contracts_traded > 0:
             print(f"Avg P&L per Contract: ${total_pnl_cents/total_contracts_traded/100:.4f}")
    print(f"Detailed log saved to: {log_file_path}")

if __name__ == "__main__":
    # (BASE_PROJECT_DIR auto-detection and final logging remains the same)
    script_path = os.path.abspath(__file__)
    current_dir_name = os.path.basename(os.path.dirname(script_path))
    if current_dir_name == "notebooks": BASE_PROJECT_DIR = os.path.dirname(script_path)
    else:
        BASE_PROJECT_DIR = os.path.join(os.path.dirname(script_path), "notebooks")
        if not os.path.isdir(os.path.join(BASE_PROJECT_DIR, "organized_market_data")):
            BASE_PROJECT_DIR = os.path.dirname(script_path) 
            logger.warning(f"Could not find 'notebooks/organized_market_data'. Assuming BASE_PROJECT_DIR is script directory: {BASE_PROJECT_DIR}")

    utils.BASE_PROJECT_DIR = BASE_PROJECT_DIR # Ensure utils uses the same base
    utils.BINANCE_DATA_PATH_TEMPLATE = os.path.join(BASE_PROJECT_DIR, "binance_data/BTCUSDT-1m-{date_nodash}/BTCUSDT-1m-{date_nodash}.csv")
    KALSHI_ORGANIZED_DATA_DIR = os.path.join(BASE_PROJECT_DIR, "organized_market_data")
    MARKET_OUTCOMES_CSV_PATH = os.path.join(BASE_PROJECT_DIR, "market_candlestick_data/kalshi_btc_hourly_market_outcomes.csv")
    LOGS_DIR = os.path.join(BASE_PROJECT_DIR, "logs"); os.makedirs(LOGS_DIR, exist_ok=True)
    log_file_path = os.path.join(LOGS_DIR, f"backtest_log_ActualLR_Sizing_{BACKTEST_DATE_YYMMMDD}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s',handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])
    logger.info(f"Final BASE_PROJECT_DIR: {BASE_PROJECT_DIR}")
    logger.info(f"Final Binance data template: {utils.BINANCE_DATA_PATH_TEMPLATE}")
    logger.info(f"Final Kalshi organized data: {KALSHI_ORGANIZED_DATA_DIR}")
    logger.info(f"Final Market outcomes CSV: {MARKET_OUTCOMES_CSV_PATH}")
    logger.info(f"Final Log file: {log_file_path}")
    
    run_backtest()