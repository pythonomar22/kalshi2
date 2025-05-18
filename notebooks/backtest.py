# backtest.py
import pandas as pd
import os
import datetime as dt
from datetime import timezone 
import logging
import numpy as np

import utils 
import sizing 
import linreg_strategy 

# --- Configuration ---
BASE_PROJECT_DIR = utils.BASE_PROJECT_DIR
KALSHI_ORGANIZED_DATA_DIR = os.path.join(BASE_PROJECT_DIR, "organized_market_data")
MARKET_OUTCOMES_CSV_PATH = os.path.join(BASE_PROJECT_DIR, "market_candlestick_data/kalshi_btc_hourly_market_outcomes.csv") 

BACKTEST_DATE_YYMMMDD = "25MAY15"
TRADE_DECISION_OFFSET_MINUTES = 5   
KALSHI_MAX_STALENESS_SECONDS = 240  
NTM_PERCENTAGE_THRESHOLD = 0.025    

TOTAL_CAPITAL_USD = 500.0 # Your total trading capital

# --- Pass model thresholds to sizing module ---
sizing.MODEL_SCORE_THRESHOLD_BUY_YES = linreg_strategy.MODEL_SCORE_THRESHOLD_BUY_YES
sizing.MODEL_SCORE_THRESHOLD_BUY_NO = linreg_strategy.MODEL_SCORE_THRESHOLD_BUY_NO
# --- Sizing specific parameters for the capital-based approach ---
sizing.MAX_CAPITAL_ALLOCATION_PER_TRADE_USD = 25.0 # e.g., max 5% of 500 USD
sizing.BASE_CAPITAL_ALLOCATION_PER_TRADE_USD = 5.0  # e.g., min 1% of 500 USD
sizing.MAX_CONTRACTS_PER_TRADE = 200 # An overall hard cap on contracts
sizing.PRACTICAL_MAX_MODEL_SCORE_FOR_SCALING = 5.0 # Tune this based on typical model score range

# --- Logging Setup ---
LOGS_DIR = os.path.join(BASE_PROJECT_DIR, "logs") 
os.makedirs(LOGS_DIR, exist_ok=True)
log_file_path = os.path.join(LOGS_DIR, f"backtest_log_LR_CapSizing_{BACKTEST_DATE_YYMMMDD}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
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
    current_capital_usd = TOTAL_CAPITAL_USD # Track capital over time

    logger.info(f"Starting backtest with LR model & Capital Sizing for date: {BACKTEST_DATE_YYMMMDD}. Initial Capital: ${current_capital_usd:.2f}")
    total_pnl_cents = 0; trades_executed = 0; markets_processed_total = 0
    markets_considered_for_trade = 0 
    markets_skipped_no_btc_data = 0; markets_skipped_no_kalshi_data = 0
    markets_skipped_relevance = 0; markets_skipped_wide_spread = 0
    markets_skipped_no_model_signal = 0; markets_skipped_insufficient_capital = 0
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
            # ... (parsing ticker info, NTM filter - same as before) ...
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
            if btc_features is None:
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

            model_prediction_score = linreg_strategy.calculate_model_score(btc_features, kalshi_strike_price)
            trade_action, _ = linreg_strategy.get_trade_decision(model_prediction_score)
            
            entry_price_cents = 0; position_size = 0; contract_cost_cents = 0
            if trade_action == "BUY_YES":
                contract_cost_cents = yes_ask_price
                entry_price_cents = yes_ask_price 
            elif trade_action == "BUY_NO":
                contract_cost_cents = 100 - yes_bid_price
                entry_price_cents = 100 - yes_bid_price # This is the cost of the "NO" contract
            
            if trade_action:
                position_size = sizing.calculate_position_size_capital_based(
                    model_score=model_prediction_score,
                    contract_cost_cents=contract_cost_cents,
                    available_capital_usd=current_capital_usd # Pass current capital
                )
                if position_size == 0: # Sizing function might return 0 if allocation is too small for even 1 contract
                    logger.info(f"Sizing for {market_ticker} resulted in 0 contracts. Skipping trade.")
                    trade_action = None # Effectively no trade
                    markets_skipped_insufficient_capital +=1 # Or a more specific counter

            if trade_action and position_size > 0 :
                # Check if capital is sufficient for this sized trade
                total_trade_cost_cents = contract_cost_cents * position_size
                if (total_trade_cost_cents / 100) > current_capital_usd:
                    logger.warning(f"INSUFFICIENT CAPITAL for {market_ticker}: Need ${total_trade_cost_cents/100:.2f}, Have ${current_capital_usd:.2f}. Skipping trade.")
                    markets_skipped_insufficient_capital += 1
                    continue # Skip this trade

                try: market_outcome_result = market_outcomes_df.loc[market_ticker, 'result']
                except KeyError: logger.error(f"Outcome for {market_ticker} not found. Skip P&L."); continue
                
                pnl_per_contract = 0
                if market_outcome_result == 'yes': 
                    pnl_per_contract = (100 - entry_price_cents) if trade_action == "BUY_YES" else (0 - entry_price_cents)
                elif market_outcome_result == 'no': 
                    pnl_per_contract = (0 - entry_price_cents) if trade_action == "BUY_YES" else (100 - entry_price_cents)
                else: logger.warning(f"Unknown outcome '{market_outcome_result}' for {market_ticker}."); continue
                
                pnl_this_trade_total_cents = pnl_per_contract * position_size
                total_pnl_cents += pnl_this_trade_total_cents
                current_capital_usd += (pnl_this_trade_total_cents / 100.0) # Update capital
                trades_executed += 1
                total_contracts_traded += position_size
                
                feature_str = (f"BTC Last={btc_features['btc_price']:.2f}, Chg1m={btc_features['btc_price_change_1m']:.2f}, "
                               f"Chg5m={btc_features['btc_price_change_5m']:.2f}, Chg15m={btc_features['btc_price_change_15m']:.2f}, "
                               f"Vol15m={btc_features['btc_volatility_15m']:.2f}, DistToStrike={(btc_features['btc_price'] - kalshi_strike_price):.2f}")
                logging.info(f"TRADE ({os.path.basename(linreg_strategy.__file__)}): {market_ticker} | Score={model_prediction_score:.2f} | {feature_str} | Strike: {kalshi_strike_price:.2f} | "
                             f"{trade_action} x{position_size} @ {entry_price_cents:.0f}c (A:{yes_ask_price:.0f} B:{yes_bid_price:.0f}) | Outcome: {market_outcome_result.upper()} | P&L: {pnl_this_trade_total_cents:.0f}c | Capital: ${current_capital_usd:.2f}")
            elif trade_action and position_size == 0: # If trade was signaled but sizing made it 0
                 markets_skipped_no_model_signal +=1 # Or could be a new category like "skipped_min_size_not_met"
            elif not trade_action: # Model did not give a strong enough signal
                 markets_skipped_no_model_signal +=1


    logger.info(f"\n--- Backtest Summary ({os.path.basename(linreg_strategy.__file__)} with Capital Sizing) for Date: {BACKTEST_DATE_YYMMMDD} ---")
    logger.info(f"Initial Capital: ${TOTAL_CAPITAL_USD:.2f}")
    logger.info(f"Final Capital: ${current_capital_usd:.2f}")
    logger.info(f"Total Kalshi market files found: {markets_processed_total}")
    logger.info(f"Markets skipped - Not NTM: {markets_skipped_relevance}")
    logger.info(f"Markets considered for trading (NTM): {markets_considered_for_trade}")
    logger.info(f"  Of NTM, skipped - No BTC data/features: {markets_skipped_no_btc_data}")
    logger.info(f"  Of NTM, skipped - No/Stale Kalshi data: {markets_skipped_no_kalshi_data}")
    logger.info(f"  Of NTM, skipped - Wide Spread: {markets_skipped_wide_spread}")
    logger.info(f"  Of NTM, skipped - No Model Signal (score too low/neutral): {markets_skipped_no_model_signal}")
    logger.info(f"  Of NTM, skipped - Insufficient Capital or Size=0: {markets_skipped_insufficient_capital}")
    logger.info(f"Total trades executed: {trades_executed}")
    logger.info(f"Total contracts traded: {total_contracts_traded}")
    logger.info(f"Total P&L: {total_pnl_cents:.2f} cents (${total_pnl_cents/100:.2f})")
    if trades_executed > 0:
        logger.info(f"Average P&L per trade: {total_pnl_cents/trades_executed:.2f} cents")
        if total_contracts_traded > 0: 
            logger.info(f"Average P&L per contract: {total_pnl_cents/total_contracts_traded:.2f} cents")

    print(f"\n--- Backtest Summary ({os.path.basename(linreg_strategy.__file__)} with Capital Sizing) for Date: {BACKTEST_DATE_YYMMMDD} ---")
    print(f"Initial Capital: ${TOTAL_CAPITAL_USD:.2f}")
    print(f"Final Capital: ${current_capital_usd:.2f}")
    # (Rest of print summary similar to log summary)
    print(f"Total Kalshi market files found: {markets_processed_total}")
    print(f"Markets skipped (Not NTM): {markets_skipped_relevance}")
    print(f"Markets considered for trading (NTM): {markets_considered_for_trade}")
    print(f"  Skipped (No BTC Data/Features): {markets_skipped_no_btc_data}")
    print(f"  Skipped (No/Stale Kalshi Data): {markets_skipped_no_kalshi_data}")
    print(f"  Skipped (Wide Spread): {markets_skipped_wide_spread}")
    print(f"  Skipped (No Model Signal): {markets_skipped_no_model_signal}")
    print(f"  Skipped (Insufficient Capital/Size 0): {markets_skipped_insufficient_capital}")
    print(f"Total Trades Executed: {trades_executed}")
    print(f"Total Contracts Traded: {total_contracts_traded}")
    print(f"Total P&L: ${total_pnl_cents/100:.2f}")
    if trades_executed > 0:
        print(f"Avg P&L per Trade: ${total_pnl_cents/trades_executed/100:.4f}")
        if total_contracts_traded > 0:
             print(f"Avg P&L per Contract: ${total_pnl_cents/total_contracts_traded/100:.4f}")
    print(f"Detailed log saved to: {log_file_path}")

if __name__ == "__main__":
    # (BASE_PROJECT_DIR auto-detection and final logging remains the same as previous version)
    script_path = os.path.abspath(__file__)
    current_dir_name = os.path.basename(os.path.dirname(script_path))
    if current_dir_name == "notebooks": BASE_PROJECT_DIR = os.path.dirname(script_path)
    else:
        BASE_PROJECT_DIR = os.path.join(os.path.dirname(script_path), "notebooks")
        if not os.path.isdir(os.path.join(BASE_PROJECT_DIR, "organized_market_data")):
            BASE_PROJECT_DIR = os.path.dirname(script_path) 
            logger.warning(f"Could not find 'notebooks/organized_market_data'. Assuming BASE_PROJECT_DIR is script directory: {BASE_PROJECT_DIR}")

    utils.BASE_PROJECT_DIR = BASE_PROJECT_DIR 
    utils.BINANCE_DATA_PATH_TEMPLATE = os.path.join(BASE_PROJECT_DIR, "binance_data/BTCUSDT-1m-{date_nodash}/BTCUSDT-1m-{date_nodash}.csv")
    KALSHI_ORGANIZED_DATA_DIR = os.path.join(BASE_PROJECT_DIR, "organized_market_data")
    MARKET_OUTCOMES_CSV_PATH = os.path.join(BASE_PROJECT_DIR, "market_candlestick_data/kalshi_btc_hourly_market_outcomes.csv")
    LOGS_DIR = os.path.join(BASE_PROJECT_DIR, "logs"); os.makedirs(LOGS_DIR, exist_ok=True)
    log_file_path = os.path.join(LOGS_DIR, f"backtest_log_LR_CapSizing_{BACKTEST_DATE_YYMMMDD}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(name)s.%(funcName)s:%(lineno)d - %(message)s',handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])
    logger.info(f"Final BASE_PROJECT_DIR: {BASE_PROJECT_DIR}")
    logger.info(f"Final Binance data template: {utils.BINANCE_DATA_PATH_TEMPLATE}")
    logger.info(f"Final Kalshi organized data: {KALSHI_ORGANIZED_DATA_DIR}")
    logger.info(f"Final Market outcomes CSV: {MARKET_OUTCOMES_CSV_PATH}")
    logger.info(f"Final Log file: {log_file_path}")
    
    run_backtest()