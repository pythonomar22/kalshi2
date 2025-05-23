# /notebooks/backtest/backtest_engine.py (MODIFIED)

import pandas as pd
import numpy as np
import datetime as dt
from datetime import timezone
import logging

from backtest import backtest_config as config # Assuming notebooks/ is in sys.path from .ipynb
from backtest import backtest_utils as utils

def run_backtest(all_features_df: pd.DataFrame, model, scaler, model_feature_names: list):
    engine_logger = logging.getLogger("backtest_engine_logger") # Use a specific logger for the engine
    engine_logger.info("Starting backtest engine for PER-MINUTE decision model...")
    
    # --- Step 1: Filter for markets RESOLVING within the overall test period ---
    eligible_markets_decision_points_df = all_features_df[
        (all_features_df['resolution_time_ts'] >= config.MARKET_RESOLUTION_START_TS) &
        (all_features_df['resolution_time_ts'] <= config.MARKET_RESOLUTION_END_TS)
    ].copy()

    if eligible_markets_decision_points_df.empty:
        engine_logger.warning("No decision points for markets resolving in the specified period.")
        return 0, 0

    engine_logger.info(f"Found {len(eligible_markets_decision_points_df)} decision points for markets resolving in the period.")

    # --- Step 2: Filter these decision points to only those OCCURRING within the decision-making calendar days ---
    decision_points_to_simulate_df = eligible_markets_decision_points_df[
        (eligible_markets_decision_points_df['decision_timestamp_s'] >= config.DECISION_MAKING_START_TS) &
        (eligible_markets_decision_points_df['decision_timestamp_s'] <= config.DECISION_MAKING_END_TS)
    ].copy()

    if decision_points_to_simulate_df.empty:
        engine_logger.warning(f"No decision points fall within the actual decision-making calendar period: {config.DECISION_MAKING_START_DATE_STR} to {config.DECISION_MAKING_END_DATE_STR}.")
        return 0, 0
        
    # Sort all decision points to simulate chronologically by their actual decision time
    decision_points_to_simulate_df.sort_values(by='decision_timestamp_s', inplace=True)
    
    num_total_decision_points_to_simulate = len(decision_points_to_simulate_df)
    engine_logger.info(f"Processing {num_total_decision_points_to_simulate} decision points that occur between {config.DECISION_MAKING_START_DATE_STR} and {config.DECISION_MAKING_END_DATE_STR}.")

    current_trade_log_file = None
    current_log_date_str = None 

    overall_pnl_cents = 0
    trade_stats = {'yes_won': 0, 'yes_lost': 0, 'no_won': 0, 'no_lost': 0}
    active_market_bets = {} 

    for index, decision_point_row in decision_points_to_simulate_df.iterrows():
        market_ticker = decision_point_row['market_ticker']
        decision_timestamp_s = decision_point_row['decision_timestamp_s']
        # ... (rest of variable assignments: resolution_time_ts, strike_price, actual_market_outcome, time_to_resolution_min)
        resolution_time_ts = decision_point_row['resolution_time_ts']
        strike_price = decision_point_row['strike_price']
        actual_market_outcome = decision_point_row['target'] 
        time_to_resolution_min = decision_point_row['time_to_resolution_minutes']


        decision_date_obj = dt.datetime.fromtimestamp(decision_timestamp_s, tz=timezone.utc)
        decision_date_str = decision_date_obj.strftime("%Y-%m-%d")
        if decision_date_str != current_log_date_str:
            _, current_trade_log_file = utils.setup_daily_trade_logger(decision_date_str)
            current_log_date_str = decision_date_str
            # engine_logger.info(f"Logging trades for decisions made on {current_log_date_str} to {current_trade_log_file}") # Can be verbose

        if config.ONE_BET_PER_KALSHI_MARKET and market_ticker in active_market_bets:
            continue 

        try:
            current_features_for_model = decision_point_row[model_feature_names].copy()
        except KeyError as e:
            engine_logger.error(f"KeyError for {market_ticker} at {decision_timestamp_s}: {e}.")
            continue
        
        if current_features_for_model.isnull().any():
            # nan_cols = current_features_for_model[current_features_for_model.isnull()].index.tolist() # For debugging
            # engine_logger.warning(f"NaNs in features for {market_ticker} at {decision_timestamp_s}: {nan_cols}. Skipping.") # Verbose
            continue
        
        single_decision_point_features_df = current_features_for_model.to_frame().T
        
        try:
            scaled_features = scaler.transform(single_decision_point_features_df)
            predicted_proba_yes = model.predict_proba(scaled_features)[0, 1]
        except Exception as e:
            engine_logger.error(f"Error scaling/predicting for {market_ticker} at {decision_timestamp_s}: {e}")
            continue
            
        action = "HOLD"; bet_cost_cents = 0; pnl_cents = 0; contracts_traded = 0

        if predicted_proba_yes > config.PROBABILITY_THRESHOLD_YES:
            action = "BUY_YES"; bet_cost_cents = int(predicted_proba_yes * 100); contracts_traded = 1
            if actual_market_outcome == 1: pnl_cents = 100 - bet_cost_cents; trade_stats['yes_won'] += 1
            else: pnl_cents = -bet_cost_cents; trade_stats['yes_lost'] += 1
        
        elif (1 - predicted_proba_yes) > config.PROBABILITY_THRESHOLD_NO:
            action = "BUY_NO"; prob_no = 1 - predicted_proba_yes; bet_cost_cents = int(prob_no * 100); contracts_traded = 1
            if actual_market_outcome == 0: pnl_cents = 100 - bet_cost_cents; trade_stats['no_won'] +=1
            else: pnl_cents = -bet_cost_cents; trade_stats['no_lost'] +=1
        
        overall_pnl_cents += pnl_cents

        if action != "HOLD":
            if config.ONE_BET_PER_KALSHI_MARKET: active_market_bets[market_ticker] = action
            trade_log_entry = (
                f"{decision_date_obj.isoformat()},"
                f"{market_ticker},{strike_price},{resolution_time_ts},"
                f"{decision_timestamp_s},{time_to_resolution_min:.2f},"
                f"{action},{predicted_proba_yes:.4f},{bet_cost_cents},{contracts_traded},"
                f"{actual_market_outcome},{pnl_cents}\n"
            )
            try:
                with open(current_trade_log_file, 'a') as f: f.write(trade_log_entry)
            except Exception as e: engine_logger.error(f"Error writing to trade log {current_trade_log_file}: {e}")

    engine_logger.info("Backtest engine (per-minute model) finished processing.")
    engine_logger.info(f"--- Backtest Summary ---")
    engine_logger.info(f"Total P&L: {overall_pnl_cents / 100:.2f} USD")
    engine_logger.info(f"YES Bets: {trade_stats['yes_won']} Won, {trade_stats['yes_lost']} Lost")
    engine_logger.info(f"NO Bets: {trade_stats['no_won']} Won, {trade_stats['no_lost']} Lost")
    total_trades = sum(trade_stats.values())
    engine_logger.info(f"Total Trades Made: {total_trades}")
    
    return overall_pnl_cents, total_trades