# /Users/omarabul-hassan/Desktop/projects/kalshi/random/backtest/live_backtest_engine.py

import pandas as pd
import numpy as np
import datetime as dt
from datetime import timezone
import logging

import live_backtest_config as config
import live_backtest_utils as utils 

def run_live_backtest(all_features_df: pd.DataFrame, model, scaler, model_feature_names: list):
    engine_logger = logging.getLogger("live_backtest_engine_logger")
    engine_logger.info("Starting LIVE backtest engine...")

    if all_features_df.empty:
        engine_logger.warning("Input features DataFrame is empty. Cannot run backtest.")
        return 0, 0
        
    decision_points_to_simulate_df = all_features_df.copy()
    # Ensure all necessary columns for logging are present, even if not in model_feature_names
    # For example, 'current_btc_price' if it's not always a model feature but good for logs.
    # Here, model_feature_names should ideally contain 'current_btc_price'.
    
    decision_points_to_simulate_df.sort_values(by='decision_timestamp_s', inplace=True)
    
    num_total_decision_points_to_simulate = len(decision_points_to_simulate_df)
    if num_total_decision_points_to_simulate == 0:
        engine_logger.warning("No decision points to simulate after sorting/initial checks.")
        return 0,0
        
    engine_logger.info(f"Processing {num_total_decision_points_to_simulate} decision points from live data.")

    current_trade_log_file = None
    current_log_date_str = None 

    overall_pnl_cents = 0
    trade_stats = {'yes_won': 0, 'yes_lost': 0, 'no_won': 0, 'no_lost': 0}
    active_market_bets = {} 

    first_buy_yes_logged = False
    first_buy_no_logged = False
    
    # +++ Define header for the rich trade log +++
    # Ensure this matches the order of fields in trade_log_entry_fields
    # Add the model_feature_names to the header
    rich_trade_log_header_base = [
        "trade_execution_time_utc", "market_ticker", "strike_price", "resolution_time_ts",
        "decision_timestamp_s", "time_to_resolution_minutes", "action", "predicted_prob_yes",
        "bet_cost_cents", "contracts_traded", "kalshi_outcome_target", "pnl_cents"
    ]
    # We'll get the actual model_feature_names from the argument
    # The model_feature_names list itself will be written to the header
    # And the corresponding values will be written for each trade.

    for index, decision_point_row in decision_points_to_simulate_df.iterrows():
        market_ticker = decision_point_row['market_ticker']
        decision_timestamp_s = decision_point_row['decision_timestamp_s']
        resolution_time_ts = decision_point_row['resolution_time_ts']
        strike_price = decision_point_row['strike_price']
        actual_market_outcome = decision_point_row['target'] 
        time_to_resolution_min = decision_point_row['time_to_resolution_minutes']

        decision_date_obj = dt.datetime.fromtimestamp(decision_timestamp_s, tz=timezone.utc)
        decision_date_str = decision_date_obj.strftime("%Y-%m-%d")
        
        if decision_date_str != current_log_date_str:
            # For rich logs, we need to write the header once with all feature names
            # The setup_daily_trade_logger in utils currently writes a basic header.
            # We need to modify how the header is written or handle it here.
            
            # Let's modify setup_daily_trade_logger to accept a custom header
            dynamic_header = ",".join(rich_trade_log_header_base + model_feature_names) + "\n"
            _, current_trade_log_file = utils.setup_daily_trade_logger(
                decision_date_str, 
                custom_header=dynamic_header
            )
            current_log_date_str = decision_date_str

        if config.ONE_BET_PER_KALSHI_MARKET and market_ticker in active_market_bets:
            continue 

        try:
            current_features_for_model_series = decision_point_row[model_feature_names].copy()
        except KeyError as e:
            engine_logger.error(f"KeyError for features in {market_ticker} at {decision_timestamp_s}: {e}.")
            continue
        
        if current_features_for_model_series.isnull().any():
            nan_cols = current_features_for_model_series[current_features_for_model_series.isnull()].index.tolist()
            # engine_logger.warning(f"NaNs in features for {market_ticker} at {decision_timestamp_s} before scaling: {nan_cols}. Skipping.") # Can be very verbose
            continue
        
        single_decision_point_features_df = current_features_for_model_series.to_frame().T
        
        try:
            scaled_features = scaler.transform(single_decision_point_features_df)
            predicted_proba_yes = model.predict_proba(scaled_features)[0, 1]
        except Exception as e:
            engine_logger.error(f"Error scaling/predicting for {market_ticker} at {decision_timestamp_s}: {e}")
            continue
            
        action = "HOLD"; bet_cost_cents = 0; pnl_cents = 0; contracts_traded = 0
        made_trade_for_debug = False 

        if predicted_proba_yes > config.PROBABILITY_THRESHOLD_YES:
            action = "BUY_YES"; bet_cost_cents = int(round(predicted_proba_yes * 100)); contracts_traded = 1 # round for bet_cost
            if actual_market_outcome == 1: pnl_cents = 100 - bet_cost_cents; trade_stats['yes_won'] += 1
            else: pnl_cents = -bet_cost_cents; trade_stats['yes_lost'] += 1
            if not first_buy_yes_logged: made_trade_for_debug = True; first_buy_yes_logged = True
        
        elif (1 - predicted_proba_yes) > config.PROBABILITY_THRESHOLD_NO:
            action = "BUY_NO"; prob_no = 1 - predicted_proba_yes; bet_cost_cents = int(round(prob_no * 100)); contracts_traded = 1 # round for bet_cost
            if actual_market_outcome == 0: pnl_cents = 100 - bet_cost_cents; trade_stats['no_won'] +=1
            else: pnl_cents = -bet_cost_cents; trade_stats['no_lost'] +=1
            if not first_buy_no_logged: made_trade_for_debug = True; first_buy_no_logged = True

        if made_trade_for_debug: # Detailed logging for first trades
            engine_logger.info(f"\n--- DETAILED TRADE DEBUG ({action}) ---")
            engine_logger.info(f"Market Ticker: {market_ticker}, Decision Timestamp: {decision_timestamp_s} ({decision_date_obj.isoformat()})")
            engine_logger.info(f"Raw Features Used ({len(model_feature_names)}):\n{current_features_for_model_series.to_string()}")
            engine_logger.info(f"Predicted P(Yes): {predicted_proba_yes:.4f}, Actual Outcome: {actual_market_outcome}, P&L: {pnl_cents}")
            engine_logger.info(f"--- END DETAILED TRADE DEBUG ---\n")
        
        overall_pnl_cents += pnl_cents

        if action != "HOLD":
            if config.ONE_BET_PER_KALSHI_MARKET: active_market_bets[market_ticker] = action
            
            # Prepare fields for the rich trade log
            trade_log_entry_fields_base = [
                f"{decision_date_obj.isoformat()}",
                market_ticker, f"{strike_price:.2f}", str(resolution_time_ts),
                str(decision_timestamp_s), f"{time_to_resolution_min:.2f}", action, 
                f"{predicted_proba_yes:.4f}", str(bet_cost_cents), str(contracts_traded),
                str(actual_market_outcome), str(pnl_cents)
            ]
            # Add the values of the model features for this decision point
            feature_values_for_log = [f"{current_features_for_model_series[feat_name]:.6f}" if isinstance(current_features_for_model_series[feat_name], (float, np.floating)) else str(current_features_for_model_series[feat_name]) for feat_name in model_feature_names]
            
            full_trade_log_entry = ",".join(trade_log_entry_fields_base + feature_values_for_log) + "\n"

            try:
                with open(current_trade_log_file, 'a') as f: f.write(full_trade_log_entry)
            except Exception as e: engine_logger.error(f"Error writing to trade log {current_trade_log_file}: {e}")

    engine_logger.info("LIVE Backtest engine finished processing.")
    engine_logger.info(f"--- LIVE Backtest Summary ---")
    engine_logger.info(f"Total P&L: {overall_pnl_cents / 100:.2f} USD")
    engine_logger.info(f"YES Bets: {trade_stats['yes_won']} Won, {trade_stats['yes_lost']} Lost")
    engine_logger.info(f"NO Bets: {trade_stats['no_won']} Won, {trade_stats['no_lost']} Lost")
    total_trades = sum(trade_stats.values())
    engine_logger.info(f"Total Trades Made: {total_trades}")
    
    return overall_pnl_cents, total_trades