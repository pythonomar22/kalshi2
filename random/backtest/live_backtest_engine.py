# /random/backtest/live_backtest_engine.py

import pandas as pd
import numpy as np
import datetime as dt
from datetime import timezone
import logging

# Import from live_backtest_config and live_backtest_data_utils
try:
    from . import live_backtest_config as live_cfg
    from . import live_backtest_data_utils as live_data_utils # For setup_daily_trade_logger
except ImportError: # For direct script execution if needed
    import live_backtest_config as live_cfg
    import live_backtest_data_utils as live_data_utils


def run_live_data_backtest(all_features_df: pd.DataFrame, model, scaler, model_feature_names: list):
    engine_logger = logging.getLogger("live_backtest_engine")
    engine_logger.info("Starting LIVE DATA backtest engine...")

    if all_features_df.empty:
        engine_logger.warning("No features provided to the live backtest engine. Exiting.")
        return 0, 0, pd.DataFrame() # pnl, trades, trade_log_df

    # Features are already filtered by market open/close times during live feature generation.
    # Timestamps (decision_timestamp_s, resolution_time_ts) are already in UTC seconds.

    # Sort all decision points to simulate chronologically by their actual decision time
    # This should already be somewhat ordered by market, then by time from feature gen, but good to ensure.
    all_features_df.sort_values(by='decision_timestamp_s', inplace=True)

    num_total_decision_points_to_simulate = len(all_features_df)
    engine_logger.info(f"Processing {num_total_decision_points_to_simulate} decision points from live data.")

    current_trade_log_file = None
    current_log_date_str = None

    overall_pnl_cents = 0
    trade_stats = {'yes_won': 0, 'yes_lost': 0, 'no_won': 0, 'no_lost': 0}
    active_market_bets = {} # For ONE_BET_PER_KALSHI_MARKET
    
    all_trades_log_list = []


    for index, decision_point_row in all_features_df.iterrows():
        market_ticker = decision_point_row['market_ticker']
        decision_timestamp_s = decision_point_row['decision_timestamp_s']
        resolution_time_ts = decision_point_row['resolution_time_ts']
        # Strike price is part of model_feature_names
        strike_price = decision_point_row['strike_price'] 
        actual_market_outcome = decision_point_row['target']
        # time_to_resolution_minutes is part of model_feature_names
        time_to_resolution_min = decision_point_row['time_to_resolution_minutes']

        decision_date_obj = dt.datetime.fromtimestamp(decision_timestamp_s, tz=timezone.utc)
        decision_date_str = decision_date_obj.strftime("%Y-%m-%d")

        if decision_date_str != current_log_date_str:
            current_trade_log_file = live_data_utils.setup_daily_trade_logger(decision_date_str, live_cfg.LOG_DIR)
            current_log_date_str = decision_date_str
            engine_logger.info(f"Logging live trades for decisions made on {current_log_date_str} to {current_trade_log_file}")

        if live_cfg.ONE_BET_PER_KALSHI_MARKET and market_ticker in active_market_bets:
            continue

        try:
            current_features_for_model = decision_point_row[model_feature_names].copy()
        except KeyError as e:
            engine_logger.error(f"KeyError for features of {market_ticker} at {decision_timestamp_s}: {e}. Available: {decision_point_row.index.tolist()}")
            continue
        
        # Handle NaNs: The model expects no NaNs. Imputation should happen before this.
        # If NaNs reach here, it's an issue in feature generation or selection.
        if current_features_for_model.isnull().any():
            nan_cols = current_features_for_model[current_features_for_model.isnull()].index.tolist()
            engine_logger.warning(f"NaNs in features for {market_ticker} at {decision_timestamp_s}: {nan_cols}. Skipping prediction.")
            continue
        
        single_decision_point_features_df = current_features_for_model.to_frame().T
        
        try:
            scaled_features = scaler.transform(single_decision_point_features_df)
            predicted_proba_yes = model.predict_proba(scaled_features)[0, 1]
        except Exception as e:
            engine_logger.error(f"Error scaling/predicting for {market_ticker} at {decision_timestamp_s}: {e}")
            continue
            
        action = "HOLD"; bet_cost_cents = 0; pnl_cents = 0; contracts_traded = 0

        if predicted_proba_yes > live_cfg.PROBABILITY_THRESHOLD_YES:
            action = "BUY_YES"
            # Bet cost is tricky in live simulation without live bid/ask.
            # For simplicity, assume we can get the trade at predicted_proba_yes price.
            # Kalshi contract cost is P(Yes) * 100 cents for a YES contract.
            bet_cost_cents = int(predicted_proba_yes * 100) 
            contracts_traded = 1
            if actual_market_outcome == 1: # Won YES bet
                pnl_cents = 100 - bet_cost_cents
                trade_stats['yes_won'] += 1
            else: # Lost YES bet
                pnl_cents = -bet_cost_cents
                trade_stats['yes_lost'] += 1
        
        elif (1 - predicted_proba_yes) > live_cfg.PROBABILITY_THRESHOLD_NO:
            action = "BUY_NO"
            prob_no = 1 - predicted_proba_yes
            bet_cost_cents = int(prob_no * 100) # Cost of a NO contract
            contracts_traded = 1
            if actual_market_outcome == 0: # Won NO bet
                pnl_cents = 100 - bet_cost_cents
                trade_stats['no_won'] +=1
            else: # Lost NO bet
                pnl_cents = -bet_cost_cents
                trade_stats['no_lost'] +=1
        
        overall_pnl_cents += pnl_cents

        if action != "HOLD":
            if live_cfg.ONE_BET_PER_KALSHI_MARKET:
                active_market_bets[market_ticker] = action
            
            trade_log_entry_dict = {
                "trade_execution_time_utc": decision_date_obj.isoformat(),
                "market_ticker": market_ticker,
                "strike_price": strike_price,
                "resolution_time_ts": resolution_time_ts,
                "decision_timestamp_s": decision_timestamp_s,
                "time_to_resolution_minutes": f"{time_to_resolution_min:.2f}",
                "action": action,
                "predicted_prob_yes": f"{predicted_proba_yes:.4f}",
                "bet_cost_cents": bet_cost_cents,
                "contracts_traded": contracts_traded,
                "actual_outcome_target": actual_market_outcome,
                "pnl_cents": pnl_cents
            }
            all_trades_log_list.append(trade_log_entry_dict)

            # Write to daily CSV log file
            trade_log_entry_csv_str = (
                f"{decision_date_obj.isoformat()},"
                f"{market_ticker},{strike_price},{resolution_time_ts},"
                f"{decision_timestamp_s},{time_to_resolution_min:.2f},"
                f"{action},{predicted_proba_yes:.4f},{bet_cost_cents},{contracts_traded},"
                f"{actual_market_outcome},{pnl_cents}\n"
            )
            try:
                if current_trade_log_file: # Ensure it's initialized
                    with open(current_trade_log_file, 'a') as f:
                        f.write(trade_log_entry_csv_str)
            except Exception as e:
                engine_logger.error(f"Error writing to trade log {current_trade_log_file}: {e}")

    engine_logger.info("Live data backtest engine finished processing.")
    engine_logger.info(f"--- Live Data Backtest Summary ---")
    engine_logger.info(f"Total P&L: {overall_pnl_cents / 100:.2f} USD")
    engine_logger.info(f"YES Bets: {trade_stats['yes_won']} Won, {trade_stats['yes_lost']} Lost")
    engine_logger.info(f"NO Bets: {trade_stats['no_won']} Won, {trade_stats['no_lost']} Lost")
    total_trades = sum(trade_stats.values())
    engine_logger.info(f"Total Trades Made: {total_trades}")
    
    trade_log_df = pd.DataFrame(all_trades_log_list)
    return overall_pnl_cents, total_trades, trade_log_df